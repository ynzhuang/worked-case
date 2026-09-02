"""The corpus: synthetic, deterministic, and rendered six ways."""

from __future__ import annotations

import collections
import csv
import json
from pathlib import Path

import pytest

from aelayer.generate import generate_corpus


def rows(root: Path, table: str) -> list[dict]:
    with (root / f"{table}.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def jsonl(root: Path, name: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (root / name).read_text(encoding="utf-8").splitlines() if line
    ]


@pytest.fixture(scope="module")
def small(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("gen")
    generate_corpus(seed=3, out_dir=root, shared_truths=8, extra_per_profile=4)
    return root


# -- determinism ------------------------------------------------------------


def test_the_same_seed_produces_byte_identical_files(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for target in (a, b):
        generate_corpus(seed=5, out_dir=target, shared_truths=4, extra_per_profile=2)
    for name in ("ae.csv", "ex.csv", "dm.csv", "suppae.csv", "co.csv",
                 "documents.jsonl", "truths.jsonl", "gold_records.jsonl",
                 "gold_episodes.jsonl", "manifest.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_a_different_seed_produces_a_different_corpus(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    generate_corpus(seed=5, out_dir=a, shared_truths=4, extra_per_profile=2)
    generate_corpus(seed=6, out_dir=b, shared_truths=4, extra_per_profile=2)
    assert (a / "ae.csv").read_bytes() != (b / "ae.csv").read_bytes()


# -- it says what it is -----------------------------------------------------


def test_every_row_is_marked_synthetic(small):
    for table in ("dm", "ae", "ex", "suppae", "co"):
        table_rows = rows(small, table)
        assert table_rows, table
        assert {r["SYNTHETIC"] for r in table_rows} == {"Y"}


def test_the_corpus_carries_a_notice_that_it_is_not_patient_data(small):
    manifest = json.loads((small / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["synthetic"] is True
    assert "No real patient data" in manifest["notice"]
    assert "SYNTHETIC DATA ONLY" in (small / "README.txt").read_text(encoding="utf-8")


# -- the six renderings -----------------------------------------------------


def test_each_shared_truth_is_rendered_under_every_profile(small):
    by_truth = collections.defaultdict(set)
    for episode in jsonl(small, "gold_episodes.jsonl"):
        if episode["cohort"] == "shared":
            by_truth[episode["truth_id"]].add(episode["profile"])
    assert by_truth
    for truth_id, seen in by_truth.items():
        assert len(seen) == 6, f"{truth_id} rendered in {sorted(seen)}"


def test_the_underlying_truth_does_not_change_between_renderings(small):
    by_truth = collections.defaultdict(set)
    for episode in jsonl(small, "gold_episodes.jsonl"):
        if episode["cohort"] == "shared":
            by_truth[episode["truth_id"]].add(
                (episode["true_location"], episode["onset_offset_days"])
            )
    for truth_id, seen in by_truth.items():
        assert len(seen) == 1, f"{truth_id} differs across renderings: {seen}"


def test_what_a_rendering_can_support_may_be_weaker_than_the_truth(small):
    weaker = [
        e for e in jsonl(small, "gold_episodes.jsonl")
        if e["true_verdict"] != e["verdict_if_location_available"]
    ]
    assert weaker
    assert all(not e["location_available"] for e in weaker)
    assert all(e["true_verdict"] == "not_ascertainable" for e in weaker)


def test_the_location_lives_in_a_different_place_in_each_profile(small):
    manifest = json.loads((small / "manifest.json").read_text(encoding="utf-8"))
    homes = {
        profile: tuple(body["location_home"])
        for profile, body in manifest["profiles"].items()
    }
    assert homes["P1_structured"] == ("AELOC",)
    assert homes["P2_text"] == ("reported_term",)
    assert homes["P3_prespecified"] == ("none",)
    assert homes["P4_sponsor"] == ("sponsor_variable",)
    assert homes["P5_comment"] == ("comment",)
    assert set(homes["P6_both"]) == {"AELOC", "reported_term"}


def test_the_sponsor_variable_is_deliberately_non_standard(small):
    supplemental = rows(small, "suppae")
    assert supplemental
    assert {r["QNAM"] for r in supplemental} == {"RASHSITE"}
    assert {r["IDVAR"] for r in supplemental} == {"AESPID"}


def test_comments_point_back_at_the_ae_record_they_describe(small):
    comments = rows(small, "co")
    assert comments
    record_ids = {r["AESPID"] for r in rows(small, "ae")}
    assert {r["IDVARVAL"] for r in comments} <= record_ids


def test_profiles_use_different_dictionary_versions_and_date_styles(small):
    manifest = json.loads((small / "manifest.json").read_text(encoding="utf-8"))
    versions = {b["dictionary_version"] for b in manifest["profiles"].values()}
    assert len(versions) > 1
    dates = {r["AESTDTC"] for r in rows(small, "ae") if r["AESTDTC"]}
    assert any("-" in d and d[2] == "-" and not d[:4].isdigit() for d in dates)


# -- the answer key ---------------------------------------------------------


def test_gold_distinguishes_no_site_from_no_site_recorded(small):
    """The distinction the availability confusion matrix is scored against."""
    gold = jsonl(small, "gold_episodes.jsonl")
    generalised = [e for e in gold if e["true_location"] == "GENERALISED"]
    not_collected = [
        e for e in gold if e["profile"] == "P3_prespecified" and e["concept"] == "RASH"
    ]
    assert generalised, "no genuinely site-less event, so 'absent' is untested"
    assert not_collected
    assert all(e["true_verdict"] != "not_ascertainable" for e in generalised
               if e["location_available"])
    assert any(e["true_verdict"] == "not_ascertainable" for e in not_collected)


def test_gold_records_where_each_route_holds_the_location(small):
    gold = jsonl(small, "gold_records.jsonl")
    assert any(g["location_in_structured"] and g["location_in_text"] for g in gold)
    assert any(g["location_in_structured"] and not g["location_in_text"] for g in gold)
    assert any(not g["location_in_structured"] and g["location_in_text"] for g in gold)


def test_the_corpus_contains_text_the_lexicon_cannot_cover(small):
    outcomes = collections.Counter(
        g["text_outcome"] for g in jsonl(small, "gold_records.jsonl")
    )
    assert outcomes["unlexiconed"] > 0
    assert outcomes["omitted"] > 0


def test_seeded_discrepancies_never_cross_a_verdict_class(small):
    """So the silver numbers and the phenotype numbers stay separable."""
    truncal = {"CHEST", "ABDOMEN", "BACK"}
    for record in jsonl(small, "gold_records.jsonl"):
        if record["text_outcome"] != "discrepant":
            continue
        structured, text = record["true_location"], record["text_location"]
        assert structured != text
        assert (structured in truncal) == (text in truncal)
