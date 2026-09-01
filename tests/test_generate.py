"""The corpus: synthetic, deterministic, and rendered six ways."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
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
    generate_corpus(seed=3, n_studies=6, out_dir=root,
                    invariance_truths=4, background_per_study=2)
    return root


# -- determinism ------------------------------------------------------------


def test_the_same_seed_produces_byte_identical_files(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    generate_corpus(seed=5, n_studies=6, out_dir=a,
                    invariance_truths=3, background_per_study=2)
    generate_corpus(seed=5, n_studies=6, out_dir=b,
                    invariance_truths=3, background_per_study=2)
    for name in ("ae.csv", "lb.csv", "ex.csv", "dm.csv", "narratives.jsonl",
                 "truths.jsonl", "gold_records.jsonl", "gold_episodes.jsonl",
                 "manifest.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_a_different_seed_produces_a_different_corpus(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    generate_corpus(seed=5, n_studies=6, out_dir=a,
                    invariance_truths=3, background_per_study=2)
    generate_corpus(seed=6, n_studies=6, out_dir=b,
                    invariance_truths=3, background_per_study=2)
    assert (a / "ae.csv").read_bytes() != (b / "ae.csv").read_bytes()


# -- it says what it is -----------------------------------------------------


def test_every_row_is_marked_synthetic(small):
    for table in ("dm", "ae", "ex", "lb"):
        table_rows = rows(small, table)
        assert table_rows
        assert {r["SYNTHETIC"] for r in table_rows} == {"Y"}


def test_the_corpus_carries_a_notice_that_it_is_not_patient_data(small):
    manifest = json.loads((small / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["synthetic"] is True
    assert "No real patient data" in manifest["notice"]
    assert "SYNTHETIC DATA ONLY" in (small / "README.txt").read_text(encoding="utf-8")


# -- the six renderings -----------------------------------------------------


def test_each_truth_in_the_invariance_cohort_is_rendered_under_every_study(small):
    episodes = jsonl(small, "gold_episodes.jsonl")
    by_truth = defaultdict(set)
    for episode in episodes:
        if episode["base_truth_id"].startswith("T"):
            by_truth[episode["base_truth_id"]].add(episode["study_id"])
    assert by_truth
    for truth_id, studies in by_truth.items():
        assert len(studies) == 6, f"{truth_id} rendered in {sorted(studies)}"


def test_the_representation_independent_verdict_is_the_same_across_renderings(small):
    """The invariance reference: what happened does not depend on the CRF."""
    by_truth = defaultdict(set)
    for episode in jsonl(small, "gold_episodes.jsonl"):
        if episode["base_truth_id"].startswith("T"):
            by_truth[episode["base_truth_id"]].add(
                episode["representation_independent_verdict"]
            )
    for truth_id, verdicts in by_truth.items():
        assert len(verdicts) == 1, f"{truth_id} has {verdicts}"


def test_what_a_rendering_can_support_may_be_weaker_than_the_truth(small):
    """V-D collects no coded terms, so it cannot reach `explicit` on coding."""
    weaker = [
        e for e in jsonl(small, "gold_episodes.jsonl")
        if e["state_as_recorded"] != e["true_evidence_state"]
    ]
    assert weaker, "no rendering loses anything, so invariance is untested"
    assert all(not e["coded_by_study"] for e in weaker)


def test_one_study_splits_a_single_episode_across_records(small):
    counts = Counter(
        (e["study_id"], e["n_records"] > 1)
        for e in jsonl(small, "gold_episodes.jsonl")
    )
    splitting = {study for (study, split) in counts if split}
    assert splitting, "no study exercises multi-record episodes"


def test_the_linked_form_study_puts_its_glucose_outside_the_ae_row(small):
    linked = rows(small, "linked_hypo_event")
    assert linked
    assert {r["SYNTHETIC"] for r in linked} == {"Y"}
    linked_ids = {r["LNKID"] for r in linked}
    ae_links = {r["AELNKID"] for r in rows(small, "ae") if r["AELNKID"]}
    assert linked_ids & ae_links


def test_studies_use_different_dictionary_versions(small):
    manifest = json.loads((small / "manifest.json").read_text(encoding="utf-8"))
    versions = {b["dictionary_version"] for b in manifest["studies"].values()}
    assert len(versions) > 1


def test_studies_use_different_glucose_units(small):
    units = {r["LBORRESU"] for r in rows(small, "lb") if r["LBORRESU"]}
    assert {"mg/dL", "mmol/L"} <= units


# -- the answer key ---------------------------------------------------------


def test_gold_records_record_what_each_field_state_should_be(small):
    gold = jsonl(small, "gold_records.jsonl")
    assert gold
    states = {s for row in gold for s in row["collection_states"].values()}
    # More than one kind of blank has to appear, or the harness measures nothing.
    assert len({s for s in states if s != "collected"}) >= 3


def test_a_value_only_the_narrative_carries_is_marked_recoverable(small):
    gold = jsonl(small, "gold_records.jsonl")
    recoverable = [
        row for row in gold
        if any(k not in row["values"] or row["values"][k] in (None, "")
               for k in row["narrated_values"])
    ]
    assert recoverable, "nothing is recoverable only from text"


def test_narratives_are_linked_to_the_records_they_describe(small):
    doc_ids = {n["doc_id"] for n in jsonl(small, "narratives.jsonl")}
    ae_docs = {r["DOCID"] for r in rows(small, "ae") if r["DOCID"]}
    assert ae_docs
    assert ae_docs <= doc_ids
