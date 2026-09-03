"""The corpus. It is synthetic, it says so, and it plants each thing measured."""

from __future__ import annotations

from collections import Counter

import pytest

from aelayer.generate import SYNTHETIC_FLAG, generate_corpus


def test_every_row_is_flagged_synthetic(store):
    for name, rows in store.tables.items():
        for row in rows:
            assert row.get("SYNTHETIC") == SYNTHETIC_FLAG, (
                f"a row in {name} is not marked synthetic"
            )


def test_every_document_carries_the_synthetic_header(store):
    for document in store.documents.values():
        assert "SYNTHETIC" in document.header
        assert "NOT REAL PATIENT DATA" in document.header


def test_the_manifest_says_the_corpus_is_generated(store):
    assert store.manifest["synthetic"] is True
    assert "No real patient data" in store.manifest["notice"]


def test_generation_is_deterministic(tmp_path):
    a, _ = generate_corpus(seed=3, out_dir=tmp_path / "a")
    b, _ = generate_corpus(seed=3, out_dir=tmp_path / "b")
    assert (a / "ae.csv").read_text() == (b / "ae.csv").read_text()
    assert (a / "gold.jsonl").read_text() == (b / "gold.jsonl").read_text()


def test_a_different_seed_produces_a_different_corpus(tmp_path):
    a, _ = generate_corpus(seed=3, out_dir=tmp_path / "a")
    b, _ = generate_corpus(seed=4, out_dir=tmp_path / "b")
    assert (a / "ae.csv").read_text() != (b / "ae.csv").read_text()


# -- what the corpus contains -------------------------------------------------


def test_all_seven_profiles_are_rendered(store, profiles):
    studies = set(store.studies())
    declared = {p.study_id for p in profiles.profiles.values()}
    assert studies == declared
    assert len(studies) == 7


def test_the_gold_records_what_each_route_holds_separately(store):
    gold = store.gold()
    assert gold
    for row in gold:
        assert set(row) >= {
            "true_assertion", "true_availability", "structured_assertion",
            "text_assertion", "in_structured", "in_text", "readable",
            "true_verdict", "verdict_if_readable",
        }


def test_the_gold_carries_a_verdict_per_ablation_stage(store):
    for row in store.gold():
        assert row["verdict_stage_structured"]
        assert row["verdict_stage_reported_term"]
        assert row["verdict_stage_comments"]


def test_each_stage_can_ascertain_at_least_as_much_as_the_one_below(store):
    gold = store.gold()
    stages = [
        "verdict_stage_structured", "verdict_stage_reported_term",
        "verdict_stage_comments",
    ]
    ascertained = [
        sum(1 for row in gold if row[stage] in ("case", "non_case"))
        for stage in stages
    ]
    assert ascertained == sorted(ascertained)
    assert ascertained[0] < ascertained[-1], (
        "text buys nothing in the answer key, so the ablation cannot measure it"
    )


def test_the_text_never_contradicts_the_truth(store):
    """The generator writes what is true, or says nothing. Never the opposite."""
    mismatches = [
        row for row in store.gold()
        if row["text_assertion"] is not None
        and row["true_assertion"] is not None
        and row["text_assertion"] != row["true_assertion"]
    ]
    assert mismatches == []


def test_only_the_negated_and_mixed_styles_write_an_absence_in_the_term(store,
                                                                        profiles):
    """The style governs AETERM. A comment is a different home with its own rule."""
    documented = {
        row["profile"] for row in store.gold() if row["term_assertion"] == "absent"
    }
    assert documented
    for profile_id in documented:
        assert profiles.profile(profile_id).reported_term_style in \
            ("negated", "mixed"), (
                f"{profile_id} writes an absence into AETERM but is not "
                f"declared to"
            )


def test_a_rich_style_says_nothing_about_an_absence_in_its_term(store, profiles):
    rich = {
        p.profile_id for p in profiles.profiles.values()
        if p.reported_term_style == "rich"
    }
    checked = 0
    for row in store.gold():
        if row["profile"] in rich and row["true_assertion"] == "absent":
            checked += 1
            assert row["term_assertion"] is None, (
                "a `rich` site wrote an absence into its reported term; the "
                "trap this profile exists to set is that an absence there "
                "looks identical to silence"
            )
    assert checked, "no rich profile was tested against an absence"


def test_unlexiconed_phrasings_exist_so_abstention_is_measurable(store):
    """Some text states the modifier in words no catalogue covers."""
    from aelayer.generate import UNLEXICONED

    terms = " ".join(row["AETERM"] for row in store.rows("ae"))
    assert any(phrase in terms for phrase in UNLEXICONED)


def test_every_dictionary_version_is_used(store, profiles, catalog):
    used = {p.dictionary_version for p in profiles.profiles.values()}
    assert used == set(catalog.dictionary_versions)


def test_a_subject_appears_in_exactly_one_study(store):
    owners = Counter()
    for row in store.rows("ae"):
        owners[row["USUBJID"]] = row["STUDYID"]
    for subject, study in owners.items():
        assert subject.startswith(study)


def test_shared_truths_are_rendered_under_every_profile(store):
    shared = [row for row in store.gold() if row["cohort"] == "shared"]
    by_truth = Counter(row["truth_id"] for row in shared)
    assert by_truth
    assert max(by_truth.values()) == len(store.studies())
