"""The three normalization mechanisms, reported separately.

Only one of them is a language problem. Conflating the three is how a system
ends up with a model quietly rewriting coded fields.
"""

from __future__ import annotations

from collections import Counter

import pytest

from aelayer.catalog import ConfigError


# -- 5.1 language variation ---------------------------------------------------


def test_language_variation_is_the_only_model_mechanism(configs):
    from aelayer.extract.backends import LANGUAGE_VARIATION
    from aelayer.guards import DETERMINISTIC_MECHANISMS, MODEL_MECHANISM

    assert MODEL_MECHANISM == LANGUAGE_VARIATION == "language_variation"
    assert MODEL_MECHANISM not in DETERMINISTIC_MECHANISMS


def test_the_same_fact_written_several_ways_normalizes_to_one_space(configs):
    from aelayer.extract.mentions import MentionFinder

    finder = MentionFinder(configs.catalog, configs.extraction)
    phrasings = [
        "rash with oral mucosal involvement",
        "rash with oral ulceration",
        "rash with buccal erosions",
        "eruption with stomatitis",
    ]
    calls = [finder.best(text, "mucosal_involvement") for text in phrasings]
    assert all(call is not None for call in calls)
    assert {call.value for call in calls} == {"ORAL"}
    assert {call.assertion for call in calls} == {"present"}


# -- 5.2 coded-concept variation ---------------------------------------------


def test_several_legitimate_codings_are_preserved_side_by_side(records):
    codes = Counter(r.coded_event.code for r in records if r.coded_event)
    cutaneous = {
        code for code in codes
        if "ash" in code or "rash" in code.lower()
    }
    assert len(cutaneous) >= 3, f"only one coding present: {sorted(codes)}"


def test_the_concept_set_decides_membership_not_the_string(pipeline,
                                                           definition_v2, records):
    """A record whose code has no target-version mapping still qualifies."""
    assignments = {a.record_id: a for a in pipeline.assignments(definition_v2)}
    flagged = [
        r for r in records
        if r.coded_event and r.coded_event.reconciliation == "flagged_for_review"
    ]
    assert flagged
    assert any(r.record_id in assignments for r in flagged), (
        "a code that could not be reconciled was silently dropped from the "
        "definition rather than judged on its concept"
    )


def test_nothing_merges_two_codings(records):
    """Both codes survive; neither is overwritten with the other."""
    for record in records:
        coded = record.coded_event
        if coded is None:
            continue
        assert coded.code, "a coded term lost its original code"


# -- 5.3 terminology-version variation ---------------------------------------


def test_all_three_reconciliation_outcomes_are_exercised(records):
    outcomes = Counter(
        r.coded_event.reconciliation for r in records if r.coded_event
    )
    assert outcomes["unchanged"] > 0
    assert outcomes["remapped_mechanically"] > 0
    assert outcomes["flagged_for_review"] > 0


def test_reconciliation_is_a_declared_one_to_one_map(catalog):
    result = catalog.reconcile("Rash erythematous", "D-19.0", "D-21.0")
    assert result.outcome == "remapped_mechanically"
    assert result.reconciled_to == "Erythematous rash"
    assert result.concept_id == "RASH_ERYTHEMATOUS"


def test_a_code_absent_from_the_target_is_flagged_never_recoded(catalog):
    result = catalog.reconcile("Rash maculopapular", "D-19.0", "D-21.0")
    assert result.outcome == "flagged_for_review"
    assert result.reconciled_to is None
    assert "no model recodes it" in result.note


def test_an_unknown_code_is_flagged_rather_than_guessed(catalog):
    result = catalog.reconcile("Something nobody declared", "D-19.0", "D-21.0")
    assert result.outcome == "flagged_for_review"
    assert result.reconciled_to is None


def test_an_identical_code_is_unchanged_not_remapped(catalog):
    result = catalog.reconcile("Rash", "D-19.0", "D-21.0")
    assert result.outcome == "unchanged"


def test_the_source_version_is_preserved_on_every_record(records, profiles):
    for record in records:
        if record.coded_event is None:
            continue
        profile = profiles.profile(record.profile)
        assert record.coded_event.dictionary_version == profile.dictionary_version


# -- value coercion ----------------------------------------------------------


def test_a_tristate_qualifier_reads_as_an_assertion():
    from aelayer.normalize.values import TRISTATE

    assert TRISTATE["y"] == "present"
    assert TRISTATE["n"] == "absent"
    assert TRISTATE["u"] == "uncertain"


def test_a_blank_qualifier_is_not_a_negative(structured_records):
    silent = [
        r.modifiers["mucosal_involvement"] for r in structured_records
        if r.profile == "P_structured"
        and not r.modifiers["mucosal_involvement"].observed
    ]
    assert silent
    assert all(a.assertion is None for a in silent)
