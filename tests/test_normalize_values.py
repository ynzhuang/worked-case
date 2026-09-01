"""Deterministic coercion. Nothing here guesses, and nothing returns a bare None."""

from __future__ import annotations

import datetime as _dt

import pytest

from aelayer.normalize.values import (
    coerce_bool,
    coerce_datetime,
    coerce_enum,
    is_blank,
    to_canonical_unit,
    unresolved_concept,
)
from aelayer.semantics import CollectionSemantics


@pytest.fixture
def study():
    return CollectionSemantics({"studies": {"ST": {
        "collected_fields": ["severity", "relatedness", "outcome",
                             "action_taken", "onset_datetime", "seriousness"],
        "blank_means": {"outcome": "pending_ongoing"},
        "restricted_codelists": {
            "action_taken": {
                "permissible": ["dose_not_changed", "drug_withdrawn"],
                "absent_concepts": ["dose_reduced"],
            }
        },
    }}}).for_study("ST")


# -- blanks -----------------------------------------------------------------


@pytest.mark.parametrize("cell", [None, "", "   "])
def test_a_blank_is_recognised_in_every_shape_it_arrives_in(cell):
    assert is_blank(cell)


def test_a_blank_asks_the_study_what_the_blank_means(study):
    assert coerce_enum("outcome", "", study) == (None, "pending_ongoing", "")
    assert coerce_enum("action_taken", None, study)[1] == "unknown"
    assert coerce_enum("dictionary", "", study)[1] == "not_collected_by_protocol"


# -- controlled values ------------------------------------------------------


@pytest.mark.parametrize("cell,expected", [
    ("Grade 3", "severe"), ("severe", "severe"), ("2", "moderate"),
    ("MILD", "mild"),
])
def test_spelling_variants_map_to_one_canonical_value(cell, expected, study):
    assert coerce_enum("severity", cell, study) == (expected, "collected", "")


def test_a_value_outside_the_codelist_is_reported_not_coerced(study):
    value, state, note = coerce_enum("severity", "catastrophic", study)
    assert (value, state) == (None, "unknown")
    assert "not in the canonical codelist" in note


def test_a_value_the_study_itself_does_not_permit_is_a_data_question(study):
    """Recorded, but outside this study's own codelist. Report the discrepancy."""
    value, state, note = coerce_enum("action_taken", "dose reduced", study)
    assert (value, state) == (None, "unknown")
    assert "not permissible" in note
    assert "ST" in note


def test_a_permissible_value_passes_through(study):
    assert coerce_enum("action_taken", "drug withdrawn", study) == (
        "drug_withdrawn", "collected", ""
    )


# -- what a restricted codelist cannot say ---------------------------------


def test_a_concept_the_codelist_cannot_express_is_not_representable(study):
    state, note = unresolved_concept("action_taken", "dose_reduced", study)
    assert state == "not_representable"
    assert "left unresolved rather than coerced" in note


def test_a_concept_the_codelist_can_express_falls_back_to_the_blank_meaning(study):
    assert unresolved_concept("action_taken", "drug_withdrawn", study)[0] == "unknown"
    assert unresolved_concept("action_taken", None, study)[0] == "unknown"
    assert unresolved_concept("severity", "anything", study)[0] == "unknown"


# -- booleans and dates -----------------------------------------------------


@pytest.mark.parametrize("cell,expected", [("Y", True), ("yes", True),
                                           ("N", False), ("0", False)])
def test_yes_and_no_are_read_in_their_usual_spellings(cell, expected, study):
    assert coerce_bool("seriousness", cell, study) == (expected, "collected", "")


def test_an_unrecognised_yes_no_is_unknown_with_a_reason(study):
    value, state, note = coerce_bool("seriousness", "maybe", study)
    assert (value, state) == (None, "unknown")
    assert "yes/no" in note


@pytest.mark.parametrize("cell", ["2024-02-06", "2024-02-06T09:30",
                                  "2024-02-06T09:30:00"])
def test_a_full_date_is_parsed(cell, study):
    value, state, _note = coerce_datetime("onset_datetime", cell, study)
    assert state == "collected"
    assert value.date() == _dt.date(2024, 2, 6)


@pytest.mark.parametrize("cell", ["2024", "2024-02"])
def test_a_partial_date_is_not_silently_rounded(cell, study):
    """A month is a real fact, and not one that supports day arithmetic."""
    value, state, note = coerce_datetime("onset_datetime", cell, study)
    assert (value, state) == (None, "unknown")
    assert "partial date" in note


def test_an_unparsable_date_says_so(study):
    value, state, note = coerce_datetime("onset_datetime", "last Tuesday", study)
    assert (value, state) == (None, "unknown")
    assert "not a parsable date" in note


# -- units ------------------------------------------------------------------


def test_a_value_is_converted_into_the_canonical_unit():
    """A threshold applied to an unconverted mmol/L value misclassifies a study."""
    assert to_canonical_unit(3.0, "mmol/L", {"mg/dL": 1.0, "mmol/L": 18.0182}) == 54.0546


def test_the_unit_match_is_case_insensitive():
    assert to_canonical_unit(3.0, "MMOL/L", {"mmol/L": 18.0182}) is not None


def test_an_unknown_unit_converts_to_nothing_rather_than_to_itself():
    assert to_canonical_unit(3.0, "furlongs", {"mg/dL": 1.0}) is None
