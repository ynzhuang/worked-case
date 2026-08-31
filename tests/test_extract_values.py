"""Lab units, and the strict separation of severity from seriousness."""

from __future__ import annotations

import datetime as _dt

import pytest

from aelayer.extract.values import ValueExtractor


@pytest.fixture(scope="module")
def values(catalog, extraction_config):
    return ValueExtractor(catalog, extraction_config)


@pytest.mark.parametrize(
    "text,value,unit,canonical",
    [
        ("Capillary glucose was 48 mg/dL.", 48.0, "mg/dL", 48.0),
        ("Fingerstick glucose 3.1 mmol/L was recorded.", 3.1, "mmol/L", 55.8564),
        ("A blood glucose of 64.0 mg/dL was documented.", 64.0, "mg/dL", 64.0),
        ("Plasma glucose measured 2.9 mmol/L.", 2.9, "mmol/L", 52.2528),
    ],
)
def test_lab_values_are_converted_to_canonical_units(values, text, value, unit, canonical):
    hits = values.find_labs(text)
    assert len(hits) == 1
    assert hits[0].value == value and hits[0].unit == unit
    assert hits[0].canonical_value == pytest.approx(canonical, abs=1e-3)
    assert hits[0].canonical_unit == "mg/dL"


def test_a_mmol_value_and_a_mgdl_value_compare_correctly(values):
    """The whole point of conversion: 3.1 mmol/L is not below a 70 mg/dL bar
    by numeric accident, it is below it because 3.1 mmol/L *is* 55.9 mg/dL."""
    si = values.find_labs("Glucose 3.1 mmol/L.")[0]
    us = values.find_labs("Glucose 55 mg/dL.")[0]
    assert si.value < us.value                       # 3.1 < 55, naively
    assert si.canonical_value == pytest.approx(us.canonical_value, abs=1.0)


def test_missing_units_are_inferred_only_where_magnitude_is_unambiguous(values):
    assert values.find_labs("Capillary glucose was 48.")[0].unit == "mg/dL"
    assert values.find_labs("Capillary glucose was 3.1.")[0].unit == "mmol/L"
    # Between the ranges: ambiguous, so nothing is reported rather than guessed.
    assert values.find_labs("Capillary glucose was 32.") == []


def test_implausible_values_are_not_offered_as_evidence(values):
    hits = [h for h in values.find_labs("Glucose 999 mg/dL.") if not h.implausible]
    assert hits == []


def test_severity_and_seriousness_never_write_to_each_other(values):
    """A mild event that put the subject in hospital.

    Severity must stay `mild`; seriousness must record hospitalisation. Any
    model that treats them as one graded scale gets this wrong.
    """
    text = (
        "The subject had a mild hypoglycaemic episode. "
        "The subject was admitted to hospital for observation."
    )
    assert values.single_value(text, "severity").value == "mild"
    assert [h.value for h in values.multi_value(text, "seriousness")] == ["hospitalisation"]


def test_a_severe_event_can_be_non_serious(values):
    text = "The subject had a severe hypoglycaemic episode. The event resolved."
    assert values.single_value(text, "severity").value == "severe"
    assert values.multi_value(text, "seriousness") == []


@pytest.mark.parametrize(
    "field,text,expected",
    [
        ("action_taken", "Study drug was held for 48 hours.", "dose_interrupted"),
        ("action_taken", "The dose was reduced at the next visit.", "dose_reduced"),
        ("action_taken", "The subject was permanently discontinued.", "drug_withdrawn"),
        ("outcome", "The event resolved the same day.", "resolved"),
        ("outcome", "The event was resolving at last contact.", "resolving"),
        ("outcome", "The event was ongoing at the end of the period.", "not_resolved"),
        ("relatedness", "Assessed as possibly related to study drug.", "possible"),
        ("relatedness", "Considered unrelated to study drug.", "not_related"),
        ("rechallenge", "On rechallenge the event recurred.", "done_recurred"),
        ("rechallenge", "Rechallenged without recurrence.", "done_no_recurrence"),
        ("rechallenge", "Rechallenge was not performed.", "not_done"),
    ],
)
def test_enumerated_fields_from_narrative_cues(values, field, text, expected):
    assert values.single_value(text, field).value == expected


def test_the_longest_cue_wins(values):
    """`not resolved` contains `resolved` and must beat it."""
    assert values.single_value("The event was not resolved.", "outcome").value == "not_resolved"


def test_rescue_treatment_is_often_the_only_trace_of_a_mild_event(values):
    assert values.rescue_treatment("Oral glucose gel was administered.") is not None
    assert values.rescue_treatment("The subject rested quietly.") is None


def test_structured_ae_columns_are_preserved_and_used(values):
    row = {"AESEV": "severe", "AEOUT": "recovered", "AEACN": "drug_withdrawn",
           "AESCAT": "hospitalisation|death"}
    assert values.from_ae_row(row, "severity") == "severe"
    assert values.from_ae_row(row, "outcome") == "resolved"
    assert values.from_ae_row(row, "action_taken") == "drug_withdrawn"
    assert values.seriousness_from_ae_row(row) == ["death", "hospitalisation"]


def test_blank_structured_columns_yield_nothing(values):
    """Older studies leave these to the narrative."""
    row = {"AESEV": "", "AEOUT": None, "AESCAT": ""}
    assert values.from_ae_row(row, "severity") is None
    assert values.from_ae_row(row, "outcome") is None
    assert values.seriousness_from_ae_row(row) == []


def test_structured_labs_are_restricted_to_the_day_of_the_event(values):
    rows = [
        {"USUBJID": "S1", "LBSEQ": 1, "LBTESTCD": "GLUC", "LBORRES": 52,
         "LBORRESU": "mg/dL", "LBDTC": "2024-02-07"},
        {"USUBJID": "S1", "LBSEQ": 2, "LBTESTCD": "GLUC", "LBORRES": 44,
         "LBORRESU": "mg/dL", "LBDTC": "2024-02-20"},
    ]
    hits = values.labs_from_lb(rows, _dt.date(2024, 2, 7))
    assert [h.value for h in hits] == [52]
    assert values.labs_from_lb(rows, None) == []
