"""The event object's contract."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aelayer.models import (
    CaseDefinition,
    AssertionPolicy,
    EventObject,
    LabValue,
    Span,
    WindowSpec,
)


def span(field: str, value: str = "x") -> Span:
    return Span(doc_id="d", start=0, end=3, field=field, extracted_value=value, text="abc")


def test_span_rejects_inverted_range():
    with pytest.raises(ValidationError):
        Span(doc_id="d", start=9, end=2, field="assertion", extracted_value="present")


def test_missing_provenance_names_the_unbacked_field():
    event = EventObject(
        event_id="e", subject_id="s", study_id="st", doc_id="d",
        concept_id="HYPOGLYCEMIA", severity="mild",
        evidence=[span("concept_id"), span("assertion")],
    )
    assert event.missing_provenance() == ["severity"]
    assert not event.has_full_provenance()


def test_full_provenance_when_every_populated_field_has_a_span():
    event = EventObject(
        event_id="e", subject_id="s", study_id="st", doc_id="d",
        concept_id="HYPOGLYCEMIA", severity="mild", seriousness=["hospitalisation"],
        evidence=[
            span("concept_id"), span("assertion"), span("severity"),
            span("seriousness"),
        ],
    )
    assert event.has_full_provenance()


def test_null_fields_need_no_span():
    event = EventObject(
        event_id="e", subject_id="s", study_id="st", doc_id="d",
        concept_id="C", evidence=[span("concept_id"), span("assertion")],
    )
    assert event.severity is None
    assert event.has_full_provenance()


def test_severity_and_seriousness_are_independent_fields():
    """A mild event can be serious. The model must permit it."""
    event = EventObject(
        event_id="e", subject_id="s", study_id="st", doc_id="d",
        concept_id="C", severity="mild", seriousness=["hospitalisation", "death"],
        evidence=[
            span("concept_id"), span("assertion"), span("severity"),
            span("seriousness"),
        ],
    )
    assert event.severity == "mild"
    assert "hospitalisation" in event.seriousness
    # And a severe event can carry no seriousness at all.
    other = event.model_copy(update={"severity": "severe", "seriousness": []})
    assert other.severity == "severe" and other.seriousness == []


def test_assertion_is_a_field_not_a_confidence_discount():
    event = EventObject(
        event_id="e", subject_id="s", study_id="st", doc_id="d",
        concept_id="C", assertion="absent", confidence={"assertion": 0.95},
        evidence=[span("concept_id"), span("assertion", "absent")],
    )
    assert event.assertion == "absent"
    assert event.confidence["assertion"] > 0.9


def test_window_rejects_inverted_bounds():
    with pytest.raises(ValidationError):
        WindowSpec(min=14, max=0)


def test_window_contains_is_inclusive():
    window = WindowSpec(min=0, max=14)
    assert window.contains(0) and window.contains(14)
    assert not window.contains(15)


def test_assertion_policy_rejects_a_class_in_two_buckets():
    with pytest.raises(ValidationError):
        AssertionPolicy(require=["present"], exclude=["present"])


def test_case_definition_maps_states_to_verdicts():
    definition = CaseDefinition()
    assert definition.verdict_for("explicit") == "case"
    assert definition.verdict_for("possible") == "review"
    assert definition.verdict_for("absent") == "excluded"


def test_lab_value_keeps_reported_and_canonical_units():
    value = LabValue(
        test="GLUCOSE", value=3.1, unit="mmol/L", canonical_value=55.86,
        canonical_unit="mg/dL", span=span("labs"),
    )
    assert value.value == 3.1 and value.unit == "mmol/L"
    assert value.canonical_value == 55.86 and value.canonical_unit == "mg/dL"
