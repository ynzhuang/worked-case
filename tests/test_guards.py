"""The deterministic/model boundary.

Definition of done: no controlled value is ever sent to the model path.
"""

from __future__ import annotations

import collections

import pytest

from aelayer.guards import (
    ControlledValueLeak,
    DETERMINISTIC_FIELDS,
    ModelRequest,
    SETTLED_STATES,
    assert_model_path_permitted,
    assert_no_structured_payload,
    unresolved_fields,
)


def test_a_model_request_can_only_carry_text():
    """Structurally: there is nowhere to put a controlled value."""
    request = ModelRequest(
        doc_id="D", text="t", requested_fields=("symptoms",),
        schema_name="CanonicalAERecord", prompt_version="p",
    )
    assert set(vars(request)) == {
        "doc_id", "text", "requested_fields", "schema_name", "prompt_version",
        "record_id",
    }


def test_a_request_for_an_undefined_field_is_refused():
    with pytest.raises(ControlledValueLeak, match="no defined source"):
        ModelRequest(
            doc_id="D", text="t", requested_fields=("bank_details",),
            schema_name="s", prompt_version="p",
        )


def test_no_collected_field_is_ever_askable(records):
    """The guarantee, asserted over the whole corpus."""
    for record in records:
        askable = set(unresolved_fields(record))
        for name, field in record.fields().items():
            if name in DETERMINISTIC_FIELDS and field.structured_state == "collected":
                assert name not in askable, (record.source_record_id, name)


def test_a_gated_field_is_settled_not_open(records):
    """A parent gate answered No is a fact the study recorded."""
    assert "not_applicable_gated" in SETTLED_STATES
    for record in records:
        askable = set(unresolved_fields(record))
        for criterion, field in record.seriousness_criteria.items():
            if field.collection_state == "not_applicable_gated":
                assert f"seriousness_criteria.{criterion}" not in askable


def test_asking_about_a_collected_field_raises(records):
    record = next(r for r in records if r.severity.structured_state == "collected")
    request = ModelRequest(
        doc_id="D", text="t", requested_fields=("severity",),
        schema_name="s", prompt_version="p",
    )
    with pytest.raises(ControlledValueLeak, match="already settled"):
        assert_model_path_permitted(request, record)


def test_asking_about_an_already_standardized_concept_raises(records):
    record = next(r for r in records if r.standardized_concept)
    request = ModelRequest(
        doc_id="D", text="t", requested_fields=("standardized_concept",),
        schema_name="s", prompt_version="p",
    )
    with pytest.raises(ControlledValueLeak):
        assert_model_path_permitted(request, record)


def test_a_structured_record_cannot_reach_a_backend(records):
    with pytest.raises(ControlledValueLeak, match="only a ModelRequest"):
        assert_no_structured_payload(records[0], where="test")


def test_the_askable_set_reflects_each_study_convention(records, semantics):
    """V-D collected almost nothing, so almost everything is open there."""
    by_representation = collections.defaultdict(set)
    for record in records:
        rep = semantics.for_study(record.study_id).representation
        by_representation[rep] |= set(unresolved_fields(record))
    assert "severity" in by_representation["V-D"]
    assert "severity" not in by_representation["V-A"]
    for representation in by_representation:
        assert {"symptoms", "assertion", "labs"} <= by_representation[representation]


def test_recovered_values_are_marked_as_coming_from_text(records):
    """A field with a structured counterpart, filled from prose, records both.

    Text-only fields such as `assertion` have no structured counterpart and are
    excluded: there was never a column for them to have been collected in.
    """
    recovered = [
        (r, name) for r in records for name, f in r.fields().items()
        if f.source == "text" and f.value is not None and name in DETERMINISTIC_FIELDS
    ]
    assert recovered
    for record, name in recovered:
        field = record.fields()[name]
        assert field.spans, f"{record.source_record_id}.{name} has no span"
        assert field.structured_state != "collected"
        assert field.prior_state is not None


def test_the_model_path_never_invents_a_coded_term(records):
    """A dictionary term is assigned by a coder, not recovered from prose."""
    for record in records:
        if record.coded_term.value is not None:
            assert record.coded_term.source == "structured"
