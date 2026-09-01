"""A blank is not a value.

The whole point of ``Field`` is that four different kinds of empty stay
distinguishable all the way to the phenotype rule that consults them.
"""

from __future__ import annotations

import collections

import pytest
from pydantic import ValidationError

from aelayer.models import (
    COLLECTION_STATES,
    Field,
    MissingnessPolicy,
    Span,
)


def span(field: str = "severity") -> Span:
    return Span(doc_id="AE:1", field=field, extracted_value="x", kind="structured")


def test_a_collected_value_can_support_an_inference():
    field = Field[str].collected("mild", [span()])
    assert field.populated and field.is_evidence_of_absence


@pytest.mark.parametrize(
    "state",
    ["not_collected_by_protocol", "not_applicable_gated", "pending_ongoing",
     "intentionally_blank", "not_representable", "unknown"],
)
def test_no_kind_of_blank_supports_an_inference(state):
    """Only a value the study actually collected can carry that weight."""
    assert not Field[str].missing(state).is_evidence_of_absence


def test_a_missing_field_cannot_claim_to_be_collected():
    with pytest.raises(ValueError, match="cannot be in state 'collected'"):
        Field[str].missing("collected")


def test_a_populated_field_needs_a_span():
    assert not Field[str](value="mild", collection_state="collected").has_provenance()
    assert Field[str].collected("mild", [span()]).has_provenance()


def test_the_structured_state_survives_a_narrative_recovery():
    """A value recovered from text does not make the CRF column collected."""
    field = Field[str](
        value="mild", collection_state="collected", source="text",
        spans=[span()], prior_state="not_collected_by_protocol",
    )
    assert field.populated
    assert field.structured_state == "not_collected_by_protocol"


def test_a_definition_may_not_read_an_uncollected_field_as_absence():
    for state in ("not_collected_by_protocol", "not_applicable_gated"):
        with pytest.raises(ValidationError, match="cannot be treated as evidence of absence"):
            MissingnessPolicy(treat_as_absent=[state])


def test_a_state_cannot_be_both_absent_and_routed_to_review():
    with pytest.raises(ValidationError, match="both treated as absent"):
        MissingnessPolicy(treat_as_absent=["unknown"], route_to_review=["unknown"])


def test_the_shipped_definition_assumes_nothing_absent(definition_v1):
    assert definition_v1.missingness.treat_as_absent == []


# -- the corpus exercises every state ---------------------------------------


def test_every_collection_state_appears_in_the_corpus(store):
    seen = {
        state
        for row in store.gold_records()
        for state in row["collection_states"].values()
    }
    assert seen == set(COLLECTION_STATES) - {"collected"} | {"collected"}


def test_the_pipeline_reproduces_the_gold_collection_states(records, store):
    gold = store.gold_records_by_id()
    disagreements = collections.Counter()
    total = 0
    for record in records:
        truth = gold.get(record.source_record_id)
        if truth is None:
            continue
        for name, field in record.fields().items():
            want = truth["collection_states"].get(name)
            if want is None:
                continue
            total += 1
            if want != field.structured_state:
                disagreements[(want, field.structured_state)] += 1
    assert total > 0
    assert disagreements == collections.Counter(), dict(disagreements)


def test_a_gated_criterion_is_not_unknown(records):
    """The gate was answered No, so the criterion was never applicable."""
    gated = [
        record for record in records
        if record.seriousness.value is False
        and any(
            f.collection_state == "not_applicable_gated"
            for f in record.seriousness_criteria.values()
        )
    ]
    assert gated
    for record in gated:
        for field in record.seriousness_criteria.values():
            assert field.collection_state == "not_applicable_gated"
            assert field.value is None


def test_a_restricted_codelist_yields_not_representable(records, semantics):
    """The concept exists; the field cannot express it."""
    restricted = [
        r for r in records
        if r.action_taken.collection_state == "not_representable"
    ]
    assert restricted, "STUDY-05 should have unrepresentable actions"
    for record in restricted:
        assert record.action_taken.value is None
        assert "no permissible" in record.action_taken.note
        assert semantics.for_study(record.study_id).codelist_for("action_taken")


def test_a_study_that_never_collected_a_field_says_so(records):
    absent = [
        r for r in records
        if r.severity.structured_state == "not_collected_by_protocol"
    ]
    assert absent
    assert {r.study_id for r in absent} == {"STUDY-04"}


def test_an_ongoing_event_has_a_pending_end_not_a_missing_one(records):
    pending = [r for r in records if r.end_datetime.collection_state == "pending_ongoing"]
    assert pending
    for record in pending:
        assert record.end_datetime.value is None
        assert record.outcome.value not in ("recovered", "recovered_with_sequelae", "fatal")


def test_a_protocol_instructed_blank_is_not_an_unanswered_question(records):
    instructed = [
        r for r in records
        if r.relatedness.structured_state == "intentionally_blank"
    ]
    assert instructed
    assert {r.study_id for r in instructed} == {"STUDY-03"}
