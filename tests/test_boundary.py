"""The boundary between the deterministic path and the model path.

A value the study already settled is never a question for a model. Not a
convention — a function every request passes through.
"""

from __future__ import annotations

import pytest

from aelayer.extract.backends import ExtractionRequest
from aelayer.guards import (
    SETTLED,
    BoundaryViolation,
    askable_attributes,
    assert_model_path_permitted,
    unresolved_attributes,
)
from aelayer.models import STRUCTURED_SOURCES, Attribute, Span


def request(**kwargs):
    base = dict(
        doc_id="AE:R1:AETERM", text="rash on the chest", attributes=("location",),
        concept_id="RASH", source_kind="reported_term", source_variable="AETERM",
    )
    base.update(kwargs)
    return ExtractionRequest(**base)


# -- what the guard refuses -------------------------------------------------


def test_no_structured_value_ever_reaches_the_model_path(records, profiles):
    """Asserted over every record in the corpus, not on one example."""
    for record in records:
        profile = profiles.for_study(record.study_id)
        for attribute in askable_attributes(record, profile,
                                            _extraction_config(record)):
            current = record.attribute(attribute)
            assert current is not None
            assert not (current.populated and current.source in STRUCTURED_SOURCES)


def _extraction_config(_record):
    from aelayer.catalog import load_configs

    return load_configs().extraction


def test_a_request_that_reads_a_structured_variable_is_refused(records, profiles):
    record = records[0]
    profile = profiles.for_study(record.study_id)
    with pytest.raises(BoundaryViolation, match="read by the deterministic path"):
        assert_model_path_permitted(
            request(source_kind="structured_standard"), record, profile
        )


def test_a_request_naming_a_settled_attribute_is_refused(records, profiles):
    record = next(r for r in records if r.location.populated)
    profile = profiles.for_study(record.study_id)
    with pytest.raises(BoundaryViolation, match="not a question for a model"):
        assert_model_path_permitted(request(), record, profile)


def test_a_request_naming_an_attribute_that_does_not_exist_is_refused(
    records, profiles
):
    record = records[0]
    profile = profiles.for_study(record.study_id)
    with pytest.raises(BoundaryViolation, match="not an attribute of"):
        assert_model_path_permitted(request(attributes=("vibes",)), record, profile)


def test_a_model_request_carries_text_and_nothing_else(records, profiles):
    record = records[0]
    profile = profiles.for_study(record.study_id)
    with pytest.raises(BoundaryViolation, match="carries text"):
        assert_model_path_permitted(request(text=None), record, profile)


def test_the_extraction_config_may_not_declare_a_structured_source_readable():
    from aelayer.catalog import ConfigError, ExtractionConfig

    with pytest.raises(ConfigError, match="not a question for a model"):
        ExtractionConfig({
            "readable_sources": ["structured_standard"],
            "extractable_attributes": ["location"],
            "modifiers": {},
        })


# -- what the guard permits -------------------------------------------------


def test_a_gated_or_representable_answer_is_settled():
    assert "collected" in SETTLED
    assert "not_applicable_gated" in SETTLED
    assert "not_representable" in SETTLED
    # Deliberately absent: a study that never asked may still have written the
    # answer in prose, which is the whole point of the layer.
    assert "not_collected_by_protocol" not in SETTLED


def test_the_askable_set_follows_each_study_convention(records, profiles, configs):
    askable_by_profile = {}
    for record in records:
        if record.standardized_concept != "RASH":
            continue
        profile = profiles.for_study(record.study_id)
        askable_by_profile.setdefault(record.profile, set()).update(
            askable_attributes(record, profile, configs.extraction)
        )
    # A study whose location lives in a structured variable is never asked
    # about it; one whose location lives in prose always is.
    assert "location" not in askable_by_profile.get("P1_structured", set())
    assert "location" not in askable_by_profile.get("P4_sponsor", set())
    assert "location" in askable_by_profile.get("P2_text", set())
    assert "location" in askable_by_profile.get("P5_comment", set())


def test_unresolved_attributes_are_the_open_questions(records):
    record = next(r for r in records if r.profile == "P3_prespecified")
    unresolved = unresolved_attributes(record)
    assert "location" in unresolved
    assert "coded_event" not in unresolved


# -- abstention -------------------------------------------------------------


def test_abstention_is_recorded_rather_than_guessed(records):
    abstained = [
        r for r in records
        if not r.location.populated and "abstained" in r.location.note
    ]
    assert abstained, "the model path never abstained, so abstention is untested"
    for record in abstained:
        assert record.location.value is None
        assert record.location.availability != "collected"


def test_a_recovered_value_says_it_came_from_text(records):
    recovered = [r for r in records if r.location.method == "extracted"]
    assert recovered
    for record in recovered:
        assert record.location.source in ("reported_term", "comment")
        assert record.location.evidence
        assert record.location.extractor_version
        span = record.location.evidence[0]
        assert span.kind == "text"


def test_every_populated_attribute_traces_to_a_span(records):
    violations = {r.source_record_id: r.missing_provenance() for r in records
                  if not r.has_full_provenance()}
    assert not violations, violations
