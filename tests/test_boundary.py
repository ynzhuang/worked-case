"""The deterministic/model boundary, enforced over the whole corpus.

Three separable claims, each tested rather than documented:

1. a value the study already settled never reaches a backend
2. the model path is used for **language variation only** — coded-concept
   variation and terminology-version variation are deterministic and cannot
   even be requested
3. no coded field is ever modified by anything
"""

from __future__ import annotations

import pytest

from aelayer.catalog import ConfigError, ExtractionConfig
from aelayer.extract import ExtractionEngine, ExtractionRequest
from aelayer.guards import (
    SETTLED,
    BoundaryViolation,
    askable_modifiers,
    assert_coded_field_untouched,
    assert_model_path_permitted,
)
from aelayer.models import STRUCTURED_SOURCES

MODIFIER = "mucosal_involvement"


def _request(record, **overrides):
    body = dict(
        doc_id="AE:R1:AETERM", text="rash with oral ulceration",
        modifiers=(MODIFIER,), source_kind="reported_term",
        source_variable="AETERM",
    )
    body.update(overrides)
    return ExtractionRequest(**body)


# -- 1. nothing settled is ever asked ----------------------------------------


def test_no_settled_value_is_ever_sent_to_a_backend(pipeline, configs):
    """Over every record in the corpus, not on a sample."""
    engine = ExtractionEngine.build(configs, pipeline.store, "rules")
    asked: list[tuple[str, str]] = []
    for record in pipeline.structured_only_records():
        profile = configs.profiles.for_study(record.study_id)
        for modifier in askable_modifiers(record, profile, configs.extraction):
            asked.append((record.record_id, modifier))
            current = record.attribute(modifier)
            assert current.availability not in SETTLED
            assert current.source not in STRUCTURED_SOURCES
    assert asked, "the boundary test is vacuous if nothing was ever asked"
    assert engine.backend.name == "rules"


def test_a_settled_attribute_is_refused_by_name(pipeline, configs):
    record = next(
        r for r in pipeline.structured_only_records()
        if r.modifiers.get(MODIFIER) and r.modifiers[MODIFIER].observed
    )
    profile = configs.profiles.for_study(record.study_id)
    with pytest.raises(BoundaryViolation) as exc:
        assert_model_path_permitted(_request(record), record, profile)
    assert "not a question for a model" in str(exc.value)


def test_a_structured_source_kind_is_refused(pipeline, configs):
    record = pipeline.structured_only_records()[0]
    profile = configs.profiles.for_study(record.study_id)
    with pytest.raises(BoundaryViolation) as exc:
        assert_model_path_permitted(
            _request(record, source_kind="structured_standard"), record, profile
        )
    assert "never by a model" in str(exc.value)


def test_a_request_carrying_something_other_than_text_is_refused(pipeline, configs):
    record = pipeline.structured_only_records()[0]
    profile = configs.profiles.for_study(record.study_id)

    class Bad:
        text = None
        mechanism = "language_variation"
        source_kind = "reported_term"
        modifiers = (MODIFIER,)

    with pytest.raises(BoundaryViolation):
        assert_model_path_permitted(Bad(), record, profile)


def test_a_structured_kind_cannot_even_be_declared_readable(configs):
    raw = {**configs.extraction.raw, "readable_sources": ["structured_standard"]}
    with pytest.raises(ConfigError) as exc:
        ExtractionConfig(raw)
    assert "already settled" in str(exc.value)


# -- 2. a model is used for language variation only --------------------------


@pytest.mark.parametrize(
    "mechanism", ["coded_concept_variation", "terminology_version_variation"]
)
def test_deterministic_mechanisms_can_never_reach_a_backend(
    pipeline, configs, mechanism
):
    record = next(
        r for r in pipeline.structured_only_records()
        if r.modifiers.get(MODIFIER) and not r.modifiers[MODIFIER].observed
    )
    profile = configs.profiles.for_study(record.study_id)
    with pytest.raises(BoundaryViolation) as exc:
        assert_model_path_permitted(
            _request(record, mechanism=mechanism), record, profile
        )
    assert "never rewrites a coded field" in str(exc.value)


def test_an_undeclared_mechanism_is_refused(pipeline, configs):
    record = next(
        r for r in pipeline.structured_only_records()
        if r.modifiers.get(MODIFIER) and not r.modifiers[MODIFIER].observed
    )
    profile = configs.profiles.for_study(record.study_id)
    with pytest.raises(BoundaryViolation) as exc:
        assert_model_path_permitted(
            _request(record, mechanism="vibes"), record, profile
        )
    assert "and nothing else" in str(exc.value)


def test_every_extracted_value_came_from_a_readable_text_source(records, configs):
    readable = set(configs.extraction.readable_sources)
    extracted = [
        r.modifiers[MODIFIER] for r in records
        if r.modifiers.get(MODIFIER) and r.modifiers[MODIFIER].method == "extracted"
    ]
    assert extracted
    for attribute in extracted:
        assert attribute.source in readable
        assert attribute.evidence


# -- 3. no coded field is modified -------------------------------------------


def test_the_model_path_never_modifies_a_coded_field(pipeline, records):
    before = {r.record_id: r for r in pipeline.structured_only_records()}
    for after in records:
        assert_coded_field_untouched(before[after.record_id], after)


def test_a_changed_code_is_caught(pipeline):
    before = next(
        r for r in pipeline.structured_only_records() if r.coded_event
    )
    after = before.model_copy(deep=True)
    after.coded_event.code = "Something else"
    with pytest.raises(BoundaryViolation) as exc:
        assert_coded_field_untouched(before, after)
    assert "No model ever rewrites a coded field" in str(exc.value)


def test_reconciliation_never_overwrites_the_original(records):
    """Preserve original codes: `reconciled_to` sits beside `code`, never on it."""
    remapped = [
        r.coded_event for r in records
        if r.coded_event and r.coded_event.reconciliation == "remapped_mechanically"
    ]
    assert remapped, "no record exercises a mechanical remap"
    for coded in remapped:
        assert coded.reconciled_to != coded.code
        assert coded.code  # the original survives
        assert coded.effective_code == coded.reconciled_to


def test_a_code_with_no_mapping_is_flagged_not_recoded(records):
    flagged = [
        r.coded_event for r in records
        if r.coded_event and r.coded_event.reconciliation == "flagged_for_review"
    ]
    assert flagged, "no record exercises the flagged-for-review path"
    for coded in flagged:
        assert coded.reconciled_to is None
        assert coded.effective_code == coded.code
        assert "human decides" in coded.note or "not a code" in coded.note
