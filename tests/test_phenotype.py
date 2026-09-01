"""Rule evaluation over episodes, and how it reads a blank."""

from __future__ import annotations

import datetime as _dt

import pytest
import yaml

from aelayer.anchors import AnchorResolver
from aelayer.models import (
    CanonicalAEEpisode,
    Field,
    LabValue,
    Span,
    SymptomMention,
)
from aelayer.phenotype.evaluator import PhenotypeEvaluator
from aelayer.phenotype.loader import DefinitionError, load_definition


def span(field: str = "x") -> Span:
    return Span(doc_id="d", field=field, extracted_value="x", text="x")


def episode(**kwargs) -> CanonicalAEEpisode:
    symptoms = kwargs.pop("symptoms", [])
    labs = kwargs.pop("labs", [])
    base = dict(
        episode_id="E1", study_id="ST", subject_id="S1",
        standardized_concept="HYPOGLYCEMIA",
        episode_start=Field[_dt.datetime].collected(
            _dt.datetime(2024, 2, 6), [span("onset_datetime")]
        ),
        symptoms=[
            SymptomMention(symptom=s, span=span("symptoms")) for s in symptoms
        ],
        labs=[
            LabValue(test="GLUCOSE", value=v, unit=u, canonical_value=c,
                     canonical_unit="mg/dL", span=span("labs"))
            for v, u, c in labs
        ],
        field_states={
            "coded_term": "collected", "symptoms": "collected",
            "labs.GLUCOSE": "collected" if labs else "unknown",
        },
    )
    base.update(kwargs)
    return CanonicalAEEpisode(**base)


# The synthetic episodes below belong to a subject invented for this module, so
# the anchor they are measured from is declared here rather than borrowed from
# the corpus. Escalation is 2024-02-01; the default episode starts five days
# later, comfortably inside the definition's [0, 14] day window.
ANCHORS = {
    "dose_escalation": {"domain": "EX", "rule": "dose_increase",
                        "date_field": "EXSTDTC"},
}
EXPOSURES = {
    "S1": [
        {"USUBJID": "S1", "EXSEQ": 1, "EXDOSE": 10, "EXSTDTC": "2024-01-01"},
        {"USUBJID": "S1", "EXSEQ": 2, "EXDOSE": 20, "EXSTDTC": "2024-02-01"},
    ],
}


@pytest.fixture
def resolver() -> AnchorResolver:
    return AnchorResolver(ANCHORS, EXPOSURES)


@pytest.fixture
def evaluator(definition_v1, catalog, resolver):
    return PhenotypeEvaluator(definition_v1, catalog, resolver)


# -- the ladder -------------------------------------------------------------


def test_a_matching_coded_term_reaches_explicit(evaluator):
    verdict = evaluator.evaluate_episode(
        episode(coded_terms=["Hypoglycaemia"], onset_offset_days=Field[int]())
    )
    assert verdict.rule_id == "explicit"


def test_a_non_specific_coded_term_does_not(evaluator):
    """Coded 'Malaise', narrative describes hypoglycemia."""
    verdict = evaluator.evaluate_episode(
        episode(coded_terms=["Malaise"], symptoms=["tremor"],
                labs=[(58.0, "mg/dL", 58.0)])
    )
    assert verdict.rule_id == "supported"


def test_a_low_value_plus_a_symptom_reaches_supported(evaluator):
    verdict = evaluator.evaluate_episode(
        episode(symptoms=["tremor"], labs=[(58.0, "mg/dL", 58.0)])
    )
    assert verdict.state == "supported"
    assert "58.0 mg/dL" in verdict.reason


def test_an_si_value_is_compared_after_conversion(evaluator):
    """3.1 mmol/L is 55.9 mg/dL, below a 70 mg/dL bar."""
    verdict = evaluator.evaluate_episode(
        episode(symptoms=["tremor"], labs=[(3.1, "mmol/L", 55.8564)])
    )
    assert verdict.state == "supported"


def test_a_normal_si_value_does_not_qualify(evaluator):
    verdict = evaluator.evaluate_episode(
        episode(symptoms=["tremor"], labs=[(5.8, "mmol/L", 104.5)])
    )
    assert verdict.state == "insufficient"


def test_symptoms_alone_reach_insufficient_and_are_reported(evaluator):
    verdict = evaluator.evaluate_episode(episode(symptoms=["tremor"]))
    assert (verdict.state, verdict.verdict) == ("insufficient", "review")


def test_treatment_action_is_not_a_criterion(definition_v1):
    """It is an attribute of the episode, not part of the case definition."""
    body = yaml.safe_dump(definition_v1.model_dump(mode="json", exclude={
        "definition_hash", "source_path"
    }))
    assert "action_taken" not in body


# -- missingness ------------------------------------------------------------


def test_a_measured_value_that_misses_the_bar_is_a_finding(evaluator):
    verdict = evaluator.evaluate_episode(
        episode(symptoms=["tremor"], labs=[(90.0, "mg/dL", 90.0)])
    )
    assert verdict.state == "insufficient"
    assert verdict.review_reasons == []


def test_a_value_that_was_never_measured_is_not_a_finding(evaluator):
    """Routed to review, because absence of a measurement is not a low result."""
    subject = episode(symptoms=["tremor"])
    subject.field_states["labs.GLUCOSE"] = "unknown"
    verdict = evaluator.evaluate_episode(subject)
    assert verdict.verdict == "review"
    assert any("labs.GLUCOSE" in r for r in verdict.review_reasons)


def test_a_field_the_protocol_never_collected_is_not_absence(evaluator):
    subject = episode(coded_terms=[], symptoms=["tremor"],
                      labs=[(58.0, "mg/dL", 58.0)])
    subject.field_states["coded_term"] = "not_collected_by_protocol"
    verdict = evaluator.evaluate_episode(subject)
    assert verdict.state == "supported"
    # not_collected_by_protocol is not in route_to_review, so no routing.
    assert not any("coded_term" in r for r in verdict.review_reasons)


# -- window and linkage -----------------------------------------------------


def test_an_episode_outside_the_window_is_excluded(evaluator):
    subject = episode(coded_terms=["Hypoglycaemia"])
    subject.episode_start = Field[_dt.datetime].collected(
        _dt.datetime(2024, 6, 1), [span("onset_datetime")]
    )
    verdict = evaluator.evaluate_episode(subject)
    assert verdict.verdict == "excluded"
    assert "outside" in verdict.reason


def test_an_unresolved_onset_follows_the_definition(evaluator):
    subject = episode(coded_terms=["Hypoglycaemia"])
    subject.episode_start = Field[_dt.datetime].missing("unknown")
    verdict = evaluator.evaluate_episode(subject)
    assert verdict.verdict == "review"
    assert "routes unresolved onsets to review" in verdict.reason


def test_flagged_linkage_routes_to_review(evaluator):
    subject = episode(coded_terms=["Hypoglycaemia"], linkage_review_required=True,
                      linkage_note="a 3-day gap is a judgement call")
    verdict = evaluator.evaluate_episode(subject)
    assert verdict.verdict == "review"
    assert "judgement call" in verdict.reason


def test_low_linkage_confidence_routes_to_review(evaluator):
    subject = episode(coded_terms=["Hypoglycaemia"], linkage_confidence=0.5)
    verdict = evaluator.evaluate_episode(subject)
    assert verdict.verdict == "review"
    assert "below the definition's threshold" in verdict.reason


def test_an_unadjudicated_candidate_cannot_be_a_case(evaluator):
    subject = episode(coded_terms=["Hypoglycaemia"], candidate=True)
    verdict = evaluator.evaluate_episode(subject)
    assert verdict.verdict == "excluded"
    assert "adjudication" in verdict.reason


# -- dictionary bridging ----------------------------------------------------


def test_bridging_matches_an_earlier_dictionary_version(evaluator):
    subject = episode(coded_terms=["Hypoglycaemic episode"],
                      dictionary_versions=["MedDRA 21.1"])
    assert evaluator.evaluate_episode(subject).rule_id == "explicit"


def test_without_bridging_only_that_version_counts(definition_v1, catalog, resolver):
    variant = definition_v1.model_copy(deep=True)
    variant.concept.bridge_dictionary_versions = False
    evaluator = PhenotypeEvaluator(variant, catalog, resolver)
    modern = episode(coded_terms=["Blood glucose decreased"],
                     dictionary_versions=["MedDRA 21.1"])
    assert evaluator.evaluate_episode(modern).rule_id != "explicit"


# -- loader -----------------------------------------------------------------


def test_a_definition_must_operate_on_episodes(tmp_path, catalog, definition_v1):
    body = definition_v1.model_dump(
        mode="json", exclude={"definition_hash", "source_path"}
    )
    body["operates_on"] = "record"
    path = tmp_path / "te_symptomatic_hypoglycemia.v1.yaml"
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    with pytest.raises(Exception):
        load_definition(path, catalog)


def test_a_definition_that_reads_a_protocol_gap_as_absence_is_rejected(
    tmp_path, catalog, definition_v1
):
    body = definition_v1.model_dump(
        mode="json", exclude={"definition_hash", "source_path"}
    )
    body["missingness"]["treat_as_absent"] = ["not_collected_by_protocol"]
    path = tmp_path / "te_symptomatic_hypoglycemia.v1.yaml"
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    with pytest.raises(DefinitionError, match="evidence of absence"):
        load_definition(path, catalog)


def test_the_loader_refuses_to_overwrite_a_frozen_definition(
    tmp_path, catalog, definition_v1
):
    from aelayer.phenotype.loader import DefinitionCatalog

    body = definition_v1.model_dump(
        mode="json", exclude={"definition_hash", "source_path"}
    )
    (tmp_path / "te_symptomatic_hypoglycemia.v1.yaml").write_text(
        yaml.safe_dump(body), encoding="utf-8"
    )
    catalogue = DefinitionCatalog(tmp_path, catalog)
    with pytest.raises(DefinitionError, match="frozen and will not be overwritten"):
        catalogue.write_candidate(body)


def test_every_assignment_names_the_rule_that_decided_it(pipeline, definition_v1):
    for assignment in pipeline.evaluate(definition_v1):
        assert assignment.reason
        if assignment.matched_rule_id:
            assert assignment.matched_rule_id in assignment.reason
        assert assignment.definition_hash == definition_v1.definition_hash


def test_v2_only_ever_weakens_a_verdict(pipeline, definition_v1, definition_v2):
    rank = {"excluded": 0, "review": 1, "case": 2}
    v1 = {a.episode_id: a for a in pipeline.evaluate(definition_v1)}
    v2 = {a.episode_id: a for a in pipeline.evaluate(definition_v2)}
    moved = [i for i in v1 if v1[i].verdict != v2[i].verdict]
    assert moved
    for episode_id in v1:
        assert rank[v2[episode_id].verdict] <= rank[v1[episode_id].verdict]
