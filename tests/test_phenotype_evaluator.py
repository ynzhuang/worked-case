"""Rule evaluation, the evidence ladder, and the reason on every row."""

from __future__ import annotations

import datetime as _dt

import pytest

from aelayer.anchors import AnchorResolver
from aelayer.models import EventObject, LabValue, Span, SymptomMention
from aelayer.phenotype.evaluator import PhenotypeEvaluator

ANCHORS = {
    "dose_escalation": {"domain": "EX", "rule": "dose_increase", "date_field": "EXSTDTC"}
}
EXPOSURES = {
    "S1": [
        {"USUBJID": "S1", "EXSEQ": 1, "EXDOSE": 10, "EXSTDTC": "2024-01-01"},
        {"USUBJID": "S1", "EXSEQ": 2, "EXDOSE": 20, "EXSTDTC": "2024-02-01"},
    ]
}


def span(field, value="x"):
    return Span(doc_id="d", start=0, end=1, field=field, extracted_value=value, text="x")


def make_event(**kwargs):
    base = dict(
        event_id=kwargs.pop("event_id", "e1"),
        subject_id="S1", study_id="ST", doc_id="d",
        concept_id="HYPOGLYCEMIA",
        evidence=[span("concept_id"), span("assertion")],
    )
    symptoms = kwargs.pop("symptoms", [])
    labs = kwargs.pop("labs", [])
    base["symptoms"] = [
        SymptomMention(symptom=s, span=span("symptoms", s)) for s in symptoms
    ]
    base["labs"] = [
        LabValue(test="GLUCOSE", value=v, unit=u, canonical_value=c,
                 canonical_unit="mg/dL", span=span("labs"))
        for v, u, c in labs
    ]
    base.update(kwargs)
    return EventObject(**base)


@pytest.fixture
def evaluator(definition_v1, catalog):
    return PhenotypeEvaluator(definition_v1, catalog, AnchorResolver(ANCHORS, EXPOSURES))


def in_window(**kwargs):
    kwargs.setdefault("onset_offset_days", 5)
    kwargs.setdefault("anchor_event", "dose_escalation")
    return kwargs


# -- the evidence ladder ---------------------------------------------------


def test_a_matching_coded_term_reaches_explicit(evaluator):
    event = make_event(
        coded_term="Hypoglycaemia", concept_match_kinds=["coded_term"], **in_window()
    )
    verdict = evaluator.evaluate_event(event)
    assert (verdict.state, verdict.verdict, verdict.rule_id) == (
        "explicit", "case", "explicit"
    )
    assert "coded term" in verdict.reason


def test_a_verbatim_mention_reaches_explicit(evaluator):
    event = make_event(concept_match_kinds=["lexicon"], **in_window())
    assert evaluator.evaluate_event(event).state == "explicit"


def test_a_non_specific_coded_term_does_not_reach_explicit(evaluator):
    """Coded `Malaise`, narrative describes hypoglycemia: not an explicit match."""
    event = make_event(
        coded_term="Malaise", concept_match_kinds=["contextual"],
        symptoms=["tremor"], **in_window()
    )
    assert evaluator.evaluate_event(event).state != "explicit"


def test_a_low_glucose_plus_a_symptom_reaches_supported(evaluator):
    event = make_event(
        concept_match_kinds=["contextual"], symptoms=["tremor"],
        labs=[(58.0, "mg/dL", 58.0)], **in_window()
    )
    verdict = evaluator.evaluate_event(event)
    assert (verdict.state, verdict.verdict) == ("supported", "case")
    assert "58.0 mg/dL" in verdict.reason and "tremor" in verdict.reason


def test_an_si_value_is_compared_after_conversion(evaluator):
    """3.1 mmol/L is 55.9 mg/dL, which is below the 70 mg/dL threshold.

    A rule that compared 3.1 to 70 numerically would call every SI study a case;
    one that ignored conversion entirely would call none of them.
    """
    event = make_event(
        concept_match_kinds=["contextual"], symptoms=["tremor"],
        labs=[(3.1, "mmol/L", 55.8564)], **in_window()
    )
    assert evaluator.evaluate_event(event).state == "supported"


def test_a_normal_si_value_does_not_reach_supported(evaluator):
    event = make_event(
        concept_match_kinds=["contextual"], symptoms=["tremor"],
        labs=[(5.8, "mmol/L", 104.5)], **in_window()
    )
    assert evaluator.evaluate_event(event).state != "supported"


def test_symptoms_plus_rescue_reach_possible(evaluator):
    event = make_event(
        concept_match_kinds=["contextual"], symptoms=["tremor"],
        rescue_treatment=True,
        evidence=[span("concept_id"), span("assertion"), span("rescue_treatment")],
        **in_window()
    )
    verdict = evaluator.evaluate_event(event)
    assert (verdict.state, verdict.verdict) == ("possible", "review")


def test_symptoms_plus_a_dose_action_reach_possible(evaluator):
    event = make_event(
        concept_match_kinds=["contextual"], symptoms=["tremor"],
        action_taken="dose_interrupted",
        evidence=[span("concept_id"), span("assertion"), span("action_taken")],
        **in_window()
    )
    assert evaluator.evaluate_event(event).state == "possible"


def test_symptoms_alone_reach_nothing(evaluator):
    """Counting these would manufacture signal."""
    event = make_event(
        concept_match_kinds=["contextual"], symptoms=["tremor"], **in_window()
    )
    verdict = evaluator.evaluate_event(event)
    assert (verdict.state, verdict.verdict, verdict.rule_id) == ("none", "excluded", None)


def test_rules_are_ordered_and_the_first_match_wins(evaluator):
    """An explicit mention with a low glucose is `explicit`, not `supported`."""
    event = make_event(
        concept_match_kinds=["lexicon"], symptoms=["tremor"],
        labs=[(48.0, "mg/dL", 48.0)], **in_window()
    )
    assert evaluator.evaluate_event(event).rule_id == "explicit"


# -- assertion policy ------------------------------------------------------


def test_a_negated_mention_becomes_absent_and_is_excluded(evaluator):
    event = make_event(assertion="absent", concept_match_kinds=["lexicon"], **in_window())
    verdict = evaluator.evaluate_event(event)
    assert (verdict.state, verdict.verdict) == ("absent", "excluded")
    assert "excluded by the definition's assertion policy" in verdict.reason


@pytest.mark.parametrize("assertion", ["hypothetical", "historical", "family_history"])
def test_excluded_assertions_never_become_cases(evaluator, assertion):
    event = make_event(
        assertion=assertion, concept_match_kinds=["lexicon"], **in_window()
    )
    assert evaluator.evaluate_event(event).verdict == "excluded"


def test_an_uncertain_mention_is_routed_to_review(evaluator):
    event = make_event(
        assertion="uncertain", concept_match_kinds=["lexicon"],
        symptoms=["tremor"], **in_window()
    )
    verdict = evaluator.evaluate_event(event)
    assert verdict.verdict == "review"
    assert "routed to review" in verdict.reason


# -- window ----------------------------------------------------------------


def test_an_event_outside_the_window_is_excluded(evaluator):
    event = make_event(
        concept_match_kinds=["lexicon"], onset_offset_days=40,
        anchor_event="dose_escalation",
    )
    verdict = evaluator.evaluate_event(event)
    assert verdict.verdict == "excluded"
    assert "outside the window" in verdict.reason


def test_the_window_boundaries_are_inclusive(evaluator):
    for offset in (0, 14):
        event = make_event(
            concept_match_kinds=["lexicon"], onset_offset_days=offset,
            anchor_event="dose_escalation",
        )
        assert evaluator.evaluate_event(event).verdict == "case"


def test_an_unresolved_onset_follows_the_definition_not_the_extractor(evaluator):
    event = make_event(concept_match_kinds=["lexicon"])
    verdict = evaluator.evaluate_event(event)
    assert verdict.verdict == "review"
    assert "routes unresolved onsets to review" in verdict.reason


def test_on_unresolved_onset_exclude_is_honoured(definition_v1, catalog):
    variant = definition_v1.model_copy(deep=True)
    variant.window.on_unresolved_onset = "exclude"
    evaluator = PhenotypeEvaluator(variant, catalog, AnchorResolver(ANCHORS, EXPOSURES))
    event = make_event(concept_match_kinds=["lexicon"])
    assert evaluator.evaluate_event(event).verdict == "excluded"


def test_the_offset_is_recomputed_under_the_definitions_index_rule(evaluator):
    """The date is authoritative; the definition picks which anchor it counts from."""
    event = make_event(
        concept_match_kinds=["lexicon"], onset_date=_dt.date(2024, 2, 6),
        onset_offset_days=99, anchor_event="dose_escalation",
    )
    offset, resolved, _detail = evaluator.resolve_offset(event)
    assert resolved and offset == 5


def test_an_offset_against_a_different_anchor_is_not_reused(evaluator):
    event = make_event(concept_match_kinds=["lexicon"], onset_offset_days=5,
                       anchor_event="first_dose")
    _offset, resolved, detail = evaluator.resolve_offset(event)
    assert not resolved and "not the definition's anchor" in detail


# -- subject level ---------------------------------------------------------


def test_the_strongest_event_decides_the_subject(evaluator):
    weak = make_event(event_id="e1", concept_match_kinds=["contextual"],
                      symptoms=["tremor"], **in_window())
    strong = make_event(event_id="e2", concept_match_kinds=["lexicon"], **in_window())
    assignment = evaluator.evaluate([weak, strong])[0]
    assert assignment.verdict == "case"
    assert assignment.evidence_state == "explicit"
    assert "e2" in assignment.contributing_event_ids


def test_a_subject_with_no_event_still_gets_a_row(evaluator):
    assignments = evaluator.evaluate([], subjects=[("S9", "ST")])
    assert len(assignments) == 1
    assert assignments[0].verdict == "excluded"
    assert "no event object" in assignments[0].reason


def test_every_assignment_names_the_rule_that_decided_it(pipeline, definition_v1):
    for assignment in pipeline.evaluate(definition_v1):
        assert assignment.reason
        if assignment.matched_rule_id:
            assert assignment.matched_rule_id in assignment.reason
        assert assignment.definition_id == definition_v1.id
        assert assignment.definition_version == definition_v1.version
        assert assignment.definition_hash == definition_v1.definition_hash


def test_cases_carry_the_spans_that_made_them_cases(pipeline, definition_v1):
    cases = [a for a in pipeline.evaluate(definition_v1) if a.verdict == "case"]
    assert cases
    assert all(a.evidence_spans for a in cases)


def test_events_for_other_concepts_are_ignored(evaluator):
    other = make_event(concept_id="NAUSEA", concept_match_kinds=["lexicon"], **in_window())
    assignments = evaluator.evaluate([other], subjects=[("S1", "ST")])
    assert assignments[0].verdict == "excluded"


def test_a_concept_group_expands_by_explicit_membership(definition_v1, catalog):
    grouped = definition_v1.model_copy(deep=True)
    grouped.concept.group = "GLYCEMIC_EVENTS"
    evaluator = PhenotypeEvaluator(grouped, catalog, AnchorResolver(ANCHORS, EXPOSURES))
    assert evaluator.concept_ids == {"HYPOGLYCEMIA", "HYPERGLYCEMIA"}


# -- version behaviour -----------------------------------------------------


def test_v2_moves_subjects_from_case_to_review(pipeline, definition_v1, definition_v2):
    """Raising the bar from 70 to 54 mg/dL must move people, and only downward."""
    v1 = {a.subject_id: a for a in pipeline.evaluate(definition_v1)}
    v2 = {a.subject_id: a for a in pipeline.evaluate(definition_v2)}

    moved = [s for s in v1 if v1[s].verdict == "case" and v2[s].verdict == "review"]
    assert moved, "a stricter threshold must reclassify somebody"

    # A stricter threshold can only weaken a verdict, never strengthen one.
    rank = {"excluded": 0, "review": 1, "case": 2}
    for subject in v1:
        assert rank[v2[subject].verdict] <= rank[v1[subject].verdict], subject

    # Every subject who moved rested on `supported` under v1 — that is the only
    # state the threshold touches. Where they land under v2 depends on their
    # other events, so the arrival state is not fixed.
    assert all(v1[s].evidence_state == "supported" for s in moved)
    assert any(v2[s].evidence_state == "possible" for s in moved)


def test_the_supported_count_shrinks_under_the_stricter_threshold(
    pipeline, definition_v1, definition_v2
):
    def supported(definition):
        return sum(
            1 for a in pipeline.evaluate(definition) if a.evidence_state == "supported"
        )

    assert supported(definition_v2) < supported(definition_v1)


def test_evaluating_v2_does_not_change_what_v1_says(pipeline, definition_v1, definition_v2):
    before = [a.model_dump_json() for a in pipeline.evaluate(definition_v1)]
    pipeline.evaluate(definition_v2)
    after = [a.model_dump_json() for a in pipeline.evaluate(definition_v1)]
    assert before == after
