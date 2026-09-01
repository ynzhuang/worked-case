"""The model path: assertion, values, temporality, and concept surface forms.

These are the pieces the extraction backends are built from. Each is tested on
text written here rather than on corpus output, so a failure names the rule that
broke rather than the pipeline stage that noticed.
"""

from __future__ import annotations

import datetime as _dt

import pytest

from aelayer.anchors import AnchorResolver
from aelayer.extract.assertion import AssertionClassifier
from aelayer.extract.concepts import ConceptMatcher
from aelayer.extract.temporal import TemporalExtractor
from aelayer.extract.text import edit_distance, split_sentences, tokenize
from aelayer.extract.values import ValueExtractor


@pytest.fixture(scope="module")
def extraction(configs):
    return configs.extraction


@pytest.fixture(scope="module")
def classifier(extraction):
    return AssertionClassifier(extraction)


@pytest.fixture(scope="module")
def matcher(catalog, extraction):
    return ConceptMatcher(catalog, extraction)


@pytest.fixture(scope="module")
def values(catalog, extraction):
    return ValueExtractor(catalog, extraction)


def classify(classifier, text: str, needle: str):
    start = text.lower().index(needle.lower())
    return classifier.classify(text, start, start + len(needle), split_sentences(text))


# -- sentences and tokens ---------------------------------------------------


def test_sentences_carry_their_own_offsets():
    text = "Onset on day 3. Glucose was 54 mg/dL. Resolved."
    sentences = split_sentences(text)
    assert len(sentences) == 3
    for sentence in sentences:
        assert text[sentence.start:sentence.end] == sentence.text


def test_a_token_points_back_at_the_characters_it_came_from():
    text = "Severe hypoglycaemia overnight"
    for token in tokenize(text):
        assert text[token.start:token.end] == token.text


def test_edit_distance_stops_counting_past_the_ceiling():
    assert edit_distance("hypoglycaemia", "hypoglycemia", maximum=2) == 1
    assert edit_distance("hypoglycaemia", "anaemia", maximum=2) > 2


# -- assertion --------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Patient reported hypoglycemia overnight.", "present"),
        ("There was no hypoglycemia during the infusion.", "absent"),
        ("The patient denied hypoglycemia at the visit.", "absent"),
        ("Possible hypoglycemia; glucose not drawn.", "uncertain"),
        ("History of hypoglycemia prior to enrolment.", "historical"),
        ("Advised to report hypoglycemia if it recurs.", "hypothetical"),
        ("Mother has hypoglycemia unawareness.", "family_history"),
    ],
)
def test_each_assertion_class_is_reachable(classifier, text, expected):
    assert classify(classifier, text, "hypoglycemia").assertion == expected


def test_a_mention_with_no_cue_defaults_to_present(classifier):
    result = classify(classifier, "Hypoglycemia at 03:00.", "hypoglycemia")
    assert result.assertion == "present"
    assert result.cue is None
    assert "default" in result.rule


def test_a_pseudo_cue_does_not_negate(classifier):
    """'no change' contains a negation token but negates nothing clinical."""
    text = "No change in dose; hypoglycemia recurred the next night."
    assert classify(classifier, text, "hypoglycemia").assertion == "present"


def test_a_cue_does_not_reach_past_a_terminator(classifier):
    text = "No nausea, but hypoglycemia was documented."
    assert classify(classifier, text, "hypoglycemia").assertion == "present"


def test_a_cue_does_not_reach_into_the_next_sentence(classifier):
    text = "There was no nausea. Hypoglycemia was documented overnight."
    assert classify(classifier, text, "Hypoglycemia").assertion == "present"


def test_a_post_cue_governs_a_mention_that_precedes_it(classifier):
    text = "Hypoglycemia was ruled out on repeat testing."
    assert classify(classifier, text, "Hypoglycemia").assertion == "absent"


def test_the_nearest_cue_wins(classifier):
    text = "History of diabetes; possible hypoglycemia this morning."
    assert classify(classifier, text, "hypoglycemia").assertion == "uncertain"


def test_the_classifier_reports_the_cue_it_used(classifier):
    result = classify(
        classifier, "There was no hypoglycemia overnight.", "hypoglycemia"
    )
    assert result.cue
    assert result.has_cue
    assert result.cue in "There was no hypoglycemia overnight."


# -- concepts ---------------------------------------------------------------


def test_a_lexicon_surface_form_is_matched(matcher):
    mentions = matcher.find_concepts("Patient reported hypoglycemia overnight.")
    assert any(m.concept_id == "HYPOGLYCEMIA" for m in mentions)


def test_an_abbreviation_matches_only_with_the_context_its_gate_requires(matcher):
    """'hypo' is ambiguous prose until something in the sentence anchors it."""
    bare = matcher.find_concepts("Patient had a hypo overnight.")
    assert not any(m.concept_id == "HYPOGLYCEMIA" for m in bare)
    gated = matcher.find_concepts("Hypo overnight with tremor and sweating.")
    assert any(m.concept_id == "HYPOGLYCEMIA" for m in gated)


def test_a_spelling_variant_is_matched(matcher):
    for surface in ("hypoglycaemia", "hypoglycemia", "low blood sugar"):
        mentions = matcher.find_concepts(f"Reported {surface} at 02:00.")
        assert any(m.concept_id == "HYPOGLYCEMIA" for m in mentions), surface


def test_every_mention_points_at_the_text_it_matched(matcher):
    text = "Reported hypoglycaemia and later anaemia."
    for mention in matcher.find_concepts(text):
        assert text[mention.start:mention.end].lower() == mention.surface.lower()


def test_overlapping_matches_resolve_to_the_longest(matcher):
    mentions = matcher.find_concepts("Severe hypoglycemia unawareness noted.")
    spans = [(m.start, m.end) for m in mentions]
    assert len(spans) == len(set(spans))
    for i, (start, end) in enumerate(spans):
        for other_start, other_end in spans[i + 1:]:
            assert end <= other_start or other_end <= start


def test_symptoms_are_matched_from_the_catalogue(matcher):
    found = {m.symptom for m in matcher.find_symptoms("Tremor and sweating noted.")}
    assert {"tremor", "sweating"} <= found


def test_a_coded_term_maps_to_a_concept(matcher):
    assert matcher.coded_term_concept("Hypoglycaemia") == "HYPOGLYCEMIA"
    assert matcher.coded_term_concept("Malaise") is None
    assert matcher.coded_term_concept(None) is None


# -- values -----------------------------------------------------------------


def test_a_lab_value_with_its_unit_is_read(values):
    hits = values.find_labs("Blood glucose was 54 mg/dL at the time.")
    assert len(hits) == 1
    assert (hits[0].test, hits[0].value, hits[0].unit) == ("GLUCOSE", 54.0, "mg/dL")


def test_an_si_value_is_read_in_its_own_unit(values):
    hit = values.find_labs("Glucose 3.1 mmol/L.")[0]
    assert (hit.value, hit.unit) == (3.1, "mmol/L")


def test_a_bare_number_is_given_a_unit_only_when_it_can_only_be_one(values):
    assert values.find_labs("Glucose was 54.")[0].unit == "mg/dL"
    assert values.find_labs("Glucose 3.1.")[0].unit == "mmol/L"


def test_a_magnitude_that_fits_no_declared_range_yields_no_value(values):
    """Between the two unit ranges there is no honest reading, so there is none."""
    assert values.find_labs("Glucose 32.") == []
    assert values.find_labs("Glucose 1.") == []


def test_a_duration_is_not_a_laboratory_result(values):
    """'glucose ... 1 days after' must not become 1.0 mmol/L."""
    assert values.find_labs("Blood glucose checked 1 days after escalation.") == []


def test_severity_and_seriousness_are_read_from_separate_cue_sets(values):
    severity = values.single_value("The event was severe.", "severity")
    assert severity.value == "severe"
    assert values.single_value("The event was severe.", "seriousness") is None
    serious = values.multi_value("The patient was hospitalised.", "seriousness")
    assert [c.value for c in serious] == ["hospitalisation"]


def test_a_cue_shadowed_by_a_longer_one_is_dropped(values):
    hit = values.single_value(
        "Required third-party assistance overnight.", "severity"
    )
    assert hit.value == "severe"


def test_a_value_cue_reports_where_it_was_found(values):
    text = "The event was moderate in intensity."
    hit = values.single_value(text, "severity")
    assert text[hit.start:hit.end].lower() == hit.cue.lower()
    assert hit.field == "severity"


# -- temporality ------------------------------------------------------------


ANCHORS = {"dose_escalation": {"domain": "EX", "rule": "dose_increase",
                               "date_field": "EXSTDTC"}}
EXPOSURES = {"S1": [
    {"USUBJID": "S1", "EXSEQ": 1, "EXDOSE": 10, "EXSTDTC": "2024-01-01"},
    {"USUBJID": "S1", "EXSEQ": 2, "EXDOSE": 20, "EXSTDTC": "2024-02-01"},
]}


@pytest.fixture(scope="module")
def temporal(extraction):
    return TemporalExtractor(extraction, AnchorResolver(ANCHORS, EXPOSURES))


def resolve(temporal, text, **kwargs):
    return temporal.resolve(
        subject_id="S1", text=text, scope=None,
        default_anchor="dose_escalation", **kwargs,
    )


def test_the_structured_onset_date_is_preferred_over_the_narrative(temporal):
    result = resolve(
        temporal, "Six days after the dose escalation.",
        recorded_onset="2024-02-04",
    )
    assert result.source == "structured_onset_date"
    assert result.onset_date == _dt.date(2024, 2, 4)
    assert result.onset_offset_days == 3


def test_a_relative_expression_is_anchored_to_the_exposure_record(temporal):
    result = resolve(temporal, "Six days after the dose escalation the patient fell.")
    assert result.onset_offset_days == 6
    assert result.onset_date == _dt.date(2024, 2, 7)
    assert result.anchor_event == "dose_escalation"


def test_a_study_day_needs_a_reference_start_and_says_so_without_one(temporal):
    unresolved = resolve(temporal, "Event on day 12.")
    assert unresolved.source == "unresolved_study_day"
    assert not unresolved.resolved
    assert "reference start date" in unresolved.detail

    resolved = resolve(
        temporal, "Event on day 12.", reference_start=_dt.date(2024, 1, 1)
    )
    assert resolved.onset_date == _dt.date(2024, 1, 12)


def test_an_absolute_date_in_the_narrative_is_read(temporal):
    result = resolve(temporal, "Event on 2024-02-09 overnight.")
    assert result.onset_date == _dt.date(2024, 2, 9)
    assert result.onset_offset_days == 8


def test_a_vague_quantifier_is_resolved_and_marked_vague(temporal):
    result = resolve(temporal, "A few days after the dose escalation.")
    assert result.source == "narrative_relative_vague"
    assert result.mention.vague
    assert result.confidence < 0.92


def test_a_subject_with_no_anchor_keeps_the_offset_and_drops_the_date(temporal):
    result = temporal.resolve(
        subject_id="NOBODY", text="Six days after the dose escalation.",
        scope=None, default_anchor="dose_escalation",
    )
    assert result.onset_offset_days == 6
    assert result.onset_date is None
    assert "no resolvable anchor date" in result.detail


def test_text_with_no_temporal_expression_resolves_to_nothing(temporal):
    result = resolve(temporal, "The patient felt unwell.")
    assert not result.resolved
    assert result.source == "unresolved"


def test_an_anchor_phrase_in_text_beats_the_default(temporal):
    assert temporal.match_anchor("the dose escalation") == "dose_escalation"
    assert temporal.match_anchor("the first dose") == "first_dose"
    assert temporal.match_anchor("the moon landing") is None
