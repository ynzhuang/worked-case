"""Concept matching: lexicon, coded terms, abbreviation gating, fuzzy match."""

from __future__ import annotations

import pytest

from aelayer.extract.concepts import ConceptMatcher
from aelayer.extract.text import edit_distance, normalise, split_sentences


@pytest.fixture(scope="module")
def matcher(catalog, extraction_config):
    return ConceptMatcher(catalog, extraction_config)


def find(matcher, text, concept="HYPOGLYCEMIA"):
    return [
        m for m in matcher.find_concepts(text, split_sentences(text))
        if m.concept_id == concept
    ]


@pytest.mark.parametrize(
    "text",
    [
        "The subject experienced hypoglycemia.",
        "The subject experienced hypoglycaemia.",          # British spelling
        "The subject reported low blood sugar.",
        "A hypoglycaemic episode was recorded.",
        "Blood glucose decreased was reported.",           # a coded term in text
    ],
)
def test_lexicon_and_coded_terms_match(matcher, text):
    assert find(matcher, text), text


def test_multiword_match_wins_over_its_parts(matcher):
    hits = find(matcher, "The subject had low blood glucose today.")
    assert len(hits) == 1
    assert hits[0].surface == "low blood glucose"


def test_fuzzy_match_tolerates_one_edit(matcher):
    hits = find(matcher, "The subject experienced hypoglycemai on study day 4.")
    assert hits and hits[0].kind == "lexicon_fuzzy"


def test_fuzzy_match_does_not_confuse_hypo_and_hyperglycemia(matcher):
    """Two edits apart, so an edit budget of one must keep them separate."""
    assert edit_distance("hypoglycemia", "hyperglycemia", maximum=2) == 2
    hits = find(matcher, "The subject reported hyperglycemia.", "HYPOGLYCEMIA")
    assert hits == []
    assert find(matcher, "The subject reported hyperglycemia.", "HYPERGLYCEMIA")


def test_transposition_counts_as_one_edit():
    """The commonest typo class. Plain Levenshtein scores it two."""
    assert edit_distance("hypoglycemia", "hypoglycemai", maximum=1) == 1


def test_abbreviation_fires_only_when_its_context_gate_is_satisfied(matcher):
    gated = "The subject had a hypo with capillary glucose of 48 mg/dL."
    ungated = "The subject uses the word hypo to describe feeling unwell."
    hits = find(matcher, gated)
    assert hits and hits[0].kind == "abbreviation"
    assert "GLUCOSE" in hits[0].gate_reason
    assert find(matcher, ungated) == []


def test_abbreviation_gate_accepts_a_qualifying_symptom(matcher):
    hits = find(matcher, "A hypo was reported, with diaphoresis and tremor.")
    assert hits and "symptom" in hits[0].gate_reason


def test_abbreviation_requires_a_word_boundary(matcher):
    """`hypo` inside `hypothyroidism` is not the abbreviation."""
    hits = find(matcher, "The subject has hypothyroidism and glucose of 50 mg/dL.")
    assert not [m for m in hits if m.kind == "abbreviation"]


def test_uppercase_abbreviations_are_case_sensitive(matcher):
    """LOC is a shorthand; `loc` in lower case is usually a different word."""
    text = "The subject had LOC with confusion."
    upper = [m for m in matcher.find_concepts(text, split_sentences(text))
             if m.concept_id == "SYNCOPE"]
    lower_text = "The subject had loc with confusion."
    lower = [m for m in matcher.find_concepts(lower_text, split_sentences(lower_text))
             if m.concept_id == "SYNCOPE"]
    assert upper and not lower


def test_coded_term_field_maps_only_by_explicit_membership(matcher):
    assert matcher.coded_term_concept("Hypoglycaemia") == "HYPOGLYCEMIA"
    assert matcher.coded_term_concept("Blood glucose decreased") == "HYPOGLYCEMIA"
    # Not a catalogue term for the concept, however plausible it sounds.
    assert matcher.coded_term_concept("Malaise") is None
    assert matcher.coded_term_concept(None) is None


def test_symptom_matching_normalises_surface_forms(matcher):
    found = {m.symptom for m in matcher.find_symptoms(
        "The subject was diaphoretic, light-headed and had the shakes."
    )}
    assert {"diaphoresis", "lightheadedness", "shakiness"} <= found


def test_symptom_matching_prefers_the_longest_span(matcher):
    found = matcher.find_symptoms("The subject reported blurred vision.")
    assert [m.symptom for m in found] == ["blurred vision"]


def test_normalise_folds_british_and_american_spelling():
    assert normalise("Hypoglycaemia") == normalise("hypoglycemia")
    assert normalise("hospitalisation") == normalise("hospitalization")


def test_matching_is_deterministic(matcher):
    text = "Hypoglycaemia with tremor and glucose 48 mg/dL."
    first = matcher.find_concepts(text, split_sentences(text))
    second = matcher.find_concepts(text, split_sentences(text))
    assert [m.__dict__ for m in first] == [m.__dict__ for m in second]
