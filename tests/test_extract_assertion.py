"""Assertion classification across all six classes."""

from __future__ import annotations

import pytest

from aelayer.extract.assertion import AssertionClassifier
from aelayer.extract.concepts import ConceptMatcher
from aelayer.extract.text import split_sentences


@pytest.fixture(scope="module")
def classify(catalog, extraction_config):
    matcher = ConceptMatcher(catalog, extraction_config)
    classifier = AssertionClassifier(extraction_config)

    def run(text: str, concept: str = "HYPOGLYCEMIA"):
        sentences = split_sentences(text)
        mentions = [
            m for m in matcher.find_concepts(text, sentences) if m.concept_id == concept
        ]
        assert mentions, f"no mention of {concept} in {text!r}"
        return classifier.classify(text, mentions[0].start, mentions[0].end, sentences)

    return run


@pytest.mark.parametrize(
    "text,expected",
    [
        ("The subject experienced hypoglycaemia on study day 4.", "present"),
        ("There was no evidence of hypoglycemia.", "absent"),
        ("The subject denies hypoglycemia.", "absent"),
        ("Screening was negative for hypoglycemia.", "absent"),
        ("Hypoglycaemia was ruled out on review of the log.", "absent"),
        ("Monitor for hypoglycemia after each dose increase.", "hypothetical"),
        ("Carbohydrate was supplied in case of hypoglycemia.", "hypothetical"),
        ("The subject has a history of hypoglycaemia.", "historical"),
        ("Hypoglycaemia was documented previously.", "historical"),
        ("The subject's mother has hypoglycemia.", "family_history"),
        ("Family history of hypoglycemia was recorded.", "family_history"),
        ("Possible hypoglycemia was considered.", "uncertain"),
        ("Hypoglycaemia cannot be excluded.", "uncertain"),
    ],
)
def test_all_six_assertion_classes(classify, text, expected):
    assert classify(text).assertion == expected


def test_every_class_is_reachable(classify):
    """All six, not just present and absent."""
    seen = {
        classify(t).assertion
        for t in [
            "The subject experienced hypoglycaemia.",
            "There was no evidence of hypoglycemia.",
            "Monitor for hypoglycemia.",
            "The subject has a history of hypoglycaemia.",
            "The subject's mother has hypoglycemia.",
            "Possible hypoglycemia was considered.",
        ]
    }
    assert seen == {
        "present", "absent", "hypothetical", "historical",
        "family_history", "uncertain",
    }


def test_a_cue_carries_a_span(classify):
    result = classify("There was no evidence of hypoglycemia.")
    assert result.cue == "no evidence of"
    assert result.cue_start is not None and result.cue_end > result.cue_start


def test_pseudo_cues_do_not_negate(classify):
    """`no dose change` contains a negation token but negates nothing."""
    text = "No dose change to study drug was made after the hypoglycaemia."
    assert classify(text).assertion == "present"


def test_negation_does_not_cross_a_sentence_boundary(classify):
    text = (
        "There was no evidence of nausea. "
        "The subject experienced hypoglycaemia the next morning."
    )
    assert classify(text).assertion == "present"


def test_negation_does_not_cross_a_terminator(classify):
    text = "The subject denies nausea but did report hypoglycaemia."
    assert classify(text).assertion == "present"


def test_default_is_present_with_a_reason(classify):
    result = classify("The subject experienced hypoglycaemia.")
    assert result.assertion == "present"
    assert result.cue is None
    assert "default" in result.rule


def test_precedence_is_config_driven(classify):
    """Family history outranks negation when both cues are in scope."""
    text = "The subject's mother had no evidence of hypoglycemia."
    assert classify(text).assertion == "family_history"


def test_post_cues_govern_a_preceding_mention(classify):
    assert classify("Hypoglycaemia was ruled out.").assertion == "absent"


def test_window_scope_is_supported(catalog, extraction_config):
    """`scope: window` is an alternative to sentence scope, not a stub."""
    import copy

    from aelayer.catalog import ExtractionConfig

    raw = copy.deepcopy(extraction_config.raw)
    raw["assertion"]["scope"] = "window"
    raw["assertion"]["window_tokens"] = 3
    classifier = AssertionClassifier(ExtractionConfig(raw))
    text = "There was no evidence of anything else at all, and hypoglycaemia occurred."
    start = text.index("hypoglycaemia")
    result = classifier.classify(text, start, start + 13, split_sentences(text))
    # The cue is well outside a three-token window, so it must not reach.
    assert result.assertion == "present"
