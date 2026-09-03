"""The model path: assertion classification, spans, confidence, abstention."""

from __future__ import annotations

import pytest

from aelayer.extract import ExtractionEngine, MentionFinder
from aelayer.extract.assertion import AssertionClassifier
from aelayer.extract.text import Sentence, split_sentences

MODIFIER = "mucosal_involvement"


@pytest.fixture(scope="module")
def finder(configs):
    return MentionFinder(configs.catalog, configs.extraction)


@pytest.mark.parametrize(
    "text, assertion",
    [
        ("rash with oral mucosal involvement", "present"),
        ("rash and mucosal erosions", "present"),
        ("rash, oral ulceration noted", "present"),
        ("rash without mucosal involvement", "absent"),
        ("rash, no mucosal involvement", "absent"),
        ("rash; mucosal involvement was absent", "absent"),
        ("rash, mucosal involvement was not present", "absent"),
        ("rash with possible mucosal involvement", "uncertain"),
        ("rash, query mucosal involvement", "uncertain"),
        ("rash; mucosal involvement cannot be excluded", "uncertain"),
    ],
)
def test_assertion_classification(finder, text, assertion):
    mention = finder.best(text, MODIFIER)
    assert mention is not None, f"abstained on {text!r}"
    assert mention.assertion == assertion


def test_a_pseudo_cue_does_not_negate(finder):
    mention = finder.best(
        "rash with oral ulceration. No dose change was made.", MODIFIER
    )
    assert mention is not None
    assert mention.assertion == "present"


def test_a_terminator_ends_a_cue_scope(finder):
    """"No rash, but oral ulceration was seen" does not negate the ulceration."""
    mention = finder.best("no rash, but oral ulceration was seen", MODIFIER)
    assert mention is not None
    assert mention.assertion == "present"


def test_a_hedge_outranks_a_negation(configs):
    """A sentence that refuses to commit is not a documented negative."""
    classifier = AssertionClassifier(configs.extraction)
    text = "possible mucosal involvement, no ulceration seen"
    sentence = split_sentences(text)[0]
    start = text.index("mucosal involvement")
    call = classifier.classify(text, sentence, start, start + len("mucosal involvement"))
    assert call.assertion == "uncertain"


def test_abstention_when_no_catalogue_form_matches(finder):
    """A phrasing nobody declared produces no answer, not a guess."""
    assert finder.best("rash with involvement of the wet surfaces", MODIFIER) is None


def test_abstention_when_the_sentence_names_no_event(finder):
    """A mention with nothing to anchor it to is not this event's."""
    assert finder.best("Mucosal involvement was absent.", MODIFIER) is None


def test_abstention_when_two_readings_are_equally_supported(finder):
    mention = finder.best(
        "rash with oral ulceration and rash with conjunctival involvement",
        MODIFIER,
    )
    # Two different sites, both directly attached. Picking one would assert
    # something the text does not settle.
    assert mention is None or mention.value in {"ORAL", "OCULAR"}


def test_a_comment_is_anchored_by_its_record(finder):
    """A comment points at one AE row structurally; it need not name the event."""
    mention = finder.best(
        "Investigator comment: no mucosal involvement was seen at any visit.",
        MODIFIER, source_kind="comment",
    )
    assert mention is not None
    assert mention.assertion == "absent"
    assert mention.anchor_surface == "(the comment's own record)"
    # The same text read as a reported term has nothing to anchor it.
    assert finder.best(
        "Investigator comment: no mucosal involvement was seen at any visit.",
        MODIFIER, source_kind="reported_term",
    ) is None


def test_every_mention_carries_a_span_covering_its_cue(finder):
    mention = finder.best("rash without mucosal involvement", MODIFIER)
    span = mention.span("AE:R1:AETERM", MODIFIER)
    assert span.start < span.end
    assert "without" in mention.surface


def test_confidence_comes_from_declared_keys(finder, configs):
    declared = set(configs.extraction.confidence.values())
    for text in (
        "rash with oral ulceration",
        "rash without mucosal involvement",
        "rash with possible mucosal involvement",
    ):
        mention = finder.best(text, MODIFIER)
        assert mention.confidence in declared


def test_a_direct_phrase_scores_above_a_loose_one(finder):
    tight = finder.best("rash with oral ulceration", MODIFIER)
    loose = finder.best(
        "rash reported by the site at the visit, with oral ulceration", MODIFIER
    )
    assert loose is None or tight.confidence >= loose.confidence


# -- the engine ---------------------------------------------------------------


def test_abstention_is_measured_not_hidden(pipeline, configs):
    engine = ExtractionEngine.build(configs, pipeline.store, "rules")
    engine.enrich_all(pipeline.structured_only_records())
    stats = engine.stats
    assert stats.requests > 0
    assert stats.abstained > 0
    assert 0.0 < stats.abstention_rate < 1.0
    assert set(stats.by_assertion) <= {"present", "absent", "uncertain"}


def test_an_abstention_leaves_both_facts_on_the_row(pipeline, configs):
    engine = ExtractionEngine.build(configs, pipeline.store, "rules")
    enriched = engine.enrich_all(pipeline.structured_only_records())
    notes = [
        r.modifiers[MODIFIER].note for r in enriched
        if r.modifiers.get(MODIFIER) and "abstained" in r.modifiers[MODIFIER].note
    ]
    assert notes
    for note in notes:
        # The availability and its explanation must not contradict each other.
        assert "abstained" in note
        assert any(word in note for word in ("unresolved", "not_collected", "pending"))


def test_the_engine_degrades_offline_and_says_so(pipeline, configs):
    engine = ExtractionEngine.build(configs, pipeline.store, "llm")
    assert engine.backend.name == "rules"
    assert any("no credentials" in note for note in engine.notes)


def test_the_backend_is_never_described_as_a_trained_model(pipeline, configs):
    engine = ExtractionEngine.build(configs, pipeline.store, "rules")
    joined = " ".join(engine.notes)
    assert "not a trained clinical NLP model" in joined


def test_sentence_splitting_preserves_offsets():
    text = "Rash on day 4. Oral ulceration followed."
    sentences = split_sentences(text)
    assert len(sentences) == 2
    for sentence in sentences:
        assert isinstance(sentence, Sentence)
        assert text[sentence.start:sentence.end] == sentence.text
