"""Finding modifier mentions in language, and normalizing them to the catalogue.

A modifier is only useful if it belongs to the event in question. "Rash with
oral ulceration; prior conjunctivitis, resolved" contains two mucosal sites and
one of them is not this event's. So a mention is anchored: it must be reachable
from a concept mention in the same sentence, within a declared token distance.

Recall here is exactly the surface forms declared in ``concepts.yaml``. That is
a property of the configuration, not of the method, and the README says so. A
phrase nobody wrote into the catalogue produces no mention at all — the
extractor abstains rather than inventing a value, and the abstention is
measured.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..catalog import ConceptCatalog, ExtractionConfig, ModifierCatalogue
from ..models import Span
from .assertion import AssertionCall, AssertionClassifier
from .text import Sentence, sentence_for, split_sentences, tokenize


@dataclass(frozen=True)
class ModifierMention:
    """One modifier mention: what it says, where it is, and how sure."""

    modifier: str
    #: The catalogue value, where the surface form names one. `None` is a real
    #: answer: "mucosal involvement" asserts the modifier without naming a site.
    value: str | None
    assertion: str
    surface: str
    start: int
    end: int
    confidence: float
    rule: str
    anchor_surface: str = ""
    cue: str | None = None

    def span(self, doc_id: str, field: str) -> Span:
        """The evidence span, widened to include the governing cue.

        A reader checking a documented negative needs to see the "without", not
        just the phrase it negates.
        """
        return Span(
            doc_id=doc_id,
            start=self.start,
            end=self.end,
            field=field,
            extracted_value=self.value or self.assertion,
            text=self.surface,
            kind="text",
        )


#: Words that carry no meaning between an event and its modifier.
_FILLERS = frozenset({"the", "a", "an", "his", "her", "their", "both", "of",
                      "with", "and"})


def phrase_pattern(phrase: str) -> re.Pattern[str]:
    parts = [re.escape(part) for part in phrase.split()]
    return re.compile(r"\b" + r"[\s\-]+".join(parts) + r"\b", re.IGNORECASE)


class MentionFinder:
    """The deterministic model-path backend.

    A lexicon and a scoping rule, not a trained model, and described as such
    everywhere it appears.
    """

    def __init__(self, catalog: ConceptCatalog, config: ExtractionConfig):
        self.catalog = catalog
        self.config = config
        self.classifier = AssertionClassifier(config)
        self.max_distance = int(config.assertion.get("max_distance_tokens", 12))
        self.modifiers: dict[str, ModifierCatalogue] = {
            name: catalog.modifier(name)
            for name in config.extractable_modifiers
            if name in catalog.modifiers
        }
        # Concept surface forms, so a mention can be anchored to the event it
        # belongs to rather than to whatever else the sentence mentions.
        self.concept_forms: list[tuple[str, str]] = sorted(
            (
                (form, concept.concept_id)
                for concept in catalog.concepts.values()
                for form in concept.lexicon
            ),
            key=lambda pair: -len(pair[0]),
        )

    # -- anchors ------------------------------------------------------------

    def concept_mentions(
        self, text: str, concept_id: str | None = None
    ) -> list[tuple[int, int, str, str]]:
        """Every concept mention in the text, longest form first."""
        found: list[tuple[int, int, str, str]] = []
        for form, cid in self.concept_forms:
            if concept_id and cid != concept_id:
                continue
            for match in phrase_pattern(form).finditer(text):
                if any(
                    start <= match.start() and match.end() <= end
                    for start, end, _surface, _cid in found
                ):
                    continue
                found.append((match.start(), match.end(), match.group(0), cid))
        return sorted(found)

    # -- mentions -----------------------------------------------------------

    def find(
        self, text: str, modifier: str, concept_id: str | None = None,
        source_kind: str = "reported_term",
    ) -> list[ModifierMention]:
        """Every mention of one modifier, anchored and classified."""
        catalogue = self.modifiers.get(modifier)
        if catalogue is None:
            return []
        sentences = split_sentences(text)
        anchors = self.concept_mentions(text, concept_id)
        mentions: list[ModifierMention] = []

        for phrase, value_id in sorted(
            catalogue.surface_index().items(), key=lambda kv: -len(kv[0])
        ):
            for match in phrase_pattern(phrase).finditer(text):
                if any(
                    m.start <= match.start() and match.end() <= m.end
                    for m in mentions
                ):
                    continue
                scored = self._score(
                    text, sentences, anchors, match, modifier, value_id,
                    source_kind,
                )
                if scored is not None:
                    mentions.append(scored)

        return sorted(mentions, key=lambda m: (m.start, -m.confidence))

    def _score(
        self, text: str, sentences: list[Sentence],
        anchors: list[tuple[int, int, str, str]], match: re.Match[str],
        modifier: str, value_id: str, source_kind: str,
    ) -> ModifierMention | None:
        """Decide whether this mention belongs to the event, and what it says."""
        sentence = sentence_for(sentences, match.start())
        if sentence is None:
            return None
        in_scope = [a for a in anchors if sentence.contains(a[0])]
        anchor: tuple[int, int, str, str] | None = None
        between = ""
        if in_scope:
            anchor = min(in_scope, key=lambda a: abs(a[0] - match.start()))
            # The words *between* the two, in whichever order they appear.
            if match.start() >= anchor[1]:
                between = text[anchor[1]:match.start()]
            else:
                between = text[match.end():anchor[0]]
            if len(tokenize(between)) > self.max_distance:
                return None
        elif source_kind == "comment":
            # A comment record points at one AE row through IDVAR/IDVARVAL. The
            # link is structural, so the record *is* the anchor and the comment
            # need not name the event again — "Investigator comment: no oral
            # ulceration was seen" is about this event by construction.
            pass
        else:
            # Nothing in this sentence says which event the modifier belongs to.
            # Abstaining is the correct answer.
            return None

        call = self.classifier.classify(text, sentence, match.start(), match.end())
        confidence_key, rule = self._support(
            between, source_kind, call, textual_anchor=anchor is not None
        )
        return ModifierMention(
            modifier=modifier,
            value=value_id if value_id != "UNSPECIFIED" else None,
            assertion=call.assertion,
            surface=self._surface(text, match, call),
            start=min(match.start(), call.cue_start if call.has_cue else match.start()),
            end=max(match.end(), call.cue_end if call.has_cue else match.end()),
            confidence=self.config.confidence_for(confidence_key, 0.7),
            rule=f"{rule}; {call.reason}",
            anchor_surface=anchor[2] if anchor else "(the comment's own record)",
            cue=call.cue,
        )

    def _support(
        self, between: str, source_kind: str, call: AssertionCall, *,
        textual_anchor: bool,
    ) -> tuple[str, str]:
        """Which declared confidence key this mention earns, and why.

        Confidence reflects how the mention was anchored and whether a cue
        governed it — not how clinically plausible the value looks.
        """
        if call.confidence_key:
            # An explicit cue is the strongest thing the text can offer about
            # what it asserts, in either direction.
            return call.confidence_key, (
                "an explicit cue governs the mention"
                if call.assertion == "absent"
                else "the source itself hedges"
            )
        if not textual_anchor:
            return "comment_record", (
                "a comment written about this record, which is the anchor"
            )
        core = [t for t in between.lower().split() if t not in _FILLERS]
        if not core:
            return "direct_phrase", "written as one phrase with the event"
        if source_kind == "comment":
            # A comment is written about this record specifically, so a
            # same-sentence mention in one is better supported than the same
            # mention in general prose.
            return "comment_record", (
                "same sentence in a comment written about this record"
            )
        return "same_sentence", "same sentence as the event mention"

    @staticmethod
    def _surface(text: str, match: re.Match[str], call: AssertionCall) -> str:
        if not call.has_cue:
            return match.group(0)
        start = min(match.start(), call.cue_start)
        end = max(match.end(), call.cue_end)
        return text[start:end]

    # -- the one answer -----------------------------------------------------

    def best(
        self, text: str, modifier: str, concept_id: str | None = None,
        source_kind: str = "reported_term",
    ) -> ModifierMention | None:
        """The single mention to believe, or ``None``.

        ``None`` is a real answer, recorded as an abstention rather than a
        guess. Two rules produce it: no mention at all, and two equally
        supported mentions that disagree.
        """
        mentions = self.find(text, modifier, concept_id, source_kind)
        if not mentions:
            return None
        best = max(mentions, key=lambda m: (m.confidence, -m.start))
        top = [m for m in mentions if m.confidence >= best.confidence]
        if len({m.assertion for m in top}) > 1:
            # The text says two different things about the same modifier with
            # equal support. Picking one would assert something it does not
            # settle.
            return None
        if len({m.value for m in top if m.value is not None}) > 1:
            return None
        return best
