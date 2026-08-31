"""Concept candidate matching.

Three kinds of surface form, deliberately distinguished because they carry
different weight downstream:

``coded_term``
    the study's own dictionary term, matched against the catalogue
``lexicon``
    a verbatim mention in text, exact or within one edit for longer terms
``abbreviation``
    an ambiguous short form that fires only when its context gate is satisfied

"hypo" is the reason the gate exists.  In a narrative it can mean a
hypoglycaemic episode or nothing at all, and the difference is whether a glucose
value or a qualifying symptom sits nearby.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from ..catalog import ConceptCatalog, ExtractionConfig
from .text import Sentence, Token, edit_distance, normalise, sentence_for, tokenize

MatchKind = Literal["coded_term", "lexicon", "lexicon_fuzzy", "abbreviation"]


@dataclass(frozen=True)
class ConceptMention:
    concept_id: str
    surface: str
    start: int
    end: int
    kind: MatchKind
    confidence: float
    gate_reason: str = ""


@dataclass(frozen=True)
class SymptomMatch:
    symptom: str
    surface: str
    start: int
    end: int


class ConceptMatcher:
    """Finds concept and symptom mentions in a document."""

    def __init__(self, catalog: ConceptCatalog, config: ExtractionConfig):
        self.catalog = catalog
        self.config = config
        norm = config.normalisation or {}
        self.fuzzy_max_edits = int(norm.get("fuzzy_max_edits", 0) or 0)
        self.fuzzy_min_length = int(norm.get("fuzzy_min_length", 9))
        self.use_variants = bool(norm.get("spelling_variants", True))

        # term (normalised, word count) -> list of (concept_id, kind, surface)
        self._exact: dict[str, list[tuple[str, MatchKind, str]]] = {}
        self._fuzzy_terms: list[tuple[str, str, MatchKind, str]] = []
        self._abbreviations: dict[str, list[str]] = {}
        self._max_words = 1
        self._build_concept_index()

        self._symptom_exact: dict[str, list[tuple[str, str]]] = {}
        self._symptom_max_words = 1
        self._build_symptom_index()

    # -- index construction -------------------------------------------------

    def _key(self, phrase: str) -> str:
        return normalise(phrase) if self.use_variants else phrase.lower()

    def _register(self, phrase: str, concept_id: str, kind: MatchKind) -> None:
        key = self._key(phrase)
        words = len(key.split())
        self._max_words = max(self._max_words, words)
        self._exact.setdefault(key, [])
        entry = (concept_id, kind, phrase)
        if entry not in self._exact[key]:
            self._exact[key].append(entry)
        if self.fuzzy_max_edits and len(key) >= self.fuzzy_min_length:
            self._fuzzy_terms.append((key, concept_id, kind, phrase))

    def _build_concept_index(self) -> None:
        for concept_id in sorted(self.catalog.concepts):
            concept = self.catalog.concepts[concept_id]
            for phrase in concept.lexicon:
                self._register(phrase, concept_id, "lexicon")
            for phrase in concept.all_coded_terms():
                # A dictionary term written out in the narrative is still a
                # verbatim mention; the coded-term *field* is handled separately.
                self._register(phrase, concept_id, "lexicon")
            for abbreviation in concept.abbreviations:
                self._abbreviations.setdefault(abbreviation, []).append(concept_id)
        self._fuzzy_terms.sort()

    def _build_symptom_index(self) -> None:
        for symptom in sorted(self.catalog.symptom_lexicon):
            for surface in self.catalog.symptom_lexicon[symptom]:
                key = self._key(surface)
                self._symptom_max_words = max(self._symptom_max_words, len(key.split()))
                self._symptom_exact.setdefault(key, [])
                if (symptom, surface) not in self._symptom_exact[key]:
                    self._symptom_exact[key].append((symptom, surface))

    # -- matching -----------------------------------------------------------

    @staticmethod
    def _windows(tokens: list[Token], max_words: int):
        for i in range(len(tokens)):
            for width in range(1, max_words + 1):
                if i + width > len(tokens):
                    break
                yield i, width, tokens[i], tokens[i + width - 1]

    def find_concepts(
        self,
        text: str,
        sentences: list[Sentence] | None = None,
        *,
        restrict_to: set[str] | None = None,
    ) -> list[ConceptMention]:
        """All concept mentions in ``text``, sorted by position.

        Overlapping matches are resolved in favour of the longest span, so
        "low blood glucose" is one mention rather than three.
        """
        tokens = tokenize(text)
        mentions: list[ConceptMention] = []
        matched_spans: set[tuple[int, int]] = set()

        for _, width, first, last in self._windows(tokens, self._max_words):
            phrase = text[first.start : last.end]
            key = self._key(phrase)
            for concept_id, kind, surface in self._exact.get(key, []):
                if restrict_to and concept_id not in restrict_to:
                    continue
                mentions.append(
                    ConceptMention(
                        concept_id=concept_id,
                        surface=phrase,
                        start=first.start,
                        end=last.end,
                        kind=kind,
                        confidence=self.config.confidence_for("lexicon_exact", 0.95),
                    )
                )
                matched_spans.add((first.start, last.end))

        if self.fuzzy_max_edits:
            mentions.extend(
                self._fuzzy_matches(text, tokens, matched_spans, restrict_to)
            )

        mentions.extend(
            self._abbreviation_matches(text, tokens, sentences, restrict_to)
        )
        return _resolve_overlaps(mentions)

    def _fuzzy_matches(
        self,
        text: str,
        tokens: list[Token],
        matched_spans: set[tuple[int, int]],
        restrict_to: set[str] | None,
    ) -> list[ConceptMention]:
        """Near-miss matches for longer terms, e.g. a transposed letter.

        Only applied to spans that did not already match something exactly, so
        a correctly spelled term is never reinterpreted as a misspelling of a
        different concept.
        """
        found: list[ConceptMention] = []
        seen: set[tuple[int, int, str]] = set()
        for _, width, first, last in self._windows(tokens, self._max_words):
            span = (first.start, last.end)
            if span in matched_spans:
                continue
            phrase = text[first.start : last.end]
            key = self._key(phrase)
            if len(key) < self.fuzzy_min_length:
                continue
            best: tuple[int, str, MatchKind] | None = None
            for term, concept_id, kind, _surface in self._fuzzy_terms:
                if restrict_to and concept_id not in restrict_to:
                    continue
                if abs(len(term) - len(key)) > self.fuzzy_max_edits:
                    continue
                distance = edit_distance(term, key, maximum=self.fuzzy_max_edits)
                if distance <= self.fuzzy_max_edits and distance > 0:
                    if best is None or distance < best[0]:
                        best = (distance, concept_id, kind)
            if best is not None and (span[0], span[1], best[1]) not in seen:
                seen.add((span[0], span[1], best[1]))
                found.append(
                    ConceptMention(
                        concept_id=best[1],
                        surface=phrase,
                        start=first.start,
                        end=last.end,
                        kind="lexicon_fuzzy",
                        confidence=self.config.confidence_for("lexicon_fuzzy", 0.75),
                        gate_reason=f"within {best[0]} edit of a catalogue term",
                    )
                )
        return found

    def _abbreviation_matches(
        self,
        text: str,
        tokens: list[Token],
        sentences: list[Sentence] | None,
        restrict_to: set[str] | None,
    ) -> list[ConceptMention]:
        if not self._abbreviations:
            return []
        sentences = sentences if sentences is not None else []
        found: list[ConceptMention] = []
        for token in tokens:
            for abbreviation in sorted(self._abbreviations):
                if not _abbreviation_equal(token.text, abbreviation):
                    continue
                for concept_id in self._abbreviations[abbreviation]:
                    if restrict_to and concept_id not in restrict_to:
                        continue
                    concept = self.catalog.concept(concept_id)
                    if not concept.abbreviations_gated:
                        found.append(
                            ConceptMention(
                                concept_id, token.text, token.start, token.end,
                                "abbreviation",
                                self.config.confidence_for("lexicon_exact", 0.95),
                            )
                        )
                        continue
                    satisfied, reason = self._gate_satisfied(
                        concept_id, text, token, sentences
                    )
                    if satisfied:
                        found.append(
                            ConceptMention(
                                concept_id, token.text, token.start, token.end,
                                "abbreviation",
                                self.config.confidence_for("abbreviation_gated", 0.70),
                                gate_reason=reason,
                            )
                        )
        return found

    def _gate_satisfied(
        self, concept_id: str, text: str, token: Token, sentences: list[Sentence]
    ) -> tuple[bool, str]:
        """Does the abbreviation's required context appear within its scope?"""
        concept = self.catalog.concept(concept_id)
        gate = concept.context_gate or {}
        sentence = sentence_for(sentences, token.start)
        if sentence is None:
            scope_text, scope_offset = text, 0
        else:
            scope_text, scope_offset = sentence.text, sentence.start

        for test_id in gate.get("lab_tests") or []:
            lab = self.catalog.lab_tests.get(test_id)
            if lab is None:
                continue
            for name in sorted(lab.names, key=len, reverse=True):
                pattern = re.compile(
                    rf"\b{re.escape(name)}\b\D{{0,20}}\d", re.IGNORECASE
                )
                if pattern.search(scope_text):
                    return True, f"{test_id} value in scope"

        set_names = gate.get("symptom_sets") or []
        if set_names:
            qualifying = self.catalog.symptoms_in_sets(list(set_names))
            for match in self.find_symptoms(scope_text, offset=scope_offset):
                if match.symptom in qualifying:
                    return True, f"symptom '{match.symptom}' in scope"
        return False, "context gate not satisfied"

    def find_symptoms(self, text: str, offset: int = 0) -> list[SymptomMatch]:
        tokens = tokenize(text, offset=0)
        found: list[SymptomMatch] = []
        claimed: set[int] = set()
        for _, width, first, last in sorted(
            self._windows(tokens, self._symptom_max_words),
            key=lambda w: (-w[1], w[2].start),
        ):
            phrase = text[first.start : last.end]
            key = self._key(phrase)
            entries = self._symptom_exact.get(key)
            if not entries:
                continue
            span = range(first.start, last.end)
            if any(position in claimed for position in span):
                continue
            claimed.update(span)
            symptom, _surface = entries[0]
            found.append(
                SymptomMatch(symptom, phrase, offset + first.start, offset + last.end)
            )
        return sorted(found, key=lambda m: (m.start, m.end))

    # -- coded term field ---------------------------------------------------

    def coded_term_concept(self, coded_term: str | None) -> str | None:
        """Which catalogue concept, if any, a dictionary term denotes.

        Membership only.  No hierarchy is walked: a term is a concept's coded
        term because the catalogue lists it, never because it sits beneath one.
        """
        if not coded_term:
            return None
        key = self._key(coded_term.strip())
        for concept_id in sorted(self.catalog.concepts):
            concept = self.catalog.concepts[concept_id]
            if any(self._key(term) == key for term in concept.all_coded_terms()):
                return concept_id
        return None


def _abbreviation_equal(token_text: str, abbreviation: str) -> bool:
    """Case-sensitive for all-caps abbreviations, case-insensitive otherwise.

    "LOC" in lower case is almost always a different word; "hypo" in any case is
    still the same shorthand.
    """
    if abbreviation.isupper():
        return token_text == abbreviation
    return token_text.lower() == abbreviation.lower()


def _resolve_overlaps(mentions: list[ConceptMention]) -> list[ConceptMention]:
    """Keep the longest mention where spans overlap, then dedupe."""
    ordered = sorted(
        mentions, key=lambda m: (m.start, -(m.end - m.start), m.concept_id, m.kind)
    )
    kept: list[ConceptMention] = []
    for mention in ordered:
        overlapping = [
            k for k in kept
            if k.start < mention.end and mention.start < k.end
            and k.concept_id == mention.concept_id
        ]
        if overlapping:
            continue
        kept.append(mention)
    return sorted(kept, key=lambda m: (m.start, m.end, m.concept_id))
