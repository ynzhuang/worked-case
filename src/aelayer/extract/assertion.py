"""Assertion classification, ConText/NegEx style.

Assertion is a first-class field, not a confidence discount.  "No evidence of
hypoglycemia" is not a low-confidence hypoglycemia event; it is a documented
absence, and it must be storable, queryable and filterable as such.

All six classes are covered:
``present`` ``absent`` ``hypothetical`` ``historical`` ``family_history``
``uncertain``

The classifier works by cue and scope.  A cue governs a concept mention when it
appears within the mention's scope on the correct side, and no terminator sits
between them.  Pseudo-cues suppress phrases that merely contain a cue token
("no change to study drug" does not negate anything).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from ..catalog import ExtractionConfig
from ..models import ASSERTION_VALUES
from .text import Sentence, sentence_for, tokenize


@dataclass(frozen=True)
class AssertionResult:
    assertion: str
    cue: str | None
    cue_start: int | None
    cue_end: int | None
    confidence: float
    rule: str

    @property
    def has_cue(self) -> bool:
        return self.cue is not None


@lru_cache(maxsize=512)
def _cue_pattern(cue: str) -> re.Pattern[str]:
    """Word-bounded, case-insensitive matcher for one cue phrase."""
    stripped = cue.strip()
    if not stripped:
        return re.compile(r"(?!x)x")  # never matches
    escaped = re.escape(stripped).replace(r"\ ", r"\s+")
    lead = r"\b" if stripped[0].isalnum() else ""
    trail = r"\b" if stripped[-1].isalnum() else ""
    return re.compile(f"{lead}{escaped}{trail}", re.IGNORECASE)


class AssertionClassifier:
    def __init__(self, config: ExtractionConfig):
        self.config = config
        assertion = config.assertion
        self.scope = assertion.get("scope", "sentence")
        self.window_tokens = int(assertion.get("window_tokens", 6))
        self.default = assertion.get("default", "present")
        self.terminators = list(assertion.get("terminators") or [])
        self.pseudo_cues = list(assertion.get("pseudo_cues") or [])
        declared = list(assertion.get("precedence") or [])
        classes = [c for c in (assertion.get("cues") or {}) if c in ASSERTION_VALUES]
        # Anything with cues but no declared precedence goes last, in a stable
        # order, so behaviour never depends on YAML key ordering.
        self.precedence = declared + sorted(set(classes) - set(declared))
        self.cues: dict[str, tuple[list[str], list[str]]] = {
            cls: config.cue_lists(cls) for cls in classes
        }

    # -- scope --------------------------------------------------------------

    def scope_for(
        self, text: str, mention_start: int, mention_end: int, sentences: list[Sentence]
    ) -> tuple[int, int]:
        """The character range a cue must fall inside to govern this mention."""
        sentence = sentence_for(sentences, mention_start)
        if sentence is None:
            low, high = 0, len(text)
        else:
            low, high = sentence.start, sentence.end
        if self.scope == "window":
            tokens = tokenize(text)
            before = [t for t in tokens if t.end <= mention_start]
            after = [t for t in tokens if t.start >= mention_end]
            if before:
                low = max(low, before[max(0, len(before) - self.window_tokens)].start)
            if after:
                index = min(len(after), self.window_tokens) - 1
                high = min(high, after[index].end)
        return low, high

    # -- cue detection ------------------------------------------------------

    def _pseudo_spans(self, text: str, low: int, high: int) -> list[tuple[int, int]]:
        spans = []
        window = text[low:high]
        for phrase in self.pseudo_cues:
            for match in _cue_pattern(phrase).finditer(window):
                spans.append((low + match.start(), low + match.end()))
        return spans

    def _terminator_positions(self, text: str, low: int, high: int) -> list[int]:
        positions = []
        window = text[low:high]
        for terminator in self.terminators:
            for match in _cue_pattern(terminator).finditer(window):
                positions.append(low + match.start())
        return sorted(positions)

    def classify(
        self,
        text: str,
        mention_start: int,
        mention_end: int,
        sentences: list[Sentence],
    ) -> AssertionResult:
        """Classify one concept mention."""
        low, high = self.scope_for(text, mention_start, mention_end, sentences)
        pseudo = self._pseudo_spans(text, low, high)
        terminators = self._terminator_positions(text, low, high)

        for assertion_class in self.precedence:
            pre_cues, post_cues = self.cues.get(assertion_class, ([], []))
            hit = self._first_governing_cue(
                text, low, high, mention_start, mention_end,
                pre_cues, post_cues, pseudo, terminators,
            )
            if hit is not None:
                cue_text, start, end = hit
                return AssertionResult(
                    assertion=assertion_class,
                    cue=cue_text,
                    cue_start=start,
                    cue_end=end,
                    confidence=self.config.confidence_for("assertion_cue", 0.90),
                    rule=f"cue '{cue_text}' governs the mention in {self.scope} scope",
                )

        return AssertionResult(
            assertion=self.default,
            cue=None,
            cue_start=None,
            cue_end=None,
            confidence=self.config.confidence_for("assertion_default", 0.80),
            rule=f"no assertion cue in {self.scope} scope; default '{self.default}'",
        )

    def _first_governing_cue(
        self,
        text: str,
        low: int,
        high: int,
        mention_start: int,
        mention_end: int,
        pre_cues: list[str],
        post_cues: list[str],
        pseudo: list[tuple[int, int]],
        terminators: list[int],
    ) -> tuple[str, int, int] | None:
        """The nearest cue of one class that actually reaches the mention."""
        best: tuple[int, str, int, int] | None = None

        for cue in pre_cues:
            for match in _cue_pattern(cue).finditer(text[low:high]):
                start, end = low + match.start(), low + match.end()
                if end > mention_start:
                    continue  # a pre-cue must precede the mention
                if _inside(start, end, pseudo):
                    continue
                if any(end <= position < mention_start for position in terminators):
                    continue
                distance = mention_start - end
                if best is None or distance < best[0]:
                    best = (distance, text[start:end], start, end)

        for cue in post_cues:
            for match in _cue_pattern(cue).finditer(text[low:high]):
                start, end = low + match.start(), low + match.end()
                if start < mention_end:
                    continue  # a post-cue must follow the mention
                if _inside(start, end, pseudo):
                    continue
                if any(mention_end <= position < start for position in terminators):
                    continue
                distance = start - mention_end
                if best is None or distance < best[0]:
                    best = (distance, text[start:end], start, end)

        if best is None:
            return None
        _, cue_text, start, end = best
        return cue_text, start, end


def _inside(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(low <= start and end <= high for low, high in spans)
