"""Deciding what a sentence *says* about a modifier: present, absent, uncertain.

This is the single most consequential thing the model path does, and the reason
`assertion` and `availability` are separate fields on every attribute. A
narrative that says "rash without mucosal involvement" is not silence. Somebody
looked, and the answer was no. That subject is a `non_case` and belongs in the
denominator. Collapse the two fields and that subject disappears into the same
bucket as one nobody ever examined, which biases every rate the layer reports.

The classifier is a cue-scoping rule over the lists declared in
``extraction.yaml``. It is not a trained model and is described as such
everywhere it appears; its recall is exactly the cues somebody wrote down.

Three properties are deliberate:

* **A cue governs a mention only within a scope.** A pre-cue governs forward
  until a terminator; a post-cue governs backward the same way. "No rash, but
  oral ulceration was noted" does not make the ulceration negative.
* **A pseudo-cue is not a cue.** "No dose change" contains "no " and asserts
  nothing about any modifier.
* **The default is `present`, and that is a claim about writing, not about
  clinical reality.** A clinician who writes the phrase at all is asserting it.
  Absence of a negation cue is never evidence of presence: where no mention is
  found at all, the extractor abstains and returns nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..catalog import ExtractionConfig
from .text import Sentence, normalise


@dataclass(frozen=True)
class AssertionCall:
    """What the text says about one mention, and what decided it."""

    assertion: str
    cue: str | None
    cue_start: int | None
    cue_end: int | None
    #: The key in ``extraction.yaml``'s `confidence` block this call implies.
    confidence_key: str
    reason: str

    @property
    def has_cue(self) -> bool:
        return self.cue is not None


def _cue_pattern(cue: str) -> re.Pattern[str]:
    """A cue matches on word boundaries, tolerating hyphens and extra spaces."""
    body = r"[\s\-]+".join(re.escape(part) for part in cue.split())
    lead = r"\b" if cue[:1].isalnum() else ""
    trail = r"\b" if cue.strip()[-1:].isalnum() else ""
    return re.compile(lead + body + trail, re.IGNORECASE)


class AssertionClassifier:
    """Cue scoping over one sentence.

    Everything it knows comes from ``extraction.yaml``; nothing is hard-coded
    here except the shape of the scoping rule itself.
    """

    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.default: str = config.assertion.get("default", "present")
        self.scope: str = config.assertion.get("scope", "sentence")
        self.max_distance = int(config.assertion.get("max_distance_tokens", 12))
        self.terminators = tuple(config.assertion.get("terminators") or ())
        self.pseudo_cues = tuple(
            normalise(p) for p in (config.assertion.get("pseudo_cues") or ())
        )
        # Longest cue first, so "cannot be excluded" wins over "cannot exclude".
        self.pre: list[tuple[str, str, re.Pattern[str]]] = []
        self.post: list[tuple[str, str, re.Pattern[str]]] = []
        for assertion_class in ("absent", "uncertain"):
            pre, post = config.cue_lists(assertion_class)
            for cue in pre:
                self.pre.append((assertion_class, cue, _cue_pattern(cue)))
            for cue in post:
                self.post.append((assertion_class, cue, _cue_pattern(cue)))
        self.pre.sort(key=lambda item: -len(item[1]))
        self.post.sort(key=lambda item: -len(item[1]))

    # -- scoping ------------------------------------------------------------

    def _is_pseudo(self, text: str, start: int, end: int) -> bool:
        """Does this cue occurrence sit inside a declared pseudo-cue phrase?"""
        window = normalise(text[max(0, start - 4):end + 40])
        return any(
            phrase and phrase in window and phrase.startswith(
                normalise(text[start:end]).strip()
            )
            for phrase in self.pseudo_cues
        )

    def _blocked(self, text: str) -> bool:
        """A terminator between the cue and the mention ends the cue's scope."""
        folded = normalise(text)
        return any(
            re.search(r"\b" + re.escape(t) + r"\b" if t.isalnum() else re.escape(t),
                      folded)
            for t in self.terminators
        )

    # -- the call -----------------------------------------------------------

    def classify(
        self, text: str, sentence: Sentence, start: int, end: int
    ) -> AssertionCall:
        """Classify the mention at ``[start, end)`` within its sentence."""
        before = text[sentence.start:start]
        after = text[end:sentence.end]

        best: AssertionCall | None = None
        for assertion_class, cue, pattern in self.pre:
            for match in pattern.finditer(before):
                gap = before[match.end():]
                if len(gap.split()) > self.max_distance:
                    continue
                if self._blocked(gap):
                    continue
                absolute = sentence.start + match.start()
                if self._is_pseudo(text, absolute, sentence.start + match.end()):
                    continue
                best = self._prefer(best, AssertionCall(
                    assertion=assertion_class,
                    cue=match.group(0),
                    cue_start=absolute,
                    cue_end=sentence.start + match.end(),
                    confidence_key=(
                        "negated" if assertion_class == "absent" else "uncertain"
                    ),
                    reason=(
                        f"{match.group(0)!r} governs the mention "
                        f"({assertion_class})"
                    ),
                ))

        for assertion_class, cue, pattern in self.post:
            for match in pattern.finditer(after):
                gap = after[:match.start()]
                if len(gap.split()) > self.max_distance:
                    continue
                if self._blocked(gap):
                    continue
                absolute = end + match.start()
                if self._is_pseudo(text, absolute, end + match.end()):
                    continue
                best = self._prefer(best, AssertionCall(
                    assertion=assertion_class,
                    cue=match.group(0),
                    cue_start=absolute,
                    cue_end=end + match.end(),
                    confidence_key=(
                        "negated" if assertion_class == "absent" else "uncertain"
                    ),
                    reason=(
                        f"{match.group(0)!r} follows the mention "
                        f"({assertion_class})"
                    ),
                ))

        if best is not None:
            return best
        return AssertionCall(
            assertion=self.default,
            cue=None,
            cue_start=None,
            cue_end=None,
            confidence_key="",
            reason=(
                f"no cue governs the mention, so it is read as "
                f"{self.default!r}: writing the phrase asserts it"
            ),
        )

    @staticmethod
    def _prefer(current: AssertionCall | None, candidate: AssertionCall):
        """`uncertain` beats `absent`: a hedge over a negation is still a hedge.

        "Cannot exclude mucosal involvement, no ulceration seen" is not a
        documented negative, and calling it one would put the subject in the
        denominator on the strength of a sentence that refuses to commit.
        """
        if current is None:
            return candidate
        rank = {"uncertain": 2, "absent": 1, "present": 0}
        if rank[candidate.assertion] > rank[current.assertion]:
            return candidate
        if rank[candidate.assertion] < rank[current.assertion]:
            return current
        # Same class: the cue nearest the mention governs.
        return candidate if len(candidate.cue or "") > len(current.cue or "") else current
