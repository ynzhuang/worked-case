"""Offset-preserving text utilities.

Every function here returns character offsets into the original document, so
that any value derived downstream can be traced back to the span it came from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.?!])\s+(?=[\"'(]?[A-Z0-9])")
_TOKEN = re.compile(r"[A-Za-z][A-Za-z'’-]*|\d+(?:\.\d+)?")

#: British/American and -ise/-ize variation, applied to both the lexicon and the
#: text so a single normalised comparison covers both.
_SPELLING_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"isation\b"), "ization"),
    (re.compile(r"ise\b"), "ize"),
    (re.compile(r"ised\b"), "ized"),
    (re.compile(r"ising\b"), "izing"),
    (re.compile(r"ae"), "e"),
    (re.compile(r"oe"), "e"),
)


@dataclass(frozen=True)
class Sentence:
    text: str
    start: int
    end: int

    def contains(self, position: int) -> bool:
        return self.start <= position < self.end


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int


def split_sentences(text: str) -> list[Sentence]:
    """Split into sentences, preserving offsets into ``text``.

    The header line of a narrative is treated as its own sentence so that a
    concept word appearing in a title can never fall inside the assertion scope
    of the body.
    """
    sentences: list[Sentence] = []
    for line_start, line in _iter_lines(text):
        cursor = 0
        for piece in _SENTENCE_BOUNDARY.split(line):
            if not piece:
                continue
            offset = line.index(piece, cursor)
            cursor = offset + len(piece)
            stripped = piece.strip()
            if not stripped:
                continue
            lead = len(piece) - len(piece.lstrip())
            start = line_start + offset + lead
            sentences.append(Sentence(stripped, start, start + len(stripped)))
    return sentences


def _iter_lines(text: str):
    position = 0
    for line in text.split("\n"):
        yield position, line
        position += len(line) + 1


def tokenize(text: str, offset: int = 0) -> list[Token]:
    return [
        Token(m.group(0), offset + m.start(), offset + m.end())
        for m in _TOKEN.finditer(text)
    ]


def normalise(text: str) -> str:
    """Lowercase and fold spelling variants for matching purposes only."""
    folded = text.lower()
    for pattern, replacement in _SPELLING_RULES:
        folded = pattern.sub(replacement, folded)
    return folded


def edit_distance(a: str, b: str, *, maximum: int = 2) -> int:
    """Optimal string alignment (Damerau-Levenshtein) distance.

    Adjacent transposition counts as a single edit, because "hypoglycemai" is
    one typo for "hypoglycemia" and not two.  Plain Levenshtein scores it 2 and
    would miss the most common class of clinical typo at an edit budget of 1.

    Abandoned once the distance exceeds ``maximum``: the only question ever
    asked is whether two strings are within a small edit budget.
    """
    if abs(len(a) - len(b)) > maximum:
        return maximum + 1
    rows = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        rows[i][0] = i
    for j in range(len(b) + 1):
        rows[0][j] = j
    for i in range(1, len(a) + 1):
        best = rows[i][0]
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            value = min(
                rows[i - 1][j] + 1,        # deletion
                rows[i][j - 1] + 1,        # insertion
                rows[i - 1][j - 1] + cost,  # substitution
            )
            if (
                i > 1
                and j > 1
                and a[i - 1] == b[j - 2]
                and a[i - 2] == b[j - 1]
            ):
                value = min(value, rows[i - 2][j - 2] + 1)  # transposition
            rows[i][j] = value
            best = min(best, value)
        if best > maximum:
            return maximum + 1
    return rows[len(a)][len(b)]


def sentence_for(sentences: list[Sentence], position: int) -> Sentence | None:
    for sentence in sentences:
        if sentence.contains(position):
            return sentence
    return None
