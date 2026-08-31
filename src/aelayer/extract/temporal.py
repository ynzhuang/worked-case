"""Onset timing: relative expressions from text, anchor dates from tables.

The anchor comes from structured data, the offset from text.  A narrative says
"six days after dose escalation"; the escalation date lives in the exposure
domain.  This module resolves the relative expression against the structured
record.  If no anchor can be resolved, ``onset_offset_days`` is populated and
``onset_date`` stays null, and the phenotype definition decides what to do with
that — not the extractor.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from typing import Any

from ..anchors import AnchorResolver, parse_date
from ..catalog import ExtractionConfig
from .text import Sentence, normalise


@dataclass(frozen=True)
class TemporalMention:
    """One temporal expression found in text."""

    pattern_id: str
    surface: str
    start: int
    end: int
    offset_days: int | None = None
    study_day: int | None = None
    absolute_date: _dt.date | None = None
    anchor_phrase: str | None = None
    anchor_event: str | None = None
    vague: bool = False


@dataclass
class OnsetResolution:
    """The extractor's finding about when an event started."""

    onset_date: _dt.date | None = None
    onset_offset_days: int | None = None
    anchor_event: str | None = None
    anchor_date: _dt.date | None = None
    source: str = "unresolved"
    confidence: float = 0.0
    mention: TemporalMention | None = None
    detail: str = "no temporal expression and no recorded onset date"

    @property
    def resolved(self) -> bool:
        return self.onset_date is not None or self.onset_offset_days is not None


class TemporalExtractor:
    def __init__(self, config: ExtractionConfig, anchor_resolver: AnchorResolver | None = None):
        self.config = config
        self.resolver = anchor_resolver
        temporality = config.temporality
        self.patterns: list[tuple[str, re.Pattern[str], dict[str, Any]]] = [
            (spec["id"], re.compile(spec["pattern"], re.IGNORECASE), spec)
            for spec in temporality.get("offset_patterns", [])
        ]
        self.number_words: dict[str, int] = {
            str(k).lower(): int(v) for k, v in (temporality.get("number_words") or {}).items()
        }
        self.vague = {str(v).lower() for v in (temporality.get("vague_quantifiers") or [])}
        self.unit_days: dict[str, int] = {
            str(k).lower(): int(v) for k, v in (temporality.get("unit_days") or {}).items()
        }
        self.anchor_aliases: dict[str, list[str]] = {
            event: [normalise(a) for a in aliases]
            for event, aliases in (temporality.get("anchor_aliases") or {}).items()
        }

    # -- text side ----------------------------------------------------------

    def find_mentions(self, text: str) -> list[TemporalMention]:
        """Every temporal expression in ``text``, in document order.

        Patterns are ordered in config; where two match the same span the
        earlier pattern wins, so behaviour is a property of the config rather
        than of regex evaluation order.
        """
        found: list[TemporalMention] = []
        claimed: list[tuple[int, int]] = []
        for pattern_id, pattern, spec in self.patterns:
            for match in pattern.finditer(text):
                span = (match.start(), match.end())
                if any(s < span[1] and span[0] < e for s, e in claimed):
                    continue
                mention = self._build_mention(pattern_id, match, text, spec)
                if mention is None:
                    continue
                claimed.append(span)
                found.append(mention)
        return sorted(found, key=lambda m: (m.start, m.end))

    def _build_mention(
        self, pattern_id: str, match: re.Match[str], text: str, spec: dict[str, Any]
    ) -> TemporalMention | None:
        groups = match.groupdict()
        surface = match.group(0)
        start, end = match.start(), match.end()

        # A pattern may declare a fixed offset ("the day after", "on the day
        # of") rather than capturing a magnitude to convert.
        if spec.get("offset_days") is not None:
            anchor_phrase = (groups.get("anchor") or "").strip() or None
            return TemporalMention(
                pattern_id, surface, start, end,
                offset_days=int(spec["offset_days"]),
                anchor_phrase=anchor_phrase,
                anchor_event=self.match_anchor(anchor_phrase) if anchor_phrase else None,
            )

        if groups.get("day") is not None:
            return TemporalMention(
                pattern_id, surface, start, end, study_day=int(groups["day"])
            )
        if groups.get("date") is not None:
            parsed = _parse_narrative_date(groups["date"])
            if parsed is None:
                return None
            return TemporalMention(pattern_id, surface, start, end, absolute_date=parsed)

        raw_n = (groups.get("n") or "").strip().lower()
        if not raw_n:
            return None
        magnitude = self.number_words.get(raw_n)
        if magnitude is None:
            try:
                magnitude = int(raw_n)
            except ValueError:
                return None
        unit = (groups.get("unit") or "day").strip().lower()
        multiplier = self.unit_days.get(unit)
        if multiplier is None:
            multiplier = self.unit_days.get(unit.rstrip("s"), 1)
        offset = magnitude * multiplier

        anchor_phrase = (groups.get("anchor") or "").strip() or None
        anchor_event = self.match_anchor(anchor_phrase) if anchor_phrase else None
        return TemporalMention(
            pattern_id, surface, start, end,
            offset_days=offset,
            anchor_phrase=anchor_phrase,
            anchor_event=anchor_event,
            vague=raw_n in self.vague,
        )

    def match_anchor(self, phrase: str | None) -> str | None:
        """Map an anchor phrase in text onto a configured anchor event."""
        if not phrase:
            return None
        folded = normalise(phrase)
        best: tuple[int, str] | None = None
        for event, aliases in sorted(self.anchor_aliases.items()):
            for alias in aliases:
                if alias in folded:
                    # Longest alias wins: "dose escalation" beats "escalation".
                    if best is None or len(alias) > best[0]:
                        best = (len(alias), event)
        return best[1] if best else None

    # -- resolution ---------------------------------------------------------

    def resolve(
        self,
        *,
        subject_id: str,
        text: str,
        scope: Sentence | None,
        default_anchor: str | None,
        index_rule: str = "first_occurrence",
        recorded_onset: Any = None,
        reference_start: _dt.date | None = None,
    ) -> OnsetResolution:
        """Determine onset, preferring the structured record where it exists.

        Order of preference:

        1. a recorded onset date in the AE table — the study's own answer
        2. an absolute date written in the narrative
        3. a study day, resolved against the subject's reference start date
        4. a relative expression, resolved against an anchor from the EX domain

        Where a relative expression cannot be tied to an anchor date, the offset
        is still reported and the date is left null.
        """
        window = text if scope is None else scope.text
        window_offset = 0 if scope is None else scope.start
        mentions = [
            TemporalMention(
                m.pattern_id, m.surface, m.start + window_offset, m.end + window_offset,
                m.offset_days, m.study_day, m.absolute_date, m.anchor_phrase,
                m.anchor_event, m.vague,
            )
            for m in self.find_mentions(window)
        ]
        mention = mentions[0] if mentions else None

        recorded = parse_date(recorded_onset)
        anchor_event = (
            (mention.anchor_event if mention else None) or default_anchor
        )
        anchor_hit = self._anchor_hit(subject_id, anchor_event, index_rule, recorded)
        anchor_date = anchor_hit.date if anchor_hit else None

        if recorded is not None:
            offset = (recorded - anchor_date).days if anchor_date else None
            return OnsetResolution(
                onset_date=recorded,
                onset_offset_days=offset,
                anchor_event=anchor_event if anchor_date else None,
                anchor_date=anchor_date,
                source="structured_onset_date",
                confidence=self.config.confidence_for("temporal_structured", 0.98),
                mention=mention,
                detail=(
                    f"onset date from the AE record ({recorded.isoformat()})"
                    + (
                        f", {offset} days from {anchor_event} on {anchor_date.isoformat()}"
                        if anchor_date else ", no anchor date resolved"
                    )
                ),
            )

        if mention is None:
            return OnsetResolution(
                anchor_event=None, anchor_date=anchor_date, mention=None
            )

        if mention.absolute_date is not None:
            offset = (mention.absolute_date - anchor_date).days if anchor_date else None
            return OnsetResolution(
                onset_date=mention.absolute_date,
                onset_offset_days=offset,
                anchor_event=anchor_event if anchor_date else None,
                anchor_date=anchor_date,
                source="narrative_absolute_date",
                confidence=self.config.confidence_for("temporal_explicit", 0.92),
                mention=mention,
                detail=f"absolute date in narrative: '{mention.surface}'",
            )

        if mention.study_day is not None:
            if reference_start is None:
                return OnsetResolution(
                    anchor_event=anchor_event, anchor_date=anchor_date, mention=mention,
                    source="unresolved_study_day",
                    detail=(
                        f"study day {mention.study_day} in narrative, but the "
                        f"subject has no reference start date to resolve it against"
                    ),
                )
            onset_date = reference_start + _dt.timedelta(days=mention.study_day - 1)
            offset = (onset_date - anchor_date).days if anchor_date else None
            return OnsetResolution(
                onset_date=onset_date,
                onset_offset_days=offset,
                anchor_event=anchor_event if anchor_date else None,
                anchor_date=anchor_date,
                source="narrative_study_day",
                confidence=self.config.confidence_for("temporal_explicit", 0.92),
                mention=mention,
                detail=(
                    f"study day {mention.study_day} resolved against RFSTDTC "
                    f"{reference_start.isoformat()}"
                ),
            )

        offset = mention.offset_days
        onset_date = anchor_date + _dt.timedelta(days=offset) if anchor_date and offset is not None else None
        confidence_key = "temporal_vague" if mention.vague else "temporal_explicit"
        return OnsetResolution(
            onset_date=onset_date,
            onset_offset_days=offset,
            anchor_event=anchor_event,
            anchor_date=anchor_date,
            source="narrative_relative_vague" if mention.vague else "narrative_relative",
            confidence=self.config.confidence_for(confidence_key, 0.9),
            mention=mention,
            detail=(
                f"relative expression '{mention.surface}'"
                + (
                    f" anchored to {anchor_event} on {anchor_date.isoformat()}"
                    if anchor_date
                    else " with no resolvable anchor date; offset reported, date left null"
                )
                + (" (vague quantifier, mapped by config)" if mention.vague else "")
            ),
        )

    def _anchor_hit(
        self,
        subject_id: str,
        anchor_event: str | None,
        index_rule: str,
        onset_date: _dt.date | None,
    ):
        if self.resolver is None or not anchor_event:
            return None
        return self.resolver.resolve(
            subject_id, anchor_event, index_rule=index_rule, onset_date=onset_date
        )


_MONTHS = {
    m: i + 1
    for i, m in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    )
}


def _parse_narrative_date(text: str) -> _dt.date | None:
    text = text.strip()
    iso = parse_date(text)
    if iso is not None:
        return iso
    match = re.match(r"(\d{1,2})[-/]([A-Za-z]{3})[-/](\d{4})", text)
    if match:
        month = _MONTHS.get(match.group(2).lower())
        if month:
            try:
                return _dt.date(int(match.group(3)), month, int(match.group(1)))
            except ValueError:
                return None
    return None
