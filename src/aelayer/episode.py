"""Episode grouping — deliberately demoted.

An evolving event may be written as several source records under one collection
convention and as a single record under another. Grouping them lets an analysis
talk about the clinical event rather than the paperwork.

In this version that grouping is **secondary**, and the demotion is the design
decision, not an omission:

* Nothing evaluates a phenotype at the episode grain. Verdicts, denominators,
  the silver standard and the ablation all run on source records, because the
  source record is the thing the study actually collected and the only grain
  every claim can be traced back to.
* No attribute is promoted from a record onto an episode. Promotion means
  choosing between two records that disagree, and that choice would sit
  underneath every downstream number while being invisible in it.
* An episode is therefore a **grouping with a stated rule and a confidence**,
  and nothing more. It is offered as a view; anything it cannot settle it
  reports rather than resolves.

The earlier version of this layer put episodes at the centre and derived cases
from them. That made a declared linkage assumption — one that is simply wrong
for recurrent conditions — load-bearing for every rate the system produced. The
records are the grain now, and this module sits above them.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field as _dc_field
from typing import Any, Iterable, Sequence

from .models import CanonicalAERecord

#: How the grouping rules are named in output. `single_record` is by far the
#: most common and is not a fallback: most studies write one record per event.
LINKAGE_RULES = (
    "single_record",
    "explicit_continuation",
    "gap_within_tolerance",
    "flagged_for_review",
)


@dataclass
class Episode:
    """A group of source records believed to describe one clinical event.

    Carries no attributes of its own. To read a value, read it off one of the
    records — where they disagree, that disagreement is a finding, not
    something for this class to average away.
    """

    episode_id: str
    subject_id: str
    study_id: str
    concept_id: str | None
    record_ids: list[str] = _dc_field(default_factory=list)
    rule: str = "single_record"
    confidence: float = 1.0
    review_required: bool = False
    note: str = ""

    @property
    def size(self) -> int:
        return len(self.record_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "subject_id": self.subject_id,
            "study_id": self.study_id,
            "concept_id": self.concept_id,
            "record_ids": list(self.record_ids),
            "n_records": self.size,
            "rule": self.rule,
            "confidence": self.confidence,
            "review_required": self.review_required,
            "note": self.note,
        }


@dataclass
class EpisodeView:
    """Every episode over one snapshot, plus what could not be settled."""

    episodes: list[Episode] = _dc_field(default_factory=list)
    note: str = (
        "Episodes are a derived view. Source records are unmodified beneath "
        "them, no attribute is promoted onto an episode, and no phenotype is "
        "evaluated at this grain."
    )

    def rules(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for episode in self.episodes:
            counts[episode.rule] = counts.get(episode.rule, 0) + 1
        return dict(sorted(counts.items()))

    def flagged(self) -> list[Episode]:
        return [e for e in self.episodes if e.review_required]

    def for_record(self, record_id: str) -> Episode | None:
        return next(
            (e for e in self.episodes if record_id in e.record_ids), None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_episodes": len(self.episodes),
            "n_records": sum(e.size for e in self.episodes),
            "rules": self.rules(),
            "n_flagged": len(self.flagged()),
            "note": self.note,
        }


def _onset(record: CanonicalAERecord) -> _dt.date | None:
    return record.onset.value if record.onset.observed else None


def _end(record: CanonicalAERecord) -> _dt.date | None:
    return record.end.value if record.end.observed else None


def group_records(
    records: Iterable[CanonicalAERecord], *, gap_tolerance_days: int = 3
) -> EpisodeView:
    """Group records into episodes under one declared rule.

    Same subject, same study, same concept, and intervals that touch or sit
    within the tolerance. Records whose dates cannot be read are never merged
    on a guess: they stand alone and say why.
    """
    view = EpisodeView()
    buckets: dict[tuple[str, str, str | None], list[CanonicalAERecord]] = {}
    for record in records:
        key = (record.study_id, record.subject_id, record.concept_id)
        buckets.setdefault(key, []).append(record)

    for (study_id, subject_id, concept_id), group in sorted(buckets.items()):
        group.sort(key=lambda r: (_onset(r) or _dt.date.max, r.record_id))
        current: list[CanonicalAERecord] = []
        rule = "single_record"
        review = False
        note = ""

        def flush() -> None:
            nonlocal current, rule, review, note
            if not current:
                return
            view.episodes.append(Episode(
                episode_id=(
                    f"{study_id}::{subject_id}::{concept_id or 'UNCODED'}::"
                    f"{len(view.episodes) + 1:02d}"
                ),
                subject_id=subject_id,
                study_id=study_id,
                concept_id=concept_id,
                record_ids=[r.record_id for r in current],
                rule=("single_record" if len(current) == 1 else rule),
                confidence=(1.0 if len(current) == 1 else 0.7),
                review_required=review and len(current) > 1,
                note=note,
            ))
            current, rule, review, note = [], "single_record", False, ""

        for record in group:
            if not current:
                current = [record]
                continue
            previous = current[-1]
            previous_end = _end(previous) or _onset(previous)
            this_onset = _onset(record)
            if previous_end is None or this_onset is None:
                # One of the two has no readable date. Merging would be a
                # guess, so they stay apart and the reason is on the record.
                flush()
                current = [record]
                note = (
                    "not merged with the preceding record: one of the two has "
                    "no readable onset or end date, and merging on a guess "
                    "would put an assumption underneath every later number"
                )
                continue
            gap = (this_onset - previous_end).days
            if gap <= gap_tolerance_days:
                current.append(record)
                rule = "gap_within_tolerance"
                review = gap > 0
                note = (
                    f"records {gap} day(s) apart, within the declared tolerance "
                    f"of {gap_tolerance_days}; a recurrent condition would need "
                    f"a different rule and this one is declared, not inferred"
                )
            else:
                flush()
                current = [record]
        flush()
    return view


def episode_table(view: EpisodeView) -> list[dict[str, Any]]:
    return [episode.to_dict() for episode in view.episodes]


def reconcile_records(
    records: Sequence[CanonicalAERecord], *, gap_tolerance_days: int = 3
) -> EpisodeView:
    """Alias kept for callers that read as "reconcile"; the same grouping."""
    return group_records(records, gap_tolerance_days=gap_tolerance_days)
