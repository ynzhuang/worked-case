"""Per-subject trajectories.

Deliberately minimal. This is not disease-progression modelling; it is the
ordered structure the rest of the system already needs — exposures and episodes
on one timeline, each with its offset from the anchor — so that a window can be
applied and "distribution by time since exposure" can be answered without every
component re-deriving the ordering for itself.

Building it here rather than inside the phenotype evaluator means the evaluator
and the agent measure time the same way, from the same records.
"""

from __future__ import annotations

import collections
import datetime as _dt
from typing import Any, Iterable, Sequence

from .anchors import AnchorResolver, parse_date
from .ingest import TrialStore
from .models import CanonicalAEEpisode, Trajectory, TrajectoryEvent


def build_trajectories(
    episodes: Iterable[CanonicalAEEpisode],
    store: TrialStore,
    resolver: AnchorResolver,
    anchor_event: str = "first_exposure",
) -> dict[str, Trajectory]:
    """One trajectory per subject who has any record at all.

    A subject with exposures and no episodes still gets a trajectory: "nothing
    happened to this subject" is a fact a denominator needs, and dropping them
    would silently shrink every rate computed from it.
    """
    by_subject: dict[str, list[CanonicalAEEpisode]] = collections.defaultdict(list)
    for episode in episodes:
        by_subject[episode.subject_id].append(episode)

    trajectories: dict[str, Trajectory] = {}
    for subject_id in store.subjects():
        study_id = store.study_of(subject_id) or ""
        hit = resolver.resolve(subject_id, anchor_event)
        anchor_date = hit.date if hit else None

        events: list[TrajectoryEvent] = []
        for row in store.subject_rows(subject_id, "ex"):
            when = parse_date(row.get("EXSTDTC"))
            if when is None:
                continue
            events.append(TrajectoryEvent(
                kind="exposure",
                identifier=f"EX:{subject_id}:{row.get('EXSEQ')}",
                date=when,
                label=str(row.get("EXTRT") or "exposure"),
                detail={"dose": row.get("EXDOSE"), "unit": row.get("EXDOSU")},
                offset_days=(when - anchor_date).days if anchor_date else None,
            ))
        for episode in sorted(by_subject.get(subject_id, []),
                              key=lambda e: e.episode_id):
            when = episode.episode_start.value
            if when is None:
                continue
            events.append(TrajectoryEvent(
                kind="episode",
                identifier=episode.episode_id,
                date=when,
                label=episode.standardized_concept or "unmapped",
                detail={
                    "location": episode.location.value,
                    "severity": episode.severity.value,
                    "records": len(episode.source_record_ids),
                },
                offset_days=(
                    episode.onset_offset_days.value
                    if episode.onset_offset_days.populated
                    else ((when - anchor_date).days if anchor_date else None)
                ),
            ))

        events.sort(key=lambda e: (e.date, e.kind, e.identifier))
        trajectories[subject_id] = Trajectory(
            subject_id=subject_id,
            study_id=study_id,
            profile=store.profile_of(study_id) or "",
            anchor_event=anchor_event if anchor_date else None,
            anchor_date=anchor_date,
            events=events,
        )
    return trajectories


def time_since_exposure(
    trajectories: Sequence[Trajectory], concept: str | None = None,
    bins: Sequence[int] = (0, 7, 14, 30, 90),
) -> dict[str, Any]:
    """Distribution of episode onsets by time since the anchor.

    Episodes with no resolvable offset are counted in their own bucket rather
    than dropped: an unknown offset is a fact about the data, and folding it
    into "over 90 days" would be an invention.
    """
    edges = list(bins)
    counts: dict[str, int] = {f"{a}-{b}": 0 for a, b in zip(edges, edges[1:])}
    counts[f">{edges[-1]}"] = 0
    counts["unresolved"] = 0

    for trajectory in trajectories:
        for event in trajectory.episodes():
            if concept and event.label != concept:
                continue
            offset = event.offset_days
            if offset is None:
                counts["unresolved"] += 1
                continue
            for low, high in zip(edges, edges[1:]):
                if low <= offset < high:
                    counts[f"{low}-{high}"] += 1
                    break
            else:
                counts[f">{edges[-1]}"] += 1
    return counts
