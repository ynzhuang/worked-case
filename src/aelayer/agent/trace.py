"""Traceability.

A reported number is only as good as the chain behind it, so ``trace_number``
walks that chain end to end:

    number -> analysis run -> cohort -> definition version -> episodes ->
    source records -> spans

Every hop names the route the attribute came by, because in this layer "where
did this number come from" and "which variable was it read from" are the same
question.

A number that cannot be traced end to end is a failing test, not a caveat:
``complete`` is false and ``broken_at`` names the level where the chain stops.
"""

from __future__ import annotations

from typing import Sequence

from ..models import (
    CanonicalAEEpisode,
    CanonicalAERecord,
    CaseAssignment,
    Manifest,
    Trace,
    TraceLink,
)


def trace_number(
    *,
    number: float | int,
    label: str,
    manifest: Manifest,
    assignments: Sequence[CaseAssignment],
    episodes: Sequence[CanonicalAEEpisode],
    records: Sequence[CanonicalAERecord],
    verdict: str = "case",
    max_examples: int = 3,
) -> Trace:
    links: list[TraceLink] = [
        TraceLink(
            level="number", identifier=label, detail=f"{label} = {number}",
            payload={"value": number, "verdict": verdict},
        ),
        TraceLink(
            level="analysis", identifier=manifest.manifest_id,
            detail=(
                f"{manifest.analysis_method} run at {manifest.created_at} by "
                f"{manifest.actor}; normalizer {manifest.normalizer_version}, "
                f"extractor {manifest.extractor_version}, snapshot "
                f"{manifest.data_snapshot_id}"
            ),
            payload={
                "results_hash": manifest.results_hash,
                "output_pointer": manifest.output_pointer,
                "attribute_methods": manifest.attribute_methods,
            },
        ),
    ]

    matching = [a for a in assignments if a.verdict == verdict]
    studies = sorted({a.study_id for a in matching})
    links.append(TraceLink(
        level="cohort", identifier="|".join(studies) or "(empty)",
        detail=(
            f"{len(matching)} episode(s) with verdict {verdict!r} across "
            f"{len(studies)} study/studies"
        ),
        payload={"episode_ids": [a.episode_id for a in matching[:max_examples]]},
    ))
    links.append(TraceLink(
        level="definition",
        identifier=(
            f"{manifest.phenotype_definition_id}."
            f"v{manifest.phenotype_definition_version}"
        ),
        detail=(
            f"status {manifest.definition_status}, content hash "
            f"{manifest.definition_hash}"
        ),
    ))

    if not matching:
        return Trace(
            number=number, label=label, complete=False, broken_at="episode",
            links=links,
        )

    by_id = {e.episode_id: e for e in episodes}
    records_by_id = {r.source_record_id: r for r in records}
    broken_at: str | None = None

    for assignment in matching[:max_examples]:
        episode = by_id.get(assignment.episode_id)
        if episode is None:
            broken_at = "episode"
            break
        satisfied = [f for f in assignment.findings if f.satisfied]
        links.append(TraceLink(
            level="episode", identifier=episode.episode_id,
            detail="; ".join(f.reason for f in satisfied) or assignment.reason,
            payload={
                "attribute_sources": assignment.attribute_sources,
                "attribute_methods": assignment.attribute_methods,
                "linkage_rule": episode.linkage_rule,
            },
        ))

        for record_id in episode.source_record_ids:
            record = records_by_id.get(record_id)
            if record is None:
                broken_at = "record"
                break
            links.append(TraceLink(
                level="record", identifier=record_id,
                detail=(
                    f"{record.source_form_id} record in {record.study_id} "
                    f"({record.profile}), normalized by "
                    f"{record.normalizer_version}"
                    + (f", enriched by {record.extractor_version}"
                       if record.extractor_version else "")
                ),
            ))
            spans = [
                span for attribute in record.attributes().values()
                for span in attribute.evidence
            ]
            if not spans:
                broken_at = "span"
                break
            for span in sorted(spans, key=lambda s: (s.field, s.start))[:6]:
                links.append(TraceLink(
                    level="span", identifier=f"{span.doc_id}:{span.start}-{span.end}",
                    detail=f"{span.field} = {span.extracted_value!r}",
                    payload={"text": span.text, "kind": span.kind},
                ))
        if broken_at:
            break

    return Trace(
        number=number, label=label, complete=broken_at is None,
        broken_at=broken_at, links=links,
    )


def render_trace(trace: Trace) -> str:
    indent = {
        "number": 0, "analysis": 1, "cohort": 2, "definition": 3,
        "episode": 4, "record": 5, "span": 6,
    }
    lines: list[str] = []
    for link in trace.links:
        pad = "  " * indent[link.level]
        lines.append(f"{pad}{link.level:<10} {link.identifier}")
        if link.detail:
            lines.append(f"{pad}{'':<10} {link.detail}")
    if not trace.complete:
        lines.append("")
        lines.append(
            f"INCOMPLETE: the chain breaks at {trace.broken_at!r}. A number that "
            f"cannot be traced to source is not reportable."
        )
    return "\n".join(lines)
