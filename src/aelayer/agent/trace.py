"""Traceability.

A reported number is only as good as the chain behind it.  ``trace`` walks that
chain end to end:

    number -> analysis run -> cohort -> phenotype definition and version ->
    contributing episodes -> source records -> text spans

**This replaces the approval gate, deliberately.**  Approving a specification
you cannot independently evaluate is ceremony: the reviewer sees a plan, not a
result, and clicking approve does not make the result checkable.  A number you
can follow back to the sentence a site wrote is checkable, by someone who was
not in the room when it was compiled.

A number that cannot be traced end to end is a failing test, not a caveat.
"""

from __future__ import annotations

from typing import Any, Sequence

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
    """Build the full chain behind one reported number.

    ``complete`` is false, with ``broken_at`` naming the level, the moment any
    hop cannot be made. It is not softened into a warning: the guarantee is
    that a number either traces to source text or is reported as untraceable.
    """
    links: list[TraceLink] = [
        TraceLink(
            level="number",
            identifier=label,
            detail=f"{label} = {number}",
            payload={"value": number, "verdict": verdict},
        )
    ]

    links.append(
        TraceLink(
            level="analysis",
            identifier=manifest.manifest_id,
            detail=(
                f"{manifest.analysis_method} run at {manifest.created_at} by "
                f"{manifest.actor}"
            ),
            payload={
                "results_hash": manifest.results_hash,
                "output_pointer": manifest.output_pointer,
                "normalizer_version": manifest.normalizer_version,
                "extractor_version": manifest.extractor_version,
                "model_version": manifest.model_version,
                "snapshot_id": manifest.data_snapshot_id,
                "terminology_versions": manifest.terminology_versions,
            },
        )
    )

    contributing = [a for a in assignments if a.verdict == verdict]
    links.append(
        TraceLink(
            level="cohort",
            identifier="|".join(manifest.specification.get("studies", [])) or "all",
            detail=(
                f"{len(contributing)} episode(s) with verdict {verdict!r} across "
                f"{len(manifest.specification.get('studies', []))} study/studies"
            ),
            payload={
                "cohort_specification": manifest.cohort_specification,
                "episode_ids": [a.episode_id for a in contributing][:50],
            },
        )
    )

    links.append(
        TraceLink(
            level="definition",
            identifier=(
                f"{manifest.phenotype_definition_id}."
                f"v{manifest.phenotype_definition_version}"
            ),
            detail=(
                f"status {manifest.definition_status}, content hash "
                f"{manifest.definition_hash}"
            ),
            payload={"definition_hash": manifest.definition_hash},
        )
    )

    if not contributing:
        return Trace(
            number=number, label=label, complete=False, broken_at="episode",
            links=links,
        )

    episode_index = {e.episode_id: e for e in episodes}
    record_index = {r.source_record_id: r for r in records}
    broken_at: str | None = None

    for assignment in contributing[:max_examples]:
        episode = episode_index.get(assignment.episode_id)
        if episode is None:
            broken_at = "episode"
            break
        links.append(
            TraceLink(
                level="episode",
                identifier=episode.episode_id,
                detail=(
                    f"{assignment.reason[:200]} "
                    f"(linkage {episode.linkage_rule}, confidence "
                    f"{episode.linkage_confidence:.2f})"
                ),
                payload={
                    "subject_id": episode.subject_id,
                    "study_id": episode.study_id,
                    "matched_rule_id": assignment.matched_rule_id,
                    "evidence_state": assignment.evidence_state,
                    "source_record_ids": episode.source_record_ids,
                    "linkage_review_required": episode.linkage_review_required,
                },
            )
        )
        for record_id in episode.source_record_ids:
            record = record_index.get(record_id)
            if record is None:
                broken_at = "record"
                break
            links.append(
                TraceLink(
                    level="record",
                    identifier=record.source_record_id,
                    detail=(
                        f"{record.source_form_id} record in {record.study_id}, "
                        f"normalized by {record.normalizer_version}"
                        + (
                            f", enriched by {record.extractor_version}"
                            if record.extractor_version else ""
                        )
                    ),
                    payload={
                        "verbatim_term": record.verbatim_term.value,
                        "coded_term": record.coded_term.value,
                        "dictionary_version": record.dictionary_version,
                        "concept_source": record.concept_source,
                        "collection_states": record.collection_states(),
                    },
                )
            )
            spans = sorted(
                record.evidence, key=lambda s: (s.field, s.doc_id, s.start)
            )
            if not spans:
                broken_at = "span"
                break
            for span in spans[:6]:
                links.append(
                    TraceLink(
                        level="span",
                        identifier=f"{span.doc_id}:{span.start}-{span.end}",
                        detail=f"{span.field} = {span.extracted_value!r}",
                        payload={
                            "doc_id": span.doc_id, "field": span.field,
                            "start": span.start, "end": span.end,
                            "text": span.text[:200], "kind": span.kind,
                        },
                    )
                )
        if broken_at:
            break

    levels = {link.level for link in links}
    required = {"number", "analysis", "cohort", "definition", "episode", "record", "span"}
    missing = sorted(required - levels)
    if missing and not broken_at:
        broken_at = missing[0]

    return Trace(
        number=number,
        label=label,
        complete=broken_at is None,
        broken_at=broken_at,
        links=links,
    )


def render_trace(trace: Trace) -> str:
    """The chain as an indented outline, for a terminal."""
    indent = {
        "number": 0, "analysis": 1, "cohort": 2, "definition": 3,
        "episode": 4, "record": 5, "span": 6,
    }
    lines = []
    for link in trace.links:
        pad = "  " * indent.get(link.level, 0)
        lines.append(f"{pad}{link.level:<10} {link.identifier}")
        if link.detail:
            lines.append(f"{pad}           {link.detail}")
    if not trace.complete:
        lines.append("")
        lines.append(
            f"  INCOMPLETE: the chain breaks at {trace.broken_at!r}. A number "
            f"that cannot be traced to source text is not reportable."
        )
    return "\n".join(lines)
