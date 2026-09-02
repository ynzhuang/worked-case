"""Querying the index: a precise path and a discovery path.

**Precise.** Cohort membership is decided by normalized attributes and a
definition's verdict. No embedding is consulted — a cohort is a claim about
which patients meet a written rule, and similarity is not that claim.

**Discovery.** Free-text search over mentions, hybrid where a local embedding
model exists and lexical where it does not. Everything it returns is a
``candidate``: calling ``as_cohort()`` on a discovery result raises, because a
mention is a place in a document where something was named, not an adjudicated
event.

The discovery path also answers the question the earlier version could not:
*find modifiers that no catalogue value covers yet*. That is the honest job of
semantic search here — surfacing what the value space is missing, so a person
can decide whether to extend it.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from typing import Any, Literal, Sequence

from ..catalog import ConceptCatalog, ConfigError
from .index import EpisodeIndex

Mode = Literal["precise", "lexical", "dense", "hybrid"]


class CandidateInCohort(RuntimeError):
    """Raised when discovery output is used as a cohort."""


@dataclass
class RetrievedEpisode:
    episode_id: str
    study_id: str
    subject_id: str
    profile: str
    concept: str | None
    location: str | None
    location_method: str | None
    location_source: str | None
    location_confidence: float | None
    pattern: str | None
    severity: str | None
    onset_offset_days: int | None
    record_count: int
    linkage_rule: str
    linkage_review: bool
    verdict: str | None
    definition: str | None
    reported_terms: str
    candidate: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id, "study_id": self.study_id,
            "subject_id": self.subject_id, "profile": self.profile,
            "concept": self.concept, "location": self.location,
            "location_method": self.location_method,
            "location_source": self.location_source,
            "location_confidence": self.location_confidence,
            "pattern": self.pattern, "severity": self.severity,
            "onset_offset_days": self.onset_offset_days,
            "record_count": self.record_count, "linkage_rule": self.linkage_rule,
            "linkage_review": self.linkage_review, "verdict": self.verdict,
            "definition": self.definition, "reported_terms": self.reported_terms,
            "candidate": self.candidate,
        }


@dataclass
class RetrievalResult:
    episodes: list[RetrievedEpisode]
    mode: Mode
    filters: dict[str, Any]
    notes: list[str] = _dc_field(default_factory=list)

    def as_cohort(self) -> list[RetrievedEpisode]:
        if self.mode != "precise":
            raise CandidateInCohort(
                f"mode {self.mode!r} is a discovery path and returns candidates; "
                f"a cohort is built on the precise path or not at all"
            )
        candidates = [e for e in self.episodes if e.candidate]
        if candidates:
            raise CandidateInCohort(
                f"{len(candidates)} result(s) are unadjudicated candidates and "
                f"cannot enter a cohort without adjudication or a definition "
                f"version that claims them"
            )
        return self.episodes

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "count": len(self.episodes),
            "filters": self.filters,
            "usable_as_cohort": self.mode == "precise"
            and not any(e.candidate for e in self.episodes),
            "notes": self.notes,
            "episodes": [e.to_dict() for e in self.episodes],
        }


@dataclass
class DiscoveredMention:
    mention_id: str
    doc_id: str
    study_id: str
    subject_id: str
    source_record_id: str | None
    profile: str | None
    attribute: str
    value: str
    surface: str
    source_variable: str | None
    confidence: float | None
    normalized: bool
    sentence: str
    score: float = 0.0
    candidate: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "mention_id": self.mention_id, "doc_id": self.doc_id,
            "study_id": self.study_id, "subject_id": self.subject_id,
            "source_record_id": self.source_record_id, "profile": self.profile,
            "attribute": self.attribute, "value": self.value,
            "surface": self.surface, "source_variable": self.source_variable,
            "confidence": self.confidence, "normalized": self.normalized,
            "sentence": self.sentence, "score": round(self.score, 6),
            "candidate": self.candidate,
        }


@dataclass
class DiscoveryResult:
    mentions: list[DiscoveredMention]
    mode: Mode
    filters: dict[str, Any]
    dense_available: bool = False
    notes: list[str] = _dc_field(default_factory=list)

    def as_cohort(self):
        raise CandidateInCohort(
            "discovery returns candidate mentions. A mention is a place in a "
            "document where something is named, not an event that occurred; it "
            "enters a cohort through adjudication or a definition version that "
            "claims it, never directly."
        )

    @property
    def unnormalized(self) -> list[DiscoveredMention]:
        return [m for m in self.mentions if not m.normalized]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "count": len(self.mentions),
            "filters": self.filters,
            "usable_as_cohort": False,
            "all_candidates": all(m.candidate for m in self.mentions),
            "unnormalized_count": len(self.unnormalized),
            "dense_available": self.dense_available,
            "notes": self.notes,
            "cohort_note": (
                "A mention is a place in a document where something is named, "
                "not an event that occurred. These enter a cohort through "
                "adjudication or a definition version that claims them, never "
                "directly."
            ),
            "mentions": [m.to_dict() for m in self.mentions],
        }


# --------------------------------------------------------------------------


def _row_to_episode(row: Any, candidate: bool = False) -> RetrievedEpisode:
    definition = (
        f"{row['definition_id']}.v{row['definition_version']}"
        if row["definition_id"] else None
    )
    return RetrievedEpisode(
        episode_id=row["episode_id"], study_id=row["study_id"],
        subject_id=row["subject_id"], profile=row["profile"],
        concept=row["concept"], location=row["location"],
        location_method=row["location_method"],
        location_source=row["location_source"],
        location_confidence=row["location_confidence"],
        pattern=row["pattern"], severity=row["severity"],
        onset_offset_days=row["onset_offset_days"],
        record_count=row["record_count"], linkage_rule=row["linkage_rule"],
        linkage_review=bool(row["linkage_review"]), verdict=row["verdict"],
        definition=definition, reported_terms=row["reported_terms"] or "",
        candidate=bool(row["candidate"]) or candidate,
    )


def retrieve(
    index: EpisodeIndex,
    catalog: ConceptCatalog,
    *,
    concept: str | None = None,
    group: str | None = None,
    location: Sequence[str] | None = None,
    region: str | None = None,
    method: Sequence[str] | None = None,
    profile: Sequence[str] | None = None,
    studies: Sequence[str] | None = None,
    verdict: Sequence[str] | None = None,
    window: tuple[int, int] | None = None,
    linkage_review: bool | None = None,
    definition_id: str | None = None,
    definition_version: int | None = None,
    top_k: int = 50,
) -> RetrievalResult:
    """Cohort-eligible episodes, filtered on normalized attributes."""
    where: list[str] = []
    params: list[Any] = []
    notes: list[str] = []

    concepts: list[str] = []
    if group:
        concepts.extend(catalog.expand_group(group))
    if concept:
        if concept in catalog.concept_groups:
            concepts.extend(catalog.expand_group(concept))
        elif concept in catalog.concepts:
            concepts.append(concept)
        else:
            raise ConfigError(
                f"unknown concept {concept!r}; known: {sorted(catalog.concepts)}"
            )
    if concepts:
        where.append(f"concept IN ({','.join('?' * len(concepts))})")
        params.extend(sorted(set(concepts)))

    values = list(location or [])
    if region:
        values.extend(catalog.attribute("location").in_region(region))
        notes.append(
            f"region {region!r} expanded to its declared members; no hierarchy "
            f"was walked as though it implied membership"
        )
    if values:
        where.append(f"location IN ({','.join('?' * len(values))})")
        params.extend(sorted(set(values)))

    for column, selected in (
        ("location_method", method), ("profile", profile),
        ("study_id", studies), ("verdict", verdict),
    ):
        if selected:
            where.append(f"{column} IN ({','.join('?' * len(selected))})")
            params.extend(list(selected))

    if window is not None:
        where.append("onset_offset_days IS NOT NULL")
        where.append("onset_offset_days BETWEEN ? AND ?")
        params.extend([window[0], window[1]])
    if linkage_review is not None:
        where.append("linkage_review = ?")
        params.append(int(linkage_review))
    if definition_id:
        where.append("definition_id = ?")
        params.append(definition_id)
    if definition_version is not None:
        where.append("definition_version = ?")
        params.append(definition_version)

    clause = f" WHERE {' AND '.join(where)}" if where else ""
    rows = index.query(
        f"SELECT * FROM episodes{clause} ORDER BY episode_id LIMIT ?",
        params + [top_k],
    )
    return RetrievalResult(
        episodes=[_row_to_episode(row) for row in rows],
        mode="precise",
        filters={
            "concept": sorted(set(concepts)) or None, "location": values or None,
            "region": region, "method": list(method or []) or None,
            "profile": list(profile or []) or None,
            "studies": list(studies or []) or None,
            "verdict": list(verdict or []) or None, "window": window,
            "linkage_review": linkage_review,
            "definition": (
                f"{definition_id}.v{definition_version}" if definition_id else None
            ),
        },
        notes=notes,
    )


def discover(
    index: EpisodeIndex,
    catalog: ConceptCatalog,
    *,
    text: str | None = None,
    attribute: Sequence[str] | None = None,
    value: Sequence[str] | None = None,
    studies: Sequence[str] | None = None,
    profile: Sequence[str] | None = None,
    unnormalized_only: bool = False,
    mode: Mode = "lexical",
    top_k: int = 50,
) -> DiscoveryResult:
    """Search free text for modifier mentions. Everything returned is a candidate."""
    notes: list[str] = []
    dense_available = False
    if mode in ("dense", "hybrid"):
        from .dense import dense_backend_available

        dense_available = dense_backend_available()
        if not dense_available:
            notes.append(
                "no local embedding model is present, so dense discovery "
                "degraded to lexical; nothing about the result is silently "
                "different, and the candidate status is unaffected"
            )
            mode = "lexical"

    where: list[str] = []
    params: list[Any] = []
    if attribute:
        where.append(f"m.attribute IN ({','.join('?' * len(attribute))})")
        params.extend(list(attribute))
    if value:
        where.append(f"m.value IN ({','.join('?' * len(value))})")
        params.extend(list(value))
    if studies:
        where.append(f"m.study_id IN ({','.join('?' * len(studies))})")
        params.extend(list(studies))
    if profile:
        where.append(f"m.profile IN ({','.join('?' * len(profile))})")
        params.extend(list(profile))
    if unnormalized_only:
        where.append("m.normalized = 0")
        notes.append(
            "restricted to mentions no catalogue value covers: this is the "
            "question the value space cannot answer yet, which is what a "
            "person needs to see in order to extend it"
        )

    if text:
        clause = " AND " + " AND ".join(where) if where else ""
        rows = index.query(
            "SELECT m.*, bm25(text_index) AS score FROM text_index "
            "JOIN mentions m ON m.mention_id = text_index.mention_id "
            f"WHERE text_index MATCH ?{clause} ORDER BY score LIMIT ?",
            [text] + params + [top_k],
        )
    else:
        clause = f" WHERE {' AND '.join(where)}" if where else ""
        rows = index.query(
            f"SELECT m.*, 0.0 AS score FROM mentions m{clause} "
            f"ORDER BY m.mention_id LIMIT ?",
            params + [top_k],
        )

    mentions = [
        DiscoveredMention(
            mention_id=row["mention_id"], doc_id=row["doc_id"],
            study_id=row["study_id"], subject_id=row["subject_id"],
            source_record_id=row["source_record_id"], profile=row["profile"],
            attribute=row["attribute"], value=row["value"], surface=row["surface"],
            source_variable=row["source_variable"], confidence=row["confidence"],
            normalized=bool(row["normalized"]), sentence=row["sentence"],
            score=float(row["score"] or 0.0),
        )
        for row in rows
    ]
    return DiscoveryResult(
        mentions=mentions, mode=mode,
        filters={
            "text": text, "attribute": list(attribute or []) or None,
            "value": list(value or []) or None,
            "studies": list(studies or []) or None,
            "profile": list(profile or []) or None,
            "unnormalized_only": unnormalized_only,
        },
        dense_available=dense_available, notes=notes,
    )
