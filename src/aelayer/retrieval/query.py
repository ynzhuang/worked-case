"""Querying the index: a precise path and a discovery path.

**Precise.** Cohort membership is decided by normalized attributes and a
definition's verdict. No embedding is consulted — a cohort is a claim about
which patients meet a written rule, and similarity is not that claim.

**Discovery.** Free-text search over mentions, hybrid where a local embedding
model exists and lexical where it does not. Everything it returns is a
``candidate``: calling ``as_cohort()`` on a discovery result raises, because a
mention is a place in a document where something was named, not an adjudicated
event.

The precise path filters on the two fields separately. ``assertion="absent"``
and ``availability="not_collected"`` select different subjects, and a query
surface that cannot express the difference would push a caller into treating
them as one — which is the error the whole model exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from typing import Any, Literal, Sequence

from ..catalog import ConceptCatalog, ConfigError
from .index import RecordIndex

Mode = Literal["precise", "lexical", "dense", "hybrid"]


class CandidateInCohort(RuntimeError):
    """Raised when discovery output is used as a cohort."""


@dataclass
class RetrievedRecord:
    record_id: str
    study_id: str
    subject_id: str
    profile: str
    concept: str | None
    code: str | None
    dictionary_version: str | None
    reconciliation: str | None
    modifier: str | None
    assertion: str | None
    availability: str
    value: str | None
    method: str | None
    source_variable: str | None
    confidence: float | None
    severity: str | None
    grade: int | None
    exposure_offset_days: int | None
    verdict: str | None
    definition: str | None
    reported_term: str
    candidate: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id, "study_id": self.study_id,
            "subject_id": self.subject_id, "profile": self.profile,
            "concept": self.concept, "code": self.code,
            "dictionary_version": self.dictionary_version,
            "reconciliation": self.reconciliation,
            "modifier": self.modifier, "assertion": self.assertion,
            "availability": self.availability, "value": self.value,
            "method": self.method, "source_variable": self.source_variable,
            "confidence": self.confidence, "severity": self.severity,
            "grade": self.grade,
            "exposure_offset_days": self.exposure_offset_days,
            "verdict": self.verdict, "definition": self.definition,
            "reported_term": self.reported_term, "candidate": self.candidate,
        }


@dataclass
class RetrievalResult:
    records: list[RetrievedRecord]
    mode: Mode
    filters: dict[str, Any]
    notes: list[str] = _dc_field(default_factory=list)

    def as_cohort(self) -> list[RetrievedRecord]:
        if self.mode != "precise":
            raise CandidateInCohort(
                f"mode {self.mode!r} is a discovery path and returns candidates; "
                f"a cohort is built on the precise path or not at all"
            )
        candidates = [r for r in self.records if r.candidate]
        if candidates:
            raise CandidateInCohort(
                f"{len(candidates)} result(s) are unadjudicated candidates and "
                f"cannot enter a cohort without adjudication or a definition "
                f"version that claims them"
            )
        return self.records

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "count": len(self.records),
            "filters": self.filters,
            "usable_as_cohort": self.mode == "precise"
            and not any(r.candidate for r in self.records),
            "notes": self.notes,
            "records": [r.to_dict() for r in self.records],
        }


@dataclass
class DiscoveredMention:
    mention_id: str
    doc_id: str
    study_id: str
    subject_id: str
    source_record_id: str | None
    profile: str | None
    modifier: str
    assertion: str
    value: str | None
    surface: str
    source_variable: str | None
    confidence: float | None
    sentence: str
    score: float = 0.0
    candidate: bool = True

    @property
    def normalized(self) -> bool:
        """Whether the mention resolved to a declared catalogue value.

        An assertion with no value is still normalized — "mucosal involvement"
        asserts the modifier without naming a site, and that is a complete
        answer, not a partial one.
        """
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "mention_id": self.mention_id, "doc_id": self.doc_id,
            "study_id": self.study_id, "subject_id": self.subject_id,
            "source_record_id": self.source_record_id, "profile": self.profile,
            "modifier": self.modifier, "assertion": self.assertion,
            "value": self.value, "surface": self.surface,
            "source_variable": self.source_variable,
            "confidence": self.confidence, "sentence": self.sentence,
            "score": round(self.score, 6), "candidate": self.candidate,
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "count": len(self.mentions),
            "filters": self.filters,
            "usable_as_cohort": False,
            "all_candidates": all(m.candidate for m in self.mentions),
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


def _row_to_record(row: Any, candidate: bool = False) -> RetrievedRecord:
    definition = (
        f"{row['definition_id']}.v{row['definition_version']}"
        if row["definition_id"] else None
    )
    return RetrievedRecord(
        record_id=row["record_id"], study_id=row["study_id"],
        subject_id=row["subject_id"], profile=row["profile"],
        concept=row["concept"], code=row["code"],
        dictionary_version=row["dictionary_version"],
        reconciliation=row["reconciliation"], modifier=row["modifier"],
        assertion=row["modifier_assertion"],
        availability=row["modifier_availability"],
        value=row["modifier_value"], method=row["modifier_method"],
        source_variable=row["modifier_source"],
        confidence=row["modifier_confidence"], severity=row["severity"],
        grade=row["grade"], exposure_offset_days=row["exposure_offset_days"],
        verdict=row["verdict"], definition=definition,
        reported_term=row["reported_term"] or "",
        candidate=bool(row["candidate"]) or candidate,
    )


def retrieve(
    index: RecordIndex,
    catalog: ConceptCatalog,
    *,
    concept: str | None = None,
    group: str | None = None,
    assertion: Sequence[str] | None = None,
    availability: Sequence[str] | None = None,
    value: Sequence[str] | None = None,
    method: Sequence[str] | None = None,
    profile: Sequence[str] | None = None,
    studies: Sequence[str] | None = None,
    verdict: Sequence[str] | None = None,
    reconciliation: Sequence[str] | None = None,
    window: tuple[int, int] | None = None,
    definition_id: str | None = None,
    definition_version: int | None = None,
    top_k: int = 50,
) -> RetrievalResult:
    """Cohort-eligible records, filtered on normalized values."""
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

    if assertion and availability:
        notes.append(
            "assertion and availability were filtered separately, which is the "
            "only way to ask this question correctly: a documented 'absent' and "
            "a variable nobody collected are different populations"
        )
    for column, selected in (
        ("modifier_assertion", assertion),
        ("modifier_availability", availability),
        ("modifier_value", value),
        ("modifier_method", method),
        ("profile", profile),
        ("study_id", studies),
        ("verdict", verdict),
        ("reconciliation", reconciliation),
    ):
        if selected:
            where.append(f"{column} IN ({','.join('?' * len(selected))})")
            params.extend(list(selected))

    if window is not None:
        where.append("exposure_offset_days IS NOT NULL")
        where.append("exposure_offset_days BETWEEN ? AND ?")
        params.extend([window[0], window[1]])
    if definition_id:
        where.append("definition_id = ?")
        params.append(definition_id)
    if definition_version is not None:
        where.append("definition_version = ?")
        params.append(definition_version)

    clause = f" WHERE {' AND '.join(where)}" if where else ""
    rows = index.query(
        f"SELECT * FROM records{clause} ORDER BY record_id LIMIT ?",
        params + [top_k],
    )
    return RetrievalResult(
        records=[_row_to_record(row) for row in rows],
        mode="precise",
        filters={
            "concept": sorted(set(concepts)) or None,
            "assertion": list(assertion or []) or None,
            "availability": list(availability or []) or None,
            "value": list(value or []) or None,
            "method": list(method or []) or None,
            "profile": list(profile or []) or None,
            "studies": list(studies or []) or None,
            "verdict": list(verdict or []) or None,
            "reconciliation": list(reconciliation or []) or None,
            "window": window,
            "definition": (
                f"{definition_id}.v{definition_version}" if definition_id else None
            ),
        },
        notes=notes,
    )


def discover(
    index: RecordIndex,
    catalog: ConceptCatalog,
    *,
    text: str | None = None,
    modifier: Sequence[str] | None = None,
    assertion: Sequence[str] | None = None,
    value: Sequence[str] | None = None,
    studies: Sequence[str] | None = None,
    profile: Sequence[str] | None = None,
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
    for column, selected in (
        ("m.modifier", modifier), ("m.assertion", assertion),
        ("m.value", value), ("m.study_id", studies), ("m.profile", profile),
    ):
        if selected:
            where.append(f"{column} IN ({','.join('?' * len(selected))})")
            params.extend(list(selected))

    if text:
        clause = " AND " + " AND ".join(where) if where else ""
        rows = index.query(
            "SELECT m.*, bm25(text_index) AS score FROM text_index "
            "JOIN mentions m ON m.doc_id = text_index.doc_id "
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
            modifier=row["modifier"], assertion=row["assertion"],
            value=row["value"], surface=row["surface"],
            source_variable=row["source_variable"], confidence=row["confidence"],
            sentence=row["sentence"], score=float(row["score"] or 0.0),
        )
        for row in rows
    ]
    return DiscoveryResult(
        mentions=mentions, mode=mode,
        filters={
            "text": text, "modifier": list(modifier or []) or None,
            "assertion": list(assertion or []) or None,
            "value": list(value or []) or None,
            "studies": list(studies or []) or None,
            "profile": list(profile or []) or None,
        },
        dense_available=dense_available, notes=notes,
    )
