"""Compositional retrieval.

The query interface mirrors how a scientific question decomposes: a concept, an
assertion, a window relative to an anchor, a set of studies, a set of evidence
states.  Each of those is a structured predicate.

Concept expansion uses the catalogue's synonyms and coded terms.  It does not
walk a MedDRA hierarchy as though it were a subsumption ontology; grouping above
the term level is an explicit, named list in config.

Assertion filtering is always a structured predicate, never a similarity
heuristic — including in dense and hybrid mode.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from ..catalog import ConceptCatalog
from ..models import EventObject
from .index import EventIndex

Mode = Literal["lexical", "dense", "hybrid"]


@dataclass
class RetrievedRecord:
    event_id: str
    doc_id: str
    subject_id: str
    study_id: str
    concept_id: str
    assertion: str
    evidence_state: str | None
    verdict: str | None
    onset_offset_days: int | None
    severity: str | None
    seriousness: list[str]
    action_taken: str | None
    coded_term: str | None
    score: float
    snippet: str
    matched_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "doc_id": self.doc_id,
            "subject_id": self.subject_id,
            "study_id": self.study_id,
            "concept_id": self.concept_id,
            "assertion": self.assertion,
            "evidence_state": self.evidence_state,
            "verdict": self.verdict,
            "onset_offset_days": self.onset_offset_days,
            "severity": self.severity,
            "seriousness": self.seriousness,
            "action_taken": self.action_taken,
            "coded_term": self.coded_term,
            "score": round(self.score, 6),
            "snippet": self.snippet,
            "matched_terms": self.matched_terms,
        }


@dataclass
class RetrievalResult:
    records: list[RetrievedRecord]
    expanded_terms: list[str]
    mode: Mode
    filters: dict[str, Any]
    total_before_filters: int
    dense_available: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def negation_false_positives(self) -> int:
        """Returned records that document an absence of the concept."""
        return sum(1 for r in self.records if r.assertion == "absent")

    @property
    def negation_false_positive_rate(self) -> float:
        if not self.records:
            return 0.0
        return self.negation_false_positives / len(self.records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "filters": self.filters,
            "expanded_terms": self.expanded_terms,
            "count": len(self.records),
            "total_before_filters": self.total_before_filters,
            "negation_false_positives": self.negation_false_positives,
            "negation_false_positive_rate": round(
                self.negation_false_positive_rate, 6
            ),
            "dense_available": self.dense_available,
            "notes": self.notes,
            "records": [r.to_dict() for r in self.records],
        }


def expand_concept(
    catalog: ConceptCatalog, concept: str | None, group: str | None = None
) -> tuple[list[str], list[str]]:
    """Concept ids and their surface forms.

    Expansion is by explicit catalogue membership: a concept's own synonyms and
    coded terms, plus the members of a named group where one is given.
    """
    concept_ids: list[str] = []
    if group:
        concept_ids.extend(catalog.expand_group(group))
    if concept:
        if concept in catalog.concept_groups:
            concept_ids.extend(catalog.expand_group(concept))
        else:
            concept_ids.append(concept)
    concept_ids = sorted(set(concept_ids))
    terms: list[str] = []
    for concept_id in concept_ids:
        terms.extend(catalog.synonyms(concept_id))
        terms.extend(catalog.concept(concept_id).abbreviations)
    return concept_ids, sorted(set(terms))


def _fts_query(terms: Sequence[str]) -> str:
    """An FTS5 MATCH expression: any of the expanded surface forms."""
    quoted = []
    for term in terms:
        cleaned = re.sub(r'["\']', " ", term).strip()
        if cleaned:
            quoted.append(f'"{cleaned}"')
    return " OR ".join(quoted)


def retrieve(
    index: EventIndex,
    catalog: ConceptCatalog,
    *,
    concept: str | None = None,
    group: str | None = None,
    text: str | None = None,
    assertion: Sequence[str] | None = None,
    evidence_state: Sequence[str] | None = None,
    verdict: Sequence[str] | None = None,
    window: tuple[int, int] | None = None,
    anchor: str | None = None,
    studies: Sequence[str] | None = None,
    severity: Sequence[str] | None = None,
    seriousness: Sequence[str] | None = None,
    action_taken: Sequence[str] | None = None,
    definition_id: str | None = None,
    definition_version: int | None = None,
    mode: Mode = "lexical",
    top_k: int = 20,
) -> RetrievalResult:
    """Retrieve event objects with their narrative context.

    ``assertion`` is applied as a SQL predicate on a column.  With it set to
    ``["present"]`` no record documenting an absence can come back, whatever the
    lexical or dense scorer thought of the text.
    """
    notes: list[str] = []
    concept_ids, terms = expand_concept(catalog, concept, group)

    dense_available = False
    if mode in ("dense", "hybrid"):
        from .dense import dense_backend_available

        dense_available = dense_backend_available()
        if not dense_available:
            notes.append(
                "no local embedding model is present; dense retrieval degraded "
                "to lexical. Assertion filtering is unaffected because it is a "
                "structured predicate, not a similarity heuristic."
            )
            mode = "lexical"

    match_terms = list(terms)
    if text:
        match_terms.append(text)

    # The lexical index scores documents; when a concept filter is present it
    # does not gate them. An event raised from symptoms and a glucose value
    # carries no surface form of the concept anywhere in its narrative, and
    # those are precisely the records the evidence ladder exists to catch.
    # Requiring a text match would drop them before any rule ran.
    structured_concept_filter = bool(concept_ids)
    lexical_scores = _lexical_scores(index, match_terms)
    if structured_concept_filter and match_terms:
        unmatched = "concept filter is structural; lexical score orders results"
        if unmatched not in notes:
            notes.append(unmatched)

    where: list[str] = []
    params: list[Any] = []

    if concept_ids:
        where.append(f"e.concept_id IN ({','.join('?' * len(concept_ids))})")
        params.extend(concept_ids)
    if assertion:
        where.append(f"e.assertion IN ({','.join('?' * len(assertion))})")
        params.extend(list(assertion))
    if studies:
        where.append(f"e.study_id IN ({','.join('?' * len(studies))})")
        params.extend(list(studies))
    if severity:
        where.append(f"e.severity IN ({','.join('?' * len(severity))})")
        params.extend(list(severity))
    if action_taken:
        where.append(f"e.action_taken IN ({','.join('?' * len(action_taken))})")
        params.extend(list(action_taken))
    if anchor:
        where.append("e.anchor_event = ?")
        params.append(anchor)
    if window is not None:
        where.append("e.onset_offset_days IS NOT NULL")
        where.append("e.onset_offset_days BETWEEN ? AND ?")
        params.extend([window[0], window[1]])
    if seriousness:
        clauses = []
        for category in seriousness:
            clauses.append(
                "(e.seriousness = ? OR e.seriousness LIKE ? "
                "OR e.seriousness LIKE ? OR e.seriousness LIKE ?)"
            )
            params.extend(
                [category, f"{category}|%", f"%|{category}", f"%|{category}|%"]
            )
        where.append("(" + " OR ".join(clauses) + ")")

    # Free-text search with no concept filter: the lexical match *is* the query,
    # so it filters rather than merely scores.
    if match_terms and not structured_concept_filter:
        if not lexical_scores:
            where.append("1 = 0")
        else:
            doc_ids = sorted(lexical_scores)
            where.append(f"e.doc_id IN ({','.join('?' * len(doc_ids))})")
            params.extend(doc_ids)

    join = " FROM events e JOIN documents d ON d.doc_id = e.doc_id"
    if evidence_state or verdict or definition_id:
        join += (
            " LEFT JOIN event_states s ON s.event_id = e.event_id"
            " AND s.definition_id = ?"
        )
        state_params: list[Any] = [definition_id or ""]
        if definition_version is not None:
            join += " AND s.definition_version = ?"
            state_params.append(definition_version)
        params = state_params + params
        if evidence_state:
            where.append(
                f"s.evidence_state IN ({','.join('?' * len(evidence_state))})"
            )
            params.extend(list(evidence_state))
        if verdict:
            where.append(f"s.verdict IN ({','.join('?' * len(verdict))})")
            params.extend(list(verdict))
    else:
        join += " LEFT JOIN event_states s ON s.event_id = e.event_id"

    clause = (" WHERE " + " AND ".join(where)) if where else ""
    query = (
        "SELECT e.*, d.text AS doc_text, s.evidence_state AS evidence_state, "
        "s.verdict AS verdict" + join + clause + " ORDER BY e.event_id"
    )
    rows = index.query(query, params)

    # Rank in Python: a document the lexical index scored comes first, then the
    # rest in a stable order. Sorting in SQL would mean threading the FTS table
    # back into a query whose filters are deliberately structural.
    scored_rows = sorted(
        rows,
        key=lambda row: (-lexical_scores.get(row["doc_id"], 0.0), row["event_id"]),
    )

    total_before = _unfiltered_count(index, match_terms)

    records: list[RetrievedRecord] = []
    for row in scored_rows[: max(top_k, 0) or None]:
        records.append(
            RetrievedRecord(
                event_id=row["event_id"],
                doc_id=row["doc_id"],
                subject_id=row["subject_id"],
                study_id=row["study_id"],
                concept_id=row["concept_id"],
                assertion=row["assertion"],
                evidence_state=row["evidence_state"],
                verdict=row["verdict"],
                onset_offset_days=row["onset_offset_days"],
                severity=row["severity"],
                seriousness=[s for s in (row["seriousness"] or "").split("|") if s],
                action_taken=row["action_taken"],
                coded_term=row["coded_term"],
                score=lexical_scores.get(row["doc_id"], 0.0),
                snippet=_snippet(row["doc_text"], terms),
                matched_terms=[
                    t for t in terms if t.lower() in (row["doc_text"] or "").lower()
                ],
            )
        )

    if mode == "hybrid" and dense_available:
        from .dense import rerank

        records = rerank(records, text or concept or "", index)
        notes.append("hybrid: lexical candidates reranked by local embeddings")

    return RetrievalResult(
        records=records,
        expanded_terms=terms,
        mode=mode,
        filters={
            "concept": concept,
            "group": group,
            "concept_ids": concept_ids,
            "text": text,
            "assertion": list(assertion) if assertion else None,
            "evidence_state": list(evidence_state) if evidence_state else None,
            "verdict": list(verdict) if verdict else None,
            "window": list(window) if window else None,
            "anchor": anchor,
            "studies": list(studies) if studies else None,
            "severity": list(severity) if severity else None,
            "seriousness": list(seriousness) if seriousness else None,
            "action_taken": list(action_taken) if action_taken else None,
            "definition": (
                f"{definition_id}.v{definition_version}" if definition_id else None
            ),
            "top_k": top_k,
        },
        total_before_filters=total_before,
        dense_available=dense_available,
        notes=notes,
    )


def _lexical_scores(index: EventIndex, match_terms: list[str]) -> dict[str, float]:
    """Document ids scored by the FTS5 index. bm25 is negative-better; negated."""
    if not match_terms:
        return {}
    rows = index.query(
        "SELECT d.doc_id AS doc_id, -bm25(documents_fts) AS score "
        "FROM documents d JOIN documents_fts f ON f.rowid = d.rowid "
        "WHERE documents_fts MATCH ?",
        (_fts_query(match_terms),),
    )
    return {row["doc_id"]: float(row["score"]) for row in rows}


def _unfiltered_count(index: EventIndex, match_terms: list[str]) -> int:
    if not match_terms:
        return index.query("SELECT COUNT(*) FROM events")[0][0]
    return index.query(
        "SELECT COUNT(*) FROM events e JOIN documents d ON d.doc_id = e.doc_id "
        "JOIN documents_fts f ON f.rowid = d.rowid WHERE documents_fts MATCH ?",
        (_fts_query(match_terms),),
    )[0][0]


def _snippet(text: str, terms: list[str], width: int = 110) -> str:
    """A window of text around the first matching term."""
    if not text:
        return ""
    lowered = text.lower()
    best = None
    for term in terms:
        position = lowered.find(term.lower())
        if position >= 0 and (best is None or position < best):
            best = position
    if best is None:
        return text[:width].strip()
    start = max(0, best - width // 3)
    end = min(len(text), best + width)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{text[start:end].strip()}{suffix}"
