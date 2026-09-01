"""Two retrieval paths, and a guard between them.

**Precise cohort.**  A phenotype definition, a window and study metadata over
canonical episodes.  No embeddings, no similarity, no candidates.  What comes
back is what the definition claims.

**Discovery.**  Lexical, optionally dense, terminology-aware, with assertion,
temporality and provenance as structured predicates.  What comes back is marked
``candidate=True``.

**The guard.**  A discovery candidate cannot enter a cohort.  Not by being
similar enough, not by being high-scoring, not by being convenient.  It enters
through adjudication, or through a new definition version that claims it on the
evidence.  ``as_cohort()`` refuses candidates outright rather than filtering
them quietly, because a silent filter is how a candidate ends up in a
denominator.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as _dc_field
from typing import Any, Literal, Sequence

from ..catalog import ConceptCatalog, ConfigError
from .index import EpisodeIndex

Mode = Literal["precise", "lexical", "dense", "hybrid"]


class CandidateInCohort(RuntimeError):
    """Raised when an unadjudicated discovery candidate is used as a cohort."""


@dataclass
class RetrievedEpisode:
    episode_id: str
    subject_id: str
    study_id: str
    representation: str
    standardized_concept: str | None
    coded_terms: list[str]
    assertions: list[str]
    evidence_state: str | None
    verdict: str | None
    onset_offset_days: int | None
    peak_severity: str | None
    record_count: int
    linkage_rule: str
    linkage_confidence: float
    linkage_review: bool
    provenance_paths: list[str]
    candidate: bool
    score: float
    snippet: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "subject_id": self.subject_id,
            "study_id": self.study_id,
            "representation": self.representation,
            "standardized_concept": self.standardized_concept,
            "coded_terms": self.coded_terms,
            "assertions": self.assertions,
            "evidence_state": self.evidence_state,
            "verdict": self.verdict,
            "onset_offset_days": self.onset_offset_days,
            "peak_severity": self.peak_severity,
            "record_count": self.record_count,
            "linkage_rule": self.linkage_rule,
            "linkage_confidence": round(self.linkage_confidence, 4),
            "linkage_review_required": self.linkage_review,
            "provenance": self.provenance_paths,
            "candidate": self.candidate,
            "score": round(self.score, 6),
            "snippet": self.snippet,
        }


@dataclass
class RetrievalResult:
    records: list[RetrievedEpisode]
    mode: Mode
    expanded_terms: list[str]
    filters: dict[str, Any]
    dense_available: bool = False
    notes: list[str] = _dc_field(default_factory=list)

    @property
    def negation_false_positives(self) -> int:
        return sum(1 for r in self.records if "absent" in r.assertions)

    @property
    def negation_false_positive_rate(self) -> float:
        return (
            self.negation_false_positives / len(self.records) if self.records else 0.0
        )

    @property
    def candidates_excluded(self) -> int:
        return sum(1 for r in self.records if r.candidate)

    def as_cohort(self) -> list[RetrievedEpisode]:
        """The results, usable as a cohort — or a refusal explaining why not.

        Discovery output is a hypothesis. Letting it become a denominator
        without adjudication is how an exploratory search turns into a claim
        nobody sanctioned.
        """
        if self.mode != "precise":
            raise CandidateInCohort(
                f"retrieval mode {self.mode!r} returns discovery candidates, "
                f"which cannot form a cohort. Adjudicate them, or write a "
                f"definition version that claims them on the evidence, then run "
                f"the precise path."
            )
        offenders = [r.episode_id for r in self.records if r.candidate]
        if offenders:
            raise CandidateInCohort(
                f"{len(offenders)} unadjudicated candidate(s) in the result "
                f"set, e.g. {offenders[:3]}; a candidate cannot enter a cohort."
            )
        return list(self.records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "count": len(self.records),
            "filters": self.filters,
            "expanded_terms": self.expanded_terms,
            "negation_false_positives": self.negation_false_positives,
            "negation_false_positive_rate": round(
                self.negation_false_positive_rate, 6
            ),
            "candidates": self.candidates_excluded,
            "usable_as_cohort": self.mode == "precise" and self.candidates_excluded == 0,
            "dense_available": self.dense_available,
            "notes": self.notes,
            "records": [r.to_dict() for r in self.records],
        }


def expand_concept(
    catalog: ConceptCatalog, concept: str | None, group: str | None = None
) -> tuple[list[str], list[str]]:
    """Concept ids and surface forms, by explicit catalogue membership.

    A named group expands to its declared members.  No hierarchy is walked as
    though it implied subsumption.
    """
    ids: list[str] = []
    if group:
        ids.extend(catalog.expand_group(group))
    if concept:
        if concept in catalog.concept_groups:
            ids.extend(catalog.expand_group(concept))
        else:
            ids.append(concept)
    ids = sorted(set(ids))
    terms: list[str] = []
    for concept_id in ids:
        terms.extend(catalog.synonyms(concept_id))
        terms.extend(catalog.concept(concept_id).abbreviations)
    return ids, sorted(set(terms))


def _fts_query(terms: Sequence[str]) -> str:
    quoted = [f'"{re.sub(chr(34) + "|" + chr(39), " ", t).strip()}"' for t in terms]
    return " OR ".join(q for q in quoted if q != '""')


def _lexical_scores(index: EpisodeIndex, terms: Sequence[str]) -> dict[str, float]:
    if not terms:
        return {}
    rows = index.query(
        "SELECT d.doc_id AS doc_id, -bm25(documents_fts) AS score "
        "FROM documents d JOIN documents_fts f ON f.rowid = d.rowid "
        "WHERE documents_fts MATCH ?",
        (_fts_query(terms),),
    )
    return {row["doc_id"]: float(row["score"]) for row in rows}


def retrieve(
    index: EpisodeIndex,
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
    representation: Sequence[str] | None = None,
    provenance: Sequence[str] | None = None,
    linkage_review: bool | None = None,
    definition_id: str | None = None,
    definition_version: int | None = None,
    mode: Mode = "precise",
    top_k: int = 20,
) -> RetrievalResult:
    """Query episodes.

    ``mode="precise"`` returns cohort-eligible episodes.  Any other mode is
    discovery, and its results are marked as candidates.
    """
    notes: list[str] = []
    concept_ids, terms = expand_concept(catalog, concept, group)
    discovery = mode != "precise"

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

    where: list[str] = []
    params: list[Any] = []

    if concept_ids:
        where.append(
            f"e.standardized_concept IN ({','.join('?' * len(concept_ids))})"
        )
        params.extend(concept_ids)
    if studies:
        where.append(f"e.study_id IN ({','.join('?' * len(studies))})")
        params.extend(list(studies))
    if severity:
        where.append(f"e.peak_severity IN ({','.join('?' * len(severity))})")
        params.extend(list(severity))
    if representation:
        where.append(f"e.representation IN ({','.join('?' * len(representation))})")
        params.extend(list(representation))
    if anchor:
        where.append("e.anchor_event = ?")
        params.append(anchor)
    if window is not None:
        where.append("e.onset_offset_days IS NOT NULL")
        where.append("e.onset_offset_days BETWEEN ? AND ?")
        params.extend([window[0], window[1]])
    if linkage_review is not None:
        where.append("e.linkage_review = ?")
        params.append(1 if linkage_review else 0)
    if provenance:
        clauses = []
        for kind in provenance:
            clauses.append(
                "(e.provenance_paths = ? OR e.provenance_paths LIKE ? "
                "OR e.provenance_paths LIKE ? OR e.provenance_paths LIKE ?)"
            )
            params.extend([kind, f"{kind}|%", f"%|{kind}", f"%|{kind}|%"])
        where.append("(" + " OR ".join(clauses) + ")")
    if assertion:
        # Assertion is a column. With `present` selected, an episode that
        # documents an absence cannot come back, whatever the scorer thought.
        clauses = []
        for value in assertion:
            clauses.append(
                "(e.assertions = ? OR e.assertions LIKE ? OR e.assertions LIKE ? "
                "OR e.assertions LIKE ?)"
            )
            params.extend([value, f"{value}|%", f"%|{value}", f"%|{value}|%"])
        where.append("(" + " OR ".join(clauses) + ")")
        if "absent" not in assertion:
            where.append(
                "NOT (e.assertions = 'absent' OR e.assertions LIKE 'absent|%' "
                "OR e.assertions LIKE '%|absent' OR e.assertions LIKE '%|absent|%')"
            )

    join = " FROM episodes e"
    state_params: list[Any] = []
    if evidence_state or verdict or definition_id:
        join += (
            " LEFT JOIN episode_states s ON s.episode_id = e.episode_id"
            " AND s.definition_id = ?"
        )
        state_params.append(definition_id or "")
        if definition_version is not None:
            join += " AND s.definition_version = ?"
            state_params.append(definition_version)
        if evidence_state:
            where.append(
                f"s.evidence_state IN ({','.join('?' * len(evidence_state))})"
            )
            params.extend(list(evidence_state))
        if verdict:
            where.append(f"s.verdict IN ({','.join('?' * len(verdict))})")
            params.extend(list(verdict))
    else:
        join += " LEFT JOIN episode_states s ON s.episode_id = e.episode_id"

    match_terms = list(terms)
    if text:
        match_terms.append(text)
    lexical = _lexical_scores(index, match_terms) if discovery or match_terms else {}

    # Free-text discovery with no concept filter: the lexical hit *is* the
    # query. With a concept filter, the structured column decides membership
    # and the lexical score only orders — an episode raised from symptoms and a
    # glucose value carries no surface form of the concept anywhere.
    if text and not concept_ids:
        docs = sorted(lexical)
        if not docs:
            where.append("1 = 0")
        else:
            clauses = []
            for doc in docs:
                clauses.append("(e.doc_ids = ? OR e.doc_ids LIKE ? OR e.doc_ids LIKE ?)")
                params.extend([doc, f"{doc}|%", f"%|{doc}"])
            where.append("(" + " OR ".join(clauses) + ")")

    clause = (" WHERE " + " AND ".join(where)) if where else ""
    sql = (
        "SELECT e.*, s.evidence_state AS evidence_state, s.verdict AS verdict"
        + join + clause + " ORDER BY e.episode_id"
    )
    rows = index.query(sql, state_params + params)

    def score_for(row) -> float:
        docs = [d for d in (row["doc_ids"] or "").split("|") if d]
        return max((lexical.get(d, 0.0) for d in docs), default=0.0)

    ordered = sorted(rows, key=lambda r: (-score_for(r), r["episode_id"]))
    records = [
        RetrievedEpisode(
            episode_id=row["episode_id"],
            subject_id=row["subject_id"],
            study_id=row["study_id"],
            representation=row["representation"],
            standardized_concept=row["standardized_concept"],
            coded_terms=[t for t in (row["coded_terms"] or "").split("|") if t],
            assertions=[a for a in (row["assertions"] or "").split("|") if a],
            evidence_state=row["evidence_state"],
            verdict=row["verdict"],
            onset_offset_days=row["onset_offset_days"],
            peak_severity=row["peak_severity"],
            record_count=row["record_count"],
            linkage_rule=row["linkage_rule"],
            linkage_confidence=row["linkage_confidence"],
            linkage_review=bool(row["linkage_review"]),
            provenance_paths=[p for p in (row["provenance_paths"] or "").split("|") if p],
            # Discovery output is a candidate whatever the index says.
            candidate=bool(row["candidate"]) or discovery,
            score=score_for(row),
            snippet=_snippet(index, row, terms),
        )
        for row in ordered[: max(top_k, 0) or None]
    ]

    if discovery:
        notes.append(
            "discovery results are candidates. They may not enter a cohort "
            "without adjudication or a definition version that claims them."
        )

    return RetrievalResult(
        records=records,
        mode=mode,
        expanded_terms=terms,
        filters={
            "concept": concept, "group": group, "concept_ids": concept_ids,
            "text": text, "assertion": list(assertion) if assertion else None,
            "evidence_state": list(evidence_state) if evidence_state else None,
            "verdict": list(verdict) if verdict else None,
            "window": list(window) if window else None, "anchor": anchor,
            "studies": list(studies) if studies else None,
            "severity": list(severity) if severity else None,
            "representation": list(representation) if representation else None,
            "provenance": list(provenance) if provenance else None,
            "linkage_review": linkage_review,
            "definition": (
                f"{definition_id}.v{definition_version}" if definition_id else None
            ),
            "top_k": top_k,
        },
        dense_available=dense_available,
        notes=notes,
    )


def _snippet(index: EpisodeIndex, row, terms: Sequence[str], width: int = 130) -> str:
    docs = [d for d in (row["doc_ids"] or "").split("|") if d]
    if not docs:
        return ""
    document = index.document(docs[0])
    if not document:
        return ""
    text = document["text"]
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
    return (
        ("..." if start else "") + text[start:end].strip() + ("..." if end < len(text) else "")
    )


# --------------------------------------------------------------------------
# Discovery over narrative mentions
# --------------------------------------------------------------------------


@dataclass
class DiscoveredMention:
    """A place in a document where a concept is named, and how it is asserted."""

    mention_id: str
    doc_id: str
    study_id: str
    subject_id: str
    concept_id: str
    assertion: str
    match_kind: str
    surface: str
    sentence: str
    cue: str | None
    score: float
    #: Always true. Discovery output is a hypothesis, never a cohort member.
    candidate: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "mention_id": self.mention_id, "doc_id": self.doc_id,
            "study_id": self.study_id, "subject_id": self.subject_id,
            "concept_id": self.concept_id, "assertion": self.assertion,
            "match_kind": self.match_kind, "surface": self.surface,
            "sentence": self.sentence, "cue": self.cue,
            "score": round(self.score, 6), "candidate": self.candidate,
        }


@dataclass
class DiscoveryResult:
    mentions: list[DiscoveredMention]
    mode: Mode
    expanded_terms: list[str]
    filters: dict[str, Any]
    total_before_filters: int
    dense_available: bool = False
    notes: list[str] = _dc_field(default_factory=list)

    @property
    def negation_false_positives(self) -> int:
        """Returned mentions that document an absence of the concept."""
        return sum(1 for m in self.mentions if m.assertion == "absent")

    @property
    def negation_false_positive_rate(self) -> float:
        return (
            self.negation_false_positives / len(self.mentions) if self.mentions else 0.0
        )

    def as_cohort(self):
        raise CandidateInCohort(
            "discovery returns candidate mentions. A mention is a place in a "
            "document where a concept is named, not an episode that occurred; "
            "it enters a cohort through adjudication or a definition version "
            "that claims it, never directly."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "count": len(self.mentions),
            "total_before_filters": self.total_before_filters,
            "filters": self.filters,
            "expanded_terms": self.expanded_terms,
            "negation_false_positives": self.negation_false_positives,
            "negation_false_positive_rate": round(
                self.negation_false_positive_rate, 6
            ),
            "all_candidates": all(m.candidate for m in self.mentions),
            "dense_available": self.dense_available,
            "notes": self.notes,
            "mentions": [m.to_dict() for m in self.mentions],
        }


def discover(
    index: EpisodeIndex,
    catalog: ConceptCatalog,
    *,
    concept: str | None = None,
    group: str | None = None,
    text: str | None = None,
    assertion: Sequence[str] | None = None,
    studies: Sequence[str] | None = None,
    match_kind: Sequence[str] | None = None,
    mode: Mode = "lexical",
    top_k: int = 20,
) -> DiscoveryResult:
    """Search narrative text for concept mentions.

    Assertion is a structured predicate here too, and this is where it earns
    its keep: a coded AE row asserts presence by construction, but a narrative
    can name hypoglycemia precisely in order to record that it did not happen.
    """
    notes: list[str] = []
    concept_ids, terms = expand_concept(catalog, concept, group)

    dense_available = False
    if mode in ("dense", "hybrid"):
        from .dense import dense_backend_available

        dense_available = dense_backend_available()
        if not dense_available:
            notes.append(
                "no local embedding model is present; dense discovery degraded "
                "to lexical. Assertion filtering is unaffected: it is a "
                "structured predicate, not a similarity heuristic."
            )
            mode = "lexical"

    where: list[str] = []
    params: list[Any] = []
    if concept_ids:
        where.append(f"m.concept_id IN ({','.join('?' * len(concept_ids))})")
        params.extend(concept_ids)
    if studies:
        where.append(f"m.study_id IN ({','.join('?' * len(studies))})")
        params.extend(list(studies))
    if match_kind:
        where.append(f"m.match_kind IN ({','.join('?' * len(match_kind))})")
        params.extend(list(match_kind))
    if assertion:
        where.append(f"m.assertion IN ({','.join('?' * len(assertion))})")
        params.extend(list(assertion))

    unfiltered_where: list[str] = []
    unfiltered_params: list[Any] = []
    if concept_ids:
        unfiltered_where.append(
            f"m.concept_id IN ({','.join('?' * len(concept_ids))})"
        )
        unfiltered_params.extend(concept_ids)

    match_terms = list(terms) + ([text] if text else [])
    lexical = _lexical_scores(index, match_terms)
    if text and not concept_ids:
        docs = sorted(lexical)
        if not docs:
            where.append("1 = 0")
        else:
            where.append(f"m.doc_id IN ({','.join('?' * len(docs))})")
            params.extend(docs)

    clause = (" WHERE " + " AND ".join(where)) if where else ""
    rows = index.query("SELECT m.* FROM mentions m" + clause, params)
    total = len(
        index.query(
            "SELECT m.mention_id FROM mentions m"
            + ((" WHERE " + " AND ".join(unfiltered_where)) if unfiltered_where else ""),
            unfiltered_params,
        )
    )

    ordered = sorted(
        rows,
        key=lambda r: (-lexical.get(r["doc_id"], 0.0), r["mention_id"]),
    )
    mentions = [
        DiscoveredMention(
            mention_id=row["mention_id"], doc_id=row["doc_id"],
            study_id=row["study_id"], subject_id=row["subject_id"],
            concept_id=row["concept_id"], assertion=row["assertion"],
            match_kind=row["match_kind"], surface=row["surface"],
            sentence=row["sentence"], cue=row["cue"],
            score=lexical.get(row["doc_id"], 0.0),
        )
        for row in ordered[: max(top_k, 0) or None]
    ]
    notes.append(
        "discovery returns candidate mentions; they may not enter a cohort "
        "without adjudication or a definition version that claims them."
    )
    return DiscoveryResult(
        mentions=mentions, mode=mode, expanded_terms=terms,
        filters={
            "concept": concept, "group": group, "text": text,
            "assertion": list(assertion) if assertion else None,
            "studies": list(studies) if studies else None,
            "match_kind": list(match_kind) if match_kind else None,
            "top_k": top_k,
        },
        total_before_filters=total, dense_available=dense_available, notes=notes,
    )
