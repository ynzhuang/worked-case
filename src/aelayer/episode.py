"""Episode reconciliation.

An evolving event may be recorded as several source records under some
collection conventions, and as one under others.  Deriving an episode view lets
an analysis reason about the clinical event; keeping the records untouched
underneath means the derivation can be redone when the assumption changes.
Collapsing rows in place would be irreversible, so nothing here modifies a
record.

**Reconciliation is a declared assumption, not a solved problem.**  The default
rule — same subject, same concept, intervals overlapping or close — is wrong for
recurrent conditions.  Two hypoglycemia records three days apart are plausibly
two episodes; two anaemia records three days apart are plausibly one condition
changing grade.  So the catalogue declares ``recurrence_expected`` per concept,
the split falls that way by default, and anything the rules cannot settle is
flagged with ``linkage_review_required`` rather than quietly resolved.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field as _dc_field
from typing import Any, Iterable, Sequence

from .anchors import AnchorResolver
from .catalog import ConceptCatalog
from .models import (
    SERIOUSNESS_CRITERIA,
    CanonicalAEEpisode,
    CanonicalAERecord,
    Field,
    LinkageRule,
    Span,
)
from .semantics import CollectionSemantics


@dataclass(frozen=True)
class LinkageDecision:
    """Whether two adjacent records belong to the same episode, and why."""

    attach: bool
    rule: LinkageRule
    confidence: float
    review_required: bool
    note: str


@dataclass
class ReconciliationConfig:
    gap_days: int = 1
    ambiguous_gap_days: int = 4
    confidence: dict[str, float] = _dc_field(default_factory=dict)

    @classmethod
    def from_catalog(cls, catalog: ConceptCatalog) -> "ReconciliationConfig":
        body = catalog.episode_reconciliation or {}
        return cls(
            gap_days=int(body.get("gap_days", 1)),
            ambiguous_gap_days=int(body.get("ambiguous_gap_days", 4)),
            confidence={k: float(v) for k, v in (body.get("confidence") or {}).items()},
        )

    def score(self, rule: str, default: float = 0.5) -> float:
        return self.confidence.get(rule, default)


class EpisodeReconciler:
    def __init__(
        self,
        catalog: ConceptCatalog,
        semantics: CollectionSemantics,
        config: ReconciliationConfig | None = None,
        anchor_resolver: AnchorResolver | None = None,
        default_anchor: str | None = None,
    ):
        self.catalog = catalog
        self.semantics = semantics
        self.config = config or ReconciliationConfig.from_catalog(catalog)
        # Used only to stamp an offset on the episode so it can be filtered on
        # without re-running a phenotype definition. A definition still resolves
        # its own anchor: this is a retrieval convenience, not a case criterion.
        self.resolver = anchor_resolver
        self.default_anchor = default_anchor

    # -- the decision -------------------------------------------------------

    def decide(
        self,
        previous: CanonicalAERecord,
        candidate: CanonicalAERecord,
        concept: str | None,
    ) -> LinkageDecision:
        """Does ``candidate`` continue the episode ``previous`` belongs to?"""
        # 1. The CRF said so. Nothing beats a declared continuation.
        if candidate.continuation_of == previous.source_record_id:
            return LinkageDecision(
                True, "explicit_continuation",
                self.config.score("explicit_continuation", 1.0), False,
                f"record declares continuation of {previous.source_record_id}",
            )

        start = candidate.onset_datetime.value
        previous_start = previous.onset_datetime.value
        previous_end = previous.end_datetime.value

        if start is None or previous_start is None:
            # Without a usable onset there is no temporal evidence either way.
            return LinkageDecision(
                False, "single_record", 0.4, True,
                "onset is not resolvable on one of the records, so temporal "
                "linkage cannot be evaluated",
            )

        # 2. The study declares that it splits a worsening event across records.
        study = self.semantics.for_study(candidate.study_id)
        gap = (start.date() - (previous_end or previous_start).date()).days
        if study.splits_on_severity_change():
            severity_changed = (
                previous.severity.value != candidate.severity.value
                and candidate.severity.populated
            )
            if severity_changed and gap <= self.config.ambiguous_gap_days:
                return LinkageDecision(
                    True, "declared_convention",
                    self.config.score("declared_convention", 0.9), False,
                    f"{study.study_id} declares record_splitting="
                    f"{study.record_splitting}; severity moved "
                    f"{previous.severity.value} -> {candidate.severity.value} "
                    f"after {gap} day(s)",
                )

        # 3. Intervals that actually overlap are one episode either way.
        if previous_end is not None and start <= previous_end:
            return LinkageDecision(
                True, "temporal_overlap",
                self.config.score("temporal_overlap", 0.95), False,
                f"onset {start.date()} falls inside the previous record's "
                f"interval, which ends {previous_end.date()}",
            )

        # 4. Everything below turns on whether this concept recurs.
        recurs = (
            self.catalog.concept(concept).recurrence_expected if concept else True
        )
        if recurs:
            review = gap <= self.config.ambiguous_gap_days
            return LinkageDecision(
                False, "recurrence_split",
                self.config.score("recurrence_split", 0.9), review,
                f"{concept} is expected to recur, so a {gap}-day gap is read as "
                f"a second episode rather than a continuation"
                + (
                    "; the gap is short enough that this is a judgement call and "
                    "is flagged for review"
                    if review else ""
                ),
            )

        if gap <= self.config.gap_days:
            return LinkageDecision(
                True, "gap_within_tolerance",
                self.config.score("gap_within_tolerance", 0.85), False,
                f"{concept} is not expected to recur and the records are "
                f"{gap} day(s) apart",
            )
        if gap <= self.config.ambiguous_gap_days:
            return LinkageDecision(
                True, "gap_within_tolerance",
                self.config.score("gap_within_tolerance", 0.85) * 0.8, True,
                f"{concept} is not expected to recur but the records are "
                f"{gap} day(s) apart, beyond the {self.config.gap_days}-day "
                f"tolerance; linked, and flagged for review",
            )
        return LinkageDecision(
            False, "single_record", 0.9, False,
            f"records are {gap} day(s) apart, beyond any linkage tolerance",
        )

    # -- grouping -----------------------------------------------------------

    def reconcile(
        self, records: Iterable[CanonicalAERecord]
    ) -> list[CanonicalAEEpisode]:
        """Derive episodes over records, leaving the records untouched."""
        records = list(records)
        # An explicit continuation declared by the CRF is the strongest evidence
        # available and outranks everything else, including whether the concept
        # could be standardized at all. Resolving those chains before grouping
        # is what stops two records the study itself linked from landing in
        # different episodes because neither coded to a catalogue term.
        component = _continuation_components(records)

        # A continuation record often carries no narrative of its own, so it
        # standardizes to nothing. It inherits the concept of the chain the CRF
        # put it in rather than being stranded as unmapped.
        component_concept: dict[str, str | None] = {}
        for record in records:
            root = component[record.source_record_id]
            if component_concept.get(root) is None:
                component_concept[root] = record.standardized_concept

        grouped: dict[tuple[str, str, str], list[CanonicalAERecord]] = {}
        for record in records:
            root = component[record.source_record_id]
            concept = record.standardized_concept or component_concept.get(root)
            key = (
                record.study_id,
                record.subject_id,
                concept or f"__unmapped__:{root}",
            )
            grouped.setdefault(key, []).append(record)

        episodes: list[CanonicalAEEpisode] = []
        for (study_id, subject_id, concept_key), group in sorted(grouped.items()):
            concept = None if concept_key.startswith("__unmapped__") else concept_key
            ordered = sorted(group, key=_ordering_key)
            chains = self._chain(ordered, concept)
            for index, (chain, decision) in enumerate(chains):
                episodes.append(
                    self.build_episode(
                        chain, concept, index, decision, study_id, subject_id
                    )
                )
        return sorted(episodes, key=lambda e: e.episode_id)

    def _chain(
        self, ordered: Sequence[CanonicalAERecord], concept: str | None
    ) -> list[tuple[list[CanonicalAERecord], LinkageDecision]]:
        """Walk the ordered records, opening a new episode where linkage fails."""
        chains: list[tuple[list[CanonicalAERecord], LinkageDecision]] = []
        current: list[CanonicalAERecord] = []
        decision = LinkageDecision(
            True, "single_record", self.config.score("single_record", 1.0), False,
            "a single source record",
        )
        for record in ordered:
            if not current:
                current = [record]
                decision = LinkageDecision(
                    True, "single_record", self.config.score("single_record", 1.0),
                    False, "a single source record",
                )
                continue
            verdict = self.decide(current[-1], record, concept)
            if verdict.attach:
                current.append(record)
                # The weakest link governs the chain's confidence, and any
                # flagged link flags the whole episode.
                decision = LinkageDecision(
                    True, verdict.rule,
                    min(decision.confidence, verdict.confidence),
                    decision.review_required or verdict.review_required,
                    _join_notes(decision.note, verdict.note),
                )
            else:
                chains.append((current, decision))
                current = [record]
                decision = LinkageDecision(
                    True, verdict.rule, verdict.confidence, verdict.review_required,
                    verdict.note,
                )
        if current:
            chains.append((current, decision))
        return chains

    # -- construction -------------------------------------------------------

    def build_episode(
        self,
        chain: Sequence[CanonicalAERecord],
        concept: str | None,
        index: int,
        decision: LinkageDecision,
        study_id: str,
        subject_id: str,
    ) -> CanonicalAEEpisode:
        """Assemble one episode from its chain, carrying every reason forward."""
        first, last = chain[0], chain[-1]
        # An episode whose concept could not be standardized is grouped by its
        # own record, so its id carries that record rather than colliding with
        # every other unmapped episode for the same subject.
        episode_id = (
            f"{subject_id}::{concept}::{index + 1:02d}" if concept
            else f"{subject_id}::UNMAPPED::{first.source_record_id}"
        )

        severity_trajectory = [
            (r.onset_datetime.value, r.severity.value)
            for r in chain if r.severity.populated
        ]
        seriousness_trajectory = [
            (
                r.onset_datetime.value,
                sorted(
                    c for c, f in r.seriousness_criteria.items()
                    if f.value is True
                ),
            )
            for r in chain if r.seriousness.value is True
        ]
        action_history = [
            (r.onset_datetime.value, r.action_taken.value)
            for r in chain if r.action_taken.populated
        ]

        spans: list[Span] = []
        for record in chain:
            spans.extend(record.evidence)
        for symptom in (s for r in chain for s in r.symptoms):
            spans.append(symptom.span)
        for lab in (l for r in chain for l in r.labs):
            spans.append(lab.span)

        offset, anchor_event, anchor_date = self._anchor(
            subject_id, first.onset_datetime.value
        )

        episode = CanonicalAEEpisode(
            episode_id=episode_id,
            study_id=study_id,
            subject_id=subject_id,
            standardized_concept=concept,
            episode_start=_carry(first.onset_datetime, "derived"),
            onset_offset_days=offset,
            anchor_event=anchor_event,
            anchor_datetime=anchor_date,
            episode_end=_episode_end(chain),
            source_record_ids=[r.source_record_id for r in chain],
            severity_trajectory=severity_trajectory,
            seriousness_trajectory=seriousness_trajectory,
            relatedness=_strongest(chain, "relatedness"),
            action_history=action_history,
            outcome=_carry(last.outcome, "derived"),
            seriousness=_any_true(chain),
            symptoms=_dedupe_symptoms(chain),
            labs=_dedupe_labs(chain),
            coded_terms=sorted({r.coded_term.value for r in chain if r.coded_term.populated}),
            verbatim_terms=sorted(
                {r.verbatim_term.value for r in chain if r.verbatim_term.populated}
            ),
            dictionary_versions=sorted(
                {r.dictionary_version for r in chain if r.dictionary_version}
            ),
            assertions=sorted({r.assertion.value for r in chain if r.assertion.populated}),
            linked_evidence=_dedupe_spans(spans),
            linkage_rule=decision.rule if len(chain) > 1 else "single_record",
            linkage_confidence=(
                decision.confidence if len(chain) > 1
                else self.config.score("single_record", 1.0)
            ),
            linkage_review_required=decision.review_required,
            linkage_note=decision.note,
            field_states=_field_states(chain),
            field_notes=_field_notes(chain),
            episode_provenance={
                "record_count": len(chain),
                "source_record_ids": [r.source_record_id for r in chain],
                "normalizer_versions": sorted({r.normalizer_version for r in chain}),
                "extractor_versions": sorted(
                    {r.extractor_version for r in chain if r.extractor_version}
                ),
                "representation_hint": self.semantics.for_study(study_id).representation,
            },
        )
        return episode

    def _anchor(
        self, subject_id: str, start: _dt.datetime | None
    ) -> tuple[Field[int], str | None, _dt.datetime | None]:
        """The episode's offset from the study's default anchor, if resolvable.

        Unresolvable is a state, not a zero: with no exposure record to measure
        from, the offset stays ``unknown`` and says why, rather than defaulting
        to a number a filter would silently trust.
        """
        if self.resolver is None or self.default_anchor is None:
            return Field[int](
                collection_state="unknown", source="derived",
                note="no anchor configuration available to resolve an offset",
            ), None, None
        if start is None:
            return Field[int](
                collection_state="unknown", source="derived",
                note="the episode has no resolvable start, so no offset exists",
            ), None, None
        hit = self.resolver.resolve(
            subject_id, self.default_anchor, onset_date=start.date()
        )
        if hit is None:
            return Field[int](
                collection_state="unknown", source="derived",
                note=(
                    f"no {self.default_anchor} occurrence in this subject's "
                    f"exposure record"
                ),
            ), None, None
        anchor_date = _dt.datetime.combine(hit.date, _dt.time.min)
        return (
            Field[int](
                value=(start.date() - hit.date).days,
                collection_state="collected",
                source="derived",
                note=f"{self.default_anchor} on {hit.date.isoformat()} ({hit.detail})",
            ),
            self.default_anchor,
            anchor_date,
        )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _continuation_components(
    records: Sequence[CanonicalAERecord],
) -> dict[str, str]:
    """Group records the CRF explicitly links, by union-find over the chain."""
    parent: dict[str, str] = {r.source_record_id: r.source_record_id for r in records}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: str, b: str) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    for record in records:
        if record.continuation_of and record.continuation_of in parent:
            union(record.source_record_id, record.continuation_of)
    return {rid: find(rid) for rid in parent}


def _ordering_key(record: CanonicalAERecord) -> tuple:
    """Records with a resolvable onset first, in time order; the rest last.

    A record whose onset cannot be resolved is not placed by guesswork.
    """
    onset = record.onset_datetime.value
    return (onset is None, onset or _dt.datetime.max, record.source_record_id)


def _join_notes(existing: str, addition: str) -> str:
    if not existing or existing == "a single source record":
        return addition
    return f"{existing}; {addition}"


def _carry(source: Field[Any], new_source: str) -> Field[Any]:
    """Lift a record field to episode level, keeping its state and spans."""
    return Field(
        value=source.value,
        collection_state=source.collection_state,
        source=new_source,  # type: ignore[arg-type]
        spans=list(source.spans),
        confidence=source.confidence,
        note=source.note,
    )


def _episode_end(chain: Sequence[CanonicalAERecord]) -> Field[_dt.datetime]:
    """The episode's end, or the reason it does not have one yet.

    An episode whose last record is still ongoing has a pending end, not a
    missing one, and the difference decides whether a duration can be computed.
    """
    ends = [r.end_datetime for r in chain if r.end_datetime.populated]
    if ends:
        latest = max(ends, key=lambda f: f.value)
        return _carry(latest, "derived")
    pending = [r.end_datetime for r in chain
               if r.end_datetime.collection_state == "pending_ongoing"]
    if pending:
        return _carry(pending[-1], "derived")
    return _carry(chain[-1].end_datetime, "derived")


def _strongest(chain: Sequence[CanonicalAERecord], name: str) -> Field[Any]:
    """The last collected value for a field, else the last field as it stands.

    Preferring a collected value over an empty one is not a merge: the record
    that carried it is unchanged and still named in `source_record_ids`.
    """
    collected = [
        getattr(r, name) for r in chain
        if getattr(r, name).collection_state == "collected"
    ]
    return _carry(collected[-1] if collected else getattr(chain[-1], name), "derived")


def _any_true(chain: Sequence[CanonicalAERecord]) -> Field[bool]:
    """Seriousness for the episode: serious if any constituent record is.

    Where no record answered the gate, the episode inherits the unanswered
    state rather than defaulting to not-serious.
    """
    for record in chain:
        if record.seriousness.value is True:
            return _carry(record.seriousness, "derived")
    collected = [r.seriousness for r in chain
                 if r.seriousness.collection_state == "collected"]
    return _carry(collected[-1] if collected else chain[-1].seriousness, "derived")


def _dedupe_symptoms(chain: Sequence[CanonicalAERecord]):
    seen: set[str] = set()
    out = []
    for record in chain:
        for symptom in record.symptoms:
            if symptom.symptom not in seen:
                seen.add(symptom.symptom)
                out.append(symptom)
    return sorted(out, key=lambda s: s.symptom)


def _dedupe_labs(chain: Sequence[CanonicalAERecord]):
    """One entry per distinct measurement.

    The same value can arrive from a narrative and from a linked form; that is
    one measurement with two pieces of provenance, not two results.
    """
    seen: dict[tuple, Any] = {}
    for record in chain:
        for lab in record.labs:
            key = (lab.test, round(lab.canonical_value or -1.0, 2),
                   lab.collection_datetime)
            if key not in seen:
                seen[key] = lab
    return [seen[k] for k in sorted(seen, key=lambda k: (k[0], k[1], str(k[2])))]


#: Most informative first. A state that names a reason beats a bare `unknown`.
_STATE_PRIORITY = (
    "collected",
    "not_representable",
    "pending_ongoing",
    "not_applicable_gated",
    "intentionally_blank",
    "not_collected_by_protocol",
    "unknown",
)


def _field_states(chain: Sequence[CanonicalAERecord]) -> dict[str, str]:
    """Summarise each field's collection state across the chain.

    A value collected on any constituent record makes the episode's field
    collected. Otherwise the most informative reason wins, because "the CRF
    never asked" tells a reader more than "unknown".
    """
    states: dict[str, list[str]] = {}
    for record in chain:
        for name, field in record.fields().items():
            states.setdefault(name, []).append(field.collection_state)

    summary = {
        name: min(values, key=lambda s: _STATE_PRIORITY.index(s)
                  if s in _STATE_PRIORITY else len(_STATE_PRIORITY))
        for name, values in states.items()
    }

    # Objective values are not a CRF column, but a rule that asks for one still
    # needs to tell "measured and not low" from "never measured".
    tests = {lab.test for record in chain for lab in record.labs}
    for test in sorted(tests):
        summary[f"labs.{test}"] = "collected"
    if "labs.GLUCOSE" not in summary:
        summary["labs.GLUCOSE"] = "unknown"
    summary["symptoms"] = (
        "collected"
        if any(r.symptoms_assessed.value is True for r in chain)
        else "unknown"
    )
    return dict(sorted(summary.items()))


def _field_notes(chain: Sequence[CanonicalAERecord]) -> dict[str, str]:
    notes: dict[str, str] = {}
    for record in chain:
        for name, field in record.fields().items():
            if field.note and name not in notes:
                notes[name] = field.note
    return dict(sorted(notes.items()))


def _dedupe_spans(spans: Sequence[Span]) -> list[Span]:
    seen: set[tuple] = set()
    out: list[Span] = []
    for span in spans:
        if span.key() not in seen:
            seen.add(span.key())
            out.append(span)
    return sorted(out, key=lambda s: (s.field, s.doc_id, s.start, s.end))


def reconcile_records(
    records: Iterable[CanonicalAERecord],
    catalog: ConceptCatalog,
    semantics: CollectionSemantics,
) -> list[CanonicalAEEpisode]:
    return EpisodeReconciler(catalog, semantics).reconcile(records)
