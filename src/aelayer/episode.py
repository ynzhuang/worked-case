"""Episode reconciliation.

An evolving event may be recorded as several source records under some
collection conventions and as one under others. Deriving an episode lets an
analysis reason about the clinical event; keeping the records untouched
underneath means the derivation can be redone when the assumption changes.

**Reconciliation is a declared assumption, not a solved problem.** The default
rule — same subject, same concept, intervals overlapping or close — is wrong for
recurrent conditions, so the catalogue declares ``recurrence_expected`` per
concept and anything the rules cannot settle is flagged rather than quietly
resolved.

Promoting an attribute from records to the episode keeps its route. Where two
records disagree, the more authoritative route wins — a value the CRF settled
outranks one read out of prose — and the losing value is recorded in the note
rather than discarded silently.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .anchors import AnchorResolver
from .catalog import ConceptCatalog
from .models import (
    Attribute,
    CanonicalAEEpisode,
    CanonicalAERecord,
    LinkageRule,
    Span,
)
from .profiles import StudyProfiles

#: How much a route is trusted when two records disagree. Not a claim that
#: extraction is unreliable — a claim that a value the study itself coded is the
#: study's own answer, and outranks a reading of its prose.
METHOD_RANK = {"direct": 3, "normalized": 2, "extracted": 1, None: 0}


@dataclass(frozen=True)
class LinkageDecision:
    attach: bool
    rule: LinkageRule
    confidence: float
    review_required: bool
    note: str


@dataclass
class ReconciliationConfig:
    overlap_tolerance_days: int = 3
    recurrence_gap_days: int = 7
    confidence: dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.confidence = self.confidence or {}

    @classmethod
    def from_catalog(cls, catalog: ConceptCatalog) -> "ReconciliationConfig":
        body = catalog.episode_reconciliation or {}
        return cls(
            overlap_tolerance_days=int(body.get("overlap_tolerance_days", 3)),
            recurrence_gap_days=int(body.get("recurrence_gap_days", 7)),
            confidence=dict(body.get("confidence") or {}),
        )

    def score(self, rule: str, default: float) -> float:
        return float(self.confidence.get(rule, default))


class EpisodeReconciler:
    def __init__(
        self,
        catalog: ConceptCatalog,
        profiles: StudyProfiles,
        config: ReconciliationConfig | None = None,
        anchor_resolver: AnchorResolver | None = None,
        default_anchor: str | None = None,
    ):
        self.catalog = catalog
        self.profiles = profiles
        self.config = config or ReconciliationConfig.from_catalog(catalog)
        self.resolver = anchor_resolver
        self.default_anchor = default_anchor

    # -- the decision -------------------------------------------------------

    def decide(
        self, previous: CanonicalAERecord, candidate: CanonicalAERecord,
        concept: str | None,
    ) -> LinkageDecision:
        """Does ``candidate`` continue the episode ``previous`` belongs to?"""
        if candidate.continuation_of == previous.source_record_id:
            return LinkageDecision(
                True, "explicit_continuation",
                self.config.score("explicit_continuation", 1.0), False,
                f"the CRF declares continuation of {previous.source_record_id}",
            )

        start = candidate.onset.value
        previous_start = previous.onset.value
        previous_end = previous.end.value

        if start is None or previous_start is None:
            return LinkageDecision(
                False, "single_record", 0.4, True,
                "onset is not resolvable on one of the records, so temporal "
                "linkage cannot be evaluated either way",
            )

        profile = self.profiles.for_study(candidate.study_id)
        if profile.splits_on_severity_change():
            same_window = abs((start - previous_start).days) <= max(
                self.config.overlap_tolerance_days, 7
            )
            if same_window:
                return LinkageDecision(
                    True, "declared_convention",
                    self.config.score("declared_convention", 0.95), False,
                    f"{profile.profile_id} declares that it splits an evolving "
                    f"event across records on severity change",
                )

        if previous_end is not None and start <= previous_end:
            return LinkageDecision(
                True, "temporal_overlap",
                self.config.score("temporal_overlap", 0.9), False,
                f"onset {start.isoformat()} falls inside the previous record's "
                f"interval, which ends {previous_end.isoformat()}",
            )

        reference = previous_end or previous_start
        gap = (start - reference).days
        recurs = self._recurrence_expected(concept)
        if gap <= self.config.overlap_tolerance_days and not recurs:
            return LinkageDecision(
                True, "gap_within_tolerance",
                self.config.score("gap_within_tolerance", 0.75), True,
                f"a {gap}-day gap, and {concept} is not a concept where "
                f"recurrence is expected, so the records are read as one "
                f"condition changing — flagged because the rule is a judgement",
            )
        return LinkageDecision(
            False, "recurrence_split",
            self.config.score("recurrence_split", 0.9), False,
            f"a {gap}-day gap"
            + (
                f", and recurrence is expected for {concept}, so these are "
                f"separate episodes"
                if recurs else ", which is beyond the tolerance for merging"
            ),
        )

    def _recurrence_expected(self, concept: str | None) -> bool:
        if not concept:
            return True
        try:
            return self.catalog.concept(concept).recurrence_expected
        except Exception:
            return True

    # -- assembly -----------------------------------------------------------

    def reconcile(
        self, records: Iterable[CanonicalAERecord]
    ) -> list[CanonicalAEEpisode]:
        records = list(records)
        component = _continuation_components(records)

        # A continuation record often carries nothing that standardizes, so it
        # inherits the concept of the chain the CRF put it in rather than being
        # stranded on its own.
        component_concept: dict[str, str | None] = {}
        for record in records:
            root = component[record.source_record_id]
            if component_concept.get(root) is None:
                component_concept[root] = record.standardized_concept

        grouped: dict[tuple[str, str, str], list[CanonicalAERecord]] = {}
        for record in records:
            root = component[record.source_record_id]
            concept = record.standardized_concept or component_concept.get(root)
            grouped.setdefault(
                (record.study_id, record.subject_id, concept or f"UNMAPPED::{root}"),
                [],
            ).append(record)

        episodes: list[CanonicalAEEpisode] = []
        for (study_id, subject_id, concept_key), chain in sorted(grouped.items()):
            chain.sort(key=lambda r: (r.onset.value or _dt.date.min, r.source_record_id))
            concept = None if concept_key.startswith("UNMAPPED::") else concept_key
            current: list[CanonicalAERecord] = [chain[0]]
            decision = LinkageDecision(
                True, "single_record", self.config.score("single_record", 1.0),
                False, "a single source record",
            )
            for record in chain[1:]:
                verdict = self.decide(current[-1], record, concept)
                if verdict.attach:
                    current.append(record)
                    decision = verdict
                else:
                    episodes.append(self._build(
                        study_id, subject_id, concept, current, decision,
                        len(episodes),
                    ))
                    current = [record]
                    decision = LinkageDecision(
                        True, "single_record",
                        self.config.score("single_record", 1.0), False,
                        f"a new episode: {verdict.note}",
                    )
            episodes.append(
                self._build(study_id, subject_id, concept, current, decision,
                            len(episodes))
            )
        return sorted(episodes, key=lambda e: e.episode_id)

    def _build(
        self, study_id: str, subject_id: str, concept: str | None,
        chain: Sequence[CanonicalAERecord], decision: LinkageDecision, index: int,
    ) -> CanonicalAEEpisode:
        first, last = chain[0], chain[-1]
        profile = self.profiles.for_study(study_id)
        episode_id = (
            f"{subject_id}::{concept or 'UNMAPPED'}::{first.source_record_id}"
        )
        spans: list[Span] = []
        for record in chain:
            spans.extend(record.spans())

        offset, anchor_event, anchor_date = self._anchor(
            subject_id, first.onset.value
        )

        return CanonicalAEEpisode(
            episode_id=episode_id,
            study_id=study_id,
            subject_id=subject_id,
            profile=profile.profile_id,
            standardized_concept=concept,
            episode_start=_carry(first.onset),
            episode_end=_carry(last.end),
            source_record_ids=[r.source_record_id for r in chain],
            location=_promote(chain, "location"),
            laterality=_promote(chain, "laterality"),
            pattern=_promote(chain, "pattern"),
            severity=_promote(chain, "severity"),
            seriousness=_promote(chain, "seriousness"),
            relatedness=_promote(chain, "relatedness"),
            outcome=_carry(last.outcome),
            action_taken=_promote(chain, "action_taken"),
            coded_events=sorted(
                {r.coded_event.value for r in chain if r.coded_event.populated}
            ),
            reported_terms=sorted(
                {r.reported_term.value for r in chain if r.reported_term.populated}
            ),
            dictionary_versions=sorted(
                {r.dictionary_version for r in chain if r.dictionary_version}
            ),
            severity_trajectory=[
                (r.onset.value, r.severity.value)
                for r in chain if r.severity.populated
            ],
            onset_offset_days=offset,
            anchor_event=anchor_event,
            anchor_date=anchor_date,
            linked_evidence=_dedupe(spans),
            linkage_rule=decision.rule if len(chain) > 1 else "single_record",
            linkage_confidence=(
                decision.confidence if len(chain) > 1
                else self.config.score("single_record", 1.0)
            ),
            linkage_review_required=decision.review_required and len(chain) > 1,
            linkage_note=decision.note,
            episode_provenance={
                "record_count": len(chain),
                "source_record_ids": [r.source_record_id for r in chain],
                "normalizer_versions": sorted({r.normalizer_version for r in chain}),
                "extractor_versions": sorted(
                    {r.extractor_version for r in chain if r.extractor_version}
                ),
                "profile": profile.profile_id,
            },
        )

    def _anchor(
        self, subject_id: str, start: _dt.date | None
    ) -> tuple[Attribute[int], str | None, _dt.date | None]:
        """The offset from the study's anchor, so a window can be applied.

        Unresolvable is a state, not a zero: with no exposure record to measure
        from, the offset stays unknown and says why rather than defaulting to a
        number a filter would silently trust.
        """
        if self.resolver is None or self.default_anchor is None:
            return Attribute[int].unavailable(
                "unknown", note="no anchor configuration to resolve an offset"
            ), None, None
        if start is None:
            return Attribute[int].unavailable(
                "unknown", note="the episode has no resolvable start"
            ), None, None
        hit = self.resolver.resolve(
            subject_id, self.default_anchor, onset_date=start
        )
        if hit is None:
            return Attribute[int].unavailable(
                "unknown",
                note=f"no {self.default_anchor} occurrence in this subject's "
                     f"exposure record",
            ), None, None
        return (
            Attribute[int](
                value=(start - hit.date).days, availability="collected",
                method="normalized", source="derived",
                source_variable=f"EX.{self.default_anchor}",
                note=f"{self.default_anchor} on {hit.date.isoformat()} ({hit.detail})",
            ),
            self.default_anchor,
            hit.date,
        )


# --------------------------------------------------------------------------


def _continuation_components(
    records: Sequence[CanonicalAERecord],
) -> dict[str, str]:
    """Union-find over declared continuation chains.

    A chain the CRF itself declares outranks everything, including whether the
    records in it could be standardized — otherwise two records the study linked
    land in different episodes because neither coded to a catalogue term.
    """
    parent: dict[str, str] = {r.source_record_id: r.source_record_id for r in records}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for record in records:
        target = record.continuation_of
        if target and target in parent:
            a, b = find(record.source_record_id), find(target)
            if a != b:
                parent[max(a, b)] = min(a, b)
    return {r.source_record_id: find(r.source_record_id) for r in records}


def _promote(
    chain: Sequence[CanonicalAERecord], name: str
) -> Attribute[Any]:
    """Lift one attribute from the records to the episode, keeping its route.

    The most authoritative populated value wins. Where a less authoritative
    route disagreed, the note says so: a disagreement between the CRF and the
    investigator's own words is a finding, not noise to be dropped.
    """
    populated = [
        (r, r.attribute(name)) for r in chain
        if r.attribute(name) and r.attribute(name).populated
    ]
    if not populated:
        # Keep the most informative reason for the emptiness.
        for record in chain:
            attribute = record.attribute(name)
            if attribute and attribute.availability != "unknown":
                return attribute.model_copy(deep=True)
        attribute = chain[0].attribute(name)
        return (
            attribute.model_copy(deep=True) if attribute
            else Attribute[Any].unavailable("unknown")
        )

    populated.sort(
        key=lambda pair: (
            METHOD_RANK.get(pair[1].method, 0), pair[1].confidence or 0.0
        ),
        reverse=True,
    )
    best = populated[0][1]
    others = {p[1].value for p in populated[1:] if p[1].value != best.value}
    if not others:
        return best.model_copy(deep=True)
    note = (
        f"{best.note + '; ' if best.note else ''}records in this episode "
        f"disagree: {sorted(others)} also recorded, and the "
        f"{best.method} value from {best.source_variable} was taken as the "
        f"study's own answer"
    )
    return best.model_copy(deep=True, update={"note": note})


def _carry(attribute: Attribute[Any]) -> Attribute[Any]:
    return attribute.model_copy(deep=True)


def _dedupe(spans: Sequence[Span]) -> list[Span]:
    seen: dict[tuple, Span] = {}
    for span in spans:
        seen.setdefault(span.key(), span)
    return sorted(seen.values(), key=lambda s: (s.field, s.doc_id, s.start, s.end))


def reconcile_records(
    records: Iterable[CanonicalAERecord], catalog: ConceptCatalog,
    profiles: StudyProfiles,
) -> list[CanonicalAEEpisode]:
    return EpisodeReconciler(catalog, profiles).reconcile(records)
