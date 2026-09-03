"""Executable comparison of two phenotype definitions.

Not a search over descriptions.  Both definitions are evaluated against the same
data snapshot, and the answer is the set of records each one claims, with the
reasons attached.  "These two definitions differ" is a sentence; "these 14
records are cases under v1 and not under v2, and here is the rule that decided
each" is a finding.

A scope is mandatory.  See ``KnowledgeRegistry.require_scope``.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from typing import Any, Sequence

from ..models import CaseAssignment, PhenotypeDefinition
from .registry import ScopeRequired


@dataclass
class DiscordantRecord:
    record_id: str
    subject_id: str
    study_id: str
    verdict_a: str
    verdict_b: str
    deciding_a: str | None
    deciding_b: str | None
    reason_a: str
    reason_b: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "subject_id": self.subject_id,
            "study_id": self.study_id,
            "verdict_a": self.verdict_a,
            "verdict_b": self.verdict_b,
            "deciding_a": self.deciding_a,
            "deciding_b": self.deciding_b,
            "reason_a": self.reason_a,
            "reason_b": self.reason_b,
        }


@dataclass
class DefinitionComparison:
    scope: str
    definition_a: str
    definition_b: str
    snapshot_id: str
    shared: list[str] = _dc_field(default_factory=list)
    gained: list[str] = _dc_field(default_factory=list)
    lost: list[str] = _dc_field(default_factory=list)
    discordant: list[DiscordantRecord] = _dc_field(default_factory=list)

    @property
    def summary_line(self) -> str:
        return (
            f"{self.definition_a} -> {self.definition_b} within scope "
            f"{self.scope!r}: {len(self.shared)} shared, {len(self.gained)} "
            f"gained, {len(self.lost)} lost, {len(self.discordant)} discordant"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "definition_a": self.definition_a,
            "definition_b": self.definition_b,
            "snapshot_id": self.snapshot_id,
            "shared": self.shared,
            "gained": self.gained,
            "lost": self.lost,
            "discordant": [d.to_dict() for d in self.discordant],
            "summary": self.summary_line,
        }


def diff_definitions(
    definition_a: PhenotypeDefinition,
    definition_b: PhenotypeDefinition,
    snapshot_id: str,
    scope: str | None,
    assignments_a: Sequence[CaseAssignment],
    assignments_b: Sequence[CaseAssignment],
) -> DefinitionComparison:
    """Compare two definitions by what they actually claim on one snapshot."""
    if not scope or not scope.strip():
        raise ScopeRequired(
            "diff_definitions requires an explicit scope — a scientific "
            "question or a phenotype family. An unscoped programme-wide sweep "
            "is refused: the capability is for evidence reuse, not for "
            "auditing colleagues' past choices."
        )

    by_a = {a.record_id: a for a in assignments_a}
    by_b = {a.record_id: a for a in assignments_b}
    cases_a = {i for i, a in by_a.items() if a.verdict == "case"}
    cases_b = {i for i, a in by_b.items() if a.verdict == "case"}

    discordant = [
        DiscordantRecord(
            record_id=record_id,
            subject_id=by_a[record_id].subject_id,
            study_id=by_a[record_id].study_id,
            verdict_a=by_a[record_id].verdict,
            verdict_b=by_b[record_id].verdict,
            deciding_a=by_a[record_id].deciding_criterion,
            deciding_b=by_b[record_id].deciding_criterion,
            reason_a=by_a[record_id].reason,
            reason_b=by_b[record_id].reason,
        )
        for record_id in sorted(set(by_a) & set(by_b))
        if by_a[record_id].verdict != by_b[record_id].verdict
    ]

    return DefinitionComparison(
        scope=scope.strip(),
        definition_a=definition_a.key,
        definition_b=definition_b.key,
        snapshot_id=snapshot_id,
        shared=sorted(cases_a & cases_b),
        gained=sorted(cases_b - cases_a),
        lost=sorted(cases_a - cases_b),
        discordant=discordant,
    )
