"""The only callable surface the agent has.

Four functions.  Each is validated code that the agent invokes with a
schema-checked specification; none of them is written or modified by a model.
Anything the agent reports has to have come through one of these.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ..models import CaseAssignment, PhenotypeQuerySpec
from ..pipeline import Pipeline

TOOLS = ("cohort", "retrieve", "evaluate", "summarise")


@dataclass
class AgentTools:
    pipeline: Pipeline

    # -- cohort -------------------------------------------------------------

    def cohort(self, studies: Sequence[str] | None = None) -> dict[str, Any]:
        """The denominator: every subject in scope, before any rule is applied."""
        pairs = self.pipeline.cohort(studies)
        by_study: dict[str, int] = {}
        for _subject, study in pairs:
            by_study[study] = by_study.get(study, 0) + 1
        return {
            "subjects": len(pairs),
            "studies": dict(sorted(by_study.items())),
            "snapshot_id": self.pipeline.snapshot_id,
        }

    # -- retrieve -----------------------------------------------------------

    def retrieve(self, spec: PhenotypeQuerySpec, top_k: int | None = None) -> dict[str, Any]:
        """Records matching the spec, with their spans."""
        result = self.pipeline.retrieve(
            concept=spec.concept,
            assertion=list(spec.assertion) or None,
            evidence_state=list(spec.evidence_state) or None,
            window=tuple(spec.window) if spec.window else None,
            anchor=spec.anchor,
            studies=list(spec.studies) or None,
            definition_id=spec.definition_id,
            definition_version=spec.definition_version,
            mode=spec.retrieval_mode,
            top_k=top_k or spec.top_k,
        )
        return result.to_dict()

    # -- evaluate -----------------------------------------------------------

    def evaluate(self, spec: PhenotypeQuerySpec) -> dict[str, Any]:
        """Run the named definition version and return its assignments."""
        definition = self.pipeline.definition(
            spec.definition_id, spec.definition_version
        )
        assignments = self.pipeline.evaluate(definition, spec.studies or None)
        return {
            "definition": {
                "id": definition.id,
                "version": definition.version,
                "status": definition.status,
                "hash": definition.definition_hash,
                "label": definition.label,
            },
            "assignments": [a.model_dump(mode="json") for a in assignments],
        }

    # -- summarise ----------------------------------------------------------

    def summarise(
        self,
        assignments: Sequence[CaseAssignment] | Sequence[dict[str, Any]],
        spec: PhenotypeQuerySpec,
    ) -> dict[str, Any]:
        """Counts by state and verdict, pooled and per study.

        The review set is reported as its own count and is never folded into
        the case count. Pooling only explicit events undercounts; counting
        everything manufactures signal; reporting both separately is the only
        defensible option.
        """
        rows = [
            a if isinstance(a, dict) else a.model_dump(mode="json")
            for a in assignments
        ]
        by_verdict: dict[str, int] = {}
        by_state: dict[str, int] = {}
        per_study: dict[str, dict[str, int]] = {}
        by_rule: dict[str, int] = {}
        for row in rows:
            by_verdict[row["verdict"]] = by_verdict.get(row["verdict"], 0) + 1
            by_state[row["evidence_state"]] = by_state.get(row["evidence_state"], 0) + 1
            study = per_study.setdefault(row["study_id"], {})
            study[row["verdict"]] = study.get(row["verdict"], 0) + 1
            rule = row.get("matched_rule_id") or "(no rule matched)"
            by_rule[rule] = by_rule.get(rule, 0) + 1
        return {
            "subjects": len(rows),
            "counts_by_verdict": dict(sorted(by_verdict.items())),
            "counts_by_state": dict(sorted(by_state.items())),
            "counts_by_rule": dict(sorted(by_rule.items())),
            "per_study": {
                study: dict(sorted(counts.items()))
                for study, counts in sorted(per_study.items())
            },
            "primary_case_count": by_verdict.get("case", 0),
            "review_set_count": by_verdict.get("review", 0),
            "studies": list(spec.studies),
        }

    # -- dispatch -----------------------------------------------------------

    def call(self, name: str, **kwargs: Any) -> Any:
        if name not in TOOLS:
            raise ValueError(
                f"{name!r} is not a callable tool. The agent may call only: "
                f"{list(TOOLS)}"
            )
        return getattr(self, name)(**kwargs)
