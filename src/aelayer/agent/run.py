"""Approval gate and execution.

The compiled spec is returned and execution blocks until it is explicitly
approved.  This is not ceremony: the spec encodes which definition version
runs, which assertion classes count, and which evidence states become cases.
Those are scientific choices, and a person has to make them before any number
is produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..models import Clarification, PhenotypeQuerySpec, RunManifest
from ..pipeline import Pipeline
from ..runs import RunStore, execute
from .compile import CompileResult, compile_question
from .tools import AgentTools


class ApprovalRequired(RuntimeError):
    """Raised when execution is attempted on an unapproved spec."""


@dataclass
class EvidencePackage:
    """What a completed agent run returns.

    Not a number in isolation: the counts, the spans behind them, the exact
    versions that produced them, and the limits on reading them.
    """

    question: str
    spec: PhenotypeQuerySpec
    summary: dict[str, Any]
    retrieval: dict[str, Any]
    definition: dict[str, Any]
    extractor_version: str
    snapshot_id: str
    run_id: str
    results_hash: str
    limitations: list[str]
    contributing_spans: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "spec": self.spec.model_dump(mode="json"),
            "summary": self.summary,
            "retrieval": self.retrieval,
            "definition": self.definition,
            "extractor_version": self.extractor_version,
            "snapshot_id": self.snapshot_id,
            "run_id": self.run_id,
            "results_hash": self.results_hash,
            "contributing_spans": self.contributing_spans,
            "limitations": self.limitations,
        }


@dataclass
class AgentSession:
    """One question, its compiled spec, and its approval state."""

    pipeline: Pipeline
    question: str
    backend: str = "deterministic"
    result: CompileResult | None = None
    approved: bool = False

    def compile(self, *, allow_draft: bool = False) -> CompileResult:
        self.result = compile_question(
            self.question, self.pipeline, backend=self.backend, allow_draft=allow_draft
        )
        self.approved = False
        return self.result

    @property
    def spec(self) -> PhenotypeQuerySpec | None:
        return self.result.spec if self.result else None

    @property
    def clarification(self) -> Clarification | None:
        return self.result.clarification if self.result else None

    def approve(self) -> None:
        if self.result is None:
            raise ApprovalRequired("nothing has been compiled to approve")
        if self.result.spec is None:
            raise ApprovalRequired(
                "the question produced a clarification request, not a "
                "specification. Answer the clarification and compile again."
            )
        self.approved = True

    def execute(
        self, *, run_store: RunStore | None = None, save: bool = True
    ) -> tuple[EvidencePackage, RunManifest]:
        if self.result is None or self.result.spec is None:
            raise ApprovalRequired("no approved specification to execute")
        if not self.approved:
            raise ApprovalRequired(
                "execution is blocked until the compiled specification is "
                "explicitly approved. Review the spec, then approve it."
            )

        spec = self.result.spec
        tools = AgentTools(self.pipeline)

        cohort = tools.call("cohort", studies=list(spec.studies) or None)
        evaluation = tools.call("evaluate", spec=spec)
        summary = tools.call(
            "summarise", assignments=evaluation["assignments"], spec=spec
        )
        retrieval = tools.call("retrieve", spec=spec)

        definition = self.pipeline.definition(
            spec.definition_id, spec.definition_version
        )
        manifest = execute(
            self.pipeline,
            definition,
            studies=list(spec.studies) or None,
            spec_extra={
                "question": spec.question,
                "backend": spec.backend,
                "assertion": list(spec.assertion),
                "evidence_state": list(spec.evidence_state),
            },
        )
        if save:
            (run_store or RunStore()).save(manifest)

        spans: list[dict[str, Any]] = []
        for assignment in manifest.assignments:
            if assignment.verdict != "case":
                continue
            for span in assignment.evidence_spans[:2]:
                spans.append(
                    {
                        "subject_id": assignment.subject_id,
                        "doc_id": span.doc_id,
                        "field": span.field,
                        "start": span.start,
                        "end": span.end,
                        "text": span.text,
                        "rule": assignment.matched_rule_id,
                    }
                )
            if len(spans) >= 25:
                break

        summary["cohort"] = cohort
        package = EvidencePackage(
            question=self.question,
            spec=spec,
            summary=summary,
            retrieval={
                "count": retrieval["count"],
                "expanded_terms": retrieval["expanded_terms"],
                "negation_false_positive_rate": retrieval[
                    "negation_false_positive_rate"
                ],
                "records": retrieval["records"][:10],
            },
            definition=evaluation["definition"],
            extractor_version=self.pipeline.extractor_version,
            snapshot_id=self.pipeline.snapshot_id,
            run_id=manifest.run_id,
            results_hash=manifest.results_hash,
            limitations=manifest.limitations,
            contributing_spans=spans,
        )
        return package, manifest
