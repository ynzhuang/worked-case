"""Execution and the evidence package.

There is no approval gate.  Approving a specification you cannot independently
evaluate is ceremony: the reviewer sees a plan, not a result.  What replaces it
is traceability — every number the package reports can be followed back through
the analysis, the cohort, the definition version, the episodes and the source
records to the text a site wrote.

The specification is still compiled, still returned, and still inspectable.  It
is just not treated as though clicking past it were a control.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from typing import Any

from ..models import Clarification, Manifest, PhenotypeQuerySpec, Trace
from ..pipeline import Pipeline
from ..runs import ManifestStore, execute
from .compile import CompileResult, compile_question
from .tools import SERVICES, AgentServices
from .trace import trace_number


class ClarificationRequired(RuntimeError):
    """Raised when execution is attempted on an underdetermined question."""


@dataclass
class EvidencePackage:
    """What a completed run returns.

    Not a number in isolation: the counts, the versions that produced them, the
    limits on reading them, and a trace that reaches source text.
    """

    question: str
    spec: PhenotypeQuerySpec
    summary: dict[str, Any]
    statistics: dict[str, Any]
    cohort_context: dict[str, Any]
    definition: dict[str, Any]
    versions: dict[str, Any]
    manifest_id: str
    results_hash: str
    output_pointer: str
    limitations: list[str]
    trace: Trace | None = None
    services_called: list[str] = _dc_field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "spec": self.spec.model_dump(mode="json"),
            "summary": self.summary,
            "statistics": self.statistics,
            "cohort_context": self.cohort_context,
            "definition": self.definition,
            "versions": self.versions,
            "manifest_id": self.manifest_id,
            "results_hash": self.results_hash,
            "output_pointer": self.output_pointer,
            "services_called": self.services_called,
            "limitations": self.limitations,
            "trace": self.trace.model_dump(mode="json") if self.trace else None,
            "traceable": bool(self.trace and self.trace.complete),
        }


@dataclass
class AgentSession:
    pipeline: Pipeline
    question: str
    backend: str = "deterministic"
    result: CompileResult | None = None

    def compile(self, *, allow_draft: bool = False) -> CompileResult:
        self.result = compile_question(
            self.question, self.pipeline, backend=self.backend,
            allow_draft=allow_draft,
        )
        return self.result

    @property
    def spec(self) -> PhenotypeQuerySpec | None:
        return self.result.spec if self.result else None

    @property
    def clarification(self) -> Clarification | None:
        return self.result.clarification if self.result else None

    def execute(
        self, *, manifest_store: ManifestStore | None = None, save: bool = True
    ) -> tuple[EvidencePackage, Manifest]:
        if self.result is None:
            self.compile()
        if self.result.spec is None:
            raise ClarificationRequired(
                "the question leaves a rule underdetermined, so nothing was "
                "executed. Answer the clarification and compile again."
            )

        spec = self.result.spec
        services = AgentServices(self.pipeline)
        called: list[str] = []

        evidence = services.call("cohort_evidence", spec=spec)
        called.append("cohort_evidence")
        assignments = evidence["assignments"]

        summary = services.call("summarise", assignments=assignments, spec=spec)
        called.append("summarise")
        statistics = services.call(
            "statistical_analysis", assignments=assignments, spec=spec
        )
        called.append("statistical_analysis")
        context = services.call("exposure_and_covariates", spec=spec)
        called.append("exposure_and_covariates")

        definition = self.pipeline.definition(
            spec.definition_id, spec.definition_version
        )
        manifest, live_assignments = execute(
            self.pipeline, definition,
            studies=list(spec.studies) or None,
            question=self.question,
            actor="agent",
            specification={
                "backend": spec.backend,
                "evidence_state": list(spec.evidence_state),
                "retrieval_mode": spec.retrieval_mode,
            },
            manifest_store=manifest_store,
            save=save,
        )

        chain = trace_number(
            number=summary["primary_case_count"],
            label="primary case count",
            manifest=manifest,
            assignments=live_assignments,
            episodes=self.pipeline.episodes(),
            records=self.pipeline.records(),
        )

        package = EvidencePackage(
            question=self.question,
            spec=spec,
            summary=summary,
            statistics=statistics,
            cohort_context=context,
            definition=evidence["definition"],
            versions=self.pipeline.versions(),
            manifest_id=manifest.manifest_id,
            results_hash=manifest.results_hash,
            output_pointer=manifest.output_pointer,
            limitations=manifest.limitations,
            trace=chain,
            services_called=called,
        )
        return package, manifest
