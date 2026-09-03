"""Executing a compiled specification.

There is no approval gate. Approving a specification you cannot independently
evaluate is ceremony: the reviewer sees a plan, not a result, and clicking
approve does not make the result checkable. What makes it checkable is the trace
returned with it, which anyone can follow back to the text a site wrote, after
the fact.

The agent computes nothing. Every number in the package comes from a registered,
typed, permissioned tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from typing import Any

from ..models import Conflict, Manifest, QuerySpec, Trace
from ..pipeline import Pipeline
from ..runs import ManifestStore, execute
from .compile import CompileResult, compile_question
from .tools import AgentServices, Permission
from .trace import trace_number


class ConflictUnresolved(RuntimeError):
    """Raised when execution is attempted on an underdetermined question."""


#: Kept so callers written against the earlier name keep working.
ClarificationRequired = ConflictUnresolved


@dataclass
class EvidencePackage:
    """What a completed run returns.

    Not a number in isolation: the counts, the routes the evidence came by, the
    versions that produced them, the limits on reading them, and a trace that
    reaches source text.
    """

    question: str
    spec: QuerySpec
    cohort: dict[str, Any]
    supportability: dict[str, Any]
    exposure: dict[str, Any]
    covariates: dict[str, Any]
    definition: dict[str, Any]
    versions: dict[str, Any]
    manifest_id: str
    results_hash: str
    output_pointer: str
    limitations: list[str]
    trace: Trace | None = None
    tools_called: list[str] = _dc_field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "spec": self.spec.model_dump(mode="json"),
            "cohort": self.cohort,
            "supportability": self.supportability,
            "exposure": self.exposure,
            "covariates": self.covariates,
            "definition": self.definition,
            "versions": self.versions,
            "manifest_id": self.manifest_id,
            "results_hash": self.results_hash,
            "output_pointer": self.output_pointer,
            "tools_called": self.tools_called,
            "limitations": self.limitations,
            "trace": self.trace.model_dump(mode="json") if self.trace else None,
            "traceable": bool(self.trace and self.trace.complete),
        }


@dataclass
class AgentSession:
    pipeline: Pipeline
    question: str
    backend: str = "deterministic"
    permissions: set[Permission] | None = None
    result: CompileResult | None = None

    def compile(self, *, allow_draft: bool = False) -> CompileResult:
        self.result = compile_question(
            self.question, self.pipeline, backend=self.backend,
            allow_draft=allow_draft,
        )
        return self.result

    @property
    def spec(self) -> QuerySpec | None:
        return self.result.spec if self.result else None

    @property
    def conflict(self) -> Conflict | None:
        return self.result.conflict if self.result else None

    def execute(
        self, *, manifest_store: ManifestStore | None = None, save: bool = True
    ) -> tuple[EvidencePackage, Manifest]:
        if self.result is None:
            self.compile()
        if self.result.spec is None:
            raise ConflictUnresolved(
                "the question conflicts with the definition it names, or "
                "leaves a rule underdetermined, so nothing was executed. The "
                "agent does not override a bound definition to accommodate a "
                "question: settle the conflict, or write a new version, and "
                "compile again."
            )
        spec = self.result.spec
        services = AgentServices(self.pipeline, self.permissions)

        definition_body = services.call(
            "phenotype.resolve",
            definition_id=spec.definition_id, version=spec.definition_version,
        )
        # The metadata screen runs *before* any patient-level query. A study
        # that records the required modifier nowhere cannot answer, and that is
        # knowable from the collection profile alone.
        definition = self.pipeline.definition(
            spec.definition_id, spec.definition_version
        )
        supportability = {}
        for requirement in definition.modifiers:
            supportability[requirement.name] = services.call(
                "study.supportability", modifier=requirement.name
            )
        cohort = services.call(
            "cohort.run", definition_id=spec.definition_id,
            version=spec.definition_version, studies=list(spec.studies),
        )
        exposure = services.call("exposure.build", studies=list(spec.studies))
        covariates = services.call("covariates.build", studies=list(spec.studies))

        manifest, assignments = execute(
            self.pipeline, definition,
            studies=list(spec.studies) or None,
            question=self.question,
            actor="agent",
            specification={
                "backend": spec.backend,
                "verdicts": list(spec.verdicts),
                "accept_methods": list(spec.accept_methods),
                "tools": services.calls,
            },
            manifest_store=manifest_store,
            save=save,
        )

        chain = trace_number(
            number=cohort["counts_by_verdict"].get("case", 0),
            label="case count",
            manifest=manifest,
            assignments=assignments,
            records=self.pipeline.records(),
        )
        return EvidencePackage(
            question=self.question,
            spec=spec,
            cohort=cohort,
            supportability=supportability,
            exposure=exposure,
            covariates=covariates,
            definition=definition_body,
            versions=self.pipeline.versions(),
            manifest_id=manifest.manifest_id,
            results_hash=manifest.results_hash,
            output_pointer=manifest.output_pointer,
            limitations=manifest.limitations,
            trace=chain,
            tools_called=list(services.calls),
        ), manifest
