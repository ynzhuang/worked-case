"""The agent's tool surface: typed, permissioned, and small.

Three properties, each enforced in code rather than documented and hoped for:

**Typed both ways.** Every tool declares an input schema and an output schema.
A call whose arguments do not validate is refused before it runs; a result that
does not validate is refused before it is returned. The agent cannot smuggle a
free-form dictionary past either boundary.

**Permissioned.** Every tool declares a permission, and a session is granted a
set. A tool outside the grant is not callable, and the refusal names the missing
permission rather than failing obscurely.

**No SQL, no writes.** There is no tool that takes a query string, and no tool
that writes to a source record. Source records are immutable; everything above
them is derived and recomputable. A tool that could edit a record would make the
provenance chain a claim rather than a fact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, ValidationError
from pydantic import Field as _PydanticField

from ..models import Verdict
from ..pipeline import Pipeline

Permission = Literal[
    "read_cohort", "read_evidence", "read_exposure", "analyse", "export"
]

ALL_PERMISSIONS: tuple[Permission, ...] = (
    "read_cohort", "read_evidence", "read_exposure", "analyse", "export",
)


class ToolError(RuntimeError):
    """Raised when a call is refused: unknown tool, bad schema, or no permission."""


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ResolveIn(_Model):
    definition_id: str
    version: int | None = None


class ResolveOut(_Model):
    id: str
    version: int
    status: str
    hash: str
    label: str
    concept: str
    required_attributes: list[dict[str, Any]]


class CohortIn(_Model):
    definition_id: str
    version: int | None = None
    studies: list[str] = _PydanticField(default_factory=list)


class CohortOut(_Model):
    definition: str
    definition_hash: str
    episodes: int
    counts_by_verdict: dict[str, int]
    subjects_by_verdict: dict[str, int]
    attribute_methods: dict[str, int]
    attribute_sources: dict[str, int]
    not_ascertainable_note: str


class EvidenceIn(_Model):
    concept: str | None = None
    location: list[str] = _PydanticField(default_factory=list)
    region: str | None = None
    studies: list[str] = _PydanticField(default_factory=list)
    verdict: list[Verdict] = _PydanticField(default_factory=list)
    window: tuple[int, int] | None = None
    top_k: int = 20


class EvidenceOut(_Model):
    count: int
    usable_as_cohort: bool
    episodes: list[dict[str, Any]]


class ExposureIn(_Model):
    studies: list[str] = _PydanticField(default_factory=list)


class ExposureOut(_Model):
    subjects: int
    with_exposure: int
    anchor_event: str
    offsets_resolved: int
    distribution_by_time_since_exposure: dict[str, int]


class CovariatesIn(_Model):
    studies: list[str] = _PydanticField(default_factory=list)


class CovariatesOut(_Model):
    subjects: int
    arms: dict[str, int]
    sex: dict[str, int]
    countries: dict[str, int]


class CompareIn(_Model):
    definition_id: str
    left: int
    right: int
    scope: str | None = None


class CompareOut(_Model):
    summary: str
    shared: int
    gained: int
    lost: int
    discordant: list[dict[str, Any]]


class OmicsIn(_Model):
    definition_id: str
    version: int | None = None
    studies: list[str] = _PydanticField(default_factory=list)


class OmicsOut(_Model):
    format: str
    rows: list[dict[str, Any]]
    note: str


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    permission: Permission
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    handler: Callable[["AgentServices", BaseModel], dict[str, Any]]
    writes_source_records: bool = False

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "permission": self.permission,
            "input_schema": self.input_model.model_json_schema(),
            "output_schema": self.output_model.model_json_schema(),
            "writes_source_records": self.writes_source_records,
        }


REGISTRY: dict[str, ToolSpec] = {}


def tool(
    name: str, permission: Permission, input_model: type[BaseModel],
    output_model: type[BaseModel], description: str,
):
    def register(handler):
        REGISTRY[name] = ToolSpec(
            name=name, description=description, permission=permission,
            input_model=input_model, output_model=output_model, handler=handler,
        )
        return handler
    return register


class AgentServices:
    """The only surface an agent execution can reach."""

    def __init__(
        self, pipeline: Pipeline, permissions: set[Permission] | None = None
    ):
        self.pipeline = pipeline
        self.permissions: set[Permission] = set(
            permissions if permissions is not None else ALL_PERMISSIONS
        )
        self.calls: list[str] = []

    # -- dispatch -----------------------------------------------------------

    def call(self, name: str, **kwargs: Any) -> dict[str, Any]:
        spec = REGISTRY.get(name)
        if spec is None:
            raise ToolError(
                f"{name!r} is not a registered tool. An execution may call only: "
                f"{sorted(REGISTRY)}"
            )
        if spec.permission not in self.permissions:
            raise ToolError(
                f"{name!r} requires the {spec.permission!r} permission, which "
                f"this session was not granted (it has "
                f"{sorted(self.permissions)})"
            )
        try:
            payload = spec.input_model.model_validate(kwargs)
        except ValidationError as exc:
            raise ToolError(f"{name}: arguments do not validate — {exc}") from exc

        result = spec.handler(self, payload)

        try:
            validated = spec.output_model.model_validate(result)
        except ValidationError as exc:
            raise ToolError(
                f"{name}: the result does not match its declared output schema "
                f"— {exc}"
            ) from exc
        self.calls.append(name)
        return validated.model_dump(mode="json")

    @staticmethod
    def catalogue() -> list[dict[str, Any]]:
        return [REGISTRY[name].schema() for name in sorted(REGISTRY)]


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------


@tool(
    "phenotype.resolve", "read_cohort", ResolveIn, ResolveOut,
    "Resolve a definition id and version to the frozen artifact that will run.",
)
def _resolve(services: AgentServices, payload: ResolveIn) -> dict[str, Any]:
    definition = services.pipeline.definition(payload.definition_id, payload.version)
    return {
        "id": definition.id, "version": definition.version,
        "status": definition.status, "hash": definition.definition_hash,
        "label": definition.label, "concept": definition.concept.primary,
        "required_attributes": [
            {
                "name": r.name, "in": r.allowed,
                "accept_methods": list(r.accept_methods),
                "on_unavailable": r.on_unavailable,
            }
            for r in definition.required_attributes
        ],
    }


@tool(
    "cohort.run", "read_cohort", CohortIn, CohortOut,
    "Evaluate a frozen definition over episodes and return the verdict counts.",
)
def _cohort(services: AgentServices, payload: CohortIn) -> dict[str, Any]:
    import collections

    pipeline = services.pipeline
    definition = pipeline.definition(payload.definition_id, payload.version)
    assignments = pipeline.evaluate(definition, payload.studies or None)
    subjects = pipeline.evaluator(definition).evaluate_subjects(
        [e for e in pipeline.episodes()
         if not payload.studies or e.study_id in set(payload.studies)]
    )
    return {
        "definition": definition.key,
        "definition_hash": definition.definition_hash,
        "episodes": len(assignments),
        "counts_by_verdict": dict(sorted(
            collections.Counter(a.verdict for a in assignments).items()
        )),
        "subjects_by_verdict": dict(sorted(
            collections.Counter(subjects.values()).items()
        )),
        "attribute_methods": dict(sorted(collections.Counter(
            m for a in assignments if a.verdict == "case"
            for m in a.attribute_methods.values()
        ).items())),
        "attribute_sources": dict(sorted(collections.Counter(
            v for a in assignments if a.verdict == "case"
            for v in a.attribute_sources.values()
        ).items())),
        "not_ascertainable_note": (
            "not_ascertainable episodes are reported separately and are neither "
            "cases nor negatives: a required attribute was never recorded and "
            "cannot be recovered, so no reviewer can settle them either"
        ),
    }


@tool(
    "evidence.search", "read_evidence", EvidenceIn, EvidenceOut,
    "Retrieve adjudicated episodes on the precise path. Never a discovery path.",
)
def _evidence(services: AgentServices, payload: EvidenceIn) -> dict[str, Any]:
    from ..retrieval.query import retrieve

    result = retrieve(
        services.pipeline.index(), services.pipeline.catalog,
        concept=payload.concept, location=payload.location or None,
        region=payload.region, studies=payload.studies or None,
        verdict=payload.verdict or None, window=payload.window,
        top_k=payload.top_k,
    )
    body = result.to_dict()
    return {
        "count": body["count"],
        "usable_as_cohort": body["usable_as_cohort"],
        "episodes": body["episodes"],
    }


@tool(
    "exposure.build", "read_exposure", ExposureIn, ExposureOut,
    "Exposure timing per subject, and the distribution of onsets since the anchor.",
)
def _exposure(services: AgentServices, payload: ExposureIn) -> dict[str, Any]:
    from ..trajectory import time_since_exposure

    pipeline = services.pipeline
    trajectories = list(pipeline.trajectories().values())
    if payload.studies:
        allowed = set(payload.studies)
        trajectories = [t for t in trajectories if t.study_id in allowed]
    anchor = pipeline.configs.extraction.default_anchor or "first_exposure"
    return {
        "subjects": len(trajectories),
        "with_exposure": sum(1 for t in trajectories if t.exposures()),
        "anchor_event": anchor,
        "offsets_resolved": sum(
            1 for t in trajectories for e in t.episodes()
            if e.offset_days is not None
        ),
        "distribution_by_time_since_exposure": time_since_exposure(trajectories),
    }


@tool(
    "covariates.build", "read_exposure", CovariatesIn, CovariatesOut,
    "Baseline covariates for the subjects in scope.",
)
def _covariates(services: AgentServices, payload: CovariatesIn) -> dict[str, Any]:
    store = services.pipeline.store
    subjects = [s for s, _study in services.pipeline.cohort(payload.studies or None)]
    demographics = {row["USUBJID"]: row for row in store.rows("dm")}

    def counted(column: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for subject in subjects:
            value = str((demographics.get(subject) or {}).get(column) or "unknown")
            counts[value] = counts.get(value, 0) + 1
        return dict(sorted(counts.items()))

    return {
        "subjects": len(subjects),
        "arms": counted("ARM"),
        "sex": counted("SEX"),
        "countries": counted("COUNTRY"),
    }


@tool(
    "stats.compare", "analyse", CompareIn, CompareOut,
    "Compare two definition versions by the episodes each claims. Scope required.",
)
def _compare(services: AgentServices, payload: CompareIn) -> dict[str, Any]:
    from ..knowledge import diff_definitions

    pipeline = services.pipeline
    a = pipeline.definition(payload.definition_id, payload.left, allow_draft=True)
    b = pipeline.definition(payload.definition_id, payload.right, allow_draft=True)
    comparison = diff_definitions(
        a, b, pipeline.snapshot_id, payload.scope,
        pipeline.evaluate(a), pipeline.evaluate(b),
    )
    body = comparison.to_dict()
    return {
        "summary": body["summary"],
        "shared": len(body["shared"]),
        "gained": len(body["gained"]),
        "lost": len(body["lost"]),
        "discordant": body["discordant"][:20],
    }


@tool(
    "omics.run", "export", OmicsIn, OmicsOut,
    "A case-control export shaped for an existing genetics pipeline.",
)
def _omics(services: AgentServices, payload: OmicsIn) -> dict[str, Any]:
    pipeline = services.pipeline
    definition = pipeline.definition(payload.definition_id, payload.version)
    assignments = pipeline.evaluate(definition, payload.studies or None)
    return {
        "format": "case_control_v1",
        "rows": [
            {
                "subject_id": a.subject_id,
                "study_id": a.study_id,
                "status": 1 if a.verdict == "case" else (
                    0 if a.verdict == "not_case" else None
                ),
                "verdict": a.verdict,
                "definition": f"{a.definition_id}.v{a.definition_version}",
                "definition_hash": a.definition_hash,
                "evidence_route": sorted(set(a.attribute_methods.values())),
            }
            for a in assignments
        ],
        "note": (
            "status is null for review and not_ascertainable subjects: they are "
            "neither cases nor controls until adjudicated, and coding them "
            "either way would put an unadjudicated judgement into someone "
            "else's analysis. The evidence route travels with each row so a "
            "downstream analysis can see which subjects depended on text."
        ),
    }


SERVICES = tuple(sorted(REGISTRY))
