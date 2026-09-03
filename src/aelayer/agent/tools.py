"""The agent's tool surface: typed, permissioned, and small.

Four properties, each enforced in code rather than documented and hoped for:

**Typed both ways.** Every tool declares an input schema and an output schema.
A call whose arguments do not validate is refused before it runs; a result that
does not validate is refused before it is returned. The agent cannot smuggle a
free-form dictionary past either boundary.

**Permissioned.** Every tool declares a permission, and a session is granted a
set. A tool outside the grant is not callable, and the refusal names the
missing permission rather than failing obscurely.

**No SQL surface, no writes.** There is no tool that takes a query string, and
no tool that writes to a source record. Source records are immutable;
everything above them is derived and recomputable. A tool that could edit a
record would make the provenance chain a claim rather than a fact.

**A definition is bound, never invented.** ``phenotype.resolve`` returns the
frozen artifact that will run, and every tool that produces a number takes the
id and version rather than a set of parameters. A question that implies
different parameters is a conflict for a person to settle, not something for a
tool to accommodate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, ValidationError
from pydantic import Field as _PydanticField

from ..models import Assertion, Availability, Method, Verdict
from ..pipeline import Pipeline

Permission = Literal[
    "read_metadata", "read_cohort", "read_evidence", "read_exposure",
    "analyse", "export",
]

ALL_PERMISSIONS: tuple[Permission, ...] = (
    "read_metadata", "read_cohort", "read_evidence", "read_exposure",
    "analyse", "export",
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
    concept_set: list[str]
    dictionary_target: str | None
    modifiers: list[dict[str, Any]]
    temporal: dict[str, Any] | None
    binding_note: str


class SupportabilityIn(_Model):
    modifier: str = "mucosal_involvement"


class SupportabilityOut(_Model):
    modifier: str
    studies: list[dict[str, Any]]
    supported: list[str]
    supported_via_extraction: list[str]
    cannot_ascertain: list[str]
    note: str


class CohortIn(_Model):
    definition_id: str
    version: int | None = None
    studies: list[str] = _PydanticField(default_factory=list)


class CohortOut(_Model):
    definition: str
    definition_hash: str
    records: int
    counts_by_verdict: dict[str, int]
    denominators: list[dict[str, Any]]
    overall: dict[str, Any]
    attribute_methods: dict[str, int]
    attribute_sources: dict[str, int]
    denominator_note: str


class EvidenceIn(_Model):
    concept: str | None = None
    assertion: list[Assertion] = _PydanticField(default_factory=list)
    availability: list[Availability] = _PydanticField(default_factory=list)
    value: list[str] = _PydanticField(default_factory=list)
    method: list[Method] = _PydanticField(default_factory=list)
    studies: list[str] = _PydanticField(default_factory=list)
    verdict: list[Verdict] = _PydanticField(default_factory=list)
    window: tuple[int, int] | None = None
    top_k: int = 20


class EvidenceOut(_Model):
    count: int
    usable_as_cohort: bool
    records: list[dict[str, Any]]
    notes: list[str]


class ExposureIn(_Model):
    studies: list[str] = _PydanticField(default_factory=list)


class ExposureOut(_Model):
    subjects: int
    records: int
    anchor_event: str
    offsets_resolved: int
    offsets_unresolved: int
    distribution_by_time_since_exposure: dict[str, int]
    method_note: str


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


class ExportIn(_Model):
    definition_id: str
    version: int | None = None
    studies: list[str] = _PydanticField(default_factory=list)


class ExportOut(_Model):
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
    "Bind a definition id and version to the frozen artifact that will run.",
)
def _resolve(services: AgentServices, payload: ResolveIn) -> dict[str, Any]:
    definition = services.pipeline.definition(payload.definition_id, payload.version)
    return {
        "id": definition.id, "version": definition.version,
        "status": definition.status, "hash": definition.definition_hash,
        "label": definition.label,
        "concept_set": sorted(definition.concept_set.include),
        "dictionary_target": definition.concept_set.dictionary_target,
        "modifiers": [
            {
                "name": r.name,
                "require_assertion": r.require_assertion,
                "accept_methods": list(r.accept_methods),
                "on_unavailable": r.on_unavailable,
            }
            for r in definition.modifiers
        ],
        "temporal": (
            {
                "anchor": definition.temporal.anchor,
                "min": definition.temporal.minimum,
                "max": definition.temporal.maximum,
            }
            if definition.temporal else None
        ),
        "binding_note": (
            f"{definition.key} is bound for this execution. Its parameters are "
            f"fixed by the frozen file, not by the question: a question that "
            f"implies a different window or a different assertion is a "
            f"conflict for a person to settle, and a new version if they "
            f"decide it should run."
        ),
    }


@tool(
    "study.supportability", "read_metadata", SupportabilityIn, SupportabilityOut,
    "Which studies can answer a question about a modifier, decided on declared "
    "metadata before any patient-level query runs.",
)
def _supportability(
    services: AgentServices, payload: SupportabilityIn
) -> dict[str, Any]:
    rows = services.pipeline.supportability(payload.modifier)
    by_status: dict[str, list[str]] = {
        "supported": [], "supported_via_extraction": [], "cannot_ascertain": [],
    }
    for row in rows:
        by_status[row["status"]].append(row["study_id"])
    return {
        "modifier": payload.modifier,
        "studies": rows,
        **{k: sorted(v) for k, v in by_status.items()},
        "note": (
            "Decided on collection metadata alone; no patient record was read "
            "to produce it. A study that records the modifier nowhere cannot "
            "answer the question, and finding that out by scanning its "
            "patients first would be both slower and worse manners."
        ),
    }


@tool(
    "cohort.run", "read_cohort", CohortIn, CohortOut,
    "Evaluate a bound definition over source records and return verdicts and "
    "denominators.",
)
def _cohort(services: AgentServices, payload: CohortIn) -> dict[str, Any]:
    import collections

    pipeline = services.pipeline
    definition = pipeline.definition(payload.definition_id, payload.version)
    result = pipeline.evaluate(definition, payload.studies or None)
    assignments = result.assignments
    from ..models import DENOMINATOR_NOTE

    return {
        "definition": definition.key,
        "definition_hash": definition.definition_hash,
        "records": len(assignments),
        "counts_by_verdict": dict(sorted(
            collections.Counter(a.verdict for a in assignments).items()
        )),
        "denominators": [d.to_dict() for d in result.denominators()],
        "overall": result.overall().to_dict(),
        "attribute_methods": dict(sorted(collections.Counter(
            m for a in assignments if a.verdict == "case"
            for m in a.attribute_methods.values()
        ).items())),
        "attribute_sources": dict(sorted(collections.Counter(
            v for a in assignments if a.verdict == "case"
            for v in a.attribute_sources.values()
        ).items())),
        "denominator_note": DENOMINATOR_NOTE,
    }


@tool(
    "evidence.search", "read_evidence", EvidenceIn, EvidenceOut,
    "Retrieve adjudicated records on the precise path. Assertion and "
    "availability are separate filters. Never a discovery path.",
)
def _evidence(services: AgentServices, payload: EvidenceIn) -> dict[str, Any]:
    result = services.pipeline.retrieve(
        concept=payload.concept,
        assertion=payload.assertion or None,
        availability=payload.availability or None,
        value=payload.value or None,
        method=payload.method or None,
        studies=payload.studies or None,
        verdict=payload.verdict or None,
        window=payload.window,
        top_k=payload.top_k,
    )
    body = result.to_dict()
    return {
        "count": body["count"],
        "usable_as_cohort": body["usable_as_cohort"],
        "records": body["records"],
        "notes": body["notes"],
    }


@tool(
    "exposure.build", "read_exposure", ExposureIn, ExposureOut,
    "Exposure timing per subject, and the distribution of onsets since the anchor.",
)
def _exposure(services: AgentServices, payload: ExposureIn) -> dict[str, Any]:
    pipeline = services.pipeline
    records = pipeline.records()
    if payload.studies:
        allowed = set(payload.studies)
        records = [r for r in records if r.study_id in allowed]
    anchor = pipeline.configs.extraction.default_anchor or "first_exposure"

    buckets: dict[str, int] = {}
    resolved = unresolved = 0
    for record in records:
        relation = record.exposure_relation
        if not relation.observed or relation.value is None:
            unresolved += 1
            continue
        resolved += 1
        offset = int(relation.value)
        label = (
            "0-7" if offset <= 7 else
            "8-30" if offset <= 30 else
            "31-90" if offset <= 90 else
            "91+" if offset >= 0 else "before exposure"
        )
        buckets[label] = buckets.get(label, 0) + 1
    return {
        "subjects": len({r.subject_id for r in records}),
        "records": len(records),
        "anchor_event": anchor,
        "offsets_resolved": resolved,
        "offsets_unresolved": unresolved,
        "distribution_by_time_since_exposure": dict(sorted(buckets.items())),
        "method_note": (
            "Each offset is AE.AESTDTC minus the anchor exposure date, computed "
            "across domains by governed code and stamped method 'derived'. It "
            "is never model reasoning, and a record whose date cannot be read "
            "is counted as unresolved rather than assigned a zero."
        ),
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
    "Compare two definition versions by the records each claims. Scope required.",
)
def _compare(services: AgentServices, payload: CompareIn) -> dict[str, Any]:
    from ..knowledge import diff_definitions

    pipeline = services.pipeline
    a = pipeline.definition(payload.definition_id, payload.left, allow_draft=True)
    b = pipeline.definition(payload.definition_id, payload.right, allow_draft=True)
    comparison = diff_definitions(
        a, b, pipeline.snapshot_id, payload.scope,
        pipeline.assignments(a), pipeline.assignments(b),
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
    "cohort.export", "export", ExportIn, ExportOut,
    "A case / non-case export, with unascertained subjects left null.",
)
def _export(services: AgentServices, payload: ExportIn) -> dict[str, Any]:
    pipeline = services.pipeline
    definition = pipeline.definition(payload.definition_id, payload.version)
    assignments = pipeline.assignments(definition, payload.studies or None)
    return {
        "format": "case_noncase_v1",
        "rows": [
            {
                "subject_id": a.subject_id,
                "study_id": a.study_id,
                "record_id": a.record_id,
                "status": 1 if a.verdict == "case" else (
                    0 if a.verdict == "non_case" else None
                ),
                "verdict": a.verdict,
                "definition": f"{a.definition_id}.v{a.definition_version}",
                "definition_hash": a.definition_hash,
                "evidence_route": sorted(set(a.attribute_methods.values())),
            }
            for a in assignments
        ],
        "note": (
            "status is null for review and not_ascertainable records: they are "
            "neither cases nor non-cases until adjudicated, and coding them "
            "either way would put an unadjudicated judgement into someone "
            "else's analysis. The evidence route travels with each row so a "
            "downstream analysis can see which subjects depended on text."
        ),
    }


SERVICES = tuple(sorted(REGISTRY))
