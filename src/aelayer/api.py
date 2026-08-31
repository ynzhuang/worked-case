"""HTTP API.

Thin.  Every endpoint delegates to the same pipeline the CLI uses, so the API
and the CLI cannot disagree about which definition version produced a number.

Nothing here mutates a frozen definition.  The candidate endpoint renders a
proposed next version as YAML and hands it back; writing it to the repository
is a deliberate act performed outside this process.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal, Optional

import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import paths
from .catalog import ConfigError
from .ingest import IngestError
from .models import PhenotypeQuerySpec
from .phenotype.loader import DefinitionError, diff_definitions
from .pipeline import Pipeline
from .runs import ReplayError, RunStore, execute

app = FastAPI(
    title="Adverse event evidence layer",
    version="0.1.0",
    description=(
        "Provenance-bearing event objects and versioned phenotype definitions "
        "over completed-trial adverse event data. All data is synthetic. The "
        "extractor is a configurable rule and lexicon baseline, not a trained "
        "clinical NLP model."
    ),
)

_SESSIONS: dict[str, Any] = {}


@lru_cache(maxsize=1)
def _pipeline_singleton() -> Pipeline:
    return Pipeline.load(store_path=paths.STORE_DB)


def get_pipeline() -> Pipeline:
    try:
        return _pipeline_singleton()
    except (IngestError, ConfigError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def reset_pipeline() -> None:
    _pipeline_singleton.cache_clear()


# --------------------------------------------------------------------------
# Request bodies
# --------------------------------------------------------------------------


class ExtractRequest(BaseModel):
    refresh: bool = Field(
        default=False, description="Re-extract even if a fresh store exists."
    )


class EvaluateRequest(BaseModel):
    definition_id: str = "te_symptomatic_hypoglycemia"
    version: Optional[int] = None
    studies: list[str] = Field(default_factory=list)
    allow_draft: bool = False
    save: bool = True


class CompileRequest(BaseModel):
    question: str
    backend: Literal["deterministic", "llm"] = "deterministic"


class AgentRunRequest(BaseModel):
    question: str
    backend: Literal["deterministic", "llm"] = "deterministic"
    approved: bool = Field(
        default=False,
        description=(
            "Must be true. The compiled specification is returned by "
            "/agent/compile and execution is blocked until it is approved."
        ),
    )


class CandidateRequest(BaseModel):
    definition_id: str
    base_version: int
    changes: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Dotted paths to new values, e.g. "
            '{"window.max": 7, "evidence_rules.supported.lab.value": 54}'
        ),
    )
    author: str = "ui"


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------


@app.get("/api/summary")
def summary() -> dict[str, Any]:
    pipeline = get_pipeline()
    return {
        **pipeline.summary(),
        "snapshot_id": pipeline.snapshot_id,
        "studies": pipeline.store.studies(),
        "study_conventions": pipeline.store.manifest.get("studies", {}),
        "concepts": sorted(pipeline.catalog.concepts),
        "concept_groups": pipeline.catalog.concept_groups,
        "notice": (
            "Synthetic data only. The extractor is a configurable rule and "
            "lexicon baseline, not a trained clinical NLP model."
        ),
    }


@app.post("/extract")
def extract(request: ExtractRequest) -> dict[str, Any]:
    pipeline = get_pipeline()
    events = pipeline.events(refresh=request.refresh)
    index = pipeline.index(refresh=request.refresh)
    violations = [
        {"event_id": e.event_id, "fields": e.missing_provenance()}
        for e in events
        if not e.has_full_provenance()
    ]
    return {
        "events": len(events),
        "documents": index.meta().document_count,
        "extractor_version": pipeline.extractor_version,
        "snapshot_id": pipeline.snapshot_id,
        "provenance_violations": violations,
    }


@app.get("/api/documents")
def documents(
    study: Optional[str] = None, limit: int = Query(50, le=500)
) -> dict[str, Any]:
    pipeline = get_pipeline()
    rows = sorted(pipeline.store.narratives.values(), key=lambda n: n.doc_id)
    if study:
        rows = [n for n in rows if n.study_id == study]
    return {
        "count": len(rows),
        "documents": [
            {
                "doc_id": n.doc_id,
                "study_id": n.study_id,
                "subject_id": n.subject_id,
                "preview": n.text[:120],
            }
            for n in rows[:limit]
        ],
    }


@app.get("/api/documents/{doc_id}")
def document(doc_id: str) -> dict[str, Any]:
    """Source text beside every event object extracted from it."""
    pipeline = get_pipeline()
    narrative = pipeline.store.narratives.get(doc_id)
    if narrative is None:
        raise HTTPException(status_code=404, detail=f"no document {doc_id!r}")
    events = [e for e in pipeline.events() if e.doc_id == doc_id]
    return {
        "doc_id": doc_id,
        "study_id": narrative.study_id,
        "subject_id": narrative.subject_id,
        "header": narrative.header,
        "text": narrative.full_text,
        "events": [e.model_dump(mode="json") for e in events],
        "extractor_version": pipeline.extractor_version,
    }


@app.get("/definitions")
def definitions() -> dict[str, Any]:
    pipeline = get_pipeline()
    return {
        "definitions": [
            {
                "id": d.id,
                "version": d.version,
                "key": d.key,
                "status": d.status,
                "label": d.label,
                "description": d.description,
                "hash": d.definition_hash,
                "body": d.model_dump(mode="json"),
            }
            for d in pipeline.definitions.all()
        ]
    }


@app.get("/definitions/{definition_id}/{version}/yaml", response_class=PlainTextResponse)
def definition_yaml(definition_id: str, version: int) -> str:
    pipeline = get_pipeline()
    try:
        definition = pipeline.definitions.get(definition_id, version, allow_draft=True)
    except DefinitionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    from pathlib import Path

    return Path(definition.source_path).read_text(encoding="utf-8")


@app.get("/definitions/{definition_id}/diff")
def definition_diff(definition_id: str, left: int, right: int) -> dict[str, Any]:
    pipeline = get_pipeline()
    try:
        a = pipeline.definitions.get(definition_id, left, allow_draft=True)
        b = pipeline.definitions.get(definition_id, right, allow_draft=True)
    except DefinitionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "left": a.key,
        "left_status": a.status,
        "right": b.key,
        "right_status": b.status,
        "changes": diff_definitions(a, b),
    }


@app.post("/definitions/candidate")
def definition_candidate(request: CandidateRequest) -> dict[str, Any]:
    """Render a proposed next version as YAML, without writing anything.

    Editing a control in the UI produces a candidate; it never mutates the
    frozen definition it was derived from.  Publishing is a separate,
    deliberate act.
    """
    pipeline = get_pipeline()
    try:
        base = pipeline.definitions.get(
            request.definition_id, request.base_version, allow_draft=True
        )
    except DefinitionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    body = base.model_dump(mode="json", exclude={"definition_hash", "source_path"})
    next_version = pipeline.definitions.next_version(base.id)
    body["version"] = next_version
    body["status"] = "draft"
    body["supersedes"] = base.key
    body["authors"] = [request.author]

    applied: list[str] = []
    for path, value in sorted(request.changes.items()):
        if _apply_change(body, path, value):
            applied.append(f"{path} = {value!r}")
        else:
            raise HTTPException(
                status_code=400, detail=f"no such definition path: {path!r}"
            )

    from .models import PhenotypeDefinition

    try:
        PhenotypeDefinition.model_validate(body)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"the proposed candidate does not validate: {exc}",
        ) from exc

    rendered = yaml.safe_dump(body, sort_keys=False, allow_unicode=True)
    return {
        "base": base.key,
        "candidate_version": next_version,
        "filename": f"{base.id}.v{next_version}.yaml",
        "applied_changes": applied,
        "yaml": rendered,
        "note": (
            f"This is a candidate. {base.key} is {base.status} and has not been "
            f"modified. Save this file into config/phenotypes/ to publish it."
        ),
    }


def _apply_change(body: dict[str, Any], path: str, value: Any) -> bool:
    """Set one dotted path inside a definition body.

    Three addressing forms, in order of preference:

    ``window.max``
        plain nested keys
    ``evidence_rules.supported.lab.value``
        an evidence rule by **id**, then a predicate by name found anywhere in
        its ``when`` block. Addressing rules by id rather than list index means
        a candidate keeps its meaning if the rule order changes.
    ``evidence_rules.supported.when.all.0.lab.value``
        the fully explicit path, list indices included
    """
    parts = path.split(".")
    node: Any = body
    for index, part in enumerate(parts[:-1]):
        if isinstance(node, list):
            if part.isdigit() and int(part) < len(node):
                node = node[int(part)]
                continue
            match = next(
                (item for item in node if isinstance(item, dict) and item.get("id") == part),
                None,
            )
            if match is None:
                return False
            node = match
            continue
        if isinstance(node, dict):
            if part in node:
                node = node[part]
                continue
            # Shorthand: name a predicate and find it inside this rule's `when`.
            if "when" in node:
                found = _find_predicate(node["when"], part)
                if found is not None:
                    node = found
                    continue
            return False
        return False

    leaf = parts[-1]
    if isinstance(node, dict):
        node[leaf] = value
        return True
    if isinstance(node, list) and leaf.isdigit() and int(leaf) < len(node):
        node[int(leaf)] = value
        return True
    return False


def _find_predicate(condition: Any, name: str) -> dict[str, Any] | None:
    """The first predicate body named ``name`` anywhere inside a `when` block."""
    if isinstance(condition, list):
        for item in condition:
            found = _find_predicate(item, name)
            if found is not None:
                return found
        return None
    if not isinstance(condition, dict):
        return None
    for key, value in condition.items():
        if key == name and isinstance(value, dict):
            return value
        found = _find_predicate(value, name)
        if found is not None:
            return found
    return None


@app.post("/evaluate")
def evaluate(request: EvaluateRequest) -> dict[str, Any]:
    pipeline = get_pipeline()
    try:
        definition = pipeline.definition(
            request.definition_id, request.version, allow_draft=request.allow_draft
        )
    except DefinitionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    manifest = execute(pipeline, definition, studies=request.studies or None)
    if request.save:
        RunStore().save(manifest)
        pipeline.index().record_assignments(manifest.assignments)
    return manifest.model_dump(mode="json")


@app.get("/retrieve")
def retrieve_endpoint(
    concept: Optional[str] = None,
    group: Optional[str] = None,
    text: Optional[str] = None,
    assertion: list[str] = Query(default=[]),
    evidence_state: list[str] = Query(default=[]),
    studies: list[str] = Query(default=[]),
    window_min: Optional[int] = None,
    window_max: Optional[int] = None,
    anchor: Optional[str] = None,
    definition_id: str = "te_symptomatic_hypoglycemia",
    definition_version: Optional[int] = None,
    mode: Literal["lexical", "dense", "hybrid"] = "lexical",
    top_k: int = Query(20, le=1000),
) -> dict[str, Any]:
    pipeline = get_pipeline()
    window = (
        (window_min, window_max)
        if window_min is not None and window_max is not None
        else None
    )
    try:
        result = pipeline.retrieve(
            concept=concept,
            group=group,
            text=text,
            assertion=assertion or None,
            evidence_state=evidence_state or None,
            window=window,
            anchor=anchor,
            studies=studies or None,
            definition_id=definition_id,
            definition_version=definition_version,
            mode=mode,
            top_k=top_k,
        )
    except ConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result.to_dict()


@app.post("/agent/compile")
def agent_compile(request: CompileRequest) -> JSONResponse:
    """Compile a question into a spec. Never executes."""
    from .agent import AgentSession

    pipeline = get_pipeline()
    session = AgentSession(pipeline, request.question, backend=request.backend)
    result = session.compile()
    _SESSIONS[request.question] = session
    payload = result.to_dict()
    payload["approval_required"] = not result.needs_clarification
    payload["executed"] = False
    return JSONResponse(payload, status_code=200 if result.spec else 409)


@app.post("/agent/run")
def agent_run(request: AgentRunRequest) -> JSONResponse:
    """Execute an approved spec. Blocked unless ``approved`` is true."""
    from .agent import AgentSession
    from .agent.run import ApprovalRequired

    pipeline = get_pipeline()
    session = _SESSIONS.get(request.question) or AgentSession(
        pipeline, request.question, backend=request.backend
    )
    result = session.result or session.compile()
    if result.needs_clarification:
        return JSONResponse(
            {
                "executed": False,
                "clarification": result.clarification.model_dump(mode="json"),
                "detail": (
                    "The question leaves a rule underdetermined. Nothing was "
                    "executed."
                ),
            },
            status_code=409,
        )
    if not request.approved:
        return JSONResponse(
            {
                "executed": False,
                "spec": result.spec.model_dump(mode="json"),
                "detail": (
                    "Execution is blocked until the compiled specification is "
                    "explicitly approved. Review the spec, then resend with "
                    "approved=true."
                ),
            },
            status_code=428,
        )
    session.approve()
    try:
        package, manifest = session.execute()
    except ApprovalRequired as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=428, detail=str(exc)) from exc
    payload = package.to_dict()
    payload["executed"] = True
    payload["run"] = manifest.model_dump(mode="json")
    return JSONResponse(payload)


@app.get("/runs")
def list_runs(limit: int = Query(25, le=200)) -> dict[str, Any]:
    manifests = RunStore().list()[-limit:]
    return {
        "runs": [
            {
                "run_id": m.run_id,
                "created_at": m.created_at,
                "definition": f"{m.definition_id}.v{m.definition_version}",
                "definition_hash": m.definition_hash,
                "extractor_version": m.extractor_version,
                "snapshot_id": m.snapshot_id,
                "counts_by_verdict": m.counts_by_verdict,
                "results_hash": m.results_hash,
            }
            for m in manifests
        ]
    }


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    try:
        return RunStore().load(run_id).model_dump(mode="json")
    except ReplayError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/runs/{run_id}/replay")
def replay_run(run_id: str) -> dict[str, Any]:
    from .runs import replay

    # Replay against the corpus this process is actually serving, not whatever
    # happens to sit at the default data path.
    pipeline = get_pipeline()
    try:
        report, manifest = replay(
            run_id,
            data_dir=pipeline.store.root,
            phenotype_dir=pipeline.definitions.directory,
        )
    except ReplayError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "run_id": run_id,
        "reproduced": report.reproduced,
        "original_results_hash": report.original_hash,
        "replayed_results_hash": report.replayed_hash,
        "differences": report.differences,
        "summary": report.summary(),
        "counts_by_verdict": manifest.counts_by_verdict,
    }


# --------------------------------------------------------------------------
# Static UI
# --------------------------------------------------------------------------

if paths.UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(paths.UI_DIR), html=True), name="ui")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(str(paths.UI_DIR / "index.html"))
