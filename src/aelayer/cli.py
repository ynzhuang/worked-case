"""Command line interface.

Every command prints what it did, which versions produced it, and where the
result went.  A number without its definition version and extractor version is
not a result.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer

from . import paths
from .catalog import ConfigError
from .ingest import IngestError
from .phenotype.loader import DefinitionError

app = typer.Typer(
    add_completion=False,
    help=(
        "Adverse event evidence layer. Converts free-text AE evidence into "
        "provenance-bearing event objects and evaluates versioned phenotype "
        "definitions over them. All data is synthetic."
    ),
)


def _echo(message: str = "") -> None:
    typer.echo(message)


def _fail(message: str) -> None:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _pipeline(data_dir: Optional[str] = None, store: Optional[str] = None):
    from .pipeline import Pipeline

    try:
        return Pipeline.load(data_dir, store_path=store or paths.STORE_DB)
    except (IngestError, ConfigError) as exc:
        _fail(str(exc))


# --------------------------------------------------------------------------


@app.command()
def generate(
    seed: int = typer.Option(7, help="Seed. The corpus is fully determined by it."),
    studies: int = typer.Option(4, min=1, max=6, help="Number of studies to generate."),
    subjects: Optional[int] = typer.Option(
        None, help="Override subjects per study (default varies by study)."
    ),
    out: Optional[str] = typer.Option(None, help="Output directory."),
) -> None:
    """Generate the synthetic corpus with its gold answer key."""
    from .generate import generate_corpus

    root, manifest = generate_corpus(
        seed=seed, n_studies=studies, out_dir=out, subjects_per_study=subjects
    )
    counts = manifest["counts"]
    _echo(f"Generated synthetic corpus in {root}")
    _echo(
        f"  {counts['studies']} studies, {counts['subjects']} subjects, "
        f"{counts['ae_records']} AE records with narratives"
    )
    for study_id, body in sorted(manifest["studies"].items()):
        _echo(
            f"  {study_id}: {body['glucose_unit']:>7} | {body['dictionary_version']:<12} "
            f"| coded={'yes' if body['codes_hypoglycemia'] else 'NO':<3} "
            f"| AE detail={body['structured_ae_detail']}"
        )
    _echo("  All records are computer generated. No real patient data.")


@app.command()
def ingest(
    data_dir: str = typer.Argument(str(paths.DATA_DIR), help="Corpus directory."),
) -> None:
    """Load a corpus and report what is in it."""
    from .ingest import load_store

    try:
        store = load_store(data_dir)
    except IngestError as exc:
        _fail(str(exc))
    summary = store.summary()
    _echo(f"Ingested {data_dir}")
    for key, value in summary.items():
        _echo(f"  {key:<16} {value}")


@app.command()
def extract(
    out: str = typer.Option(str(paths.STORE_DB), "--out", help="Store path."),
    data_dir: Optional[str] = typer.Option(None, help="Corpus directory."),
) -> None:
    """Extract event objects and build the retrieval index."""
    pipeline = _pipeline(data_dir, out)
    events = pipeline.events(refresh=True)
    index = pipeline.index(refresh=True)
    violations = [e.event_id for e in events if not e.has_full_provenance()]

    _echo(f"Extracted {len(events)} event objects -> {out}")
    _echo(f"  extractor version {pipeline.extractor_version}")
    _echo(f"  data snapshot     {pipeline.snapshot_id}")
    _echo(f"  indexed documents {index.meta().document_count}")
    if violations:
        _fail(
            f"  {len(violations)} event(s) have a populated field with no span. "
            f"Every derived value must trace to a span: {violations[:5]}"
        )
    _echo("  every populated field on every event traces to a span")


@app.command()
def definitions(
    show: Optional[str] = typer.Option(None, help="Show one definition in full."),
    diff: Optional[str] = typer.Option(
        None, help="Diff two versions, e.g. te_symptomatic_hypoglycemia:1:2"
    ),
) -> None:
    """List phenotype definitions, or show or diff one."""
    from .catalog import load_configs
    from .phenotype.loader import DefinitionCatalog, diff_definitions

    catalog, _config, _version = load_configs()
    definition_catalog = DefinitionCatalog(catalog=catalog)

    if diff:
        try:
            definition_id, left_version, right_version = diff.split(":")
            left = definition_catalog.get(definition_id, int(left_version), allow_draft=True)
            right = definition_catalog.get(definition_id, int(right_version), allow_draft=True)
        except (ValueError, DefinitionError) as exc:
            _fail(f"cannot diff {diff!r}: {exc}")
        changes = diff_definitions(left, right)
        _echo(f"{left.key} ({left.status}) -> {right.key} ({right.status})")
        if not changes:
            _echo("  no differences")
        for change in changes:
            _echo(f"  {change['path']}: {change['from']!r} -> {change['to']!r}")
        return

    if show:
        try:
            definition = definition_catalog.get(show, allow_draft=True)
        except DefinitionError as exc:
            _fail(str(exc))
        _echo(json.dumps(definition.model_dump(mode="json"), indent=2, default=str))
        return

    for definition in definition_catalog.all():
        _echo(
            f"{definition.key:<38} {definition.status:<11} "
            f"{definition.definition_hash}  {definition.label}"
        )


@app.command()
def evaluate(
    definition: str = typer.Option(
        "te_symptomatic_hypoglycemia", "--definition", help="Definition id."
    ),
    version: Optional[int] = typer.Option(None, help="Version. Default: latest frozen."),
    study: list[str] = typer.Option([], "--study", help="Restrict to a study. Repeatable."),
    allow_draft: bool = typer.Option(False, help="Permit a draft definition to run."),
    limit: int = typer.Option(15, help="Case table rows to print."),
    save: bool = typer.Option(True, help="Write a run manifest."),
    data_dir: Optional[str] = typer.Option(None),
    store: Optional[str] = typer.Option(None),
) -> None:
    """Evaluate a phenotype definition and print the case table with reasons."""
    from .runs import RunStore, execute

    pipeline = _pipeline(data_dir, store)
    try:
        resolved = pipeline.definition(definition, version, allow_draft=allow_draft)
    except DefinitionError as exc:
        _fail(str(exc))

    manifest = execute(pipeline, resolved, studies=list(study) or None)
    if save:
        path = RunStore().save(manifest)
        pipeline.index().record_assignments(manifest.assignments)
    else:
        path = None

    _echo(f"{resolved.label}")
    _echo(f"  definition   {resolved.key}  status={resolved.status}  hash={resolved.definition_hash}")
    _echo(f"  extractor    {manifest.extractor_version}")
    _echo(f"  snapshot     {manifest.snapshot_id}")
    _echo(f"  run          {manifest.run_id}  results={manifest.results_hash}")
    _echo("")
    _echo(f"  verdicts     {manifest.counts_by_verdict}")
    _echo(f"  states       {manifest.counts_by_state}")
    _echo("")

    cases = [a for a in manifest.assignments if a.verdict in ("case", "review")]
    _echo(f"  {'subject':<16} {'verdict':<9} {'state':<10} {'rule':<10} reason")
    _echo(f"  {'-'*16} {'-'*9} {'-'*10} {'-'*10} {'-'*58}")
    for assignment in cases[:limit]:
        reason = assignment.reason
        if len(reason) > 88:
            reason = reason[:85] + "..."
        _echo(
            f"  {assignment.subject_id:<16} {assignment.verdict:<9} "
            f"{assignment.evidence_state:<10} "
            f"{(assignment.matched_rule_id or '-'):<10} {reason}"
        )
    if len(cases) > limit:
        _echo(f"  ... {len(cases) - limit} more (raise --limit to see them)")
    _echo("")
    review = manifest.counts_by_verdict.get("review", 0)
    _echo(
        f"  The review set holds {review} subject(s). It is reported separately "
        f"and is not folded into the case count."
    )
    if path:
        _echo(f"  manifest -> {path}")


@app.command()
def retrieve(
    query: str = typer.Argument(..., help="Concept id, group id, or free text."),
    assertion: list[str] = typer.Option([], "--assertion", help="Repeatable."),
    evidence_state: list[str] = typer.Option([], "--evidence-state", help="Repeatable."),
    window: Optional[str] = typer.Option(None, help="Offset window, e.g. 0:14"),
    anchor: Optional[str] = typer.Option(None, help="Anchor event."),
    study: list[str] = typer.Option([], "--study", help="Repeatable."),
    definition: str = typer.Option("te_symptomatic_hypoglycemia"),
    version: Optional[int] = typer.Option(None),
    mode: str = typer.Option("lexical", help="lexical | dense | hybrid"),
    top_k: int = typer.Option(10, "--top-k"),
    as_json: bool = typer.Option(False, "--json"),
    data_dir: Optional[str] = typer.Option(None),
    store: Optional[str] = typer.Option(None),
) -> None:
    """Retrieve records. Assertion is a structured filter, not a hope."""
    pipeline = _pipeline(data_dir, store)
    catalog = pipeline.catalog

    bounds = None
    if window:
        try:
            low, high = window.split(":")
            bounds = (int(low), int(high))
        except ValueError:
            _fail(f"--window must look like 0:14, got {window!r}")

    is_concept = query in catalog.concepts or query in catalog.concept_groups
    resolved_version = version
    if evidence_state and resolved_version is None:
        try:
            resolved_version = pipeline.definition(definition).version
        except DefinitionError:
            resolved_version = None

    result = pipeline.retrieve(
        concept=query if is_concept else None,
        text=None if is_concept else query,
        assertion=list(assertion) or None,
        evidence_state=list(evidence_state) or None,
        window=bounds,
        anchor=anchor,
        studies=list(study) or None,
        definition_id=definition,
        definition_version=resolved_version,
        mode=mode,  # type: ignore[arg-type]
        top_k=top_k,
    )
    if as_json:
        _echo(json.dumps(result.to_dict(), indent=2))
        return

    _echo(f"{len(result.records)} record(s), mode={result.mode}")
    if result.expanded_terms:
        _echo(f"  concept expanded to {len(result.expanded_terms)} surface forms")
    _echo(
        f"  records asserting absence: {result.negation_false_positives} "
        f"(rate {result.negation_false_positive_rate:.4f})"
    )
    for note in result.notes:
        _echo(f"  note: {note}")
    _echo("")
    for record in result.records:
        state = record.evidence_state or "-"
        _echo(
            f"  {record.subject_id:<16} {record.assertion:<14} {state:<10} "
            f"off={str(record.onset_offset_days):<5} {record.doc_id}"
        )
        _echo(f"      {record.snippet}")


@app.command()
def ask(
    question: str = typer.Argument(..., help="A question in plain language."),
    approve: bool = typer.Option(
        False, "--approve", help="Approve the compiled spec and execute it."
    ),
    backend: str = typer.Option("deterministic", help="deterministic | llm"),
    as_json: bool = typer.Option(False, "--json"),
    data_dir: Optional[str] = typer.Option(None),
    store: Optional[str] = typer.Option(None),
) -> None:
    """Compile a question into a spec. Execution requires --approve."""
    from .agent import AgentSession

    pipeline = _pipeline(data_dir, store)
    session = AgentSession(pipeline, question, backend=backend)
    result = session.compile()

    if result.needs_clarification:
        clarification = result.clarification
        if as_json:
            _echo(json.dumps(result.to_dict(), indent=2))
            raise typer.Exit(code=2)
        _echo("Clarification needed before anything can be run.")
        _echo("")
        _echo(f"  Ambiguity: {clarification.ambiguity}")
        _echo(f"  Effect:    {clarification.effect}")
        if clarification.options:
            _echo("  Options:")
            for option in clarification.options:
                _echo(f"    - {option}")
        _echo("")
        _echo("  No specification was compiled and nothing was executed.")
        raise typer.Exit(code=2)

    spec = result.spec
    if as_json and not approve:
        _echo(json.dumps(result.to_dict(), indent=2))
        return

    _echo("Compiled specification (nothing has been executed yet):")
    _echo("")
    for line in json.dumps(spec.model_dump(mode="json"), indent=2).splitlines():
        _echo(f"  {line}")
    _echo("")

    if not approve:
        _echo("  Execution is blocked until this specification is approved.")
        _echo("  Re-run with --approve to execute it.")
        return

    session.approve()
    package, manifest = session.execute()
    if as_json:
        _echo(json.dumps(package.to_dict(), indent=2))
        return

    summary = package.summary
    _echo("Approved and executed.")
    _echo("")
    _echo(f"  primary cases   {summary['primary_case_count']}")
    _echo(f"  review set      {summary['review_set_count']} (reported separately)")
    _echo(f"  counts by state {summary['counts_by_state']}")
    _echo(f"  by rule         {summary['counts_by_rule']}")
    _echo("")
    _echo("  per study:")
    for study_id, counts in summary["per_study"].items():
        _echo(f"    {study_id:<12} {counts}")
    _echo("")
    _echo(f"  definition      {package.definition['id']}.v{package.definition['version']} "
          f"({package.definition['status']}) hash={package.definition['hash']}")
    _echo(f"  extractor       {package.extractor_version}")
    _echo(f"  snapshot        {package.snapshot_id}")
    _echo(f"  run             {package.run_id}  results={package.results_hash}")
    _echo("")
    _echo("  example contributing spans:")
    for span in package.contributing_spans[:5]:
        _echo(f"    {span['subject_id']:<16} {span['field']:<12} {span['text'][:64]!r}")
    _echo("")
    _echo("  limitations:")
    for limitation in package.limitations:
        _echo(f"    - {limitation}")


@app.command(name="eval")
def eval_command(
    report: Optional[str] = typer.Option(
        str(paths.REPORTS_DIR / "eval.md"), "--report", help="Markdown report path."
    ),
    definition: str = typer.Option("te_symptomatic_hypoglycemia"),
    version: Optional[int] = typer.Option(None),
    as_json: Optional[str] = typer.Option(None, "--json", help="Also write JSON here."),
    data_dir: Optional[str] = typer.Option(None),
    store: Optional[str] = typer.Option(None),
) -> None:
    """Run the full evaluation harness and write the report."""
    from .eval.harness import run_evaluation

    pipeline = _pipeline(data_dir, store)
    results, written = run_evaluation(pipeline, definition, version, report)

    extraction = results["extraction"]
    phenotype = results["phenotype"]
    retrieval = results["retrieval"]
    stability = results["stability"]

    _echo("Evaluation")
    _echo(f"  extraction   assertion accuracy "
          f"{extraction['assertion_confusion']['accuracy']:.3f}, "
          f"concept F1 {extraction['concept_detection']['f1']:.3f}")
    _echo(f"  phenotype    PPV {phenotype['pooled']['ppv']:.3f}, "
          f"sensitivity {phenotype['pooled']['sensitivity']:.3f} "
          f"over {phenotype['subjects']} subjects")
    _echo(
        f"  retrieval    negation FP rate "
        f"{retrieval['assertion_filter_on']['negation_false_positive_rate']:.4f} "
        f"with the assertion filter on, "
        f"{retrieval['assertion_filter_off']['negation_false_positive_rate']:.4f} off"
    )
    _echo(f"  stability    run id stable={stability['run_id_stable']}, "
          f"results stable={stability['results_stable']}")
    violations = extraction["provenance_violations"]
    _echo(f"  provenance   {len(violations)} violation(s)")
    for sweep in results["sensitivity"]["sweeps"]:
        low, high = sweep["case_count_range"]
        _echo(f"  sensitivity  {sweep['parameter']}: case count {low}-{high}")

    if as_json:
        target = Path(as_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        serialisable = {
            k: v for k, v in results.items() if not k.endswith("_matrix")
        }
        serialisable["extraction"] = {
            k: v for k, v in results["extraction"].items() if not k.endswith("_matrix")
        }
        serialisable["phenotype"] = {
            k: v for k, v in results["phenotype"].items() if not k.endswith("_matrix")
        }
        target.write_text(json.dumps(serialisable, indent=2, default=str), encoding="utf-8")
        _echo(f"  json   -> {target}")
    if written:
        _echo(f"  report -> {written}")
    if violations:
        raise typer.Exit(code=1)


@app.command()
def replay(
    run_id: str = typer.Argument(..., help="Run id from a manifest."),
    data_dir: Optional[str] = typer.Option(None),
) -> None:
    """Reproduce a prior run and verify it matches, hash for hash."""
    from .runs import ReplayError, replay as do_replay

    try:
        report, _manifest = do_replay(run_id, data_dir=data_dir)
    except ReplayError as exc:
        _fail(str(exc))
    _echo(report.summary())
    if not report.reproduced:
        for difference in report.differences:
            _echo(f"  - {difference}")
        raise typer.Exit(code=1)


@app.command()
def runs(limit: int = typer.Option(20)) -> None:
    """List recorded runs."""
    from .runs import RunStore

    recorded = RunStore().list()
    if not recorded:
        _echo("No runs recorded yet.")
        return
    for manifest in recorded[-limit:]:
        _echo(
            f"{manifest.run_id}  {manifest.created_at}  "
            f"{manifest.definition_id}.v{manifest.definition_version}  "
            f"cases={manifest.counts_by_verdict.get('case', 0)}  "
            f"review={manifest.counts_by_verdict.get('review', 0)}"
        )


@app.command()
def demo(
    seed: int = typer.Option(7),
    studies: int = typer.Option(4, min=1, max=6),
    limit: int = typer.Option(12),
) -> None:
    """Generate, extract, evaluate and print the case table, end to end."""
    from .generate import generate_corpus
    from .runs import RunStore, execute

    _echo("=== 1. generate ===")
    root, manifest_data = generate_corpus(seed=seed, n_studies=studies)
    counts = manifest_data["counts"]
    _echo(
        f"  {counts['studies']} studies, {counts['subjects']} subjects, "
        f"{counts['ae_records']} AE records with narratives -> {root}"
    )
    _echo("  synthetic data only; no real patient records anywhere in this repo")

    _echo("")
    _echo("=== 2. extract ===")
    pipeline = _pipeline(None, str(paths.STORE_DB))
    events = pipeline.events(refresh=True)
    pipeline.index(refresh=True)
    violations = [e.event_id for e in events if not e.has_full_provenance()]
    _echo(f"  {len(events)} event objects, extractor {pipeline.extractor_version}")
    _echo(
        f"  {len(violations)} populated field(s) without a span"
        + ("" if violations else "  (every derived value traces to source)")
    )

    _echo("")
    _echo("=== 3. evaluate te_symptomatic_hypoglycemia v1 ===")
    definition = pipeline.definition("te_symptomatic_hypoglycemia", 1)
    run = execute(pipeline, definition)
    RunStore().save(run)
    pipeline.index().record_assignments(run.assignments)
    _echo(f"  verdicts {run.counts_by_verdict}")
    _echo(f"  states   {run.counts_by_state}")
    _echo(f"  run {run.run_id}  results {run.results_hash}")

    _echo("")
    _echo("=== 4. case table ===")
    _echo(f"  {'subject':<16} {'verdict':<9} {'state':<10} {'rule':<10} reason")
    _echo(f"  {'-'*16} {'-'*9} {'-'*10} {'-'*10} {'-'*60}")
    shown = [a for a in run.assignments if a.verdict in ("case", "review")][:limit]
    for assignment in shown:
        reason = assignment.reason
        if len(reason) > 90:
            reason = reason[:87] + "..."
        _echo(
            f"  {assignment.subject_id:<16} {assignment.verdict:<9} "
            f"{assignment.evidence_state:<10} "
            f"{(assignment.matched_rule_id or '-'):<10} {reason}"
        )
    _echo("")
    _echo(
        f"  {run.counts_by_verdict.get('review', 0)} subject(s) are in the review "
        f"set, reported separately rather than discarded."
    )
    _echo(f"  Replay this run with:  aelayer replay {run.run_id}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
) -> None:
    """Serve the API and the single-page UI."""
    import uvicorn

    _echo(f"UI and API on http://{host}:{port}/")
    uvicorn.run("aelayer.api:app", host=host, port=port, log_level="warning")


def main() -> None:  # pragma: no cover - console entry point
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
