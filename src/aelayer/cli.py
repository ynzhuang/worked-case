"""Command line interface.

Every command prints what it did and which versions produced it.  A number
without its definition version, normalizer version and extraction backend is not
a result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from . import paths
from .catalog import ConfigError
from .ingest import IngestError
from .phenotype.loader import DefinitionError
from .semantics import SemanticsError

app = typer.Typer(
    add_completion=False,
    help=(
        "Adverse event evidence layer. Normalizes mixed-format AE evidence into "
        "source-faithful canonical records, derives episodes above them, and "
        "evaluates versioned phenotype definitions. All data is synthetic."
    ),
)
knowledge_app = typer.Typer(help="Program knowledge layer.")
app.add_typer(knowledge_app, name="knowledge")


def _echo(message: str = "") -> None:
    typer.echo(message)


def _fail(message: str) -> None:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _pipeline(data_dir: Optional[str] = None, store: Optional[str] = None,
              backend: str = "auto"):
    from .pipeline import Pipeline

    try:
        return Pipeline.load(
            data_dir, store_path=store or paths.STORE_DB, backend=backend
        )
    except (IngestError, ConfigError, SemanticsError) as exc:
        _fail(str(exc))


# --------------------------------------------------------------------------


@app.command()
def generate(
    seed: int = typer.Option(7, help="Seed. The corpus is fully determined by it."),
    studies: int = typer.Option(6, min=1, max=6),
    truths: int = typer.Option(24, help="Truths rendered under every variant."),
    background: int = typer.Option(18, help="Extra subjects per study."),
    out: Optional[str] = typer.Option(None),
) -> None:
    """Generate the synthetic corpus: ground truth, then six renderings of it."""
    from .generate import generate_corpus

    root, manifest = generate_corpus(
        seed=seed, n_studies=studies, out_dir=out,
        invariance_truths=truths, background_per_study=background,
    )
    counts = manifest["counts"]
    _echo(f"Generated synthetic corpus in {root}")
    _echo(
        f"  {counts['studies']} studies, {counts['subjects']} subjects, "
        f"{counts['ae_records']} source records, "
        f"{counts['episodes_expected']} true episodes"
    )
    _echo(f"  {manifest['invariance_truths']} truths rendered under every variant")
    for study_id, body in sorted(manifest["studies"].items()):
        _echo(
            f"  {study_id} {body['representation']:<4} {body['glucose_unit']:>7} "
            f"| {body['dictionary_version']:<12} | split={body['record_splitting']:<19}"
            f"| forms={len(body['linked_forms'])}"
        )
    _echo("  All records are computer generated. No real patient data.")


@app.command()
def ingest(data_dir: str = typer.Argument(str(paths.DATA_DIR))) -> None:
    """Load a corpus and report what is in it."""
    from .ingest import load_store

    try:
        store = load_store(data_dir)
    except IngestError as exc:
        _fail(str(exc))
    _echo(f"Ingested {data_dir}")
    for key, value in store.summary().items():
        _echo(f"  {key:<22} {value}")


@app.command()
def normalize(
    data_dir: Optional[str] = typer.Option(None),
    limit: int = typer.Option(0, help="Print this many records' collection states."),
) -> None:
    """Run the deterministic path and report collection states."""
    import collections

    pipeline = _pipeline(data_dir)
    from .normalize import normalize_store

    records = normalize_store(pipeline.store, pipeline.configs)
    states = collections.Counter(
        state for r in records for state in r.collection_states().values()
    )
    _echo(f"Normalized {len(records)} source records")
    _echo(f"  normalizer {pipeline.normalizer_version}")
    _echo("  collection states:")
    for state, count in sorted(states.items()):
        _echo(f"    {state:<28} {count}")
    violations = [r.source_record_id for r in records if not r.has_full_provenance()]
    if violations:
        _fail(f"  {len(violations)} record(s) have a populated field with no span")
    _echo("  every populated field traces to a span")
    for record in records[:limit]:
        _echo(f"\n  {record.source_record_id} ({record.study_id})")
        for name, state in record.collection_states().items():
            field = record.fields()[name]
            _echo(f"    {name:<28} {str(field.value)[:24]:<26} {state}")


@app.command()
def extract(
    out: str = typer.Option(str(paths.STORE_DB), "--out"),
    data_dir: Optional[str] = typer.Option(None),
    backend: str = typer.Option("auto", help="auto | rules | llm"),
) -> None:
    """Normalize, enrich from narrative, reconcile episodes, build the index."""
    pipeline = _pipeline(data_dir, out, backend)
    records = pipeline.records(refresh=True)
    episodes = pipeline.episodes(refresh=True)
    index = pipeline.index(refresh=True)
    versions = pipeline.versions()

    _echo(f"{len(records)} canonical records -> {len(episodes)} episodes -> {out}")
    _echo(f"  normalizer  {versions['normalizer_version']}")
    _echo(f"  extractor   {versions['extractor_version']} "
          f"(backend: {versions['extraction_backend']})")
    _echo(f"  snapshot    {versions['snapshot_id']}")
    _echo(f"  indexed     {index.meta().document_count} documents, "
          f"{index.meta().mention_count} narrative mentions")
    for note in pipeline.engine().notes:
        _echo(f"  note: {note}")
    violations = [r.source_record_id for r in records if not r.has_full_provenance()]
    if violations:
        _fail(f"  {len(violations)} record(s) have a populated field with no span: "
              f"{violations[:5]}")
    _echo("  every populated field on every record traces to a span")
    flagged = sum(1 for e in episodes if e.linkage_review_required)
    _echo(f"  {flagged} episode(s) flagged for linkage review, reported not resolved")


@app.command()
def definitions(
    show: Optional[str] = typer.Option(None),
    compare: Optional[str] = typer.Option(
        None, help="Compare two versions, e.g. te_symptomatic_hypoglycemia:1:2"
    ),
    scope: Optional[str] = typer.Option(
        None, help="Required with --compare: the scientific question it applies to."
    ),
    data_dir: Optional[str] = typer.Option(None),
) -> None:
    """List phenotype definitions, show one, or compare two by what they claim."""
    from .knowledge import ScopeRequired, diff_definitions

    pipeline = _pipeline(data_dir)

    if compare:
        try:
            definition_id, left, right = compare.split(":")
            a = pipeline.definition(definition_id, int(left), allow_draft=True)
            b = pipeline.definition(definition_id, int(right), allow_draft=True)
        except (ValueError, DefinitionError) as exc:
            _fail(f"cannot compare {compare!r}: {exc}")
        try:
            comparison = diff_definitions(
                a, b, pipeline.snapshot_id, scope,
                pipeline.evaluate(a), pipeline.evaluate(b),
            )
        except ScopeRequired as exc:
            _fail(str(exc))
        _echo(comparison.summary_line)
        _echo("")
        _echo(f"  {'episode':<40} {'v' + str(a.version):<10} {'v' + str(b.version)}")
        for entry in comparison.discordant[:12]:
            _echo(f"  {entry.episode_id:<40} {entry.verdict_a:<10} {entry.verdict_b}")
        if len(comparison.discordant) > 12:
            _echo(f"  ... {len(comparison.discordant) - 12} more")
        if comparison.discordant:
            first = comparison.discordant[0]
            _echo("")
            _echo(f"  why {first.episode_id} moved:")
            _echo(f"    v{a.version}: {first.reason_a[:150]}")
            _echo(f"    v{b.version}: {first.reason_b[:150]}")
        return

    if show:
        try:
            definition = pipeline.definition(show, allow_draft=True)
        except DefinitionError as exc:
            _fail(str(exc))
        _echo(json.dumps(definition.model_dump(mode="json"), indent=2, default=str))
        return

    for definition in pipeline.definitions.all():
        _echo(
            f"{definition.key:<40} {definition.status:<11} "
            f"{definition.definition_hash}  {definition.label}"
        )


@app.command()
def evaluate(
    definition: str = typer.Option("te_symptomatic_hypoglycemia", "--definition"),
    version: Optional[int] = typer.Option(None),
    study: list[str] = typer.Option([], "--study"),
    allow_draft: bool = typer.Option(False),
    limit: int = typer.Option(12),
    save: bool = typer.Option(True),
    data_dir: Optional[str] = typer.Option(None),
    store: Optional[str] = typer.Option(None),
) -> None:
    """Evaluate a definition over episodes and print the case table with reasons."""
    from .runs import execute

    pipeline = _pipeline(data_dir, store)
    try:
        resolved = pipeline.definition(definition, version, allow_draft=allow_draft)
    except DefinitionError as exc:
        _fail(str(exc))

    manifest, assignments = execute(
        pipeline, resolved, studies=list(study) or None, save=save
    )
    if save:
        pipeline.index().record_assignments(assignments)

    _echo(resolved.label)
    _echo(f"  definition  {resolved.key}  status={resolved.status}  "
          f"hash={resolved.definition_hash}")
    _echo(f"  normalizer  {manifest.normalizer_version}")
    _echo(f"  extractor   {manifest.extractor_version} "
          f"({manifest.parameters.get('backend')})")
    _echo(f"  snapshot    {manifest.data_snapshot_id}")
    _echo(f"  manifest    {manifest.manifest_id}  results={manifest.results_hash}")
    _echo("")
    _echo(f"  verdicts    {manifest.counts_by_verdict}")
    _echo(f"  states      {manifest.counts_by_state}")
    _echo("")
    _echo(f"  {'episode':<40} {'verdict':<9} {'state':<13} {'rule':<13} reason")
    _echo(f"  {'-'*40} {'-'*9} {'-'*13} {'-'*13} {'-'*40}")
    shown = [a for a in assignments if a.verdict in ("case", "review")]
    for assignment in shown[:limit]:
        reason = assignment.reason
        if len(reason) > 76:
            reason = reason[:73] + "..."
        _echo(
            f"  {assignment.episode_id:<40} {assignment.verdict:<9} "
            f"{assignment.evidence_state:<13} "
            f"{(assignment.matched_rule_id or '-'):<13} {reason}"
        )
    if len(shown) > limit:
        _echo(f"  ... {len(shown) - limit} more (raise --limit)")
    _echo("")
    review = manifest.counts_by_verdict.get("review", 0)
    flagged = sum(1 for a in assignments if a.linkage_review_required)
    _echo(
        f"  {review} episode(s) in the review set, reported separately; "
        f"{flagged} carry a flagged episode linkage."
    )
    if save:
        _echo(f"  results -> {manifest.output_pointer}")
        _echo(f"  replay with:  aelayer replay {manifest.manifest_id}")


@app.command()
def retrieve(
    query: str = typer.Argument(..., help="Concept id, group id, or free text."),
    mode: str = typer.Option("precise", help="precise | lexical | dense | hybrid"),
    assertion: list[str] = typer.Option([], "--assertion"),
    verdict: list[str] = typer.Option([], "--verdict"),
    study: list[str] = typer.Option([], "--study"),
    representation: list[str] = typer.Option([], "--representation"),
    window: Optional[str] = typer.Option(None, help="Offset window, e.g. 0:14"),
    definition: str = typer.Option("te_symptomatic_hypoglycemia"),
    version: Optional[int] = typer.Option(None),
    top_k: int = typer.Option(10, "--top-k"),
    as_json: bool = typer.Option(False, "--json"),
    data_dir: Optional[str] = typer.Option(None),
    store: Optional[str] = typer.Option(None),
) -> None:
    """Retrieve episodes (precise) or narrative mentions (discovery)."""
    from .retrieval.query import discover, retrieve as retrieve_episodes

    pipeline = _pipeline(data_dir, store)
    index = pipeline.index()
    catalog = pipeline.catalog
    is_concept = query in catalog.concepts or query in catalog.concept_groups

    if mode == "precise":
        bounds = None
        if window:
            try:
                low, high = window.split(":")
                bounds = (int(low), int(high))
            except ValueError:
                _fail(f"--window must look like 0:14, got {window!r}")
        result = retrieve_episodes(
            index, catalog,
            concept=query if is_concept else None,
            text=None if is_concept else query,
            assertion=list(assertion) or None,
            verdict=list(verdict) or None,
            studies=list(study) or None,
            representation=list(representation) or None,
            window=bounds, definition_id=definition,
            definition_version=version, mode="precise", top_k=top_k,
        )
        if as_json:
            _echo(json.dumps(result.to_dict(), indent=2))
            return
        _echo(f"{len(result.records)} episode(s), precise cohort path")
        _echo(f"  usable as a cohort: {result.to_dict()['usable_as_cohort']}")
        for record in result.records:
            _echo(
                f"  {record.episode_id:<40} {str(record.verdict):<9} "
                f"{str(record.evidence_state):<13} "
                f"{record.representation:<5} off={record.onset_offset_days}"
            )
        return

    result = discover(
        index, catalog,
        concept=query if is_concept else None,
        text=None if is_concept else query,
        assertion=list(assertion) or None,
        studies=list(study) or None,
        mode=mode, top_k=top_k,
    )
    if as_json:
        _echo(json.dumps(result.to_dict(), indent=2))
        return
    _echo(f"{len(result.mentions)} mention(s), discovery path ({result.mode})")
    _echo(
        f"  mentions asserting absence: {result.negation_false_positives} "
        f"(rate {result.negation_false_positive_rate:.4f})"
    )
    for note in result.notes:
        _echo(f"  note: {note}")
    _echo("")
    for mention in result.mentions:
        _echo(
            f"  {mention.subject_id:<18} {mention.assertion:<15} "
            f"{mention.match_kind:<14} {mention.surface!r}"
        )
        _echo(f"      {mention.sentence[:110]}")


@app.command()
def ask(
    question: str = typer.Argument(...),
    backend: str = typer.Option("deterministic", help="deterministic | llm"),
    show_trace: bool = typer.Option(True, "--trace/--no-trace"),
    as_json: bool = typer.Option(False, "--json"),
    data_dir: Optional[str] = typer.Option(None),
    store: Optional[str] = typer.Option(None),
) -> None:
    """Compile a question, execute it, and trace the number back to source."""
    from .agent import AgentSession, render_trace

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
        for option in clarification.options:
            _echo(f"    - {option}")
        _echo("")
        _echo("  No specification was compiled and nothing was executed.")
        raise typer.Exit(code=2)

    package, manifest = session.execute()
    if as_json:
        _echo(json.dumps(package.to_dict(), indent=2, default=str))
        return

    _echo("Compiled specification:")
    for line in json.dumps(package.spec.model_dump(mode="json"), indent=2).splitlines():
        _echo(f"  {line}")
    _echo("")
    summary = package.summary
    _echo(f"  primary cases      {summary['primary_case_count']}")
    _echo(f"  review set         {summary['review_set_count']} (reported separately)")
    _echo(f"  linkage flagged    {summary['linkage_flagged']}")
    _echo(f"  incidence          {package.statistics['incidence_proportion']}")
    _echo(f"  counts by state    {summary['counts_by_state']}")
    _echo("")
    _echo(f"  definition  {package.definition['id']}.v{package.definition['version']} "
          f"({package.definition['status']}) hash={package.definition['hash']}")
    _echo(f"  normalizer  {package.versions['normalizer_version']}")
    _echo(f"  extractor   {package.versions['extractor_version']} "
          f"({package.versions['extraction_backend']})")
    _echo(f"  manifest    {package.manifest_id}  results={package.results_hash}")
    _echo(f"  services    {', '.join(package.services_called)}")
    _echo("")
    if show_trace and package.trace:
        _echo(f"  traceable to source: {package.trace.complete}")
        _echo("")
        for line in render_trace(package.trace).splitlines()[:24]:
            _echo(f"  {line}")
        _echo("")
    _echo("  limitations:")
    for limitation in package.limitations:
        _echo(f"    - {limitation}")


@app.command()
def trace(
    manifest_id: str = typer.Argument(..., help="Manifest id from a prior run."),
    verdict: str = typer.Option("case"),
    data_dir: Optional[str] = typer.Option(None),
    store: Optional[str] = typer.Option(None),
) -> None:
    """Follow a reported number back to the text a site wrote."""
    from .agent import render_trace, trace_number
    from .runs import ManifestStore, ReplayError, ResultStore

    pipeline = _pipeline(data_dir, store)
    try:
        manifest = ManifestStore(paths.RUNS_DIR).load(manifest_id)
        assignments = ResultStore(paths.RUNS_DIR / "results").read(manifest_id)
    except ReplayError as exc:
        _fail(str(exc))

    count = sum(1 for a in assignments if a.verdict == verdict)
    chain = trace_number(
        number=count, label=f"{verdict} count", manifest=manifest,
        assignments=assignments, episodes=pipeline.episodes(),
        records=pipeline.records(),
    )
    _echo(render_trace(chain))
    if not chain.complete:
        raise typer.Exit(code=1)


@app.command(name="eval")
def eval_command(
    report: Optional[str] = typer.Option(str(paths.REPORTS_DIR / "eval.md"), "--report"),
    definition: str = typer.Option("te_symptomatic_hypoglycemia"),
    version: Optional[int] = typer.Option(None),
    as_json: Optional[str] = typer.Option(None, "--json"),
    data_dir: Optional[str] = typer.Option(None),
    store: Optional[str] = typer.Option(None),
) -> None:
    """Run the full evaluation harness and write the report."""
    from .eval.harness import run_evaluation

    pipeline = _pipeline(data_dir, store)
    results, written = run_evaluation(pipeline, definition, version, report)

    layer1, layer2, layer3 = results["layer1"], results["layer2"], results["layer3"]
    invariance, repro = results["invariance"], results["reproducibility"]
    _echo("Evaluation")
    _echo(f"  layer 1  collection-state accuracy "
          f"{layer1['collection_state_confusion']['accuracy']:.3f}, "
          f"abstention precision {layer1['abstention']['abstention_precision']:.3f}, "
          f"answer precision {layer1['abstention']['answer_precision']:.3f}")
    _echo(f"  layer 2  boundary agreement {layer2['boundary_agreement']:.3f}, "
          f"over-merge {layer2['over_merge']}, over-split {layer2['over_split']}")
    pooled = layer3["pooled"]
    _echo(f"  layer 3  PPV {pooled['ppv']:.3f}, sensitivity "
          f"{pooled['sensitivity']:.3f} "
          f"({pooled['false_negatives_from_linkage_review']} of "
          f"{pooled['fn']} misses are declined linkage)")
    _echo(f"  transport dev {layer3['transportability']['development']['sensitivity']:.3f} "
          f"-> held-out {layer3['transportability']['held_out']['sensitivity']:.3f}")
    _echo(f"  invariance verdict {invariance['verdict_agreement']:.3f}, "
          f"state {invariance['state_agreement']:.3f} "
          f"({invariance['discordant_count']} discordant)")
    retrieval = results.get("retrieval") or {}
    if retrieval.get("available"):
        _echo(
            f"  retrieval negation FP rate "
            f"{retrieval['assertion_filter_on']['negation_false_positive_rate']:.4f} "
            f"with the assertion filter on, "
            f"{retrieval['assertion_filter_off']['negation_false_positive_rate']:.4f} off"
        )
    _echo(f"  repro    manifest stable={repro['manifest_id_stable']}, "
          f"results stable={repro['results_stable']}")
    violations = layer1["provenance_violations"]
    _echo(f"  provenance {len(violations)} violation(s)")

    if as_json:
        target = Path(as_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        cleaned = {
            k: ({kk: vv for kk, vv in v.items() if not kk.endswith("_matrix")}
                if isinstance(v, dict) else v)
            for k, v in results.items()
        }
        target.write_text(json.dumps(cleaned, indent=2, default=str), encoding="utf-8")
        _echo(f"  json   -> {target}")
    if written:
        _echo(f"  report -> {written}")
    if violations:
        raise typer.Exit(code=1)


@app.command()
def replay(
    manifest_id: str = typer.Argument(...),
    data_dir: Optional[str] = typer.Option(None),
) -> None:
    """Reproduce a prior run and verify it matches, hash for hash."""
    from .runs import ReplayError, replay as do_replay

    try:
        report, _manifest = do_replay(manifest_id, data_dir=data_dir)
    except ReplayError as exc:
        _fail(str(exc))
    _echo(report.summary())
    if not report.reproduced:
        for difference in report.differences:
            _echo(f"  - {difference}")
        raise typer.Exit(code=1)


@knowledge_app.command("status")
def knowledge_status(data_dir: Optional[str] = typer.Option(None)) -> None:
    """What the registry holds. Empty on day one, and it says so."""
    from .knowledge import KnowledgeRegistry

    pipeline = _pipeline(data_dir)
    status = KnowledgeRegistry.open(definitions=pipeline.definitions).status()
    _echo(f"Manifests recorded: {status['manifests']}")
    _echo(f"Capture mode:       {status['capture_mode']}")
    _echo(f"Definitions:        {', '.join(status['definitions'])}")
    if status["definitions_used"]:
        _echo(f"Used in runs:       {', '.join(status['definitions_used'])}")
    _echo("")
    _echo(f"  {status['note']}")


@knowledge_app.command("backfill")
def knowledge_backfill(
    manifest: str = typer.Option(..., "--manifest", help="Path to a manifest JSON."),
) -> None:
    """Add one historical execution, deliberately."""
    from .knowledge import KnowledgeRegistry
    from .models import Manifest

    path = Path(manifest)
    if not path.exists():
        _fail(f"no manifest file at {path}")
    try:
        parsed = Manifest.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _fail(f"{path} is not a valid manifest: {exc}")
    written = KnowledgeRegistry.open().backfill(parsed)
    _echo(f"Backfilled {parsed.manifest_id} -> {written}")
    _echo(
        "  Backfill adds exactly what you gave it. Nothing about earlier work "
        "is reconstructed."
    )


@app.command()
def demo(
    seed: int = typer.Option(7),
    studies: int = typer.Option(6, min=1, max=6),
    limit: int = typer.Option(10),
) -> None:
    """Generate, normalize, extract, reconcile, evaluate — end to end."""
    import collections

    from .generate import generate_corpus
    from .runs import execute

    _echo("=== 1. generate ===")
    root, manifest_data = generate_corpus(seed=seed, n_studies=studies)
    counts = manifest_data["counts"]
    _echo(f"  {counts['studies']} studies, {counts['subjects']} subjects, "
          f"{counts['ae_records']} source records -> {root}")
    _echo("  synthetic data only; no real patient records anywhere in this repo")

    pipeline = _pipeline(None, str(paths.STORE_DB))

    _echo("")
    _echo("=== 2. normalize (deterministic path) ===")
    records = pipeline.records(refresh=True)
    states = collections.Counter(
        s for r in records for s in r.collection_states().values()
    )
    _echo(f"  {len(records)} canonical records, normalizer "
          f"{pipeline.normalizer_version}")
    for state, count in sorted(states.items()):
        _echo(f"    {state:<28} {count}")
    _echo("  a blank is not a value: each of these means something different")

    _echo("")
    _echo("=== 3. extract (model path, unresolved fields only) ===")
    versions = pipeline.versions()
    _echo(f"  backend {versions['extraction_backend']}")
    for note in pipeline.engine().notes:
        _echo(f"  {note}")
    recovered = sum(
        1 for r in records for f in r.fields().values() if f.source == "text"
    )
    _echo(f"  {recovered} field value(s) recovered from narrative where the "
          f"structured field was unresolved")
    violations = [r.source_record_id for r in records if not r.has_full_provenance()]
    _echo(f"  {len(violations)} populated field(s) without a span"
          + ("" if violations else "  (every derived value traces to source)"))

    _echo("")
    _echo("=== 4. reconcile episodes ===")
    episodes = pipeline.episodes(refresh=True)
    rules = collections.Counter(e.linkage_rule for e in episodes)
    flagged = sum(1 for e in episodes if e.linkage_review_required)
    _echo(f"  {len(records)} records -> {len(episodes)} episodes")
    for rule, count in sorted(rules.items()):
        _echo(f"    {rule:<24} {count}")
    _echo(f"  {flagged} flagged for review, reported rather than silently resolved")
    _echo("  source records are unmodified: episodes are derived above them")

    _echo("")
    _echo("=== 5. evaluate te_symptomatic_hypoglycemia v1 ===")
    definition = pipeline.definition("te_symptomatic_hypoglycemia", 1)
    manifest, assignments = execute(pipeline, definition)
    pipeline.index(refresh=True).record_assignments(assignments)
    _echo(f"  verdicts {manifest.counts_by_verdict}")
    _echo(f"  states   {manifest.counts_by_state}")
    _echo(f"  manifest {manifest.manifest_id}  results {manifest.results_hash}")

    _echo("")
    _echo("=== 6. case table ===")
    _echo(f"  {'episode':<40} {'verdict':<9} {'state':<13} {'rule':<13} reason")
    _echo(f"  {'-'*40} {'-'*9} {'-'*13} {'-'*13} {'-'*40}")
    shown = [a for a in assignments if a.verdict in ("case", "review")][:limit]
    for assignment in shown:
        reason = assignment.reason
        if len(reason) > 72:
            reason = reason[:69] + "..."
        _echo(f"  {assignment.episode_id:<40} {assignment.verdict:<9} "
              f"{assignment.evidence_state:<13} "
              f"{(assignment.matched_rule_id or '-'):<13} {reason}")
    _echo("")
    _echo(f"  {manifest.counts_by_verdict.get('review', 0)} episode(s) in the review "
          f"set, reported separately rather than discarded.")
    _echo(f"  Trace any number:  aelayer trace {manifest.manifest_id}")
    _echo(f"  Replay this run:   aelayer replay {manifest.manifest_id}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1"), port: int = typer.Option(8000)
) -> None:
    """Serve the API and the single-page UI."""
    import uvicorn

    _echo(f"UI and API on http://{host}:{port}/")
    uvicorn.run("aelayer.api:app", host=host, port=port, log_level="warning")


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
