"""Command line interface.

Every command prints what it did and which versions produced it. A number
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
from .profiles import ProfileError

app = typer.Typer(
    add_completion=False,
    help=(
        "Adverse event evidence layer. One clinical attribute can live in five "
        "different places; this reads all of them into one provenance-bearing "
        "shape and evaluates versioned phenotype definitions over it. All data "
        "is synthetic."
    ),
)
eval_app = typer.Typer(help="Evaluation harnesses.")
knowledge_app = typer.Typer(help="Program knowledge layer.")
app.add_typer(eval_app, name="eval")
app.add_typer(knowledge_app, name="knowledge")

DEFINITION = "te_truncal_rash"


def _echo(message: str = "") -> None:
    typer.echo(message)


def _fail(message: str) -> None:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _gold_version(pipeline, definition_id: str) -> Optional[int]:
    """The version the corpus's answer key was written against.

    Defaulting to the highest published version would quietly evaluate a
    different definition from the one the gold labels describe, and the numbers
    would look like a regression rather than a different question.
    """
    recorded = str(pipeline.store.manifest.get("gold_case_definition") or "")
    if recorded.startswith(f"{definition_id}.v"):
        try:
            return int(recorded.rsplit(".v", 1)[-1])
        except ValueError:
            return None
    return None


def _pipeline(data_dir: Optional[str] = None, store: Optional[str] = None,
              backend: str = "auto"):
    from .pipeline import Pipeline

    try:
        return Pipeline.load(
            data_dir, store_path=store or paths.STORE_DB, backend=backend
        )
    except (IngestError, ConfigError, ProfileError) as exc:
        _fail(str(exc))


# --------------------------------------------------------------------------


@app.command()
def generate(
    seed: int = typer.Option(7, help="Seed. The corpus is fully determined by it."),
    truths: int = typer.Option(24, help="Truths rendered under every profile."),
    background: int = typer.Option(14, help="Extra subjects per profile."),
    out: Optional[str] = typer.Option(None),
) -> None:
    """Generate the synthetic corpus: one truth, rendered under six profiles."""
    from .generate import generate_corpus

    root, manifest = generate_corpus(
        seed=seed, out_dir=out, shared_truths=truths, extra_per_profile=background,
    )
    counts = manifest["counts"]
    _echo(f"Generated synthetic corpus in {root}")
    _echo(
        f"  {counts['profiles']} profiles, {counts['subjects']} subjects, "
        f"{counts['ae_records']} AE records, {counts['suppae_records']} "
        f"supplemental qualifiers, {counts['comments']} comments"
    )
    _echo(f"  {manifest['shared_truths']} truths rendered under every profile")
    _echo("")
    _echo(f"  {'profile':<18} {'term style':<13} {'location home':<26} dictionary")
    for profile_id, body in sorted(manifest["profiles"].items()):
        _echo(
            f"  {profile_id:<18} {body['reported_term_style']:<13} "
            f"{','.join(body['location_home']):<26} {body['dictionary_version']}"
        )
    _echo("")
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
    limit: int = typer.Option(0, help="Print this many records' attributes."),
) -> None:
    """Run the deterministic path and report where each attribute came from."""
    import collections

    pipeline = _pipeline(data_dir)
    records = pipeline.structured_only_records()
    routes = collections.Counter(
        (r.profile, r.location.method or r.location.availability) for r in records
    )
    _echo(f"Normalized {len(records)} source records")
    _echo(f"  normalizer {pipeline.normalizer_version}")
    _echo("")
    _echo("  location, by profile and route (deterministic path only):")
    for (profile, route), count in sorted(routes.items()):
        _echo(f"    {profile:<18} {route:<28} {count}")
    _echo("")
    _echo("  a route is part of the fact: `direct` is the study's own qualifier,")
    _echo("  `normalized` is a declared mapping, and anything else is a question")
    _echo("  for the model path or a value nobody recorded.")
    for record in records[:limit]:
        _echo(f"\n  {record.source_record_id} ({record.profile})")
        for name, attribute in record.attributes().items():
            _echo(f"    {name:<24} {str(attribute.value)[:22]:<24} "
                  f"{attribute.availability}")


@app.command()
def extract(
    out: str = typer.Option(str(paths.STORE_DB), "--out"),
    data_dir: Optional[str] = typer.Option(None),
    backend: str = typer.Option("auto", help="auto | rules | llm"),
) -> None:
    """Normalize, extract from text, reconcile episodes, build the index."""
    import collections

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
    _echo(f"  indexed     {index.meta().episode_count} episodes, "
          f"{index.meta().mention_count} text mentions")
    for note in pipeline.engine().notes:
        _echo(f"  note: {note}")
    routes = collections.Counter(
        r.location.method for r in records if r.location.populated
    )
    _echo("")
    _echo("  location by route:")
    for route, count in sorted(routes.items(), key=lambda kv: str(kv[0])):
        _echo(f"    {str(route):<14} {count}")
    violations = [r.source_record_id for r in records if not r.has_full_provenance()]
    if violations:
        _fail(f"  {len(violations)} record(s) have a populated attribute with no "
              f"span: {violations[:5]}")
    _echo("  every populated attribute on every record traces to a span")


@app.command()
def definitions(
    show: Optional[str] = typer.Option(None),
    compare: Optional[str] = typer.Option(
        None, help="Compare two versions, e.g. te_truncal_rash:1:2"
    ),
    scope: Optional[str] = typer.Option(
        None, help="Required with --compare: the scientific question it applies to."
    ),
    data_dir: Optional[str] = typer.Option(None),
) -> None:
    """List definitions, show one, or compare two by what they claim."""
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
        _echo(f"  {'episode':<46} {'v' + str(a.version):<18} v{b.version}")
        for entry in comparison.discordant[:12]:
            _echo(f"  {entry.episode_id:<46} {entry.verdict_a:<18} {entry.verdict_b}")
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
        accepts = sorted(
            {m for r in definition.required_attributes for m in r.accept_methods}
        )
        _echo(
            f"{definition.key:<26} {definition.status:<11} "
            f"{definition.definition_hash}  accepts={','.join(accepts):<32} "
            f"{definition.label}"
        )


@app.command()
def evaluate(
    definition: str = typer.Option(DEFINITION, "--definition"),
    version: Optional[int] = typer.Option(None),
    study: list[str] = typer.Option([], "--study"),
    allow_draft: bool = typer.Option(False),
    limit: int = typer.Option(12),
    save: bool = typer.Option(True),
    data_dir: Optional[str] = typer.Option(None),
    store: Optional[str] = typer.Option(None),
) -> None:
    """Evaluate a definition and print the verdicts with the route behind each."""
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
    _echo(f"  routes      {manifest.attribute_methods}")
    _echo("")
    _echo(f"  {'episode':<44} {'verdict':<18} {'route':<32} reason")
    _echo(f"  {'-'*44} {'-'*18} {'-'*32} {'-'*40}")
    shown = [a for a in assignments if a.verdict in ("case", "not_ascertainable",
                                                     "review")]
    for assignment in shown[:limit]:
        route = ",".join(
            f"{k}={v}" for k, v in sorted(assignment.attribute_sources.items())
        ) or "-"
        reason = assignment.reason
        if len(reason) > 60:
            reason = reason[:57] + "..."
        _echo(
            f"  {assignment.episode_id:<44} {assignment.verdict:<18} "
            f"{route[:32]:<32} {reason}"
        )
    if len(shown) > limit:
        _echo(f"  ... {len(shown) - limit} more (raise --limit)")
    _echo("")
    na = manifest.counts_by_verdict.get("not_ascertainable", 0)
    _echo(
        f"  {na} episode(s) are not ascertainable: a required attribute was "
        f"never recorded and cannot be recovered."
    )
    _echo("  They are neither cases nor negatives, and are reported separately.")
    if save:
        _echo(f"  results -> {manifest.output_pointer}")
        _echo(f"  replay with:  aelayer replay {manifest.manifest_id}")


@app.command()
def retrieve(
    query: str = typer.Argument(..., help="Concept id, or free text for discovery."),
    mode: str = typer.Option("precise", help="precise | lexical | dense | hybrid"),
    location: list[str] = typer.Option([], "--location"),
    region: Optional[str] = typer.Option(None),
    method: list[str] = typer.Option([], "--method", help="direct|normalized|extracted"),
    verdict: list[str] = typer.Option([], "--verdict"),
    profile: list[str] = typer.Option([], "--profile"),
    window: Optional[str] = typer.Option(None, help="Offset window, e.g. 0:14"),
    unnormalized: bool = typer.Option(
        False, help="Discovery only: mentions no catalogue value covers yet."
    ),
    top_k: int = typer.Option(10, "--top-k"),
    as_json: bool = typer.Option(False, "--json"),
    data_dir: Optional[str] = typer.Option(None),
    store: Optional[str] = typer.Option(None),
) -> None:
    """Retrieve episodes (precise) or text mentions (discovery)."""
    from .retrieval.query import discover, retrieve as retrieve_episodes

    pipeline = _pipeline(data_dir, store)
    index = pipeline.index()

    if mode == "precise":
        bounds = None
        if window:
            try:
                low, high = window.split(":")
                bounds = (int(low), int(high))
            except ValueError:
                _fail(f"--window must look like 0:14, got {window!r}")
        result = retrieve_episodes(
            index, pipeline.catalog,
            concept=query if query in pipeline.catalog.concepts else None,
            location=list(location) or None, region=region,
            method=list(method) or None, verdict=list(verdict) or None,
            profile=list(profile) or None, window=bounds, top_k=top_k,
        )
        if as_json:
            _echo(json.dumps(result.to_dict(), indent=2))
            return
        _echo(f"{len(result.episodes)} episode(s), precise cohort path")
        _echo(f"  usable as a cohort: {result.to_dict()['usable_as_cohort']}")
        for note in result.notes:
            _echo(f"  note: {note}")
        for episode in result.episodes:
            _echo(
                f"  {episode.episode_id:<46} {str(episode.verdict):<18} "
                f"{str(episode.location):<12} {str(episode.location_method):<11} "
                f"{episode.location_source}"
            )
        return

    result = discover(
        index, pipeline.catalog,
        text=None if query in pipeline.catalog.concepts else query,
        profile=list(profile) or None, unnormalized_only=unnormalized,
        mode=mode, top_k=top_k,
    )
    if as_json:
        _echo(json.dumps(result.to_dict(), indent=2))
        return
    _echo(f"{len(result.mentions)} mention(s), discovery path ({result.mode})")
    _echo(f"  usable as a cohort: no — every result is a candidate")
    _echo(f"  {len(result.unnormalized)} of them are not covered by any "
          f"catalogue value")
    for note in result.notes:
        _echo(f"  note: {note}")
    _echo("")
    for mention in result.mentions:
        _echo(
            f"  {mention.subject_id:<22} {mention.attribute:<10} "
            f"{mention.value:<14} {'normalized' if mention.normalized else 'NEW':<11} "
            f"{mention.surface!r}"
        )


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
    cohort = package.cohort
    _echo(f"  verdicts           {cohort['counts_by_verdict']}")
    _echo(f"  subjects           {cohort['subjects_by_verdict']}")
    _echo(f"  evidence routes    {cohort['attribute_methods']}")
    _echo(f"  source variables   {cohort['attribute_sources']}")
    _echo("")
    _echo(f"  {cohort['not_ascertainable_note']}")
    _echo("")
    _echo(f"  definition  {package.definition['id']}.v{package.definition['version']} "
          f"({package.definition['status']}) hash={package.definition['hash']}")
    _echo(f"  normalizer  {package.versions['normalizer_version']}")
    _echo(f"  extractor   {package.versions['extractor_version']} "
          f"({package.versions['extraction_backend']})")
    _echo(f"  manifest    {package.manifest_id}  results={package.results_hash}")
    _echo(f"  tools       {', '.join(package.tools_called)}")
    _echo("")
    if show_trace and package.trace:
        _echo(f"  traceable to source: {package.trace.complete}")
        _echo("")
        for line in render_trace(package.trace).splitlines()[:22]:
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
        records=pipeline.records(), verdict=verdict,
    )
    _echo(render_trace(chain))
    if not chain.complete:
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


# -- eval ------------------------------------------------------------------


@eval_app.command("silver")
def eval_silver(
    attribute: str = typer.Option("location", "--attribute"),
    profile: list[str] = typer.Option([], "--profile"),
    queue: str = typer.Option(str(paths.REPORTS_DIR / "adjudication.jsonl")),
    sample: int = typer.Option(20, help="Agreements sampled into the queue."),
    as_json: Optional[str] = typer.Option(None, "--json"),
    data_dir: Optional[str] = typer.Option(None),
    store: Optional[str] = typer.Option(None),
) -> None:
    """Score extraction against the study's own structured field."""
    from .silver import SilverHarness

    pipeline = _pipeline(data_dir, store)
    harness = SilverHarness(pipeline.configs, pipeline.store, pipeline.engine())
    eligible = list(profile) or harness.eligible_profiles(attribute)
    if not eligible:
        _fail(
            f"no profile records {attribute} both structurally and in text, so "
            f"there is nothing to build a silver standard from"
        )
    report = harness.run(pipeline.records(), attribute, eligible)
    body = report.to_dict()
    overall = body["overall"]

    _echo(f"Silver standard — {attribute}")
    _echo(f"  profiles           {', '.join(eligible)}")
    _echo(f"  eligible records   {overall['eligible_records']}")
    _echo(f"  precision          {overall['precision']:.3f}")
    _echo(f"  recall             {overall['recall']:.3f}")
    _echo(f"  f1                 {overall['f1']:.3f}")
    _echo(f"  coverage           {overall['coverage']:.3f}")
    _echo(f"  abstention rate    {overall['abstention_rate']:.3f}")
    _echo(f"  normalized agreement {overall['normalized_agreement']:.3f}")
    _echo("")
    _echo("  by reported-term style:")
    for style, metrics in body["by_reported_term_style"].items():
        _echo(f"    {style:<14} precision {metrics['precision']:.3f}  "
              f"coverage {metrics['coverage']:.3f}  "
              f"abstention {metrics['abstention_rate']:.3f}")
    _echo("")
    written = report.write_adjudication(queue, agreement_sample=sample)
    rows = report.adjudication_queue(agreement_sample=sample)
    disagreements = sum(1 for r in rows if r["agreement"] == "disagree")
    sampled = sum(1 for r in rows if r["queue_reason"].startswith("sampled"))
    _echo(f"  adjudication queue -> {written}")
    _echo(f"    {len(rows)} row(s): {disagreements} disagreement(s), "
          f"{sampled} sampled agreement(s), the rest low confidence")
    _echo("")
    _echo(f"  {body['caveat']}")
    if as_json:
        target = Path(as_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
        _echo(f"  json -> {target}")


@eval_app.command("transport")
def eval_transport(
    holdout: str = typer.Option("", help="Comma-separated profiles to hold out."),
    definition: str = typer.Option(DEFINITION),
    version: Optional[int] = typer.Option(None),
    data_dir: Optional[str] = typer.Option(None),
    store: Optional[str] = typer.Option(None),
) -> None:
    """Hold out whole studies and report the drop."""
    from .eval.transport import transportability

    pipeline = _pipeline(data_dir, store)
    try:
        resolved = pipeline.definition(
            definition, version if version is not None
            else _gold_version(pipeline, definition)
        )
    except DefinitionError as exc:
        _fail(str(exc))
    try:
        result = transportability(
            pipeline, resolved,
            [p.strip() for p in holdout.split(",") if p.strip()] or None,
        )
    except ValueError as exc:
        _fail(str(exc))

    _echo(f"Transportability — whole studies held out ({resolved.key})")
    _echo(f"  {result['note']}")
    _echo("")
    for side in ("development", "held_out"):
        body = result[side]
        _echo(
            f"  {side:<12} {','.join(result[side + '_profiles']):<40} "
            f"episodes={body['episodes']:<5} PPV={body['ppv']:.3f} "
            f"sens={body['sensitivity']:.3f} "
            f"not-ascertainable={body['not_ascertainable_rate']:.3f}"
        )
    _echo("")
    _echo(f"  sensitivity drop {result['sensitivity_drop']:+.3f}, "
          f"PPV drop {result['ppv_drop']:+.3f}, "
          f"not-ascertainable change {result['not_ascertainable_rate_change']:+.3f}")
    _echo(f"  {result['not_fitted']}")


@eval_app.command("all")
def eval_all(
    report: Optional[str] = typer.Option(str(paths.REPORTS_DIR / "eval.md"), "--report"),
    definition: str = typer.Option(DEFINITION),
    version: Optional[int] = typer.Option(None),
    holdout: str = typer.Option(""),
    as_json: Optional[str] = typer.Option(None, "--json"),
    data_dir: Optional[str] = typer.Option(None),
    store: Optional[str] = typer.Option(None),
) -> None:
    """Run every harness and write the report."""
    from .eval.harness import run_evaluation

    pipeline = _pipeline(data_dir, store)
    results, written = run_evaluation(
        pipeline, definition, version, report,
        [p.strip() for p in holdout.split(",") if p.strip()] or None,
    )
    silver = results["silver"]["overall"]
    phenotype = results["phenotype"]["pooled"]
    ablation = results["ablation"]
    availability = results["availability"]
    transport = results["transport"]
    invariance = results["invariance"]
    repro = results["reproducibility"]

    _echo("Evaluation")
    _echo(f"  silver     precision {silver['precision']:.3f}, recall "
          f"{silver['recall']:.3f}, coverage {silver['coverage']:.3f}, "
          f"abstention {silver['abstention_rate']:.3f} (silver standard, not truth)")
    _echo(f"  phenotype  PPV {phenotype['ppv']:.3f}, sensitivity "
          f"{phenotype['sensitivity']:.3f}, not-ascertainable rate "
          f"{phenotype['not_ascertainable_rate']:.3f}")
    _echo(f"  ablation   {ablation['cases_only_findable_through_text']} of "
          f"{ablation['cases_with_text']} cases "
          f"({ablation['fraction_only_findable_through_text']:.1%}) are findable "
          f"only through text")
    _echo(f"  availability accuracy {availability['accuracy']:.3f}, "
          f"{availability['missing_read_as_collected']} missing read as collected")
    _echo(f"  transport  sensitivity {transport['sensitivity_drop']:+.3f}, "
          f"not-ascertainable {transport['not_ascertainable_rate_change']:+.3f} "
          f"(held out: {','.join(transport['held_out_profiles'])})")
    _echo(f"  invariance {invariance['agreement_where_evidence_supports_it']:.3f} "
          f"where the evidence supports it, {invariance['raw_agreement']:.3f} raw")
    _echo(f"  repro      manifest stable={repro['manifest_id_stable']}, "
          f"results stable={repro['results_stable']}")

    if as_json:
        target = Path(as_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        cleaned = {
            k: ({kk: vv for kk, vv in v.items() if not kk.endswith("matrix")}
                if isinstance(v, dict) else v)
            for k, v in results.items()
        }
        target.write_text(json.dumps(cleaned, indent=2, default=str), encoding="utf-8")
        _echo(f"  json   -> {target}")
    if written:
        _echo(f"  report -> {written}")


# -- knowledge -------------------------------------------------------------


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


@knowledge_app.command("tools")
def knowledge_tools() -> None:
    """The agent's entire callable surface, with schemas and permissions."""
    from .agent.tools import AgentServices

    for spec in AgentServices.catalogue():
        _echo(f"{spec['name']:<20} {spec['permission']:<16} {spec['description']}")
    _echo("")
    _echo("  Every call is validated against its input schema before it runs and")
    _echo("  its output schema before it returns. There is no SQL surface and no")
    _echo("  tool that writes to a source record.")


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


# --------------------------------------------------------------------------


@app.command()
def demo(seed: int = typer.Option(7), limit: int = typer.Option(8)) -> None:
    """Generate, normalize, extract, evaluate — end to end, offline."""
    import collections

    from .generate import generate_corpus
    from .runs import execute
    from .silver import SilverHarness

    _echo("=== 1. generate: one truth, six renderings ===")
    root, manifest_data = generate_corpus(seed=seed)
    counts = manifest_data["counts"]
    _echo(f"  {counts['profiles']} profiles, {counts['subjects']} subjects, "
          f"{counts['ae_records']} AE records -> {root}")
    for profile_id, body in sorted(manifest_data["profiles"].items()):
        _echo(f"    {profile_id:<18} location in {','.join(body['location_home'])}")
    _echo("  synthetic data only; no real patient records anywhere in this repo")

    pipeline = _pipeline(None, str(paths.STORE_DB))

    _echo("")
    _echo("=== 2. normalize + extract: five homes, one shape ===")
    records = pipeline.records(refresh=True)
    routes = collections.Counter(
        (r.profile, r.location.method or "unresolved") for r in records
        if r.standardized_concept == "RASH"
    )
    for (profile, route), count in sorted(routes.items()):
        _echo(f"    {profile:<18} {route:<12} {count}")
    _echo(f"  normalizer {pipeline.normalizer_version}")
    _echo(f"  extractor  {pipeline.extractor_version} "
          f"({pipeline.versions()['extraction_backend']})")
    violations = [r.source_record_id for r in records if not r.has_full_provenance()]
    _echo(f"  {len(violations)} populated attribute(s) without a span"
          + ("" if violations else "  (every value traces to source)"))

    _echo("")
    _echo("=== 3. episodes ===")
    episodes = pipeline.episodes(refresh=True)
    _echo(f"  {len(records)} records -> {len(episodes)} episodes")
    for rule, count in sorted(collections.Counter(
        e.linkage_rule for e in episodes
    ).items()):
        _echo(f"    {rule:<24} {count}")

    _echo("")
    _echo("=== 4. silver standard (extraction vs the study's own field) ===")
    harness = SilverHarness(pipeline.configs, pipeline.store, pipeline.engine())
    report = harness.run(records, "location")
    overall = report.to_dict()["overall"]
    _echo(f"  precision {overall['precision']:.3f}  recall {overall['recall']:.3f}  "
          f"coverage {overall['coverage']:.3f}  "
          f"abstention {overall['abstention_rate']:.3f}")
    _echo(f"  {overall['disagreements']} disagreement(s) for adjudication, over "
          f"{overall['eligible_records']} records where both routes speak")
    _echo("  silver standard, not ground truth: the comparator has its own error rate")

    _echo("")
    _echo("=== 5. evaluate te_truncal_rash v1 ===")
    definition = pipeline.definition(DEFINITION, 1)
    manifest, assignments = execute(pipeline, definition)
    pipeline.index(refresh=True).record_assignments(assignments)
    _echo(f"  verdicts {manifest.counts_by_verdict}")
    _echo(f"  routes   {manifest.attribute_methods}")
    _echo(f"  manifest {manifest.manifest_id}  results {manifest.results_hash}")

    _echo("")
    _echo("=== 6. verdicts, with the route that produced each ===")
    _echo(f"  {'episode':<44} {'verdict':<18} {'location':<10} route")
    _echo(f"  {'-'*44} {'-'*18} {'-'*10} {'-'*24}")
    by_id = {e.episode_id: e for e in episodes}
    interesting = [
        a for a in assignments
        if a.verdict in ("case", "not_ascertainable")
    ][:limit]
    for assignment in interesting:
        episode = by_id[assignment.episode_id]
        _echo(
            f"  {assignment.episode_id:<44} {assignment.verdict:<18} "
            f"{str(episode.location.value):<10} "
            f"{episode.location.method or episode.location.availability}"
        )
    na = manifest.counts_by_verdict.get("not_ascertainable", 0)
    _echo("")
    _echo(f"  {na} episode(s) not ascertainable — the location was never recorded")
    _echo(f"  and cannot be recovered. Not a negative, and reported separately.")
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
