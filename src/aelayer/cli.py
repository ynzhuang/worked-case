"""Command line interface.

Every command prints what it did and which versions produced it. A number
without its definition version, normalizer version and extraction backend is
not a result.

Two commands carry the weight of the whole prototype and are named
accordingly: ``ablation`` states whether reading narrative text was worth it,
and ``silver`` scores the extraction against a masked comparator and prints
both of its caveats verbatim.
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
        "Adverse event evidence layer. One clinical modifier can live in five "
        "different places; this reads all of them into one provenance-bearing "
        "shape and evaluates versioned phenotype definitions over it. All data "
        "is synthetic."
    ),
)
eval_app = typer.Typer(help="Evaluation harnesses.")
knowledge_app = typer.Typer(help="Program knowledge layer.")
app.add_typer(eval_app, name="eval")
app.add_typer(knowledge_app, name="knowledge")

DEFINITION = "cutaneous_mucosal"
MODIFIER = "mucosal_involvement"


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
    except (IngestError, ConfigError, ProfileError) as exc:
        _fail(str(exc))


# --------------------------------------------------------------------------


@app.command()
def generate(
    seed: int = typer.Option(7, help="Deterministic seed."),
    out: str = typer.Option(str(paths.DATA_DIR), help="Output directory."),
    shared: int = typer.Option(24, help="Truths rendered under every profile."),
    extra: int = typer.Option(12, help="Extra truths per profile."),
) -> None:
    """Generate the synthetic corpus: one truth, rendered under seven profiles."""
    from .generate import generate_corpus

    path, manifest = generate_corpus(
        seed=seed, out_dir=out, shared_truths=shared, extra_per_profile=extra,
    )
    counts = manifest["counts"]
    _echo(f"{counts['profiles']} studies, {counts['subjects']} subjects, "
          f"{counts['ae_records']} source records -> {path}")
    _echo("  synthetic data only; no real patient records anywhere in this repo")
    _echo()
    _echo(f"  {'profile':<20} {'term style':<12} {'modifier home':<24} dictionary")
    for name, body in sorted(manifest["profiles"].items()):
        _echo(
            f"  {name:<20} {body['reported_term_style']:<12} "
            f"{','.join(body['modifier_homes']):<24} "
            f"{body['dictionary_version']}"
        )


@app.command()
def ingest(data_dir: str = typer.Argument(str(paths.DATA_DIR))) -> None:
    """Load a corpus and report what is in it."""
    from .ingest import load_store

    try:
        store = load_store(data_dir)
    except IngestError as exc:
        _fail(str(exc))
    summary = store.summary()
    _echo(f"snapshot {store.snapshot_id}")
    for key, value in sorted(summary.items()):
        _echo(f"  {key:<24} {value}")


@app.command()
def normalize(
    data_dir: Optional[str] = typer.Option(None),
    limit: int = typer.Option(0, help="Print this many records' attributes."),
) -> None:
    """Run the deterministic path and report where each modifier came from."""
    import collections

    pipeline = _pipeline(data_dir)
    records = pipeline.structured_only_records()
    _echo(f"{len(records)} canonical records, normalizer "
          f"{pipeline.normalizer_version}")
    _echo()
    routes = collections.Counter(
        (r.profile, (r.modifiers.get(MODIFIER).describe_route()
                     if r.modifiers.get(MODIFIER) else "missing"))
        for r in records
    )
    _echo(f"  {MODIFIER}, by profile and route (deterministic path only):")
    for (profile, route), count in sorted(routes.items()):
        _echo(f"    {profile:<20} {route:<48} {count}")
    _echo()
    _echo("  assertion and availability are separate fields: a route reading")
    _echo("  'absent' is an observed negative, and 'not_collected' is silence")
    _echo()
    reconciliation = collections.Counter(
        r.coded_event.reconciliation for r in records if r.coded_event
    )
    _echo("  dictionary version reconciliation (mechanical, never a model):")
    for outcome, count in sorted(reconciliation.items()):
        _echo(f"    {outcome:<26} {count}")
    for record in records[:limit]:
        _echo()
        _echo(f"  {record.record_id} ({record.profile})")
        for name, attribute in record.attributes().items():
            _echo(f"    {name:<24} {str(attribute.value)[:20]:<22} "
                  f"{str(attribute.assertion):<10} {attribute.availability}")


@app.command()
def extract(
    data_dir: Optional[str] = typer.Option(None),
    backend: str = typer.Option("auto", help="auto | rules | llm"),
    store: Optional[str] = typer.Option(None),
) -> None:
    """Normalize, extract from text, and build the retrieval index."""
    import collections

    pipeline = _pipeline(data_dir, store, backend)
    records = pipeline.records(refresh=True)
    engine = pipeline.engine()
    index = pipeline.index(refresh=True)

    _echo(f"{len(records)} canonical records, backend {engine.backend.name}")
    for note in engine.notes:
        _echo(f"  {note}")
    _echo()
    stats = engine.stats.to_dict()
    _echo(f"  requests            {stats['requests']}")
    _echo(f"  recovered           {stats['recovered']}  {stats['by_assertion']}")
    _echo(f"  abstained           {stats['abstained']}")
    _echo(f"  abstention rate     {stats['abstention_rate']:.3f}")
    _echo("  an abstention is a valid answer, recorded as one; a guess is a defect")
    _echo()
    routes = collections.Counter(
        r.modifiers[MODIFIER].method for r in records
        if r.modifiers.get(MODIFIER) and r.modifiers[MODIFIER].observed
    )
    _echo(f"  {MODIFIER} by route: {dict(sorted(routes.items()))}")
    _echo(f"  indexed     {index.meta().record_count} records, "
          f"{index.meta().mention_count} text mentions")

    violations = [r.record_id for r in records if not r.has_full_provenance()]
    if violations:
        _fail(f"  {len(violations)} record(s) have an observed attribute with no "
              f"span: {violations[:5]}")
    _echo("  every observed attribute on every record traces to a span")


@app.command()
def supportability(
    modifier: str = typer.Option(MODIFIER),
    data_dir: Optional[str] = typer.Option(None),
) -> None:
    """Which studies can answer a question, decided on metadata alone."""
    pipeline = _pipeline(data_dir)
    rows = pipeline.supportability(modifier)
    _echo(f"supportability for {modifier!r}, from declared collection metadata")
    _echo("  no patient record was read to produce this")
    _echo()
    _echo(f"  {'study':<10} {'profile':<20} {'status':<26} reason")
    for row in rows:
        _echo(f"  {row['study_id']:<10} {row['profile']:<20} "
              f"{row['status']:<26} {row['reason'][:70]}")


@app.command()
def definitions(
    definition_id: Optional[str] = typer.Argument(None),
    version: Optional[int] = typer.Option(None),
    compare: Optional[str] = typer.Option(
        None, "--compare", help="Compare two versions, as '1:2'."
    ),
    scope: Optional[str] = typer.Option(
        None, "--scope", help="Required for --compare: the question it applies to."
    ),
    data_dir: Optional[str] = typer.Option(None),
) -> None:
    """List definitions, show one, or compare two by what they claim."""
    from .knowledge import ScopeRequired, diff_definitions

    pipeline = _pipeline(data_dir)
    catalogue = pipeline.definitions

    if definition_id is None:
        _echo(f"{len(catalogue.all())} definition(s) in {catalogue.directory}")
        for definition in catalogue.all():
            _echo(f"  {definition.key:<24} {definition.status:<8} "
                  f"{definition.definition_hash[:12]}  {definition.label}")
        return

    if compare:
        try:
            left, right = (int(p) for p in compare.split(":", 1))
        except ValueError:
            _fail("--compare takes two versions, as '1:2'")
        try:
            a = catalogue.get(definition_id, left, allow_draft=True)
            b = catalogue.get(definition_id, right, allow_draft=True)
            comparison = diff_definitions(
                a, b, pipeline.snapshot_id, scope,
                pipeline.assignments(a), pipeline.assignments(b),
            )
        except ScopeRequired as exc:
            _fail(str(exc))
        except DefinitionError as exc:
            _fail(str(exc))
        _echo(comparison.summary_line)
        _echo()
        _echo(f"  {'record':<44} {'v' + str(a.version):<20} v{b.version}")
        for entry in comparison.discordant[:15]:
            _echo(f"  {entry.record_id:<44} {entry.verdict_a:<20} {entry.verdict_b}")
        if comparison.discordant:
            first = comparison.discordant[0]
            _echo()
            _echo(f"  why {first.record_id} moved:")
            _echo(f"    v{a.version}: {first.reason_a}")
            _echo(f"    v{b.version}: {first.reason_b}")
        return

    try:
        definition = catalogue.get(definition_id, version, allow_draft=True)
    except DefinitionError as exc:
        _fail(str(exc))
    _echo(f"{definition.key}  ({definition.status})")
    _echo(f"  hash        {definition.definition_hash}")
    _echo(f"  label       {definition.label}")
    _echo(f"  concepts    {sorted(definition.concept_set.include)}")
    _echo(f"  target      {definition.concept_set.dictionary_target}")
    for requirement in definition.modifiers:
        _echo(f"  modifier    {requirement.name} must assert "
              f"{requirement.require_assertion!r} via "
              f"{requirement.accept_methods}; otherwise "
              f"{requirement.on_unavailable}")
    if definition.temporal:
        _echo(f"  temporal    {definition.temporal.minimum}-"
              f"{definition.temporal.maximum} days from "
              f"{definition.temporal.anchor}")
    if definition.grade:
        _echo(f"  grade       minimum {definition.grade.minimum}")
    if definition.cumulative_exposure:
        _echo(f"  exposure    minimum {definition.cumulative_exposure.minimum:g} "
              f"{definition.cumulative_exposure.unit} before onset")
    _echo(f"  verdicts    {definition.verdicts}")


@app.command()
def evaluate(
    definition_id: str = typer.Argument(DEFINITION),
    version: Optional[int] = typer.Option(None),
    studies: list[str] = typer.Option([], "--study"),
    limit: int = typer.Option(10),
    data_dir: Optional[str] = typer.Option(None),
    backend: str = typer.Option("auto"),
    save: bool = typer.Option(True, help="Record a manifest for this run."),
) -> None:
    """Evaluate a definition and print the verdicts with the route behind each."""
    from .models import DENOMINATOR_NOTE
    from .runs import execute

    pipeline = _pipeline(data_dir, backend=backend)
    try:
        definition = pipeline.definition(definition_id, version)
    except DefinitionError as exc:
        _fail(str(exc))

    manifest, assignments = execute(
        pipeline, definition, studies=list(studies) or None,
        question=f"cli evaluate {definition.key}", actor="cli", save=save,
    )
    _echo(f"{definition.key} ({definition.status}) over "
          f"{len(assignments)} source record(s)")
    _echo(f"  verdicts    {manifest.counts_by_verdict}")
    _echo(f"  routes      {manifest.attribute_methods}")
    _echo(f"  manifest    {manifest.manifest_id}  results {manifest.results_hash}")
    _echo()
    _echo(f"  {'study':<10} {'total':>6} {'case':>5} {'non':>5} {'rev':>5} "
          f"{'n/a':>5} {'asc.f':>7} {'incidence':>10}")
    for row in manifest.denominators:
        _echo(f"  {row['study_id']:<10} {row['n_total']:6d} {row['n_case']:5d} "
              f"{row['n_non_case']:5d} {row['n_review']:5d} "
              f"{row['n_not_ascertainable']:5d} "
              f"{row['ascertainable_fraction']:7.3f} "
              f"{str(row['incidence_within_ascertainable']):>10}")
    _echo()
    _echo(f"  {DENOMINATOR_NOTE}")
    _echo()
    _echo(f"  {'record':<40} {'verdict':<18} {'route':<30} reason")
    _echo("  " + "-" * 110)
    for assignment in assignments[:limit]:
        route = ",".join(
            f"{k}={v}" for k, v in sorted(assignment.attribute_sources.items())
        ) or "—"
        _echo(f"  {assignment.record_id[-38:]:<40} {assignment.verdict:<18} "
              f"{route[:28]:<30} {assignment.reason[:60]}")
    _echo()
    _echo(f"  Trace any number:  aelayer trace {manifest.manifest_id}")
    _echo(f"  Replay this run:   aelayer replay {manifest.manifest_id}")


@app.command()
def ablation(
    definition_id: str = typer.Argument(DEFINITION),
    version: Optional[int] = typer.Option(None),
    data_dir: Optional[str] = typer.Option(None),
    backend: str = typer.Option("auto"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Is reading narrative text worth it? Three stages, ending in a decision."""
    from .ablation import format_ablation, run_ablation

    pipeline = _pipeline(data_dir, backend=backend)
    try:
        definition = pipeline.definition(definition_id, version)
    except DefinitionError as exc:
        _fail(str(exc))
    report = run_ablation(
        definition, pipeline.store, pipeline.configs, pipeline.backend_preference
    )
    if as_json:
        _echo(json.dumps(report.to_dict(), indent=2))
        return
    _echo(format_ablation(report))


@app.command()
def retrieve(
    text: Optional[str] = typer.Option(None, "--text", help="Discovery path."),
    concept: Optional[str] = typer.Option(None, "--concept"),
    assertion: list[str] = typer.Option([], "--assertion"),
    availability: list[str] = typer.Option([], "--availability"),
    value: list[str] = typer.Option([], "--value"),
    method: list[str] = typer.Option([], "--method"),
    verdict: list[str] = typer.Option([], "--verdict"),
    studies: list[str] = typer.Option([], "--study"),
    top_k: int = typer.Option(20),
    data_dir: Optional[str] = typer.Option(None),
    store: Optional[str] = typer.Option(None),
) -> None:
    """Retrieve records (precise) or text mentions (discovery)."""
    pipeline = _pipeline(data_dir, store)
    if text:
        result = pipeline.discover(text=text, top_k=top_k, studies=list(studies) or None)
        _echo(f"{len(result.mentions)} mention(s), discovery path")
        for note in result.notes:
            _echo(f"  {note}")
        _echo(f"  {'subject':<24} {'assertion':<10} {'value':<12} surface")
        for mention in result.mentions:
            _echo(f"  {mention.subject_id:<24} {mention.assertion:<10} "
                  f"{str(mention.value):<12} {mention.surface[:44]!r}")
        _echo()
        _echo("  every row above is a CANDIDATE. A mention is a place in a")
        _echo("  document where something is named, not an event that occurred.")
        return

    try:
        result = pipeline.retrieve(
            concept=concept, assertion=list(assertion) or None,
            availability=list(availability) or None, value=list(value) or None,
            method=list(method) or None, verdict=list(verdict) or None,
            studies=list(studies) or None, top_k=top_k,
        )
    except ConfigError as exc:
        _fail(str(exc))
    _echo(f"{len(result.records)} record(s), precise cohort path")
    for note in result.notes:
        _echo(f"  {note}")
    _echo(f"  {'record':<40} {'verdict':<18} {'assertion':<10} "
          f"{'availability':<16} source")
    for record in result.records:
        _echo(f"  {record.record_id[-38:]:<40} {str(record.verdict):<18} "
              f"{str(record.assertion):<10} {record.availability:<16} "
              f"{record.source_variable}")


@app.command()
def ask(
    question: str = typer.Argument(...),
    backend: str = typer.Option("deterministic", help="deterministic | llm"),
    data_dir: Optional[str] = typer.Option(None),
    save: bool = typer.Option(True),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Compile a question, execute it, and trace the number back to source."""
    from .agent import AgentSession, render_trace

    pipeline = _pipeline(data_dir)
    session = AgentSession(pipeline, question, backend=backend)
    result = session.compile()

    if result.conflict is not None:
        conflict = result.conflict
        _echo("NOT RUN — the question conflicts with the definition it names,")
        _echo("or leaves a rule underdetermined. Nothing was computed.")
        _echo()
        _echo(f"  conflict   {conflict.conflict}")
        if conflict.bound_definition:
            _echo(f"  bound to   {conflict.bound_definition}")
        _echo(f"  effect     {conflict.effect}")
        _echo("  options:")
        for option in conflict.options:
            _echo(f"    - {option}")
        _echo()
        _echo("  The agent does not override a bound definition to accommodate")
        _echo("  a question. A different rule is a new version, not a parameter.")
        raise typer.Exit(code=2)

    package, manifest = session.execute(save=save)
    body = package.to_dict()
    if as_json:
        _echo(json.dumps(body, indent=2, default=str))
        return

    spec = package.spec
    _echo(f"bound      {spec.definition_id}.v{spec.definition_version} "
          f"({spec.definition_hash[:12]})")
    _echo(f"tools      {body['tools_called']}")
    _echo()
    for modifier, screen in body["supportability"].items():
        _echo(f"  supportability for {modifier} (metadata only, no patient read):")
        _echo(f"    supported             {screen['supported']}")
        _echo(f"    via extraction        {screen['supported_via_extraction']}")
        _echo(f"    cannot ascertain      {screen['cannot_ascertain']}")
    _echo()
    cohort = body["cohort"]
    _echo(f"  verdicts           {cohort['counts_by_verdict']}")
    _echo(f"  evidence routes    {cohort['attribute_methods']}")
    _echo(f"  source variables   {cohort['attribute_sources']}")
    overall = cohort["overall"]
    _echo(f"  ascertainable      {overall['n_ascertainable']}/"
          f"{overall['n_total']} ({overall['ascertainable_fraction']:.3f})")
    _echo(f"  incidence          "
          f"{overall['incidence_within_ascertainable']} within the "
          f"ascertainable population")
    _echo()
    _echo(f"  manifest {package.manifest_id}  results {package.results_hash}")
    _echo()
    _echo(render_trace(package.trace))


@app.command()
def trace(
    manifest_id: str = typer.Argument(...),
    verdict: str = typer.Option("case"),
    data_dir: Optional[str] = typer.Option(None),
) -> None:
    """Follow a reported number back to the text a site wrote."""
    from .agent.trace import render_trace, trace_number
    from .runs import ManifestStore, ReplayError, ResultStore

    store = ManifestStore(paths.RUNS_DIR)
    try:
        manifest = store.load(manifest_id)
        assignments = ResultStore(paths.RUNS_DIR / "results").read(manifest_id)
    except ReplayError as exc:
        _fail(str(exc))

    pipeline = _pipeline(data_dir)
    chain = trace_number(
        number=manifest.counts_by_verdict.get(verdict, 0),
        label=f"{verdict} count",
        manifest=manifest,
        assignments=assignments,
        records=pipeline.records(),
        verdict=verdict,
    )
    _echo(render_trace(chain))


@app.command()
def replay(
    manifest_id: str = typer.Argument(...),
    data_dir: Optional[str] = typer.Option(None),
) -> None:
    """Reproduce a prior run and verify it matches, hash for hash."""
    from .runs import ReplayError, replay as replay_run

    try:
        report, _manifest = replay_run(manifest_id, data_dir=data_dir)
    except ReplayError as exc:
        _fail(str(exc))
    _echo(report.summary())
    for difference in report.differences:
        _echo(f"  - {difference}")
    if not report.reproduced:
        raise typer.Exit(code=1)


# -- eval -------------------------------------------------------------------


@eval_app.command("silver")
def eval_silver(
    modifier: str = typer.Option(MODIFIER),
    queue: Optional[str] = typer.Option(
        None, "--queue", help="Write the adjudication queue here."
    ),
    data_dir: Optional[str] = typer.Option(None),
    backend: str = typer.Option("auto"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Score extraction against a masked structured comparator."""
    from .silver import SILVER_CAVEATS, SilverHarness

    pipeline = _pipeline(data_dir, backend=backend)
    harness = SilverHarness(pipeline.configs, pipeline.store, pipeline.engine())
    report = harness.run(pipeline.records(), modifier)
    body = report.to_dict()
    if as_json:
        _echo(json.dumps(body, indent=2))
        return

    _echo(f"SILVER standard for {modifier!r} over {body['profiles']}")
    _echo()
    for caveat in SILVER_CAVEATS:
        _echo(f"  {caveat}")
        _echo()
    for key, value in body["overall"].items():
        _echo(f"  {key:<24} {value}")
    _echo()
    _echo(f"  {'assertion':<12} {'n':>4} {'answered':>9} {'correct':>8} "
          f"{'recall':>7} {'precision':>10}")
    for name, row in body["by_assertion"].items():
        _echo(f"  {name:<12} {row['n']:4d} {row['answered']:9d} "
              f"{row['correct']:8d} {row['recall']:7.3f} {row['precision']:10.3f}")
    _echo()
    calibration = body["calibration"]
    _echo(f"  Brier score              {calibration['brier_score']}")
    _echo(f"  expected calib. error    "
          f"{calibration['expected_calibration_error']}")
    _echo(f"  {'bin':<12} {'n':>4} {'mean conf':>10} {'observed':>9} {'gap':>7}")
    for row in calibration["reliability"]:
        _echo(f"  {row['bin']:<12} {row['n']:4d} {row['mean_confidence']:10.3f} "
              f"{row['observed_accuracy']:9.3f} {row['gap']:7.3f}")
    _echo()
    _echo(f"  {calibration['note']}")

    if queue:
        path = report.write_adjudication(queue)
        _echo()
        _echo(f"  adjudication queue -> {path} "
              f"({len(report.adjudication_queue())} rows, including a random "
              f"sample of agreements)")


@eval_app.command("transport")
def eval_transport(
    definition_id: str = typer.Argument(DEFINITION),
    version: Optional[int] = typer.Option(None),
    holdout: Optional[str] = typer.Option(
        None, "--holdout", help="Comma-separated profiles to hold out."
    ),
    data_dir: Optional[str] = typer.Option(None),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Hold out whole studies and report what protocol shift costs."""
    from .eval.transport import transportability

    pipeline = _pipeline(data_dir)
    try:
        definition = pipeline.definition(definition_id, version)
    except DefinitionError as exc:
        _fail(str(exc))
    held = [p.strip() for p in holdout.split(",")] if holdout else None
    try:
        body = transportability(pipeline, definition, held)
    except ValueError as exc:
        _fail(str(exc))
    if as_json:
        _echo(json.dumps(body, indent=2))
        return
    _echo(f"transportability for {definition.key}")
    _echo(f"  {body['note']}")
    _echo()
    _echo(f"  development  {body['development_profiles']}")
    _echo(f"  held out     {body['held_out_profiles']}")
    _echo()
    _echo(f"  {'':<26} {'development':>12} {'held out':>12}")
    for label, key in (
        ("records", "n"), ("PPV", "ppv"), ("sensitivity", "sensitivity"),
        ("not-ascertainable rate", "not_ascertainable_rate"),
    ):
        _echo(f"  {label:<26} {body['development'][key]:>12} "
              f"{body['held_out'][key]:>12}")
    _echo()
    _echo(f"  sensitivity drop {body['sensitivity_drop']}, "
          f"PPV drop {body['ppv_drop']}")
    _echo()
    _echo(f"  {body['holdout_character']}")


@eval_app.command("all")
def eval_all(
    definition_id: str = typer.Argument(DEFINITION),
    version: Optional[int] = typer.Option(None),
    report: Optional[str] = typer.Option(
        str(paths.REPORTS_DIR / "evaluation.md"), "--report"
    ),
    holdout: Optional[str] = typer.Option(None, "--holdout"),
    data_dir: Optional[str] = typer.Option(None),
    backend: str = typer.Option("auto"),
) -> None:
    """Run every harness and write the markdown report."""
    from .eval.harness import run_evaluation

    pipeline = _pipeline(data_dir, backend=backend)
    held = [p.strip() for p in holdout.split(",")] if holdout else None
    results, path = run_evaluation(
        pipeline, definition_id, version, report_path=report, holdout=held,
    )
    _echo(results["disclaimer"])
    _echo()
    _echo(f"  DECISION: {results['ablation']['decision']}")
    _echo()
    phenotype = results["phenotype"]["pooled"]
    _echo(f"  phenotype     PPV {phenotype['ppv']:.3f}, "
          f"sensitivity {phenotype['sensitivity']:.3f}, "
          f"not-ascertainable {phenotype['not_ascertainable_rate']:.3f}")
    silver = results["silver"]["overall"]
    _echo(f"  silver        precision {silver['precision']:.3f}, "
          f"recall {silver['recall']:.3f}, "
          f"abstention {silver['abstention_rate']:.3f}")
    _echo(f"  calibration   Brier "
          f"{results['silver']['calibration']['brier_score']}")
    assertion = results["assertion"]
    _echo(f"  silence       read as an assertion "
          f"{assertion['silence_read_as_an_assertion']} time(s); documented "
          f"negatives recovered {assertion['documented_negatives_recovered']}")
    transport = results["transport"]
    _echo(f"  transport     sensitivity drop "
          f"{transport['sensitivity_drop']:.3f} on held-out studies")
    invariance = results["invariance"]
    _echo(f"  invariance    {invariance['agreement_where_evidence_supports_it']:.3f} "
          f"where the evidence supports it")
    _echo(f"                {INVARIANCE_SHORT}")
    reproducibility = results["reproducibility"]
    _echo(f"  reproducible  results stable: "
          f"{reproducibility['results_stable']}")
    if path:
        _echo()
        _echo(f"  report -> {path}")


INVARIANCE_SHORT = (
    "(consistency across representations is not clinical validity)"
)


# -- knowledge ---------------------------------------------------------------


@knowledge_app.command("status")
def knowledge_status(data_dir: Optional[str] = typer.Option(None)) -> None:
    """What the registry holds. An empty registry is the expected day-one state."""
    from .knowledge import KnowledgeRegistry

    pipeline = _pipeline(data_dir)
    registry = KnowledgeRegistry.open(definitions=pipeline.definitions)
    status = registry.status()
    _echo(f"{status['manifests']} governed execution(s) recorded "
          f"({status['capture_mode']} capture)")
    _echo(f"  {status['note']}")
    _echo(f"  definitions        {status['definitions']}")
    _echo(f"  definitions used   {status['definitions_used']}")
    _echo(f"  snapshots          {status['snapshots']}")


@knowledge_app.command("tools")
def knowledge_tools() -> None:
    """The agent's tool surface: every tool, its permission and its schemas."""
    from .agent import REGISTRY

    _echo(f"{len(REGISTRY)} registered tool(s). No SQL surface; no tool writes "
          f"to a source record.")
    _echo()
    for name in sorted(REGISTRY):
        spec = REGISTRY[name]
        _echo(f"  {name:<24} {spec.permission:<16} "
              f"writes_source_records={spec.writes_source_records}")
        _echo(f"    {spec.description}")


@knowledge_app.command("backfill")
def knowledge_backfill(
    manifest: str = typer.Option(..., "--manifest", help="A manifest JSON file."),
) -> None:
    """Add one historical execution, deliberately."""
    from .knowledge import KnowledgeRegistry
    from .models import Manifest

    body = json.loads(Path(manifest).read_text(encoding="utf-8"))
    registry = KnowledgeRegistry.open()
    path = registry.backfill(Manifest.model_validate(body))
    _echo(f"backfilled {path}")
    _echo("  the registry captures forward by default; this row was added by "
          "an explicit, scoped act")


# -- demo and serve ----------------------------------------------------------


@app.command()
def demo(seed: int = typer.Option(7), limit: int = typer.Option(8)) -> None:
    """Generate, normalize, extract, evaluate, ablate — end to end, offline."""
    import collections

    from .ablation import format_ablation, run_ablation
    from .generate import generate_corpus
    from .runs import execute
    from .silver import SILVER_CAVEATS, SilverHarness

    _echo("=== 1. generate ===")
    path, manifest = generate_corpus(seed=seed)
    _echo(f"  {manifest['counts']['profiles']} studies, "
          f"{manifest['counts']['subjects']} subjects, "
          f"{manifest['counts']['ae_records']} source records -> {path}")
    _echo("  synthetic data only; no real patient records anywhere in this repo")

    pipeline = _pipeline()

    _echo()
    _echo("=== 2. normalize (deterministic path) ===")
    structured = pipeline.structured_only_records()
    _echo(f"  {len(structured)} canonical records, {pipeline.normalizer_version}")
    availabilities = collections.Counter(
        r.modifiers[MODIFIER].availability for r in structured
        if MODIFIER in r.modifiers
    )
    for name, count in sorted(availabilities.items()):
        _echo(f"    {name:<20} {count}")
    _echo("  a blank is not a value, and an observed 'no' is not a blank")
    reconciliation = collections.Counter(
        r.coded_event.reconciliation for r in structured if r.coded_event
    )
    _echo(f"  dictionary reconciliation: {dict(sorted(reconciliation.items()))}")
    _echo("  mechanical only; what does not map is flagged, never auto-recoded")

    _echo()
    _echo("=== 3. extract (model path, language variation only) ===")
    records = pipeline.records()
    engine = pipeline.engine()
    _echo(f"  backend {engine.backend.name}")
    for note in engine.notes:
        _echo(f"  {note}")
    stats = engine.stats.to_dict()
    _echo(f"  recovered {stats['recovered']} {stats['by_assertion']}, "
          f"abstained {stats['abstained']} "
          f"(rate {stats['abstention_rate']:.3f})")
    defects = [r.record_id for r in records if not r.has_full_provenance()]
    _echo(f"  {len(defects)} observed attribute(s) without a span "
          f"(every value traces to source)")

    _echo()
    _echo("=== 4. supportability (metadata only) ===")
    for row in pipeline.supportability(MODIFIER):
        _echo(f"  {row['study_id']:<10} {row['profile']:<20} {row['status']}")

    definition = pipeline.definition(DEFINITION)
    _echo()
    _echo(f"=== 5. evaluate {definition.key} ===")
    run_manifest, assignments = execute(
        pipeline, definition, question="cli demo", actor="cli", save=True
    )
    _echo(f"  verdicts {run_manifest.counts_by_verdict}")
    _echo(f"  manifest {run_manifest.manifest_id}  "
          f"results {run_manifest.results_hash}")
    _echo()
    _echo(f"  {'study':<10} {'total':>6} {'case':>5} {'non':>5} {'rev':>5} "
          f"{'n/a':>5} {'asc.f':>7}")
    for row in run_manifest.denominators:
        _echo(f"  {row['study_id']:<10} {row['n_total']:6d} {row['n_case']:5d} "
              f"{row['n_non_case']:5d} {row['n_review']:5d} "
              f"{row['n_not_ascertainable']:5d} "
              f"{row['ascertainable_fraction']:7.3f}")

    _echo()
    _echo("=== 6. silver standard ===")
    report = SilverHarness(
        pipeline.configs, pipeline.store, engine
    ).run(records, MODIFIER)
    metrics = report.metrics()
    _echo(f"  precision {metrics['precision']:.3f}, "
          f"recall {metrics['recall']:.3f}, "
          f"abstention {metrics['abstention_rate']:.3f}")
    _echo(f"  Brier {report.calibration()['brier_score']}")
    _echo()
    for caveat in SILVER_CAVEATS:
        _echo(f"  {caveat}")
        _echo()

    _echo("=== 7. value ablation ===")
    _echo(format_ablation(
        run_ablation(definition, pipeline.store, pipeline.configs)
    ))
    _echo()
    _echo(f"  Trace any number:  aelayer trace {run_manifest.manifest_id}")
    _echo(f"  Replay this run:   aelayer replay {run_manifest.manifest_id}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    data_dir: Optional[str] = typer.Option(None),
) -> None:
    """Serve the API and the single-page UI."""
    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(data_dir), host=host, port=port, log_level="info")


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
