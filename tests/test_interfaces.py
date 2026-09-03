"""The CLI, the API and the agent — the three surfaces, on one pipeline."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from aelayer.agent import AgentServices, AgentSession, ConflictUnresolved, ToolError
from aelayer.cli import app
from aelayer.retrieval import CandidateInCohort

runner = CliRunner()


# -- the agent binds a definition ---------------------------------------------


def test_the_agent_binds_a_version_and_stamps_its_hash(pipeline):
    session = AgentSession(pipeline, "cutaneous events with mucosal involvement")
    result = session.compile()
    assert result.conflict is None
    assert result.spec.definition_id == "cutaneous_mucosal"
    assert result.spec.definition_hash
    definition = pipeline.definition(
        result.spec.definition_id, result.spec.definition_version
    )
    assert result.spec.definition_hash == definition.definition_hash


def test_the_spec_carries_no_parameters_of_its_own(pipeline):
    """It names a version. Windows and thresholds live in the frozen file."""
    session = AgentSession(pipeline, "cutaneous events with mucosal involvement")
    body = session.compile().spec.model_dump()
    for invented in ("window", "min_confidence", "require_assertion", "grade"):
        assert invented not in body


@pytest.mark.parametrize(
    "question, expected",
    [
        ("cutaneous events with mucosal involvement within 14 days", "14-day window"),
        ("cutaneous events without mucosal involvement", "two readings"),
        ("severe cutaneous events requiring hospitalisation", "seriousness language"),
        ("how many patients had a stroke?", "does not name a phenotype"),
    ],
)
def test_a_question_that_conflicts_is_refused_not_accommodated(pipeline, question,
                                                               expected):
    session = AgentSession(pipeline, question)
    result = session.compile()
    assert result.conflict is not None, f"{question!r} was silently accommodated"
    assert expected in result.conflict.conflict
    assert result.conflict.options
    with pytest.raises(ConflictUnresolved):
        session.execute(save=False)


def test_the_window_conflict_offers_a_new_version_not_an_override(pipeline):
    session = AgentSession(
        pipeline, "cutaneous events with mucosal involvement within 14 days"
    )
    conflict = session.compile().conflict
    assert any("Create v" in option for option in conflict.options)
    assert any("as written" in option for option in conflict.options)
    assert "new definition version" in conflict.effect


def test_the_negative_conflict_names_both_populations(pipeline):
    session = AgentSession(pipeline, "cutaneous events without mucosal involvement")
    conflict = session.compile().conflict
    joined = " ".join(conflict.options)
    assert "assertion=absent" in joined
    assert "not_collected" in joined
    assert "denominator" in conflict.effect


# -- the supportability screen ------------------------------------------------


def test_supportability_runs_before_any_patient_level_query(pipeline):
    session = AgentSession(pipeline, "cutaneous events with mucosal involvement")
    package, _manifest = session.execute(save=False)
    calls = package.tools_called
    assert "study.supportability" in calls
    assert calls.index("study.supportability") < calls.index("cohort.run")
    screen = package.supportability["mucosal_involvement"]
    assert screen["cannot_ascertain"]


# -- the tool surface ---------------------------------------------------------


def test_no_tool_takes_a_query_string_or_writes(pipeline):
    for spec in AgentServices(pipeline).catalogue():
        assert spec["writes_source_records"] is False
        properties = spec["input_schema"].get("properties", {})
        for name in properties:
            assert name not in ("sql", "query", "where", "raw"), (
                f"{spec['name']} exposes a query surface"
            )


def test_a_tool_outside_the_grant_is_refused(pipeline):
    services = AgentServices(pipeline, permissions={"read_cohort"})
    with pytest.raises(ToolError) as exc:
        services.call("stats.compare", definition_id="cutaneous_mucosal",
                      left=1, right=2, scope="x")
    assert "requires the 'analyse' permission" in str(exc.value)


def test_arguments_are_validated_before_a_tool_runs(pipeline):
    services = AgentServices(pipeline)
    with pytest.raises(ToolError) as exc:
        services.call("cohort.run", definition_id="cutaneous_mucosal", nonsense=1)
    assert "do not validate" in str(exc.value)


def test_an_unscoped_comparison_is_refused(pipeline):
    services = AgentServices(pipeline)
    with pytest.raises(Exception) as exc:
        services.call("stats.compare", definition_id="cutaneous_mucosal",
                      left=1, right=2, scope=None)
    assert "scope" in str(exc.value).lower()


def test_the_export_leaves_unascertained_subjects_null(pipeline):
    body = AgentServices(pipeline).call(
        "cohort.export", definition_id="cutaneous_mucosal", version=2
    )
    statuses = {row["status"] for row in body["rows"]}
    assert None in statuses
    for row in body["rows"]:
        if row["verdict"] in ("review", "not_ascertainable"):
            assert row["status"] is None
    assert "unadjudicated judgement" in body["note"]


# -- traceability -------------------------------------------------------------


def test_every_reported_number_traces_to_source(pipeline):
    session = AgentSession(pipeline, "cutaneous events with mucosal involvement")
    package, _manifest = session.execute(save=False)
    assert package.trace.complete, f"chain breaks at {package.trace.broken_at}"
    levels = set(package.trace.levels())
    assert {"result", "analysis", "cohort", "definition", "record", "span"} <= levels


# -- retrieval ----------------------------------------------------------------


def test_discovery_output_cannot_become_a_cohort(pipeline):
    result = pipeline.discover(text="mucosal", top_k=5)
    with pytest.raises(CandidateInCohort):
        result.as_cohort()


def test_the_precise_path_is_cohort_eligible(pipeline, index):
    result = pipeline.retrieve(assertion=["present"], top_k=5)
    assert result.as_cohort() is not None


def test_assertion_and_availability_are_separate_filters(pipeline, index):
    absent = pipeline.retrieve(assertion=["absent"], top_k=200)
    silent = pipeline.retrieve(availability=["not_collected"], top_k=200)
    assert absent.records and silent.records
    assert not (
        {r.record_id for r in absent.records}
        & {r.record_id for r in silent.records}
    ), "a documented negative and an uncollected variable overlap"


# -- the CLI ------------------------------------------------------------------


def test_cli_help_lists_the_two_headline_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ablation" in result.output
    assert "supportability" in result.output


def test_cli_ask_exits_non_zero_on_a_conflict(corpus_dir):
    result = runner.invoke(app, [
        "ask", "cutaneous events without mucosal involvement",
        "--data-dir", str(corpus_dir), "--no-save",
    ])
    assert result.exit_code == 2
    assert "NOT RUN" in result.output
    assert "does not override" in result.output


def test_cli_supportability_reads_no_patient_record(corpus_dir):
    result = runner.invoke(app, ["supportability", "--data-dir", str(corpus_dir)])
    assert result.exit_code == 0
    assert "no patient record was read" in result.output
    assert "cannot_ascertain" in result.output


def test_cli_ablation_ends_in_a_decision(corpus_dir):
    result = runner.invoke(app, ["ablation", "--data-dir", str(corpus_dir)])
    assert result.exit_code == 0
    assert "DECISION:" in result.output


def test_cli_silver_prints_both_caveats(corpus_dir):
    from aelayer.silver import SILVER_CAVEATS

    result = runner.invoke(app, ["eval", "silver", "--data-dir", str(corpus_dir)])
    assert result.exit_code == 0
    for caveat in SILVER_CAVEATS:
        # The CLI wraps, so compare on a distinctive fragment.
        assert caveat.split(".")[0] in result.output.replace("\n", " ")


def test_cli_normalize_reports_the_reconciliation_split(corpus_dir):
    result = runner.invoke(app, ["normalize", "--data-dir", str(corpus_dir)])
    assert result.exit_code == 0
    for outcome in ("unchanged", "remapped_mechanically", "flagged_for_review"):
        assert outcome in result.output
