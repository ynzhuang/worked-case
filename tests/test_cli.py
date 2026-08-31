"""The command line surface, end to end."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from aelayer.cli import app

runner = CliRunner()


@pytest.fixture(scope="module")
def cli_env(tmp_path_factory):
    """A corpus and store the CLI owns, so tests never touch the repo's."""
    root = tmp_path_factory.mktemp("cli")
    data = root / "data"
    result = runner.invoke(
        app, ["generate", "--seed", "4", "--studies", "2", "--subjects", "12",
              "--out", str(data)]
    )
    assert result.exit_code == 0, result.output
    store = str(root / "store.db")
    result = runner.invoke(app, ["extract", "--out", store, "--data-dir", str(data)])
    assert result.exit_code == 0, result.output
    return {"data": str(data), "store": store}


def test_generate_says_the_data_is_synthetic(cli_env, tmp_path):
    result = runner.invoke(
        app, ["generate", "--seed", "1", "--studies", "2", "--subjects", "5",
              "--out", str(tmp_path / "d")]
    )
    assert result.exit_code == 0
    assert "No real patient data" in result.output


def test_ingest_reports_the_corpus(cli_env):
    result = runner.invoke(app, ["ingest", cli_env["data"]])
    assert result.exit_code == 0
    assert "snapshot_id" in result.output and "ae_records" in result.output


def test_extract_confirms_the_provenance_invariant(cli_env):
    result = runner.invoke(
        app, ["extract", "--out", cli_env["store"], "--data-dir", cli_env["data"]]
    )
    assert result.exit_code == 0
    assert "every populated field on every event traces to a span" in result.output


def test_definitions_lists_versions_with_status_and_hash(cli_env):
    result = runner.invoke(app, ["definitions"])
    assert result.exit_code == 0
    assert "te_symptomatic_hypoglycemia.v1" in result.output
    assert "frozen" in result.output


def test_definitions_diff_isolates_the_threshold(cli_env):
    result = runner.invoke(
        app, ["definitions", "--diff", "te_symptomatic_hypoglycemia:1:2"]
    )
    assert result.exit_code == 0
    assert "lab.value: 70 -> 54" in result.output


def test_evaluate_prints_a_case_table_with_reasons(cli_env):
    result = runner.invoke(app, [
        "evaluate", "--version", "1", "--data-dir", cli_env["data"],
        "--store", cli_env["store"], "--no-save",
    ])
    assert result.exit_code == 0
    assert "verdicts" in result.output
    assert "rule 'explicit'" in result.output or "rule 'supported'" in result.output
    assert "review set" in result.output


def test_evaluate_refuses_an_unknown_definition(cli_env):
    result = runner.invoke(app, [
        "evaluate", "--definition", "no_such_phenotype",
        "--data-dir", cli_env["data"], "--store", cli_env["store"],
    ])
    assert result.exit_code == 1


def test_retrieve_shows_the_assertion_filter_effect(cli_env):
    on = runner.invoke(app, [
        "retrieve", "HYPOGLYCEMIA", "--assertion", "present", "--top-k", "200",
        "--data-dir", cli_env["data"], "--store", cli_env["store"], "--json",
    ])
    off = runner.invoke(app, [
        "retrieve", "HYPOGLYCEMIA", "--top-k", "200",
        "--data-dir", cli_env["data"], "--store", cli_env["store"], "--json",
    ])
    assert on.exit_code == 0 and off.exit_code == 0
    on_body = json.loads(on.output)
    off_body = json.loads(off.output)
    assert on_body["negation_false_positives"] == 0
    assert off_body["negation_false_positives"] > 0


def test_retrieve_rejects_a_malformed_window(cli_env):
    result = runner.invoke(app, [
        "retrieve", "HYPOGLYCEMIA", "--window", "nonsense",
        "--data-dir", cli_env["data"], "--store", cli_env["store"],
    ])
    assert result.exit_code == 1


def test_ask_without_approve_compiles_but_does_not_execute(cli_env):
    result = runner.invoke(app, [
        "ask", "symptomatic hypoglycemia within 14 days of escalation",
        "--data-dir", cli_env["data"], "--store", cli_env["store"],
    ])
    assert result.exit_code == 0
    assert "nothing has been executed yet" in result.output
    assert "Execution is blocked" in result.output
    assert "primary cases" not in result.output


def test_ask_with_approve_executes_and_states_its_limits(cli_env):
    result = runner.invoke(app, [
        "ask", "symptomatic hypoglycemia within 14 days of escalation", "--approve",
        "--data-dir", cli_env["data"], "--store", cli_env["store"],
    ])
    assert result.exit_code == 0
    assert "Approved and executed" in result.output
    assert "primary cases" in result.output
    assert "review set" in result.output
    assert "limitations" in result.output


def test_an_ambiguous_question_exits_with_a_clarification(cli_env):
    result = runner.invoke(app, [
        "ask", "serious severe hypoglycemia cases", "--approve",
        "--data-dir", cli_env["data"], "--store", cli_env["store"],
    ])
    assert result.exit_code == 2
    assert "Clarification needed" in result.output
    assert "nothing was executed" in result.output.lower()


def test_replay_reproduces_a_recorded_run(cli_env, tmp_path, monkeypatch):
    from aelayer import runs as runs_module

    runs_dir = tmp_path / "runs"
    monkeypatch.setattr(runs_module.paths, "RUNS_DIR", runs_dir)

    evaluate = runner.invoke(app, [
        "evaluate", "--version", "1", "--data-dir", cli_env["data"],
        "--store", cli_env["store"],
    ])
    assert evaluate.exit_code == 0
    run_id = [
        line.split()[1] for line in evaluate.output.splitlines()
        if line.strip().startswith("run ")
    ][0]

    listed = runner.invoke(app, ["runs"])
    assert run_id in listed.output

    replayed = runner.invoke(app, ["replay", run_id, "--data-dir", cli_env["data"]])
    assert replayed.exit_code == 0, replayed.output
    assert "reproduced exactly" in replayed.output


def test_replay_of_an_unknown_run_fails_loudly(tmp_path, monkeypatch):
    from aelayer import runs as runs_module

    monkeypatch.setattr(runs_module.paths, "RUNS_DIR", tmp_path / "empty")
    result = runner.invoke(app, ["replay", "notarun"])
    assert result.exit_code == 1


def test_eval_writes_a_report_with_real_numbers(cli_env, tmp_path):
    report = tmp_path / "eval.md"
    result = runner.invoke(app, [
        "eval", "--report", str(report), "--data-dir", cli_env["data"],
        "--store", cli_env["store"],
    ])
    assert result.exit_code == 0, result.output
    assert "negation FP rate" in result.output
    body = report.read_text(encoding="utf-8")
    for heading in ("## 1. Extraction", "## 2. Phenotype", "## 3. Retrieval",
                    "## 4. Stability", "## 5. Definition sensitivity"):
        assert heading in body
