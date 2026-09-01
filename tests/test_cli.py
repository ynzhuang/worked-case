"""The command line. Every command prints the versions behind its numbers."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from aelayer import paths
from aelayer.cli import app


@pytest.fixture(scope="module")
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(scope="module")
def store_path(tmp_path_factory):
    return str(tmp_path_factory.mktemp("cli-store") / "store.db")


@pytest.fixture(scope="module", autouse=True)
def runs_dir(tmp_path_factory, monkeypatch_module):
    directory = tmp_path_factory.mktemp("cli-runs")
    monkeypatch_module.setattr(paths, "RUNS_DIR", directory)
    return directory


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch

    patch = MonkeyPatch()
    yield patch
    patch.undo()


def run(runner, *args, expect: int = 0):
    result = runner.invoke(app, list(args))
    assert result.exit_code == expect, result.output + str(result.exception)
    return result.output


@pytest.fixture(scope="module")
def built(runner, corpus_dir, store_path):
    return run(runner, "extract", "--data-dir", str(corpus_dir), "--out", store_path)


def common(corpus_dir, store_path):
    return ["--data-dir", str(corpus_dir), "--store", store_path]


# -- generate and ingest ----------------------------------------------------


def test_generate_says_the_corpus_is_synthetic(runner, tmp_path):
    output = run(
        runner, "generate", "--seed", "3", "--truths", "2", "--background", "1",
        "--out", str(tmp_path / "corpus"),
    )
    assert "No real patient data" in output
    assert (tmp_path / "corpus" / "ae.csv").exists()


def test_generate_reports_each_study_representation(runner, tmp_path):
    output = run(
        runner, "generate", "--seed", "3", "--truths", "2", "--background", "1",
        "--out", str(tmp_path / "corpus"),
    )
    for representation in ("V-A", "V-B", "V-C", "V-D", "V-E", "V-F"):
        assert representation in output


def test_ingest_reports_what_it_loaded(runner, corpus_dir):
    output = run(runner, "ingest", str(corpus_dir))
    assert "studies" in output
    assert "subjects" in output


# -- normalize and extract --------------------------------------------------


def test_normalize_reports_the_collection_states_it_assigned(runner, corpus_dir):
    output = run(runner, "normalize", "--data-dir", str(corpus_dir))
    assert "collection states:" in output
    assert "not_collected_by_protocol" in output
    assert "a blank is not a value" in output or "traces to a span" in output


def test_extract_reports_the_versions_and_the_linkage_review_set(built):
    assert "normalizer" in built
    assert "extractor" in built
    assert "snapshot" in built
    assert "flagged for linkage review" in built
    assert "every populated field on every record traces to a span" in built


# -- definitions ------------------------------------------------------------


def test_definitions_lists_versions_with_their_hashes(runner, corpus_dir):
    output = run(runner, "definitions", "--data-dir", str(corpus_dir))
    assert "te_symptomatic_hypoglycemia.v1" in output
    assert "frozen" in output


def test_comparing_definitions_without_a_scope_is_refused(runner, corpus_dir):
    result = runner.invoke(app, [
        "definitions", "--data-dir", str(corpus_dir),
        "--compare", "te_symptomatic_hypoglycemia:1:2",
    ])
    assert result.exit_code == 1
    assert "requires an explicit scope" in result.output


def test_a_scoped_comparison_names_the_episodes_that_moved(runner, corpus_dir):
    output = run(
        runner, "definitions", "--data-dir", str(corpus_dir),
        "--compare", "te_symptomatic_hypoglycemia:1:2",
        "--scope", "hypoglycemia incidence question",
    )
    assert "discordant" in output
    assert "why" in output


# -- evaluate ---------------------------------------------------------------


@pytest.fixture(scope="module")
def evaluated(runner, corpus_dir, store_path, built):
    return run(runner, "evaluate", *common(corpus_dir, store_path), "--version", "1")


def test_evaluate_prints_the_definition_hash_and_the_manifest(evaluated):
    assert "definition  te_symptomatic_hypoglycemia.v1" in evaluated
    assert "hash=" in evaluated
    assert "manifest" in evaluated and "results=" in evaluated


def test_every_case_row_names_the_rule_that_decided_it(evaluated):
    assert "rule" in evaluated
    assert "explicit" in evaluated or "supported" in evaluated


def test_the_review_set_is_reported_separately(evaluated):
    assert "review set, reported separately" in evaluated


def test_evaluating_an_unknown_definition_fails_loudly(runner, corpus_dir, store_path):
    result = runner.invoke(app, [
        "evaluate", *common(corpus_dir, store_path), "--definition", "nope",
    ])
    assert result.exit_code == 1
    assert "no definition" in result.output


# -- retrieve and discover --------------------------------------------------


def test_precise_retrieval_reports_whether_it_is_cohort_usable(
    runner, corpus_dir, store_path, evaluated
):
    output = run(
        runner, "retrieve", "HYPOGLYCEMIA", *common(corpus_dir, store_path)
    )
    assert "precise cohort path" in output
    assert "usable as a cohort: True" in output


def test_discovery_reports_its_negation_false_positive_rate(
    runner, corpus_dir, store_path, built
):
    output = run(
        runner, "retrieve", "HYPOGLYCEMIA", "--mode", "lexical",
        "--assertion", "present", *common(corpus_dir, store_path),
    )
    assert "discovery path" in output
    assert "mentions asserting absence: 0" in output


def test_retrieval_can_be_asked_for_json(runner, corpus_dir, store_path, evaluated):
    output = run(
        runner, "retrieve", "HYPOGLYCEMIA", "--json",
        *common(corpus_dir, store_path),
    )
    body = json.loads(output)
    assert body["mode"] == "precise"
    assert body["usable_as_cohort"] is True


# -- ask, trace, replay -----------------------------------------------------


def test_an_underdetermined_question_stops_before_executing(
    runner, corpus_dir, store_path, built
):
    result = runner.invoke(app, [
        "ask", "how many severe hypoglycemia cases required hospitalisation?",
        *common(corpus_dir, store_path),
    ])
    assert result.exit_code == 2
    assert "Clarification needed" in result.output
    assert "No specification was compiled and nothing was executed" in result.output


@pytest.fixture(scope="module")
def asked(runner, corpus_dir, store_path, built):
    return run(
        runner, "ask", "how many subjects had symptomatic hypoglycemia?",
        *common(corpus_dir, store_path),
    )


def test_asking_prints_the_specification_it_compiled(asked):
    assert "Compiled specification:" in asked
    assert "definition_id" in asked


def test_asking_traces_the_number_it_reports(asked):
    assert "traceable to source: True" in asked
    assert "number" in asked


def test_asking_states_its_limitations(asked):
    assert "limitations:" in asked
    assert "synthetic" in asked


def test_a_recorded_run_replays(runner, corpus_dir, store_path, evaluated, runs_dir):
    manifest_id = [
        line.split()[1] for line in evaluated.splitlines()
        if line.strip().startswith("manifest ")
    ][0]
    output = run(runner, "replay", manifest_id, "--data-dir", str(corpus_dir))
    assert "reproduced exactly" in output


def test_replaying_an_unknown_run_fails_with_what_it_knows(runner, corpus_dir):
    result = runner.invoke(app, ["replay", "nope", "--data-dir", str(corpus_dir)])
    assert result.exit_code == 1
    assert "no manifest" in result.output


def test_a_number_traces_back_to_source_text(
    runner, corpus_dir, store_path, evaluated
):
    manifest_id = [
        line.split()[1] for line in evaluated.splitlines()
        if line.strip().startswith("manifest ")
    ][0]
    output = run(runner, "trace", manifest_id, *common(corpus_dir, store_path))
    assert "number" in output
    assert "span" in output


# -- eval and knowledge -----------------------------------------------------


def test_eval_writes_a_report_and_prints_the_headline_numbers(
    runner, corpus_dir, store_path, tmp_path_factory, built
):
    report = tmp_path_factory.mktemp("report") / "eval.md"
    output = run(
        runner, "eval", "--report", str(report), *common(corpus_dir, store_path),
    )
    assert "layer 1" in output and "layer 3" in output
    assert "invariance" in output
    assert "provenance 0 violation(s)" in output
    body = report.read_text(encoding="utf-8")
    assert "Consistency across representations does not establish clinical validity" in body


def test_knowledge_status_says_how_it_accrues(runner, corpus_dir):
    output = run(runner, "knowledge", "status", "--data-dir", str(corpus_dir))
    assert "Capture mode:       forward" in output
    assert "backfill" in output
