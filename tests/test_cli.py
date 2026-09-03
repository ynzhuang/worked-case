"""Every CLI command runs, and each says which versions produced its numbers."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from aelayer.cli import app

runner = CliRunner()


@pytest.fixture(scope="module")
def data(corpus_dir):
    return ["--data-dir", str(corpus_dir)]


def _run(*args):
    result = runner.invoke(app, list(args))
    assert result.exit_code == 0, result.output + str(result.exception)
    return result.output


def test_generate_reports_the_profiles_it_rendered(tmp_path):
    output = _run("generate", "--out", str(tmp_path), "--shared", "4", "--extra", "2")
    assert "no real patient records anywhere in this repo" in output
    for profile in ("P_structured", "P_text", "P_absent", "P_both", "P_negated",
                    "P_version", "P_concept_variant"):
        assert profile in output


def test_ingest_reports_the_snapshot(corpus_dir):
    output = _run("ingest", str(corpus_dir))
    assert "snapshot" in output
    assert "ae_records" in output


def test_normalize_separates_assertion_from_availability(data):
    output = _run("normalize", *data)
    assert "assertion and availability are separate fields" in output
    assert "observed negative" in output


def test_extract_reports_abstention_and_provenance(data, tmp_path):
    output = _run("extract", *data, "--store", str(tmp_path / "s.db"))
    assert "abstention rate" in output
    assert "a valid answer" in output
    assert "every observed attribute on every record traces to a span" in output


def test_definitions_lists_and_shows(data):
    listing = _run("definitions", *data)
    assert "cutaneous_mucosal.v1" in listing
    assert "graded_toxicity.v1" in listing
    detail = _run("definitions", "cutaneous_mucosal", "--version", "2", *data)
    assert "must assert 'present'" in detail
    assert "temporal" in detail


def test_definitions_compare_requires_a_scope(data):
    result = runner.invoke(app, [
        "definitions", "cutaneous_mucosal", "--compare", "1:2", *data,
    ])
    assert result.exit_code == 1
    assert "scope" in result.output.lower()


def test_definitions_compare_reports_why_a_record_moved(data):
    output = _run(
        "definitions", "cutaneous_mucosal", "--compare", "1:2",
        "--scope", "cutaneous adverse events", *data,
    )
    assert "gained" in output
    assert "why" in output


def test_evaluate_prints_denominators_and_the_note(data):
    output = _run("evaluate", "cutaneous_mucosal", "--version", "2",
                  "--limit", "3", "--no-save", *data)
    assert "asc.f" in output
    assert "neither the numerator nor the denominator" in output
    assert "Trace any number" in output


def test_evaluate_runs_the_second_definition_unchanged(data):
    output = _run("evaluate", "graded_toxicity", "--limit", "2", "--no-save", *data)
    assert "graded_toxicity.v1" in output


def test_ablation_json_is_machine_readable(data):
    output = _run("ablation", "--json", *data)
    body = json.loads(output)
    assert body["decision"]
    assert len(body["stages"]) == 3


def test_retrieve_precise_and_discovery(data, tmp_path):
    _run("extract", *data, "--store", str(tmp_path / "s.db"))
    precise = _run("retrieve", "--assertion", "absent", "--top-k", "3",
                   "--store", str(tmp_path / "s.db"), *data)
    assert "precise cohort path" in precise
    discovery = _run("retrieve", "--text", "mucosal", "--top-k", "3",
                     "--store", str(tmp_path / "s.db"), *data)
    assert "every row above is a CANDIDATE" in discovery


def test_ask_executes_and_traces(data):
    output = _run("ask", "cutaneous events with mucosal involvement",
                  "--no-save", *data)
    assert "bound" in output
    assert "supportability" in output
    assert "cannot ascertain" in output
    assert "result" in output and "span" in output


def test_evaluate_then_trace_then_replay(data, tmp_path):
    evaluated = _run("evaluate", "cutaneous_mucosal", "--version", "2",
                     "--limit", "1", *data)
    manifest_id = evaluated.split("aelayer trace ")[1].split()[0]
    traced = _run("trace", manifest_id, *data)
    assert "case count" in traced
    replayed = _run("replay", manifest_id, *data)
    assert "reproduced exactly" in replayed


def test_eval_transport_names_the_two_sides(data):
    output = _run("eval", "transport", *data)
    assert "development" in output
    assert "held out" in output
    assert "sensitivity drop" in output


def test_eval_transport_rejects_an_unknown_holdout(data):
    result = runner.invoke(app, ["eval", "transport", "--holdout", "P_nope", *data])
    assert result.exit_code == 1
    assert "no such profile" in result.output


def test_eval_all_writes_a_report(data, tmp_path):
    output = _run("eval", "all", "--report", str(tmp_path / "r.md"), *data)
    assert "DECISION:" in output
    assert "consistency across representations is not clinical validity" in output
    assert (tmp_path / "r.md").exists()


def test_eval_silver_writes_an_adjudication_queue(data, tmp_path):
    output = _run("eval", "silver", "--queue", str(tmp_path / "q.jsonl"), *data)
    assert "sample of agreements" in output
    rows = (tmp_path / "q.jsonl").read_text().splitlines()
    assert rows


def test_knowledge_tools_lists_the_surface(data):
    output = _run("knowledge", "tools")
    assert "No SQL surface" in output
    assert "writes_source_records=False" in output


def test_knowledge_status_reports_forward_capture(data):
    output = _run("knowledge", "status", *data)
    assert "forward" in output


def test_knowledge_backfill_is_deliberate(data, tmp_path):
    evaluated = _run("evaluate", "cutaneous_mucosal", "--version", "2",
                     "--limit", "1", *data)
    manifest_id = evaluated.split("aelayer trace ")[1].split()[0]
    from aelayer import paths

    source = paths.RUNS_DIR / f"{manifest_id}.json"
    output = _run("knowledge", "backfill", "--manifest", str(source))
    assert "explicit, scoped act" in output


def test_demo_runs_the_whole_path(tmp_path, monkeypatch):
    output = _run("demo", "--seed", "5", "--limit", "3")
    for heading in ("1. generate", "2. normalize", "3. extract",
                    "4. supportability", "5. evaluate", "6. silver standard",
                    "7. value ablation"):
        assert heading in output
    assert "DECISION:" in output
    assert "CAVEAT 1" in output and "CAVEAT 2" in output


def test_a_missing_corpus_fails_with_advice(tmp_path):
    result = runner.invoke(app, ["normalize", "--data-dir", str(tmp_path / "nope")])
    assert result.exit_code == 1
    assert "aelayer generate" in result.output
