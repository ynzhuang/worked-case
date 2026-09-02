"""The CLI and the HTTP API, over the same pipeline."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from aelayer import api, paths
from aelayer.cli import app


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch

    patch = MonkeyPatch()
    yield patch
    patch.undo()


@pytest.fixture(scope="module")
def runs_dir(tmp_path_factory, monkeypatch_module):
    directory = tmp_path_factory.mktemp("runs")
    monkeypatch_module.setattr(paths, "RUNS_DIR", directory)
    return directory


@pytest.fixture(scope="module")
def client(pipeline, index, runs_dir, monkeypatch_module):
    monkeypatch_module.setattr(api, "_pipeline_singleton", lambda: pipeline)
    monkeypatch_module.setattr(api, "_runs_dir", lambda: runs_dir)
    with TestClient(api.app) as test_client:
        yield test_client


def ok(response):
    assert response.status_code == 200, response.text
    return response.json()


# --------------------------------------------------------------------- API


def test_the_summary_says_what_the_data_and_the_extractor_are(client):
    body = ok(client.get("/api/summary"))
    assert "synthetic" in body["notice"]
    assert "not a trained clinical NLP model" in body["notice"]
    assert "illustrative placeholders" in body["notice"]


def test_the_summary_shows_where_each_profile_keeps_the_attribute(client):
    body = ok(client.get("/api/summary"))
    assert len(body["profiles"]) == 6
    homes = {tuple(p["location_home"]) for p in body["profiles"].values()}
    assert len(homes) == 6
    both = [p for p in body["profiles"].values() if p["carries_both_location"]]
    assert len(both) == 1


def test_records_can_be_filtered_by_route(client):
    body = ok(client.get("/api/records", params={"method": "extracted", "limit": 5}))
    assert body["count"]
    assert {r["location_method"] for r in body["records"]} == {"extracted"}


def test_one_record_exposes_every_attribute_with_its_route(client):
    listing = ok(client.get("/api/records", params={"limit": 1}))
    record_id = listing["records"][0]["source_record_id"]
    body = ok(client.get(f"/api/records/{record_id}"))
    assert body["provenance_complete"]
    for attribute in body["attributes"].values():
        if attribute["value"] is not None:
            assert attribute["evidence"]
            assert attribute["method"]
            assert attribute["source_variable"]


def test_an_unknown_record_is_a_404(client):
    assert client.get("/api/records/NOPE").status_code == 404


def test_an_episode_shows_the_records_it_was_derived_from(client):
    listing = ok(client.get("/api/episodes", params={"limit": 1}))
    episode_id = listing["episodes"][0]["episode_id"]
    body = ok(client.get(f"/api/episodes/{episode_id}"))
    assert body["records"]
    assert len(body["records"]) == len(body["episode"]["source_record_ids"])


def test_a_trajectory_is_served_per_subject(client):
    listing = ok(client.get("/api/episodes", params={"limit": 1}))
    subject = listing["episodes"][0]["subject_id"]
    body = ok(client.get(f"/api/trajectories/{subject}"))
    assert body["events"]
    assert {e["kind"] for e in body["events"]} <= {"exposure", "episode"}


def test_evaluating_reports_the_unascertainable_separately(client):
    body = ok(client.post("/evaluate", json={
        "definition_id": "te_truncal_rash", "version": 1,
    }))
    manifest = body["manifest"]
    assert manifest["counts_by_verdict"]["not_ascertainable"] == len(
        body["not_ascertainable"]
    )
    assert manifest["attribute_methods"]


def test_the_silver_endpoint_labels_itself_and_carries_a_queue(client):
    body = ok(client.get("/eval/silver"))
    assert body["standard"] == "silver"
    assert "not ground truth" in body["caveat"]
    assert body["adjudication_queue"]
    assert any(
        r["queue_reason"].startswith("sampled agreement")
        for r in body["adjudication_queue"]
    )


def test_the_ablation_endpoint_names_the_business_case(client):
    body = ok(client.get("/eval/ablation"))
    assert body["cases_only_findable_through_text"] > 0
    assert body["cases_with_text"] > body["cases_structured_only"]


def test_the_transport_endpoint_holds_out_whole_studies(client):
    body = ok(client.get("/eval/transport"))
    assert body["split"] == "whole_study"
    assert not set(body["development_profiles"]) & set(body["held_out_profiles"])


def test_holding_out_an_unknown_profile_is_a_400(client):
    assert client.get(
        "/eval/transport", params={"holdout": "P9_imaginary"}
    ).status_code == 400


def test_the_precise_path_is_a_cohort_and_discovery_is_not(client):
    precise = ok(client.get("/retrieve", params={"concept": "RASH"}))
    assert precise["usable_as_cohort"]
    discovery = ok(client.get("/discover", params={"attribute": "location"}))
    assert discovery["usable_as_cohort"] is False
    assert discovery["all_candidates"]
    assert "never directly" in discovery["cohort_note"]


def test_discovery_can_be_asked_for_what_the_catalogue_misses(client):
    body = ok(client.get("/discover", params={"unnormalized_only": True}))
    assert body["unnormalized_count"] == body["count"]
    assert body["count"] > 0


def test_comparing_versions_without_a_scope_is_refused(client):
    response = client.get(
        "/definitions/te_truncal_rash/compare", params={"left": 1, "right": 2}
    )
    assert response.status_code == 400
    assert "requires an explicit scope" in response.json()["detail"]


def test_a_scoped_comparison_shows_what_the_route_change_costs(client):
    body = ok(client.get("/definitions/te_truncal_rash/compare", params={
        "left": 1, "right": 2, "scope": "truncal rash incidence",
    }))
    assert body["lost"]
    assert any("does not accept" in d["reason_b"] for d in body["discordant"])


def test_a_candidate_is_rendered_and_nothing_on_disk_changes(client, pipeline):
    before = pipeline.definition("te_truncal_rash", 1).definition_hash
    body = ok(client.post("/definitions/candidate", json={
        "definition_id": "te_truncal_rash", "base_version": 1,
        "changes": {"required_attributes.location.min_confidence": 0.9},
    }))
    assert body["applied_changes"] == [
        "required_attributes.location.min_confidence = 0.9"
    ]
    assert "has not been modified" in body["note"]
    assert pipeline.definitions.get(
        "te_truncal_rash", 1
    ).definition_hash == before


def test_a_candidate_naming_a_path_that_does_not_exist_is_refused(client):
    response = client.post("/definitions/candidate", json={
        "definition_id": "te_truncal_rash", "base_version": 1,
        "changes": {"required_attributes.location.vibes": 1},
    })
    assert response.status_code == 400
    assert "no such definition path" in response.json()["detail"]


def test_the_tool_surface_is_published_with_its_schemas(client):
    body = ok(client.get("/agent/tools"))
    assert len(body["tools"]) >= 5
    assert all(t["writes_source_records"] is False for t in body["tools"])
    assert "no SQL surface" in body["note"]


def test_an_underdetermined_question_returns_no_number(client):
    response = client.post("/agent/ask", json={
        "question": "how many severe rash cases were hospitalised?", "save": False,
    })
    assert response.status_code == 409
    body = response.json()
    assert body["executed"] is False
    assert "no number was produced" in body["detail"]


def test_asking_executes_without_an_approval_step_and_returns_a_trace(client):
    body = ok(client.post("/agent/ask", json={
        "question": "how many rash cases after first exposure?",
    }))
    assert body["executed"] is True
    assert body["traceable"] is True
    assert body["cohort"]["counts_by_verdict"]
    assert body["tools_called"]


def test_a_run_replays_hash_for_hash_and_traces_to_a_span(client):
    manifest = ok(client.post("/evaluate", json={
        "definition_id": "te_truncal_rash", "version": 1,
    }))["manifest"]
    replayed = ok(client.post(f"/runs/{manifest['manifest_id']}/replay"))
    assert replayed["reproduced"], replayed["differences"]
    traced = ok(client.get(f"/trace/{manifest['manifest_id']}"))
    assert traced["complete"]
    assert traced["links"][-1]["level"] == "span"


def test_the_ui_is_served_from_the_same_process(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "<html" in response.text.lower()


# --------------------------------------------------------------------- CLI


@pytest.fixture(scope="module")
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(scope="module")
def store_path(tmp_path_factory):
    return str(tmp_path_factory.mktemp("cli-store") / "store.db")


def run(runner, *args, expect: int = 0):
    result = runner.invoke(app, list(args))
    assert result.exit_code == expect, result.output + str(result.exception)
    return result.output


def common(corpus_dir, store_path):
    return ["--data-dir", str(corpus_dir), "--store", store_path]


@pytest.fixture(scope="module")
def built(runner, corpus_dir, store_path, runs_dir):
    return run(runner, "extract", "--data-dir", str(corpus_dir), "--out", store_path)


def test_generate_says_the_corpus_is_synthetic(runner, tmp_path_factory):
    out = tmp_path_factory.mktemp("gen-cli")
    output = run(runner, "generate", "--seed", "3", "--truths", "2",
                 "--background", "1", "--out", str(out))
    assert "No real patient data" in output
    for profile in ("P1_structured", "P6_both"):
        assert profile in output


def test_normalize_reports_the_route_per_profile(runner, corpus_dir):
    output = run(runner, "normalize", "--data-dir", str(corpus_dir))
    assert "location, by profile and route" in output
    assert "a route is part of the fact" in output


def test_extract_reports_versions_and_provenance(built):
    assert "normalizer" in built and "extractor" in built
    assert "every populated attribute on every record traces to a span" in built


def test_definitions_show_which_routes_each_accepts(runner, corpus_dir):
    output = run(runner, "definitions", "--data-dir", str(corpus_dir))
    assert "te_truncal_rash.v1" in output
    assert "accepts=direct,extracted,normalized" in output


def test_comparing_without_a_scope_is_refused(runner, corpus_dir):
    result = runner.invoke(app, [
        "definitions", "--data-dir", str(corpus_dir),
        "--compare", "te_truncal_rash:1:2",
    ])
    assert result.exit_code == 1
    assert "requires an explicit scope" in result.output


@pytest.fixture(scope="module")
def evaluated(runner, corpus_dir, store_path, built):
    return run(runner, "evaluate", *common(corpus_dir, store_path), "--version", "1")


def test_evaluate_reports_the_unascertainable_apart(evaluated):
    assert "not ascertainable" in evaluated
    assert "Not a negative" in evaluated or "neither cases nor negatives" in evaluated


def test_evaluate_prints_the_route_behind_each_verdict(evaluated):
    assert "routes" in evaluated
    assert "location=" in evaluated


def test_silver_reports_every_metric_and_writes_a_queue(
    runner, corpus_dir, store_path, built, tmp_path_factory
):
    queue = tmp_path_factory.mktemp("queue") / "adjudication.jsonl"
    output = run(runner, "eval", "silver", "--queue", str(queue),
                 *common(corpus_dir, store_path))
    for label in ("precision", "recall", "coverage", "abstention rate",
                  "normalized agreement"):
        assert label in output
    assert "silver standard, not ground truth" in output
    assert "sampled agreement" in output
    rows = [json.loads(line) for line in queue.read_text().splitlines() if line]
    assert rows


def test_transport_holds_out_whole_studies(runner, corpus_dir, store_path, built):
    output = run(runner, "eval", "transport", *common(corpus_dir, store_path))
    assert "never rows" in output
    assert "development" in output and "held_out" in output


def test_eval_all_writes_a_report(runner, corpus_dir, store_path, built,
                                  tmp_path_factory):
    report = tmp_path_factory.mktemp("report") / "eval.md"
    output = run(runner, "eval", "all", "--report", str(report),
                 *common(corpus_dir, store_path))
    assert "silver" in output and "ablation" in output
    body = report.read_text(encoding="utf-8")
    assert "Consistency across representations is not clinical validity" in body


def test_an_underdetermined_question_stops_before_executing(
    runner, corpus_dir, store_path, built
):
    result = runner.invoke(app, [
        "ask", "how many severe rash cases were hospitalised?",
        *common(corpus_dir, store_path),
    ])
    assert result.exit_code == 2
    assert "No specification was compiled and nothing was executed" in result.output


def test_asking_traces_the_number_it_reports(runner, corpus_dir, store_path, built):
    output = run(runner, "ask", "how many rash cases after first exposure?",
                 *common(corpus_dir, store_path))
    assert "traceable to source: True" in output
    assert "tools" in output


def test_the_tool_surface_is_listed(runner):
    output = run(runner, "knowledge", "tools")
    assert "cohort.run" in output
    assert "no SQL surface" in output


def test_a_recorded_run_replays(runner, corpus_dir, store_path, evaluated, runs_dir):
    manifest_id = [
        line.split()[1] for line in evaluated.splitlines()
        if line.strip().startswith("manifest ")
    ][0]
    output = run(runner, "replay", manifest_id, "--data-dir", str(corpus_dir))
    assert "reproduced exactly" in output


def test_retrieval_separates_the_two_paths(runner, corpus_dir, store_path, evaluated):
    precise = run(runner, "retrieve", "RASH", *common(corpus_dir, store_path))
    assert "usable as a cohort: True" in precise
    discovery = run(runner, "retrieve", "rash", "--mode", "lexical",
                    *common(corpus_dir, store_path))
    assert "every result is a candidate" in discovery
