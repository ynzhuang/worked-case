"""The HTTP surface."""

from __future__ import annotations

import warnings

import pytest

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def client(pipeline, monkeypatch_session=None):
    from fastapi.testclient import TestClient

    import aelayer.api as api_module

    api_module._pipeline_singleton.cache_clear()
    api_module._SESSIONS.clear()
    original = api_module.get_pipeline
    api_module.get_pipeline = lambda: pipeline
    api_module.app.dependency_overrides = {}
    try:
        yield TestClient(api_module.app)
    finally:
        api_module.get_pipeline = original


def test_summary_states_what_the_system_is(client):
    body = client.get("/api/summary").json()
    assert "synthetic" in body["notice"].lower()
    assert "not a trained clinical nlp model" in body["notice"].lower()
    assert body["studies"] and body["concepts"]


def test_extract_reports_provenance_violations(client):
    body = client.post("/extract", json={"refresh": False}).json()
    assert body["events"] > 0
    assert body["provenance_violations"] == []
    assert body["extractor_version"] and body["snapshot_id"]


def test_a_document_comes_back_with_its_event_objects(client):
    doc_id = client.get("/api/documents?limit=1").json()["documents"][0]["doc_id"]
    body = client.get(f"/api/documents/{doc_id}").json()
    assert body["text"]
    for event in body["events"]:
        for span in event["evidence"]:
            if span["doc_id"] == doc_id:
                assert body["text"][span["start"]:span["end"]] == span["text"]


def test_a_missing_document_is_a_404(client):
    assert client.get("/api/documents/nope").status_code == 404


def test_definitions_are_listed_with_status_and_hash(client):
    body = client.get("/definitions").json()["definitions"]
    keys = {d["key"] for d in body}
    assert {"te_symptomatic_hypoglycemia.v1", "te_symptomatic_hypoglycemia.v2"} <= keys
    assert all(d["hash"] and d["status"] for d in body)


def test_the_diff_endpoint_isolates_the_changed_threshold(client):
    body = client.get(
        "/definitions/te_symptomatic_hypoglycemia/diff?left=1&right=2"
    ).json()
    paths = {c["path"]: c for c in body["changes"]}
    assert paths["evidence_rules[supported].when.all[0].lab.value"]["to"] == 54


def test_evaluate_returns_a_full_manifest(client):
    body = client.post(
        "/evaluate",
        json={"definition_id": "te_symptomatic_hypoglycemia", "version": 1,
              "save": False},
    ).json()
    assert body["definition_version"] == 1
    assert body["results_hash"] and body["run_id"]
    assert sum(body["counts_by_verdict"].values()) == len(body["assignments"])
    assert all(a["reason"] for a in body["assignments"])


def test_evaluating_an_unknown_version_is_a_400(client):
    response = client.post(
        "/evaluate", json={"definition_id": "te_symptomatic_hypoglycemia", "version": 99}
    )
    assert response.status_code == 400


def test_retrieve_applies_the_assertion_filter(client):
    on = client.get("/retrieve", params={
        "concept": "HYPOGLYCEMIA", "assertion": ["present"], "top_k": 500}).json()
    off = client.get("/retrieve", params={
        "concept": "HYPOGLYCEMIA", "top_k": 500}).json()
    assert on["negation_false_positives"] == 0
    assert off["negation_false_positives"] > 0


def test_the_candidate_endpoint_never_mutates_the_frozen_file(client, pipeline):
    from pathlib import Path

    source = Path(pipeline.definition("te_symptomatic_hypoglycemia", 1).source_path)
    before = source.read_text(encoding="utf-8")
    body = client.post("/definitions/candidate", json={
        "definition_id": "te_symptomatic_hypoglycemia",
        "base_version": 1,
        "changes": {"window.max": 7, "evidence_rules.supported.lab.value": 54},
    }).json()
    assert source.read_text(encoding="utf-8") == before
    assert body["filename"].endswith(".yaml")
    assert "status: draft" in body["yaml"]
    assert "supersedes: te_symptomatic_hypoglycemia.v1" in body["yaml"]
    assert len(body["applied_changes"]) == 2


def test_a_candidate_that_would_not_validate_is_refused(client):
    response = client.post("/definitions/candidate", json={
        "definition_id": "te_symptomatic_hypoglycemia",
        "base_version": 1,
        "changes": {"window.max": -5},
    })
    assert response.status_code == 400


def test_an_unknown_candidate_path_is_refused(client):
    response = client.post("/definitions/candidate", json={
        "definition_id": "te_symptomatic_hypoglycemia",
        "base_version": 1,
        "changes": {"not.a.path": 1},
    })
    assert response.status_code == 400
    assert "no such definition path" in response.json()["detail"]


def test_agent_compile_returns_a_spec_and_does_not_execute(client):
    response = client.post("/agent/compile", json={
        "question": "symptomatic hypoglycemia within 14 days of escalation"})
    body = response.json()
    assert response.status_code == 200
    assert body["executed"] is False
    assert body["approval_required"] is True
    assert body["spec"]["definition_id"] == "te_symptomatic_hypoglycemia"


def test_an_ambiguous_question_is_a_conflict_with_a_clarification(client):
    response = client.post("/agent/compile",
                           json={"question": "serious severe hypoglycemia cases"})
    assert response.status_code == 409
    assert response.json()["clarification"]["ambiguity"]


def test_agent_run_is_blocked_without_approval(client):
    response = client.post("/agent/run", json={
        "question": "symptomatic hypoglycemia within 14 days of escalation",
        "approved": False})
    assert response.status_code == 428
    assert response.json()["executed"] is False
    assert "blocked until" in response.json()["detail"]


def test_agent_run_executes_once_approved(client):
    response = client.post("/agent/run", json={
        "question": "symptomatic hypoglycemia within 14 days of escalation",
        "approved": True})
    body = response.json()
    assert response.status_code == 200 and body["executed"]
    assert body["summary"]["primary_case_count"] >= 0
    assert body["definition"]["hash"] and body["extractor_version"]
    assert body["limitations"]


def test_agent_run_refuses_an_ambiguous_question_even_when_approved(client):
    response = client.post("/agent/run", json={
        "question": "serious severe hypoglycemia cases", "approved": True})
    assert response.status_code == 409
    assert response.json()["executed"] is False


def test_runs_can_be_listed_fetched_and_replayed(client):
    created = client.post("/evaluate", json={
        "definition_id": "te_symptomatic_hypoglycemia", "version": 1, "save": True,
    }).json()
    run_id = created["run_id"]

    listed = {r["run_id"] for r in client.get("/runs").json()["runs"]}
    assert run_id in listed
    assert client.get(f"/runs/{run_id}").json()["run_id"] == run_id

    replayed = client.post(f"/runs/{run_id}/replay").json()
    assert replayed["reproduced"], replayed["differences"]
    assert replayed["replayed_results_hash"] == created["results_hash"]


def test_an_unknown_run_is_a_404(client):
    assert client.get("/runs/deadbeef").status_code == 404


def test_the_ui_is_served(client):
    assert client.get("/").status_code == 200
    assert client.get("/ui/app.js").status_code == 200
