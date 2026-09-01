"""The HTTP surface. Same pipeline as the CLI, so they cannot disagree."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from aelayer import api


@pytest.fixture(scope="module")
def client(pipeline, tmp_path_factory, monkeypatch_module):
    runs = tmp_path_factory.mktemp("api-runs")
    monkeypatch_module.setattr(api, "_pipeline_singleton", lambda: pipeline)
    monkeypatch_module.setattr(api, "_runs_dir", lambda: runs)
    with TestClient(api.app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch

    patch = MonkeyPatch()
    yield patch
    patch.undo()


def ok(response):
    assert response.status_code == 200, response.text
    return response.json()


# -- what it says about itself ---------------------------------------------


def test_the_summary_says_the_data_is_synthetic_and_what_the_extractor_is(client):
    body = ok(client.get("/api/summary"))
    assert "synthetic" in body["notice"]
    assert "not a trained clinical NLP model" in body["notice"]
    assert "illustrative placeholders" in body["notice"]


def test_the_summary_reports_each_study_collection_convention(client):
    body = ok(client.get("/api/summary"))
    assert len(body["study_semantics"]) == 6
    for study in body["study_semantics"].values():
        assert study["representation"]
        assert study["dictionary_version"]
    assert body["collection_states"]


def test_the_summary_names_the_versions_behind_every_number(client):
    body = ok(client.get("/api/summary"))
    for key in ("normalizer_version", "extractor_version", "snapshot_id",
                "extraction_backend"):
        assert body[key]


# -- records ----------------------------------------------------------------


def test_records_carry_their_collection_states(client):
    body = ok(client.get("/api/records", params={"limit": 5}))
    assert body["count"] > 5
    for record in body["records"]:
        assert record["collection_states"]


def test_records_can_be_filtered_to_a_collection_state(client):
    body = ok(client.get(
        "/api/records",
        params={"collection_state": "not_collected_by_protocol", "limit": 5},
    ))
    assert body["count"]
    for record in body["records"]:
        assert "not_collected_by_protocol" in record["collection_states"].values()


def test_one_record_exposes_every_span_behind_every_field(client):
    listing = ok(client.get("/api/records", params={"limit": 1}))
    record_id = listing["records"][0]["source_record_id"]
    body = ok(client.get(f"/api/records/{record_id}"))
    assert body["provenance_complete"]
    for field in body["fields"].values():
        if field["value"] is not None:
            assert field["spans"]


def test_an_unknown_record_is_a_404(client):
    assert client.get("/api/records/NOPE").status_code == 404


# -- episodes ---------------------------------------------------------------


def test_episodes_report_the_rule_that_linked_them(client):
    body = ok(client.get("/api/episodes", params={"limit": 10}))
    for episode in body["episodes"]:
        assert episode["linkage_rule"]
        assert 0.0 <= episode["linkage_confidence"] <= 1.0


def test_episodes_flagged_for_review_are_queryable_as_such(client):
    body = ok(client.get("/api/episodes", params={"review_only": True}))
    assert all(e["linkage_review_required"] for e in body["episodes"])
    assert all(e["linkage_note"] for e in body["episodes"])


def test_an_episode_shows_the_records_it_was_derived_from(client):
    listing = ok(client.get("/api/episodes", params={"limit": 1}))
    episode_id = listing["episodes"][0]["episode_id"]
    body = ok(client.get(f"/api/episodes/{episode_id}"))
    assert body["records"]
    assert len(body["records"]) == len(body["episode"]["source_record_ids"])
    assert body["field_states"]


# -- definitions ------------------------------------------------------------


def test_definitions_are_listed_with_their_content_hashes(client):
    body = ok(client.get("/definitions"))
    assert len(body["definitions"]) >= 2
    assert all(d["hash"] for d in body["definitions"])


def test_a_definition_is_served_as_the_yaml_on_disk(client):
    body = client.get("/definitions/te_symptomatic_hypoglycemia/1/yaml")
    assert body.status_code == 200
    assert "evidence_rules" in body.text


def test_comparing_two_versions_without_a_scope_is_refused(client):
    response = client.get(
        "/definitions/te_symptomatic_hypoglycemia/compare",
        params={"left": 1, "right": 2},
    )
    assert response.status_code == 400
    assert "requires an explicit scope" in response.json()["detail"]


def test_a_scoped_comparison_returns_the_episodes_that_moved(client):
    body = ok(client.get(
        "/definitions/te_symptomatic_hypoglycemia/compare",
        params={"left": 1, "right": 2,
                "scope": "post-hoc hypoglycemia incidence question"},
    ))
    assert body["discordant"]
    first = body["discordant"][0]
    assert first["reason_a"] and first["reason_b"]
    assert first["verdict_a"] != first["verdict_b"]


def test_a_candidate_is_rendered_and_nothing_on_disk_changes(client, pipeline):
    before = pipeline.definition("te_symptomatic_hypoglycemia", 1).definition_hash
    body = ok(client.post("/definitions/candidate", json={
        "definition_id": "te_symptomatic_hypoglycemia",
        "base_version": 1,
        "changes": {"window.max": 7},
    }))
    assert "window" in body["yaml"]
    assert body["applied_changes"] == ["window.max = 7"]
    assert "has not been modified" in body["note"]
    after = pipeline.definitions.get("te_symptomatic_hypoglycemia", 1)
    assert after.definition_hash == before


def test_a_candidate_naming_a_path_that_does_not_exist_is_refused(client):
    response = client.post("/definitions/candidate", json={
        "definition_id": "te_symptomatic_hypoglycemia",
        "base_version": 1,
        "changes": {"window.vibes": 7},
    })
    assert response.status_code == 400
    assert "no such definition path" in response.json()["detail"]


# -- evaluation -------------------------------------------------------------


def test_evaluating_returns_a_manifest_and_the_review_set_separately(client):
    body = ok(client.post("/evaluate", json={
        "definition_id": "te_symptomatic_hypoglycemia", "version": 1,
    }))
    manifest = body["manifest"]
    assert manifest["manifest_id"] and manifest["results_hash"]
    assert manifest["output_pointer"]
    assert len(body["review_set"]) == manifest["counts_by_verdict"].get("review", 0)


def test_evaluating_an_unknown_definition_is_a_400(client):
    response = client.post("/evaluate", json={"definition_id": "nope"})
    assert response.status_code == 400


# -- retrieval and discovery ------------------------------------------------


def test_the_precise_path_is_usable_as_a_cohort(client):
    body = ok(client.get("/retrieve", params={"concept": "HYPOGLYCEMIA"}))
    assert body["usable_as_cohort"]
    assert body["mode"] == "precise"


def test_the_discovery_path_says_it_is_not(client):
    body = ok(client.get("/discover", params={"concept": "HYPOGLYCEMIA"}))
    assert body["usable_as_cohort"] is False
    assert body["all_candidates"]
    assert "never directly" in body["cohort_note"]


def test_assertion_is_a_filter_on_the_discovery_path(client):
    filtered = ok(client.get("/discover", params={
        "concept": "HYPOGLYCEMIA", "assertion": "present", "top_k": 200,
    }))
    unfiltered = ok(client.get("/discover", params={
        "concept": "HYPOGLYCEMIA", "top_k": 200,
    }))
    assert filtered["negation_false_positives"] == 0
    assert unfiltered["negation_false_positives"] > 0


def test_absence_is_queryable_in_its_own_right(client):
    body = ok(client.get("/discover", params={
        "concept": "HYPOGLYCEMIA", "assertion": "absent", "top_k": 200,
    }))
    assert body["count"]
    assert all(m["assertion"] == "absent" for m in body["mentions"])


# -- the agent --------------------------------------------------------------


def test_an_underdetermined_question_returns_a_clarification_and_no_number(client):
    response = client.post("/agent/ask", json={
        "question": "how many subjects had severe hypoglycemia requiring "
                    "hospitalisation?",
        "save": False,
    })
    assert response.status_code == 409
    body = response.json()
    assert body["executed"] is False
    assert body["clarification"]["options"]
    assert "no number was produced" in body["detail"]


def test_compiling_never_executes(client):
    body = ok(client.post("/agent/compile", json={
        "question": "how many subjects had symptomatic hypoglycemia?",
    }))
    assert body["executed"] is False
    assert body["spec"]


def test_asking_executes_without_an_approval_step_and_returns_a_trace(client):
    """Traceability after the fact is the check, not a click before it."""
    body = ok(client.post("/agent/ask", json={
        "question": "how many subjects had symptomatic hypoglycemia?",
    }))
    assert body["executed"] is True
    assert body["traceable"] is True
    assert body["trace"]["complete"]
    assert body["summary"]["primary_case_count"] >= 0
    assert any("synthetic" in limitation for limitation in body["limitations"])


# -- runs, replay, trace ----------------------------------------------------


def test_a_run_is_listed_then_replays_hash_for_hash(client):
    manifest = ok(client.post("/evaluate", json={
        "definition_id": "te_symptomatic_hypoglycemia", "version": 1,
    }))["manifest"]
    listing = ok(client.get("/runs"))
    assert manifest["manifest_id"] in [r["manifest_id"] for r in listing["runs"]]

    replayed = ok(client.post(f"/runs/{manifest['manifest_id']}/replay"))
    assert replayed["reproduced"], replayed["differences"]
    assert replayed["original_results_hash"] == replayed["replayed_results_hash"]


def test_a_number_traces_back_to_a_span(client):
    manifest = ok(client.post("/evaluate", json={
        "definition_id": "te_symptomatic_hypoglycemia", "version": 1,
    }))["manifest"]
    body = ok(client.get(f"/trace/{manifest['manifest_id']}"))
    assert body["complete"]
    assert [link["level"] for link in body["links"]][0] == "number"
    assert body["links"][-1]["level"] == "span"
    assert "number" in body["rendered"]


def test_an_unknown_run_is_a_404(client):
    assert client.get("/runs/deadbeef").status_code == 404
    assert client.post("/runs/deadbeef/replay").status_code == 404


def test_the_knowledge_registry_reports_what_it_actually_holds(client):
    body = ok(client.get("/knowledge/status"))
    assert body["capture_mode"] == "forward"
    assert "backfill" in body["note"]


# -- the UI is served -------------------------------------------------------


def test_the_ui_is_served_from_the_same_process(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "<html" in response.text.lower() or "<!doctype" in response.text.lower()
