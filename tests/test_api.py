"""The HTTP surface. Thin by design: it must not disagree with the CLI."""

from __future__ import annotations

def test_summary_reports_every_version(client):
    body = client.get("/api/summary").json()
    for key in ("normalizer_version", "extractor_version", "snapshot_id",
                "dictionary_target"):
        assert body[key]
    assert len(body["profiles"]) == 7
    assert set(body["reconciliation"]) == {
        "unchanged", "remapped_mechanically", "flagged_for_review"
    }


def test_build_reports_no_provenance_violations(client):
    body = client.post("/build", json={"refresh": True}).json()
    assert body["provenance_violations"] == []
    assert body["extraction"]["requests"] > 0
    assert body["mentions"] > 0


def test_a_record_carries_both_fields_and_its_evidence(client):
    listing = client.get("/api/records?assertion=present&limit=1").json()
    record_id = listing["records"][0]["record_id"]
    body = client.get(f"/api/records/{record_id}").json()
    modifier = body["attributes"]["mucosal_involvement"]
    assert modifier["assertion"] == "present"
    assert modifier["availability"] == "observed"
    assert modifier["evidence"]
    assert modifier["route"]
    assert body["coded_event"]["code"]


def test_an_unknown_record_is_404(client):
    assert client.get("/api/records/nope").status_code == 404


def test_records_can_be_filtered_by_reconciliation(client):
    body = client.get("/api/records?reconciliation=flagged_for_review").json()
    assert body["count"] > 0
    for row in body["records"]:
        assert row["reconciliation"] == "flagged_for_review"
        assert row["reconciled_to"] is None


def test_episodes_are_labelled_as_a_derived_view(client):
    body = client.get("/api/episodes?limit=3").json()
    assert "no phenotype is evaluated at this grain" in body["note"]
    assert body["n_episodes"] > 0


def test_supportability_names_the_study_that_cannot_answer(client):
    body = client.get("/api/supportability").json()
    statuses = {row["status"] for row in body["studies"]}
    assert "cannot_ascertain" in statuses
    assert "no patient record was read" in body["note"]


def test_definitions_expose_their_accept_methods(client):
    body = client.get("/definitions").json()
    keys = {d["key"]: d for d in body["definitions"]}
    assert keys["cutaneous_mucosal.v1"]["accept_methods"] == ["direct"]
    assert keys["cutaneous_mucosal.v2"]["accept_methods"] == ["direct", "extracted"]


def test_the_definition_yaml_is_served_verbatim(client):
    text = client.get("/definitions/cutaneous_mucosal/2/yaml").text
    assert "concept_set" in text
    assert "accept_methods" in text


def test_an_unknown_definition_version_is_404(client):
    assert client.get("/definitions/cutaneous_mucosal/99/yaml").status_code == 404


def test_comparing_without_a_scope_is_400(client):
    response = client.get("/definitions/cutaneous_mucosal/compare?left=1&right=2")
    assert response.status_code == 400
    assert "scope" in response.json()["detail"].lower()


def test_comparing_with_a_scope_reports_what_moved(client):
    body = client.get(
        "/definitions/cutaneous_mucosal/compare"
        "?left=1&right=2&scope=cutaneous+adverse+events"
    ).json()
    assert body["gained"]
    assert body["discordant"]


def test_evaluate_returns_denominators_and_the_note(client):
    body = client.post("/evaluate", json={
        "definition_id": "cutaneous_mucosal", "version": 2, "save": False,
    }).json()
    assert len(body["denominators"]) > 1
    assert "ascertainable fraction" in body["denominator_note"]
    assert body["not_ascertainable"]


def test_evaluating_an_unknown_definition_is_400(client):
    response = client.post("/evaluate", json={"definition_id": "nope"})
    assert response.status_code == 400


def test_retrieve_is_cohort_eligible_and_discover_is_not(client):
    precise = client.get("/retrieve?assertion=present&top_k=5").json()
    assert precise["mode"] == "precise"
    assert precise["usable_as_cohort"] is True
    discovery = client.get("/discover?text=mucosal&top_k=5").json()
    assert discovery["usable_as_cohort"] is False
    assert discovery["all_candidates"] is True
    assert "not an event that occurred" in discovery["cohort_note"]


def test_retrieve_notes_when_both_filters_are_used(client):
    body = client.get(
        "/retrieve?assertion=absent&availability=observed&top_k=5"
    ).json()
    assert any("separately" in note for note in body["notes"])


def test_retrieving_an_unknown_concept_is_400(client):
    assert client.get("/retrieve?concept=NOPE").status_code == 400


def test_silver_carries_its_queue_and_caveats(client):
    body = client.get("/eval/silver").json()
    assert len(body["caveats"]) == 2
    assert body["adjudication_queue"]
    assert body["calibration"]["brier_score"] is not None


def test_transport_holds_out_whole_studies(client):
    body = client.get("/eval/transport").json()
    assert body["split"] == "whole_study"
    assert body["row_splits"] == "disallowed"


def test_transport_rejects_an_unknown_holdout(client):
    assert client.get("/eval/transport?holdout=P_nope").status_code == 400


def test_compile_only_does_not_execute(client):
    body = client.post("/agent/compile", json={
        "question": "cutaneous events with mucosal involvement",
    }).json()
    assert body["executed"] is False
    assert body["spec"]["definition_hash"]


def test_compile_returns_409_on_a_conflict(client):
    response = client.post("/agent/compile", json={
        "question": "cutaneous events with mucosal involvement within 14 days",
    })
    assert response.status_code == 409


def test_ask_returns_a_complete_trace(client):
    body = client.post("/agent/ask", json={
        "question": "cutaneous events with mucosal involvement", "save": True,
    }).json()
    assert body["executed"] is True
    assert body["traceable"] is True
    assert body["supportability"]


def test_a_run_can_be_listed_traced_and_replayed(client):
    asked = client.post("/agent/ask", json={
        "question": "cutaneous events with mucosal involvement", "save": True,
    }).json()
    manifest_id = asked["manifest_id"]

    runs = client.get("/runs").json()["runs"]
    assert any(r["manifest_id"] == manifest_id for r in runs)

    detail = client.get(f"/runs/{manifest_id}").json()
    assert detail["definition_id"] == "cutaneous_mucosal"

    traced = client.get(f"/trace/{manifest_id}").json()
    assert traced["complete"] is True
    assert "rendered" in traced

    replayed = client.post(f"/runs/{manifest_id}/replay").json()
    assert replayed["reproduced"] is True


def test_an_unknown_run_is_404(client):
    assert client.get("/runs/deadbeef").status_code == 404
    assert client.get("/trace/deadbeef").status_code == 404
    assert client.post("/runs/deadbeef/replay").status_code == 404


def test_the_ui_is_served(client):
    assert client.get("/").status_code == 200
    assert client.get("/ui/app.js").status_code == 200
