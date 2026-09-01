"""Governed execution: a manifest points at a result, and a run replays."""

from __future__ import annotations

import json

import pytest

from aelayer.models import Manifest
from aelayer.runs import (
    ManifestStore,
    ReplayError,
    ResultStore,
    compute_manifest_id,
    execute,
    replay,
)


@pytest.fixture
def stores(tmp_path):
    return ManifestStore(tmp_path / "runs"), ResultStore(tmp_path / "runs" / "results")


@pytest.fixture
def run(pipeline, definition_v1, stores):
    manifests, results = stores
    manifest, assignments = execute(
        pipeline, definition_v1, manifest_store=manifests, result_store=results,
    )
    return manifest, assignments


# -- what a manifest is -----------------------------------------------------


def test_a_manifest_points_at_the_result_and_does_not_copy_it(run, stores):
    """A second copy of the result is a second thing that can drift."""
    manifest, _assignments = run
    _manifests, results = stores
    body = manifest.model_dump(mode="json")
    assert manifest.output_pointer
    assert manifest.output_pointer.endswith(f"{manifest.manifest_id}.results.json")
    assert results.path_for(manifest.manifest_id).exists()
    for value in body.values():
        assert not isinstance(value, list) or not any(
            isinstance(v, dict) and "episode_id" in v for v in value
        )


def test_a_manifest_records_every_version_that_produced_the_number(run):
    manifest, _ = run
    assert manifest.normalizer_version
    assert manifest.extractor_version
    assert manifest.data_snapshot_id
    assert manifest.definition_hash
    assert manifest.terminology_versions
    assert manifest.results_hash


def test_a_manifest_carries_its_standing_limitations(run):
    manifest, _ = run
    body = " ".join(manifest.limitations)
    assert "synthetic" in body
    assert "illustrative placeholders" in body


def test_counts_in_the_manifest_match_the_assignments(run):
    manifest, assignments = run
    assert sum(manifest.counts_by_verdict.values()) == len(assignments)
    assert manifest.counts_by_verdict["case"] == sum(
        1 for a in assignments if a.verdict == "case"
    )


# -- the id is content-derived ---------------------------------------------


def test_the_manifest_id_is_derived_from_content_not_from_the_clock(run, pipeline,
                                                                   definition_v1,
                                                                   stores):
    first, _ = run
    manifests, results = stores
    second, _ = execute(
        pipeline, definition_v1, manifest_store=manifests, result_store=results,
    )
    assert first.manifest_id == second.manifest_id
    assert first.results_hash == second.results_hash


def test_a_different_definition_yields_a_different_manifest_id(
    pipeline, definition_v1, definition_v2, stores
):
    manifests, results = stores
    a, _ = execute(pipeline, definition_v1, manifest_store=manifests,
                   result_store=results)
    b, _ = execute(pipeline, definition_v2, manifest_store=manifests,
                   result_store=results)
    assert a.manifest_id != b.manifest_id


def test_a_narrower_study_scope_yields_a_different_manifest_id(
    pipeline, definition_v1, stores
):
    manifests, results = stores
    everything, _ = execute(pipeline, definition_v1, manifest_store=manifests,
                            result_store=results)
    one, _ = execute(pipeline, definition_v1, studies=["STUDY-01"],
                     manifest_store=manifests, result_store=results)
    assert everything.manifest_id != one.manifest_id


def test_the_manifest_id_ignores_fields_that_do_not_affect_the_answer():
    versions = {"normalizer_version": "n1", "extractor_version": "e1",
                "extraction_backend": "rules", "model_version": None}
    base = compute_manifest_id({"a": 1}, "dh", "snap", versions)
    assert base == compute_manifest_id(
        {"a": 1}, "dh", "snap", {**versions, "terminology_versions": {"x": "1"}}
    )
    assert base != compute_manifest_id({"a": 2}, "dh", "snap", versions)
    assert base != compute_manifest_id({"a": 1}, "dh", "other", versions)


# -- saving and not saving --------------------------------------------------


def test_a_run_that_is_not_saved_writes_nothing(pipeline, definition_v1, stores):
    manifests, results = stores
    manifest, _ = execute(pipeline, definition_v1, manifest_store=manifests,
                          result_store=results, save=False)
    assert manifest.output_pointer == ""
    assert not manifests.path_for(manifest.manifest_id).exists()
    assert not results.path_for(manifest.manifest_id).exists()


# -- replay -----------------------------------------------------------------


def test_a_recorded_run_replays_exactly(run, stores, corpus_dir):
    manifest, _ = run
    manifests, _results = stores
    report, _replayed = replay(
        manifest.manifest_id, manifest_store=manifests, data_dir=corpus_dir
    )
    assert report.reproduced, report.differences
    assert report.original_hash == report.replayed_hash
    assert "reproduced exactly" in report.summary()


def test_a_run_recorded_against_a_superseded_definition_still_replays(
    pipeline, definition_v1, definition_v2, stores, corpus_dir
):
    """v1 is superseded by v2 and remains a file on disk, so it still runs."""
    assert definition_v2.supersedes
    manifests, results = stores
    manifest, _ = execute(pipeline, definition_v1, manifest_store=manifests,
                          result_store=results)
    report, _ = replay(manifest.manifest_id, manifest_store=manifests,
                       data_dir=corpus_dir)
    assert report.reproduced, report.differences


def test_replaying_an_unknown_run_says_what_it_knows(stores):
    manifests, _ = stores
    with pytest.raises(ReplayError, match="Nothing has been executed yet"):
        replay("deadbeef", manifest_store=manifests)


def test_a_changed_definition_is_reported_as_a_named_difference(
    run, stores, corpus_dir, tmp_path, definition_v1
):
    """Replay says which input moved, not merely that the run failed."""
    import yaml

    manifest, _ = run
    manifests, _results = stores
    altered = tmp_path / "phenotypes"
    altered.mkdir()
    body = definition_v1.model_dump(
        mode="json", exclude={"definition_hash", "source_path"}
    )
    body["window"]["max"] = 30
    (altered / "te_symptomatic_hypoglycemia.v1.yaml").write_text(
        yaml.safe_dump(body), encoding="utf-8"
    )
    report, _ = replay(manifest.manifest_id, manifest_store=manifests,
                       data_dir=corpus_dir, phenotype_dir=altered)
    assert not report.reproduced
    assert any("has changed on disk" in d for d in report.differences)


def test_a_missing_result_file_is_a_real_gap_and_says_so(run, stores):
    manifest, _ = run
    _manifests, results = stores
    results.path_for(manifest.manifest_id).unlink()
    with pytest.raises(ReplayError, match="pointer, not a copy"):
        results.read(manifest.manifest_id)


# -- the store tolerates a messy directory ---------------------------------


def test_listing_skips_files_that_are_not_manifests(run, stores, tmp_path):
    manifests, _ = stores
    (manifests.directory / "notes.json").write_text("{}", encoding="utf-8")
    (manifests.directory / "broken.json").write_text("{not json", encoding="utf-8")
    listed = manifests.list()
    assert [m.manifest_id for m in listed] == [run[0].manifest_id]


def test_listing_an_empty_registry_is_not_an_error(tmp_path):
    assert ManifestStore(tmp_path / "nothing").list() == []


def test_a_saved_manifest_round_trips_through_json(run, stores):
    manifest, _ = run
    manifests, _ = stores
    raw = json.loads(manifests.path_for(manifest.manifest_id).read_text())
    assert Manifest.model_validate(raw) == manifest
