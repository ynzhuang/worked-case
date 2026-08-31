"""Provenance, run manifests and replay."""

from __future__ import annotations

import pytest

from aelayer.runs import ReplayError, RunStore, compute_run_id, execute, replay


def test_a_run_stamps_all_three_hashes(pipeline, definition_v1):
    manifest = execute(pipeline, definition_v1)
    assert manifest.extractor_version == pipeline.extractor_version
    assert manifest.definition_hash == definition_v1.definition_hash
    assert manifest.snapshot_id == pipeline.snapshot_id
    assert manifest.definition_version == 1
    assert manifest.definition_status == "frozen"


def test_every_assignment_carries_the_definition_that_produced_it(pipeline, definition_v1):
    manifest = execute(pipeline, definition_v1)
    for assignment in manifest.assignments:
        assert assignment.definition_hash == definition_v1.definition_hash
        assert assignment.definition_version == 1


def test_the_run_id_is_content_derived(pipeline, definition_v1):
    first = execute(pipeline, definition_v1)
    second = execute(pipeline, definition_v1)
    assert first.run_id == second.run_id
    assert first.results_hash == second.results_hash
    assert first.created_at is not None  # only the timestamp may differ


def test_a_different_definition_version_is_a_different_run(pipeline, definition_v1,
                                                           definition_v2):
    assert execute(pipeline, definition_v1).run_id != execute(
        pipeline, definition_v2
    ).run_id


def test_the_run_id_moves_when_any_input_moves():
    spec = {"definition_id": "d", "definition_version": 1, "studies": []}
    base = compute_run_id(spec, "ex-1", "def-1", "snap-1")
    assert compute_run_id(spec, "ex-2", "def-1", "snap-1") != base
    assert compute_run_id(spec, "ex-1", "def-2", "snap-1") != base
    assert compute_run_id(spec, "ex-1", "def-1", "snap-2") != base
    assert compute_run_id(dict(spec, studies=["S"]), "ex-1", "def-1", "snap-1") != base


def test_a_run_is_reproduced_byte_for_byte(pipeline, definition_v1, tmp_path, corpus_dir):
    store = RunStore(tmp_path / "runs")
    manifest = execute(pipeline, definition_v1)
    store.save(manifest)
    report, replayed = replay(manifest.run_id, run_store=store, data_dir=corpus_dir)
    assert report.reproduced, report.differences
    assert replayed.results_hash == manifest.results_hash
    assert [a.model_dump_json() for a in replayed.assignments] == [
        a.model_dump_json() for a in manifest.assignments
    ]


def test_a_superseded_definition_is_still_replayable(pipeline, definition_v1,
                                                     definition_v2, tmp_path, corpus_dir):
    """A later version must never rewrite the cohort a prior analysis rests on."""
    store = RunStore(tmp_path / "runs")
    old = execute(pipeline, definition_v1)
    store.save(old)
    store.save(execute(pipeline, definition_v2))     # the newer version also runs
    report, replayed = replay(old.run_id, run_store=store, data_dir=corpus_dir)
    assert report.reproduced
    assert replayed.definition_version == 1


def test_replay_says_which_input_moved(pipeline, definition_v1, tmp_path, corpus_dir):
    store = RunStore(tmp_path / "runs")
    manifest = execute(pipeline, definition_v1)
    tampered = manifest.model_copy(update={"snapshot_id": "not-the-real-snapshot"})
    store.save(tampered)
    report, _replayed = replay(tampered.run_id, run_store=store, data_dir=corpus_dir)
    assert not report.reproduced
    assert any("data snapshot differs" in d for d in report.differences)


def test_replay_of_an_unknown_run_is_an_explicit_error(tmp_path):
    with pytest.raises(ReplayError, match="no run"):
        RunStore(tmp_path / "runs").load("deadbeef")


def test_a_manifest_round_trips_through_disk(pipeline, definition_v1, tmp_path):
    store = RunStore(tmp_path / "runs")
    manifest = execute(pipeline, definition_v1)
    path = store.save(manifest)
    assert path.exists()
    # Compared as data, not as a JSON string: the saved file sorts its keys,
    # so byte equality of the serialisation is not the claim being made.
    assert store.load(manifest.run_id).model_dump(mode="json") == manifest.model_dump(
        mode="json"
    )


def test_runs_are_deterministic_by_default(pipeline, definition_v1):
    manifest = execute(pipeline, definition_v1)
    assert manifest.deterministic
    assert manifest.nondeterministic_paths == []


def test_an_llm_compiled_spec_is_marked_nondeterministic(pipeline, definition_v1):
    manifest = execute(pipeline, definition_v1, spec_extra={"backend": "llm"})
    assert not manifest.deterministic
    assert manifest.nondeterministic_paths


def test_limitations_are_stated_on_every_run(pipeline, definition_v1):
    manifest = execute(pipeline, definition_v1)
    joined = " ".join(manifest.limitations).lower()
    assert "synthetic" in joined
    assert "not a trained clinical nlp model" in joined
    assert "meddra" in joined


def test_counts_are_recorded_by_state_and_by_verdict(pipeline, definition_v1):
    manifest = execute(pipeline, definition_v1)
    assert sum(manifest.counts_by_verdict.values()) == len(manifest.assignments)
    assert sum(manifest.counts_by_state.values()) == len(manifest.assignments)
