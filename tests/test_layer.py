"""End to end: governed runs, replay, traceability, and the demoted grain."""

from __future__ import annotations

import pytest

from aelayer.episode import group_records
from aelayer.knowledge import KnowledgeRegistry, ScopeRequired, diff_definitions
from aelayer.runs import (
    STANDING_LIMITATIONS, ManifestStore, ReplayError, ResultStore, execute,
    replay,
)


@pytest.fixture(scope="module")
def run(pipeline, definition_v2, tmp_path_factory):
    directory = tmp_path_factory.mktemp("runs")
    manifest, assignments = execute(
        pipeline, definition_v2, question="test run", actor="pytest",
        manifest_store=ManifestStore(directory),
        result_store=ResultStore(directory / "results"),
    )
    return manifest, assignments, directory


# -- manifests ----------------------------------------------------------------


def test_the_manifest_records_every_version_that_produced_the_number(run):
    manifest, _assignments, _dir = run
    assert manifest.definition_hash
    assert manifest.normalizer_version
    assert manifest.extractor_version
    assert manifest.data_snapshot_id
    assert manifest.dictionary_versions
    assert manifest.results_hash


def test_the_manifest_holds_a_pointer_not_a_copy(run):
    manifest, assignments, _dir = run
    body = manifest.model_dump()
    assert manifest.output_pointer
    assert "assignments" not in body
    assert len(str(body)) < 20000, (
        "the manifest is carrying result payload, which would create a second "
        "uncontrolled result store"
    )


def test_the_manifest_carries_per_study_denominators(run):
    manifest, _assignments, _dir = run
    assert len(manifest.denominators) > 1
    for row in manifest.denominators:
        assert "ascertainable_fraction" in row


def test_the_manifest_records_which_routes_supplied_the_evidence(run):
    manifest, _assignments, _dir = run
    assert manifest.attribute_methods
    assert "extracted" in manifest.attribute_methods


def test_the_standing_limitations_are_attached_to_every_run(run):
    manifest, _assignments, _dir = run
    assert manifest.limitations == STANDING_LIMITATIONS
    joined = " ".join(manifest.limitations)
    assert "synthetic" in joined
    assert "not_ascertainable" in joined
    assert "source-record grain" in joined
    assert "illustrative placeholders" in joined


def test_the_manifest_id_is_content_derived(pipeline, definition_v2):
    first, _a = execute(pipeline, definition_v2, save=False)
    second, _b = execute(pipeline, definition_v2, save=False)
    assert first.manifest_id == second.manifest_id
    assert first.results_hash == second.results_hash
    assert first.created_at != "" and "created_at" not in first.manifest_id


def test_a_different_definition_gives_a_different_id(pipeline, definition_v1,
                                                     definition_v2):
    a, _x = execute(pipeline, definition_v1, save=False)
    b, _y = execute(pipeline, definition_v2, save=False)
    assert a.manifest_id != b.manifest_id
    assert a.results_hash != b.results_hash


# -- replay -------------------------------------------------------------------


def test_a_recorded_run_replays_exactly(run, corpus_dir):
    manifest, _assignments, directory = run
    report, _replayed = replay(
        manifest.manifest_id, manifest_store=ManifestStore(directory),
        data_dir=corpus_dir,
    )
    assert report.reproduced, report.differences
    assert report.original_hash == report.replayed_hash


def test_an_unknown_run_is_refused(tmp_path):
    with pytest.raises(ReplayError) as exc:
        replay("deadbeef", manifest_store=ManifestStore(tmp_path))
    assert "Nothing has been executed yet" in str(exc.value)


def test_a_missing_result_file_is_a_real_gap(run, tmp_path):
    manifest, _assignments, directory = run
    with pytest.raises(ReplayError) as exc:
        ResultStore(tmp_path).read(manifest.manifest_id)
    assert "pointer, not a copy" in str(exc.value)


# -- the knowledge layer ------------------------------------------------------


def test_an_empty_registry_is_not_an_error(tmp_path, pipeline):
    status = KnowledgeRegistry.open(tmp_path, pipeline.definitions).status()
    assert status["manifests"] == 0
    assert status["capture_mode"] == "forward"
    assert "empty until something has been run" in status["note"]


def test_the_registry_accrues_from_executions(run, pipeline):
    _manifest, _assignments, directory = run
    status = KnowledgeRegistry.open(directory, pipeline.definitions).status()
    assert status["manifests"] >= 1
    assert status["definitions_used"]


def test_an_unscoped_sweep_is_refused(pipeline, definition_v1, definition_v2):
    with pytest.raises(ScopeRequired) as exc:
        diff_definitions(
            definition_v1, definition_v2, pipeline.snapshot_id, None,
            pipeline.assignments(definition_v1),
            pipeline.assignments(definition_v2),
        )
    assert "auditing colleagues' past choices" in str(exc.value)


def test_a_definition_diff_is_executed_not_textual(pipeline, definition_v1,
                                                   definition_v2):
    comparison = diff_definitions(
        definition_v1, definition_v2, pipeline.snapshot_id,
        "cutaneous adverse events",
        pipeline.assignments(definition_v1), pipeline.assignments(definition_v2),
    )
    assert comparison.gained
    assert comparison.discordant
    example = comparison.discordant[0]
    assert example.reason_a != example.reason_b
    assert example.verdict_a != example.verdict_b


# -- the demoted episode grain ------------------------------------------------


def test_episodes_are_a_view_and_carry_no_attributes(records):
    view = group_records(records)
    assert view.episodes
    for episode in view.episodes:
        assert episode.record_ids
        assert not hasattr(episode, "modifiers")
    assert "no phenotype is evaluated at this grain" in view.note


def test_no_verdict_is_assigned_at_the_episode_grain(assignments, records):
    """Every assignment is keyed by a source record, not an episode."""
    record_ids = {r.record_id for r in records}
    for assignment in assignments:
        assert assignment.record_id in record_ids


def test_records_with_unreadable_dates_are_never_merged_on_a_guess(records):
    view = group_records(records)
    for episode in view.episodes:
        if episode.size > 1:
            assert episode.rule != "single_record"


# -- the whole path -----------------------------------------------------------


def test_the_source_records_are_never_modified(pipeline, records):
    """Enrichment adds attributes above the row; it never edits the row."""
    rows = {str(r["AESPID"]): dict(r) for r in pipeline.store.rows("ae")}
    pipeline.records(refresh=True)
    after = {str(r["AESPID"]): dict(r) for r in pipeline.store.rows("ae")}
    assert rows == after


def test_the_summary_names_every_version(pipeline):
    summary = pipeline.summary()
    for key in ("normalizer_version", "extractor_version", "snapshot_id",
                "dictionary_versions", "dictionary_target"):
        assert summary[key]
    assert summary["extraction"]["requests"] > 0
