"""Episodes, trajectories, retrieval, the knowledge layer, and the agent."""

from __future__ import annotations

import datetime as _dt

import pytest

from aelayer.agent import AgentServices, AgentSession, ToolError
from aelayer.agent.tools import REGISTRY
from aelayer.episode import EpisodeReconciler, ReconciliationConfig
from aelayer.knowledge import KnowledgeRegistry, ScopeRequired, diff_definitions
from aelayer.models import Attribute
from aelayer.retrieval import CandidateInCohort, discover, retrieve
from aelayer.runs import ManifestStore, ResultStore, execute
from aelayer.trajectory import time_since_exposure


# -- episodes ---------------------------------------------------------------


def test_source_records_survive_derivation_unmodified(pipeline, episodes, records):
    """Deleting every episode loses nothing; deleting a record loses the source."""
    before = [r.model_dump_json() for r in records]
    pipeline.reconcile(records)
    assert [r.model_dump_json() for r in records] == before


def test_no_record_is_lost_or_duplicated(episodes, records):
    assigned = [r for e in episodes for r in e.source_record_ids]
    assert sorted(assigned) == sorted(r.source_record_id for r in records)


def test_a_declared_continuation_outranks_everything(episodes, records):
    chains = [e for e in episodes if e.linkage_rule == "explicit_continuation"]
    assert chains
    for episode in chains:
        assert len(episode.source_record_ids) > 1


def test_an_attribute_promoted_to_the_episode_keeps_its_route(episodes):
    populated = [e for e in episodes if e.location.populated]
    assert populated
    for episode in populated:
        assert episode.location.method in ("direct", "normalized", "extracted")
        assert episode.location.source_variable


def test_the_more_authoritative_route_wins_a_disagreement(catalog, profiles, records):
    """A value the CRF settled outranks a reading of the study's own prose."""
    from aelayer.models import Span

    base = next(r for r in records if r.profile == "P6_both").model_copy(deep=True)
    other = base.model_copy(deep=True)
    other.source_record_id = base.source_record_id + "-B"
    other.continuation_of = base.source_record_id
    base.location = Attribute[str].direct(
        "CHEST", "AELOC",
        [Span(doc_id="AE:x", field="location", extracted_value="CHEST")],
    )
    other.location = Attribute[str].extracted(
        "BACK", "AETERM",
        [Span(doc_id="AE:x:AETERM", start=0, end=4, field="location",
              extracted_value="BACK")],
    )
    episodes = EpisodeReconciler(catalog, profiles, ReconciliationConfig()).reconcile(
        [base, other]
    )
    assert len(episodes) == 1
    assert episodes[0].location.value == "CHEST"
    assert "disagree" in episodes[0].location.note


def test_an_unresolvable_onset_is_not_placed_by_guesswork(catalog, profiles, records):
    pair = [r.model_copy(deep=True) for r in records[:2]]
    for record in pair:
        record.onset = Attribute[_dt.date].unavailable("unknown")
        record.continuation_of = None
    pair[1].subject_id = pair[0].subject_id
    pair[1].study_id = pair[0].study_id
    pair[1].standardized_concept = pair[0].standardized_concept
    episodes = EpisodeReconciler(catalog, profiles, ReconciliationConfig()).reconcile(pair)
    assert len(episodes) == 2
    assert any(e.linkage_review_required or e.linkage_confidence < 1.0
               for e in episodes) or all(
        len(e.source_record_ids) == 1 for e in episodes)


def test_every_episode_carries_an_offset_or_says_why_not(episodes):
    for episode in episodes:
        if episode.onset_offset_days.populated:
            assert episode.anchor_event == "first_exposure"
            assert episode.anchor_date is not None
        else:
            assert episode.onset_offset_days.note


# -- trajectories -----------------------------------------------------------


def test_a_trajectory_orders_exposures_and_episodes_together(pipeline):
    trajectories = pipeline.trajectories()
    assert trajectories
    trajectory = next(t for t in trajectories.values() if t.episodes())
    dates = [e.date for e in trajectory.events]
    assert dates == sorted(dates)
    assert trajectory.exposures()
    assert trajectory.anchor_event == "first_exposure"


def test_time_since_exposure_keeps_unresolved_offsets_in_their_own_bucket(pipeline):
    distribution = time_since_exposure(list(pipeline.trajectories().values()))
    assert "unresolved" in distribution
    assert sum(distribution.values()) > 0


# -- retrieval --------------------------------------------------------------


def test_the_precise_path_is_usable_as_a_cohort(index, catalog):
    result = retrieve(index, catalog, concept="RASH", verdict=["case"], top_k=500)
    assert result.episodes
    assert result.as_cohort() == result.episodes


def test_a_region_expands_to_its_declared_members(index, catalog):
    result = retrieve(index, catalog, concept="RASH", region="trunk", top_k=500)
    assert result.episodes
    assert {e.location for e in result.episodes} <= {"CHEST", "ABDOMEN", "BACK"}
    assert any("no hierarchy was walked" in note for note in result.notes)


def test_the_route_is_a_filter(index, catalog):
    result = retrieve(index, catalog, method=["extracted"], top_k=500)
    assert result.episodes
    assert {e.location_method for e in result.episodes} == {"extracted"}


def test_discovery_returns_candidates_that_cannot_enter_a_cohort(index, catalog):
    result = discover(index, catalog, attribute=["location"], top_k=20)
    assert result.mentions
    assert all(m.candidate for m in result.mentions)
    with pytest.raises(CandidateInCohort, match="not an event that occurred"):
        result.as_cohort()


def test_discovery_surfaces_modifiers_the_catalogue_does_not_cover(index, catalog):
    """The honest job of semantic search here: showing what the value space misses."""
    result = discover(index, catalog, unnormalized_only=True, top_k=100)
    assert result.mentions
    assert all(not m.normalized for m in result.mentions)
    assert any("cannot answer yet" in note for note in result.notes)


def test_dense_degrades_to_lexical_and_says_so(index, catalog):
    result = discover(index, catalog, mode="dense", top_k=5)
    assert result.mode == "lexical"
    assert any("degraded to lexical" in note for note in result.notes)


def test_the_index_knows_when_it_is_stale(index, pipeline):
    assert not index.is_stale(pipeline.extractor_version, pipeline.normalizer_version)
    assert index.is_stale("extract-0.0.0", pipeline.normalizer_version)


# -- knowledge --------------------------------------------------------------


def test_an_unscoped_sweep_is_refused(pipeline, definition_v1, definition_v2):
    with pytest.raises(ScopeRequired, match="requires an explicit scope"):
        diff_definitions(
            definition_v1, definition_v2, pipeline.snapshot_id, None,
            pipeline.evaluate(definition_v1), pipeline.evaluate(definition_v2),
        )


def test_a_scoped_comparison_is_executed_not_textual(
    pipeline, definition_v1, definition_v2
):
    comparison = diff_definitions(
        definition_v1, definition_v2, pipeline.snapshot_id, "truncal rash incidence",
        pipeline.evaluate(definition_v1), pipeline.evaluate(definition_v2),
    )
    assert comparison.discordant
    entry = comparison.discordant[0]
    assert entry.reason_a and entry.reason_b
    assert entry.verdict_a != entry.verdict_b


def test_narrowing_the_accepted_routes_loses_cases(
    pipeline, definition_v1, definition_v2
):
    comparison = diff_definitions(
        definition_v1, definition_v2, pipeline.snapshot_id, "route comparison",
        pipeline.evaluate(definition_v1), pipeline.evaluate(definition_v2),
    )
    assert comparison.lost
    assert not comparison.gained


def test_a_manifest_records_which_routes_supplied_the_evidence(
    pipeline, definition_v1, tmp_path
):
    manifest, _assignments = execute(
        pipeline, definition_v1,
        manifest_store=ManifestStore(tmp_path / "runs"),
        result_store=ResultStore(tmp_path / "runs" / "results"),
    )
    assert manifest.attribute_methods
    assert manifest.attribute_sources
    assert set(manifest.attribute_methods) <= {"direct", "normalized", "extracted"}


def test_the_registry_says_plainly_that_it_starts_empty(tmp_path, pipeline):
    status = KnowledgeRegistry.open(tmp_path / "runs", pipeline.definitions).status()
    assert status["manifests"] == 0
    assert status["capture_mode"] == "forward"
    assert "empty until something has been run" in status["note"]


# -- the agent --------------------------------------------------------------


def test_every_tool_declares_a_schema_and_a_permission():
    for spec in AgentServices.catalogue():
        assert spec["input_schema"] and spec["output_schema"]
        assert spec["permission"]
        assert spec["writes_source_records"] is False


def test_there_is_no_sql_surface():
    for name, spec in REGISTRY.items():
        fields = spec.input_model.model_fields
        assert not {"sql", "query", "statement"} & set(fields), name


def test_arguments_are_validated_before_a_tool_runs(pipeline):
    with pytest.raises(ToolError, match="do not validate"):
        AgentServices(pipeline).call(
            "cohort.run", definition_id="te_truncal_rash", sql="DROP TABLE ae"
        )


def test_a_tool_outside_the_grant_is_refused(pipeline):
    services = AgentServices(pipeline, permissions={"read_cohort"})
    with pytest.raises(ToolError, match="requires the 'export' permission"):
        services.call("omics.run", definition_id="te_truncal_rash")


def test_an_unregistered_tool_is_refused(pipeline):
    with pytest.raises(ToolError, match="not a registered tool"):
        AgentServices(pipeline).call("run_sql", q="select 1")


def test_the_export_leaves_unadjudicated_subjects_uncoded(pipeline, index):
    body = AgentServices(pipeline).call(
        "omics.run", definition_id="te_truncal_rash", version=1
    )
    unascertainable = [r for r in body["rows"] if r["verdict"] == "not_ascertainable"]
    assert unascertainable
    assert all(r["status"] is None for r in unascertainable)
    assert all(r["status"] in (0, 1) for r in body["rows"]
               if r["verdict"] in ("case", "not_case"))
    assert "neither cases nor controls" in body["note"]


def test_an_export_row_carries_the_route_its_verdict_depended_on(pipeline, index):
    body = AgentServices(pipeline).call(
        "omics.run", definition_id="te_truncal_rash", version=1
    )
    cases = [r for r in body["rows"] if r["verdict"] == "case"]
    assert cases
    assert any(r["evidence_route"] for r in cases)


def test_severity_and_seriousness_together_stop_the_run(pipeline):
    session = AgentSession(pipeline, "how many severe rash cases were hospitalised?")
    result = session.compile()
    assert result.needs_clarification
    assert "seriousness" in result.clarification.ambiguity.lower()
    assert result.spec is None


def test_a_window_the_definition_does_not_use_needs_a_new_version(pipeline):
    session = AgentSession(pipeline, "rash within 30 days of first exposure")
    result = session.compile()
    assert result.needs_clarification
    assert "new definition version" in result.clarification.effect


def test_a_question_naming_no_phenotype_is_refused(pipeline):
    result = AgentSession(pipeline, "how many patients felt off?").compile()
    assert result.needs_clarification
    assert "does not name a phenotype" in result.clarification.ambiguity


def test_the_agent_computes_nothing_itself(pipeline, index):
    session = AgentSession(pipeline, "how many rash cases after first exposure?")
    session.compile()
    package, _manifest = session.execute(save=False)
    assert set(package.tools_called) <= set(REGISTRY)
    definition = pipeline.definition(
        package.spec.definition_id, package.spec.definition_version
    )
    assignments = pipeline.evaluate(definition, package.spec.studies or None)
    assert package.cohort["counts_by_verdict"].get("case", 0) == sum(
        1 for a in assignments if a.verdict == "case"
    )


def test_the_number_traces_back_to_source_text(pipeline, index):
    session = AgentSession(pipeline, "how many rash cases after first exposure?")
    session.compile()
    package, _manifest = session.execute(save=False)
    assert package.trace.complete
    levels = [link.level for link in package.trace.links]
    assert levels[:4] == ["number", "analysis", "cohort", "definition"]
    assert levels[-1] == "span"
