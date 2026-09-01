"""The knowledge layer, and traceability in place of an approval gate."""

from __future__ import annotations

import pytest

from aelayer.agent import AgentSession, render_trace, trace_number
from aelayer.knowledge import KnowledgeRegistry, ScopeRequired, diff_definitions
from aelayer.runs import ManifestStore, ResultStore, execute


# -- diff_definitions -------------------------------------------------------


def test_comparison_is_executable_not_textual(pipeline, definition_v1, definition_v2):
    """The answer is which episodes each definition claims, with reasons."""
    comparison = diff_definitions(
        definition_v1, definition_v2, pipeline.snapshot_id,
        "treatment-emergent hypoglycemia",
        pipeline.evaluate(definition_v1), pipeline.evaluate(definition_v2),
    )
    assert comparison.shared or comparison.lost
    assert comparison.discordant
    for entry in comparison.discordant:
        assert entry.reason_a and entry.reason_b
        assert entry.verdict_a != entry.verdict_b


def test_an_unscoped_sweep_is_refused(pipeline, definition_v1, definition_v2):
    """The capability is for evidence reuse, not auditing past choices."""
    for scope in (None, "", "   "):
        with pytest.raises(ScopeRequired, match="explicit scope"):
            diff_definitions(
                definition_v1, definition_v2, pipeline.snapshot_id, scope,
                pipeline.evaluate(definition_v1), pipeline.evaluate(definition_v2),
            )


def test_a_stricter_definition_only_loses_cases(pipeline, definition_v1, definition_v2):
    comparison = diff_definitions(
        definition_v1, definition_v2, pipeline.snapshot_id, "hypoglycemia",
        pipeline.evaluate(definition_v1), pipeline.evaluate(definition_v2),
    )
    assert comparison.lost
    assert comparison.gained == []


# -- registry ---------------------------------------------------------------


def test_the_registry_says_plainly_that_it_starts_empty(tmp_path, pipeline):
    registry = KnowledgeRegistry.open(tmp_path / "runs", pipeline.definitions)
    status = registry.status()
    assert status["manifests"] == 0
    assert status["capture_mode"] == "forward"
    assert "empty until something has been run" in status["note"]


def test_the_registry_accrues_from_executions(tmp_path, pipeline, definition_v1):
    store = ManifestStore(tmp_path / "runs")
    registry = KnowledgeRegistry.open(tmp_path / "runs", pipeline.definitions)
    execute(
        pipeline, definition_v1, manifest_store=store,
        result_store=ResultStore(tmp_path / "runs" / "results"),
    )
    status = registry.status()
    assert status["manifests"] == 1
    assert definition_v1.key in status["definitions_used"]


def test_backfill_adds_exactly_what_it_is_given(tmp_path, pipeline, definition_v1):
    store = ManifestStore(tmp_path / "runs")
    manifest, _ = execute(
        pipeline, definition_v1, manifest_store=store,
        result_store=ResultStore(tmp_path / "runs" / "results"),
    )
    other = KnowledgeRegistry.open(tmp_path / "other", pipeline.definitions)
    assert other.status()["manifests"] == 0
    other.backfill(manifest)
    assert other.status()["manifests"] == 1


def test_the_registry_can_be_asked_what_ran_against_what(
    tmp_path, pipeline, definition_v1, definition_v2
):
    store = ManifestStore(tmp_path / "runs")
    results = ResultStore(tmp_path / "runs" / "results")
    first, _ = execute(pipeline, definition_v1, manifest_store=store,
                       result_store=results, actor="cli")
    second, _ = execute(pipeline, definition_v2, manifest_store=store,
                        result_store=results, actor="agent")
    registry = KnowledgeRegistry.open(tmp_path / "runs", pipeline.definitions)

    assert len(registry.find()) == 2
    assert [m.manifest_id for m in registry.find(actor="agent")] == [second.manifest_id]
    assert len(registry.find(definition_id=definition_v1.id)) == 2
    assert registry.find(snapshot_id="nothing-like-this") == []
    assert len(registry.find(snapshot_id=first.data_snapshot_id)) == 2


def test_the_registry_knows_every_version_of_a_definition(tmp_path, pipeline):
    registry = KnowledgeRegistry.open(tmp_path / "runs", pipeline.definitions)
    versions = registry.definition_versions("te_symptomatic_hypoglycemia")
    assert [d.version for d in versions] == [1, 2]
    assert registry.definition_versions("not_a_definition") == []


def test_a_registry_with_no_definition_catalogue_still_reports_executions(tmp_path):
    registry = KnowledgeRegistry.open(tmp_path / "runs")
    assert registry.status()["definitions"] == []
    assert registry.definition_versions("anything") == []


# -- manifest ---------------------------------------------------------------


def test_the_manifest_points_at_results_it_does_not_copy(
    tmp_path, pipeline, definition_v1
):
    results = ResultStore(tmp_path / "results")
    manifest, assignments = execute(
        pipeline, definition_v1, manifest_store=ManifestStore(tmp_path / "runs"),
        result_store=results,
    )
    payload = manifest.model_dump()
    assert "assignments" not in payload
    assert manifest.output_pointer
    assert len(results.read(manifest.manifest_id)) == len(assignments)


def test_the_manifest_records_every_version_that_produced_it(
    pipeline, definition_v1
):
    manifest, _ = execute(pipeline, definition_v1, save=False)
    assert manifest.normalizer_version == pipeline.normalizer_version
    assert manifest.extractor_version == pipeline.extractor_version
    assert manifest.definition_hash == definition_v1.definition_hash
    assert manifest.data_snapshot_id == pipeline.snapshot_id
    assert manifest.terminology_versions
    assert manifest.parameters["backend"]


# -- trace ------------------------------------------------------------------


def test_a_reported_number_traces_to_source_text(pipeline, definition_v1):
    """Definition of done: the full chain, or a failure."""
    manifest, assignments = execute(pipeline, definition_v1, save=False)
    chain = trace_number(
        number=manifest.counts_by_verdict.get("case", 0), label="case count",
        manifest=manifest, assignments=assignments,
        episodes=pipeline.episodes(), records=pipeline.records(),
    )
    assert chain.complete, chain.broken_at
    assert set(chain.levels()) == {
        "number", "analysis", "cohort", "definition", "episode", "record", "span"
    }


def test_the_chain_is_ordered_from_number_to_span(pipeline, definition_v1):
    manifest, assignments = execute(pipeline, definition_v1, save=False)
    chain = trace_number(
        number=1, label="case count", manifest=manifest, assignments=assignments,
        episodes=pipeline.episodes(), records=pipeline.records(),
    )
    levels = chain.levels()
    assert levels[:4] == ["number", "analysis", "cohort", "definition"]
    assert levels.index("episode") < levels.index("record") < levels.index("span")


def test_a_span_in_the_chain_carries_real_text(pipeline, definition_v1):
    manifest, assignments = execute(pipeline, definition_v1, save=False)
    chain = trace_number(
        number=1, label="case count", manifest=manifest, assignments=assignments,
        episodes=pipeline.episodes(), records=pipeline.records(),
    )
    spans = [l for l in chain.links if l.level == "span"]
    assert spans
    assert all(l.payload["text"] for l in spans)


def test_a_broken_chain_is_reported_as_broken(pipeline, definition_v1):
    """Not softened into a caveat: an untraceable number is a failure."""
    manifest, assignments = execute(pipeline, definition_v1, save=False)
    chain = trace_number(
        number=0, label="case count", manifest=manifest, assignments=assignments,
        episodes=[], records=[],
    )
    assert not chain.complete
    assert chain.broken_at == "episode"
    assert "INCOMPLETE" in render_trace(chain)


def test_the_agent_returns_a_traceable_package(pipeline):
    session = AgentSession(
        pipeline, "symptomatic hypoglycemia within 14 days of escalation"
    )
    session.compile()
    package, manifest = session.execute(save=False)
    assert package.trace is not None and package.trace.complete
    assert package.to_dict()["traceable"]
    assert package.summary["primary_case_count"] >= 0
    assert package.summary["review_set_count"] is not None
    assert package.limitations
    assert package.versions["normalizer_version"]


def test_the_agent_calls_only_registered_services(pipeline):
    from aelayer.agent import SERVICES, AgentServices

    session = AgentSession(pipeline, "symptomatic hypoglycemia within 14 days")
    session.compile()
    package, _ = session.execute(save=False)
    assert set(package.services_called) <= set(SERVICES)
    with pytest.raises(ValueError, match="not a registered service"):
        AgentServices(pipeline).call("rm_minus_rf")


def test_the_handoff_refuses_to_code_unadjudicated_subjects(pipeline, definition_v1):
    from aelayer.agent import AgentServices

    assignments = [a.model_dump(mode="json") for a in pipeline.evaluate(definition_v1)]
    handoff = AgentServices(pipeline).call("genetics_handoff", assignments=assignments)
    review = [r for r in handoff["rows"] if r["verdict"] == "review"]
    assert review
    assert all(r["status"] is None for r in review)
