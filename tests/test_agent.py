"""The agent: a spec, an approval gate, and a clarification when it cannot tell."""

from __future__ import annotations

import pytest

from aelayer.agent import AgentSession, TOOLS, AgentTools, compile_question
from aelayer.agent.run import ApprovalRequired


def compile_for(pipeline, question):
    return compile_question(question, pipeline)


def test_a_clear_question_compiles_to_a_spec(pipeline):
    result = compile_for(pipeline, "symptomatic hypoglycemia within 14 days of escalation")
    assert result.spec is not None
    assert result.spec.definition_id == "te_symptomatic_hypoglycemia"
    assert result.spec.concept == "HYPOGLYCEMIA"
    assert result.spec.assertion == ["present"]


def test_severity_and_seriousness_together_produce_a_clarification(pipeline):
    """The most consequential ambiguity available: they are different fields."""
    result = compile_for(pipeline, "serious severe hypoglycemia cases")
    assert result.needs_clarification
    clarification = result.clarification
    assert "severity" in clarification.ambiguity.lower()
    assert "seriousness" in clarification.ambiguity.lower()
    assert "mild event can be serious" in clarification.effect
    assert len(clarification.options) >= 2


def test_severe_alone_is_still_underdetermined(pipeline):
    result = compile_for(pipeline, "how many subjects had severe hypoglycemia?")
    assert result.needs_clarification
    assert "third-party assistance" in " ".join(result.clarification.options)


def test_a_window_that_differs_from_the_definition_is_named_not_assumed(pipeline):
    result = compile_for(pipeline, "hypoglycemia within 7 days of dose escalation")
    assert result.needs_clarification
    assert "7-day window" in result.clarification.ambiguity
    assert "new definition version" in result.clarification.effect
    assert any("v3" in o or "v2" in o or "as written" in o
               for o in result.clarification.options)


def test_a_threshold_that_differs_is_named_not_assumed(pipeline):
    """63 mg/dL is neither v1's 70 nor v2's 54, so no version answers it."""
    result = compile_for(pipeline, "symptomatic hypoglycemia with glucose below 63")
    assert result.needs_clarification
    assert "threshold" in result.clarification.ambiguity
    assert "part of the definition, not of the query" in result.clarification.effect


def test_a_threshold_that_a_published_version_already_uses_compiles(pipeline):
    """v2 exists precisely to answer this; the agent should just run it."""
    result = compile_for(pipeline, "symptomatic hypoglycemia with glucose below 54")
    assert result.spec is not None
    assert result.spec.definition_version == 2


def test_the_latest_published_version_is_selected_by_default(pipeline):
    result = compile_for(pipeline, "symptomatic hypoglycemia within 14 days of escalation")
    latest = max(pipeline.definitions.versions("te_symptomatic_hypoglycemia"))
    assert result.spec.definition_version == latest


def test_an_unmatched_question_does_not_invent_a_cohort(pipeline):
    result = compile_for(pipeline, "how many patients had a stroke?")
    assert result.needs_clarification
    assert "would be an invention" in result.clarification.effect


def test_an_unknown_study_is_refused_rather_than_silently_dropped(pipeline):
    result = compile_for(pipeline, "symptomatic hypoglycemia in STUDY-99")
    assert result.needs_clarification
    assert "STUDY-99" in result.clarification.ambiguity


def test_named_studies_narrow_the_spec(pipeline):
    study = pipeline.store.studies()[0]
    result = compile_for(
        pipeline, f"symptomatic hypoglycemia within 14 days of escalation in {study}"
    )
    assert result.spec.studies == [study]


def test_asking_about_review_widens_the_evidence_states(pipeline):
    plain = compile_for(pipeline, "symptomatic hypoglycemia within 14 days of escalation")
    with_review = compile_for(
        pipeline,
        "symptomatic hypoglycemia within 14 days of escalation including review cases",
    )
    assert "possible" not in plain.spec.evidence_state
    assert "possible" in with_review.spec.evidence_state


def test_execution_is_blocked_until_the_spec_is_approved(pipeline):
    session = AgentSession(pipeline, "symptomatic hypoglycemia within 14 days of escalation")
    session.compile()
    with pytest.raises(ApprovalRequired, match="blocked until"):
        session.execute(save=False)
    session.approve()
    package, _manifest = session.execute(save=False)
    assert package.summary["primary_case_count"] >= 0


def test_a_clarification_cannot_be_approved(pipeline):
    session = AgentSession(pipeline, "serious severe hypoglycemia cases")
    session.compile()
    with pytest.raises(ApprovalRequired, match="clarification"):
        session.approve()


def test_recompiling_revokes_a_previous_approval(pipeline):
    session = AgentSession(pipeline, "symptomatic hypoglycemia within 14 days of escalation")
    session.compile()
    session.approve()
    session.compile()
    assert not session.approved
    with pytest.raises(ApprovalRequired):
        session.execute(save=False)


def test_the_evidence_package_carries_everything_needed_to_read_it(pipeline, tmp_path):
    from aelayer.runs import RunStore

    session = AgentSession(pipeline, "symptomatic hypoglycemia within 14 days of escalation")
    session.compile()
    session.approve()
    package, manifest = session.execute(run_store=RunStore(tmp_path / "runs"))

    assert package.summary["counts_by_state"]
    assert package.summary["per_study"]
    assert package.summary["review_set_count"] is not None
    assert package.contributing_spans
    assert package.definition["version"] == manifest.definition_version
    assert package.definition["hash"] == manifest.definition_hash
    assert package.extractor_version == pipeline.extractor_version
    assert package.snapshot_id == pipeline.snapshot_id
    assert package.run_id and package.results_hash
    assert package.limitations


def test_the_review_set_is_reported_separately_from_cases(pipeline):
    session = AgentSession(pipeline, "symptomatic hypoglycemia within 14 days of escalation")
    session.compile()
    session.approve()
    package, _ = session.execute(save=False)
    counts = package.summary["counts_by_verdict"]
    assert package.summary["primary_case_count"] == counts.get("case", 0)
    assert package.summary["review_set_count"] == counts.get("review", 0)


def test_the_tool_surface_is_closed(pipeline):
    tools = AgentTools(pipeline)
    assert set(TOOLS) == {"cohort", "retrieve", "evaluate", "summarise"}
    with pytest.raises(ValueError, match="not a callable tool"):
        tools.call("rm_minus_rf")


def test_the_cohort_tool_reports_the_denominator(pipeline):
    tools = AgentTools(pipeline)
    cohort = tools.call("cohort")
    assert cohort["subjects"] == len(pipeline.store.subjects())
    assert cohort["snapshot_id"] == pipeline.snapshot_id


def test_the_llm_backend_declines_cleanly_without_a_key(pipeline, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = compile_question(
        "symptomatic hypoglycemia", pipeline, backend="llm"
    )
    assert result.needs_clarification
    assert "no API key" in result.clarification.ambiguity
    assert "deterministic backend" in " ".join(result.clarification.options)


def test_the_deterministic_backend_is_stable(pipeline):
    question = "symptomatic hypoglycemia within 14 days of escalation"
    first = compile_for(pipeline, question).spec.model_dump_json()
    second = compile_for(pipeline, question).spec.model_dump_json()
    assert first == second
