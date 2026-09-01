"""The agent: it compiles a specification, it does not compute an answer.

The boundary is the point of this module. The agent may choose *which* rule to
run; the rule itself, and every number that comes out of it, is produced by code
that was written before the question was asked.
"""

from __future__ import annotations

import pytest

from aelayer.agent import AgentSession
from aelayer.agent.compile import compile_question
from aelayer.agent.run import ClarificationRequired
from aelayer.agent.tools import SERVICES, AgentServices


def compile_it(pipeline, question, **kwargs):
    return compile_question(question, pipeline, backend="deterministic", **kwargs)


# -- compiling --------------------------------------------------------------


def test_a_clear_question_compiles_to_a_specification(pipeline):
    result = compile_it(
        pipeline, "How many subjects had treatment-emergent symptomatic "
                  "hypoglycemia after dose escalation?"
    )
    assert not result.needs_clarification
    spec = result.spec
    assert spec.definition_id == "te_symptomatic_hypoglycemia"
    assert spec.definition_version
    assert spec.retrieval_mode == "precise"
    assert spec.evidence_state


def test_the_specification_names_the_definition_version_that_will_run(pipeline):
    result = compile_it(pipeline, "count symptomatic hypoglycemia cases")
    definition = pipeline.definition(
        result.spec.definition_id, result.spec.definition_version
    )
    assert definition.status != "draft"


def test_the_compiled_spec_says_it_applied_no_defaults_of_its_own(pipeline):
    result = compile_it(pipeline, "count symptomatic hypoglycemia cases")
    assert any("no default of its own" in n for n in result.spec.notes)


def test_a_question_naming_no_known_phenotype_is_refused(pipeline):
    result = compile_it(pipeline, "how many subjects felt a bit off")
    assert result.needs_clarification
    assert "does not name a phenotype" in result.clarification.ambiguity
    assert result.clarification.options


def test_a_question_naming_an_absent_study_is_refused(pipeline):
    result = compile_it(
        pipeline, "hypoglycemia cases in STUDY-99 and STUDY-01"
    )
    assert result.needs_clarification
    assert "STUDY-99" in result.clarification.ambiguity
    assert "silently omit" in result.clarification.effect


def test_a_named_study_narrows_the_specification(pipeline):
    result = compile_it(pipeline, "hypoglycemia cases in STUDY-01")
    assert result.spec.studies == ["STUDY-01"]


def test_asking_about_review_widens_the_states_rather_than_the_verdict(pipeline):
    plain = compile_it(pipeline, "count symptomatic hypoglycemia cases").spec
    with_review = compile_it(
        pipeline, "count symptomatic hypoglycemia cases including borderline "
                  "ones for adjudication"
    ).spec
    assert set(plain.evidence_state) < set(with_review.evidence_state)


# -- what it refuses to settle silently ------------------------------------


def test_severity_and_seriousness_together_stop_the_run(pipeline):
    """Different fields; a single count would answer neither question."""
    result = compile_it(
        pipeline, "how many subjects had severe hypoglycemia requiring "
                  "hospitalisation?"
    )
    assert result.needs_clarification
    assert "seriousness" in result.clarification.ambiguity.lower()
    assert len(result.clarification.options) >= 2
    assert result.spec is None


def test_severe_alone_is_still_underdetermined(pipeline):
    result = compile_it(pipeline, "how many severe hypoglycemia cases were there?")
    assert result.needs_clarification
    assert "term of art" in result.clarification.ambiguity


def test_a_window_the_definition_does_not_use_is_a_new_version_not_a_parameter(
    pipeline
):
    result = compile_it(
        pipeline, "symptomatic hypoglycemia within 30 days of dose escalation"
    )
    assert result.needs_clarification
    assert "new definition version" in result.clarification.effect
    assert any("Create v" in option for option in result.clarification.options)


def test_a_threshold_the_definition_does_not_use_is_also_refused(pipeline):
    result = compile_it(
        pipeline, "symptomatic hypoglycemia with glucose below 70 mg/dL"
    )
    assert result.needs_clarification
    assert "threshold" in result.clarification.ambiguity.lower()


def test_the_definitions_window_survives_when_the_question_agrees_with_it(pipeline):
    result = compile_it(
        pipeline, "symptomatic hypoglycemia within 14 days of dose escalation"
    )
    assert not result.needs_clarification
    assert tuple(result.spec.window) == (0, 14)


def test_a_clarification_is_returned_rather_than_a_guess(pipeline):
    session = AgentSession(pipeline, "how many subjects felt a bit off")
    session.compile()
    with pytest.raises(ClarificationRequired, match="nothing was executed"):
        session.execute(save=False)


# -- executing --------------------------------------------------------------


@pytest.fixture(scope="module")
def package(pipeline):
    session = AgentSession(
        pipeline, "How many subjects had treatment-emergent symptomatic "
                  "hypoglycemia after dose escalation?"
    )
    session.compile()
    result, _manifest = session.execute(save=False)
    return result


def test_the_package_carries_the_versions_that_produced_the_number(package):
    for key in ("normalizer_version", "extractor_version", "snapshot_id"):
        assert package.versions[key]
    assert package.definition["hash"]
    assert package.manifest_id and package.results_hash


def test_the_package_reports_the_review_set_separately(package):
    assert "review_set_count" in package.summary
    assert package.summary["primary_case_count"] >= 0
    assert (
        package.summary["counts_by_verdict"].get("review", 0)
        == package.summary["review_set_count"]
    )


def test_the_package_states_its_limitations(package):
    body = " ".join(package.limitations)
    assert "synthetic" in body
    assert "illustrative placeholders" in body


def test_the_number_traces_back_to_source_text(package):
    assert package.trace is not None
    assert package.trace.complete
    assert [link.level for link in package.trace.links][:3] == [
        "number", "analysis", "cohort"
    ]


def test_the_agent_computes_nothing_itself(package, pipeline):
    """Every number in the package comes from a registered service."""
    assert set(package.services_called) <= set(SERVICES)
    definition = pipeline.definition(
        package.spec.definition_id, package.spec.definition_version
    )
    assignments = pipeline.evaluate(definition, package.spec.studies or None)
    assert package.summary["primary_case_count"] == sum(
        1 for a in assignments if a.verdict == "case"
    )


def test_an_unregistered_service_cannot_be_called(pipeline):
    with pytest.raises(ValueError, match="not a registered service"):
        AgentServices(pipeline).call("delete_everything")


# -- statistics stay descriptive -------------------------------------------


def test_the_statistics_service_offers_no_inferential_test(package):
    assert "incidence_proportion" in package.statistics
    assert "caveat" in package.statistics
    assert not any(
        key in package.statistics for key in ("p_value", "hazard_ratio", "ci")
    )
