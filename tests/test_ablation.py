"""The value ablation — the other thing this build exists for.

The experiment that can falsify the proposal. What is tested is that it
produces a *decision*, that the decision is about **correctly** ascertained
cases, and that it is capable of saying no.
"""

from __future__ import annotations

import pytest

from aelayer.ablation import (
    MATERIALITY,
    STAGES,
    AblationReport,
    Increment,
    StageResult,
    format_ablation,
    run_ablation,
)


@pytest.fixture(scope="module")
def report(pipeline, definition_v2):
    return run_ablation(
        definition_v2, pipeline.store, pipeline.configs, "rules"
    )


# -- shape --------------------------------------------------------------------


def test_three_cumulative_stages(report):
    assert len(report.stages) == 3
    assert [s.stage_id for s in report.stages] == [
        "structured", "reported_term", "comments"
    ]
    # Cumulative: each stage reads everything the one below it read.
    for lower, upper in zip(report.stages, report.stages[1:]):
        assert set(lower.readable_sources) < set(upper.readable_sources)


def test_the_first_stage_runs_no_model_at_all(report):
    assert report.stages[0].readable_sources == ()


def test_each_stage_ascertains_at_least_as_much_as_the_one_below(report):
    fractions = [s.metrics()["ascertainable_fraction"] for s in report.stages]
    assert fractions == sorted(fractions)


# -- the decision -------------------------------------------------------------


def test_the_output_states_a_decision_not_just_numbers(report):
    body = report.to_dict()
    assert body["decision"]
    assert body["decision"].startswith(("ADOPT", "DO NOT ADOPT"))
    for increment in body["increments"]:
        assert increment["decision"].startswith(("ADOPT", "DO NOT ADOPT"))
        assert increment["reasons"]


def test_the_decision_is_about_correctly_ascertained_cases(report):
    for increment in report.increments():
        assert increment.added_correct <= increment.added_cases
        assert increment.added_correct + increment.added_incorrect == \
            increment.added_cases
        assert "correctly ascertained" in increment.reasons[0]


def test_materiality_thresholds_are_declared_in_advance(report):
    assert report.criteria == MATERIALITY
    assert set(MATERIALITY) == {
        "min_added_correct_cases", "min_relative_gain", "min_precision_on_added"
    }


def test_a_stage_that_changes_nothing_is_not_adopted(pipeline, definition_v1):
    """v1 refuses extracted evidence, so reading text can buy it nothing."""
    body = run_ablation(
        definition_v1, pipeline.store, pipeline.configs, "rules"
    ).to_dict()
    assert body["decision"].startswith("DO NOT ADOPT")
    assert body["increments"][0]["added_cases"] == 0


def test_volume_bought_by_guessing_is_refused():
    """Precision on the added cases is a veto, not a tiebreak."""
    stage_a = StageResult(
        stage_id="a", label="a", readable_sources=(), description="",
        gold={"r1": "case", "r2": "non_case", "r3": "non_case"},
    )
    stage_b = StageResult(
        stage_id="b", label="b", readable_sources=("reported_term",),
        description="", gold=stage_a.gold,
    )

    class FakeAssignment:
        def __init__(self, record_id, verdict):
            self.record_id, self.verdict = record_id, verdict
            self.ascertained = verdict in ("case", "non_case")

    stage_a.assignments = [FakeAssignment("r1", "case")]
    stage_b.assignments = [
        FakeAssignment("r1", "case"),
        FakeAssignment("r2", "case"),
        FakeAssignment("r3", "case"),
    ]
    report = AblationReport(definition=None, stages=[stage_a, stage_b])
    increment = report.increments()[0]
    assert increment.added_cases == 2
    assert increment.added_correct == 0
    assert not increment.material
    assert "worse cohort" in increment.decision


def test_the_headline_is_the_first_increment(report):
    assert report.headline() == report.increments()[0].decision


# -- rendering ----------------------------------------------------------------


def test_the_rendered_output_ends_in_the_decision(report):
    text = format_ablation(report)
    assert text.strip().splitlines()[-1].startswith("  DECISION:")
    assert "correctly ascertained" in text or "changed no verdict" in text


def test_the_note_separates_extraction_accuracy_from_value(report):
    body = report.to_dict()
    assert "not about extraction accuracy" in body["note"]


# -- the stages are isolated from each other ---------------------------------

def test_a_stage_does_not_contaminate_the_next(pipeline, configs):
    """The configs are copied per stage, never mutated in place."""
    before = list(configs.extraction.readable_sources)
    run_ablation(
        pipeline.definition("cutaneous_mucosal", 2), pipeline.store, configs,
        "rules",
    )
    assert list(configs.extraction.readable_sources) == before


def test_stage_definitions_are_declared_once(report):
    assert [s[0] for s in STAGES] == [s.stage_id for s in report.stages]
