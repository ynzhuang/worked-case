"""The evaluation harness and its caveats."""

from __future__ import annotations

import pytest

from aelayer.eval.harness import (
    DISCLAIMER, INVARIANCE_CAVEAT, SILENCE_CAVEAT, EvaluationHarness,
)
from aelayer.eval.report import render_markdown
from aelayer.eval.transport import DEVELOPMENT_PROFILES, transportability


@pytest.fixture(scope="module")
def harness(pipeline, definition_v2):
    return EvaluationHarness.build(pipeline, definition_v2)


@pytest.fixture(scope="module")
def results(harness):
    return harness.run_all()


# -- the phenotype numbers ----------------------------------------------------


def test_the_not_ascertainable_rate_is_its_own_number(harness):
    pooled = harness.phenotype()["pooled"]
    assert "not_ascertainable_rate" in pooled
    assert 0.0 < pooled["not_ascertainable_rate"] < 1.0
    assert pooled["ppv"] > 0 and pooled["sensitivity"] > 0


def test_per_study_numbers_are_reported_separately(harness):
    per_study = harness.phenotype()["per_study"]
    assert len(per_study) == 7
    absent = next(
        m for study, m in per_study.items() if m["not_ascertainable_rate"] == 1.0
    )
    assert absent["ppv"] == 0.0, (
        "a study that cannot ascertain anything should not report a PPV above 0"
    )


# -- silence versus a documented negative -------------------------------------


def test_silence_is_never_read_as_an_assertion(harness):
    body = harness.assertion_confusion()
    assert body["silence_read_as_an_assertion"] == 0, (
        "the system invented an assertion where the source said nothing"
    )


def test_documented_negatives_are_recovered_or_missed_never_flipped(harness):
    body = harness.assertion_confusion()
    matrix = body["assertion_matrix"]
    assert body["documented_negatives_recovered"] > 0
    # An absence may be missed (read as silence) but must never be read as a
    # presence: that would turn a non-case into a case.
    assert matrix.get("absent", "present") == 0


def test_the_silence_caveat_is_stated(harness):
    assert "biases" in SILENCE_CAVEAT or "wrong by however many" in SILENCE_CAVEAT
    assert harness.assertion_confusion()["caveat"] == SILENCE_CAVEAT


# -- invariance ---------------------------------------------------------------


def test_invariance_says_it_is_not_validity(harness):
    body = harness.invariance()
    assert body["caveat"] == INVARIANCE_CAVEAT
    assert "is not clinical validity" in body["caveat"]
    assert "consistently wrong" in body["caveat"]


def test_invariance_is_measured_where_evidence_supports_it(harness):
    body = harness.invariance()
    assert body["truths_compared"] > 0
    assert body["agreement_where_evidence_supports_it"] >= body["raw_agreement"], (
        "a study that could not record the modifier is being counted as a "
        "disagreement, which punishes the system for a collection decision"
    )


# -- transportability ---------------------------------------------------------


def test_the_holdout_is_by_study_never_by_row(pipeline, definition_v2):
    body = transportability(pipeline, definition_v2)
    assert body["split"] == "whole_study"
    assert body["row_splits"] == "disallowed"
    assert set(body["development_profiles"]) == set(DEVELOPMENT_PROFILES)
    assert not set(body["development_profiles"]) & set(body["held_out_profiles"])


def test_the_holdout_reports_a_drop_and_says_nothing_is_fitted(pipeline,
                                                               definition_v2):
    body = transportability(pipeline, definition_v2)
    assert "sensitivity_drop" in body
    assert "is fitted to data" in body["not_fitted"]
    assert body["held_out"]["not_ascertainable_rate"] > \
        body["development"]["not_ascertainable_rate"], (
            "the held-out studies are supposed to be harder to ascertain"
        )


def test_an_unknown_holdout_profile_is_refused(pipeline, definition_v2):
    with pytest.raises(ValueError) as exc:
        transportability(pipeline, definition_v2, ["P_nonexistent"])
    assert "no such profile" in str(exc.value)


def test_a_split_needs_both_sides(pipeline, definition_v2, profiles):
    with pytest.raises(ValueError):
        transportability(pipeline, definition_v2, profiles.profile_ids())


# -- reproducibility ----------------------------------------------------------


def test_identical_inputs_reproduce_identically(harness):
    body = harness.reproducibility(repeats=2)
    assert body["manifest_id_stable"]
    assert body["results_stable"]
    assert body["normalization_stable"]


# -- the report ---------------------------------------------------------------


def test_the_report_leads_with_the_disclaimer_and_the_decision(results):
    markdown = render_markdown(results)
    assert markdown.index(DISCLAIMER) < markdown.index("## The decision")
    assert markdown.index("## The decision") < markdown.index("## Phenotype")


def test_the_report_prints_both_silver_caveats_verbatim(results):
    from aelayer.silver import SILVER_CAVEATS

    markdown = render_markdown(results)
    for caveat in SILVER_CAVEATS:
        assert caveat in markdown


def test_the_report_prints_the_denominator_note(results):
    from aelayer.models import DENOMINATOR_NOTE

    assert DENOMINATOR_NOTE in render_markdown(results)


def test_the_report_states_the_decision_in_bold(results):
    markdown = render_markdown(results)
    assert f"**{results['ablation']['decision']}**" in markdown


def test_the_report_writes_to_disk(pipeline, tmp_path):
    from aelayer.eval.harness import run_evaluation

    _results, path = run_evaluation(
        pipeline, "cutaneous_mucosal", 2, report_path=tmp_path / "r.md"
    )
    assert path.exists()
    assert "evaluation report" in path.read_text()
