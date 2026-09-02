"""The evaluation harnesses: ablation, availability, transport, invariance."""

from __future__ import annotations

import pytest

from aelayer.eval.harness import (
    ABLATION_NOTE,
    DISCLAIMER,
    INVARIANCE_CAVEAT,
    EvaluationHarness,
    run_evaluation,
)
from aelayer.eval.report import render_markdown
from aelayer.eval.transport import DEVELOPMENT_PROFILES, transportability


@pytest.fixture(scope="module")
def harness(pipeline, definition_v1):
    return EvaluationHarness.build(pipeline, definition_v1)


@pytest.fixture(scope="module")
def results(harness):
    return harness.run_all()


# -- phenotype --------------------------------------------------------------


def test_the_not_ascertainable_rate_is_a_first_class_number(results):
    pooled = results["phenotype"]["pooled"]
    assert "not_ascertainable_rate" in pooled
    assert pooled["not_ascertainable_predicted"] > 0
    assert 0.0 <= pooled["not_ascertainable_rate"] <= 1.0
    # It is not folded into the negatives.
    assert pooled["fp"] + pooled["tp"] == pooled["predicted_cases"]


def test_the_not_ascertainable_rate_is_reported_per_profile(results):
    per_profile = results["phenotype"]["per_profile"]
    assert per_profile["P3_prespecified"]["not_ascertainable_rate"] > 0
    assert per_profile["P1_structured"]["not_ascertainable_rate"] < \
        per_profile["P3_prespecified"]["not_ascertainable_rate"]


def test_cases_are_attributed_to_the_routes_that_produced_them(results):
    phenotype = results["phenotype"]
    assert set(phenotype["attribute_methods"]) <= {"direct", "normalized", "extracted"}
    assert phenotype["cases_depending_on_text"] > 0


# -- ablation ---------------------------------------------------------------


def test_the_ablation_names_what_text_recovery_is_worth(results):
    ablation = results["ablation"]
    assert ablation["note"] == ABLATION_NOTE
    assert ablation["cases_with_text"] > ablation["cases_structured_only"]
    assert ablation["cases_only_findable_through_text"] > 0
    assert 0.0 < ablation["fraction_only_findable_through_text"] <= 1.0


def test_the_ablation_shows_text_resolving_unascertainable_episodes(results):
    """Without text those events are not negatives — they are unanswerable."""
    ablation = results["ablation"]
    assert ablation["not_ascertainable_structured_only"] > \
        ablation["not_ascertainable_with_text"]
    assert ablation["not_ascertainable_resolved_by_text"] > 0


def test_the_ablation_runs_the_same_definition_both_times(harness, results):
    """Only the accepted routes differ, so nothing else can explain the gap."""
    examples = results["ablation"]["examples"]
    assert examples
    assert {e["profile"] for e in examples} <= {"P2_text", "P5_comment", "P6_both"}
    assert all(e["without_text"] != "case" for e in examples)


# -- availability -----------------------------------------------------------


def test_the_availability_matrix_separates_the_kinds_of_missing(results):
    availability = results["availability"]
    assert availability["confusion"]["total"] > 0
    assert availability["accuracy"] > 0.9
    # The failure that quietly biases everything downstream.
    assert availability["missing_read_as_collected"] == 0


def test_not_collected_is_never_read_as_a_negative(records):
    for record in records:
        if record.location.availability == "not_collected_by_protocol":
            assert not record.location.is_evidence_of_absence


# -- transport --------------------------------------------------------------


def test_the_holdout_is_by_study_not_by_row(results):
    transport = results["transport"]
    assert transport["split"] == "whole_study"
    assert "never rows" in transport["note"]
    assert not set(transport["development_profiles"]) & set(
        transport["held_out_profiles"]
    )


def test_the_development_side_is_the_profiles_the_rules_were_written_against(results):
    assert set(results["transport"]["development_profiles"]) == set(
        DEVELOPMENT_PROFILES
    )


def test_a_custom_holdout_is_honoured(pipeline, definition_v1):
    result = transportability(pipeline, definition_v1, ["P6_both"])
    assert result["held_out_profiles"] == ["P6_both"]
    assert len(result["development_profiles"]) == 5


def test_holding_out_everything_is_refused(pipeline, definition_v1, profiles):
    with pytest.raises(ValueError, match="at least one profile on each side"):
        transportability(pipeline, definition_v1, profiles.profile_ids())


def test_holding_out_an_unknown_profile_is_refused(pipeline, definition_v1):
    with pytest.raises(ValueError, match="no such profile"):
        transportability(pipeline, definition_v1, ["P9_imaginary"])


def test_the_report_says_the_gap_is_not_overfitting(results):
    assert "not overfitting" in results["transport"]["not_fitted"]


# -- invariance -------------------------------------------------------------


def test_invariance_separates_raw_agreement_from_supported_agreement(results):
    invariance = results["invariance"]
    assert invariance["truths_compared"] > 0
    assert invariance["agreement_where_evidence_supports_it"] >= \
        invariance["raw_agreement"]


def test_invariance_states_what_it_does_not_establish(results):
    caveat = results["invariance"]["caveat"]
    assert "Consistency across representations is not clinical validity" in caveat
    assert caveat == INVARIANCE_CAVEAT


# -- reproducibility --------------------------------------------------------


def test_the_same_inputs_produce_the_same_run_twice(results):
    repro = results["reproducibility"]
    assert repro["manifest_id_stable"]
    assert repro["results_stable"]
    assert repro["normalization_stable"]


# -- the report -------------------------------------------------------------


def test_the_report_disclaims_before_any_number(results):
    body = render_markdown(results)
    assert DISCLAIMER in body
    assert body.index(DISCLAIMER) < body.index("## Silver standard")


def test_the_report_labels_the_silver_standard_as_silver(results):
    body = render_markdown(results)
    assert "Silver standard" in body
    assert "not ground truth" in body


def test_the_report_names_every_version_behind_the_numbers(results):
    body = render_markdown(results)
    for value in (
        results["versions"]["normalizer_version"],
        results["versions"]["extractor_version"],
        results["definition"]["hash"],
    ):
        assert value in body


def test_run_evaluation_writes_where_it_is_told(pipeline, tmp_path):
    results, written = run_evaluation(
        pipeline, "te_truncal_rash", 1, tmp_path / "nested" / "eval.md"
    )
    assert written.exists()
    assert written.read_text(encoding="utf-8").startswith("# Adverse event")
    assert results["definition"]["version"] == 1


def test_run_evaluation_can_be_asked_for_no_report(pipeline):
    _results, written = run_evaluation(pipeline, "te_truncal_rash", 1, None)
    assert written is None
