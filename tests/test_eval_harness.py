"""The harness: three layers, a stress test, and what it refuses to claim."""

from __future__ import annotations

import datetime as _dt

import pytest

from aelayer.eval.harness import (
    DISCLAIMER,
    INVARIANCE_CAVEAT,
    EvaluationHarness,
    run_evaluation,
)
from aelayer.eval.report import render_markdown


@pytest.fixture(scope="module")
def harness(pipeline, definition_v1):
    return EvaluationHarness.build(pipeline, definition_v1)


@pytest.fixture(scope="module")
def results(harness):
    return harness.run_all()


# -- Layer 1 ----------------------------------------------------------------


def test_layer1_reports_a_collection_state_confusion_matrix(results):
    confusion = results["layer1"]["collection_state_confusion"]
    assert confusion["total"] > 0
    assert 0.0 <= confusion["accuracy"] <= 1.0
    assert len({r for r in confusion["labels"]}) >= 5


def test_layer1_separates_the_deterministic_path_from_the_model_path(results):
    paths = results["layer1"]["by_source_path"]
    assert "structured" in paths
    assert "text" in paths, "the model path recovered nothing, so it is unmeasured"


def test_layer1_scores_abstention_as_its_own_outcome(results):
    """Declining is a behaviour to be measured, not a gap in the numbers."""
    abstention = results["layer1"]["abstention"]
    asked = sum(
        abstention[k] for k in
        ("correct_abstention", "wrong_abstention", "correct_answer", "wrong_answer")
    )
    assert asked > 0
    assert 0.0 <= abstention["abstention_precision"] <= 1.0
    assert 0.0 <= abstention["answer_precision"] <= 1.0


def test_dates_are_compared_at_the_resolution_the_corpus_records_them(results):
    """A parsed midnight datetime and a gold date are the same answer."""
    from aelayer.eval.harness import _comparable

    assert _comparable("onset_datetime", _dt.datetime(2024, 2, 6, 0, 0)) == "2024-02-06"
    assert _comparable("onset_datetime", "2024-02-06T00:00:00") == "2024-02-06"
    assert _comparable("severity", "severe") == "severe"
    for name in ("onset_datetime", "end_datetime"):
        assert results["layer1"]["overall"][name]["recall"] == 1.0


def test_a_populated_field_without_a_span_is_reported_as_a_defect(results):
    assert results["layer1"]["provenance_violations"] == []


# -- Layer 2 ----------------------------------------------------------------


def test_layer2_reports_over_merge_and_over_split_separately(results):
    layer2 = results["layer2"]
    assert "over_merge" in layer2 and "over_split" in layer2
    assert layer2["gold_episodes"] > 0
    assert 0.0 <= layer2["boundary_agreement"] <= 1.0


def test_layer2_breaks_recurrence_out_from_everything_else(results):
    """The default merge rule is wrong exactly where recurrence is expected."""
    layer2 = results["layer2"]
    assert "recurrence_expected" in layer2
    assert "recurrence_not_expected" in layer2
    assert layer2["recurrence_expected"]["episodes"] > 0


def test_layer2_reports_which_linkage_rules_did_the_work(results):
    rules = results["layer2"]["linkage_rules"]
    assert rules
    assert set(rules) <= {
        "single_record", "explicit_continuation", "declared_convention",
        "temporal_overlap", "gap_within_tolerance", "recurrence_split",
    }, rules


# -- Layer 3 ----------------------------------------------------------------


def test_layer3_reports_ppv_and_sensitivity_against_gold(results):
    pooled = results["layer3"]["pooled"]
    assert 0.0 <= pooled["ppv"] <= 1.0
    assert 0.0 <= pooled["sensitivity"] <= 1.0
    assert pooled["gold_cases"] > 0


def test_a_miss_the_system_declined_to_call_is_counted_apart(results):
    """Routing to review is a different failure from silently missing a case."""
    pooled = results["layer3"]["pooled"]
    assert "false_negatives_from_linkage_review" in pooled
    assert pooled["false_negatives_from_linkage_review"] <= pooled["fn"]
    assert pooled["sensitivity_excluding_declined"] >= pooled["sensitivity"]


def test_layer3_reports_the_review_set_rather_than_folding_it_in(results):
    layer3 = results["layer3"]
    assert layer3["review_set_size"] == layer3["counts_by_verdict"].get("review", 0)


def test_transportability_is_reported_with_what_the_gap_means(results):
    transport = results["layer3"]["transportability"]
    assert transport["held_out_studies"]
    assert "not overfitting" in transport["note"]


# -- invariance -------------------------------------------------------------


def test_invariance_compares_one_truth_across_its_renderings(results):
    invariance = results["invariance"]
    assert invariance["truths_compared"] > 0
    assert len(invariance["representations"]) > 1
    assert 0.0 <= invariance["verdict_agreement"] <= 1.0


def test_invariance_says_in_so_many_words_what_it_does_not_establish(results):
    caveat = results["invariance"]["caveat"]
    assert (
        "Consistency across representations does not establish clinical validity"
        in caveat
    )
    assert caveat == INVARIANCE_CAVEAT


def test_a_discordant_truth_names_the_renderings_that_disagreed(results):
    for entry in results["invariance"]["discordant"]:
        assert entry["truth_id"]
        assert len(set(entry["verdicts"].values())) > 1
        assert entry["majority"] in entry["verdicts"].values()
        assert entry["example_reason"]


# -- reproducibility --------------------------------------------------------


def test_the_same_inputs_produce_the_same_run_twice(results):
    repro = results["reproducibility"]
    assert repro["manifest_id_stable"]
    assert repro["results_stable"]
    assert repro["normalization_stable"]


# -- retrieval --------------------------------------------------------------


def test_the_assertion_filter_is_measured_on_and_off(results):
    retrieval = results["retrieval"]
    assert retrieval["available"]
    on = retrieval["assertion_filter_on"]["negation_false_positive_rate"]
    off = retrieval["assertion_filter_off"]["negation_false_positive_rate"]
    assert on <= off, "the assertion filter is not earning its place"


# -- the report -------------------------------------------------------------


def test_the_report_says_the_numbers_are_synthetic_before_any_number(results):
    body = render_markdown(results)
    assert DISCLAIMER in body
    assert body.index(DISCLAIMER) < body.index("## Layer 1")


def test_the_report_carries_the_invariance_caveat_verbatim(results):
    assert INVARIANCE_CAVEAT in render_markdown(results)


def test_the_report_names_every_version_that_produced_the_numbers(results):
    body = render_markdown(results)
    for value in (
        results["versions"]["normalizer_version"],
        results["versions"]["extractor_version"],
        results["versions"]["snapshot_id"],
        results["definition"]["hash"],
    ):
        assert value in body


def test_run_evaluation_writes_the_report_where_it_is_told(pipeline, tmp_path):
    target = tmp_path / "nested" / "eval.md"
    written_results, written = run_evaluation(
        pipeline, "te_symptomatic_hypoglycemia", 1, target
    )
    assert written == target
    assert target.read_text(encoding="utf-8").startswith("# Adverse event")
    assert written_results["definition"]["version"] == 1


def test_run_evaluation_can_be_asked_for_no_report(pipeline):
    _results, written = run_evaluation(pipeline, "te_symptomatic_hypoglycemia", 1, None)
    assert written is None
