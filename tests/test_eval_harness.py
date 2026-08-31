"""The evaluation harness itself."""

from __future__ import annotations

import pytest

from aelayer.eval.harness import EvaluationHarness
from aelayer.eval.metrics import ConfusionMatrix, PRF, recall_at_k, reciprocal_rank, scalar_prf


@pytest.fixture(scope="module")
def harness(pipeline, definition_v1):
    return EvaluationHarness.build(pipeline, definition_v1)


def test_extraction_metrics_report_counts_behind_every_rate(harness):
    metrics = harness.extraction_metrics()
    for name, body in metrics["overall"].items():
        assert body["tp"] + body["fn"] == body["support"], name
        assert 0.0 <= body["f1"] <= 1.0, name


def test_extraction_metrics_break_out_by_assertion_and_pattern(harness):
    metrics = harness.extraction_metrics()
    assert len(metrics["by_assertion"]) >= 5
    assert len(metrics["by_pattern"]) >= 10
    assert "vague" in metrics["onset_by_phrasing"] or metrics["onset_by_phrasing"]


def test_the_assertion_confusion_matrix_is_produced(harness):
    metrics = harness.extraction_metrics()
    matrix = metrics["assertion_confusion_matrix"]
    assert matrix.total > 0
    assert 0.0 <= matrix.accuracy <= 1.0
    assert "gold \\ predicted" in matrix.to_markdown()


def test_the_harness_reports_provenance_violations_as_defects(harness):
    assert harness.extraction_metrics()["provenance_violations"] == []


def test_phenotype_metrics_give_ppv_and_sensitivity_pooled_and_per_study(harness):
    metrics = harness.phenotype_metrics()
    assert 0.0 <= metrics["pooled"]["ppv"] <= 1.0
    assert 0.0 <= metrics["pooled"]["sensitivity"] <= 1.0
    assert metrics["per_study"]
    for body in metrics["per_study"].values():
        assert body["tp"] + body["fn"] == body["gold_cases"]


def test_phenotype_metrics_include_the_full_three_way_confusion(harness):
    matrix = harness.phenotype_metrics()["verdict_confusion_matrix"]
    assert set(matrix.labels) >= {"case", "review", "excluded"}
    assert matrix.total > 0


def test_the_review_set_is_counted_separately(harness):
    metrics = harness.phenotype_metrics()
    assert metrics["review_set_size"] == metrics["counts_by_verdict"].get("review", 0)


def test_retrieval_metrics_contrast_the_assertion_filter(harness):
    metrics = harness.retrieval_metrics()
    on = metrics["assertion_filter_on"]
    off = metrics["assertion_filter_off"]
    assert on["negation_false_positive_rate"] == 0.0
    assert off["negation_false_positive_rate"] > 0.0
    assert off["records_with_assertion_absent"] > 0
    assert on["gold_negated_documents_returned"] == 0


def test_retrieval_metrics_report_recall_at_k_and_mrr(harness):
    metrics = harness.retrieval_metrics()
    for key in ("recall@5", "recall@10", "recall@20", "recall@50", "mrr"):
        assert 0.0 <= metrics["assertion_filter_on"][key] <= 1.0
    assert metrics["per_study"]
    assert 0.0 <= metrics["mean_mrr_per_study"] <= 1.0


def test_stability_is_measured_not_asserted(harness):
    stability = harness.stability_check(repeats=2)
    assert stability["extraction_stable"]
    assert stability["run_id_stable"]
    assert stability["results_stable"]
    assert len(set(stability["result_hashes"])) == 1


def test_the_sensitivity_sweep_moves_the_case_count(harness):
    """Definitional drift as a measurement rather than an argument."""
    sweeps = harness.sensitivity_sweep()["sweeps"]
    assert len(sweeps) >= 2
    for sweep in sweeps:
        low, high = sweep["case_count_range"]
        assert low <= high
        assert sweep["rows"]
    assert any(s["case_count_range"][0] != s["case_count_range"][1] for s in sweeps)


def test_the_glucose_sweep_is_monotone(harness):
    """A stricter threshold cannot produce more cases."""
    sweep = next(s for s in harness.sensitivity_sweep()["sweeps"]
                 if "glucose" in s["parameter"])
    rows = sorted(sweep["rows"], key=lambda r: -r["value"])
    counts = [r["case"] for r in rows]
    assert counts == sorted(counts, reverse=True)


def test_the_window_sweep_is_monotone(harness):
    sweep = next(s for s in harness.sensitivity_sweep()["sweeps"]
                 if "window" in s["parameter"])
    rows = sorted(sweep["rows"], key=lambda r: -r["value"])
    counts = [r["case"] for r in rows]
    assert counts == sorted(counts, reverse=True)


def test_the_markdown_report_carries_real_numbers(harness):
    from aelayer.eval.report import render_markdown

    results = {
        "generated_at": "now",
        "snapshot_id": "snap",
        "extractor_version": "ex",
        "definition": {"id": "d", "version": 1, "status": "frozen",
                       "hash": "h", "label": "L"},
        "corpus": harness.pipeline.store.summary(),
        "extraction": harness.extraction_metrics(),
        "phenotype": harness.phenotype_metrics(),
        "retrieval": harness.retrieval_metrics(),
        "stability": harness.stability_check(repeats=2),
        "sensitivity": harness.sensitivity_sweep(),
    }
    markdown = render_markdown(results)
    for heading in ("## 1. Extraction", "## 2. Phenotype", "## 3. Retrieval",
                    "## 4. Stability", "## 5. Definition sensitivity"):
        assert heading in markdown
    assert "synthetic corpus" in markdown
    assert "negation false positive rate" in markdown


def test_the_primary_event_is_chosen_without_consulting_gold(harness):
    """Otherwise the assertion metric would be measuring itself."""
    by_doc = harness.events_by_doc()
    doc_id = next(iter(by_doc))
    first = EvaluationHarness.primary_event(by_doc[doc_id], "HYPOGLYCEMIA")
    second = EvaluationHarness.primary_event(list(reversed(by_doc[doc_id])),
                                             "HYPOGLYCEMIA")
    assert (first is None and second is None) or first.event_id == second.event_id


# -- metric primitives -----------------------------------------------------


def test_prf_arithmetic():
    prf = PRF(tp=8, fp=2, fn=2)
    assert prf.precision == 0.8 and prf.recall == 0.8 and prf.f1 == pytest.approx(0.8)
    assert PRF().precision == 0.0 and PRF().f1 == 0.0


def test_a_wrong_slot_costs_both_precision_and_recall():
    assert scalar_prf("mild", "severe") == (0, 1, 1)
    assert scalar_prf("mild", "mild") == (1, 0, 0)
    assert scalar_prf(None, "mild") == (0, 1, 0)
    assert scalar_prf("mild", None) == (0, 0, 1)
    assert scalar_prf(None, None) == (0, 0, 0)


def test_confusion_matrix_per_label():
    matrix = ConfusionMatrix(labels=["a", "b"])
    matrix.add("a", "a")
    matrix.add("a", "b")
    matrix.add("b", "b")
    per_label = matrix.per_label()
    assert per_label["a"].tp == 1 and per_label["a"].fn == 1
    assert per_label["b"].tp == 1 and per_label["b"].fp == 1
    assert matrix.accuracy == pytest.approx(2 / 3)


def test_ranking_metrics():
    ranked = ["a", "b", "c", "d"]
    assert reciprocal_rank(ranked, {"c"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranked, {"z"}) == 0.0
    assert recall_at_k(ranked, {"a", "d"}, 2) == 0.5
    assert recall_at_k(ranked, set(), 2) == 0.0
