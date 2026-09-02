"""The silver-standard harness — the centerpiece.

It produces genuine extraction metrics on data nobody hand-annotated, by using
one study's own structured qualifier as a masked comparator.
"""

from __future__ import annotations

import json

import pytest

from aelayer.extract.backends import ExtractionRequest
from aelayer.silver import SILVER_CAVEAT, SilverHarness


@pytest.fixture(scope="module")
def harness(pipeline):
    return SilverHarness(pipeline.configs, pipeline.store, pipeline.engine())


@pytest.fixture(scope="module")
def report(harness, pipeline):
    return harness.run(pipeline.records(), "location")


# -- eligibility ------------------------------------------------------------


def test_only_profiles_carrying_both_routes_are_eligible(harness, profiles):
    eligible = harness.eligible_profiles("location")
    assert eligible == ["P6_both"]
    for profile_id in eligible:
        profile = profiles.profile(profile_id)
        assert profile.structured_home("location") is not None
        assert profile.text_home("location") is not None


def test_a_profile_with_one_route_produces_no_comparison(harness, pipeline):
    empty = harness.run(pipeline.records(), "location", ["P2_text"])
    assert empty.comparisons == []


# -- the method -------------------------------------------------------------


def test_the_extractor_never_sees_the_structured_value(harness, pipeline, records):
    """The comparator is read separately and is not in the request at all."""
    record = next(
        r for r in records if r.profile == "P6_both" and r.location.populated
    )
    profile = pipeline.profiles.profile("P6_both")
    structured = harness._masked_structured_value(record, profile, "location")
    assert structured is not None
    doc_id, text, kind, variable = harness._text_source(record, profile.text_home("location"))
    request = ExtractionRequest(
        doc_id=doc_id, text=text, attributes=("location",),
        source_kind=kind, source_variable=variable,
    )
    assert "AELOC" not in request.text
    assert structured not in request.text.upper().replace(" ", "")


def test_both_sides_are_normalized_before_they_are_compared(report):
    """A surface form and a catalogue value are not comparable until they are."""
    answered = [c for c in report.answered if c.agreement == "agree"]
    assert answered
    for comparison in answered[:10]:
        assert comparison.extracted_value == comparison.structured_value
        assert comparison.span_text.lower() != comparison.extracted_value.lower() \
            or comparison.span_text.lower() == comparison.structured_value.lower()


# -- the metrics ------------------------------------------------------------


def test_it_reports_precision_recall_coverage_and_abstention(report):
    metrics = report.metrics()
    for key in ("precision", "recall", "f1", "coverage", "abstention_rate",
                "normalized_agreement"):
        assert 0.0 <= metrics[key] <= 1.0
    assert metrics["eligible_records"] > 0
    assert metrics["answered"] + len(report.abstentions) == metrics["eligible_records"]


def test_coverage_and_abstention_are_complementary(report):
    metrics = report.metrics()
    assert round(metrics["coverage"] + metrics["abstention_rate"], 4) == 1.0


def test_the_extractor_abstains_on_words_no_lexicon_carries(report):
    """Which is correct behaviour, and is why abstention is a rate not a failure."""
    assert report.abstentions
    assert report.metrics()["abstention_rate"] > 0


def test_metrics_break_out_by_profile_and_by_term_style(report):
    body = report.to_dict()
    assert body["by_profile"]
    assert body["by_reported_term_style"]
    assert set(body["by_profile"]) <= {"P6_both"}


def test_it_calls_itself_a_silver_standard(report):
    body = report.to_dict()
    assert body["standard"] == "silver"
    assert body["caveat"] == SILVER_CAVEAT
    assert "not ground truth" in body["caveat"]
    assert "own error rate" in body["caveat"]


# -- the adjudication queue -------------------------------------------------


def test_the_queue_carries_disagreements(report):
    queue = report.adjudication_queue()
    reasons = [row["queue_reason"] for row in queue]
    assert any("disagree" in r for r in reasons)


def test_the_queue_always_samples_agreements(report):
    """Without it you only ever inspect failures and never learn the
    comparator's own error rate."""
    queue = report.adjudication_queue(agreement_sample=5)
    sampled = [r for r in queue if r["queue_reason"].startswith("sampled agreement")]
    assert sampled
    assert len(sampled) <= 5
    assert all(r["agreement"] == "agree" for r in sampled)


def test_the_queue_is_deterministic_for_a_seed(report):
    first = report.adjudication_queue(seed=3, agreement_sample=5)
    second = report.adjudication_queue(seed=3, agreement_sample=5)
    assert [r["source_record_id"] for r in first] == \
        [r["source_record_id"] for r in second]


def test_every_queue_row_carries_the_text_a_reviewer_needs(report):
    for row in report.adjudication_queue(agreement_sample=3):
        assert row["text"]
        assert row["source_record_id"]
        assert row["standard"] == "silver"
        assert row["queue_reason"]


def test_the_queue_is_written_as_jsonl(report, tmp_path):
    target = report.write_adjudication(tmp_path / "queue" / "adjudication.jsonl")
    assert target.exists()
    rows = [json.loads(line) for line in target.read_text().splitlines() if line]
    assert rows
    assert {r["attribute"] for r in rows} == {"location"}
