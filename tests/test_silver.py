"""The silver-standard harness — one of the two things this build exists for.

What is tested here is not the score. It is that the harness cannot report a
number without the two things that bound it: an independence caveat and a
sampling caveat, both printed verbatim.
"""

from __future__ import annotations

import json

import pytest

from aelayer.silver import SILVER_CAVEATS, SilverHarness

MODIFIER = "mucosal_involvement"


@pytest.fixture(scope="module")
def report(pipeline, records):
    harness = SilverHarness(pipeline.configs, pipeline.store, pipeline.engine())
    return harness.run(records, MODIFIER)


# -- the caveats --------------------------------------------------------------


def test_there_are_exactly_two_caveats_and_they_say_the_right_things():
    assert len(SILVER_CAVEATS) == 2
    independence, sampling = SILVER_CAVEATS
    assert "NOT INDEPENDENT" in independence
    assert "same investigator" in independence
    assert "UPPER BOUND" in independence
    assert "NOT A RANDOM SUBSET" in sampling
    assert "only in free text" in sampling


def test_every_report_carries_both_caveats_verbatim(report):
    body = report.to_dict()
    assert body["caveats"] == list(SILVER_CAVEATS)
    for caveat in SILVER_CAVEATS:
        assert caveat in json.dumps(body)


def test_it_is_called_a_silver_standard_in_the_output(report):
    body = report.to_dict()
    assert body["standard"] == "silver"
    assert "not ground truth" in " ".join(body["caveats"])


# -- what is compared ---------------------------------------------------------


def test_the_comparator_is_only_available_where_both_routes_exist(report, profiles):
    assert report.profiles
    for profile_id in report.profiles:
        assert profiles.profile(profile_id).carries_both(MODIFIER)


def test_assertions_are_compared_not_just_values(report):
    """An extractor that turned every "no" into silence must not score well."""
    by_assertion = report.by_assertion()
    assert "present" in by_assertion
    assert "absent" in by_assertion, (
        "the comparator recorded no documented negative, so the class that "
        "decides the denominator is unmeasurable"
    )


def test_the_structured_value_never_reaches_the_extractor(pipeline, records):
    """The masking is real: the request carries text and the modifier name."""
    harness = SilverHarness(pipeline.configs, pipeline.store, pipeline.engine())
    seen: list = []
    original = pipeline.engine().backend.extract

    def spy(request):
        seen.append(request)
        return original(request)

    pipeline.engine().backend.extract = spy  # type: ignore[method-assign]
    try:
        harness.run(records, MODIFIER)
    finally:
        pipeline.engine().backend.extract = original  # type: ignore[method-assign]
    assert seen
    for request in seen:
        assert request.source_kind in ("reported_term", "comment")
        assert "AEMUCOS" not in request.text
        assert request.modifiers == (MODIFIER,)


# -- calibration --------------------------------------------------------------


def test_calibration_is_reported(report):
    calibration = report.calibration()
    assert calibration["brier_score"] is not None
    assert 0.0 <= calibration["brier_score"] <= 1.0
    assert calibration["reliability"]
    for row in calibration["reliability"]:
        assert row["n"] > 0
        assert 0.0 <= row["observed_accuracy"] <= 1.0
        assert row["gap"] == round(
            row["mean_confidence"] - row["observed_accuracy"], 4
        )


def test_calibration_says_which_direction_matters(report):
    assert "more confident than it was right" in report.calibration()["note"]


# -- the adjudication queue ---------------------------------------------------


def test_the_queue_includes_a_sample_of_agreements(report):
    queue = report.adjudication_queue(agreement_sample=5)
    reasons = {row["queue_reason"] for row in queue}
    assert any(r.startswith("sampled agreement") for r in reasons), (
        "the sample of agreements is not optional: without it the comparator's "
        "own error rate can never be estimated"
    )


def test_the_queue_is_deterministic(report):
    first = report.adjudication_queue(seed=3)
    second = report.adjudication_queue(seed=3)
    assert [r["source_record_id"] for r in first] == \
        [r["source_record_id"] for r in second]


def test_the_queue_can_be_written_out(report, tmp_path):
    path = report.write_adjudication(tmp_path / "queue.jsonl", agreement_sample=3)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows
    assert all(row["standard"] == "silver" for row in rows)


# -- coverage is reported beside precision ------------------------------------


def test_coverage_and_abstention_sit_beside_precision(report):
    metrics = report.metrics()
    for key in ("precision", "recall", "coverage", "abstention_rate"):
        assert key in metrics
    assert 0.0 <= metrics["coverage"] <= 1.0
    assert round(metrics["coverage"] + metrics["abstention_rate"], 4) == 1.0
