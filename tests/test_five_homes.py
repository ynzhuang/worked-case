"""The claim the prototype exists to demonstrate.

One clinically relevant attribute lives in any of five places, and which one
applies is a per-study collection decision. One frozen rule runs across all
five, returning the same verdict where the evidence supports it and
`not_ascertainable` where it does not.
"""

from __future__ import annotations

import collections

import pytest


def rash_records(records, profile):
    return [
        r for r in records
        if r.profile == profile and r.standardized_concept == "RASH"
    ]


# -- the five homes --------------------------------------------------------


def test_every_home_is_exercised_by_some_profile(profiles):
    homes = {
        home for profile in profiles.profiles.values()
        for home in profile.home_kinds("location")
    }
    assert homes == {
        "AELOC", "reported_term", "sponsor_variable", "comment", "none",
    }


@pytest.mark.parametrize(
    "profile,method,variable",
    [
        ("P1_structured", "direct", "AELOC"),
        ("P2_text", "extracted", "AETERM"),
        ("P4_sponsor", "normalized", "SUPPAE.RASHSITE"),
        ("P5_comment", "extracted", "CO.COVAL"),
        ("P6_both", "direct", "AELOC"),
    ],
)
def test_each_home_resolves_by_the_route_it_implies(records, profile, method, variable):
    resolved = [
        r for r in rash_records(records, profile) if r.location.populated
    ]
    assert resolved, f"{profile} resolved no location at all"
    assert {r.location.method for r in resolved} == {method}
    assert {r.location.source_variable for r in resolved} == {variable}


def test_the_study_that_records_it_nowhere_resolves_nothing(records):
    """Not a bug, and not a negative: the site was never collected."""
    rows = rash_records(records, "P3_prespecified")
    assert rows
    assert all(not r.location.populated for r in rows)
    assert {r.location.availability for r in rows} == {"not_collected_by_protocol"}
    assert all("not recoverable" in r.location.note for r in rows)


def test_the_sponsor_code_is_resolved_through_a_declared_mapping(records, profiles):
    profile = profiles.profile("P4_sponsor")
    resolved = [r for r in rash_records(records, "P4_sponsor") if r.location.populated]
    assert resolved
    for record in resolved:
        assert record.location.value in profile.sponsor_codelist
        assert "sponsor codelist" in record.location.note


def test_an_unmapped_sponsor_code_is_not_representable_rather_than_guessed(profiles):
    from aelayer.normalize.values import resolve_sponsor_value

    profile = profiles.profile("P4_sponsor")
    value, availability, note, variable = resolve_sponsor_value(
        profile, [{"QNAM": "RASHSITE", "QVAL": "ZZ"}], "location"
    )
    assert (value, availability) == (None, "not_representable")
    assert "does not cover" in note
    assert variable == "SUPPAE.RASHSITE"


# -- one rule, five homes ---------------------------------------------------


def test_one_definition_runs_across_every_profile(assignments):
    profiles_seen = {a.profile for a in assignments}
    assert len(profiles_seen) == 6


def test_the_verdict_is_the_same_where_the_evidence_supports_it(pipeline, assignments):
    """The invariance claim, checked directly rather than through the harness."""
    gold = {}
    for entry in pipeline.store.gold_episodes():
        for record_id in entry["source_record_ids"]:
            gold[record_id] = entry

    by_truth: dict[str, dict[str, str]] = collections.defaultdict(dict)
    for assignment in assignments:
        truth = next(
            (gold[r] for r in assignment.source_record_ids if r in gold), None
        )
        if truth is None or truth["cohort"] != "shared":
            continue
        if truth["location_available"]:
            by_truth[truth["truth_id"]][assignment.profile] = assignment.verdict

    compared = {t: b for t, b in by_truth.items() if len(b) >= 2}
    assert compared, "no truth was rendered with evidence under two profiles"
    disagreeing = {t: b for t, b in compared.items() if len(set(b.values())) > 1}
    assert not disagreeing, disagreeing


def test_the_study_that_cannot_record_it_returns_not_ascertainable(assignments):
    verdicts = collections.Counter(
        a.verdict for a in assignments if a.profile == "P3_prespecified"
    )
    assert verdicts["not_ascertainable"] > 0
    assert verdicts["case"] == 0


def test_not_ascertainable_is_counted_apart_from_not_case(assignments):
    """The distinction is the point: one is a finding, the other is a gap."""
    counts = collections.Counter(a.verdict for a in assignments)
    assert counts["not_ascertainable"] > 0
    assert counts["not_case"] > 0
    for assignment in assignments:
        if assignment.verdict == "not_ascertainable":
            assert assignment.deciding_attribute
            assert "cannot be evaluated" in assignment.reason \
                or "not recoverable" in assignment.reason


def test_a_case_names_the_route_its_evidence_came_by(assignments):
    cases = [a for a in assignments if a.verdict == "case"]
    assert cases
    for assignment in cases:
        assert assignment.attribute_methods
        assert assignment.attribute_sources
        assert set(assignment.attribute_methods) <= {"location", "onset"}


def test_cases_arrive_by_all_three_routes(assignments):
    methods = {
        m for a in assignments if a.verdict == "case"
        for m in a.attribute_methods.values()
    }
    assert methods == {"direct", "normalized", "extracted"}
