"""One modifier, five collection homes, one rule.

The claim under test: the same phenotype definition runs across studies that
record mucosal involvement in a structured qualifier, in a linked form, in the
investigator's own words, in a comment, or nowhere — and the fifth case is
`not_ascertainable`, not a negative.
"""

from __future__ import annotations

from collections import Counter

import pytest

MODIFIER = "mucosal_involvement"


def _by_profile(records, profile):
    return [r for r in records if r.profile == profile]


def test_every_declared_home_is_exercised(profiles):
    kinds = set()
    for profile in profiles.profiles.values():
        for home in profile.homes_for(MODIFIER):
            kinds.add(home.kind)
    assert kinds == {
        "structured_standard", "linked_form", "reported_term", "comment", None
    }


def test_structured_qualifier_arrives_by_the_direct_route(structured_records):
    rows = [
        r.modifiers[MODIFIER] for r in _by_profile(structured_records, "P_structured")
        if r.modifiers[MODIFIER].observed
    ]
    assert rows
    assert all(a.method == "direct" and a.source_variable == "AEMUCOS" for a in rows)


def test_linked_form_arrives_by_the_direct_route(structured_records):
    rows = [
        r.modifiers[MODIFIER] for r in _by_profile(structured_records, "P_version")
        if r.modifiers[MODIFIER].observed
    ]
    assert rows
    assert all(
        a.method == "direct" and a.source_variable == "SC.MUCOSAL" for a in rows
    )


def test_reported_term_needs_the_model_path(structured_records, records):
    """Before extraction the modifier is unresolved; after it, it is observed."""
    before = _by_profile(structured_records, "P_text")
    after = _by_profile(records, "P_text")
    assert all(not r.modifiers[MODIFIER].observed for r in before)
    recovered = [r for r in after if r.modifiers[MODIFIER].observed]
    assert recovered
    assert all(
        r.modifiers[MODIFIER].method == "extracted"
        and r.modifiers[MODIFIER].source_variable == "AETERM"
        for r in recovered
    )


def test_comment_records_are_read_as_a_separate_home(records):
    rows = [
        r.modifiers[MODIFIER] for r in _by_profile(records, "P_concept_variant")
        if r.modifiers[MODIFIER].observed
    ]
    assert rows
    assert all(a.source_variable == "CO.COVAL" for a in rows)


def test_no_home_means_not_collected_and_never_a_negative(records, pipeline,
                                                          definition_v2):
    rows = _by_profile(records, "P_absent")
    assert rows
    for record in rows:
        attribute = record.modifiers[MODIFIER]
        assert attribute.availability == "not_collected"
        assert attribute.assertion is None

    verdicts = Counter(
        a.verdict for a in pipeline.assignments(definition_v2)
        if a.profile == "P_absent"
    )
    assert verdicts["not_ascertainable"] > 0
    assert verdicts["case"] == 0
    assert verdicts["non_case"] == 0, (
        "a study that never asked cannot produce an evaluated negative"
    )


def test_the_negated_profile_produces_documented_negatives(records):
    """The only way to prove an observed negative is told from silence."""
    rows = [
        r.modifiers[MODIFIER] for r in _by_profile(records, "P_negated")
        if r.modifiers[MODIFIER].documented_negative
    ]
    assert rows, "P_negated produced no documented negative"
    for attribute in rows:
        assert attribute.method == "extracted"
        assert attribute.evidence
        assert attribute.availability == "observed"


def test_documented_negatives_become_non_cases(pipeline, definition_v2):
    """And therefore enter the denominator."""
    assignments = [
        a for a in pipeline.assignments(definition_v2)
        if a.profile == "P_negated" and a.verdict == "non_case"
    ]
    assert assignments
    assert any(
        "documented negative" in a.reason for a in assignments
    )


def test_one_rule_runs_across_every_profile(pipeline, definition_v2, profiles):
    """No branch in the definition names a study, a variable or a route."""
    seen = {a.profile for a in pipeline.assignments(definition_v2)}
    assert seen == set(profiles.profile_ids())
    text = (definition_v2.model_dump_json())
    for variable in ("AEMUCOS", "SC.MUCOSAL", "AETERM", "CO.COVAL"):
        assert variable not in text, (
            f"the definition names {variable}; the rule must be route-agnostic"
        )


def test_the_route_travels_with_every_case(pipeline, definition_v2):
    cases = [a for a in pipeline.assignments(definition_v2) if a.verdict == "case"]
    assert cases
    for assignment in cases:
        assert assignment.attribute_methods
        assert assignment.attribute_sources
        assert assignment.evidence_spans


@pytest.mark.parametrize("method", ["direct", "extracted"])
def test_both_accepted_methods_actually_produce_cases(pipeline, definition_v2,
                                                      method):
    """`accept_methods: [direct, extracted]` is the demonstration."""
    cases = [a for a in pipeline.assignments(definition_v2) if a.verdict == "case"]
    assert any(
        a.attribute_methods.get(MODIFIER) == method for a in cases
    ), f"no case was decided by a {method} reading of the modifier"


def test_exposure_relation_is_derived_across_domains(records):
    resolved = [r.exposure_relation for r in records if r.exposure_relation.observed]
    assert resolved
    for attribute in resolved:
        assert attribute.method == "derived"
        assert attribute.source == "cross_domain"
        assert attribute.source_variable == "AE+EX"
        assert attribute.evidence


def test_every_observed_attribute_traces_to_a_span(records):
    defects = [r.record_id for r in records if not r.has_full_provenance()]
    assert defects == []
