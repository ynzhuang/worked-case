"""Four verdicts, and the denominators they make possible."""

from __future__ import annotations

from collections import Counter

import pytest

from aelayer.models import ASCERTAINED, VERDICTS, Attribute, Span
from aelayer.phenotype import DefinitionError, PhenotypeEvaluator, load_definition

MODIFIER = "mucosal_involvement"


# -- the four verdicts --------------------------------------------------------


def test_all_four_verdicts_occur(result):
    counts = result.verdicts()
    for verdict in VERDICTS:
        assert counts.get(verdict, 0) > 0, f"no record reached {verdict}"


def test_a_documented_negative_is_a_non_case(pipeline, definition_v2):
    assignments = [
        a for a in pipeline.assignments(definition_v2) if a.verdict == "non_case"
    ]
    reasons = [a for a in assignments if "documented negative" in a.reason]
    assert reasons
    for assignment in reasons:
        finding = next(f for f in assignment.findings if f.name == MODIFIER)
        assert finding.assertion == "absent"
        assert finding.availability == "observed"


def test_silence_is_not_a_negative(pipeline, definition_v2):
    unascertainable = [
        a for a in pipeline.assignments(definition_v2)
        if a.verdict == "not_ascertainable"
    ]
    assert unascertainable
    for assignment in unascertainable:
        finding = next(
            (f for f in assignment.findings if not f.satisfied
             and f.verdict == "not_ascertainable"),
            None,
        )
        if finding is None or finding.name != MODIFIER:
            continue
        assert finding.assertion is None
        assert "not the same as saying no" in finding.reason


def test_an_uncertain_source_reaches_review_not_a_verdict(pipeline, definition_v2):
    reviews = [
        a for a in pipeline.assignments(definition_v2) if a.verdict == "review"
    ]
    assert reviews
    assert any("hedged" in a.reason for a in reviews)


def test_a_record_outside_the_concept_set_is_not_a_negative(pipeline,
                                                            definition_v2, records):
    evaluated = {a.record_id for a in pipeline.assignments(definition_v2)}
    distractors = [
        r for r in records
        if r.concept_id not in definition_v2.concept_set.include
    ]
    assert distractors
    for record in distractors:
        assert record.record_id not in evaluated, (
            "an unrelated event was counted as an evaluated negative, which "
            "would inflate the denominator with events nobody asked about"
        )


# -- denominators -------------------------------------------------------------


def test_the_ascertainable_fraction_is_reported_per_study(result):
    denominators = result.denominators()
    assert len(denominators) > 1
    for row in denominators:
        assert 0.0 <= row.ascertainable_fraction <= 1.0
        assert row.n_ascertainable == row.n_case + row.n_non_case


def test_unascertainable_records_enter_neither_numerator_nor_denominator(result):
    for row in result.denominators():
        assert row.n_ascertainable + row.n_review + row.n_not_ascertainable \
            == row.n_total
        if row.incidence is not None:
            assert row.incidence == round(row.n_case / row.n_ascertainable, 4)


def test_a_study_that_never_collects_has_no_incidence(result):
    absent = next(d for d in result.denominators() if d.profile == "P_absent")
    assert absent.ascertainable_fraction == 0.0
    assert absent.incidence is None


def test_the_denominator_note_says_what_was_excluded():
    from aelayer.models import DENOMINATOR_NOTE

    assert "neither the numerator nor the denominator" in DENOMINATOR_NOTE
    assert "ascertainable fraction" in DENOMINATOR_NOTE


def test_ascertained_is_case_and_non_case_only():
    assert ASCERTAINED == {"case", "non_case"}


# -- a second definition, no code changes ------------------------------------


def test_a_structurally_different_definition_runs_unchanged(pipeline, graded):
    """Grade and cumulative exposure, no modifier requirement at all."""
    assert graded.modifiers == []
    assert graded.grade is not None
    assert graded.cumulative_exposure is not None
    result = pipeline.evaluate(graded)
    assert result.assignments
    counts = result.verdicts()
    assert counts.get("case", 0) > 0
    assert counts.get("non_case", 0) > 0


def test_the_graded_definition_uses_a_derived_exposure_total(pipeline, graded):
    result = pipeline.evaluate(graded)
    findings = [
        f for a in result.assignments for f in a.findings
        if f.name == "cumulative_exposure" and f.satisfied
    ]
    assert findings
    for finding in findings:
        assert finding.method == "derived"
        assert finding.source == "cross_domain"


# -- accept_methods -----------------------------------------------------------


def test_a_definition_that_refuses_extracted_evidence_declines_to_use_it(
    pipeline, definition_v1, definition_v2
):
    v1 = {a.record_id: a for a in pipeline.assignments(definition_v1)}
    v2 = {a.record_id: a for a in pipeline.assignments(definition_v2)}
    moved = [
        rid for rid in v1
        if v1[rid].verdict == "not_ascertainable" and v2[rid].verdict == "case"
    ]
    assert moved, "the two versions claim the same records"
    example = v1[moved[0]]
    assert "does not accept" in example.reason


# -- the loader ---------------------------------------------------------------


def test_a_definition_naming_an_unknown_concept_is_refused(tmp_path, catalog):
    path = tmp_path / "bad.v1.yaml"
    path.write_text(
        "id: bad\nversion: 1\nstatus: frozen\nlabel: bad\n"
        "concept_set:\n  include: [NOT_A_CONCEPT]\n"
        "modifiers:\n  - name: mucosal_involvement\n",
        encoding="utf-8",
    )
    with pytest.raises(DefinitionError) as exc:
        load_definition(path, catalog)
    assert "catalogue does not define" in str(exc.value)


def test_a_definition_naming_an_unknown_modifier_is_refused(tmp_path, catalog):
    path = tmp_path / "bad2.v1.yaml"
    path.write_text(
        "id: bad2\nversion: 1\nstatus: frozen\nlabel: bad\n"
        "concept_set:\n  include: [RASH]\n"
        "modifiers:\n  - name: not_a_modifier\n",
        encoding="utf-8",
    )
    with pytest.raises(DefinitionError) as exc:
        load_definition(path, catalog)
    assert "not a configured modifier" in str(exc.value)


def test_a_definition_with_no_criterion_beyond_its_concept_is_refused(tmp_path,
                                                                      catalog):
    path = tmp_path / "bad3.v1.yaml"
    path.write_text(
        "id: bad3\nversion: 1\nstatus: frozen\nlabel: bad\n"
        "concept_set:\n  include: [RASH]\n",
        encoding="utf-8",
    )
    with pytest.raises(DefinitionError):
        load_definition(path, catalog)


def test_a_modifier_cannot_accept_the_derived_method(tmp_path, catalog):
    path = tmp_path / "bad4.v1.yaml"
    path.write_text(
        "id: bad4\nversion: 1\nstatus: frozen\nlabel: bad\n"
        "concept_set:\n  include: [RASH]\n"
        "modifiers:\n  - name: mucosal_involvement\n"
        "    accept_methods: [derived]\n",
        encoding="utf-8",
    )
    with pytest.raises(DefinitionError) as exc:
        load_definition(path, catalog)
    assert "computed across domains" in str(exc.value)


def test_a_frozen_definition_carries_a_content_hash(definition_v2):
    assert definition_v2.definition_hash
    assert definition_v2.status == "frozen"


def test_every_finding_names_what_decided_it(assignments):
    for assignment in assignments:
        assert assignment.reason
        assert assignment.findings
        if assignment.verdict != "case":
            assert assignment.deciding_criterion


# -- the evaluator in isolation ----------------------------------------------


def _record(monkeypatch=None, **modifier):
    from aelayer.models import CanonicalAERecord, CodedTerm

    span = Span(
        doc_id="AE:R1:AEMUCOS", start=0, end=1, field=MODIFIER,
        extracted_value="Y", text="Y", kind="structured",
    )
    return CanonicalAERecord(
        record_id="S:R1", study_id="S", subject_id="S-1", source_record_id="R1",
        concept_id="RASH",
        coded_event=CodedTerm(code="Rash", dictionary_version="D-21.0",
                              concept_id="RASH", reconciliation="unchanged"),
        modifiers={MODIFIER: modifier["attribute"]},
        exposure_relation=Attribute[int].derived(5, "AE+EX", [span]),
    )


@pytest.mark.parametrize(
    "attribute, expected",
    [
        (Attribute[str].direct("present", "AEMUCOS"), "case"),
        (Attribute[str].direct("absent", "AEMUCOS"), "non_case"),
        (Attribute[str].direct("uncertain", "AEMUCOS"), "review"),
        (Attribute[str].silent_because("not_collected"), "not_ascertainable"),
        (Attribute[str].silent_because("unresolved"), "not_ascertainable"),
    ],
)
def test_the_four_verdicts_map_from_the_two_fields(definition_v2, catalog,
                                                   attribute, expected):
    record = _record(attribute=attribute)
    assignment = PhenotypeEvaluator(definition_v2, catalog).evaluate(record)
    assert assignment.verdict == expected
