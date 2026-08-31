"""Onset timing: text expressions resolved against structured anchors."""

from __future__ import annotations

import datetime as _dt

import pytest

from aelayer.anchors import AnchorResolver, parse_date
from aelayer.extract.temporal import TemporalExtractor

ANCHOR_CONFIG = {
    "dose_escalation": {"domain": "EX", "rule": "dose_increase", "date_field": "EXSTDTC"},
    "first_dose": {"domain": "EX", "rule": "first_record", "date_field": "EXSTDTC"},
    "dose_reduction": {"domain": "EX", "rule": "dose_decrease", "date_field": "EXSTDTC"},
}

EXPOSURES = {
    "S1": [
        {"USUBJID": "S1", "EXSEQ": 1, "EXDOSE": 10, "EXSTDTC": "2024-01-01"},
        {"USUBJID": "S1", "EXSEQ": 2, "EXDOSE": 20, "EXSTDTC": "2024-02-01"},
        {"USUBJID": "S1", "EXSEQ": 3, "EXDOSE": 40, "EXSTDTC": "2024-03-01"},
    ],
    "S2": [],
}


@pytest.fixture(scope="module")
def resolver():
    return AnchorResolver(ANCHOR_CONFIG, EXPOSURES)


@pytest.fixture(scope="module")
def temporal(extraction_config, resolver):
    return TemporalExtractor(extraction_config, resolver)


def resolve(temporal, text, **kwargs):
    return temporal.resolve(
        subject_id=kwargs.pop("subject_id", "S1"),
        text=text,
        scope=None,
        default_anchor="dose_escalation",
        reference_start=kwargs.pop("reference_start", _dt.date(2024, 1, 1)),
        **kwargs,
    )


@pytest.mark.parametrize(
    "text,offset",
    [
        ("Hypoglycaemia six days after the dose escalation.", 6),
        ("Hypoglycaemia 6 days after the dose escalation.", 6),
        ("Within 8 days of the uptitration the subject was unwell.", 8),
        ("The event occurred 3 weeks post dose increase.", 21),
        ("Symptoms began four days later.", 4),
        ("The day after the dose escalation the subject felt shaky.", 1),
        ("On the day of the dose escalation the subject felt shaky.", 0),
    ],
)
def test_relative_expressions_resolve_to_day_offsets(temporal, text, offset):
    assert resolve(temporal, text).onset_offset_days == offset


def test_study_day_resolves_against_the_reference_start(temporal):
    result = resolve(temporal, "On study day 40 the subject was unwell.")
    assert result.onset_date == _dt.date(2024, 2, 9)
    assert result.onset_offset_days == 8  # 40 days from 1 Jan, 8 from 1 Feb
    assert result.source == "narrative_study_day"


def test_absolute_date_in_the_narrative(temporal):
    result = resolve(temporal, "Hypoglycaemia occurred on 14-Feb-2024.")
    assert result.onset_date == _dt.date(2024, 2, 14)
    assert result.onset_offset_days == 13


def test_the_ae_table_onset_date_takes_precedence(temporal):
    result = resolve(
        temporal,
        "Hypoglycaemia six days after the dose escalation.",
        recorded_onset="2024-02-10",
    )
    assert result.source == "structured_onset_date"
    assert result.onset_date == _dt.date(2024, 2, 10)
    assert result.onset_offset_days == 9


def test_the_anchor_named_in_text_beats_the_default(temporal):
    result = resolve(temporal, "Hypoglycaemia two days after the first dose.")
    assert result.anchor_event == "first_dose"
    assert result.onset_date == _dt.date(2024, 1, 3)


def test_offset_survives_when_no_anchor_date_resolves(temporal):
    """The offset is reported and the date stays null.

    The definition, not the extractor, decides what to do with that.
    """
    result = resolve(temporal, "Hypoglycaemia six days after the dose escalation.",
                     subject_id="S2")
    assert result.onset_offset_days == 6
    assert result.onset_date is None
    assert "no resolvable anchor" in result.detail


def test_unresolved_when_nothing_says_when(temporal):
    result = resolve(temporal, "The subject experienced hypoglycaemia at some point.")
    assert not result.resolved
    assert result.onset_offset_days is None


def test_vague_quantifiers_are_flagged_and_scored_lower(temporal):
    result = resolve(temporal, "Hypoglycaemia several days after the dose escalation.")
    assert result.onset_offset_days == 3
    assert result.source == "narrative_relative_vague"
    assert "vague" in result.detail
    precise = resolve(temporal, "Hypoglycaemia three days after the dose escalation.")
    assert result.confidence < precise.confidence


def test_anchor_alias_matching_prefers_the_longest_alias(temporal):
    assert temporal.match_anchor("dose escalation") == "dose_escalation"
    assert temporal.match_anchor("uptitration") == "dose_escalation"
    assert temporal.match_anchor("first dose") == "first_dose"
    assert temporal.match_anchor("a walk in the park") is None


def test_dose_increase_anchor_finds_each_escalation(resolver):
    hits = resolver.occurrences("S1", "dose_escalation")
    assert [h.date for h in hits] == [_dt.date(2024, 2, 1), _dt.date(2024, 3, 1)]
    assert "10->20" in hits[0].detail


def test_index_rule_changes_which_occurrence_counts(resolver):
    first = resolver.resolve("S1", "dose_escalation", index_rule="first_occurrence")
    recent = resolver.resolve(
        "S1", "dose_escalation", index_rule="most_recent_before_onset",
        onset_date=_dt.date(2024, 3, 5),
    )
    assert first.date == _dt.date(2024, 2, 1)
    assert recent.date == _dt.date(2024, 3, 1)


def test_first_record_anchor(resolver):
    assert resolver.resolve("S1", "first_dose").date == _dt.date(2024, 1, 1)


def test_missing_subject_yields_no_anchor(resolver):
    assert resolver.resolve("S2", "dose_escalation") is None
    assert resolver.occurrences("nobody", "dose_escalation") == []


@pytest.mark.parametrize(
    "value,expected",
    [
        ("2024-02-01", _dt.date(2024, 2, 1)),
        ("01-Feb-2024", _dt.date(2024, 2, 1)),
        ("", None),
        (None, None),
        ("not a date", None),
    ],
)
def test_date_parsing_tolerates_the_shapes_sdtm_uses(value, expected):
    assert parse_date(value) == expected
