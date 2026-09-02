"""The Attribute: a value, and the route that produced it."""

from __future__ import annotations

import datetime as _dt

import pytest

from aelayer.models import (
    AVAILABILITY_VALUES,
    METHODS,
    NOT_EVIDENCE_OF_ABSENCE,
    Attribute,
    Span,
)


def span(field: str = "location") -> Span:
    return Span(doc_id="AE:R1", start=0, end=5, field=field, extracted_value="CHEST",
                text="chest")


# -- the four invariants ---------------------------------------------------


def test_an_extracted_value_must_point_at_the_text_it_came_from():
    with pytest.raises(ValueError, match="must carry at least one span"):
        Attribute[str](
            value="CHEST", method="extracted", source="reported_term",
            availability="collected",
        )


def test_a_direct_value_must_come_from_a_standard_variable():
    """`direct` is a claim about the route, not a synonym for "confident"."""
    with pytest.raises(ValueError, match="means a standard structured variable"):
        Attribute[str](
            value="CHEST", method="direct", source="reported_term",
            availability="collected",
        )


@pytest.mark.parametrize(
    "availability", [a for a in AVAILABILITY_VALUES if a != "collected"]
)
def test_an_unavailable_attribute_never_carries_a_value(availability):
    with pytest.raises(ValueError, match="only a collected attribute has one"):
        Attribute[str](value="CHEST", availability=availability)


def test_a_collected_attribute_must_say_what_was_collected():
    with pytest.raises(ValueError, match="say which kind of empty"):
        Attribute[str](value=None, availability="collected")


def test_only_the_model_path_stamps_an_extractor_version():
    body = Attribute[str].direct("CHEST", "AELOC", [span()]).model_dump()
    with pytest.raises(ValueError, match="only the model path stamps it"):
        Attribute[str].model_validate({**body, "extractor_version": "extract-3"})


# -- there is no "inferred" ------------------------------------------------


def test_inferred_is_not_a_method():
    assert "inferred" not in METHODS
    with pytest.raises(ValueError):
        Attribute[str](
            value="CHEST", method="inferred", source="derived",
            availability="collected",
        )


def test_the_word_inferred_appears_nowhere_in_the_package():
    """A value the system worked out for itself is not an attribute of a patient."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "aelayer"
    offending = []
    for path in root.rglob("*.py"):
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if re.search(r"[\"']inferred[\"']", line):
                offending.append(f"{path.name}:{number}")
    assert not offending, offending


# -- what an availability means --------------------------------------------


def test_only_a_collected_value_is_evidence_of_absence():
    assert Attribute[str].direct("CHEST", "AELOC", [span()]).is_evidence_of_absence
    for availability in NOT_EVIDENCE_OF_ABSENCE:
        assert not Attribute[str].unavailable(availability).is_evidence_of_absence


def test_an_unavailable_attribute_cannot_claim_to_be_collected():
    with pytest.raises(ValueError, match="cannot be 'collected'"):
        Attribute[str].unavailable("collected")


def test_the_structured_availability_survives_a_text_recovery():
    """Recovering a site from prose does not make the CRF column collected."""
    recovered = Attribute[str].extracted(
        "CHEST", "AETERM", [span()], prior_availability="not_collected_by_protocol",
    )
    assert recovered.availability == "collected"
    assert recovered.structured_availability == "not_collected_by_protocol"
    assert recovered.from_text


# -- constructors ----------------------------------------------------------


def test_each_route_produces_the_method_it_claims():
    assert Attribute[str].direct("CHEST", "AELOC", [span()]).method == "direct"
    assert Attribute[str].normalized("CHEST", "SUPPAE.RASHSITE").method == "normalized"
    assert Attribute[str].extracted("CHEST", "AETERM", [span()]).method == "extracted"
    assert Attribute[str].unavailable("unknown").method is None


def test_a_route_reads_back_as_a_sentence():
    attribute = Attribute[str].direct("CHEST", "AELOC", [span()])
    assert attribute.describe_route() == "'CHEST' via direct from AELOC"
    assert Attribute[str].unavailable(
        "not_collected_by_protocol"
    ).describe_route() == "not_collected_by_protocol"


def test_a_populated_attribute_without_provenance_is_a_defect():
    assert Attribute[str].normalized("CHEST", "SUPPAE.X").has_provenance() is False
    assert Attribute[str].unavailable("unknown").has_provenance() is True


def test_a_date_attribute_round_trips():
    attribute = Attribute[_dt.date].direct(
        _dt.date(2024, 2, 6), "AESTDTC", [span("onset")]
    )
    assert attribute.value == _dt.date(2024, 2, 6)
    restored = Attribute[_dt.date].model_validate_json(attribute.model_dump_json())
    assert restored.value == attribute.value
