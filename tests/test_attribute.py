"""The two-field model, and the invariant the whole system rests on.

`assertion` is what the source said. `availability` is whether it said
anything. They are orthogonal, and merging them is the error that biases every
downstream estimate — so the model refuses to represent a merged state at all.
"""

from __future__ import annotations

import datetime as _dt

import pytest
from pydantic import ValidationError

from aelayer.models import (
    ASCERTAINED,
    ASSERTIONS,
    AVAILABILITIES,
    METHODS,
    SILENT,
    VERDICTS,
    Attribute,
    Span,
)

SPAN = Span(
    doc_id="AE:R1:AETERM", start=0, end=4, field="mucosal_involvement",
    extracted_value="ORAL", text="oral", kind="text",
)


# -- the orthogonality invariant --------------------------------------------


def test_silence_cannot_carry_an_assertion():
    """The error the spec names: a silent attribute asserting something."""
    for availability in AVAILABILITIES:
        if availability == "observed":
            continue
        for assertion in ASSERTIONS:
            with pytest.raises(ValidationError) as exc:
                Attribute[str](availability=availability, assertion=assertion)
            assert "orthogonal" in str(exc.value)


def test_observed_must_carry_an_assertion():
    """The mirror error: the source spoke, but nothing records what it said."""
    with pytest.raises(ValidationError) as exc:
        Attribute[str](availability="observed", assertion=None)
    assert "must carry an assertion" in str(exc.value)


def test_every_assertion_and_availability_combination_is_decided():
    """No combination is left implicitly legal. Each is allowed or refused."""
    allowed = 0
    for availability in AVAILABILITIES:
        for assertion in (*ASSERTIONS, None):
            try:
                Attribute[str](availability=availability, assertion=assertion)
                allowed += 1
            except ValidationError:
                continue
    # observed x 3 assertions, plus each non-observed availability with None.
    assert allowed == len(ASSERTIONS) + (len(AVAILABILITIES) - 1)


def test_documented_negative_is_not_silence():
    absent = Attribute[str].direct("absent", "AEMUCOS", [SPAN])
    silent = Attribute[str].silent_because("not_collected", variable="AEMUCOS")
    assert absent.documented_negative and not absent.silent
    assert silent.silent and not silent.documented_negative
    assert absent.availability != silent.availability
    assert absent.assertion == "absent" and silent.assertion is None


def test_silent_is_exactly_the_complement_of_observed():
    assert SILENT == frozenset(set(AVAILABILITIES) - {"observed"})


def test_a_value_requires_an_observation():
    with pytest.raises(ValidationError) as exc:
        Attribute[str](availability="not_collected", value="ORAL")
    assert "only an observed attribute has one" in str(exc.value)


# -- method invariants -------------------------------------------------------


def test_extracted_requires_a_span():
    with pytest.raises(ValidationError) as exc:
        Attribute[str](
            availability="observed", assertion="present", method="extracted",
            source="reported_term", evidence=[],
        )
    assert "at least one span" in str(exc.value)


def test_direct_means_a_structured_variable():
    with pytest.raises(ValidationError):
        Attribute[str](
            availability="observed", assertion="present", method="direct",
            source="reported_term",
        )


def test_derived_means_a_cross_domain_computation():
    with pytest.raises(ValidationError):
        Attribute[int](
            availability="observed", assertion="present", method="derived",
            source="structured_standard", value=3,
        )
    ok = Attribute[int].derived(3, "AE+EX", [SPAN])
    assert ok.method == "derived" and ok.source == "cross_domain"


def test_methods_are_exactly_three_and_none_is_called_inferred():
    assert METHODS == ("direct", "derived", "extracted")
    assert "inferred" not in METHODS


def test_silent_because_refuses_to_manufacture_an_observation():
    with pytest.raises(ValueError) as exc:
        Attribute[str].silent_because("observed")
    assert "state the assertion" in str(exc.value)


# -- reading -----------------------------------------------------------------


def test_route_description_names_the_variable():
    attribute = Attribute[str].direct("present", "AEMUCOS", [SPAN], value="ORAL")
    assert "AEMUCOS" in attribute.describe_route()
    assert "direct" in attribute.describe_route()


def test_prior_availability_keeps_both_facts():
    """Recovering a value from prose does not make the CRF column collected."""
    recovered = Attribute[str].extracted(
        "present", "AETERM", [SPAN], value="ORAL",
        prior_availability="not_collected",
    )
    assert recovered.availability == "observed"
    assert recovered.structured_availability == "not_collected"


def test_verdicts_and_the_ascertained_set():
    assert VERDICTS == ("case", "non_case", "review", "not_ascertainable")
    assert ASCERTAINED == frozenset({"case", "non_case"})
    assert "not_ascertainable" not in ASCERTAINED
    assert "review" not in ASCERTAINED


def test_date_attribute_is_generic():
    onset = Attribute[_dt.date].direct(
        "present", "AESTDTC", [SPAN], value=_dt.date(2022, 4, 1)
    )
    assert onset.value.year == 2022
