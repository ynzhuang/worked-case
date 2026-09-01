"""Deterministic coercion of controlled values, dates and units.

Every function here is total and explicit: given a cell and the study's
conventions, it returns a value and the collection state that describes it.
Nothing guesses, and nothing returns a bare ``None`` that a caller could mistake
for "the site said no".
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

from ..models import (
    ACTION_TAKEN_VALUES,
    OUTCOME_VALUES,
    RELATEDNESS_VALUES,
    SEVERITY_VALUES,
    CollectionState,
)
from ..semantics import Codelist, StudySemantics

#: Spellings that appear in real extracts for the same controlled concept.
#: Mapping them is deterministic; inferring beyond them is not attempted.
_ALIASES: dict[str, dict[str, str]] = {
    "severity": {
        "grade 1": "mild", "grade_1": "mild", "1": "mild", "mild": "mild",
        "grade 2": "moderate", "grade_2": "moderate", "2": "moderate",
        "moderate": "moderate",
        "grade 3": "severe", "grade_3": "severe", "3": "severe", "severe": "severe",
    },
    "relatedness": {
        "not related": "not_related", "unrelated": "not_related",
        "not_related": "not_related", "unlikely": "unlikely",
        "unlikely related": "unlikely", "possible": "possible",
        "possibly related": "possible", "related": "possible",
        "probable": "probable", "probably related": "probable",
        "definite": "definite", "definitely related": "definite",
        "unknown": "unknown", "not assessable": "unknown",
    },
    "action_taken": {
        "dose not changed": "dose_not_changed", "dose_not_changed": "dose_not_changed",
        "none": "dose_not_changed", "no change": "dose_not_changed",
        "dose reduced": "dose_reduced", "dose_reduced": "dose_reduced",
        "drug interrupted": "drug_interrupted", "drug_interrupted": "drug_interrupted",
        "dose interrupted": "drug_interrupted",
        "drug withdrawn": "drug_withdrawn", "drug_withdrawn": "drug_withdrawn",
        "not applicable": "not_applicable", "not_applicable": "not_applicable",
        "unknown": "unknown",
    },
    "outcome": {
        "recovered": "recovered", "resolved": "recovered",
        "recovered/resolved": "recovered",
        "recovering": "recovering", "resolving": "recovering",
        "not recovered": "not_recovered", "not_recovered": "not_recovered",
        "not resolved": "not_recovered", "ongoing": "not_recovered",
        "recovered with sequelae": "recovered_with_sequelae",
        "recovered_with_sequelae": "recovered_with_sequelae",
        "fatal": "fatal", "death": "fatal",
        "unknown": "unknown",
    },
}

_ALLOWED: dict[str, tuple[str, ...]] = {
    "severity": SEVERITY_VALUES,
    "relatedness": RELATEDNESS_VALUES,
    "action_taken": ACTION_TAKEN_VALUES,
    "outcome": OUTCOME_VALUES,
}

_BOOLEAN_TRUE = {"y", "yes", "true", "1"}
_BOOLEAN_FALSE = {"n", "no", "false", "0"}


def is_blank(cell: Any) -> bool:
    return cell is None or (isinstance(cell, str) and not cell.strip())


def coerce_enum(
    field: str, cell: Any, study: StudySemantics
) -> tuple[str | None, CollectionState, str]:
    """Map a controlled cell onto a canonical value.

    Returns ``(value, collection_state, note)``.  A blank asks the study what
    the blank means; a value outside the codelist is reported as such rather
    than silently coerced to the nearest permissible code.
    """
    if is_blank(cell):
        return None, study.state_for_blank(field), ""

    raw = str(cell).strip().lower().replace("-", " ")
    canonical = _ALIASES.get(field, {}).get(raw)
    if canonical is None:
        squashed = raw.replace(" ", "_")
        if squashed in _ALLOWED.get(field, ()):
            canonical = squashed
    if canonical is None:
        return None, "unknown", (
            f"value {cell!r} is not in the canonical codelist for {field}"
        )

    codelist: Codelist | None = study.codelist_for(field)
    if codelist is not None and canonical not in codelist.permissible:
        # The study recorded something its own codelist does not permit. Report
        # it rather than accept it: the discrepancy is a data question.
        return None, "unknown", (
            f"value {canonical!r} is not permissible for {field} in "
            f"{study.study_id}; permissible: {list(codelist.permissible)}"
        )
    return canonical, "collected", ""


def unresolved_concept(
    field: str, concept: str | None, study: StudySemantics
) -> tuple[CollectionState, str]:
    """The state for a field whose intended concept the codelist cannot express.

    ``not_representable`` is a statement about the field, not an inference
    about what the site did instead.  Substituting the nearest permissible
    code would assert something stronger than the evidence supports.
    """
    codelist = study.codelist_for(field)
    if codelist is None or concept is None:
        return study.state_for_blank(field), ""
    if concept in codelist.absent_concepts:
        return "not_representable", (
            f"{study.study_id} has no permissible {field} value for {concept!r}; "
            f"the field is left unresolved rather than coerced"
        )
    return study.state_for_blank(field), ""


def coerce_bool(
    field: str, cell: Any, study: StudySemantics
) -> tuple[bool | None, CollectionState, str]:
    if is_blank(cell):
        return None, study.state_for_blank(field), ""
    raw = str(cell).strip().lower()
    if raw in _BOOLEAN_TRUE:
        return True, "collected", ""
    if raw in _BOOLEAN_FALSE:
        return False, "collected", ""
    return None, "unknown", f"value {cell!r} is not a recognised yes/no for {field}"


def coerce_datetime(
    field: str, cell: Any, study: StudySemantics
) -> tuple[_dt.datetime | None, CollectionState, str]:
    """Parse an ISO-8601 or partial SDTM date.

    A partial date (year, or year-month) is a real value, but not one that
    supports day-level arithmetic, so it is reported as ``unknown`` with a note
    rather than silently rounded to the first of the month.
    """
    if is_blank(cell):
        return None, study.state_for_blank(field), ""
    text = str(cell).strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return _dt.datetime.strptime(text, fmt), "collected", ""
        except ValueError:
            continue
    if len(text) in (4, 7):
        return None, "unknown", (
            f"{field} is a partial date ({text!r}); day-level arithmetic is not "
            f"supported by it"
        )
    return None, "unknown", f"{field} value {cell!r} is not a parsable date"


def to_canonical_unit(
    value: float, unit: str, conversions: dict[str, float]
) -> float | None:
    """Convert a reported value into the catalogue's canonical unit.

    Not decoration: a threshold applied to an unconverted mmol/L value
    misclassifies an entire study in silence.
    """
    factor = conversions.get(unit)
    if factor is None:
        for known, known_factor in conversions.items():
            if known.lower() == unit.lower():
                factor = known_factor
                break
    return None if factor is None else round(value * factor, 4)
