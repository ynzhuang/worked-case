"""Deterministic coercion of controlled values and dates.

Every function here is total and explicit: given a cell and the study's profile,
it returns a value and the availability that describes it. Nothing guesses, and
nothing returns a bare ``None`` a caller could mistake for "the site said no".
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

from ..models import (
    ACTION_TAKEN_VALUES,
    OUTCOME_VALUES,
    RELATEDNESS_VALUES,
    SEVERITY_VALUES,
    Availability,
)
from ..profiles import StudyProfile

#: Spellings that appear in real extracts for the same controlled concept.
#: Mapping them is deterministic; inferring beyond them is not attempted.
ALIASES: dict[str, dict[str, str]] = {
    "severity": {
        "grade 1": "mild", "1": "mild", "mild": "mild",
        "grade 2": "moderate", "2": "moderate", "moderate": "moderate",
        "grade 3": "severe", "3": "severe", "severe": "severe",
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
        "dose not changed": "dose_not_changed",
        "dose_not_changed": "dose_not_changed", "none": "dose_not_changed",
        "dose reduced": "dose_reduced", "dose_reduced": "dose_reduced",
        "drug interrupted": "drug_interrupted",
        "drug_interrupted": "drug_interrupted",
        "drug withdrawn": "drug_withdrawn", "drug_withdrawn": "drug_withdrawn",
        "not applicable": "not_applicable", "not_applicable": "not_applicable",
        "unknown": "unknown",
    },
    "outcome": {
        "recovered": "recovered", "resolved": "recovered",
        "recovering": "recovering", "resolving": "recovering",
        "not recovered": "not_recovered", "not_recovered": "not_recovered",
        "ongoing": "not_recovered",
        "recovered with sequelae": "recovered_with_sequelae",
        "recovered_with_sequelae": "recovered_with_sequelae",
        "fatal": "fatal", "death": "fatal", "unknown": "unknown",
    },
}

ALLOWED: dict[str, tuple[str, ...]] = {
    "severity": SEVERITY_VALUES,
    "relatedness": RELATEDNESS_VALUES,
    "action_taken": ACTION_TAKEN_VALUES,
    "outcome": OUTCOME_VALUES,
}

_TRUE = {"y", "yes", "true", "1"}
_FALSE = {"n", "no", "false", "0"}

#: Formats a study might write a date in. Partial dates are handled separately:
#: a month is a real fact, and not one that supports day arithmetic.
_DATE_FORMATS = ("%Y-%m-%d", "%d-%b-%Y", "%d/%m/%Y", "%Y-%m-%dT%H:%M:%S",
                 "%Y-%m-%dT%H:%M")


def is_blank(cell: Any) -> bool:
    return cell is None or (isinstance(cell, str) and not cell.strip())


def coerce_enum(
    attribute: str, cell: Any, profile: StudyProfile, variable: str,
) -> tuple[str | None, Availability, str]:
    """Map a controlled cell onto a canonical value.

    A blank asks the profile what the blank means; a value outside the codelist
    is reported as such rather than silently coerced to the nearest permissible
    code.
    """
    if is_blank(cell):
        return None, profile.availability_for_blank(variable), ""
    raw = str(cell).strip().lower().replace("-", " ")
    canonical = ALIASES.get(attribute, {}).get(raw)
    if canonical is None:
        squashed = raw.replace(" ", "_")
        if squashed in ALLOWED.get(attribute, ()):
            canonical = squashed
    if canonical is None:
        return None, "unknown", (
            f"value {cell!r} is not in the canonical codelist for {attribute}"
        )
    return canonical, "collected", ""


def coerce_bool(
    cell: Any, profile: StudyProfile, variable: str,
) -> tuple[bool | None, Availability, str]:
    if is_blank(cell):
        return None, profile.availability_for_blank(variable), ""
    raw = str(cell).strip().lower()
    if raw in _TRUE:
        return True, "collected", ""
    if raw in _FALSE:
        return False, "collected", ""
    return None, "unknown", f"value {cell!r} is not a recognised yes/no"


def coerce_date(
    cell: Any, profile: StudyProfile, variable: str,
) -> tuple[_dt.date | None, Availability, str]:
    """Parse a date written in whichever convention the study uses.

    A partial date is a real value, but not one that supports day-level
    arithmetic, so it is reported as ``unknown`` with a note rather than being
    silently rounded to the first of the month.
    """
    if is_blank(cell):
        return None, profile.availability_for_blank(variable), ""
    if isinstance(cell, _dt.datetime):
        return cell.date(), "collected", ""
    if isinstance(cell, _dt.date):
        return cell, "collected", ""
    text = str(cell).strip()
    for fmt in _DATE_FORMATS:
        try:
            return _dt.datetime.strptime(text, fmt).date(), "collected", ""
        except ValueError:
            continue
    if len(text) in (4, 7):
        return None, "unknown", (
            f"{variable} is a partial date ({text!r}); day-level arithmetic is "
            f"not supported by it"
        )
    return None, "unknown", f"{variable} value {cell!r} is not a parsable date"


def resolve_sponsor_value(
    profile: StudyProfile, rows: list[dict[str, Any]], attribute: str,
) -> tuple[str | None, Availability, str, str | None]:
    """Resolve a sponsor-defined supplemental qualifier.

    Returns ``(value, availability, note, variable)``. The mapping is declared
    in the profile; a code the mapping does not cover resolves to nothing,
    because guessing which catalogue value a sponsor meant is exactly the silent
    substitution this system exists to avoid.
    """
    name = profile.sponsor_variable_name
    if not name:
        return None, profile.availability_for_blank(attribute), "", None
    variable = f"SUPPAE.{name}"
    matching = [r for r in rows if str(r.get("QNAM") or "") == name]
    if not matching:
        return None, profile.availability_for_blank(variable), "", variable
    code = matching[0].get("QVAL")
    if is_blank(code):
        return None, "unknown", f"{variable} is present but empty", variable
    value = profile.resolve_sponsor_code(str(code))
    if value is None:
        return None, "not_representable", (
            f"{variable} holds {code!r}, which the declared mapping for "
            f"{profile.profile_id} does not cover; the value is left unresolved "
            f"rather than guessed"
        ), variable
    return value, "collected", (
        f"{variable}={code!r} resolved through the sponsor codelist"
    ), variable
