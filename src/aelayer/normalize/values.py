"""Deterministic coercion of controlled values and dates.

Every function is total and explicit: given a cell and the study's profile it
returns a value, the assertion the source makes, and the availability that
describes it. Nothing guesses, and nothing returns a bare ``None`` a caller
could mistake for "the site said no".
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

from ..models import (
    ACTION_VALUES,
    OUTCOME_VALUES,
    RELATEDNESS_VALUES,
    SEVERITY_VALUES,
    Assertion,
    Availability,
)
from ..profiles import StudyProfile

ALIASES: dict[str, dict[str, str]] = {
    "severity": {
        "grade 1": "mild", "1": "mild", "mild": "mild",
        "grade 2": "moderate", "2": "moderate", "moderate": "moderate",
        "grade 3": "severe", "3": "severe", "severe": "severe",
    },
    "relatedness": {
        "not related": "not_related", "unrelated": "not_related",
        "not_related": "not_related", "unlikely": "unlikely",
        "possible": "possible", "possibly related": "possible",
        "probable": "probable", "definite": "definite", "unknown": "unknown",
    },
    "action": {
        "dose not changed": "dose_not_changed",
        "dose_not_changed": "dose_not_changed", "none": "dose_not_changed",
        "dose reduced": "dose_reduced", "dose_reduced": "dose_reduced",
        "drug interrupted": "drug_interrupted",
        "drug_interrupted": "drug_interrupted",
        "drug withdrawn": "drug_withdrawn", "drug_withdrawn": "drug_withdrawn",
        "not applicable": "not_applicable", "unknown": "unknown",
    },
    "outcome": {
        "recovered": "recovered", "resolved": "recovered",
        "recovering": "recovering", "not recovered": "not_recovered",
        "not_recovered": "not_recovered", "ongoing": "not_recovered",
        "recovered_with_sequelae": "recovered_with_sequelae",
        "fatal": "fatal", "unknown": "unknown",
    },
}

ALLOWED: dict[str, tuple[str, ...]] = {
    "severity": SEVERITY_VALUES,
    "relatedness": RELATEDNESS_VALUES,
    "action": ACTION_VALUES,
    "outcome": OUTCOME_VALUES,
}

#: How a study writes a three-state qualifier. "U" is the source hedging, which
#: is an assertion of uncertainty rather than an absence of one.
TRISTATE: dict[str, Assertion] = {
    "y": "present", "yes": "present", "true": "present", "1": "present",
    "n": "absent", "no": "absent", "false": "absent", "0": "absent",
    "u": "uncertain", "unk": "uncertain", "unknown": "uncertain",
}

_DATE_FORMATS = ("%Y-%m-%d", "%d-%b-%Y", "%d/%m/%Y", "%Y-%m-%dT%H:%M:%S",
                 "%Y-%m-%dT%H:%M")


def is_blank(cell: Any) -> bool:
    return cell is None or (isinstance(cell, str) and not cell.strip())


def coerce_enum(
    attribute: str, cell: Any, profile: StudyProfile, variable: str,
) -> tuple[str | None, Availability, str]:
    """Map a controlled cell onto a canonical value.

    A silence asks the profile what it means; a value outside the codelist is
    reported as such rather than coerced to the nearest permissible code.
    """
    if is_blank(cell):
        return None, profile.availability_for_silence(variable), ""
    raw = str(cell).strip().lower().replace("-", " ")
    canonical = ALIASES.get(attribute, {}).get(raw)
    if canonical is None:
        squashed = raw.replace(" ", "_")
        if squashed in ALLOWED.get(attribute, ()):
            canonical = squashed
    if canonical is None:
        return None, "unresolved", (
            f"value {cell!r} is not in the canonical codelist for {attribute}"
        )
    return canonical, "observed", ""


def coerce_tristate(
    cell: Any, profile: StudyProfile, variable: str,
) -> tuple[Assertion | None, Availability, str]:
    """Read a three-state structured qualifier.

    Returns an *assertion*, never a bare boolean: "N" means the source looked
    and found nothing, which is a different fact from the cell being empty.
    """
    if is_blank(cell):
        return None, profile.availability_for_silence(variable), ""
    raw = str(cell).strip().lower()
    assertion = TRISTATE.get(raw)
    if assertion is None:
        return None, "unresolved", (
            f"{variable} holds {cell!r}, which is not a recognised yes/no/unknown"
        )
    return assertion, "observed", ""


def coerce_bool(
    cell: Any, profile: StudyProfile, variable: str,
) -> tuple[bool | None, Availability, str]:
    assertion, availability, note = coerce_tristate(cell, profile, variable)
    if assertion is None or assertion == "uncertain":
        return None, "unresolved" if assertion == "uncertain" else availability, note
    return assertion == "present", availability, note


def coerce_int(
    cell: Any, profile: StudyProfile, variable: str,
) -> tuple[int | None, Availability, str]:
    if is_blank(cell):
        return None, profile.availability_for_silence(variable), ""
    try:
        return int(float(str(cell).strip())), "observed", ""
    except (TypeError, ValueError):
        return None, "unresolved", f"{variable} value {cell!r} is not a number"


def coerce_date(
    cell: Any, profile: StudyProfile, variable: str,
) -> tuple[_dt.date | None, Availability, str]:
    """Parse a date written in whichever convention the study uses.

    A partial date is a real value, but not one that supports day-level
    arithmetic, so it is reported as unresolved with a note rather than being
    silently rounded to the first of the month.
    """
    if is_blank(cell):
        return None, profile.availability_for_silence(variable), ""
    if isinstance(cell, _dt.datetime):
        return cell.date(), "observed", ""
    if isinstance(cell, _dt.date):
        return cell, "observed", ""
    text = str(cell).strip()
    for fmt in _DATE_FORMATS:
        try:
            return _dt.datetime.strptime(text, fmt).date(), "observed", ""
        except ValueError:
            continue
    if len(text) in (4, 7):
        return None, "unresolved", (
            f"{variable} is a partial date ({text!r}); day-level arithmetic is "
            f"not supported by it"
        )
    return None, "unresolved", f"{variable} value {cell!r} is not a parsable date"
