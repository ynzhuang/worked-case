"""The boundary between the deterministic path and the model path, in code.

The rule: **a value the study already settled is never a question for a model.**
Not a convention, not a code review item — a function every request passes
through, which raises rather than degrading.

The second rule is narrower and just as load-bearing. Three different things in
this system are called "normalization", and only one of them is a language
problem:

``language_variation``
    the same clinical fact written a dozen ways in prose. **This is the only
    mechanism a model is used for.**
``coded_concept_variation``
    several legitimate codings of one situation. Resolved by the concept set;
    the codes are preserved side by side and neither is overwritten.
``terminology_version_variation``
    the same concept coded under different dictionary versions. Reconciled by a
    mechanical 1:1 map, and what does not map is flagged for review.

The last two never reach a backend. That is enforced here, not documented and
hoped for, and it is tested over the whole corpus.
"""

from __future__ import annotations

from typing import Any

from .catalog import ExtractionConfig
from .models import STRUCTURED_SOURCES, CanonicalAERecord
from .profiles import StudyProfile


class BoundaryViolation(RuntimeError):
    """Raised when the model path is asked about something already settled."""


#: Availabilities that close the question. `not_collected` is deliberately not
#: here: a study that never asked for a variable may still have written the
#: answer in prose, and recovering it is the whole point of the layer.
SETTLED = frozenset({"observed", "not_applicable"})

#: The one normalization mechanism a model is used for.
MODEL_MECHANISM = "language_variation"

#: The mechanisms that are deterministic by construction. Naming one of these
#: in a model request is not a degraded mode; it is a bug, and it raises.
DETERMINISTIC_MECHANISMS = frozenset(
    {"coded_concept_variation", "terminology_version_variation"}
)


def askable_modifiers(
    record: CanonicalAERecord, profile: StudyProfile, config: ExtractionConfig
) -> tuple[str, ...]:
    """The modifiers the model path may be asked about, for one record.

    A modifier is askable when the extraction config allows it, the record has
    not settled it, and the study keeps it somewhere a model can read.
    """
    askable: list[str] = []
    for modifier in config.extractable_modifiers:
        current = record.attribute(modifier)
        if current is None or current.availability in SETTLED:
            continue
        homes = profile.homes_for(modifier)
        if not homes or not any(home.is_text for home in homes):
            # The study keeps this modifier somewhere structural, or nowhere.
            # Either way there is no prose to read it out of.
            continue
        if any(home.kind in config.readable_sources for home in homes if home.is_text):
            askable.append(modifier)
    return tuple(askable)


def unresolved_modifiers(record: CanonicalAERecord) -> tuple[str, ...]:
    return tuple(
        name for name, attribute in record.modifiers.items()
        if attribute.availability not in SETTLED
    )


def assert_model_path_permitted(
    request: Any, record: CanonicalAERecord, profile: StudyProfile
) -> None:
    """Refuse a request that would send a settled value to a model."""
    if not isinstance(getattr(request, "text", None), str):
        raise BoundaryViolation(
            "a model request carries text; anything else is a controlled value "
            "and does not belong in a prompt"
        )
    mechanism = getattr(request, "mechanism", None)
    if mechanism in DETERMINISTIC_MECHANISMS:
        raise BoundaryViolation(
            f"request declares mechanism {mechanism!r}, which is resolved by a "
            f"declared map or a concept set. A model never rewrites a coded "
            f"field, and never decides a dictionary version mapping"
        )
    if mechanism != MODEL_MECHANISM:
        raise BoundaryViolation(
            f"request declares mechanism {mechanism!r}; the model path exists "
            f"for {MODEL_MECHANISM!r} and nothing else"
        )
    if request.source_kind in STRUCTURED_SOURCES:
        raise BoundaryViolation(
            f"request reads {request.source_kind!r}: a structured variable is "
            f"read by the deterministic path, never by a model"
        )
    for modifier in request.modifiers:
        current = record.attribute(modifier)
        if current is None:
            raise BoundaryViolation(
                f"request names {modifier!r}, which is not an attribute of "
                f"{record.source_record_id}"
            )
        if current.availability in SETTLED:
            raise BoundaryViolation(
                f"{record.source_record_id}.{modifier} is already "
                f"{current.availability!r} via {current.method!r} from "
                f"{current.source_variable!r}; a settled value is not a "
                f"question for a model"
            )
        if current.source in STRUCTURED_SOURCES:
            raise BoundaryViolation(
                f"{record.source_record_id}.{modifier} came from a structured "
                f"variable and must not reach the model path"
            )


def assert_coded_field_untouched(
    before: CanonicalAERecord, after: CanonicalAERecord
) -> None:
    """A coded value is the same before and after the model path ran.

    Enrichment adds attributes; it never edits the coded term, its dictionary
    version, or the reconciliation outcome.
    """
    if (before.coded_event is None) != (after.coded_event is None):
        raise BoundaryViolation(
            f"{before.source_record_id}: the model path added or removed a "
            f"coded term, which it must never do"
        )
    if before.coded_event is None:
        return
    for field in ("code", "dictionary", "dictionary_version", "concept_id",
                  "reconciled_to", "reconciliation"):
        was = getattr(before.coded_event, field)
        now = getattr(after.coded_event, field)
        if was != now:
            raise BoundaryViolation(
                f"{before.source_record_id}: coded_event.{field} changed from "
                f"{was!r} to {now!r}. No model ever rewrites a coded field"
            )
