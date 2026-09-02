"""The boundary between the deterministic path and the model path, in code.

The rule: **a value the study already settled is never a question for a model.**
Not a convention, not a code review item — a function every request passes
through, which raises rather than degrading.

Two things follow from it, and both are tested over the whole corpus:

- no attribute whose source is a structured variable ever reaches a backend
- a request carries text and the names of unresolved attributes, and nothing
  else that could leak a controlled value into a prompt
"""

from __future__ import annotations

from typing import Any

from .catalog import ExtractionConfig
from .models import STRUCTURED_SOURCES, CanonicalAERecord
from .profiles import StudyProfile


class BoundaryViolation(RuntimeError):
    """Raised when the model path is asked about something already settled."""


#: Availabilities that mean the question is closed. `not_collected_by_protocol`
#: is deliberately not here: a study that never asked may still have written the
#: answer in prose, and that is the whole point of the layer.
SETTLED = frozenset({"collected", "not_applicable_gated", "not_representable"})


def askable_attributes(
    record: CanonicalAERecord, profile: StudyProfile, config: ExtractionConfig
) -> tuple[str, ...]:
    """The attributes the model path may be asked about, for one record.

    An attribute is askable when the extraction config allows it, the record has
    not settled it, and the study keeps it somewhere a model can read.
    """
    askable: list[str] = []
    for attribute in config.extractable_attributes:
        if attribute == "quality":
            continue
        current = record.attribute(attribute)
        if current is None or current.availability in SETTLED:
            continue
        if current.populated:
            continue
        homes = profile.homes_for(attribute)
        if homes and not any(home.is_text for home in homes):
            # The study keeps this attribute somewhere structural, or nowhere.
            # Either way there is no prose to read it out of.
            continue
        askable.append(attribute)
    return tuple(askable)


def unresolved_attributes(record: CanonicalAERecord) -> tuple[str, ...]:
    return tuple(
        name for name, attribute in record.attributes().items()
        if attribute.availability not in SETTLED and not attribute.populated
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
    if request.source_kind in ("structured_standard", "structured_sponsor"):
        raise BoundaryViolation(
            f"request reads {request.source_kind!r}: a structured variable is "
            f"read by the deterministic path, never by a model"
        )
    for attribute in request.attributes:
        if attribute == "quality":
            continue
        current = record.attribute(attribute)
        if current is None:
            raise BoundaryViolation(
                f"request names {attribute!r}, which is not an attribute of "
                f"{record.source_record_id}"
            )
        if current.availability in SETTLED or current.populated:
            raise BoundaryViolation(
                f"{record.source_record_id}.{attribute} is already "
                f"{current.availability!r} via {current.method!r} from "
                f"{current.source_variable!r}; a settled value is not a "
                f"question for a model"
            )
        if current.source in STRUCTURED_SOURCES and current.populated:
            raise BoundaryViolation(
                f"{record.source_record_id}.{attribute} came from a structured "
                f"variable and must not reach the model path"
            )
