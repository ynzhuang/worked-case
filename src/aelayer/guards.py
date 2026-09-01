"""The boundary between the deterministic path and the model path.

Enforced here, in code, rather than left to convention.

The rule is narrow and absolute: **a field the deterministic path already
resolved is never put to a model.**  Asking a model to read severity out of a
narrative when ``AESEV`` is populated invites it to disagree with a controlled
value that a coder entered and a monitor checked, and there is no upside to
that trade.

The mechanism is a request type that can only carry text, plus a check that the
fields it asks about are genuinely unresolved on the record it came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from typing import Any, Iterable

from .models import CanonicalAERecord


class ControlledValueLeak(RuntimeError):
    """Raised when the model path is asked about an already-controlled value."""


#: Fields the deterministic path owns. A model may be asked about one only when
#: the record shows it unresolved, and never otherwise.
DETERMINISTIC_FIELDS: frozenset[str] = frozenset({
    "coded_term", "dictionary_version", "onset_datetime", "end_datetime",
    "severity", "seriousness", "seriousness_criteria", "relatedness",
    "action_taken", "outcome",
})

#: Fields no CRF column carries, so the model path is their only source.
TEXT_ONLY_FIELDS: frozenset[str] = frozenset(
    {"symptoms", "symptoms_assessed", "assertion", "labs"}
)

#: Derivable from either path. The concept comes from the coded term where that
#: term denotes it, and from the verbatim and narrative where it does not — a
#: study that never coded the event, and an event coded to a non-specific term
#: like "Malaise", are both cases where the coded value is preserved untouched
#: and the concept has to come from somewhere else. The model is asked only
#: when the deterministic path produced no concept at all.
DUAL_SOURCE_FIELDS: tuple[str, ...] = ("standardized_concept",)

#: Collection states that settle a field rather than leave it open. A parent
#: gate answered No is a fact the study recorded, not a gap for a model to
#: fill: the criterion was never applicable, and asking anyway invites the
#: model to contradict the CRF's own logic.
SETTLED_STATES: frozenset[str] = frozenset({"collected", "not_applicable_gated"})


@dataclass(frozen=True)
class ModelRequest:
    """The only thing the model path is ever handed.

    It carries text and a list of fields to look for.  It structurally cannot
    carry a controlled value: there is nowhere to put one.
    """

    doc_id: str
    text: str
    requested_fields: tuple[str, ...]
    schema_name: str
    prompt_version: str
    record_id: str = ""

    def __post_init__(self) -> None:
        unknown = sorted(
            set(self.requested_fields)
            - DETERMINISTIC_FIELDS
            - TEXT_ONLY_FIELDS
            - set(DUAL_SOURCE_FIELDS)
        )
        if unknown:
            raise ControlledValueLeak(
                f"model request asks for fields with no defined source: {unknown}"
            )


def unresolved_fields(record: CanonicalAERecord) -> tuple[str, ...]:
    """Fields the model path may legitimately be asked about for this record.

    Text-only fields always; deterministic fields only where the record shows
    them unresolved.  A field whose blank means the study never collected it is
    still unresolved — the CRF's silence is not the patient's.
    """
    askable: list[str] = sorted(TEXT_ONLY_FIELDS)
    for name in sorted(DETERMINISTIC_FIELDS):
        field = record.fields().get(name)
        if field is None:
            continue
        if field.collection_state not in SETTLED_STATES:
            askable.append(name)

    if record.standardized_concept is None:
        askable.append("standardized_concept")
    return tuple(sorted(set(askable)))


def assert_model_path_permitted(
    request: ModelRequest, record: CanonicalAERecord
) -> None:
    """Raise unless every requested field is genuinely unresolved.

    This is the check the definition-of-done requires: no controlled value is
    ever sent to the model path.
    """
    fields = record.fields()
    leaked = sorted(
        name for name in request.requested_fields
        if name in DETERMINISTIC_FIELDS
        and (field := fields.get(name)) is not None
        and field.collection_state in SETTLED_STATES
    )
    if (
        "standardized_concept" in request.requested_fields
        and record.standardized_concept is not None
    ):
        leaked.append("standardized_concept")
    if leaked:
        raise ControlledValueLeak(
            f"record {record.source_record_id} has {leaked} already settled by "
            f"the deterministic path; the model path must not be asked about "
            f"them. A model disagreeing with a coded value, or filling a field "
            f"a gate ruled inapplicable, is a defect rather than a signal."
        )


def assert_no_structured_payload(payload: Any, *, where: str) -> None:
    """Raise if anything but a ModelRequest is about to reach a backend."""
    if not isinstance(payload, ModelRequest):
        raise ControlledValueLeak(
            f"{where}: the model path accepts only a ModelRequest, got "
            f"{type(payload).__name__}. Structured records must not be passed "
            f"to a model."
        )
