"""The collection-semantics resolver.

This module supplies no clinical values.  It answers one question for every
field the other two paths read: *given this study's collection conventions, what
does this cell mean?*

Without it, every empty cell collapses to ``unknown`` and four distinct facts
become one:

* the CRF never asked (``not_collected_by_protocol``)
* a parent gate was answered No, so the child was never applicable
  (``not_applicable_gated``)
* the event has not ended, so the field cannot be filled yet (``pending_ongoing``)
* the concept exists but the study's codelist has no permissible value for it
  (``not_representable``)

Only the first meaning of "empty" that a study actually collected can ever
support an inference about the patient, and none of these are it.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from pathlib import Path
from typing import Any

from .models import COLLECTION_STATES, CollectionState


class SemanticsError(ValueError):
    """Raised when the collection-semantics config is structurally invalid."""


@dataclass(frozen=True)
class GateSpec:
    """A field whose meaning depends on a parent question."""

    field: str
    gate: str
    when_gate_false: CollectionState

    def resolve(self, gate_value: Any) -> CollectionState | None:
        """The state this field takes given the gate's answer, or None."""
        if gate_value is False:
            return self.when_gate_false
        return None


@dataclass(frozen=True)
class GateValueSpec:
    """How to read a gate's answer off a record."""

    gate: str
    from_field: str
    true_when_value: Any = None
    true_when_in: tuple[str, ...] = ()

    def evaluate(self, record_values: dict[str, Any]) -> bool | None:
        """The gate's answer, or None when the gating field itself is empty."""
        raw = record_values.get(self.from_field)
        if raw in (None, ""):
            return None
        if self.true_when_in:
            return str(raw) in self.true_when_in
        if self.true_when_value is not None:
            return bool(raw) is bool(self.true_when_value)
        return bool(raw)


@dataclass(frozen=True)
class Codelist:
    """A field's permissible values, and the concepts it cannot express."""

    field: str
    permissible: tuple[str, ...]
    absent_concepts: tuple[str, ...]

    def resolve(self, concept: str | None) -> tuple[str | None, CollectionState]:
        """Map an intended concept onto a permissible value.

        A concept the codelist cannot express yields ``not_representable``
        rather than the nearest available code.  Substituting
        ``drug_interrupted`` for a dose reduction would assert something the
        evidence does not support; leaving the field unresolved and saying why
        is the weaker and correct claim.
        """
        if concept is None:
            return None, "unknown"
        if concept in self.permissible:
            return concept, "collected"
        if concept in self.absent_concepts:
            return None, "not_representable"
        return None, "not_representable"


@dataclass(frozen=True)
class StudySemantics:
    """Everything about how one study collected its data."""

    study_id: str
    label: str = ""
    representation: str = ""
    dictionary: str = "MedDRA"
    dictionary_version: str = ""
    glucose_unit: str = "mg/dL"
    record_splitting: str = "none"
    narrative_detail: str = "standard"
    linked_forms: tuple[str, ...] = ()
    collected_fields: tuple[str, ...] = ()
    blank_means: dict[str, CollectionState] = _dc_field(default_factory=dict)
    gates: dict[str, GateSpec] = _dc_field(default_factory=dict)
    codelists: dict[str, Codelist] = _dc_field(default_factory=dict)
    gate_values: dict[str, GateValueSpec] = _dc_field(default_factory=dict)
    default_blank: CollectionState = "unknown"
    note: str = ""

    # -- the one question this class exists to answer ----------------------

    def state_for_blank(self, field: str) -> CollectionState:
        """What an empty ``field`` means in this study.

        Order of precedence: an explicit per-field declaration, then whether
        the field appears in the study's collected set at all, then the
        default.
        """
        if field in self.blank_means:
            return self.blank_means[field]
        if self.collected_fields and field not in self.collected_fields:
            return "not_collected_by_protocol"
        return self.default_blank

    #: Declared blank meanings that mean the study never puts a value here.
    _NOT_COLLECTED = ("not_collected_by_protocol", "intentionally_blank")

    def collects(self, field: str) -> bool:
        """Does this study ever put a value in this field?

        False both where the CRF lacks the column and where the protocol
        instructs the site to leave it blank. The two are different facts and
        stay distinguishable through `state_for_blank`; what they share is that
        no value will ever appear.
        """
        if field in self.blank_means:
            return self.blank_means[field] not in self._NOT_COLLECTED
        return (not self.collected_fields) or field in self.collected_fields

    def gate_for(self, field: str) -> GateSpec | None:
        return self.gates.get(field)

    def gate_answer(self, gate: str, record_values: dict[str, Any]) -> bool | None:
        spec = self.gate_values.get(gate)
        return spec.evaluate(record_values) if spec else None

    def codelist_for(self, field: str) -> Codelist | None:
        return self.codelists.get(field)

    def splits_on_severity_change(self) -> bool:
        return self.record_splitting == "on_severity_change"


class CollectionSemantics:
    """Read-only view over ``collection_semantics.yaml``."""

    def __init__(self, raw: dict[str, Any], source_path: Path | None = None):
        if not isinstance(raw, dict) or "studies" not in raw:
            raise SemanticsError("collection_semantics.yaml must define `studies`")
        self.raw = raw
        self.source_path = source_path
        defaults = raw.get("defaults") or {}
        self.linked_forms: dict[str, Any] = raw.get("linked_forms") or {}

        default_gates = _parse_gates(defaults.get("gated_fields") or {})
        default_gate_values = _parse_gate_values(defaults.get("gate_values") or {})
        default_codelists = _parse_codelists(defaults)
        default_blank = _check_state(
            defaults.get("blank_means", "unknown"), "defaults.blank_means"
        )
        default_dictionary = defaults.get("dictionary", "MedDRA")

        self.studies: dict[str, StudySemantics] = {}
        for study_id, body in (raw["studies"] or {}).items():
            if not isinstance(body, dict):
                raise SemanticsError(f"study {study_id!r} must be a mapping")
            blank_means_raw = body.get("blank_means") or {}
            if isinstance(blank_means_raw, str):
                study_default = _check_state(
                    blank_means_raw, f"{study_id}.blank_means"
                )
                per_field: dict[str, CollectionState] = {}
            else:
                study_default = default_blank
                per_field = {
                    name: _check_state(state, f"{study_id}.blank_means.{name}")
                    for name, state in blank_means_raw.items()
                }
            unknown_forms = [
                f for f in (body.get("linked_forms") or []) if f not in self.linked_forms
            ]
            if unknown_forms:
                raise SemanticsError(
                    f"study {study_id!r} references undefined linked forms "
                    f"{unknown_forms}; known: {sorted(self.linked_forms)}"
                )
            self.studies[study_id] = StudySemantics(
                study_id=study_id,
                label=body.get("label", study_id),
                representation=body.get("representation", ""),
                dictionary=body.get("dictionary", default_dictionary),
                dictionary_version=body.get("dictionary_version", ""),
                glucose_unit=body.get("glucose_unit", "mg/dL"),
                record_splitting=body.get("record_splitting", "none"),
                narrative_detail=body.get("narrative_detail", "standard"),
                linked_forms=tuple(body.get("linked_forms") or []),
                collected_fields=tuple(body.get("collected_fields") or []),
                blank_means=per_field,
                gates={**default_gates, **_parse_gates(body.get("gated_fields") or {})},
                gate_values={
                    **default_gate_values,
                    **_parse_gate_values(body.get("gate_values") or {}),
                },
                codelists={**default_codelists, **_parse_codelists(body)},
                default_blank=study_default,
                note=body.get("note", "").strip(),
            )

    def for_study(self, study_id: str) -> StudySemantics:
        try:
            return self.studies[study_id]
        except KeyError:
            raise SemanticsError(
                f"no collection semantics for study {study_id!r}; known: "
                f"{sorted(self.studies)}. A study with no declared semantics "
                f"cannot be read: every blank in it would be guesswork."
            ) from None

    def study_ids(self) -> list[str]:
        return sorted(self.studies)

    def dictionary_versions(self) -> dict[str, str]:
        return {s.study_id: s.dictionary_version for s in self.studies.values()}

    def form(self, form_id: str) -> dict[str, Any]:
        try:
            return self.linked_forms[form_id]
        except KeyError:
            raise SemanticsError(f"unknown linked form {form_id!r}") from None


# --------------------------------------------------------------------------


def _check_state(value: Any, where: str) -> CollectionState:
    if value not in COLLECTION_STATES:
        raise SemanticsError(
            f"{where}: {value!r} is not a collection state; "
            f"known: {list(COLLECTION_STATES)}"
        )
    return value  # type: ignore[return-value]


def _parse_gates(body: dict[str, Any]) -> dict[str, GateSpec]:
    gates: dict[str, GateSpec] = {}
    for field_name, spec in body.items():
        if not isinstance(spec, dict) or "gate" not in spec:
            raise SemanticsError(f"gated field {field_name!r} needs a `gate`")
        gates[field_name] = GateSpec(
            field=field_name,
            gate=spec["gate"],
            when_gate_false=_check_state(
                spec.get("when_gate_false", "not_applicable_gated"),
                f"gated_fields.{field_name}.when_gate_false",
            ),
        )
    return gates


def _parse_gate_values(body: dict[str, Any]) -> dict[str, GateValueSpec]:
    specs: dict[str, GateValueSpec] = {}
    for gate, spec in body.items():
        if not isinstance(spec, dict) or "from_field" not in spec:
            raise SemanticsError(f"gate_values.{gate} needs `from_field`")
        specs[gate] = GateValueSpec(
            gate=gate,
            from_field=spec["from_field"],
            true_when_value=spec.get("true_when_value"),
            true_when_in=tuple(spec.get("true_when_in") or []),
        )
    return specs


def _parse_codelists(body: dict[str, Any]) -> dict[str, Codelist]:
    codelists: dict[str, Codelist] = {}
    for field_name, spec in (body.get("restricted_codelists") or {}).items():
        codelists[field_name] = _codelist(field_name, spec)
    return codelists


def _codelist(field_name: str, spec: dict[str, Any]) -> Codelist:
    if "permissible" not in spec:
        raise SemanticsError(f"codelist for {field_name!r} needs `permissible`")
    return Codelist(
        field=field_name,
        permissible=tuple(spec["permissible"]),
        absent_concepts=tuple(spec.get("absent_concepts") or []),
    )
