"""Study profiles: where each attribute lives, and what a blank means.

A profile is a collection decision. It is not a property of the patients, and
nothing about it should be inferred at read time — a study whose profile is not
declared cannot be read at all, because every blank in it would be guesswork.

The generator renders a clinical truth *through* a profile; the normalizer reads
a record *through* the same profile. One file, so the two cannot disagree.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from pathlib import Path
from typing import Any

from .models import AVAILABILITY_VALUES, SOURCE_KINDS, Availability, SourceKind


class ProfileError(ValueError):
    """Raised when the profile config is structurally invalid."""


@dataclass(frozen=True)
class AttributeHome:
    """One place an attribute can live."""

    home_id: str
    kind: SourceKind | None
    variable: str | None
    description: str = ""

    @property
    def is_nowhere(self) -> bool:
        return self.kind is None or self.variable is None

    @property
    def is_structured(self) -> bool:
        return self.kind in ("structured_standard", "structured_sponsor")

    @property
    def is_text(self) -> bool:
        return self.kind in ("reported_term", "comment")


@dataclass(frozen=True)
class GateSpec:
    variable: str
    gate: str
    when_gate_false: Availability

    def resolve(self, gate_value: Any) -> Availability | None:
        return self.when_gate_false if gate_value is False else None


@dataclass(frozen=True)
class GateValueSpec:
    gate: str
    from_variable: str
    true_when_in: tuple[str, ...] = ()

    def evaluate(self, row: dict[str, Any]) -> bool | None:
        raw = row.get(self.from_variable)
        if raw in (None, ""):
            return None
        return str(raw) in self.true_when_in if self.true_when_in else bool(raw)


@dataclass(frozen=True)
class StudyProfile:
    """One study's collection conventions."""

    profile_id: str
    study_id: str
    label: str = ""
    reported_term_style: str = "terse"
    location_homes: tuple[AttributeHome, ...] = ()
    pattern_homes: tuple[AttributeHome, ...] = ()
    sponsor_variable_name: str | None = None
    sponsor_codelist: dict[str, str] = _dc_field(default_factory=dict)
    dictionary: str = ""
    dictionary_version: str = ""
    collects: tuple[str, ...] = ()
    default_blank: Availability = "unknown"
    gates: dict[str, GateSpec] = _dc_field(default_factory=dict)
    gate_values: dict[str, GateValueSpec] = _dc_field(default_factory=dict)
    conventions: dict[str, Any] = _dc_field(default_factory=dict)
    note: str = ""

    # -- where an attribute lives ------------------------------------------

    def homes_for(self, attribute: str) -> tuple[AttributeHome, ...]:
        if attribute == "location":
            return self.location_homes
        if attribute == "pattern":
            return self.pattern_homes
        return ()

    def home_kinds(self, attribute: str) -> list[str]:
        return [h.home_id for h in self.homes_for(attribute)]

    def collects_attribute(self, attribute: str) -> bool:
        """Does this study record the attribute anywhere at all?"""
        return any(not home.is_nowhere for home in self.homes_for(attribute))

    def structured_home(self, attribute: str) -> AttributeHome | None:
        return next(
            (h for h in self.homes_for(attribute) if h.is_structured), None
        )

    def text_home(self, attribute: str) -> AttributeHome | None:
        return next((h for h in self.homes_for(attribute) if h.is_text), None)

    def carries_both(self, attribute: str) -> bool:
        """Structured *and* text: the shape a silver standard needs."""
        return (
            self.structured_home(attribute) is not None
            and self.text_home(attribute) is not None
        )

    # -- what a blank means -------------------------------------------------

    def availability_for_blank(self, variable: str) -> Availability:
        """What an empty ``variable`` means in this study."""
        if self.collects and variable not in self.collects:
            return "not_collected_by_protocol"
        return self.default_blank

    def collects_variable(self, variable: str) -> bool:
        return (not self.collects) or variable in self.collects

    def gate_for(self, name: str) -> GateSpec | None:
        return self.gates.get(name)

    def gate_answer(self, gate: str, row: dict[str, Any]) -> bool | None:
        spec = self.gate_values.get(gate)
        return spec.evaluate(row) if spec else None

    # -- sponsor codelists --------------------------------------------------

    def sponsor_code_for(self, value: str) -> str | None:
        return self.sponsor_codelist.get(value)

    def resolve_sponsor_code(self, code: str | None) -> str | None:
        """Map a sponsor code back to a catalogue value.

        A code the declared mapping does not cover resolves to nothing. Guessing
        which catalogue value a sponsor meant is exactly the kind of silent
        substitution this system exists to avoid.
        """
        if code in (None, ""):
            return None
        for value, mapped in self.sponsor_codelist.items():
            if str(mapped).strip().upper() == str(code).strip().upper():
                return value
        return None

    def splits_on_severity_change(self) -> bool:
        return bool(self.conventions.get("split_on_severity_change"))


class StudyProfiles:
    """Read-only view over ``study_profiles.yaml``."""

    def __init__(self, raw: dict[str, Any], source_path: Path | None = None):
        if not isinstance(raw, dict) or "profiles" not in raw:
            raise ProfileError("study_profiles.yaml must define `profiles`")
        self.raw = raw
        self.source_path = source_path

        self.homes: dict[str, AttributeHome] = {}
        for home_id, body in (raw.get("attribute_homes") or {}).items():
            kind = body.get("kind")
            if kind is not None and kind not in SOURCE_KINDS:
                raise ProfileError(
                    f"attribute home {home_id!r} names unknown source kind {kind!r}"
                )
            self.homes[home_id] = AttributeHome(
                home_id=home_id, kind=kind, variable=body.get("variable"),
                description=body.get("description", "").strip(),
            )
        if not self.homes:
            raise ProfileError("study_profiles.yaml must define `attribute_homes`")

        defaults = raw.get("defaults") or {}
        default_blank = _check_availability(
            defaults.get("blank_means", "unknown"), "defaults.blank_means"
        )
        default_gates = _parse_gates(defaults.get("gated_variables") or {})
        default_gate_values = _parse_gate_values(defaults.get("gate_values") or {})
        conventions = raw.get("conventions") or {}

        self.profiles: dict[str, StudyProfile] = {}
        for profile_id, body in (raw["profiles"] or {}).items():
            if not isinstance(body, dict):
                raise ProfileError(f"profile {profile_id!r} must be a mapping")
            self.profiles[profile_id] = StudyProfile(
                profile_id=profile_id,
                study_id=body.get("study_id", profile_id),
                label=body.get("label", profile_id),
                reported_term_style=body.get("reported_term_style", "terse"),
                location_homes=self._homes(profile_id, body.get("location_home")),
                pattern_homes=self._homes(profile_id, body.get("pattern_home")),
                sponsor_variable_name=body.get("sponsor_variable_name"),
                sponsor_codelist=dict(body.get("sponsor_codelist") or {}),
                dictionary=body.get("dictionary", defaults.get("dictionary", "")),
                dictionary_version=body.get(
                    "dictionary_version", defaults.get("dictionary_version", "")
                ),
                collects=tuple(body.get("collects") or []),
                default_blank=_check_availability(
                    body.get("blank_means", default_blank),
                    f"{profile_id}.blank_means",
                ),
                gates={**default_gates,
                       **_parse_gates(body.get("gated_variables") or {})},
                gate_values={**default_gate_values,
                             **_parse_gate_values(body.get("gate_values") or {})},
                conventions=dict(conventions.get(profile_id) or {}),
                note=body.get("note", "").strip(),
            )

        self._by_study = {p.study_id: p for p in self.profiles.values()}
        self._check_sponsor_mappings()

    def _homes(self, profile_id: str, declared: Any) -> tuple[AttributeHome, ...]:
        if declared is None:
            declared = ["none"]
        if isinstance(declared, str):
            declared = [declared]
        unknown = [h for h in declared if h not in self.homes]
        if unknown:
            raise ProfileError(
                f"profile {profile_id!r} names undefined attribute homes "
                f"{unknown}; known: {sorted(self.homes)}"
            )
        return tuple(self.homes[h] for h in declared)

    def _check_sponsor_mappings(self) -> None:
        for profile in self.profiles.values():
            uses_sponsor = any(
                home.kind == "structured_sponsor"
                for attribute in ("location", "pattern")
                for home in profile.homes_for(attribute)
            )
            if uses_sponsor and not (
                profile.sponsor_variable_name and profile.sponsor_codelist
            ):
                raise ProfileError(
                    f"profile {profile.profile_id!r} puts an attribute in a "
                    f"sponsor variable but declares no name and codelist to "
                    f"resolve it with; an unmapped sponsor code is unreadable"
                )

    # -- lookups ------------------------------------------------------------

    def profile(self, profile_id: str) -> StudyProfile:
        try:
            return self.profiles[profile_id]
        except KeyError:
            raise ProfileError(
                f"no profile {profile_id!r}; known: {sorted(self.profiles)}"
            ) from None

    def for_study(self, study_id: str) -> StudyProfile:
        try:
            return self._by_study[study_id]
        except KeyError:
            raise ProfileError(
                f"no study profile for {study_id!r}; known: "
                f"{sorted(self._by_study)}. A study with no declared profile "
                f"cannot be read: every blank in it would be guesswork."
            ) from None

    def profile_ids(self) -> list[str]:
        return sorted(self.profiles)

    def study_ids(self) -> list[str]:
        return sorted(self._by_study)

    def dictionary_versions(self) -> dict[str, str]:
        return {p.study_id: p.dictionary_version for p in self.profiles.values()}

    def evaluation_profiles(self, attribute: str = "location") -> list[str]:
        """Profiles that carry ``attribute`` both structurally and in text.

        These are the ones a silver standard can be built from: the structured
        value is a comparator the extractor never sees.
        """
        return sorted(
            p.profile_id for p in self.profiles.values() if p.carries_both(attribute)
        )


# --------------------------------------------------------------------------


def _check_availability(value: Any, where: str) -> Availability:
    if value not in AVAILABILITY_VALUES:
        raise ProfileError(
            f"{where}: {value!r} is not an availability; "
            f"known: {list(AVAILABILITY_VALUES)}"
        )
    return value  # type: ignore[return-value]


def _parse_gates(body: dict[str, Any]) -> dict[str, GateSpec]:
    gates: dict[str, GateSpec] = {}
    for variable, spec in (body or {}).items():
        gates[variable] = GateSpec(
            variable=variable,
            gate=spec["gate"],
            when_gate_false=_check_availability(
                spec.get("when_gate_false", "not_applicable_gated"),
                f"gated_variables.{variable}.when_gate_false",
            ),
        )
    return gates


def _parse_gate_values(body: dict[str, Any]) -> dict[str, GateValueSpec]:
    values: dict[str, GateValueSpec] = {}
    for gate, spec in (body or {}).items():
        values[gate] = GateValueSpec(
            gate=gate,
            from_variable=spec["from_variable"],
            true_when_in=tuple(spec.get("true_when_in") or []),
        )
    return values
