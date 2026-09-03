"""Study profiles: where each modifier lives, and what a silence means.

A profile is a collection decision. Nothing about it is inferred at read time —
a study whose profile is not declared cannot be read, because every silence in
it would be guesswork.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from pathlib import Path
from typing import Any

from .models import AVAILABILITIES, SOURCE_KINDS, Availability, SourceKind


class ProfileError(ValueError):
    """Raised when the profile config is structurally invalid."""


@dataclass(frozen=True)
class ModifierHome:
    """One place a modifier can live."""

    home_id: str
    kind: SourceKind | None
    variable: str | None
    description: str = ""

    @property
    def is_nowhere(self) -> bool:
        return self.kind is None or self.variable is None

    @property
    def is_structured(self) -> bool:
        return self.kind in ("structured_standard", "structured_sponsor", "linked_form")

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
    homes: dict[str, tuple[ModifierHome, ...]] = _dc_field(default_factory=dict)
    dictionary: str = ""
    dictionary_version: str = ""
    # One or more concepts this study codes a cutaneous event to. More than
    # one is not a defect: several codings of one clinical situation is the
    # normal state of the world, and the concept set is what reconciles them.
    prefer_concept: tuple[str, ...] = ()
    collects: tuple[str, ...] = ()
    default_silence: Availability = "not_collected"
    gates: dict[str, GateSpec] = _dc_field(default_factory=dict)
    gate_values: dict[str, GateValueSpec] = _dc_field(default_factory=dict)
    conventions: dict[str, Any] = _dc_field(default_factory=dict)
    note: str = ""

    # -- where a modifier lives --------------------------------------------

    def homes_for(self, modifier: str) -> tuple[ModifierHome, ...]:
        return self.homes.get(modifier, ())

    def home_ids(self, modifier: str) -> list[str]:
        return [h.home_id for h in self.homes_for(modifier)]

    def collects_modifier(self, modifier: str) -> bool:
        return any(not home.is_nowhere for home in self.homes_for(modifier))

    def structured_home(self, modifier: str) -> ModifierHome | None:
        return next((h for h in self.homes_for(modifier) if h.is_structured), None)

    def text_home(self, modifier: str) -> ModifierHome | None:
        return next((h for h in self.homes_for(modifier) if h.is_text), None)

    def carries_both(self, modifier: str) -> bool:
        """Structured *and* text: the shape a silver standard needs."""
        return (
            self.structured_home(modifier) is not None
            and self.text_home(modifier) is not None
        )

    def supportability(self, modifier: str) -> tuple[str, str]:
        """Can this study answer a question requiring the modifier?

        Decided on metadata alone, before any patient-level query.
        """
        if not self.collects_modifier(modifier):
            return "cannot_ascertain", (
                f"{self.profile_id} records {modifier} nowhere, so no query "
                f"over it can be answered from this study"
            )
        if self.structured_home(modifier) is not None:
            return "supported", (
                f"{modifier} is a structured variable "
                f"({self.structured_home(modifier).variable}) in {self.profile_id}"
            )
        home = self.text_home(modifier)
        return "supported_via_extraction", (
            f"{modifier} lives only in {home.variable} in {self.profile_id}, so "
            f"answering requires text extraction and carries its measured error "
            f"rate"
        )

    # -- what a silence means ----------------------------------------------

    def availability_for_silence(self, variable: str) -> Availability:
        if self.collects and variable not in self.collects:
            return "not_collected"
        return "unresolved"

    def collects_variable(self, variable: str) -> bool:
        return (not self.collects) or variable in self.collects

    def gate_for(self, name: str) -> GateSpec | None:
        return self.gates.get(name)

    def gate_answer(self, gate: str, row: dict[str, Any]) -> bool | None:
        spec = self.gate_values.get(gate)
        return spec.evaluate(row) if spec else None


class StudyProfiles:
    """Read-only view over ``study_profiles.yaml``."""

    def __init__(self, raw: dict[str, Any], source_path: Path | None = None):
        if not isinstance(raw, dict) or "profiles" not in raw:
            raise ProfileError("study_profiles.yaml must define `profiles`")
        self.raw = raw
        self.source_path = source_path

        self.homes: dict[str, ModifierHome] = {}
        for home_id, body in (raw.get("modifier_homes") or {}).items():
            kind = body.get("kind")
            if kind is not None and kind not in SOURCE_KINDS:
                raise ProfileError(
                    f"modifier home {home_id!r} names unknown source kind {kind!r}"
                )
            self.homes[home_id] = ModifierHome(
                home_id=home_id, kind=kind, variable=body.get("variable"),
                description=(body.get("description") or "").strip(),
            )
        if not self.homes:
            raise ProfileError("study_profiles.yaml must define `modifier_homes`")

        defaults = raw.get("defaults") or {}
        default_silence = _check(
            defaults.get("silence_means", "not_collected"), "defaults.silence_means"
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
                homes={
                    modifier: self._homes(profile_id, declared)
                    for modifier, declared in (body.get("modifier_homes") or {}).items()
                },
                dictionary=body.get("dictionary", defaults.get("dictionary", "")),
                dictionary_version=body.get(
                    "dictionary_version", defaults.get("dictionary_version", "")
                ),
                prefer_concept=_as_tuple(body.get("prefer_concept")),
                collects=tuple(body.get("collects") or []),
                default_silence=_check(
                    body.get("silence_means", default_silence),
                    f"{profile_id}.silence_means",
                ),
                gates={**default_gates,
                       **_parse_gates(body.get("gated_variables") or {})},
                gate_values={**default_gate_values,
                             **_parse_gate_values(body.get("gate_values") or {})},
                conventions=dict(conventions.get(profile_id) or {}),
                note=(body.get("note") or "").strip(),
            )
        self._by_study = {p.study_id: p for p in self.profiles.values()}

    def _homes(self, profile_id: str, declared: Any) -> tuple[ModifierHome, ...]:
        if declared is None:
            declared = ["none"]
        if isinstance(declared, str):
            declared = [declared]
        unknown = [h for h in declared if h not in self.homes]
        if unknown:
            raise ProfileError(
                f"profile {profile_id!r} names undefined modifier homes {unknown}; "
                f"known: {sorted(self.homes)}"
            )
        return tuple(self.homes[h] for h in declared)

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
                f"cannot be read: every silence in it would be guesswork."
            ) from None

    def profile_ids(self) -> list[str]:
        return sorted(self.profiles)

    def study_ids(self) -> list[str]:
        return sorted(self._by_study)

    def dictionary_versions(self) -> dict[str, str]:
        return {p.study_id: p.dictionary_version for p in self.profiles.values()}

    def evaluation_profiles(self, modifier: str) -> list[str]:
        """Profiles carrying ``modifier`` both structurally and in text.

        These are the ones a silver standard can be built from: the structured
        value is a comparator the extractor never sees.
        """
        return sorted(
            p.profile_id for p in self.profiles.values() if p.carries_both(modifier)
        )

    def supportability(self, modifier: str) -> list[dict[str, Any]]:
        """Per study: supported, supported-via-extraction, or cannot-ascertain."""
        rows = []
        for profile in sorted(self.profiles.values(), key=lambda p: p.study_id):
            status, reason = profile.supportability(modifier)
            rows.append({
                "study_id": profile.study_id,
                "profile": profile.profile_id,
                "status": status,
                "reason": reason,
                "homes": profile.home_ids(modifier),
            })
        return rows


def _as_tuple(value: Any) -> tuple[str, ...]:
    """Accept one concept id or a list of them; store one shape."""
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _check(value: Any, where: str) -> Availability:
    if value not in AVAILABILITIES:
        raise ProfileError(
            f"{where}: {value!r} is not an availability; known: "
            f"{list(AVAILABILITIES)}"
        )
    return value  # type: ignore[return-value]


def _parse_gates(body: dict[str, Any]) -> dict[str, GateSpec]:
    return {
        variable: GateSpec(
            variable=variable, gate=spec["gate"],
            when_gate_false=_check(
                spec.get("when_gate_false", "not_applicable"),
                f"gated_variables.{variable}.when_gate_false",
            ),
        )
        for variable, spec in (body or {}).items()
    }


def _parse_gate_values(body: dict[str, Any]) -> dict[str, GateValueSpec]:
    return {
        gate: GateValueSpec(
            gate=gate, from_variable=spec["from_variable"],
            true_when_in=tuple(spec.get("true_when_in") or []),
        )
        for gate, spec in (body or {}).items()
    }
