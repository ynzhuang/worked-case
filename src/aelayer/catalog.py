"""Configuration: concepts, modifier catalogues, dictionary versions, extraction.

Three separate normalization problems live here, and the code keeps them
separate because they fail differently:

**Language variation.** "oral mucosal involvement" and "mucosal lesions of the
mouth" resolve to one modifier value. Handled by the modifier catalogue's
surface forms, read by the model path.

**Coded-concept variation.** ``Rash`` and ``Rash erythematous`` are both
legitimate codings. Nothing merges them; a phenotype's concept set decides which
qualify, and no code path rewrites a coded value.

**Terminology-version variation.** A code recorded under one dictionary version
is reconciled to a target version only where the mapping is mechanical. What
does not persist is flagged for review, never auto-recoded.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml

from . import paths
from .hashing import extractor_version, hash_file, normalizer_version


class ConfigError(ValueError):
    """Raised when a config file is structurally invalid."""


Reconciliation = Literal[
    "unchanged", "remapped_mechanically", "flagged_for_review", "not_attempted"
]


# --------------------------------------------------------------------------
# Concepts
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Concept:
    concept_id: str
    label: str
    codes: dict[str, str]          # dictionary version -> coded string
    lexicon: tuple[str, ...] = ()

    def code_in(self, version: str | None) -> str | None:
        return self.codes.get(version or "")

    def versions(self) -> list[str]:
        return sorted(self.codes)


@dataclass(frozen=True)
class ModifierValue:
    value_id: str
    label: str
    surface_forms: tuple[str, ...]


@dataclass(frozen=True)
class ModifierCatalogue:
    """The normalized value space for one modifier."""

    name: str
    label: str
    description: str
    values: dict[str, ModifierValue]

    def value_ids(self) -> list[str]:
        return sorted(self.values)

    def surface_index(self) -> dict[str, str]:
        """Every declared surface form, mapped to its value id, longest first."""
        index: dict[str, str] = {}
        for value in self.values.values():
            for form in value.surface_forms:
                index[" ".join(form.lower().replace("-", " ").split())] = value.value_id
        return index

    def normalize(self, surface: str) -> str | None:
        """Map a surface form onto a catalogue value.

        Declared forms only. A phrase nobody wrote into the catalogue returns
        None, and the extractor abstains rather than inventing a value.
        """
        folded = " ".join(surface.strip().lower().replace("-", " ").split())
        return self.surface_index().get(folded)


@dataclass(frozen=True)
class ReconciledCode:
    """The result of reconciling one coded value to a target version."""

    code: str
    source_version: str
    target_version: str
    concept_id: str | None
    reconciled_to: str | None
    outcome: Reconciliation
    note: str = ""


class ConceptCatalog:
    """Read-only view over ``concepts.yaml``."""

    def __init__(self, raw: dict[str, Any], source_path: Path | None = None):
        self.raw = raw
        self.source_path = source_path
        self._validate(raw)

        self.dictionaries: dict[str, Any] = raw.get("dictionaries") or {}
        self.dictionary_name: str = self.dictionaries.get("name", "")
        self.dictionary_versions: tuple[str, ...] = tuple(
            self.dictionaries.get("versions") or []
        )
        self.target_version: str = self.dictionaries.get("target", "")
        #: The licensing notice declared beside the terms. Carried on the
        #: object rather than left in the YAML, so anything that reports a
        #: coded value can print it without re-reading the file.
        self.notice: str = (self.dictionaries.get("notice") or "").strip()

        self.concepts: dict[str, Concept] = {
            cid: Concept(
                concept_id=cid,
                label=body.get("label", cid),
                codes={str(k): str(v) for k, v in (body.get("codes") or {}).items()},
                lexicon=tuple(body.get("lexicon") or []),
            )
            for cid, body in (raw.get("concepts") or {}).items()
        }

        self.concept_groups: dict[str, list[str]] = {}
        for gid, body in (raw.get("concept_groups") or {}).items():
            members = list(body.get("members") or [])
            unknown = [m for m in members if m not in self.concepts]
            if unknown:
                raise ConfigError(
                    f"concept group {gid!r} names unknown concepts: {unknown}"
                )
            self.concept_groups[gid] = members

        self.modifiers: dict[str, ModifierCatalogue] = {}
        for name, body in (raw.get("modifiers") or {}).items():
            self.modifiers[name] = ModifierCatalogue(
                name=name,
                label=body.get("label", name),
                description=(body.get("description") or "").strip(),
                values={
                    vid: ModifierValue(
                        value_id=vid,
                        label=vbody.get("label", vid),
                        surface_forms=tuple(vbody.get("surface_forms") or []),
                    )
                    for vid, vbody in (body.get("values") or {}).items()
                },
            )

        # code string -> (concept_id, version), for reading a coded value back
        self._by_code: dict[tuple[str, str], str] = {}
        for concept in self.concepts.values():
            for version, code in concept.codes.items():
                self._by_code[(code.casefold(), version)] = concept.concept_id

    # -- validation ---------------------------------------------------------

    @staticmethod
    def _validate(raw: dict[str, Any]) -> None:
        if not isinstance(raw, dict):
            raise ConfigError("concepts.yaml must be a mapping")
        if not raw.get("concepts"):
            raise ConfigError("concepts.yaml must define at least one concept")
        dictionaries = raw.get("dictionaries") or {}
        versions = set(dictionaries.get("versions") or [])
        if not versions:
            raise ConfigError("concepts.yaml must declare dictionary versions")
        if dictionaries.get("target") not in versions:
            raise ConfigError(
                f"the target dictionary version "
                f"{dictionaries.get('target')!r} is not one of {sorted(versions)}"
            )
        for cid, body in raw["concepts"].items():
            codes = body.get("codes") or {}
            if not codes:
                raise ConfigError(f"concept {cid!r} declares no coded strings")
            unknown = sorted(set(codes) - versions)
            if unknown:
                raise ConfigError(
                    f"concept {cid!r} declares codes under unknown dictionary "
                    f"versions {unknown}"
                )
        for name, body in (raw.get("modifiers") or {}).items():
            values = body.get("values") or {}
            if not values:
                raise ConfigError(f"modifier {name!r} defines no values")
            for vid, vbody in values.items():
                if not vbody.get("surface_forms"):
                    raise ConfigError(
                        f"{name}.{vid} declares no surface forms, so nothing in "
                        f"text could ever normalize to it"
                    )

    # -- lookups ------------------------------------------------------------

    def concept(self, concept_id: str) -> Concept:
        try:
            return self.concepts[concept_id]
        except KeyError:
            raise ConfigError(f"unknown concept {concept_id!r}") from None

    def expand_group(self, group_id: str) -> list[str]:
        try:
            return list(self.concept_groups[group_id])
        except KeyError:
            raise ConfigError(f"unknown concept group {group_id!r}") from None

    def modifier(self, name: str) -> ModifierCatalogue:
        try:
            return self.modifiers[name]
        except KeyError:
            raise ConfigError(
                f"unknown modifier {name!r}; known: {sorted(self.modifiers)}"
            ) from None

    def concept_for_code(self, code: str | None, version: str | None) -> str | None:
        """Which concept a coded string belongs to, under its own version."""
        if not code:
            return None
        folded = code.strip().casefold()
        if version and (folded, version) in self._by_code:
            return self._by_code[(folded, version)]
        for (candidate, _version), concept_id in self._by_code.items():
            if candidate == folded:
                return concept_id
        return None

    # -- terminology-version reconciliation --------------------------------

    def reconcile(
        self, code: str, source_version: str, target_version: str | None = None
    ) -> ReconciledCode:
        """Reconcile one coded value to a target dictionary version.

        Mechanical only. A code that persists across versions remaps; one that
        does not is **flagged for review, never auto-recoded**. Nothing here
        consults a model, and the original code is never modified — the result
        is an additional field, not a replacement.
        """
        target = target_version or self.target_version
        concept_id = self.concept_for_code(code, source_version)
        if concept_id is None:
            return ReconciledCode(
                code=code, source_version=source_version, target_version=target,
                concept_id=None, reconciled_to=None, outcome="flagged_for_review",
                note=(
                    f"{code!r} is not a code this catalogue knows under "
                    f"{source_version}, so no mechanical mapping exists"
                ),
            )
        concept = self.concepts[concept_id]
        if source_version == target:
            return ReconciledCode(
                code=code, source_version=source_version, target_version=target,
                concept_id=concept_id, reconciled_to=code, outcome="unchanged",
                note="already recorded under the target version",
            )
        mapped = concept.code_in(target)
        if mapped is None:
            return ReconciledCode(
                code=code, source_version=source_version, target_version=target,
                concept_id=concept_id, reconciled_to=None,
                outcome="flagged_for_review",
                note=(
                    f"{concept_id} has no code under {target}; a human decides "
                    f"what it becomes, and no model recodes it"
                ),
            )
        if mapped == code:
            return ReconciledCode(
                code=code, source_version=source_version, target_version=target,
                concept_id=concept_id, reconciled_to=mapped, outcome="unchanged",
                note=f"the code is identical under {target}",
            )
        return ReconciledCode(
            code=code, source_version=source_version, target_version=target,
            concept_id=concept_id, reconciled_to=mapped,
            outcome="remapped_mechanically",
            note=f"{code!r} is {mapped!r} under {target}; the mapping is 1:1",
        )


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


class ExtractionConfig:
    """Read-only view over ``extraction.yaml``."""

    REQUIRED = ("readable_sources", "extractable_modifiers", "assertion")

    def __init__(self, raw: dict[str, Any], source_path: Path | None = None):
        if not isinstance(raw, dict):
            raise ConfigError("extraction.yaml must be a mapping")
        for section in self.REQUIRED:
            if section not in raw:
                raise ConfigError(f"extraction.yaml is missing section {section!r}")
        self.raw = raw
        self.source_path = source_path
        self.version: str = raw.get("version", "extract-4.0.0")
        self.readable_sources: tuple[str, ...] = tuple(raw["readable_sources"])
        self.extractable_modifiers: tuple[str, ...] = tuple(
            raw["extractable_modifiers"]
        )
        self.assertion: dict[str, Any] = raw["assertion"]
        self.confidence: dict[str, float] = raw.get("confidence") or {}
        self.anchors: dict[str, Any] = raw.get("anchors") or {}
        self.default_anchor: str | None = raw.get("default_anchor")
        self._validate()

    def _validate(self) -> None:
        from .models import ASSERTIONS, SOURCE_KINDS, STRUCTURED_SOURCES

        unknown = sorted(set(self.readable_sources) - set(SOURCE_KINDS))
        if unknown:
            raise ConfigError(f"readable_sources names unknown source kinds: {unknown}")
        forbidden = sorted(set(self.readable_sources) & STRUCTURED_SOURCES)
        if forbidden:
            raise ConfigError(
                f"readable_sources includes {forbidden}: a value the CRF already "
                f"settled is not a question for a model, and the guard would "
                f"refuse it anyway"
            )
        for assertion_class in (self.assertion.get("cues") or {}):
            if assertion_class not in ASSERTIONS:
                raise ConfigError(
                    f"unknown assertion class in cues: {assertion_class!r}"
                )
        if self.assertion.get("default", "present") not in ASSERTIONS:
            raise ConfigError(
                f"default assertion {self.assertion.get('default')!r} is not one "
                f"of {list(ASSERTIONS)}"
            )
        scope = self.assertion.get("scope", "sentence")
        if scope not in ("sentence", "window"):
            raise ConfigError(f"assertion scope must be sentence|window, got {scope!r}")

    def cue_lists(self, assertion_class: str) -> tuple[list[str], list[str]]:
        """``(pre_cues, post_cues)`` for one assertion class."""
        body = (self.assertion.get("cues") or {}).get(assertion_class)
        if body is None:
            return [], []
        if isinstance(body, list):
            return list(body), []
        return list(body.get("pre") or []), list(body.get("post") or [])

    def confidence_for(self, key: str, default: float = 0.5) -> float:
        value = self.confidence.get(key)
        return float(default if value is None else value)


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Configs:
    catalog: ConceptCatalog
    extraction: ExtractionConfig
    profiles: Any
    extractor_version: str
    normalizer_version: str

    def dictionary_versions(self) -> dict[str, str]:
        return self.profiles.dictionary_versions()


def load_yaml(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if data is None:
        raise ConfigError(f"{path}: file is empty")
    return data


@lru_cache(maxsize=16)
def _load_cached(
    concepts_path: str, extraction_path: str, profiles_path: str,
    _concepts_hash: str, _extraction_hash: str, _profiles_hash: str,
) -> Configs:
    from .profiles import StudyProfiles

    catalog = ConceptCatalog(load_yaml(concepts_path), Path(concepts_path))
    extraction = ExtractionConfig(load_yaml(extraction_path), Path(extraction_path))
    profiles = StudyProfiles(load_yaml(profiles_path), Path(profiles_path))
    return Configs(
        catalog=catalog,
        extraction=extraction,
        profiles=profiles,
        extractor_version=extractor_version(concepts_path, extraction_path),
        normalizer_version=normalizer_version(concepts_path, profiles_path),
    )


def load_configs(
    concepts_path: str | Path | None = None,
    extraction_path: str | Path | None = None,
    profiles_path: str | Path | None = None,
) -> Configs:
    """Load every config file and compute the versions they imply.

    Cached on content hash, so editing a file in place is picked up without
    restarting the process.
    """
    cp = str(concepts_path or paths.CONCEPTS_YAML)
    ep = str(extraction_path or paths.EXTRACTION_YAML)
    pp = str(profiles_path or paths.PROFILES_YAML)
    return _load_cached(cp, ep, pp, hash_file(cp), hash_file(ep), hash_file(pp))
