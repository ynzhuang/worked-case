"""Configuration: the concept catalogue, the attribute catalogues, extraction.

The catalogue is the shared value space. Every route into the system — a
standard variable, a sponsor codelist, an investigator's phrase, a comment —
resolves into the *same* normalized value, which is the only reason a phenotype
rule can accept all of them without caring which one it got.

Everything is loaded from ``config/`` and hashed by content, so a version string
changes exactly when the thing it names changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from . import paths
from .hashing import extractor_version, hash_file, normalizer_version


class ConfigError(ValueError):
    """Raised when a config file is structurally invalid."""


# --------------------------------------------------------------------------
# Concepts
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Concept:
    concept_id: str
    label: str
    coded_terms: dict[str, Any]
    lexicon: tuple[str, ...]
    abbreviations: tuple[str, ...]
    recurrence_expected: bool = True
    #: Whether the coded term for this concept can carry a body site. False for
    #: every concept here, which is why the site has to survive somewhere else.
    carries_body_site: bool = False

    def all_coded_terms(self) -> list[str]:
        terms: set[str] = set()
        for key, value in self.coded_terms.items():
            if key == "by_dictionary_version":
                for version_terms in (value or {}).values():
                    terms.update(version_terms or [])
            else:
                terms.update(value or [])
        return sorted(terms)

    def coded_terms_for_version(self, version: str | None) -> list[str]:
        """The terms this concept carried under one dictionary version.

        Falls back to the version-independent lists where a study's version is
        unknown or not enumerated — which is a real situation, not an error.
        """
        by_version = self.coded_terms.get("by_dictionary_version") or {}
        if version and version in by_version:
            return sorted(by_version[version] or [])
        flat: set[str] = set()
        for key, value in self.coded_terms.items():
            if key != "by_dictionary_version":
                flat.update(value or [])
        return sorted(flat)


@dataclass(frozen=True)
class AttributeValue:
    """One permissible value of an attribute, and how it appears in text."""

    value_id: str
    label: str
    surface_forms: tuple[str, ...]
    region: str | None = None


@dataclass(frozen=True)
class AttributeCatalogue:
    """The normalized value space for one attribute."""

    name: str
    label: str
    values: dict[str, AttributeValue]
    regions: dict[str, tuple[str, ...]] = _dc_field(default_factory=dict)

    def value_ids(self) -> list[str]:
        return sorted(self.values)

    def normalize(self, surface: str) -> str | None:
        """Map a surface form onto a catalogue value.

        Exact declared forms only. A phrase nobody wrote into the catalogue
        returns None, and the extractor abstains rather than inventing a value.
        """
        folded = " ".join(surface.strip().lower().replace("-", " ").split())
        for value in self.values.values():
            for form in value.surface_forms:
                if folded == " ".join(form.lower().replace("-", " ").split()):
                    return value.value_id
        return None

    def surface_index(self) -> dict[str, str]:
        """Every declared surface form, mapped to its value id."""
        index: dict[str, str] = {}
        for value in self.values.values():
            for form in value.surface_forms:
                index[" ".join(form.lower().replace("-", " ").split())] = value.value_id
        return index

    def in_region(self, region: str) -> list[str]:
        if region not in self.regions:
            raise ConfigError(
                f"unknown region {region!r} for {self.name}; "
                f"known: {sorted(self.regions)}"
            )
        return list(self.regions[region])


class ConceptCatalog:
    """Read-only view over ``concepts.yaml``."""

    def __init__(self, raw: dict[str, Any], source_path: Path | None = None):
        self.raw = raw
        self.source_path = source_path
        self._validate(raw)

        self.terminology: dict[str, Any] = raw.get("terminology") or {}
        self.concepts: dict[str, Concept] = {}
        for cid, body in (raw.get("concepts") or {}).items():
            self.concepts[cid] = Concept(
                concept_id=cid,
                label=body.get("label", cid),
                coded_terms=body.get("coded_terms") or {},
                lexicon=tuple(body.get("lexicon") or []),
                abbreviations=tuple(body.get("abbreviations") or []),
                recurrence_expected=bool(body.get("recurrence_expected", True)),
                carries_body_site=bool(body.get("carries_body_site", False)),
            )

        self.concept_groups: dict[str, list[str]] = {}
        for gid, body in (raw.get("concept_groups") or {}).items():
            members = list(body.get("members") or [])
            unknown = [m for m in members if m not in self.concepts]
            if unknown:
                raise ConfigError(
                    f"concept group {gid!r} names unknown concepts: {unknown}"
                )
            self.concept_groups[gid] = members

        self.attributes: dict[str, AttributeCatalogue] = {}
        for name, body in (raw.get("attribute_catalogues") or {}).items():
            values = {
                vid: AttributeValue(
                    value_id=vid,
                    label=vbody.get("label", vid),
                    surface_forms=tuple(vbody.get("surface_forms") or []),
                    region=vbody.get("region"),
                )
                for vid, vbody in (body.get("values") or {}).items()
            }
            regions = {
                region: tuple(members)
                for region, members in (body.get("regions") or {}).items()
            }
            unknown = sorted(
                {m for members in regions.values() for m in members} - set(values)
            )
            if unknown:
                raise ConfigError(
                    f"attribute {name!r} places unknown values in regions: {unknown}"
                )
            self.attributes[name] = AttributeCatalogue(
                name=name, label=body.get("label", name), values=values, regions=regions
            )

        self.symptom_lexicon: dict[str, list[str]] = {
            k: list(v) for k, v in (raw.get("symptom_lexicon") or {}).items()
        }
        self.episode_reconciliation: dict[str, Any] = (
            raw.get("episode_reconciliation") or {}
        )

    # -- validation ---------------------------------------------------------

    @staticmethod
    def _validate(raw: dict[str, Any]) -> None:
        if not isinstance(raw, dict):
            raise ConfigError("concepts.yaml must be a mapping")
        if not raw.get("concepts"):
            raise ConfigError("concepts.yaml must define at least one concept")
        for cid, body in raw["concepts"].items():
            if not isinstance(body, dict):
                raise ConfigError(f"concept {cid!r} must be a mapping")
            if not body.get("lexicon") and not body.get("coded_terms"):
                raise ConfigError(
                    f"concept {cid!r} needs a lexicon or coded terms to be matchable"
                )
        for name, body in (raw.get("attribute_catalogues") or {}).items():
            values = body.get("values") or {}
            if not values:
                raise ConfigError(f"attribute catalogue {name!r} defines no values")
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

    def attribute(self, name: str) -> AttributeCatalogue:
        try:
            return self.attributes[name]
        except KeyError:
            raise ConfigError(
                f"unknown attribute catalogue {name!r}; "
                f"known: {sorted(self.attributes)}"
            ) from None

    def normalize(self, attribute: str, surface: str) -> str | None:
        return self.attribute(attribute).normalize(surface)

    def concept_for_coded_term(self, term: str | None) -> str | None:
        if not term:
            return None
        folded = term.strip().casefold()
        for concept in self.concepts.values():
            if folded in {t.casefold() for t in concept.all_coded_terms()}:
                return concept.concept_id
        return None

    def dictionary_versions(self) -> list[str]:
        return list(self.terminology.get("versions") or [])


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


class ExtractionConfig:
    """Read-only view over ``extraction.yaml``."""

    REQUIRED = ("modifiers", "readable_sources", "extractable_attributes")

    def __init__(self, raw: dict[str, Any], source_path: Path | None = None):
        if not isinstance(raw, dict):
            raise ConfigError("extraction.yaml must be a mapping")
        for section in self.REQUIRED:
            if section not in raw:
                raise ConfigError(f"extraction.yaml is missing section {section!r}")
        self.raw = raw
        self.source_path = source_path
        self.version: str = raw.get("version", "extract-3.0.0")
        self.readable_sources: tuple[str, ...] = tuple(raw["readable_sources"])
        self.extractable_attributes: tuple[str, ...] = tuple(
            raw["extractable_attributes"]
        )
        self.modifiers: dict[str, Any] = raw["modifiers"]
        self.quality_lexicon: dict[str, list[str]] = raw.get("quality_lexicon") or {}
        self.severity: dict[str, list[str]] = raw.get("severity") or {}
        self.anchors: dict[str, Any] = raw.get("anchors") or {}
        self.default_anchor: str | None = raw.get("default_anchor")
        self.confidence: dict[str, float] = raw.get("confidence") or {}
        self._validate()

    def _validate(self) -> None:
        from .models import SOURCE_KINDS

        unknown = sorted(set(self.readable_sources) - set(SOURCE_KINDS))
        if unknown:
            raise ConfigError(f"readable_sources names unknown source kinds: {unknown}")
        forbidden = sorted(
            set(self.readable_sources)
            & {"structured_standard", "structured_sponsor"}
        )
        if forbidden:
            raise ConfigError(
                f"readable_sources includes {forbidden}: a value the CRF already "
                f"settled is not a question for a model, and the guard would "
                f"refuse it anyway"
            )
        scope = self.modifiers.get("scope", "sentence")
        if scope not in ("sentence", "window"):
            raise ConfigError(f"modifier scope must be sentence|window, got {scope!r}")

    def confidence_for(self, key: str, default: float = 0.5) -> float:
        value = self.confidence.get(key)
        if value is None:
            value = (self.modifiers.get("confidence") or {}).get(key)
        return float(default if value is None else value)

    @property
    def review_below(self) -> float:
        return self.confidence_for("review_below", 0.6)


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Configs:
    """Everything loaded from `config/`, with the versions they imply."""

    catalog: ConceptCatalog
    extraction: ExtractionConfig
    profiles: Any
    extractor_version: str
    normalizer_version: str

    def terminology_versions(self) -> dict[str, str]:
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
    concepts_path: str,
    extraction_path: str,
    profiles_path: str,
    _concepts_hash: str,
    _extraction_hash: str,
    _profiles_hash: str,
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
