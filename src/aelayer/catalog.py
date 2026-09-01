"""The concept catalogue and the extraction rule set, loaded and validated.

Both files are read once and exposed as small typed accessors.  Nothing here
interprets clinical meaning; it just makes the config queryable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from . import paths
from .hashing import extractor_version, hash_file, normalizer_version


class ConfigError(ValueError):
    """Raised when a config file is structurally invalid.

    A definition or catalogue that fails validation does not run.
    """


@dataclass(frozen=True)
class Concept:
    concept_id: str
    label: str
    coded_terms: dict[str, list[str]] = field(default_factory=dict)
    lexicon: list[str] = field(default_factory=list)
    abbreviations: list[str] = field(default_factory=list)
    context_required: list[str] = field(default_factory=list)
    context_gate: dict[str, Any] = field(default_factory=dict)
    #: Whether two adjacent records of this concept are more likely two
    #: episodes than one evolving episode. Hypoglycemia recurs; anaemia
    #: evolves. The reconciliation default is wrong for the former, so the
    #: catalogue states it rather than letting the linker assume.
    recurrence_expected: bool = True
    #: Evidence that makes this concept a candidate with no mention present.
    #: Empty means the concept is only ever raised by an explicit mention.
    candidate_evidence: dict[str, Any] = field(default_factory=dict)

    def all_coded_terms(self) -> list[str]:
        """Every coded term for this concept, across every dictionary version."""
        out: list[str] = []
        for key, values in self.coded_terms.items():
            if key == "by_dictionary_version":
                for body in values.values():
                    for terms in body.values():
                        out.extend(terms)
                continue
            out.extend(values)
        return sorted(set(out))

    def coded_terms_for_version(self, dictionary_version: str | None) -> list[str]:
        """Coded terms in force under one dictionary version.

        Used when a definition does *not* bridge versions: a study coded under
        an earlier dictionary is then matched only on the terms that dictionary
        actually had.
        """
        by_version = self.coded_terms.get("by_dictionary_version") or {}
        if dictionary_version and dictionary_version in by_version:
            body = by_version[dictionary_version]
            return sorted({t for terms in body.values() for t in terms})
        return sorted(
            {
                t
                for key, values in self.coded_terms.items()
                if key != "by_dictionary_version"
                for t in values
            }
        )

    @property
    def abbreviations_gated(self) -> bool:
        return "abbreviations" in self.context_required


@dataclass(frozen=True)
class LabTest:
    test_id: str
    label: str
    names: list[str]
    canonical_unit: str
    conversions: dict[str, float]
    plausible_range: tuple[float, float] | None = None

    def to_canonical(self, value: float, unit: str) -> float | None:
        """Convert to the canonical unit, or None if the unit is unknown.

        Unit conversion is not decoration.  A threshold rule applied to an
        unconverted mmol/L value misclassifies an entire study silently.
        """
        factor = self.conversions.get(unit)
        if factor is None:
            for known, known_factor in self.conversions.items():
                if known.lower() == unit.lower():
                    factor = known_factor
                    break
        if factor is None:
            return None
        return round(value * factor, 4)

    def plausible(self, canonical_value: float) -> bool:
        if not self.plausible_range:
            return True
        low, high = self.plausible_range
        return low <= canonical_value <= high


class ConceptCatalog:
    """Read-only view over ``concepts.yaml``."""

    def __init__(self, raw: dict[str, Any], source_path: Path | None = None):
        self.raw = raw
        self.source_path = source_path
        self._validate(raw)

        self.concepts: dict[str, Concept] = {}
        for cid, body in (raw.get("concepts") or {}).items():
            self.concepts[cid] = Concept(
                concept_id=cid,
                label=body.get("label", cid),
                coded_terms=body.get("coded_terms") or {},
                lexicon=list(body.get("lexicon") or []),
                abbreviations=list(body.get("abbreviations") or []),
                context_required=list(body.get("context_required") or []),
                context_gate=body.get("context_gate") or {},
                candidate_evidence=body.get("candidate_evidence") or {},
                recurrence_expected=bool(body.get("recurrence_expected", True)),
            )

        self.symptom_sets: dict[str, list[str]] = {
            k: list(v) for k, v in (raw.get("symptom_sets") or {}).items()
        }
        self.symptom_lexicon: dict[str, list[str]] = {
            k: list(v) for k, v in (raw.get("symptom_lexicon") or {}).items()
        }
        for symptoms in self.symptom_sets.values():
            for symptom in symptoms:
                self.symptom_lexicon.setdefault(symptom, [symptom])

        self.lab_tests: dict[str, LabTest] = {}
        for tid, body in (raw.get("lab_tests") or {}).items():
            rng = body.get("plausible_range")
            self.lab_tests[tid] = LabTest(
                test_id=tid,
                label=body.get("label", tid),
                names=list(body.get("names") or []),
                canonical_unit=body["canonical_unit"],
                conversions={str(k): float(v) for k, v in (body.get("conversions") or {}).items()},
                plausible_range=(float(rng[0]), float(rng[1])) if rng else None,
            )

        self.episode_reconciliation: dict[str, Any] = (
            raw.get("episode_reconciliation") or {}
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
            gate_required = body.get("context_required") or []
            if "abbreviations" in gate_required and not body.get("context_gate"):
                raise ConfigError(
                    f"concept {cid!r} gates abbreviations but defines no context_gate"
                )
        for tid, body in (raw.get("lab_tests") or {}).items():
            if "canonical_unit" not in body:
                raise ConfigError(f"lab test {tid!r} has no canonical_unit")
            conversions = body.get("conversions") or {}
            if body["canonical_unit"] not in conversions:
                raise ConfigError(
                    f"lab test {tid!r} has no conversion entry for its own "
                    f"canonical unit {body['canonical_unit']!r}"
                )

    # -- lookups ------------------------------------------------------------

    def concept(self, concept_id: str) -> Concept:
        try:
            return self.concepts[concept_id]
        except KeyError:
            raise ConfigError(f"unknown concept {concept_id!r}") from None

    def expand_group(self, group_id: str) -> list[str]:
        """Members of an explicitly named group.

        Grouping above term level is always an explicit list.  No hierarchy is
        walked as though it implied subsumption.
        """
        try:
            return list(self.concept_groups[group_id])
        except KeyError:
            raise ConfigError(f"unknown concept group {group_id!r}") from None

    def symptoms_in_sets(self, set_names: list[str]) -> set[str]:
        out: set[str] = set()
        for name in set_names:
            if name not in self.symptom_sets:
                raise ConfigError(f"unknown symptom set {name!r}")
            out.update(self.symptom_sets[name])
        return out

    def set_for_symptom(self, symptom: str) -> list[str]:
        return sorted(
            name for name, members in self.symptom_sets.items() if symptom in members
        )

    def synonyms(self, concept_id: str) -> list[str]:
        """Every surface form for a concept: lexicon plus coded terms."""
        concept = self.concept(concept_id)
        return sorted(set(concept.lexicon) | set(concept.all_coded_terms()))


class ExtractionConfig:
    """Read-only view over ``extraction.yaml``."""

    def __init__(self, raw: dict[str, Any], source_path: Path | None = None):
        if not isinstance(raw, dict):
            raise ConfigError("extraction.yaml must be a mapping")
        for section in ("assertion", "temporality", "values", "labs"):
            if section not in raw:
                raise ConfigError(f"extraction.yaml is missing section {section!r}")
        self.raw = raw
        self.source_path = source_path
        self.assertion = raw["assertion"]
        self.temporality = raw["temporality"]
        self.anchors = raw.get("anchors") or {}
        #: Where a relative expression names no anchor of its own, and where an
        #: episode is stamped with an offset for retrieval, this is the event
        #: measured from.
        self.default_anchor = (self.temporality or {}).get("default_anchor")
        self.labs = raw["labs"]
        self.values = raw["values"]
        self.normalisation = raw.get("normalisation") or {}
        self.confidence = raw.get("confidence") or {}
        self._validate()

    def _validate(self) -> None:
        scope = self.assertion.get("scope", "sentence")
        if scope not in ("sentence", "window"):
            raise ConfigError(f"assertion scope must be sentence|window, got {scope!r}")
        from .models import ASSERTION_VALUES

        for cls in (self.assertion.get("cues") or {}):
            if cls not in ASSERTION_VALUES:
                raise ConfigError(f"unknown assertion class in cues: {cls!r}")
        for cls in self.assertion.get("precedence") or []:
            if cls not in ASSERTION_VALUES:
                raise ConfigError(f"unknown assertion class in precedence: {cls!r}")

    def cue_lists(self, assertion_class: str) -> tuple[list[str], list[str]]:
        """Return ``(pre_cues, post_cues)`` for a class.

        A flat list in the YAML is treated as ``pre``, which keeps the simple
        form in the brief valid while allowing post-cues where they matter.
        """
        body = (self.assertion.get("cues") or {}).get(assertion_class)
        if body is None:
            return [], []
        if isinstance(body, list):
            return list(body), []
        return list(body.get("pre") or []), list(body.get("post") or [])

    def confidence_for(self, key: str, default: float = 0.5) -> float:
        return float(self.confidence.get(key, default))


@dataclass(frozen=True)
class Configs:
    """Everything loaded from `config/`, with the versions they imply."""

    catalog: "ConceptCatalog"
    extraction: "ExtractionConfig"
    semantics: Any
    extractor_version: str
    normalizer_version: str

    def __iter__(self):
        return iter(
            (self.catalog, self.extraction, self.semantics, self.extractor_version)
        )

    def terminology_versions(self) -> dict[str, str]:
        return self.semantics.dictionary_versions()


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
    semantics_path: str,
    _concepts_hash: str,
    _extraction_hash: str,
    _semantics_hash: str,
) -> Configs:
    from .semantics import CollectionSemantics

    catalog = ConceptCatalog(load_yaml(concepts_path), Path(concepts_path))
    extraction = ExtractionConfig(load_yaml(extraction_path), Path(extraction_path))
    semantics = CollectionSemantics(
        load_yaml(semantics_path), Path(semantics_path)
    )
    return Configs(
        catalog=catalog,
        extraction=extraction,
        semantics=semantics,
        extractor_version=extractor_version(concepts_path, extraction_path),
        normalizer_version=normalizer_version(concepts_path, semantics_path),
    )


def load_configs(
    concepts_path: str | Path | None = None,
    extraction_path: str | Path | None = None,
    semantics_path: str | Path | None = None,
) -> Configs:
    """Load every config file and compute the versions they imply.

    Results are cached on content hash, so editing a file in place is picked up
    without restarting the process.
    """
    cp = str(concepts_path or paths.CONCEPTS_YAML)
    ep = str(extraction_path or paths.EXTRACTION_YAML)
    sp = str(semantics_path or paths.SEMANTICS_YAML)
    return _load_cached(
        cp, ep, sp, hash_file(cp), hash_file(ep), hash_file(sp)
    )
