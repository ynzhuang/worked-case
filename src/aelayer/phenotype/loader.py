"""Load, validate, hash and version phenotype definitions.

A phenotype definition is a versioned scientific artifact with its own
lifecycle:

* ``draft``      — being written; will not run in a reproducible run without an
                   explicit opt-in flag, because its content can still change.
* ``frozen``     — published; the loader refuses to overwrite the file.
* ``superseded`` — replaced by a later version; still loadable and still
                   replayable, because a prior analysis was built on it.

A v2 is a new file, never an edit of v1.  Changing what qualifies as a case
must never rewrite the cohort a prior analysis was built on.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from pydantic import ValidationError

from .. import paths
from ..catalog import ConceptCatalog, ConfigError, load_yaml
from ..hashing import hash_file, hash_payload
from ..models import (
    ACTION_TAKEN_VALUES,
    ASSERTION_VALUES,
    COLLECTION_STATES,
    EVIDENCE_STATE_VALUES,
    OUTCOME_VALUES,
    RELATEDNESS_VALUES,
    SERIOUSNESS_CRITERIA,
    SEVERITY_VALUES,
    PhenotypeDefinition,
)

FILENAME_RE = re.compile(r"^(?P<id>[a-z0-9_]+)\.v(?P<version>\d+)\.yaml$")

#: Comparison operators a `lab` predicate may use.
LAB_OPS = ("<", "<=", ">", ">=", "==", "!=")

#: Every leaf predicate the rule language accepts, with the validator for its
#: body.  An unknown key is a config error, not a silently-false rule: a
#: definition that fails validation does not run.
_ENUM_PREDICATES: dict[str, tuple[str, ...]] = {
    "assertion": ASSERTION_VALUES,
    # Treatment action is a valid *attribute* to filter on, and deliberately
    # not part of the shipped hypoglycemia definition: gating a case
    # definition on what the site did to the dose imports a field the clinical
    # question never referenced, and one some studies cannot even express.
    "action_taken": ACTION_TAKEN_VALUES,
    "peak_severity": SEVERITY_VALUES,
    "seriousness_criteria": SERIOUSNESS_CRITERIA,
    "outcome": OUTCOME_VALUES,
    "relatedness": RELATEDNESS_VALUES,
}
_BOOL_PREDICATES = (
    "coded_term_matches_concept",
    "has_coded_term",
    "seriousness",
    "linkage_review_required",
)
_COMBINATORS = ("any", "all", "not")


class DefinitionError(ValueError):
    """Raised when a definition is invalid, missing, or used against its status."""


# --------------------------------------------------------------------------
# Rule language validation
# --------------------------------------------------------------------------


def validate_condition(
    condition: Any, catalog: ConceptCatalog | None, *, where: str
) -> None:
    """Recursively validate one ``when`` block of the rule language."""
    if isinstance(condition, list):
        for index, item in enumerate(condition):
            validate_condition(item, catalog, where=f"{where}[{index}]")
        return
    if not isinstance(condition, dict):
        raise DefinitionError(f"{where}: expected a mapping or list, got {condition!r}")
    if not condition:
        raise DefinitionError(f"{where}: empty condition")

    for key, body in condition.items():
        path = f"{where}.{key}"
        if key in _COMBINATORS:
            if key == "not":
                validate_condition(body, catalog, where=path)
            else:
                if not isinstance(body, list) or not body:
                    raise DefinitionError(f"{path}: `{key}` needs a non-empty list")
                validate_condition(body, catalog, where=path)
            continue

        if key in _BOOL_PREDICATES:
            if not isinstance(body, bool):
                raise DefinitionError(f"{path}: expected true or false, got {body!r}")
            continue

        if key == "lexicon_match":
            if not isinstance(body, dict):
                raise DefinitionError(f"{path}: expected a mapping")
            unknown = set(body) - {"assertion", "concept"}
            if unknown:
                raise DefinitionError(f"{path}: unknown keys {sorted(unknown)}")
            _check_enum_values(body.get("assertion"), ASSERTION_VALUES, path)
            continue

        if key == "collection_state":
            if not isinstance(body, dict) or "field" not in body:
                raise DefinitionError(f"{path}: expected a mapping with `field`")
            unknown = set(body) - {"field", "is"}
            if unknown:
                raise DefinitionError(f"{path}: unknown keys {sorted(unknown)}")
            _check_enum_values(body.get("is"), COLLECTION_STATES, path)
            continue

        if key == "lab":
            _validate_lab(body, catalog, path)
            continue

        if key == "symptoms":
            _validate_symptoms(body, catalog, path)
            continue

        if key == "onset_offset_days":
            if not isinstance(body, dict):
                raise DefinitionError(f"{path}: expected a mapping with min/max")
            unknown = set(body) - {"min", "max"}
            if unknown:
                raise DefinitionError(f"{path}: unknown keys {sorted(unknown)}")
            continue

        if key in _ENUM_PREDICATES:
            _check_enum_values(body, _ENUM_PREDICATES[key], path)
            continue

        raise DefinitionError(
            f"{path}: unknown predicate {key!r}. "
            f"Known predicates: {sorted(_known_predicates())}"
        )


def _known_predicates() -> Iterable[str]:
    return (
        list(_COMBINATORS)
        + list(_BOOL_PREDICATES)
        + list(_ENUM_PREDICATES)
        + ["lexicon_match", "lab", "symptoms", "onset_offset_days",
           "collection_state"]
    )


def _check_enum_values(value: Any, allowed: tuple[str, ...], path: str) -> None:
    if value is None:
        return
    values = value if isinstance(value, list) else [value]
    bad = [v for v in values if v not in allowed]
    if bad:
        raise DefinitionError(f"{path}: {bad} not in {list(allowed)}")


def _validate_lab(body: Any, catalog: ConceptCatalog | None, path: str) -> None:
    if not isinstance(body, dict):
        raise DefinitionError(f"{path}: expected a mapping")
    missing = {"test", "op", "value"} - set(body)
    if missing:
        raise DefinitionError(f"{path}: missing {sorted(missing)}")
    unknown = set(body) - {"test", "op", "value", "unit"}
    if unknown:
        raise DefinitionError(f"{path}: unknown keys {sorted(unknown)}")
    if body["op"] not in LAB_OPS:
        raise DefinitionError(f"{path}.op: {body['op']!r} not in {list(LAB_OPS)}")
    if not isinstance(body["value"], (int, float)):
        raise DefinitionError(f"{path}.value: expected a number")
    if catalog is not None:
        test = body["test"]
        if test not in catalog.lab_tests:
            raise DefinitionError(
                f"{path}.test: unknown lab test {test!r}; "
                f"known: {sorted(catalog.lab_tests)}"
            )
        unit = body.get("unit")
        lab = catalog.lab_tests[test]
        if unit is not None and unit not in lab.conversions:
            raise DefinitionError(
                f"{path}.unit: {unit!r} has no conversion for {test!r}; "
                f"known: {sorted(lab.conversions)}"
            )


def _validate_symptoms(body: Any, catalog: ConceptCatalog | None, path: str) -> None:
    if not isinstance(body, dict):
        raise DefinitionError(f"{path}: expected a mapping")
    unknown = set(body) - {"min_count", "from", "any_of"}
    if unknown:
        raise DefinitionError(f"{path}: unknown keys {sorted(unknown)}")
    if "from" not in body and "any_of" not in body:
        raise DefinitionError(f"{path}: needs `from` (symptom sets) or `any_of`")
    min_count = body.get("min_count", 1)
    if not isinstance(min_count, int) or min_count < 1:
        raise DefinitionError(f"{path}.min_count: expected a positive integer")
    if catalog is not None and "from" in body:
        unknown_sets = [s for s in body["from"] if s not in catalog.symptom_sets]
        if unknown_sets:
            raise DefinitionError(
                f"{path}.from: unknown symptom sets {unknown_sets}; "
                f"known: {sorted(catalog.symptom_sets)}"
            )


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_definition(
    path: str | Path, catalog: ConceptCatalog | None = None
) -> PhenotypeDefinition:
    """Load one definition file, validate it, and stamp its content hash."""
    path = Path(path)
    if not path.exists():
        raise DefinitionError(f"no definition file at {path}")
    raw = load_yaml(path)
    if not isinstance(raw, dict):
        raise DefinitionError(f"{path}: definition must be a mapping")

    body = {k: v for k, v in raw.items() if k not in ("definition_hash", "source_path")}
    try:
        definition = PhenotypeDefinition.model_validate(body)
    except ValidationError as exc:
        raise DefinitionError(f"{path}: {_format_validation_error(exc)}") from exc

    match = FILENAME_RE.match(path.name)
    if match:
        if match.group("id") != definition.id:
            raise DefinitionError(
                f"{path}: filename id {match.group('id')!r} does not match "
                f"definition id {definition.id!r}"
            )
        if int(match.group("version")) != definition.version:
            raise DefinitionError(
                f"{path}: filename version v{match.group('version')} does not match "
                f"declared version {definition.version}"
            )

    if catalog is not None:
        if definition.concept.primary not in catalog.concepts:
            raise DefinitionError(
                f"{path}: unknown primary concept {definition.concept.primary!r}"
            )
        if definition.concept.group is not None:
            catalog.expand_group(definition.concept.group)

    for rule in definition.evidence_rules:
        validate_condition(
            rule.when, catalog, where=f"{path.name}:evidence_rules.{rule.id}.when"
        )

    if definition.operates_on != "episode":
        raise DefinitionError(
            f"{path}: definitions operate on episodes; got "
            f"{definition.operates_on!r}"
        )
    referenced = {r.state for r in definition.evidence_rules}
    declared = set(
        definition.case_definition.primary
        + definition.case_definition.review
        + definition.case_definition.excluded
    )
    undeclared = referenced - declared
    if undeclared:
        raise DefinitionError(
            f"{path}: evidence rules assign states {sorted(undeclared)} that the "
            f"case_definition never places in primary_set, review_set or excluded"
        )
    if definition.window is not None and definition.anchor is None:
        raise DefinitionError(
            f"{path}: a window is defined but no anchor; an offset needs "
            f"something to be relative to"
        )

    definition.definition_hash = hash_file(path)
    definition.source_path = str(path)
    return definition


def _format_validation_error(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"{loc}: {err['msg']}")
    return "; ".join(lines)


def definition_content_hash(definition: PhenotypeDefinition) -> str:
    """Hash of the definition's semantic content, ignoring the source file.

    Used to detect whether an edited candidate actually changes behaviour.
    """
    payload = definition.model_dump(
        mode="json", exclude={"definition_hash", "source_path", "created", "authors"}
    )
    return hash_payload(payload)


# --------------------------------------------------------------------------
# Catalogue
# --------------------------------------------------------------------------


class DefinitionCatalog:
    """All definition versions found in a directory."""

    def __init__(
        self,
        directory: str | Path | None = None,
        catalog: ConceptCatalog | None = None,
    ):
        self.directory = Path(directory or paths.PHENOTYPE_DIR)
        self.concept_catalog = catalog
        self._cache: dict[tuple[str, int], PhenotypeDefinition] = {}

    def paths(self) -> list[Path]:
        if not self.directory.exists():
            return []
        return sorted(p for p in self.directory.glob("*.yaml") if FILENAME_RE.match(p.name))

    def all(self) -> list[PhenotypeDefinition]:
        out = []
        for path in self.paths():
            out.append(self.load_path(path))
        return sorted(out, key=lambda d: (d.id, d.version))

    def load_path(self, path: Path) -> PhenotypeDefinition:
        match = FILENAME_RE.match(path.name)
        key = (match.group("id"), int(match.group("version"))) if match else None
        if key and key in self._cache:
            return self._cache[key]
        definition = load_definition(path, self.concept_catalog)
        self._cache[(definition.id, definition.version)] = definition
        return definition

    def versions(self, definition_id: str) -> list[int]:
        return sorted(d.version for d in self.all() if d.id == definition_id)

    def get(
        self,
        definition_id: str,
        version: int | None = None,
        *,
        allow_draft: bool = False,
    ) -> PhenotypeDefinition:
        """Fetch a definition by id and version.

        ``version=None`` selects the highest non-draft version, falling back to
        the highest draft only when ``allow_draft`` is set.  A draft never
        becomes the default for a reproducible run by accident.
        """
        candidates = [d for d in self.all() if d.id == definition_id]
        if not candidates:
            known = sorted({d.id for d in self.all()})
            raise DefinitionError(
                f"no definition with id {definition_id!r}; known: {known}"
            )
        if version is not None:
            for definition in candidates:
                if definition.version == version:
                    self._check_runnable(definition, allow_draft)
                    return definition
            raise DefinitionError(
                f"{definition_id} has no version {version}; "
                f"available: {[d.version for d in candidates]}"
            )

        publishable = [d for d in candidates if d.status != "draft"]
        pool = publishable or (candidates if allow_draft else [])
        if not pool:
            raise DefinitionError(
                f"{definition_id} has only draft versions. Pass allow_draft to run "
                f"one, and know that the run is not reproducible: a draft's "
                f"content can still change under the same version number."
            )
        chosen = max(pool, key=lambda d: d.version)
        self._check_runnable(chosen, allow_draft)
        return chosen

    @staticmethod
    def _check_runnable(definition: PhenotypeDefinition, allow_draft: bool) -> None:
        if definition.status == "draft" and not allow_draft:
            raise DefinitionError(
                f"{definition.key} is a draft. A draft's content can change under "
                f"the same version number, so it will not run in a reproducible "
                f"run without an explicit opt-in (--allow-draft)."
            )

    def next_version(self, definition_id: str) -> int:
        versions = self.versions(definition_id)
        return (max(versions) + 1) if versions else 1

    def path_for(self, definition_id: str, version: int) -> Path:
        return self.directory / f"{definition_id}.v{version}.yaml"

    def write_candidate(
        self,
        body: dict[str, Any],
        *,
        directory: str | Path | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Write a new definition version to disk.

        Refuses to overwrite an existing frozen definition under any
        circumstances.  A published definition is the record a prior analysis
        rests on; editing it in place would silently rewrite that cohort.
        """
        import yaml

        target_dir = Path(directory or self.directory)
        definition_id = body.get("id")
        version = body.get("version")
        if not definition_id or not isinstance(version, int):
            raise DefinitionError("a candidate definition needs `id` and integer `version`")
        target = target_dir / f"{definition_id}.v{version}.yaml"

        if target.exists():
            existing = load_definition(target, self.concept_catalog)
            if existing.status == "frozen":
                raise DefinitionError(
                    f"{target.name} is frozen and will not be overwritten. "
                    f"Create v{self.next_version(definition_id)} instead: a change "
                    f"to what qualifies as a case is a new version, not an edit."
                )
            if not overwrite:
                raise DefinitionError(
                    f"{target.name} already exists (status={existing.status}). "
                    f"Pass overwrite=True to replace a draft."
                )

        target_dir.mkdir(parents=True, exist_ok=True)
        target.write_text(
            yaml.safe_dump(body, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        # Validate what was actually written; a candidate that does not load is
        # not a candidate.
        load_definition(target, self.concept_catalog)
        self._cache.pop((definition_id, version), None)
        return target


# --------------------------------------------------------------------------
# Diffing
# --------------------------------------------------------------------------


def diff_definitions(
    left: PhenotypeDefinition, right: PhenotypeDefinition
) -> list[dict[str, Any]]:
    """Field-level differences between two definitions, for the version badge."""
    a = left.model_dump(mode="json", exclude={"definition_hash", "source_path"})
    b = right.model_dump(mode="json", exclude={"definition_hash", "source_path"})
    changes: list[dict[str, Any]] = []

    def walk(prefix: str, x: Any, y: Any) -> None:
        if isinstance(x, dict) and isinstance(y, dict):
            for key in sorted(set(x) | set(y)):
                walk(f"{prefix}.{key}" if prefix else key, x.get(key), y.get(key))
            return
        if isinstance(x, list) and isinstance(y, list):
            # Descend into lists element by element so a single changed
            # threshold reads as one line rather than two dumped rule lists.
            # Where elements carry an `id`, they are matched on it, so
            # reordering rules is not reported as a change to every rule.
            if _keyed_by_id(x) and _keyed_by_id(y):
                left = {item["id"]: item for item in x}
                right = {item["id"]: item for item in y}
                for key in sorted(set(left) | set(right)):
                    walk(f"{prefix}[{key}]", left.get(key), right.get(key))
            else:
                for index in range(max(len(x), len(y))):
                    walk(
                        f"{prefix}[{index}]",
                        x[index] if index < len(x) else None,
                        y[index] if index < len(y) else None,
                    )
            return
        if x != y:
            changes.append({"path": prefix, "from": x, "to": y})

    walk("", a, b)
    return changes


def _keyed_by_id(items: list[Any]) -> bool:
    return bool(items) and all(
        isinstance(item, dict) and isinstance(item.get("id"), str) for item in items
    )
