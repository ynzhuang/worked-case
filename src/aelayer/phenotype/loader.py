"""Loading, validating and versioning phenotype definitions.

A definition is a scientific artifact with a version and a content hash. Frozen
versions are never edited: a change to what qualifies as a case is a new
version, not an edit, and the loader refuses to overwrite a frozen file.

Validation is strict on purpose. A requirement that names an attribute nothing
produces, or a value outside the catalogue, is a definition that will silently
select nobody — and silence is the failure mode worth spending code on.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from pydantic import ValidationError

from .. import paths
from ..catalog import ConceptCatalog, load_yaml
from ..hashing import hash_file, hash_payload
from ..models import METHODS, SOURCE_KINDS, VERDICTS, PhenotypeDefinition

FILENAME_RE = re.compile(r"^(?P<id>[a-z0-9_]+)\.v(?P<version>\d+)\.ya?ml$")

#: Criteria a definition may name in a temporal rule's anchor. An anchor that
#: nothing resolves is a definition that silently reviews everybody.
KNOWN_ANCHORS = ("first_exposure", "dose_escalation")


class DefinitionError(ValueError):
    """Raised when a definition file is invalid or cannot be run."""


def validate_definition(
    definition: PhenotypeDefinition, catalog: ConceptCatalog | None, where: str
) -> None:
    """Everything the schema cannot express on its own.

    Strict on purpose. A requirement naming a modifier nothing produces, or a
    concept outside the catalogue, is a definition that will silently select
    nobody — and silence is the failure mode worth spending code on.
    """
    if catalog is not None:
        unknown = sorted(
            c for c in (*definition.concept_set.include, *definition.concept_set.exclude)
            if c not in catalog.concepts
        )
        if unknown:
            raise DefinitionError(
                f"{where}: concept set names {unknown}, which the catalogue does "
                f"not define, so nothing could ever match them"
            )
        target = definition.concept_set.dictionary_target
        if target is not None and target not in catalog.dictionary_versions:
            raise DefinitionError(
                f"{where}: dictionary_target {target!r} is not a version this "
                f"catalogue knows ({list(catalog.dictionary_versions)})"
            )

    for requirement in definition.modifiers:
        if catalog is not None and requirement.name not in catalog.modifiers:
            raise DefinitionError(
                f"{where}: requirement {requirement.name!r} is not a configured "
                f"modifier; known: {sorted(catalog.modifiers)}"
            )
        unknown_methods = sorted(set(requirement.accept_methods) - set(METHODS))
        if unknown_methods:
            raise DefinitionError(
                f"{where}: requirement {requirement.name!r} accepts unknown "
                f"methods {unknown_methods}; known: {list(METHODS)}"
            )
        if "derived" in requirement.accept_methods:
            raise DefinitionError(
                f"{where}: requirement {requirement.name!r} accepts method "
                f"'derived', but a modifier is read from a source, not computed "
                f"across domains"
            )
        if requirement.accept_sources is not None:
            unknown_sources = sorted(
                set(requirement.accept_sources) - set(SOURCE_KINDS)
            )
            if unknown_sources:
                raise DefinitionError(
                    f"{where}: requirement {requirement.name!r} names unknown "
                    f"source kinds {unknown_sources}"
                )

    if definition.temporal is not None:
        if definition.temporal.anchor not in KNOWN_ANCHORS:
            raise DefinitionError(
                f"{where}: temporal anchor {definition.temporal.anchor!r} is not "
                f"one of {list(KNOWN_ANCHORS)}; an offset needs something real "
                f"to be relative to"
            )

    if set(definition.verdicts) != set(VERDICTS):
        raise DefinitionError(
            f"{where}: the verdicts block declares {sorted(definition.verdicts)}, "
            f"but the evaluator returns {sorted(VERDICTS)}"
        )


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
        raise DefinitionError(f"{path}: {_format(exc)}") from exc

    match = FILENAME_RE.match(path.name)
    if match:
        if match.group("id") != definition.id:
            raise DefinitionError(
                f"{path}: filename id {match.group('id')!r} does not match "
                f"definition id {definition.id!r}"
            )
        if int(match.group("version")) != definition.version:
            raise DefinitionError(
                f"{path}: filename version v{match.group('version')} does not "
                f"match declared version {definition.version}"
            )

    validate_definition(definition, catalog, str(path))
    definition.definition_hash = hash_file(path)
    definition.source_path = str(path)
    return definition


def _format(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
        for err in exc.errors()
    )


def definition_content_hash(definition: PhenotypeDefinition) -> str:
    """Hash of the definition's meaning, independent of file formatting."""
    return hash_payload(
        definition.model_dump(
            mode="json", exclude={"definition_hash", "source_path", "created",
                                  "authors", "description", "label"}
        )
    )


class DefinitionCatalog:
    """Every definition on disk, by id and version."""

    def __init__(
        self, directory: str | Path | None = None,
        concept_catalog: ConceptCatalog | None = None,
    ):
        self.directory = Path(directory or paths.PHENOTYPE_DIR)
        self.concept_catalog = concept_catalog
        self._cache: dict[tuple[str, int], PhenotypeDefinition] = {}

    def paths(self) -> list[Path]:
        if not self.directory.exists():
            return []
        return sorted(
            p for p in self.directory.iterdir()
            if FILENAME_RE.match(p.name)
        )

    def all(self) -> list[PhenotypeDefinition]:
        return sorted(
            (self.load_path(p) for p in self.paths()),
            key=lambda d: (d.id, d.version),
        )

    def load_path(self, path: Path) -> PhenotypeDefinition:
        match = FILENAME_RE.match(path.name)
        key = (match.group("id"), int(match.group("version"))) if match else None
        if key and key in self._cache:
            cached = self._cache[key]
            if cached.definition_hash == hash_file(path):
                return cached
        definition = load_definition(path, self.concept_catalog)
        if key:
            self._cache[key] = definition
        return definition

    def versions(self, definition_id: str) -> list[int]:
        return sorted(d.version for d in self.all() if d.id == definition_id)

    def get(
        self, definition_id: str, version: int | None = None, *,
        allow_draft: bool = False,
    ) -> PhenotypeDefinition:
        """Fetch by id and version.

        ``version=None`` selects the highest non-draft version, so a draft never
        becomes the default for a reproducible run by accident.
        """
        candidates = [d for d in self.all() if d.id == definition_id]
        if not candidates:
            raise DefinitionError(
                f"no definition with id {definition_id!r}; known: "
                f"{sorted({d.id for d in self.all()})}"
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
                f"{definition_id} has only draft versions. Pass allow_draft to "
                f"run one, and know that the run is not reproducible: a draft's "
                f"content can change under the same version number."
            )
        chosen = max(pool, key=lambda d: d.version)
        self._check_runnable(chosen, allow_draft)
        return chosen

    @staticmethod
    def _check_runnable(definition: PhenotypeDefinition, allow_draft: bool) -> None:
        if definition.status == "draft" and not allow_draft:
            raise DefinitionError(
                f"{definition.key} is a draft. A draft's content can change "
                f"under the same version number, so it will not run in a "
                f"reproducible run without an explicit opt-in (--allow-draft)."
            )

    def next_version(self, definition_id: str) -> int:
        versions = self.versions(definition_id)
        return (max(versions) + 1) if versions else 1

    def path_for(self, definition_id: str, version: int) -> Path:
        return self.directory / f"{definition_id}.v{version}.yaml"

    def write_candidate(
        self, body: dict[str, Any], *, directory: str | Path | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Write a new definition version to disk.

        Refuses to overwrite a frozen definition under any circumstances. A
        published definition is the record a prior analysis rests on; editing it
        in place would silently rewrite that cohort.
        """
        import yaml

        target_dir = Path(directory or self.directory)
        definition_id = body.get("id")
        version = body.get("version")
        if not definition_id or not isinstance(version, int):
            raise DefinitionError(
                "a candidate definition needs `id` and an integer `version`"
            )
        target = target_dir / f"{definition_id}.v{version}.yaml"
        if target.exists():
            existing = load_definition(target, self.concept_catalog)
            if existing.status == "frozen":
                raise DefinitionError(
                    f"{target.name} is frozen and will not be overwritten. "
                    f"Create v{self.next_version(definition_id)} instead: a "
                    f"change to what qualifies as a case is a new version, not "
                    f"an edit."
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
        load_definition(target, self.concept_catalog)  # a candidate that does not load is not one
        self._cache.pop((definition_id, version), None)
        return target


def diff_definitions(
    a: PhenotypeDefinition, b: PhenotypeDefinition
) -> list[dict[str, Any]]:
    """Field-level differences between two versions, for display only.

    The comparison that matters is executed, not textual — see
    ``knowledge.diff_definitions``, which runs both against the same snapshot.
    """
    left = a.model_dump(mode="json", exclude={"definition_hash", "source_path"})
    right = b.model_dump(mode="json", exclude={"definition_hash", "source_path"})
    changes: list[dict[str, Any]] = []

    def walk(prefix: str, x: Any, y: Any) -> None:
        if isinstance(x, dict) and isinstance(y, dict):
            for key in sorted(set(x) | set(y)):
                walk(f"{prefix}.{key}" if prefix else key, x.get(key), y.get(key))
        elif x != y:
            changes.append({"path": prefix, "from": x, "to": y})

    walk("", left, right)
    return changes


def load_definitions(
    directory: str | Path | None = None, catalog: ConceptCatalog | None = None
) -> Iterable[PhenotypeDefinition]:
    return DefinitionCatalog(directory, catalog).all()
