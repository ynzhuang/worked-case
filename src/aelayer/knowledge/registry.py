"""Registry of governed executions and published definitions.

**Forward capture by default.**  The registry accrues from executions.  On day
one it is empty, and nothing here pretends otherwise.  Historical backfill is a
scoped, deliberate task — ``aelayer knowledge backfill --manifest <file>`` — and
no code path implies that retrospective reconstruction happens for free.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .. import paths
from ..models import Manifest, PhenotypeDefinition
from ..runs import ManifestStore


class ScopeRequired(ValueError):
    """Raised when a program-wide sweep is attempted without a stated scope."""


@dataclass
class KnowledgeRegistry:
    """What has been run, and against which definitions."""

    manifests: ManifestStore
    definitions: Any = None

    @classmethod
    def open(
        cls, runs_dir: str | Path | None = None, definitions: Any = None
    ) -> "KnowledgeRegistry":
        return cls(ManifestStore(Path(runs_dir or paths.RUNS_DIR)), definitions)

    # -- executions ---------------------------------------------------------

    def all_manifests(self) -> list[Manifest]:
        return self.manifests.list()

    def status(self) -> dict[str, Any]:
        """What the registry actually holds, stated plainly.

        An empty registry is the expected state of a new deployment, not a
        fault, and the message says so rather than reading as an error.
        """
        recorded = self.all_manifests()
        definitions = list(self.definitions.all()) if self.definitions else []
        return {
            "manifests": len(recorded),
            "definitions": [d.key for d in definitions],
            "capture_mode": "forward",
            "note": (
                "The registry accrues from governed executions. It is empty "
                "until something has been run, and historical work is added "
                "only by an explicit backfill."
                if not recorded else
                f"{len(recorded)} governed execution(s) recorded. Historical "
                f"work before this registry existed is present only where it "
                f"was explicitly backfilled."
            ),
            "definitions_used": sorted(
                {f"{m.definition_id}.v{m.definition_version}"
                 for m in recorded}
            ),
            "snapshots": sorted({m.data_snapshot_id for m in recorded}),
        }

    def find(
        self,
        *,
        definition_id: str | None = None,
        snapshot_id: str | None = None,
        actor: str | None = None,
    ) -> list[Manifest]:
        results = self.all_manifests()
        if definition_id:
            results = [m for m in results if m.definition_id == definition_id]
        if snapshot_id:
            results = [m for m in results if m.data_snapshot_id == snapshot_id]
        if actor:
            results = [m for m in results if m.actor == actor]
        return results

    def backfill(self, manifest: Manifest) -> Path:
        """Add a historical execution, deliberately and one at a time."""
        return self.manifests.save(manifest)

    # -- definitions --------------------------------------------------------

    def definition_versions(self, definition_id: str) -> list[PhenotypeDefinition]:
        if self.definitions is None:
            return []
        return [d for d in self.definitions.all() if d.id == definition_id]

    def require_scope(self, scope: str | None, *, action: str) -> str:
        """Refuse an unscoped sweep.

        The comparison capability exists so that evidence can be reused, not so
        that a colleague's past choices can be audited in bulk. A caller has to
        name the scientific question or phenotype family it applies to.
        """
        if not scope or not scope.strip():
            raise ScopeRequired(
                f"{action} requires an explicit scope — a scientific question or "
                f"a phenotype family. This capability is for evidence reuse "
                f"within a stated question, not for sweeping a whole programme "
                f"to audit past choices."
            )
        return scope.strip()
