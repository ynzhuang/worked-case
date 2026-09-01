"""One assembled pipeline, shared by the CLI, the API and the agent.

Order of work, and it does not vary:

``ingest`` -> ``normalize`` (deterministic) -> ``extract`` (model path, unresolved
fields only) -> ``reconcile`` (episodes) -> ``evaluate`` (phenotype).

Having a single path means the CLI and the API cannot disagree about which
definition version, which normalizer or which extraction backend produced a
number.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from pathlib import Path
from typing import Any, Sequence

from . import paths
from .catalog import Configs, load_configs
from .episode import EpisodeReconciler, ReconciliationConfig
from .extract.engine import ExtractionEngine
from .ingest import TrialStore, load_store
from .models import CanonicalAEEpisode, CanonicalAERecord, CaseAssignment, PhenotypeDefinition
from .normalize import normalize_store
from .phenotype.evaluator import PhenotypeEvaluator
from .phenotype.loader import DefinitionCatalog


@dataclass
class Pipeline:
    store: TrialStore
    configs: Configs
    definitions: DefinitionCatalog
    backend_preference: str = "auto"
    store_path: Path | None = None
    _records: list[CanonicalAERecord] | None = _dc_field(default=None, repr=False)
    _episodes: list[CanonicalAEEpisode] | None = _dc_field(default=None, repr=False)
    _engine: ExtractionEngine | None = _dc_field(default=None, repr=False)
    _index: Any = _dc_field(default=None, repr=False)

    # -- construction -------------------------------------------------------

    @classmethod
    def load(
        cls,
        data_dir: str | Path | None = None,
        *,
        concepts_path: str | Path | None = None,
        extraction_path: str | Path | None = None,
        semantics_path: str | Path | None = None,
        phenotype_dir: str | Path | None = None,
        store_path: str | Path | None = None,
        backend: str = "auto",
    ) -> "Pipeline":
        configs = load_configs(concepts_path, extraction_path, semantics_path)
        return cls(
            store=load_store(data_dir),
            configs=configs,
            definitions=DefinitionCatalog(phenotype_dir, configs.catalog),
            backend_preference=backend,
            store_path=Path(store_path) if store_path else None,
        )

    # -- convenience accessors ---------------------------------------------

    @property
    def catalog(self):
        return self.configs.catalog

    @property
    def semantics(self):
        return self.configs.semantics

    @property
    def snapshot_id(self) -> str:
        return self.store.snapshot_id

    @property
    def normalizer_version(self) -> str:
        return self.configs.normalizer_version

    @property
    def extractor_version(self) -> str:
        return self.configs.extractor_version

    # -- the pipeline -------------------------------------------------------

    def engine(self) -> ExtractionEngine:
        if self._engine is None:
            self._engine = ExtractionEngine.build(
                self.configs.catalog, self.configs.extraction,
                self.configs.extractor_version, self.backend_preference,
            )
        return self._engine

    def records(self, *, refresh: bool = False) -> list[CanonicalAERecord]:
        """Normalized records, enriched from narrative where unresolved."""
        if self._records is not None and not refresh:
            return self._records
        records = normalize_store(self.store, self.configs)
        self._records = self.engine().enrich_all(records, self.store.narratives)
        self._episodes = None
        return self._records

    def episodes(self, *, refresh: bool = False) -> list[CanonicalAEEpisode]:
        if self._episodes is not None and not refresh:
            return self._episodes
        reconciler = EpisodeReconciler(
            self.configs.catalog, self.configs.semantics,
            ReconciliationConfig.from_catalog(self.configs.catalog),
        )
        self._episodes = reconciler.reconcile(self.records(refresh=refresh))
        return self._episodes

    def definition(
        self, definition_id: str, version: int | None = None, *,
        allow_draft: bool = False,
    ) -> PhenotypeDefinition:
        return self.definitions.get(definition_id, version, allow_draft=allow_draft)

    def evaluator(self, definition: PhenotypeDefinition) -> PhenotypeEvaluator:
        return PhenotypeEvaluator(
            definition, self.configs.catalog,
            self.store.anchor_resolver(self.configs.extraction.anchors),
        )

    def evaluate(
        self,
        definition: PhenotypeDefinition,
        studies: Sequence[str] | None = None,
        episodes: Sequence[CanonicalAEEpisode] | None = None,
    ) -> list[CaseAssignment]:
        source = list(episodes) if episodes is not None else self.episodes()
        if studies:
            allowed = set(studies)
            source = [e for e in source if e.study_id in allowed]
        return self.evaluator(definition).evaluate(source)

    def cohort(self, studies: Sequence[str] | None = None) -> list[tuple[str, str]]:
        pairs = [(s, self.store.study_of(s) or "") for s in self.store.subjects()]
        if studies:
            allowed = set(studies)
            pairs = [p for p in pairs if p[1] in allowed]
        return pairs

    # -- retrieval ----------------------------------------------------------

    def index(self, *, refresh: bool = False, persist: bool = True):
        from .retrieval.index import build_index

        if self._index is not None and not refresh:
            return self._index
        self._index = build_index(
            self.store_path if persist else None,
            self.store, self.episodes(refresh=refresh),
            self.configs.extractor_version, self.configs.normalizer_version,
        )
        return self._index

    def retrieve(self, **kwargs: Any):
        from .retrieval.query import retrieve

        return retrieve(self.index(), self.configs.catalog, **kwargs)

    # -- provenance ---------------------------------------------------------

    def versions(self) -> dict[str, Any]:
        engine = self.engine()
        return {
            "normalizer_version": self.configs.normalizer_version,
            "extractor_version": self.configs.extractor_version,
            "extraction_backend": engine.backend.name,
            "model_version": getattr(engine.backend, "model_version", None),
            "snapshot_id": self.snapshot_id,
            "terminology_versions": self.configs.terminology_versions(),
        }

    def summary(self) -> dict[str, Any]:
        return {
            **self.store.summary(),
            **self.versions(),
            "records": len(self.records()),
            "episodes": len(self.episodes()),
            "definitions": [d.key for d in self.definitions.all()],
            "backend_notes": list(self.engine().notes),
        }
