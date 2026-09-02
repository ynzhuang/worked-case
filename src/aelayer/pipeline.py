"""One assembled pipeline, shared by the CLI, the API and the agent.

Order of work, and it does not vary:

``ingest`` -> ``normalize`` (deterministic) -> ``extract`` (model path, on
unresolved attributes only) -> ``reconcile`` (episodes) -> ``trajectory`` ->
``evaluate`` (phenotype).

One path means the CLI and the API cannot disagree about which definition
version, which normalizer or which extraction backend produced a number.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from pathlib import Path
from typing import Any, Sequence

from . import paths
from .anchors import AnchorResolver
from .catalog import Configs, load_configs
from .episode import EpisodeReconciler, ReconciliationConfig
from .extract.engine import ExtractionEngine
from .ingest import TrialStore, load_store
from .models import (
    CanonicalAEEpisode,
    CanonicalAERecord,
    CaseAssignment,
    PhenotypeDefinition,
    Trajectory,
)
from .normalize import normalize_store
from .phenotype.evaluator import PhenotypeEvaluator
from .phenotype.loader import DefinitionCatalog
from .trajectory import build_trajectories


@dataclass
class Pipeline:
    store: TrialStore
    configs: Configs
    definitions: DefinitionCatalog
    backend_preference: str = "auto"
    store_path: Path | None = None
    _records: list[CanonicalAERecord] | None = _dc_field(default=None, repr=False)
    _episodes: list[CanonicalAEEpisode] | None = _dc_field(default=None, repr=False)
    _trajectories: dict[str, Trajectory] | None = _dc_field(default=None, repr=False)
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
        profiles_path: str | Path | None = None,
        phenotype_dir: str | Path | None = None,
        store_path: str | Path | None = None,
        backend: str = "auto",
    ) -> "Pipeline":
        configs = load_configs(concepts_path, extraction_path, profiles_path)
        return cls(
            store=load_store(data_dir),
            configs=configs,
            definitions=DefinitionCatalog(phenotype_dir, configs.catalog),
            backend_preference=backend,
            store_path=Path(store_path) if store_path else None,
        )

    # -- accessors ----------------------------------------------------------

    @property
    def catalog(self):
        return self.configs.catalog

    @property
    def profiles(self):
        return self.configs.profiles

    @property
    def snapshot_id(self) -> str:
        return self.store.snapshot_id

    @property
    def normalizer_version(self) -> str:
        return self.configs.normalizer_version

    @property
    def extractor_version(self) -> str:
        return self.configs.extractor_version

    def anchor_resolver(self) -> AnchorResolver:
        return self.store.anchor_resolver(self.configs.extraction.anchors)

    # -- the pipeline -------------------------------------------------------

    def engine(self) -> ExtractionEngine:
        if self._engine is None:
            self._engine = ExtractionEngine.build(
                self.configs, self.store, self.backend_preference
            )
        return self._engine

    def records(self, *, refresh: bool = False) -> list[CanonicalAERecord]:
        """Normalized records, enriched from text where the study kept it there."""
        if self._records is not None and not refresh:
            return self._records
        records = normalize_store(self.store, self.configs)
        self._records = self.engine().enrich_all(records)
        self._episodes = None
        self._trajectories = None
        return self._records

    def structured_only_records(self) -> list[CanonicalAERecord]:
        """The same records with the model path never run.

        This is the comparator for the value ablation: what the layer would
        have without text recovery at all.
        """
        return normalize_store(self.store, self.configs)

    def episodes(self, *, refresh: bool = False) -> list[CanonicalAEEpisode]:
        if self._episodes is not None and not refresh:
            return self._episodes
        self._episodes = self.reconcile(self.records(refresh=refresh))
        return self._episodes

    def reconcile(
        self, records: Sequence[CanonicalAERecord]
    ) -> list[CanonicalAEEpisode]:
        reconciler = EpisodeReconciler(
            self.configs.catalog, self.configs.profiles,
            ReconciliationConfig.from_catalog(self.configs.catalog),
            anchor_resolver=self.anchor_resolver(),
            default_anchor=self.configs.extraction.default_anchor,
        )
        return reconciler.reconcile(records)

    def trajectories(self, *, refresh: bool = False) -> dict[str, Trajectory]:
        if self._trajectories is not None and not refresh:
            return self._trajectories
        self._trajectories = build_trajectories(
            self.episodes(refresh=refresh), self.store, self.anchor_resolver(),
            self.configs.extraction.default_anchor or "first_exposure",
        )
        return self._trajectories

    # -- definitions and evaluation ----------------------------------------

    def definition(
        self, definition_id: str, version: int | None = None, *,
        allow_draft: bool = False,
    ) -> PhenotypeDefinition:
        return self.definitions.get(definition_id, version, allow_draft=allow_draft)

    def evaluator(self, definition: PhenotypeDefinition) -> PhenotypeEvaluator:
        return PhenotypeEvaluator(definition, self.configs.catalog)

    def evaluate(
        self, definition: PhenotypeDefinition,
        studies: Sequence[str] | None = None,
        episodes: Sequence[CanonicalAEEpisode] | None = None,
    ) -> list[CaseAssignment]:
        pool = list(episodes if episodes is not None else self.episodes())
        if studies:
            allowed = set(studies)
            pool = [e for e in pool if e.study_id in allowed]
        return self.evaluator(definition).evaluate(pool)

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
            self.mentions(),
        )
        return self._index

    def mentions(self) -> list[dict[str, Any]]:
        """Modifier mentions in free text, for the discovery path.

        Episodes are built from records, not from mentions: a mention is a place
        in a document where something is named, which is a different thing from
        an event having occurred.
        """
        backend = self.engine().backend
        extractor = getattr(backend, "modifiers", None)
        if extractor is None:
            return []
        found: list[dict[str, Any]] = []
        for record in self.records():
            for doc_id, text, source_kind, variable in self.engine().sources_for(
                record, self.configs.profiles.for_study(record.study_id)
            ):
                for attribute in ("location", "pattern"):
                    for hit in extractor.find(
                        text, attribute, record.standardized_concept, source_kind
                    ):
                        found.append({
                            "mention_id": f"{doc_id}:{hit.start}:{hit.value}",
                            "doc_id": doc_id,
                            "study_id": record.study_id,
                            "subject_id": record.subject_id,
                            "source_record_id": record.source_record_id,
                            "profile": record.profile,
                            "attribute": attribute,
                            "value": hit.value,
                            "surface": hit.surface,
                            "sentence": text,
                            "source_kind": source_kind,
                            "source_variable": variable,
                            "confidence": hit.confidence,
                            "rule": hit.rule,
                            "normalized": True,
                        })
                for hit in extractor.qualities(text):
                    found.append({
                        "mention_id": f"{doc_id}:{hit.start}:{hit.value}",
                        "doc_id": doc_id,
                        "study_id": record.study_id,
                        "subject_id": record.subject_id,
                        "source_record_id": record.source_record_id,
                        "profile": record.profile,
                        "attribute": "quality",
                        "value": hit.value,
                        "surface": hit.surface,
                        "sentence": text,
                        "source_kind": source_kind,
                        "source_variable": variable,
                        "confidence": hit.confidence,
                        "rule": hit.rule,
                        # A quality descriptor is not in any catalogue value
                        # space, which is exactly what makes it worth surfacing
                        # on the discovery path.
                        "normalized": False,
                    })
        return found

    def retrieve(self, **kwargs: Any):
        from .retrieval.query import retrieve

        return retrieve(self.index(), self.configs.catalog, **kwargs)

    # -- provenance ---------------------------------------------------------

    def versions(self) -> dict[str, Any]:
        engine = self.engine()
        return {
            "normalizer_version": self.configs.normalizer_version,
            "extractor_version": self.configs.extractor_version,
            "snapshot_id": self.snapshot_id,
            "terminology_versions": self.configs.terminology_versions(),
            **engine.versions(),
        }

    def summary(self) -> dict[str, Any]:
        return {
            **self.store.summary(),
            **self.versions(),
            "records": len(self.records()),
            "episodes": len(self.episodes()),
            "profiles": self.configs.profiles.profile_ids(),
            "definitions": [d.key for d in self.definitions.all()],
            "backend_notes": list(self.engine().notes),
        }
