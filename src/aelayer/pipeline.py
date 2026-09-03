"""One object that holds a snapshot, its configuration, and the derived views.

Everything below the CLI, the API and the agent goes through here, so that
there is exactly one place where "records" means the same thing.

Two properties are worth stating outright:

* The **source-record grain is primary.** ``records()`` is what phenotypes,
  denominators, the silver standard and the ablation all run over. Episodes are
  a derived view offered alongside; nothing is evaluated at that grain.
* ``structured_only_records()`` is the ablation's comparator and is produced by
  the same normalizer with the model path never run — not by deleting values
  afterwards, which would leave the record claiming a provenance it no longer
  has.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from pathlib import Path
from typing import Any, Sequence

from .anchors import AnchorResolver
from .catalog import Configs, ExtractionConfig, load_configs
from .episode import EpisodeView, group_records
from .extract import ExtractionEngine
from .ingest import TrialStore, load_store
from .models import CanonicalAERecord, CaseAssignment, PhenotypeDefinition
from .normalize import normalize_store
from .normalize.records import RecordNormalizer
from .phenotype import DefinitionCatalog, EvaluationResult, PhenotypeEvaluator


@dataclass
class Pipeline:
    store: TrialStore
    configs: Configs
    definitions: DefinitionCatalog
    backend_preference: str = "auto"
    store_path: Path | None = None
    _records: list[CanonicalAERecord] | None = _dc_field(default=None, repr=False)
    _episodes: EpisodeView | None = _dc_field(default=None, repr=False)
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

    def engine(self) -> ExtractionEngine:
        if self._engine is None:
            self._engine = ExtractionEngine.build(
                self.configs, self.store, self.backend_preference
            )
        return self._engine

    # -- the two grains -----------------------------------------------------

    def records(self, *, refresh: bool = False) -> list[CanonicalAERecord]:
        """Normalized records, enriched from text where the study kept it there.

        The primary grain. Everything that produces a number runs over this.
        """
        if self._records is not None and not refresh:
            return self._records
        self._records = self.engine().enrich_all(
            normalize_store(self.store, self.configs)
        )
        self._episodes = None
        return self._records

    def structured_only_records(self) -> list[CanonicalAERecord]:
        """The same records with the model path never run.

        Produced by re-running the normalizer, not by stripping values from an
        enriched record: a stripped record would still be stamped with an
        extractor version it did not use.
        """
        return normalize_store(self.store, self.configs)

    def episodes(self, *, refresh: bool = False) -> EpisodeView:
        """A derived grouping. Demoted on purpose — see ``episode.py``."""
        if self._episodes is not None and not refresh:
            return self._episodes
        self._episodes = group_records(self.records(refresh=refresh))
        return self._episodes

    # -- definitions and evaluation ----------------------------------------

    def definition(
        self, definition_id: str, version: int | None = None, *,
        allow_draft: bool = False,
    ) -> PhenotypeDefinition:
        return self.definitions.get(definition_id, version, allow_draft=allow_draft)

    def exposure_totals(
        self, records: Sequence[CanonicalAERecord] | None = None
    ) -> dict[str, float]:
        """Cumulative dose before onset, per record. Governed computation."""
        normalizer = RecordNormalizer(self.configs, self.store)
        totals: dict[str, float] = {}
        for record in (records if records is not None else self.records()):
            if not record.onset.observed:
                continue
            total, _why = normalizer.cumulative_exposure(
                record.subject_id, record.onset.value
            )
            if total is not None:
                totals[record.record_id] = total
        return totals

    def evaluator(self, definition: PhenotypeDefinition,
                  records: Sequence[CanonicalAERecord] | None = None
                  ) -> PhenotypeEvaluator:
        pool = records if records is not None else self.records()
        totals = (
            self.exposure_totals(pool)
            if definition.cumulative_exposure is not None else {}
        )
        return PhenotypeEvaluator(definition, self.configs.catalog, totals)

    def evaluate(
        self, definition: PhenotypeDefinition,
        studies: Sequence[str] | None = None,
        records: Sequence[CanonicalAERecord] | None = None,
    ) -> EvaluationResult:
        pool = list(records if records is not None else self.records())
        if studies:
            allowed = set(studies)
            pool = [r for r in pool if r.study_id in allowed]
        return self.evaluator(definition, pool).evaluate_all(pool)

    def assignments(
        self, definition: PhenotypeDefinition,
        studies: Sequence[str] | None = None,
    ) -> list[CaseAssignment]:
        return self.evaluate(definition, studies).assignments

    def cohort(self, studies: Sequence[str] | None = None) -> list[tuple[str, str]]:
        pairs = [(s, self.store.study_of(s) or "") for s in self.store.subjects()]
        if studies:
            allowed = set(studies)
            pairs = [p for p in pairs if p[1] in allowed]
        return pairs

    # -- supportability -----------------------------------------------------

    def supportability(self, modifier: str) -> list[dict[str, Any]]:
        """Which studies can answer a question about ``modifier``, and how.

        Decided on declared metadata alone, before any patient-level query
        runs. A study that records the modifier nowhere cannot answer, and
        finding that out by scanning its patients first is both slower and
        worse manners.
        """
        return self.configs.profiles.supportability(modifier)

    # -- text mentions, for the discovery path -------------------------------

    def mentions(self, modifier: str = "mucosal_involvement") -> list[dict[str, Any]]:
        """Every modifier mention in readable free text.

        A mention is a place in a document where something is named, which is a
        different claim from the event having occurred — so mentions are kept
        separate from records rather than folded into them.
        """
        finder = getattr(self.engine().backend, "finder", None)
        if finder is None:
            return []
        found: list[dict[str, Any]] = []
        for record in self.records():
            profile = self.configs.profiles.for_study(record.study_id)
            for doc_id, text, source_kind, variable in self.engine().sources_for(
                record, profile
            ):
                for mention in finder.find(text, modifier, None, source_kind):
                    found.append({
                        "mention_id": f"{doc_id}:{mention.start}:{mention.assertion}",
                        "doc_id": doc_id,
                        "study_id": record.study_id,
                        "subject_id": record.subject_id,
                        "source_record_id": record.source_record_id,
                        "profile": record.profile,
                        "modifier": modifier,
                        "assertion": mention.assertion,
                        "value": mention.value,
                        "surface": mention.surface,
                        "sentence": text,
                        "source_kind": source_kind,
                        "source_variable": variable,
                        "confidence": mention.confidence,
                        "rule": mention.rule,
                    })
        return found

    # -- retrieval ----------------------------------------------------------

    def index(self, *, refresh: bool = False, persist: bool = True):
        from .retrieval.index import build_index

        if self._index is not None and not refresh:
            return self._index
        self._index = build_index(
            self.store_path if persist else None,
            self.store, self.records(refresh=refresh),
            self.configs.extractor_version, self.configs.normalizer_version,
            self.mentions(),
        )
        return self._index

    def retrieve(self, **kwargs: Any):
        from .retrieval.query import retrieve

        return retrieve(self.index(), self.configs.catalog, **kwargs)

    def discover(self, **kwargs: Any):
        from .retrieval.query import discover

        return discover(self.index(), self.configs.catalog, **kwargs)

    # -- provenance ---------------------------------------------------------

    def versions(self) -> dict[str, Any]:
        return {
            "normalizer_version": self.configs.normalizer_version,
            "extractor_version": self.configs.extractor_version,
            "snapshot_id": self.snapshot_id,
            "dictionary_versions": self.configs.dictionary_versions(),
            "dictionary_target": self.configs.catalog.target_version,
            **self.engine().versions(),
        }

    def summary(self) -> dict[str, Any]:
        records = self.records()
        return {
            **self.store.summary(),
            **self.versions(),
            "records": len(records),
            "episodes": len(self.episodes().episodes),
            "profiles": self.configs.profiles.profile_ids(),
            "definitions": [d.key for d in self.definitions.all()],
            "extraction": self.engine().stats.to_dict(),
            "backend_notes": list(self.engine().notes),
        }


def restrict_readable_sources(
    configs: Configs, sources: Sequence[str]
) -> Configs:
    """A copy of ``configs`` whose model path may read only ``sources``.

    A copy, never a mutation: the ablation runs several stages in one process,
    and a stage that edited shared configuration would contaminate the next.
    """
    raw = {**configs.extraction.raw, "readable_sources": list(sources)}
    return Configs(
        catalog=configs.catalog,
        extraction=ExtractionConfig(raw, configs.extraction.source_path),
        profiles=configs.profiles,
        extractor_version=configs.extractor_version,
        normalizer_version=configs.normalizer_version,
    )
