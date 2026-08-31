"""One assembled pipeline: data, config, extraction, evaluation, retrieval.

The CLI, the API and the agent all go through here, so they cannot drift apart
on which definition version ran or which extractor produced a result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import paths
from .catalog import ConceptCatalog, ExtractionConfig, load_configs
from .extract.engine import ExtractionEngine
from .ingest import TrialStore, load_store
from .models import CaseAssignment, EventObject, PhenotypeDefinition
from .phenotype.evaluator import PhenotypeEvaluator
from .phenotype.loader import DefinitionCatalog
from .retrieval.index import EventIndex, build_index
from .retrieval.query import RetrievalResult, retrieve


@dataclass
class Pipeline:
    store: TrialStore
    catalog: ConceptCatalog
    config: ExtractionConfig
    extractor_version: str
    definitions: DefinitionCatalog
    store_path: Path | None = None
    _events: list[EventObject] | None = field(default=None, repr=False)
    _index: EventIndex | None = field(default=None, repr=False)

    # -- construction -------------------------------------------------------

    @classmethod
    def load(
        cls,
        data_dir: str | Path | None = None,
        *,
        concepts_path: str | Path | None = None,
        extraction_path: str | Path | None = None,
        phenotype_dir: str | Path | None = None,
        store_path: str | Path | None = None,
    ) -> "Pipeline":
        catalog, config, version = load_configs(concepts_path, extraction_path)
        store = load_store(data_dir)
        return cls(
            store=store,
            catalog=catalog,
            config=config,
            extractor_version=version,
            definitions=DefinitionCatalog(phenotype_dir, catalog),
            store_path=Path(store_path) if store_path else None,
        )

    # -- extraction ---------------------------------------------------------

    @property
    def snapshot_id(self) -> str:
        return self.store.snapshot_id

    def engine(self) -> ExtractionEngine:
        return ExtractionEngine(
            self.catalog,
            self.config,
            self.extractor_version,
            self.store.anchor_resolver(self.config.anchors),
        )

    def events(self, *, refresh: bool = False) -> list[EventObject]:
        """Extract, or reuse a cached store built from the same inputs.

        The cache is keyed on the snapshot id and the extractor version, so a
        changed config or a changed corpus can never be served from a stale
        index.
        """
        if self._events is not None and not refresh:
            return self._events
        if not refresh and self.store_path and Path(self.store_path).exists():
            try:
                index = EventIndex.open(self.store_path)
                if index.meta().matches(self.snapshot_id, self.extractor_version):
                    self._index = index
                    self._events = index.events()
                    return self._events
                index.close()
            except (FileNotFoundError, Exception):  # noqa: B014 - sqlite variants
                pass
        self._events = self.engine().extract_store(self.store)
        self._index = None
        return self._events

    def index(self, *, refresh: bool = False, persist: bool = True) -> EventIndex:
        """The retrieval index, built on demand."""
        if self._index is not None and not refresh:
            return self._index
        events = self.events(refresh=refresh)
        target = self.store_path if persist else None
        self._index = build_index(target, self.store, events, self.extractor_version)
        return self._index

    # -- definitions --------------------------------------------------------

    def definition(
        self,
        definition_id: str,
        version: int | None = None,
        *,
        allow_draft: bool = False,
    ) -> PhenotypeDefinition:
        return self.definitions.get(definition_id, version, allow_draft=allow_draft)

    def cohort(self, studies: Sequence[str] | None = None) -> list[tuple[str, str]]:
        """Every subject in scope, as ``(subject_id, study_id)``.

        Subjects with no qualifying event are part of the denominator and must
        appear, or every rate computed downstream is wrong.
        """
        pairs = [(s, self.store.study_of(s) or "") for s in self.store.subjects()]
        if studies:
            allowed = set(studies)
            pairs = [p for p in pairs if p[1] in allowed]
        return pairs

    # -- evaluation ---------------------------------------------------------

    def evaluator(self, definition: PhenotypeDefinition) -> PhenotypeEvaluator:
        return PhenotypeEvaluator(
            definition, self.catalog, self.store.anchor_resolver(self.config.anchors)
        )

    def evaluate(
        self,
        definition: PhenotypeDefinition,
        studies: Sequence[str] | None = None,
        events: Iterable[EventObject] | None = None,
    ) -> list[CaseAssignment]:
        cohort = self.cohort(studies)
        in_scope = {subject for subject, _study in cohort}
        source = list(events) if events is not None else self.events()
        scoped = [e for e in source if e.subject_id in in_scope]
        return self.evaluator(definition).evaluate(scoped, cohort)

    # -- retrieval ----------------------------------------------------------

    def retrieve(self, **kwargs: Any) -> RetrievalResult:
        return retrieve(self.index(), self.catalog, **kwargs)

    def summary(self) -> dict[str, Any]:
        return {
            **self.store.summary(),
            "extractor_version": self.extractor_version,
            "definitions": [d.key for d in self.definitions.all()],
        }
