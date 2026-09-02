"""Orchestration of the model path.

The engine asks the backend about exactly the attributes a record left
unresolved, in exactly the places the study's profile says they live. It never
asks about a value the deterministic path already settled — ``guards.py``
refuses that, and a test asserts it over the whole corpus.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from typing import Any, Iterable, Sequence

from ..catalog import Configs
from ..guards import assert_model_path_permitted, askable_attributes
from ..ingest import TrialStore
from ..models import Attribute, CanonicalAERecord, Modifier
from ..profiles import StudyProfile
from .backends import Backend, ExtractionRequest, select_backend


@dataclass
class ExtractionEngine:
    configs: Configs
    store: TrialStore
    backend: Backend
    notes: list[str] = _dc_field(default_factory=list)

    @classmethod
    def build(
        cls, configs: Configs, store: TrialStore, preference: str = "auto"
    ) -> "ExtractionEngine":
        backend, notes = select_backend(
            configs.catalog, configs.extraction, configs.extractor_version,
            preference,
        )
        return cls(configs=configs, store=store, backend=backend, notes=notes)

    # -- one record ---------------------------------------------------------

    def sources_for(
        self, record: CanonicalAERecord, profile: StudyProfile
    ) -> list[tuple[str, str, str, str]]:
        """Every readable text source for a record.

        Returns ``(doc_id, text, source_kind, source_variable)``. Only sources
        the extraction config declares readable appear here.
        """
        readable = set(self.configs.extraction.readable_sources)
        found: list[tuple[str, str, str, str]] = []
        if "reported_term" in readable and record.reported_term.populated:
            found.append((
                f"AE:{record.source_record_id}:AETERM",
                str(record.reported_term.value), "reported_term", "AETERM",
            ))
        if "comment" in readable:
            for document in self.store.documents_of(record.source_record_id):
                found.append((
                    document.doc_id, document.full_text, "comment", "CO.COVAL",
                ))
        return found

    def enrich(self, record: CanonicalAERecord) -> CanonicalAERecord:
        """Fill what the deterministic path left open, and nothing else."""
        profile = self.configs.profiles.for_study(record.study_id)
        askable = askable_attributes(record, profile, self.configs.extraction)
        if not askable:
            return record

        for doc_id, text, source_kind, variable in self.sources_for(record, profile):
            remaining = tuple(a for a in askable if not self._filled(record, a))
            if not remaining:
                break
            request = ExtractionRequest(
                doc_id=doc_id, text=text,
                attributes=remaining + ("quality",),
                concept_id=record.standardized_concept,
                source_kind=source_kind, source_variable=variable,
            )
            # The boundary is enforced here, on every request, not asserted once
            # in a test and trusted thereafter.
            assert_model_path_permitted(request, record, profile)
            result = self.backend.extract(request)

            for attribute, value in result.values.items():
                current = record.attribute(attribute)
                setattr(record, attribute, value.model_copy(update={
                    "prior_availability": current.availability if current else None,
                }))
            for attribute in result.abstained:
                current = record.attribute(attribute)
                if current is not None and not current.populated:
                    # Both facts survive: what the structured side already said
                    # about this attribute, and that the text was read and
                    # supported nothing. Replacing the first with the second
                    # would make an availability and its explanation contradict
                    # each other on the same row.
                    setattr(record, attribute, current.model_copy(update={
                        "note": (
                            f"{attribute} is {current.availability}"
                            + (f" ({current.note})" if current.note else "")
                            + f"; the model path was then asked about it in "
                              f"{variable} and abstained — nothing in the text "
                              f"supports a value"
                        ),
                    }))
            record.modifiers.extend(
                Modifier(
                    kind="quality", value=quality, surface=quality,
                    span=self._quality_span(doc_id, text, quality),
                )
                for quality in result.qualities
            )
            self.notes.extend(n for n in result.notes if n not in self.notes)

        record.extractor_version = self.configs.extractor_version
        return record

    @staticmethod
    def _filled(record: CanonicalAERecord, attribute: str) -> bool:
        current = record.attribute(attribute)
        return bool(current and current.populated)

    @staticmethod
    def _quality_span(doc_id: str, text: str, quality: str):
        from ..models import Span

        lowered = text.lower()
        start = lowered.find(quality.lower())
        if start < 0:
            start = 0
        return Span(
            doc_id=doc_id, start=start, end=start + len(quality), field="quality",
            extracted_value=quality, text=text[start:start + len(quality)],
            kind="text",
        )

    def enrich_all(
        self, records: Iterable[CanonicalAERecord]
    ) -> list[CanonicalAERecord]:
        return [self.enrich(record) for record in records]

    def versions(self) -> dict[str, Any]:
        return {
            "extraction_backend": self.backend.name,
            "model_version": getattr(self.backend, "model_version", None),
            "prompt_version": getattr(self.backend, "prompt_version", None),
        }


def enrich_records(
    records: Sequence[CanonicalAERecord], configs: Configs, store: TrialStore,
    preference: str = "auto",
) -> list[CanonicalAERecord]:
    return ExtractionEngine.build(configs, store, preference).enrich_all(records)
