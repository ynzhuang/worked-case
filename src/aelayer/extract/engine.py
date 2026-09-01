"""Orchestration for the model path.

For each record: work out which fields the deterministic path left unresolved,
ask the backend about those and no others, then accept only what comes back
grounded in a span.

Nothing here overwrites a value the deterministic path resolved.  Extracted
values land on the record marked ``source="text"``, so a reader can always tell
which system produced which field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from ..catalog import ConceptCatalog, ExtractionConfig
from ..guards import (
    ControlledValueLeak,
    ModelRequest,
    assert_model_path_permitted,
    assert_no_structured_payload,
    unresolved_fields,
)
from ..ingest import Narrative
from ..models import (
    CanonicalAERecord,
    Field,
    LabValue,
    Span,
    SymptomMention,
)
from .backends import Backend, ExtractionResult, PROMPT_VERSION, select_backend


@dataclass
class ExtractionEngine:
    catalog: ConceptCatalog
    config: ExtractionConfig
    backend: Backend
    extractor_version: str
    notes: tuple[str, ...] = ()

    @classmethod
    def build(
        cls,
        catalog: ConceptCatalog,
        config: ExtractionConfig,
        extractor_version: str,
        prefer: str = "auto",
    ) -> "ExtractionEngine":
        backend, notes = select_backend(catalog, config, prefer)
        return cls(catalog, config, backend, extractor_version, tuple(notes))

    # -- one record ---------------------------------------------------------

    def enrich(
        self, record: CanonicalAERecord, narrative: Narrative | None
    ) -> CanonicalAERecord:
        """Fill unresolved fields on a record from its narrative.

        Returns the record, mutated in place for the unresolved fields only.
        A field the deterministic path settled is never touched — the guard
        raises rather than allowing it.
        """
        record.extractor_version = self.extractor_version
        record.model_version = getattr(self.backend, "model_version", None)
        record.prompt_version = PROMPT_VERSION

        if narrative is None:
            record.assertion = Field[str].missing(
                "not_collected_by_protocol", source="text",
                note="no narrative is attached to this record",
            )
            return record

        askable = unresolved_fields(record)
        if not askable:
            return record

        request = ModelRequest(
            doc_id=narrative.doc_id,
            text=narrative.full_text,
            requested_fields=askable,
            schema_name="CanonicalAERecord",
            prompt_version=PROMPT_VERSION,
            record_id=record.source_record_id,
        )
        assert_no_structured_payload(request, where="ExtractionEngine.enrich")
        assert_model_path_permitted(request, record)

        result = self.backend.extract(request)
        self._apply(record, result, request)
        return record

    def _apply(
        self, record: CanonicalAERecord, result: ExtractionResult,
        request: ModelRequest,
    ) -> None:
        extracted = result.by_field()

        for name, value in extracted.items():
            if not value.is_grounded():
                # A populated value with no span is rejected. Accepting it
                # would put a number in a report that traces to nothing.
                result.abstained.append(name)
                continue

        if "symptoms" in extracted and extracted["symptoms"].is_grounded():
            record.symptoms = [
                SymptomMention(
                    symptom=item["symptom"], span=item["span"], assertion="present"
                )
                for item in extracted["symptoms"].value
            ]
        if "symptoms_assessed" in extracted:
            value = extracted["symptoms_assessed"]
            record.symptoms_assessed = Field[bool](
                value=True, collection_state="collected", source="text",
                spans=list(value.spans),
                note="the source addressed symptoms",
            )
        else:
            record.symptoms_assessed = Field[bool].missing(
                "unknown", source="text",
                note="the source does not address symptoms, so an empty symptom "
                     "list cannot be read as asymptomatic",
            )
        if "labs" in extracted and extracted["labs"].is_grounded():
            # Narrative values supplement, never replace, structured results;
            # a value already read from LB or a linked form stays as it is.
            existing = {
                (l.test, round(l.canonical_value or -1.0, 2)) for l in record.labs
            }
            for item in extracted["labs"].value:
                key = (item["test"], round(item["canonical_value"] or -1.0, 2))
                if key in existing:
                    continue
                existing.add(key)
                record.labs.append(
                    LabValue(
                        test=item["test"], value=item["value"], unit=item["unit"],
                        canonical_value=item["canonical_value"],
                        canonical_unit=item["canonical_unit"],
                        source="text", span=item["span"],
                    )
                )

        if (
            "standardized_concept" in extracted
            and extracted["standardized_concept"].is_grounded()
            and record.standardized_concept is None
        ):
            value = extracted["standardized_concept"]
            record.standardized_concept = value.value
            record.concept_source = "text"
            record.evidence = list(record.evidence) + list(value.spans)

        if "assertion" in extracted and extracted["assertion"].is_grounded():
            value = extracted["assertion"]
            record.assertion = Field[str](
                value=value.value, collection_state="collected", source="text",
                spans=list(value.spans), confidence=value.confidence,
                note=value.note,
            )
        else:
            record.assertion = Field[str].missing(
                "unknown", source="text",
                note="no concept mention was found in the narrative",
            )

        for name in ("severity", "relatedness", "action_taken", "outcome"):
            if name not in extracted:
                continue
            value = extracted[name]
            if not value.is_grounded():
                continue
            current: Field[Any] = getattr(record, name)
            if current.collection_state == "collected":  # pragma: no cover - guarded
                raise ControlledValueLeak(
                    f"model path returned {name!r} for a record that already "
                    f"has it collected"
                )
            setattr(
                record, name,
                Field[str](
                    value=value.value, collection_state="collected", source="text",
                    spans=list(value.spans), confidence=value.confidence,
                    note=(
                        f"recovered from narrative; the structured field was "
                        f"{current.collection_state}"
                    ),
                ),
            )

        # Abstention is recorded, not silently dropped: knowing the model
        # declined is different from never having asked.
        for name in sorted(set(result.abstained)):
            field: Field[Any] | None = record.fields().get(name)
            if field is not None and field.collection_state not in ("collected",):
                field.note = (
                    f"{field.note + '; ' if field.note else ''}"
                    f"the model path was asked and abstained"
                )

        record.evidence = _merge_spans(record)

    # -- corpus -------------------------------------------------------------

    def enrich_all(
        self, records: Sequence[CanonicalAERecord],
        narratives: dict[str, Narrative],
    ) -> list[CanonicalAERecord]:
        for record in records:
            self.enrich(record, narratives.get(record.narrative_doc_id or ""))
        return list(records)


def _merge_spans(record: CanonicalAERecord) -> list[Span]:
    spans: list[Span] = list(record.evidence)
    for field in record.fields().values():
        spans.extend(field.spans)
    spans.extend(s.span for s in record.symptoms)
    spans.extend(l.span for l in record.labs)
    seen: set[tuple] = set()
    unique: list[Span] = []
    for span in spans:
        if span.key() not in seen:
            seen.add(span.key())
            unique.append(span)
    return sorted(unique, key=lambda s: (s.field, s.doc_id, s.start, s.end))


def extract_records(
    records: Sequence[CanonicalAERecord],
    store,
    configs,
    prefer: str = "auto",
) -> tuple[list[CanonicalAERecord], ExtractionEngine]:
    engine = ExtractionEngine.build(
        configs.catalog, configs.extraction, configs.extractor_version, prefer
    )
    return engine.enrich_all(records, store.narratives), engine
