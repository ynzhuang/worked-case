"""Build source-faithful canonical records from structured CRF data.

One ``CanonicalAERecord`` per source record.  Nothing here merges rows, and
nothing here interprets prose: this is the deterministic path, and the fields it
resolves are the fields the model path is never asked about.

The order of work for each field is fixed:

1. Is the field gated behind a parent question that was answered No?
2. Does the study collect the field at all?
3. Is there a value, and does it map onto a canonical controlled value?
4. If not, what does this study's blank mean?

Only step 3 can produce ``collected``.  Every other outcome carries the reason.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Any, Iterable

from ..catalog import ConceptCatalog
from ..models import (
    SERIOUSNESS_CRITERIA,
    CanonicalAERecord,
    CollectionState,
    Field,
    LabValue,
    Span,
)
from ..semantics import CollectionSemantics, StudySemantics
from . import values as V

#: AE table columns, and the canonical field each carries.
COLUMN_FOR: dict[str, str] = {
    "verbatim_term": "AETERM",
    "coded_term": "AEDECOD",
    "onset_datetime": "AESTDTC",
    "end_datetime": "AEENDTC",
    "severity": "AESEV",
    "seriousness": "AESER",
    "relatedness": "AEREL",
    "action_taken": "AEACN",
    "outcome": "AEOUT",
}


def render_row(row: dict[str, Any]) -> str:
    """A stable, checkable rendering of a source row, for span text."""
    return " | ".join(
        f"{column}={row.get(column) if row.get(column) not in (None, '') else ''}"
        for column in sorted(row)
        if column not in ("SYNTHETIC",)
    )


@dataclass
class RecordNormalizer:
    """Deterministic normalization of one study's AE records."""

    catalog: ConceptCatalog
    semantics: CollectionSemantics
    normalizer_version: str

    # -- field construction -------------------------------------------------

    def _span(self, row: dict[str, Any], field: str, value: Any) -> Span:
        rendered = render_row(row)
        return Span(
            doc_id=f"AE:{row.get('AESPID') or row.get('USUBJID')}",
            start=0,
            end=len(rendered),
            field=field,
            extracted_value="" if value is None else str(value),
            text=rendered,
            kind="structured",
        )

    def _gated_state(
        self, field: str, row: dict[str, Any], study: StudySemantics,
        gate_inputs: dict[str, Any],
    ) -> CollectionState | None:
        """The state a parent gate forces on this field, if any."""
        gate = study.gate_for(field)
        if gate is None:
            return None
        answer = study.gate_answer(gate.gate, gate_inputs)
        return gate.resolve(answer)

    def _enum_field(
        self, field: str, row: dict[str, Any], study: StudySemantics,
    ) -> Field[str]:
        cell = row.get(COLUMN_FOR[field])
        value, state, note = V.coerce_enum(field, cell, study)
        if value is not None:
            return Field[str](
                value=value, collection_state="collected", source="structured",
                spans=[self._span(row, field, value)],
            )
        return Field[str](
            value=None, collection_state=state, source="structured",
            spans=[self._span(row, field, None)], note=note,
        )

    # -- record -------------------------------------------------------------

    def normalize_record(
        self,
        row: dict[str, Any],
        *,
        linked_rows: Iterable[dict[str, Any]] = (),
        lab_rows: Iterable[dict[str, Any]] = (),
    ) -> CanonicalAERecord:
        study = self.semantics.for_study(str(row.get("STUDYID")))
        source_record_id = str(row.get("AESPID") or f"{row.get('USUBJID')}-{row.get('AESEQ')}")

        verbatim = self._text_field("verbatim_term", row, study)
        coded = self._text_field("coded_term", row, study)
        dictionary_version = (row.get("AEDICTVER") or None) or (
            study.dictionary_version if coded.populated else None
        )

        # Gate answers are computed from the record itself, per the declared
        # gate_values in collection semantics.
        seriousness = self._boolean_field("seriousness", row, study)
        outcome = self._enum_field("outcome", row, study)
        gate_inputs = {
            "seriousness": seriousness.value,
            "outcome": outcome.value,
        }

        end_state = self._gated_state("end_datetime", row, study, gate_inputs)
        record = CanonicalAERecord(
            record_id=source_record_id,
            study_id=study.study_id,
            subject_id=str(row.get("USUBJID")),
            source_record_id=source_record_id,
            source_form_id="AE",
            verbatim_term=verbatim,
            coded_term=coded,
            dictionary=study.dictionary if coded.populated else None,
            dictionary_version=dictionary_version if coded.populated else None,
            standardized_concept=self.standardize(coded.value, dictionary_version),
            concept_source="coded" if self.standardize(coded.value, dictionary_version) else None,
            onset_datetime=self._datetime_field("onset_datetime", row, study),
            end_datetime=self._datetime_field(
                "end_datetime", row, study, forced_state=end_state
            ),
            severity=self._enum_field("severity", row, study),
            seriousness=seriousness,
            seriousness_criteria=self._criteria(row, study, gate_inputs),
            relatedness=self._enum_field("relatedness", row, study),
            action_taken=self._action_field(row, study),
            outcome=outcome,
            linked_form_ids=[f for f in [row.get("AELNKID")] if f],
            narrative_doc_id=(row.get("DOCID") or None),
            continuation_of=(row.get("AECONTRP") or None),
            normalizer_version=self.normalizer_version,
        )
        # Laboratory results are controlled values in a structured domain, so
        # they are read here and never put to the model path.
        record.labs = list(self._linked_labs(record, linked_rows)) + list(
            self._domain_labs(record, lab_rows)
        )
        record.evidence = self._collect_spans(record)
        return record

    def _text_field(
        self, field: str, row: dict[str, Any], study: StudySemantics
    ) -> Field[str]:
        cell = row.get(COLUMN_FOR[field])
        if V.is_blank(cell):
            return Field[str](
                value=None, collection_state=study.state_for_blank(field),
                source="structured", spans=[self._span(row, field, None)],
            )
        return Field[str](
            value=str(cell).strip(), collection_state="collected",
            source="structured", spans=[self._span(row, field, cell)],
        )

    def _boolean_field(
        self, field: str, row: dict[str, Any], study: StudySemantics
    ) -> Field[bool]:
        value, state, note = V.coerce_bool(field, row.get(COLUMN_FOR[field]), study)
        return Field[bool](
            value=value,
            collection_state="collected" if value is not None else state,
            source="structured", spans=[self._span(row, field, value)], note=note,
        )

    def _datetime_field(
        self, field: str, row: dict[str, Any], study: StudySemantics,
        forced_state: CollectionState | None = None,
    ) -> Field[_dt.datetime]:
        value, state, note = V.coerce_datetime(field, row.get(COLUMN_FOR[field]), study)
        if value is None and forced_state is not None:
            # A gate outranks the generic blank meaning: an event that has not
            # ended has a pending end date, not an unknown one.
            state = forced_state
        return Field[_dt.datetime](
            value=value,
            collection_state="collected" if value is not None else state,
            source="structured", spans=[self._span(row, field, value)], note=note,
        )

    def _action_field(
        self, row: dict[str, Any], study: StudySemantics
    ) -> Field[str]:
        """Treatment action, including the case the codelist cannot express.

        Where the study's permissible values have no code for what happened,
        the field stays unresolved and says so. Choosing the nearest available
        code would assert a stronger claim than the evidence supports.
        """
        field = "action_taken"
        cell = row.get(COLUMN_FOR[field])
        value, state, note = V.coerce_enum(field, cell, study)
        if value is not None:
            return Field[str](
                value=value, collection_state="collected", source="structured",
                spans=[self._span(row, field, value)],
            )
        codelist = study.codelist_for(field)
        if (
            V.is_blank(cell)
            and codelist is not None
            and codelist.absent_concepts
            and study.collects(field)
        ):
            # The study collects the field, the cell is empty, and this
            # study's codelist is known to be missing concepts. The blank is
            # therefore unresolved-from-this-field, not evidence of no action.
            state = "not_representable"
            note = (
                f"{study.study_id} has no permissible {field} value for "
                f"{list(codelist.absent_concepts)}; a blank here cannot be read "
                f"as 'no action taken'"
            )
        return Field[str](
            value=None, collection_state=state, source="structured",
            spans=[self._span(row, field, None)], note=note,
        )

    def _criteria(
        self, row: dict[str, Any], study: StudySemantics, gate_inputs: dict[str, Any]
    ) -> dict[str, Field[bool]]:
        """The seriousness criteria vector, gated on the seriousness answer.

        A criterion left blank because the gate was answered No is
        ``not_applicable_gated``. Recording it as ``unknown``, or as False,
        would invent a fact the CRF never asked for.
        """
        forced = self._gated_state("seriousness_criteria", row, study, gate_inputs)
        recorded = {
            part.strip()
            for part in str(row.get("AESCAT") or "").split("|")
            if part.strip()
        }
        out: dict[str, Field[bool]] = {}
        for criterion in SERIOUSNESS_CRITERIA:
            span = self._span(row, f"seriousness_criteria.{criterion}", None)
            if not study.collects("seriousness"):
                out[criterion] = Field[bool](
                    value=None, collection_state=study.state_for_blank("seriousness"),
                    source="structured", spans=[span],
                )
            elif forced is not None:
                out[criterion] = Field[bool](
                    value=None, collection_state=forced, source="structured",
                    spans=[span],
                    note="the seriousness gate was answered No, so this "
                         "criterion was never applicable",
                )
            else:
                out[criterion] = Field[bool](
                    value=criterion in recorded, collection_state="collected",
                    source="structured", spans=[span],
                )
        return out

    def _linked_labs(
        self, record: CanonicalAERecord, linked_rows: Iterable[dict[str, Any]]
    ) -> Iterable[LabValue]:
        """Objective values carried on a linked event form.

        Controlled fields on a linked form are structured data and are read
        deterministically; the model path never sees them.
        """
        lab = self.catalog.lab_tests.get("GLUCOSE")
        if lab is None:
            return
        for row in linked_rows:
            raw = row.get("GLUCVAL")
            if V.is_blank(raw):
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            unit = str(row.get("GLUCUNIT") or "").strip()
            canonical = V.to_canonical_unit(value, unit, lab.conversions)
            if canonical is None or not lab.plausible(canonical):
                continue
            rendered = render_row(row)
            collected, _state, _note = V.coerce_datetime(
                "linked_form_date", row.get("HEFDTC"),
                self.semantics.for_study(record.study_id),
            )
            yield LabValue(
                test="GLUCOSE", value=value, unit=unit,
                canonical_value=canonical, canonical_unit=lab.canonical_unit,
                collection_datetime=collected, source="structured",
                span=Span(
                    doc_id=f"FORM:{row.get('LNKID')}", start=0, end=len(rendered),
                    field="labs", extracted_value=f"GLUCOSE={value}{unit}",
                    text=rendered, kind="structured",
                ),
            )

    def _domain_labs(
        self, record: CanonicalAERecord, lab_rows: Iterable[dict[str, Any]]
    ) -> Iterable[LabValue]:
        """Results from the LB domain collected on the day of the event.

        Same day only: a wider window borrows a result belonging to a
        neighbouring event on the same subject and offers it as corroboration
        for this one.
        """
        onset = record.onset_datetime.value
        lab = self.catalog.lab_tests.get("GLUCOSE")
        if onset is None or lab is None:
            return
        study = self.semantics.for_study(record.study_id)
        for row in lab_rows:
            if str(row.get("LBTESTCD") or "").upper() != "GLUC":
                continue
            collected, state, _note = V.coerce_datetime("LBDTC", row.get("LBDTC"), study)
            if collected is None or collected.date() != onset.date():
                continue
            try:
                value = float(row.get("LBORRES"))
            except (TypeError, ValueError):
                continue
            unit = str(row.get("LBORRESU") or "").strip()
            canonical = V.to_canonical_unit(value, unit, lab.conversions)
            if canonical is None or not lab.plausible(canonical):
                continue
            rendered = render_row(row)
            yield LabValue(
                test="GLUCOSE", value=value, unit=unit, canonical_value=canonical,
                canonical_unit=lab.canonical_unit, collection_datetime=collected,
                source="structured",
                span=Span(
                    doc_id=f"LB:{row.get('USUBJID')}:{row.get('LBSEQ')}",
                    start=0, end=len(rendered), field="labs",
                    extracted_value=f"GLUCOSE={value}{unit}", text=rendered,
                    kind="structured",
                ),
            )

    def standardize(
        self, coded_term: str | None, dictionary_version: str | None
    ) -> str | None:
        """Which catalogue concept a coded term denotes.

        Membership only: a term belongs to a concept because the catalogue
        lists it, never because it sits beneath one in a hierarchy.
        """
        if not coded_term:
            return None
        key = coded_term.strip().casefold()
        for concept_id in sorted(self.catalog.concepts):
            concept = self.catalog.concepts[concept_id]
            if any(term.strip().casefold() == key for term in concept.all_coded_terms()):
                return concept_id
        return None

    @staticmethod
    def _collect_spans(record: CanonicalAERecord) -> list[Span]:
        spans: list[Span] = []
        for field in record.fields().values():
            spans.extend(field.spans)
        spans.extend(lab.span for lab in record.labs)
        seen: set[tuple] = set()
        unique: list[Span] = []
        for span in spans:
            if span.key() not in seen:
                seen.add(span.key())
                unique.append(span)
        return sorted(unique, key=lambda s: (s.field, s.doc_id, s.start))


def normalize_store(store, configs) -> list[CanonicalAERecord]:
    """Normalize every AE row in a snapshot, in a stable order."""
    normalizer = RecordNormalizer(
        catalog=configs.catalog,
        semantics=configs.semantics,
        normalizer_version=configs.normalizer_version,
    )
    linked_by_record: dict[str, list[dict[str, Any]]] = {}
    for row in store.rows("linked_hypo_event"):
        linked_by_record.setdefault(str(row.get("AESPID")), []).append(row)
    labs_by_subject: dict[str, list[dict[str, Any]]] = {}
    for row in store.rows("lb"):
        labs_by_subject.setdefault(str(row.get("USUBJID")), []).append(row)

    records = [
        normalizer.normalize_record(
            row,
            linked_rows=linked_by_record.get(str(row.get("AESPID")), []),
            lab_rows=labs_by_subject.get(str(row.get("USUBJID")), []),
        )
        for row in store.rows("ae")
    ]
    return sorted(records, key=lambda r: (r.study_id, r.subject_id, r.source_record_id))
