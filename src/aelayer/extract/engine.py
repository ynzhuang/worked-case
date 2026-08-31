"""Extraction orchestration.

Given a narrative and the structured record for a subject, produce zero or more
``EventObject``s.  The order of work is fixed:

concept candidates -> assertion classification within scope -> symptoms, labs,
severity, seriousness, action, outcome, rescue, rechallenge -> temporal offset
and anchor resolution against EX -> span assembly -> confidence.

Two invariants hold for everything this module emits:

* every non-null field carries at least one span
* the same input plus the same config always produces byte-identical output

The extractor deliberately does **not** assign evidence states and does not
decide cases.  It reports what the text and the tables say.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Any, Iterable

from ..anchors import AnchorResolver, parse_date
from ..catalog import ConceptCatalog, ExtractionConfig, load_configs
from ..ingest import Narrative, TrialStore
from ..models import EventObject, LabValue, Span, SymptomMention
from .assertion import AssertionClassifier, AssertionResult
from .concepts import ConceptMatcher, ConceptMention
from .temporal import TemporalExtractor
from .text import Sentence, sentence_for, split_sentences
from .values import CueHit, LabHit, ValueExtractor

#: Rendered stand-ins for structured rows, so a value read from a table still
#: points at a concrete, checkable string rather than at nothing.
def _ae_doc_id(row: dict[str, Any]) -> str:
    return f"AE:{row.get('USUBJID')}:{row.get('AESEQ')}"


def _render_ae_row(row: dict[str, Any]) -> str:
    fields = ("AETERM", "AEDECOD", "AEDICTVER", "AESTDTC", "AESEV", "AESER",
              "AESCAT", "AEREL", "AEACN", "AEOUT")
    return " | ".join(f"{f}={row.get(f) or ''}" for f in fields)


@dataclass
class SubjectContext:
    """Everything about a subject the extractor may consult."""

    subject_id: str
    study_id: str
    reference_start: _dt.date | None = None
    lb_rows: list[dict[str, Any]] = None  # type: ignore[assignment]
    cm_rows: list[dict[str, Any]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.lb_rows = self.lb_rows or []
        self.cm_rows = self.cm_rows or []


class ExtractionEngine:
    def __init__(
        self,
        catalog: ConceptCatalog,
        config: ExtractionConfig,
        extractor_version: str,
        anchor_resolver: AnchorResolver | None = None,
    ):
        self.catalog = catalog
        self.config = config
        self.extractor_version = extractor_version
        self.matcher = ConceptMatcher(catalog, config)
        self.assertions = AssertionClassifier(config)
        self.temporal = TemporalExtractor(config, anchor_resolver)
        self.values = ValueExtractor(catalog, config)
        self.default_anchor = config.temporality.get("default_anchor")

    # -- record level -------------------------------------------------------

    def extract_record(
        self,
        ae_row: dict[str, Any],
        narrative: Narrative | None,
        context: SubjectContext,
    ) -> list[EventObject]:
        """Every event object derivable from one AE record."""
        text = narrative.full_text if narrative else ""
        doc_id = narrative.doc_id if narrative else _ae_doc_id(ae_row)
        sentences = split_sentences(text)

        mentions = self.matcher.find_concepts(text, sentences)
        coded_term = (ae_row.get("AEDECOD") or None)
        coded_concept = self.matcher.coded_term_concept(coded_term)

        symptoms = self._present_symptoms(text, sentences)
        narrative_labs = self.values.find_labs(text)

        candidates = self._candidate_concepts(
            mentions, coded_concept, symptoms, narrative_labs
        )

        events: list[EventObject] = []
        for concept_id in sorted(candidates):
            concept_mentions = [m for m in mentions if m.concept_id == concept_id]
            if concept_mentions:
                grouped = self._group_by_assertion(text, sentences, concept_mentions)
                for assertion, group in sorted(grouped.items()):
                    events.append(
                        self._build_event(
                            concept_id, assertion, group, None, ae_row, narrative,
                            doc_id, text, sentences, context, symptoms,
                            narrative_labs, coded_term,
                        )
                    )
            else:
                # A candidate raised by contextual evidence alone. Its assertion
                # is classified at the evidence that raised it, so "denies
                # shakiness" cannot become a present event.
                anchor_span = self._contextual_anchor(symptoms, narrative_labs)
                assertion_result = (
                    self.assertions.classify(text, anchor_span[0], anchor_span[1], sentences)
                    if anchor_span
                    else None
                )
                assertion = assertion_result.assertion if assertion_result else "present"
                events.append(
                    self._build_event(
                        concept_id, assertion, [], (assertion_result, anchor_span),
                        ae_row, narrative, doc_id, text, sentences, context,
                        symptoms, narrative_labs, coded_term,
                    )
                )
        return events

    # -- candidate selection ------------------------------------------------

    def _candidate_concepts(
        self,
        mentions: list[ConceptMention],
        coded_concept: str | None,
        symptoms: list[tuple[str, int, int, str]],
        labs: list[LabHit],
    ) -> set[str]:
        candidates = {m.concept_id for m in mentions}
        if coded_concept:
            candidates.add(coded_concept)
        for concept_id in sorted(self.catalog.concepts):
            if concept_id in candidates:
                continue
            if self._contextual_candidate(concept_id, symptoms, labs):
                candidates.add(concept_id)
        return candidates

    def _contextual_candidate(
        self,
        concept_id: str,
        symptoms: list[tuple[str, int, int, str]],
        labs: list[LabHit],
    ) -> bool:
        """Does contextual evidence alone raise this concept as a candidate?

        Driven entirely by ``candidate_evidence`` in the concept catalogue.  A
        concept that declares none is only ever raised by an explicit mention.
        """
        spec = self.catalog.concept(concept_id).candidate_evidence
        if not spec:
            return False
        set_names = spec.get("symptom_sets") or []
        if set_names:
            qualifying = self.catalog.symptoms_in_sets(list(set_names))
            found = {s[0] for s in symptoms if s[0] in qualifying}
            if len(found) >= int(spec.get("min_symptoms", 1)):
                return True
        for test_id in spec.get("lab_tests") or []:
            if any(lab.test == test_id for lab in labs):
                return True
        return False

    @staticmethod
    def _contextual_anchor(
        symptoms: list[tuple[str, int, int, str]], labs: list[LabHit]
    ) -> tuple[int, int] | None:
        spans = [(s[1], s[2]) for s in symptoms]
        spans += [(lab.start, lab.end) for lab in labs if lab.source == "narrative"]
        return min(spans) if spans else None

    def _group_by_assertion(
        self, text: str, sentences: list[Sentence], mentions: list[ConceptMention]
    ) -> dict[str, list[tuple[ConceptMention, AssertionResult]]]:
        """One event object per distinct assertion class within a record.

        A record that both records an occurrence and documents an absence
        yields two event objects, because both are true and both must remain
        queryable.
        """
        grouped: dict[str, list[tuple[ConceptMention, AssertionResult]]] = {}
        for mention in mentions:
            result = self.assertions.classify(text, mention.start, mention.end, sentences)
            grouped.setdefault(result.assertion, []).append((mention, result))
        return grouped

    def _present_symptoms(
        self, text: str, sentences: list[Sentence]
    ) -> list[tuple[str, int, int, str]]:
        """Symptom mentions that are asserted, not negated.

        A denied symptom is not evidence of the event; it is evidence against.
        """
        out: list[tuple[str, int, int, str]] = []
        for match in self.matcher.find_symptoms(text):
            result = self.assertions.classify(text, match.start, match.end, sentences)
            if result.assertion == "present":
                out.append((match.symptom, match.start, match.end, match.surface))
        return sorted(out)

    # -- event assembly -----------------------------------------------------

    def _build_event(
        self,
        concept_id: str,
        assertion: str,
        mention_group: list[tuple[ConceptMention, AssertionResult]],
        contextual: tuple[AssertionResult | None, tuple[int, int] | None] | None,
        ae_row: dict[str, Any],
        narrative: Narrative | None,
        doc_id: str,
        text: str,
        sentences: list[Sentence],
        context: SubjectContext,
        symptoms: list[tuple[str, int, int, str]],
        narrative_labs: list[LabHit],
        coded_term: str | None,
    ) -> EventObject:
        evidence: list[Span] = []
        confidence: dict[str, float] = {}
        match_kinds: set[str] = {m.kind for m, _ in mention_group}
        if not mention_group:
            match_kinds.add("contextual")
        ae_doc = _ae_doc_id(ae_row)
        ae_text = _render_ae_row(ae_row)

        def add_span(field: str, start: int, end: int, value: str, *, doc=doc_id, source=text):
            evidence.append(
                Span(
                    doc_id=doc, start=start, end=end, field=field,
                    extracted_value=value, text=source[start:end],
                )
            )

        # -- concept identity ------------------------------------------------
        if mention_group:
            for mention, _result in mention_group:
                add_span("concept_id", mention.start, mention.end, concept_id)
            confidence["concept_id"] = round(
                max(m.confidence for m, _ in mention_group), 4
            )
        else:
            anchor = contextual[1] if contextual else None
            if anchor:
                add_span("concept_id", anchor[0], anchor[1], concept_id)
            else:  # pragma: no cover - a candidate always has an anchor
                add_span("concept_id", 0, 0, concept_id, doc=ae_doc, source=ae_text)
            confidence["concept_id"] = self.config.confidence_for("abbreviation_gated", 0.7)

        # -- assertion --------------------------------------------------------
        if mention_group:
            best = max(mention_group, key=lambda pair: pair[1].confidence)
            result = best[1]
            if result.cue_start is not None and result.cue_end is not None:
                add_span("assertion", result.cue_start, result.cue_end, assertion)
            else:
                add_span("assertion", best[0].start, best[0].end, assertion)
            confidence["assertion"] = round(result.confidence, 4)
        else:
            result = contextual[0] if contextual else None
            anchor = contextual[1] if contextual else None
            if result and result.cue_start is not None and result.cue_end is not None:
                add_span("assertion", result.cue_start, result.cue_end, assertion)
            elif anchor:
                add_span("assertion", anchor[0], anchor[1], assertion)
            confidence["assertion"] = round(
                result.confidence if result else
                self.config.confidence_for("assertion_default", 0.8), 4
            )

        # -- coded term (preserved verbatim) ----------------------------------
        coded_version = ae_row.get("AEDICTVER") or None
        if self.matcher.coded_term_concept(coded_term) == concept_id:
            match_kinds.add("coded_term")
        if coded_term:
            add_span("coded_term", 0, len(ae_text), coded_term, doc=ae_doc, source=ae_text)
            confidence["coded_term"] = self.config.confidence_for("coded_term_exact", 0.99)

        # -- symptoms ---------------------------------------------------------
        symptom_mentions: list[SymptomMention] = []
        for symptom, start, end, surface in symptoms:
            span = Span(
                doc_id=doc_id, start=start, end=end, field="symptoms",
                extracted_value=symptom, text=surface,
            )
            symptom_mentions.append(SymptomMention(symptom=symptom, span=span))
            evidence.append(span)

        # -- temporal ---------------------------------------------------------
        onset = self.temporal.resolve(
            subject_id=context.subject_id,
            text=text,
            scope=None,
            default_anchor=self.default_anchor,
            recorded_onset=ae_row.get("AESTDTC"),
            reference_start=context.reference_start,
        )
        if onset.onset_date is not None:
            if onset.source == "structured_onset_date":
                add_span("onset_date", 0, len(ae_text), onset.onset_date.isoformat(),
                         doc=ae_doc, source=ae_text)
            elif onset.mention is not None:
                add_span("onset_date", onset.mention.start, onset.mention.end,
                         onset.onset_date.isoformat())
            confidence["onset_date"] = round(onset.confidence, 4)
        if onset.onset_offset_days is not None:
            if onset.mention is not None:
                add_span("onset_offset_days", onset.mention.start, onset.mention.end,
                         str(onset.onset_offset_days))
            else:
                add_span("onset_offset_days", 0, len(ae_text),
                         str(onset.onset_offset_days), doc=ae_doc, source=ae_text)
            confidence["onset_offset_days"] = round(onset.confidence, 4)

        # -- labs -------------------------------------------------------------
        lab_values: list[LabValue] = []
        structured_labs = self.values.labs_from_lb(context.lb_rows, onset.onset_date)
        for hit in list(narrative_labs) + structured_labs:
            if hit.implausible:
                continue
            if hit.source == "narrative":
                span = Span(
                    doc_id=doc_id, start=hit.start, end=hit.end, field="labs",
                    extracted_value=f"{hit.test}={hit.value}{hit.unit}",
                    text=text[hit.start : hit.end],
                )
            else:
                span = Span(
                    doc_id=hit.source, start=hit.start, end=hit.end, field="labs",
                    extracted_value=f"{hit.test}={hit.value}{hit.unit}",
                    text=hit.surface,
                )
            lab_values.append(
                LabValue(
                    test=hit.test, value=hit.value, unit=hit.unit,
                    canonical_value=hit.canonical_value,
                    canonical_unit=hit.canonical_unit,
                    collection_date=hit.collection_date, span=span,
                )
            )
            evidence.append(span)
        if lab_values:
            confidence["labs"] = round(
                max(h.confidence for h in list(narrative_labs) + structured_labs), 4
            )

        # -- severity and seriousness, kept strictly apart ---------------------
        severity = self._scalar_field(
            "severity", ae_row, text, ae_doc, ae_text, add_span, confidence
        )
        seriousness = self._seriousness(
            ae_row, text, ae_doc, ae_text, add_span, confidence
        )
        relatedness = self._scalar_field(
            "relatedness", ae_row, text, ae_doc, ae_text, add_span, confidence
        )
        action_taken = self._scalar_field(
            "action_taken", ae_row, text, ae_doc, ae_text, add_span, confidence
        )
        outcome = self._scalar_field(
            "outcome", ae_row, text, ae_doc, ae_text, add_span, confidence
        )

        # Rechallenge is narrative-only: it is not consistently structured
        # across studies, and in this corpus it is never in the AE table.
        rechallenge = None
        hit = self.values.single_value(text, "rechallenge")
        if hit:
            rechallenge = hit.value
            add_span("rechallenge", hit.start, hit.end, hit.value)
            confidence["rechallenge"] = round(hit.confidence, 4)

        rescue = False
        rescue_hit = self.values.rescue_treatment(text)
        if rescue_hit:
            rescue = True
            add_span("rescue_treatment", rescue_hit.start, rescue_hit.end, "true")
            confidence["rescue_treatment"] = round(rescue_hit.confidence, 4)
        else:
            cm = self._rescue_from_cm(context, onset.onset_date)
            if cm is not None:
                rescue = True
                rendered, cm_doc = cm
                evidence.append(
                    Span(
                        doc_id=cm_doc, start=0, end=len(rendered),
                        field="rescue_treatment", extracted_value="true", text=rendered,
                    )
                )
                confidence["rescue_treatment"] = self.config.confidence_for(
                    "lab_with_unit", 0.95
                )

        event_id = f"{doc_id}::{concept_id}::{assertion}"
        return EventObject(
            event_id=event_id,
            subject_id=context.subject_id,
            study_id=context.study_id,
            doc_id=doc_id,
            source_record_id=ae_doc,
            concept_id=concept_id,
            coded_term=coded_term,
            coded_term_version=coded_version if coded_term else None,
            verbatim_term=(ae_row.get("AETERM") or None),
            assertion=assertion,  # type: ignore[arg-type]
            concept_match_kinds=sorted(match_kinds),
            symptoms=symptom_mentions,
            labs=lab_values,
            onset_date=onset.onset_date,
            onset_offset_days=onset.onset_offset_days,
            anchor_event=onset.anchor_event,
            anchor_date=onset.anchor_date,
            severity=severity,  # type: ignore[arg-type]
            seriousness=seriousness,  # type: ignore[arg-type]
            relatedness=relatedness,  # type: ignore[arg-type]
            action_taken=action_taken,  # type: ignore[arg-type]
            rechallenge=rechallenge,  # type: ignore[arg-type]
            rescue_treatment=rescue,
            outcome=outcome,  # type: ignore[arg-type]
            evidence=sorted(evidence, key=lambda s: (s.field, s.doc_id, s.start, s.end)),
            confidence=dict(sorted(confidence.items())),
            extractor_version=self.extractor_version,
        )

    # -- field helpers ------------------------------------------------------

    def _scalar_field(
        self, field, ae_row, text, ae_doc, ae_text, add_span, confidence
    ) -> str | None:
        """Structured column first, narrative cue where the column is blank."""
        structured = self.values.from_ae_row(ae_row, field)
        if structured is not None:
            add_span(field, 0, len(ae_text), structured, doc=ae_doc, source=ae_text)
            confidence[field] = self.config.confidence_for("coded_term_exact", 0.99)
            return structured
        hit = self.values.single_value(text, field)
        if hit is None:
            return None
        add_span(field, hit.start, hit.end, hit.value)
        confidence[field] = round(hit.confidence, 4)
        return hit.value

    def _seriousness(
        self, ae_row, text, ae_doc, ae_text, add_span, confidence
    ) -> list[str]:
        structured = self.values.seriousness_from_ae_row(ae_row)
        if structured:
            for value in structured:
                add_span("seriousness", 0, len(ae_text), value, doc=ae_doc, source=ae_text)
            confidence["seriousness"] = self.config.confidence_for("coded_term_exact", 0.99)
            return structured
        hits = self.values.multi_value(text, "seriousness")
        for hit in hits:
            add_span("seriousness", hit.start, hit.end, hit.value)
        if hits:
            confidence["seriousness"] = round(max(h.confidence for h in hits), 4)
        return sorted({h.value for h in hits})

    def _rescue_from_cm(
        self, context: SubjectContext, onset_date: _dt.date | None
    ) -> tuple[str, str] | None:
        """Rescue medication recorded in the CM domain around the event."""
        if onset_date is None:
            return None
        cues = (self.config.values.get("rescue_treatment") or {}).get("cues") or []
        folded_cues = [c.lower() for c in cues]
        for row in sorted(context.cm_rows, key=lambda r: str(r.get("CMSEQ"))):
            started = parse_date(row.get("CMSTDTC"))
            if started is None or abs((started - onset_date).days) > 1:
                continue
            blob = f"{row.get('CMTRT') or ''} {row.get('CMINDC') or ''}".lower()
            if any(cue in blob for cue in folded_cues):
                rendered = (
                    f"CM {row.get('CMTRT')} for {row.get('CMINDC')} "
                    f"on {started.isoformat()}"
                )
                return rendered, f"CM:{row.get('USUBJID')}:{row.get('CMSEQ')}"
        return None

    # -- corpus level -------------------------------------------------------

    def extract_store(self, store: TrialStore) -> list[EventObject]:
        """Every event object in a data snapshot, in a stable order."""
        contexts: dict[str, SubjectContext] = {}
        for subject_id in store.subjects():
            dm_rows = store.subject_rows(subject_id, "dm")
            reference = parse_date(dm_rows[0].get("RFSTDTC")) if dm_rows else None
            contexts[subject_id] = SubjectContext(
                subject_id=subject_id,
                study_id=store.study_of(subject_id) or "",
                reference_start=reference,
                lb_rows=store.subject_rows(subject_id, "lb"),
                cm_rows=store.subject_rows(subject_id, "cm"),
            )

        events: list[EventObject] = []
        for ae_row, narrative in store.iter_ae_with_narrative():
            subject_id = str(ae_row.get("USUBJID"))
            context = contexts.get(subject_id) or SubjectContext(
                subject_id=subject_id, study_id=str(ae_row.get("STUDYID") or "")
            )
            events.extend(self.extract_record(ae_row, narrative, context))
        return sorted(events, key=lambda e: (e.subject_id, e.doc_id, e.event_id))


def extract_corpus(
    store: TrialStore,
    concepts_path: str | None = None,
    extraction_path: str | None = None,
) -> tuple[list[EventObject], str]:
    """Extract a whole snapshot. Returns the events and the extractor version."""
    catalog, config, version = load_configs(concepts_path, extraction_path)
    resolver = store.anchor_resolver(config.anchors)
    engine = ExtractionEngine(catalog, config, version, resolver)
    return engine.extract_store(store), version
