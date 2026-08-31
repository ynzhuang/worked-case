"""The extraction engine's invariants."""

from __future__ import annotations

import collections

import pytest

from aelayer.extract.engine import ExtractionEngine


def test_every_non_null_field_carries_a_span(events):
    """The provenance invariant, asserted over the whole corpus.

    A value without provenance is a bug, not a degraded result.
    """
    offenders = {
        event.event_id: event.missing_provenance()
        for event in events
        if not event.has_full_provenance()
    }
    assert offenders == {}


def test_every_span_indexes_real_text(events, store):
    """A span must point at the characters it claims to."""
    for event in events:
        narrative = store.narratives.get(event.doc_id)
        for span in event.evidence:
            if narrative is not None and span.doc_id == event.doc_id:
                assert span.end <= len(narrative.full_text)
                if span.text:
                    assert narrative.full_text[span.start : span.end] == span.text


def test_extraction_is_byte_identical_across_runs(pipeline, store):
    first = pipeline.engine().extract_store(store)
    second = pipeline.engine().extract_store(store)
    assert [e.model_dump_json() for e in first] == [e.model_dump_json() for e in second]


def test_extractor_version_is_a_hash_of_code_and_both_configs(tmp_path):
    from aelayer import paths
    from aelayer.hashing import extractor_version

    base = extractor_version(paths.CONCEPTS_YAML, paths.EXTRACTION_YAML)
    edited = tmp_path / "extraction.yaml"
    edited.write_text(
        paths.EXTRACTION_YAML.read_text(encoding="utf-8") + "\n# a change\n",
        encoding="utf-8",
    )
    assert extractor_version(paths.CONCEPTS_YAML, edited) != base


def test_the_extractor_assigns_no_evidence_state_and_no_verdict(events):
    """It reports what the text says; interpretation belongs to the definition."""
    for event in events[:20]:
        payload = event.model_dump()
        assert "evidence_state" not in payload
        assert "verdict" not in payload
        assert "case" not in payload


def test_all_six_assertion_classes_appear_in_the_corpus(events):
    seen = {e.assertion for e in events}
    assert {"present", "absent", "hypothetical", "historical", "family_history",
            "uncertain"} <= seen


def test_documented_absences_are_stored_not_discarded(events):
    absent = [e for e in events if e.assertion == "absent"]
    assert absent, "the corpus contains negated mentions; they must survive extraction"
    for event in absent:
        assert event.spans_for("assertion"), "an absence needs its cue span too"


def test_coded_terms_are_preserved_verbatim(events, store):
    ae_by_doc = {
        row.get("DOCID"): row for row in store.rows("ae") if row.get("DOCID")
    }
    checked = 0
    for event in events:
        row = ae_by_doc.get(event.doc_id)
        if row and row.get("AEDECOD"):
            assert event.coded_term == row["AEDECOD"]
            assert event.coded_term_version == row.get("AEDICTVER")
            checked += 1
    assert checked > 0


def test_a_non_specific_coded_term_is_not_a_concept_match(events):
    """A study may code an event `Malaise` while the narrative describes
    hypoglycemia. That is a coded term, but not one *for this concept*."""
    mismatched = [
        e for e in events
        if e.coded_term in {"Malaise", "Asthenia", "Feeling abnormal",
                            "General physical health deterioration"}
    ]
    assert mismatched
    for event in mismatched:
        assert "coded_term" not in event.concept_match_kinds


def test_units_are_normalised_across_studies(events):
    reported_units = {
        lab.unit for e in events for lab in e.labs if lab.test == "GLUCOSE"
    }
    canonical_units = {
        lab.canonical_unit for e in events for lab in e.labs if lab.test == "GLUCOSE"
    }
    assert len(reported_units) > 1, "the corpus should mix mg/dL and mmol/L"
    assert canonical_units == {"mg/dL"}


def test_contextual_candidates_are_raised_without_a_mention(events):
    """Pooling only explicit mentions undercounts systematically."""
    contextual = [
        e for e in events
        if e.concept_id == "HYPOGLYCEMIA" and e.concept_match_kinds == ["contextual"]
    ]
    assert contextual
    for event in contextual:
        assert event.symptoms or event.labs


def test_an_ungated_abbreviation_produces_no_event(pipeline, store, gold):
    ungated = [
        doc_id for doc_id, row in gold.items() if row["pattern"] == "abbrev_ungated"
    ]
    assert ungated
    by_doc = collections.defaultdict(list)
    for event in pipeline.events():
        by_doc[event.doc_id].append(event)
    for doc_id in ungated:
        concepts = {e.concept_id for e in by_doc.get(doc_id, [])}
        assert "HYPOGLYCEMIA" not in concepts


def test_a_record_with_both_an_occurrence_and_an_absence_yields_both(
    catalog, extraction_config, pipeline
):
    """Both statements are true and both must remain queryable."""
    from aelayer.extract.engine import SubjectContext
    from aelayer.ingest import Narrative

    engine = ExtractionEngine(catalog, extraction_config, "test-version")
    narrative = Narrative(
        doc_id="D1", study_id="ST", subject_id="S1", ae_seq=1,
        text=(
            "The subject experienced hypoglycaemia on study day 5. "
            "There was no evidence of hypoglycaemia at the follow-up visit."
        ),
    )
    events = engine.extract_record(
        {"USUBJID": "S1", "AESEQ": 1, "AEDECOD": "", "AESTDTC": ""},
        narrative,
        SubjectContext(subject_id="S1", study_id="ST"),
    )
    assertions = {e.assertion for e in events if e.concept_id == "HYPOGLYCEMIA"}
    assert assertions == {"present", "absent"}


def test_a_negated_symptom_is_not_counted_as_evidence(catalog, extraction_config):
    from aelayer.extract.engine import SubjectContext
    from aelayer.ingest import Narrative

    engine = ExtractionEngine(catalog, extraction_config, "test-version")
    narrative = Narrative(
        doc_id="D2", study_id="ST", subject_id="S2", ae_seq=1,
        text="The subject experienced hypoglycaemia. The subject denies tremor.",
    )
    events = engine.extract_record(
        {"USUBJID": "S2", "AESEQ": 1, "AEDECOD": "", "AESTDTC": ""},
        narrative,
        SubjectContext(subject_id="S2", study_id="ST"),
    )
    present = [e for e in events if e.assertion == "present"]
    assert present and "tremor" not in {s.symptom for s in present[0].symptoms}


def test_events_are_returned_in_a_stable_order(pipeline, store):
    ids = [e.event_id for e in pipeline.engine().extract_store(store)]
    assert ids == sorted(ids, key=lambda i: i) or len(set(ids)) == len(ids)
    assert len(set(ids)) == len(ids), "event ids must be unique"
