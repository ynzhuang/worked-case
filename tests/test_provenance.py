"""Definition of done: no populated field lacks a span."""

from __future__ import annotations


def test_every_populated_record_field_carries_a_span(records):
    offenders = {
        r.source_record_id: r.missing_provenance()
        for r in records if not r.has_full_provenance()
    }
    assert offenders == {}


def test_text_spans_index_the_characters_they_claim(records, store):
    checked = 0
    for record in records:
        narrative = store.narratives.get(record.narrative_doc_id or "")
        if narrative is None:
            continue
        text = narrative.full_text
        for span in record.evidence:
            if span.kind != "text" or span.doc_id != narrative.doc_id:
                continue
            assert span.end <= len(text)
            if span.text:
                assert text[span.start : span.end] == span.text
                checked += 1
    assert checked > 0


def test_structured_spans_point_at_a_rendered_row(records):
    checked = 0
    for record in records:
        for span in record.evidence:
            if span.kind != "structured":
                continue
            assert span.doc_id
            assert span.text, f"{span.doc_id}:{span.field} renders to nothing"
            assert span.end == len(span.text) or span.end <= len(span.text)
            checked += 1
    assert checked > 0


def test_every_episode_carries_the_spans_of_its_records(episodes, records):
    by_id = {r.source_record_id: r for r in records}
    for episode in episodes:
        assert episode.linked_evidence
        record_spans = {
            s.key() for rid in episode.source_record_ids
            for s in by_id[rid].evidence
        }
        assert {s.key() for s in episode.linked_evidence} >= record_spans


def test_cases_carry_the_spans_that_made_them_cases(pipeline, definition_v1):
    cases = [a for a in pipeline.evaluate(definition_v1) if a.verdict == "case"]
    assert cases
    assert all(a.evidence_spans for a in cases)
