"""Retrieval: concept expansion and structured predicates.

The headline claim is that assertion is a column, not a hope about what an
embedding encodes. These tests hold it to that.
"""

from __future__ import annotations

import pytest

from aelayer.catalog import ConfigError
from aelayer.retrieval.query import expand_concept, retrieve


def test_assertion_present_returns_no_documented_absence(index, catalog):
    result = retrieve(index, catalog, concept="HYPOGLYCEMIA",
                      assertion=["present"], top_k=1000)
    assert result.records
    assert [r for r in result.records if r.assertion == "absent"] == []
    assert result.negation_false_positive_rate == 0.0


def test_without_the_filter_negated_records_come_back(index, catalog):
    result = retrieve(index, catalog, concept="HYPOGLYCEMIA", top_k=1000)
    absent = [r for r in result.records if r.assertion == "absent"]
    assert absent, "the corpus contains negated mentions; unfiltered they appear"
    assert result.negation_false_positive_rate > 0


def test_the_filter_is_what_makes_the_difference(index, catalog):
    """Same query, same scorer, one structured predicate apart."""
    on = retrieve(index, catalog, concept="HYPOGLYCEMIA",
                  assertion=["present"], top_k=1000)
    off = retrieve(index, catalog, concept="HYPOGLYCEMIA", top_k=1000)
    assert on.negation_false_positives == 0
    assert off.negation_false_positives > on.negation_false_positives


def test_absence_is_queryable_in_its_own_right(index, catalog):
    """A documented absence is a fact, not a low-confidence occurrence."""
    result = retrieve(index, catalog, concept="HYPOGLYCEMIA",
                      assertion=["absent"], top_k=1000)
    assert result.records
    assert {r.assertion for r in result.records} == {"absent"}


def test_concept_expansion_uses_catalogue_synonyms_and_coded_terms(catalog):
    ids, terms = expand_concept(catalog, "HYPOGLYCEMIA")
    assert ids == ["HYPOGLYCEMIA"]
    assert "hypoglycaemia" in terms and "hypoglycemia" in terms
    assert "Blood glucose decreased" in terms      # a coded term
    assert "hyperglycemia" not in terms            # a different concept


def test_a_group_expands_by_explicit_membership_only(catalog):
    ids, _terms = expand_concept(catalog, None, group="GLYCEMIC_EVENTS")
    assert ids == ["HYPERGLYCEMIA", "HYPOGLYCEMIA"]


def test_an_unknown_group_is_an_error_not_an_empty_result(catalog):
    with pytest.raises(ConfigError):
        expand_concept(catalog, None, group="NOT_A_GROUP")


def test_group_retrieval_returns_both_members(index, catalog):
    result = retrieve(index, catalog, group="GLYCEMIC_EVENTS",
                      assertion=["present"], top_k=1000)
    assert {r.concept_id for r in result.records} <= {"HYPOGLYCEMIA", "HYPERGLYCEMIA"}


def test_the_window_filter_bounds_the_offset(index, catalog):
    result = retrieve(index, catalog, concept="HYPOGLYCEMIA", assertion=["present"],
                      window=(0, 14), anchor="dose_escalation", top_k=1000)
    assert result.records
    assert all(0 <= r.onset_offset_days <= 14 for r in result.records)


def test_the_study_filter_restricts_to_named_studies(index, catalog):
    studies = index.studies()[:1]
    result = retrieve(index, catalog, concept="HYPOGLYCEMIA", studies=studies, top_k=1000)
    assert {r.study_id for r in result.records} == set(studies)


def test_the_evidence_state_filter_is_keyed_to_a_definition_version(index, catalog):
    result = retrieve(
        index, catalog, concept="HYPOGLYCEMIA", assertion=["present"],
        evidence_state=["explicit"], definition_id="te_symptomatic_hypoglycemia",
        definition_version=1, top_k=1000,
    )
    assert result.records
    assert {r.evidence_state for r in result.records} == {"explicit"}


def test_filters_compose(index, catalog):
    """A scientific question decomposes; so does the query."""
    result = retrieve(
        index, catalog, concept="HYPOGLYCEMIA", assertion=["present"],
        evidence_state=["explicit", "supported"], window=(0, 14),
        anchor="dose_escalation", studies=index.studies(),
        definition_id="te_symptomatic_hypoglycemia", definition_version=1,
        top_k=1000,
    )
    for record in result.records:
        assert record.assertion == "present"
        assert record.evidence_state in {"explicit", "supported"}
        assert 0 <= record.onset_offset_days <= 14


def test_top_k_is_honoured(index, catalog):
    assert len(retrieve(index, catalog, concept="HYPOGLYCEMIA", top_k=3).records) <= 3


def test_free_text_search_works_without_a_concept(index, catalog):
    result = retrieve(index, catalog, text="diaphoresis", top_k=20)
    assert result.records


def test_results_carry_a_snippet_and_a_score(index, catalog):
    result = retrieve(index, catalog, concept="HYPOGLYCEMIA", top_k=5)
    for record in result.records:
        assert record.snippet
        assert isinstance(record.score, float)


def test_dense_mode_degrades_to_lexical_when_no_model_is_present(index, catalog):
    """The default path must run with the network cable pulled."""
    result = retrieve(index, catalog, concept="HYPOGLYCEMIA",
                      assertion=["present"], mode="dense", top_k=10)
    assert result.mode == "lexical"
    assert not result.dense_available
    assert any("degraded to lexical" in note for note in result.notes)
    # And the structured predicate still held while it degraded.
    assert result.negation_false_positives == 0


def test_the_index_knows_when_it_is_stale(index, pipeline):
    meta = index.meta()
    assert meta.matches(pipeline.snapshot_id, pipeline.extractor_version)
    assert not meta.matches("a-different-snapshot", pipeline.extractor_version)
    assert not meta.matches(pipeline.snapshot_id, "extract-9.9.9")


def test_events_round_trip_through_the_store(index, events):
    stored = {e.event_id: e for e in index.events()}
    assert len(stored) == len(events)
    for event in events:
        assert stored[event.event_id].model_dump_json() == event.model_dump_json()


def test_assignments_round_trip_through_the_store(index, pipeline, definition_v1):
    stored = index.assignments("te_symptomatic_hypoglycemia", 1)
    assert stored
    live = pipeline.evaluate(definition_v1)
    assert [a.model_dump_json() for a in stored] == [a.model_dump_json() for a in live]


def test_the_index_is_usable_from_several_threads(index, catalog):
    """The API serves sync endpoints from a threadpool."""
    import concurrent.futures as futures

    with futures.ThreadPoolExecutor(6) as pool:
        results = [
            future.result()
            for future in [
                pool.submit(retrieve, index, catalog, concept="HYPOGLYCEMIA",
                            assertion=["present"], top_k=5)
                for _ in range(12)
            ]
        ]
    assert {len(r.records) for r in results} == {len(results[0].records)}
