"""Two paths, and the guard between them."""

from __future__ import annotations

import pytest

from aelayer.catalog import ConfigError
from aelayer.retrieval.query import CandidateInCohort, discover, expand_concept, retrieve


# -- the guard --------------------------------------------------------------


def test_discovery_results_are_candidates(index, catalog):
    result = discover(index, catalog, concept="HYPOGLYCEMIA", top_k=20)
    assert result.mentions
    assert all(m.candidate for m in result.mentions)


def test_a_discovery_candidate_cannot_enter_a_cohort(index, catalog):
    """Definition of done, and it refuses rather than filtering quietly."""
    result = discover(index, catalog, concept="HYPOGLYCEMIA", top_k=20)
    with pytest.raises(CandidateInCohort, match="adjudication"):
        result.as_cohort()


def test_a_non_precise_episode_search_is_also_refused(index, catalog):
    result = retrieve(index, catalog, concept="HYPOGLYCEMIA", mode="lexical", top_k=5)
    assert all(r.candidate for r in result.records)
    with pytest.raises(CandidateInCohort):
        result.as_cohort()


def test_the_precise_path_yields_a_usable_cohort(index, catalog):
    result = retrieve(index, catalog, concept="HYPOGLYCEMIA", mode="precise", top_k=1000)
    assert result.records
    assert result.as_cohort() == result.records


# -- assertion --------------------------------------------------------------


def test_assertion_present_returns_no_documented_absence(index, catalog):
    result = discover(index, catalog, concept="HYPOGLYCEMIA",
                      assertion=["present"], top_k=1000)
    assert result.mentions
    assert result.negation_false_positives == 0


def test_without_the_filter_negated_mentions_come_back(index, catalog):
    result = discover(index, catalog, concept="HYPOGLYCEMIA", top_k=1000)
    assert result.negation_false_positives > 0
    assert result.negation_false_positive_rate > 0


def test_the_filter_is_the_only_difference(index, catalog):
    on = discover(index, catalog, concept="HYPOGLYCEMIA",
                  assertion=["present"], top_k=1000)
    off = discover(index, catalog, concept="HYPOGLYCEMIA", top_k=1000)
    assert on.negation_false_positive_rate == 0.0
    assert off.negation_false_positive_rate > on.negation_false_positive_rate


def test_absence_is_queryable_in_its_own_right(index, catalog):
    result = discover(index, catalog, concept="HYPOGLYCEMIA",
                      assertion=["absent"], top_k=1000)
    assert result.mentions
    assert {m.assertion for m in result.mentions} == {"absent"}


def test_every_assertion_class_is_present_in_the_index(index):
    seen = {r["assertion"] for r in index.query("SELECT assertion FROM mentions")}
    assert {"present", "absent"} <= seen
    assert len(seen) >= 3


def test_dense_degrades_to_lexical_and_says_so(index, catalog):
    result = discover(index, catalog, concept="HYPOGLYCEMIA",
                      assertion=["present"], mode="dense", top_k=10)
    assert result.mode == "lexical"
    assert any("degraded to lexical" in n for n in result.notes)
    assert result.negation_false_positives == 0


# -- expansion --------------------------------------------------------------


def test_expansion_uses_catalogue_membership(catalog):
    ids, terms = expand_concept(catalog, "HYPOGLYCEMIA")
    assert ids == ["HYPOGLYCEMIA"]
    assert "hypoglycaemia" in terms
    assert "Blood glucose decreased" in terms
    assert "hyperglycemia" not in terms


def test_a_group_is_an_explicit_list(catalog):
    ids, _ = expand_concept(catalog, None, group="GLYCEMIC_EVENTS")
    assert ids == ["HYPERGLYCEMIA", "HYPOGLYCEMIA"]
    with pytest.raises(ConfigError):
        expand_concept(catalog, None, group="EVERYTHING_ENDOCRINE")


# -- structured predicates --------------------------------------------------


def test_filters_compose_on_the_precise_path(index, catalog, definition_v1):
    result = retrieve(
        index, catalog, concept="HYPOGLYCEMIA", verdict=["case"],
        window=(0, 14), studies=index.studies(),
        definition_id=definition_v1.id, definition_version=definition_v1.version,
        mode="precise", top_k=1000,
    )
    for record in result.records:
        assert record.verdict == "case"
        assert 0 <= record.onset_offset_days <= 14


def test_the_window_filter_actually_selects(index, catalog):
    """Every episode carries an offset, so a window is a real filter."""
    narrow = retrieve(index, catalog, concept="HYPOGLYCEMIA", window=(0, 3),
                      mode="precise", top_k=1000)
    wide = retrieve(index, catalog, concept="HYPOGLYCEMIA", window=(0, 60),
                    mode="precise", top_k=1000)
    assert narrow.records
    assert len(narrow.records) < len(wide.records)
    assert all(0 <= r.onset_offset_days <= 3 for r in narrow.records)


def test_provenance_is_a_predicate(index, catalog):
    result = retrieve(index, catalog, concept="HYPOGLYCEMIA",
                      provenance=["text"], mode="precise", top_k=1000)
    assert result.records
    assert all("text" in r.provenance_paths for r in result.records)


def test_representation_is_a_predicate(index, catalog):
    result = retrieve(index, catalog, representation=["V-D"], mode="precise", top_k=1000)
    assert result.records
    assert {r.representation for r in result.records} == {"V-D"}
    assert {r.study_id for r in result.records} == {"STUDY-04"}


def test_linkage_review_is_a_predicate(index, catalog):
    flagged = retrieve(index, catalog, linkage_review=True, mode="precise", top_k=1000)
    assert flagged.records
    assert all(r.linkage_review for r in flagged.records)


def test_episodes_round_trip_through_the_store(index, episodes):
    stored = {e.episode_id: e for e in index.episodes()}
    assert len(stored) == len(episodes)
    for episode in episodes:
        assert stored[episode.episode_id].model_dump_json() == episode.model_dump_json()


def test_the_index_knows_when_it_is_stale(index, pipeline):
    meta = index.meta()
    assert meta.matches(
        pipeline.snapshot_id, pipeline.extractor_version, pipeline.normalizer_version
    )
    assert not meta.matches("other", pipeline.extractor_version, pipeline.normalizer_version)
    assert not meta.matches(pipeline.snapshot_id, "other", pipeline.normalizer_version)
    assert not meta.matches(pipeline.snapshot_id, pipeline.extractor_version, "other")


def test_the_index_is_usable_from_several_threads(index, catalog):
    import concurrent.futures as futures

    with futures.ThreadPoolExecutor(6) as pool:
        results = [
            f.result() for f in [
                pool.submit(discover, index, catalog, concept="HYPOGLYCEMIA",
                            assertion=["present"], top_k=5)
                for _ in range(12)
            ]
        ]
    assert {len(r.mentions) for r in results} == {len(results[0].mentions)}
