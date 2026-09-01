"""Episode derivation is additive, and its assumptions are declared."""

from __future__ import annotations

import collections
import datetime as _dt


from aelayer.episode import EpisodeReconciler, ReconciliationConfig


def test_source_records_survive_derivation_unmodified(pipeline, store, records):
    """The lower level is the authority and is never edited in place."""
    from aelayer.normalize import normalize_store

    fresh = {r.source_record_id: r for r in normalize_store(store, pipeline.configs)}
    for record in records:
        original = fresh[record.source_record_id]
        # Extraction fills unresolved fields; nothing collected ever changes.
        for name, field in record.fields().items():
            prior = original.fields()[name]
            if prior.collection_state == "collected":
                assert field.value == prior.value, name
                assert field.collection_state == "collected"


def test_no_record_is_lost_or_duplicated(episodes, records):
    assigned = [rid for e in episodes for rid in e.source_record_ids]
    assert sorted(assigned) == sorted(r.source_record_id for r in records)
    assert len(assigned) == len(set(assigned))


def test_ambiguous_linkage_is_flagged_not_resolved(episodes):
    flagged = [e for e in episodes if e.linkage_review_required]
    assert flagged
    for episode in flagged:
        assert episode.linkage_note, "a flag without a reason is not a flag"


def test_a_recurrent_concept_splits_by_default(pipeline, catalog, semantics):
    """Two hypoglycemia records days apart are two episodes, not one."""
    assert catalog.concept("HYPOGLYCEMIA").recurrence_expected
    splits = [
        e for e in pipeline.episodes() if e.linkage_rule == "recurrence_split"
    ]
    multi_hypo = [
        e for e in pipeline.episodes()
        if e.standardized_concept == "HYPOGLYCEMIA" and len(e.source_record_ids) > 1
    ]
    # Where hypoglycemia records do merge, it is by an explicit continuation or
    # a declared convention, never by a bare temporal gap.
    for episode in multi_hypo:
        assert episode.linkage_rule in (
            "explicit_continuation", "declared_convention", "temporal_overlap"
        ), episode.linkage_rule


def test_a_non_recurrent_concept_merges_across_a_gap(pipeline, catalog):
    assert not catalog.concept("ANAEMIA").recurrence_expected
    chronic = [
        e for e in pipeline.episodes() if e.standardized_concept == "ANAEMIA"
    ]
    assert chronic
    merged = [e for e in chronic if len(e.source_record_ids) > 1]
    assert merged, "a chronic condition should merge its grade changes"


def test_an_explicit_continuation_beats_everything(pipeline, records):
    linked = [r for r in records if r.continuation_of]
    assert linked
    by_episode = {
        rid: e for e in pipeline.episodes() for rid in e.source_record_ids
    }
    for record in linked:
        episode = by_episode[record.source_record_id]
        assert record.continuation_of in episode.source_record_ids


def test_a_declared_splitting_convention_links_records(pipeline, semantics):
    assert semantics.for_study("STUDY-02").splits_on_severity_change()
    rules = collections.Counter(e.linkage_rule for e in pipeline.episodes())
    assert rules["declared_convention"] > 0


def test_episode_boundaries_agree_with_gold(pipeline, store):
    gold = {frozenset(g["source_record_ids"]) for g in store.gold_episodes()}
    predicted = {frozenset(e.source_record_ids) for e in pipeline.episodes()}
    agreement = len(gold & predicted) / len(gold)
    assert agreement > 0.9, agreement


def test_an_unresolvable_onset_does_not_get_placed_by_guesswork(
    pipeline, catalog, semantics
):
    """Two records that might be one episode, and no dates to tell. Both stay."""
    from aelayer.models import Field

    subject = pipeline.records()[0].subject_id
    records = [
        r.model_copy(deep=True)
        for r in pipeline.records() if r.subject_id == subject
    ][:2]
    if len(records) < 2:
        records = [r.model_copy(deep=True) for r in pipeline.records()[:2]]
        records[1].subject_id = records[0].subject_id
        records[1].study_id = records[0].study_id
        records[1].standardized_concept = records[0].standardized_concept
    for record in records:
        record.onset_datetime = Field[_dt.datetime](
            collection_state="unknown",
            note="cleared for this test: nothing says when this started",
        )
        record.continuation_of = None

    reconciler = EpisodeReconciler(catalog, semantics, ReconciliationConfig())
    episodes = reconciler.reconcile(records)
    assert len(episodes) == 2
    assert all(len(e.source_record_ids) == 1 for e in episodes)
    assert any(e.linkage_review_required for e in episodes)


def test_field_states_carry_the_most_informative_reason(episodes):
    seen = collections.Counter(
        state for e in episodes for state in e.field_states.values()
    )
    assert seen["collected"] > 0
    assert seen["not_collected_by_protocol"] > 0
    assert seen["not_representable"] > 0


def test_every_episode_is_stamped_with_its_offset_from_the_anchor(episodes):
    """So a window filter can be applied without re-running a definition."""
    stamped = [e for e in episodes if e.onset_offset_days.value is not None]
    assert len(stamped) == len(episodes)
    for episode in stamped:
        assert episode.anchor_event == "dose_escalation"
        assert episode.onset_offset_days.collection_state == "collected"
        assert episode.anchor_datetime is not None


def test_an_offset_that_cannot_be_resolved_is_unknown_not_zero(catalog, semantics,
                                                              records):
    """A filter that trusts a defaulted zero is worse than one that finds nothing."""
    reconciler = EpisodeReconciler(catalog, semantics, ReconciliationConfig())
    episodes = reconciler.reconcile(records[:3])
    for episode in episodes:
        assert episode.onset_offset_days.value is None
        assert episode.onset_offset_days.collection_state == "unknown"
        assert episode.onset_offset_days.note
        assert episode.anchor_event is None
