"""Ingest and snapshot hashing."""

from __future__ import annotations

import pytest

from aelayer.hashing import hash_payload, hash_text
from aelayer.ingest import IngestError, load_store


def test_the_store_reports_what_it_holds(store):
    summary = store.summary()
    assert summary["studies"] >= 2
    assert summary["ae_records"] > 0
    assert summary["narratives"] == summary["ae_records"]


def test_subjects_and_studies_are_indexed(store):
    subject = store.subjects()[0]
    assert store.study_of(subject) in store.studies()
    assert subject in store.subjects_in_study(store.study_of(subject))


def test_ae_records_pair_with_their_narratives(store):
    pairs = list(store.iter_ae_with_narrative())
    assert pairs
    assert all(narrative is not None for _row, narrative in pairs)


def test_gold_is_reachable_only_through_its_own_accessor(store):
    """Nothing in extraction or evaluation may reach the answer key by accident."""
    assert store.gold()
    assert "gold" not in store.tables
    assert not any("gold" in name for name in store.tables)


def test_the_snapshot_id_is_content_derived(corpus_dir, tmp_path):
    first = load_store(corpus_dir).snapshot_id
    assert first == load_store(corpus_dir).snapshot_id

    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(corpus_dir, copy)
    assert load_store(copy).snapshot_id == first

    (copy / "ae.csv").write_text(
        (copy / "ae.csv").read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    assert load_store(copy).snapshot_id != first


def test_the_snapshot_excludes_the_answer_key(corpus_dir, tmp_path):
    """Changing gold must not change a run id: gold is not input data."""
    import shutil

    copy = tmp_path / "gold-changed"
    shutil.copytree(corpus_dir, copy)
    before = load_store(copy).snapshot_id
    (copy / "gold.jsonl").write_text("", encoding="utf-8")
    assert load_store(copy).snapshot_id == before


def test_a_missing_corpus_says_what_to_do(tmp_path):
    with pytest.raises(IngestError, match="aelayer generate"):
        load_store(tmp_path / "nothing-here")


def test_a_corpus_missing_a_required_table_is_refused(tmp_path, corpus_dir):
    import shutil

    copy = tmp_path / "broken"
    shutil.copytree(corpus_dir, copy)
    (copy / "ae.csv").unlink()
    with pytest.raises(IngestError, match="required table"):
        load_store(copy)


def test_hashing_is_stable_and_order_insensitive_for_mappings():
    assert hash_payload({"a": 1, "b": 2}) == hash_payload({"b": 2, "a": 1})
    assert hash_payload([1, 2]) != hash_payload([2, 1])
    assert hash_text("x") == hash_text("x")
    assert hash_text("x") != hash_text("y")
