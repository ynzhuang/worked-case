"""The synthetic corpus and its gold answer key."""

from __future__ import annotations

import json

import pytest

from aelayer.generate import CorpusGenerator, generate_corpus

SYNTHETIC_TABLES = ("dm", "ae", "ex", "lb", "cm")


def test_generation_is_deterministic(tmp_path):
    first = tmp_path / "a"
    second = tmp_path / "b"
    generate_corpus(seed=3, n_studies=2, out_dir=first, subjects_per_study=6)
    generate_corpus(seed=3, n_studies=2, out_dir=second, subjects_per_study=6)
    for name in ("ae.csv", "narratives.jsonl", "gold.jsonl", "ex.csv", "lb.csv"):
        assert (first / name).read_text() == (second / name).read_text(), name


def test_a_different_seed_gives_a_different_corpus(tmp_path):
    generate_corpus(seed=1, n_studies=2, out_dir=tmp_path / "a", subjects_per_study=6)
    generate_corpus(seed=2, n_studies=2, out_dir=tmp_path / "b", subjects_per_study=6)
    assert (tmp_path / "a" / "narratives.jsonl").read_text() != (
        tmp_path / "b" / "narratives.jsonl"
    ).read_text()


def test_every_table_row_is_marked_synthetic(store):
    for name in SYNTHETIC_TABLES:
        rows = store.rows(name)
        assert rows, name
        assert all(row.get("SYNTHETIC") == "Y" for row in rows), name


def test_every_narrative_header_says_synthetic(store):
    for narrative in store.narratives.values():
        assert "SYNTHETIC" in narrative.header
        assert "NOT REAL PATIENT DATA" in narrative.header


def test_the_corpus_covers_every_narrative_pattern(tmp_path):
    """Checked against a full-size corpus; the shared fixture is deliberately
    small and cannot be expected to sample every low-weight pattern."""
    import json as _json

    root, _manifest = generate_corpus(
        seed=5, n_studies=4, out_dir=tmp_path / "full", subjects_per_study=40
    )
    patterns = {
        _json.loads(line)["pattern"]
        for line in (root / "gold.jsonl").read_text().splitlines()
        if line.strip()
    }
    expected = {
        "explicit_coded", "explicit_verbatim_only", "explicit_british",
        "explicit_misspelled", "abbrev_gated", "abbrev_ungated", "lab_symptom",
        "split_sentence", "context_rescue", "context_action", "symptom_only",
        "negated", "hypothetical", "historical", "family_history", "uncertain",
        "out_of_window", "unresolved_onset", "distractor",
    }
    assert expected <= patterns


def test_the_corpus_covers_every_assertion_class(store):
    assertions = {row["assertion"] for row in store.gold() if row["assertion"]}
    assert assertions == {
        "present", "absent", "hypothetical", "historical", "family_history",
        "uncertain",
    }


def test_the_corpus_covers_every_evidence_state(store):
    states = {row["evidence_state"] for row in store.gold()}
    assert states == {"explicit", "supported", "possible", "absent", "none"}


def test_studies_differ_on_purpose(corpus_dir):
    manifest = json.loads((corpus_dir / "manifest.json").read_text())
    studies = manifest["studies"]
    assert len({s["glucose_unit"] for s in studies.values()}) > 1
    assert len({s["dictionary_version"] for s in studies.values()}) > 1


def test_a_study_reports_glucose_in_si_units(store):
    units = {row["LBORRESU"] for row in store.rows("lb")}
    assert "mmol/L" in units or "mg/dL" in units
    assert len(units) >= 1


def test_gold_onset_is_always_recoverable_from_the_record(store):
    """Gold must never assert a value nothing in the record states."""
    ae_by_doc = {row.get("DOCID"): row for row in store.rows("ae")}
    for row in store.gold():
        if row["onset_offset_days"] is None:
            continue
        in_table = bool(ae_by_doc.get(row["doc_id"], {}).get("AESTDTC"))
        in_text = row["onset_phrasing"] not in (None, "none", "structured")
        assert in_table or in_text, row["doc_id"]


def test_gold_case_status_rolls_up_to_subjects(corpus_dir):
    records = [json.loads(line) for line in
               (corpus_dir / "gold.jsonl").read_text().splitlines() if line.strip()]
    subjects = {
        json.loads(line)["subject_id"]: json.loads(line)
        for line in (corpus_dir / "gold_cases.jsonl").read_text().splitlines()
        if line.strip()
    }
    rank = {"excluded": 0, "review": 1, "case": 2}
    by_subject = {}
    for row in records:
        best = by_subject.get(row["subject_id"])
        if best is None or rank[row["verdict"]] > rank[best]:
            by_subject[row["subject_id"]] = row["verdict"]
    for subject, verdict in by_subject.items():
        assert subjects[subject]["verdict"] == verdict


def test_severity_and_seriousness_are_not_correlated_by_construction(store):
    """The corpus must contain counterexamples to their conflation."""
    gold = store.gold()
    mild_and_serious = [
        r for r in gold if r["severity"] in ("mild", "moderate") and r["seriousness"]
    ]
    severe_and_not_serious = [
        r for r in gold if r["severity"] == "severe" and not r["seriousness"]
    ]
    assert mild_and_serious, "a mild event can be serious"
    assert severe_and_not_serious, "a severe event can be non-serious"


def test_a_study_never_codes_the_concept_at_all(corpus_dir):
    manifest = json.loads((corpus_dir / "manifest.json").read_text())
    if len(manifest["studies"]) >= 4:
        assert any(
            not s["codes_hypoglycemia"] for s in manifest["studies"].values()
        )


def test_gold_state_mirrors_v1_semantics():
    """Spot-check the answer key's own logic, independent of the evaluator."""
    base = {
        "concept": "HYPOGLYCEMIA", "assertion": "present", "symptoms": [],
        "labs": [], "coded_term_matches_concept": False, "explicit_mention": False,
        "rescue_treatment": False, "action_taken": None, "onset_offset_days": 5,
    }
    assert CorpusGenerator.gold_state(dict(base, explicit_mention=True)) == "explicit"
    assert CorpusGenerator.gold_state(
        dict(base, symptoms=["tremor"],
             labs=[{"test": "GLUCOSE", "canonical_mgdl": 50}])
    ) == "supported"
    assert CorpusGenerator.gold_state(
        dict(base, symptoms=["tremor"], rescue_treatment=True)
    ) == "possible"
    assert CorpusGenerator.gold_state(dict(base, symptoms=["tremor"])) == "none"
    assert CorpusGenerator.gold_state(dict(base, assertion="absent")) == "absent"
    assert CorpusGenerator.gold_state(dict(base, concept="NAUSEA")) == "none"


def test_gold_verdict_honours_the_window_and_the_assertion_policy():
    base = {"concept": "HYPOGLYCEMIA", "assertion": "present", "onset_offset_days": 5}
    assert CorpusGenerator.gold_verdict(base, "explicit") == "case"
    assert CorpusGenerator.gold_verdict(dict(base, onset_offset_days=40), "explicit") == "excluded"
    assert CorpusGenerator.gold_verdict(dict(base, onset_offset_days=None), "explicit") == "review"
    assert CorpusGenerator.gold_verdict(dict(base, assertion="uncertain"), "none") == "review"
    assert CorpusGenerator.gold_verdict(dict(base, assertion="historical"), "none") == "excluded"


def test_generation_rejects_an_impossible_study_count():
    with pytest.raises(ValueError):
        CorpusGenerator(n_studies=99)
