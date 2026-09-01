"""Configuration and ingest: what is refused, and what the versions mean."""

from __future__ import annotations

import shutil

import pytest
import yaml

from aelayer.catalog import ConceptCatalog, ConfigError, ExtractionConfig, load_yaml
from aelayer.hashing import (
    extractor_version,
    hash_file,
    hash_payload,
    normalizer_version,
    snapshot_id,
)
from aelayer.ingest import IngestError, load_store


def catalog_body(**overrides):
    body = {
        "concepts": {
            "THING": {"label": "Thing", "lexicon": ["thing"], "abbreviations": []},
        },
        "lab_tests": {
            "GLUCOSE": {
                "canonical_unit": "mg/dL",
                "conversions": {"mg/dL": 1.0, "mmol/L": 18.0182},
                "plausible_range": [10, 900],
            }
        },
    }
    body.update(overrides)
    return body


# -- the concept catalogue --------------------------------------------------


def test_a_concept_nothing_can_match_is_rejected():
    body = catalog_body(concepts={"THING": {"label": "Thing"}})
    with pytest.raises(ConfigError, match="needs a lexicon or coded terms"):
        ConceptCatalog(body)


def test_gating_abbreviations_without_a_gate_is_rejected():
    """An ungated abbreviation is a false-positive generator, not a shortcut."""
    body = catalog_body(concepts={"THING": {
        "lexicon": ["thing"], "abbreviations": ["th"],
        "context_required": ["abbreviations"],
    }})
    with pytest.raises(ConfigError, match="defines no context_gate"):
        ConceptCatalog(body)


def test_a_lab_test_with_no_conversion_for_its_own_unit_is_rejected():
    body = catalog_body(lab_tests={"GLUCOSE": {
        "canonical_unit": "mg/dL", "conversions": {"mmol/L": 18.0182},
    }})
    with pytest.raises(ConfigError, match="canonical unit"):
        ConceptCatalog(body)


def test_a_group_naming_a_concept_that_does_not_exist_is_rejected():
    body = catalog_body(concept_groups={"G": {"members": ["THING", "GHOST"]}})
    with pytest.raises(ConfigError, match="unknown concepts"):
        ConceptCatalog(body)


def test_an_empty_catalogue_is_rejected():
    with pytest.raises(ConfigError, match="at least one concept"):
        ConceptCatalog({"concepts": {}})


def test_unknown_lookups_say_what_was_asked_for(catalog):
    with pytest.raises(ConfigError, match="unknown concept"):
        catalog.concept("NOPE")
    with pytest.raises(ConfigError, match="unknown concept group"):
        catalog.expand_group("NOPE")
    with pytest.raises(ConfigError, match="unknown symptom set"):
        catalog.symptoms_in_sets(["NOPE"])


# -- coded terms by dictionary version -------------------------------------


def test_a_concept_reports_its_terms_per_dictionary_version(catalog):
    concept = catalog.concept("HYPOGLYCEMIA")
    every = set(concept.all_coded_terms())
    assert every
    versions = [
        v for v in (concept.coded_terms or {}).get("by_dictionary_version", {})
    ]
    for version in versions:
        assert set(concept.coded_terms_for_version(version)) <= every


def test_recurrence_is_declared_per_concept_not_assumed(catalog):
    assert catalog.concept("HYPOGLYCEMIA").recurrence_expected
    assert not catalog.concept("ANAEMIA").recurrence_expected


def test_a_group_is_the_members_it_lists_and_nothing_more(catalog):
    for group, members in catalog.concept_groups.items():
        assert catalog.expand_group(group) == members


# -- the extraction config --------------------------------------------------


def test_a_missing_section_is_rejected():
    with pytest.raises(ConfigError, match="missing section"):
        ExtractionConfig({"assertion": {}, "temporality": {}, "values": {}})


def test_an_unknown_assertion_class_is_rejected():
    with pytest.raises(ConfigError, match="unknown assertion class"):
        ExtractionConfig({
            "assertion": {"cues": {"vibes": ["maybe"]}},
            "temporality": {}, "values": {}, "labs": {},
        })


def test_a_bad_assertion_scope_is_rejected():
    with pytest.raises(ConfigError, match="scope must be"):
        ExtractionConfig({
            "assertion": {"scope": "paragraph"},
            "temporality": {}, "values": {}, "labs": {},
        })


def test_a_flat_cue_list_is_read_as_pre_cues(configs):
    pre, post = configs.extraction.cue_lists("absent")
    assert pre and isinstance(pre, list)
    assert isinstance(post, list)
    assert configs.extraction.cue_lists("not_a_class") == ([], [])


def test_the_default_anchor_is_read_from_the_config(configs):
    assert configs.extraction.default_anchor == "dose_escalation"
    assert "dose_escalation" in configs.extraction.anchors


def test_a_confidence_falls_back_to_its_default(configs):
    assert configs.extraction.confidence_for("not_a_key", 0.42) == 0.42


# -- yaml loading -----------------------------------------------------------


def test_an_empty_config_file_is_an_error(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="file is empty"):
        load_yaml(path)


def test_invalid_yaml_names_the_file(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("a: [1, 2\nb: }", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_yaml(path)


# -- versions are content-derived ------------------------------------------


def test_the_extractor_version_follows_the_config_that_drives_it(tmp_path, configs):
    concepts = tmp_path / "concepts.yaml"
    extraction = tmp_path / "extraction.yaml"
    shutil.copy(configs.catalog.source_path, concepts)
    shutil.copy(configs.extraction.source_path, extraction)
    before = extractor_version(concepts, extraction)

    body = yaml.safe_load(extraction.read_text(encoding="utf-8"))
    body["assertion"]["cues"]["absent"]["pre"].append("no sign whatsoever of")
    extraction.write_text(yaml.safe_dump(body), encoding="utf-8")
    assert extractor_version(concepts, extraction) != before


def test_the_normalizer_version_follows_the_semantics_that_drive_it(tmp_path, configs):
    concepts = tmp_path / "concepts.yaml"
    semantics = tmp_path / "semantics.yaml"
    shutil.copy(configs.catalog.source_path, concepts)
    shutil.copy(configs.semantics.source_path, semantics)
    before = normalizer_version(concepts, semantics)

    body = yaml.safe_load(semantics.read_text(encoding="utf-8"))
    body["studies"]["STUDY-01"]["note"] = "edited"
    semantics.write_text(yaml.safe_dump(body), encoding="utf-8")
    assert normalizer_version(concepts, semantics) != before


def test_a_hash_ignores_line_endings_so_it_travels(tmp_path):
    unix, windows = tmp_path / "a.txt", tmp_path / "b.txt"
    unix.write_bytes(b"one\ntwo\n")
    windows.write_bytes(b"one\r\ntwo\r\n")
    assert hash_file(unix) == hash_file(windows)


def test_a_payload_hash_ignores_key_order():
    assert hash_payload({"a": 1, "b": 2}) == hash_payload({"b": 2, "a": 1})


def test_the_snapshot_id_changes_when_any_input_file_changes(tmp_path, corpus_dir):
    copy = tmp_path / "corpus"
    shutil.copytree(corpus_dir, copy)
    before = snapshot_id(copy)
    assert before == snapshot_id(copy)
    (copy / "ae.csv").write_text(
        (copy / "ae.csv").read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    assert snapshot_id(copy) != before


# -- ingest -----------------------------------------------------------------


def test_a_missing_data_directory_says_where_it_looked(tmp_path):
    with pytest.raises(IngestError):
        load_store(tmp_path / "nothing")


def test_a_corpus_missing_a_required_table_is_refused(tmp_path, corpus_dir):
    copy = tmp_path / "corpus"
    shutil.copytree(corpus_dir, copy)
    (copy / "ae.csv").unlink()
    with pytest.raises(IngestError, match="required table ae.csv missing"):
        load_store(copy)


def test_an_optional_table_may_be_absent(tmp_path, corpus_dir):
    copy = tmp_path / "corpus"
    shutil.copytree(corpus_dir, copy)
    (copy / "linked_hypo_event.csv").unlink()
    store = load_store(copy)
    assert store.rows("linked_hypo_event") == []


def test_the_store_summarises_what_it_holds(store):
    summary = store.summary()
    assert summary["studies"] == 6
    assert summary["subjects"] > 0
    assert summary["ae_records"] > 0
    assert summary["narratives"] > 0
    assert summary["snapshot_id"]


def test_numeric_columns_arrive_as_numbers(store):
    row = store.rows("ex")[0]
    assert isinstance(row["EXSEQ"], int)
    assert isinstance(row["EXDOSE"], (int, float))


def test_an_empty_cell_arrives_as_none_not_as_an_empty_string(store):
    assert any(row.get("AEENDTC") is None for row in store.rows("ae"))


def test_a_narrative_reaches_the_record_it_belongs_to(store):
    pairs = list(store.iter_ae_with_narrative())
    assert pairs
    with_text = [(row, n) for row, n in pairs if n is not None]
    assert with_text
    row, narrative = with_text[0]
    assert narrative.subject_id == row["USUBJID"]
    assert narrative.header in narrative.full_text
