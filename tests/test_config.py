"""Configuration: the catalogue, the profiles, and what each refuses."""

from __future__ import annotations

import shutil

import pytest
import yaml

from aelayer.catalog import ConceptCatalog, ConfigError, ExtractionConfig, load_yaml
from aelayer.hashing import extractor_version, hash_file, normalizer_version, snapshot_id
from aelayer.ingest import IngestError, load_store
from aelayer.profiles import ProfileError, StudyProfiles


def catalogue_body(**overrides):
    body = {
        "concepts": {"THING": {"label": "Thing", "lexicon": ["thing"]}},
        "attribute_catalogues": {
            "location": {
                "label": "Location",
                "values": {
                    "CHEST": {"label": "Chest", "region": "trunk",
                              "surface_forms": ["chest"]},
                },
                "regions": {"trunk": ["CHEST"]},
            }
        },
    }
    body.update(overrides)
    return body


# -- the concept catalogue --------------------------------------------------


def test_a_concept_nothing_can_match_is_rejected():
    with pytest.raises(ConfigError, match="needs a lexicon or coded terms"):
        ConceptCatalog(catalogue_body(concepts={"THING": {"label": "Thing"}}))


def test_an_attribute_value_with_no_surface_forms_is_rejected():
    """Nothing in text could ever normalize to it, so it is a silent dead end."""
    with pytest.raises(ConfigError, match="declares no surface forms"):
        ConceptCatalog(catalogue_body(attribute_catalogues={
            "location": {"values": {"CHEST": {"label": "Chest"}}}
        }))


def test_a_region_naming_an_unknown_value_is_rejected():
    with pytest.raises(ConfigError, match="places unknown values in regions"):
        ConceptCatalog(catalogue_body(attribute_catalogues={
            "location": {
                "values": {"CHEST": {"surface_forms": ["chest"]}},
                "regions": {"trunk": ["CHEST", "GHOST"]},
            }
        }))


def test_normalization_maps_declared_surface_forms_only(catalog):
    for surface in ("chest", "anterior chest", "chest wall", "Chest Wall"):
        assert catalog.normalize("location", surface) == "CHEST"
    assert catalog.normalize("location", "torso") is None


def test_a_region_expands_to_its_declared_members(catalog):
    assert set(catalog.attribute("location").in_region("trunk")) == {
        "CHEST", "ABDOMEN", "BACK"
    }
    with pytest.raises(ConfigError, match="unknown region"):
        catalog.attribute("location").in_region("everything")


def test_coded_terms_differ_between_dictionary_versions(catalog):
    concept = catalog.concept("RASH")
    early = set(concept.coded_terms_for_version("24.0"))
    late = set(concept.coded_terms_for_version("26.0"))
    assert early < late
    assert set(concept.all_coded_terms()) >= late


def test_no_concept_in_the_catalogue_carries_a_body_site(catalog):
    """Which is the reason the site has to survive somewhere else."""
    assert not any(c.carries_body_site for c in catalog.concepts.values())


def test_recurrence_is_declared_per_concept(catalog):
    assert catalog.concept("RASH").recurrence_expected
    assert not catalog.concept("ANAEMIA").recurrence_expected


# -- the extraction config --------------------------------------------------


def test_a_missing_section_is_rejected():
    with pytest.raises(ConfigError, match="missing section"):
        ExtractionConfig({"modifiers": {}})


def test_a_structured_source_cannot_be_declared_readable():
    with pytest.raises(ConfigError, match="not a question for a model"):
        ExtractionConfig({
            "readable_sources": ["structured_sponsor"],
            "extractable_attributes": ["location"], "modifiers": {},
        })


def test_the_shipped_config_reads_text_only(configs):
    assert set(configs.extraction.readable_sources) == {"reported_term", "comment"}
    assert "location" in configs.extraction.extractable_attributes


# -- profiles ---------------------------------------------------------------


def test_a_profile_referencing_an_undefined_home_is_rejected():
    with pytest.raises(ProfileError, match="undefined attribute homes"):
        StudyProfiles({
            "attribute_homes": {"AELOC": {"kind": "structured_standard",
                                          "variable": "AELOC"}},
            "profiles": {"P": {"location_home": "nowhere_at_all"}},
        })


def test_a_sponsor_home_without_a_mapping_is_rejected():
    """An unmapped sponsor code is unreadable, so the config refuses it."""
    with pytest.raises(ProfileError, match="declares no name and codelist"):
        StudyProfiles({
            "attribute_homes": {
                "sponsor_variable": {"kind": "structured_sponsor",
                                     "variable": "SUPPAE"},
                "none": {"kind": None, "variable": None},
            },
            "profiles": {"P": {"location_home": "sponsor_variable"}},
        })


def test_an_undeclared_study_cannot_be_read(profiles):
    with pytest.raises(ProfileError, match="every blank in it would be guesswork"):
        profiles.for_study("STUDY-UNKNOWN")


def test_a_blank_in_an_uncollected_variable_is_not_collected_by_protocol(profiles):
    profile = profiles.profile("P2_text")
    assert profile.availability_for_blank("AELOC") == "not_collected_by_protocol"
    assert profile.availability_for_blank("AESEV") == "unknown"


def test_exactly_one_profile_carries_both_routes(profiles):
    assert profiles.evaluation_profiles("location") == ["P6_both"]


def test_the_shipped_profiles_cover_every_home(profiles):
    homes = {
        home for profile in profiles.profiles.values()
        for home in profile.home_kinds("location")
    }
    assert homes == {"AELOC", "reported_term", "sponsor_variable", "comment", "none"}


# -- versions ---------------------------------------------------------------


def test_the_extractor_version_follows_its_config(tmp_path, configs):
    concepts = tmp_path / "concepts.yaml"
    extraction = tmp_path / "extraction.yaml"
    shutil.copy(configs.catalog.source_path, concepts)
    shutil.copy(configs.extraction.source_path, extraction)
    before = extractor_version(concepts, extraction)
    body = yaml.safe_load(extraction.read_text(encoding="utf-8"))
    body["modifiers"]["connectors"].append("upon")
    extraction.write_text(yaml.safe_dump(body), encoding="utf-8")
    assert extractor_version(concepts, extraction) != before


def test_the_normalizer_version_follows_the_profiles(tmp_path, configs):
    concepts = tmp_path / "concepts.yaml"
    profiles_path = tmp_path / "profiles.yaml"
    shutil.copy(configs.catalog.source_path, concepts)
    shutil.copy(configs.profiles.source_path, profiles_path)
    before = normalizer_version(concepts, profiles_path)
    body = yaml.safe_load(profiles_path.read_text(encoding="utf-8"))
    body["profiles"]["P1_structured"]["note"] = "edited"
    profiles_path.write_text(yaml.safe_dump(body), encoding="utf-8")
    assert normalizer_version(concepts, profiles_path) != before


def test_a_hash_ignores_line_endings(tmp_path):
    unix, windows = tmp_path / "a.txt", tmp_path / "b.txt"
    unix.write_bytes(b"one\ntwo\n")
    windows.write_bytes(b"one\r\ntwo\r\n")
    assert hash_file(unix) == hash_file(windows)


def test_the_snapshot_changes_when_an_input_changes(tmp_path, corpus_dir):
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
    with pytest.raises(IngestError, match="generates its own corpus"):
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
    (copy / "suppae.csv").unlink()
    assert load_store(copy).rows("suppae") == []


def test_supplemental_and_comment_records_key_back_to_an_ae_record(store):
    record_ids = {r["AESPID"] for r in store.rows("ae")}
    for table in ("suppae", "co"):
        rows = store.rows(table)
        assert rows, table
        assert {r["IDVARVAL"] for r in rows} <= record_ids


def test_an_empty_config_file_is_an_error(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="file is empty"):
        load_yaml(path)
