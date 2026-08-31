"""Config loading and validation."""

from __future__ import annotations

import pytest
import yaml

from aelayer.catalog import ConceptCatalog, ConfigError, ExtractionConfig, load_configs


def test_the_shipped_catalogue_loads(catalog):
    assert "HYPOGLYCEMIA" in catalog.concepts
    assert catalog.concept("HYPOGLYCEMIA").label == "Hypoglycemia"


def test_an_unknown_concept_is_an_explicit_error(catalog):
    with pytest.raises(ConfigError, match="unknown concept"):
        catalog.concept("NOT_A_CONCEPT")


def test_unit_conversion_reaches_the_canonical_unit(catalog):
    glucose = catalog.lab_tests["GLUCOSE"]
    assert glucose.canonical_unit == "mg/dL"
    assert glucose.to_canonical(70, "mg/dL") == 70
    assert glucose.to_canonical(3.9, "mmol/L") == pytest.approx(70.3, abs=0.2)
    assert glucose.to_canonical(70, "furlongs") is None


def test_implausible_values_are_recognised(catalog):
    glucose = catalog.lab_tests["GLUCOSE"]
    assert glucose.plausible(54)
    assert not glucose.plausible(9999)


def test_symptom_sets_resolve(catalog):
    symptoms = catalog.symptoms_in_sets(["neuroglycopenic", "autonomic"])
    assert "confusion" in symptoms and "tremor" in symptoms
    with pytest.raises(ConfigError, match="unknown symptom set"):
        catalog.symptoms_in_sets(["nope"])


def test_a_group_is_an_explicit_list_not_an_inferred_hierarchy(catalog):
    assert catalog.expand_group("GLYCEMIC_EVENTS") == ["HYPOGLYCEMIA", "HYPERGLYCEMIA"]
    with pytest.raises(ConfigError, match="unknown concept group"):
        catalog.expand_group("EVERYTHING_ENDOCRINE")


def test_synonyms_combine_lexicon_and_coded_terms(catalog):
    synonyms = catalog.synonyms("HYPOGLYCEMIA")
    assert "hypoglycaemia" in synonyms
    assert "Blood glucose decreased" in synonyms


def test_a_catalogue_with_no_concepts_is_rejected():
    with pytest.raises(ConfigError, match="at least one concept"):
        ConceptCatalog({"concepts": {}})


def test_an_unmatchable_concept_is_rejected():
    with pytest.raises(ConfigError, match="lexicon or coded terms"):
        ConceptCatalog({"concepts": {"X": {"label": "X"}}})


def test_gating_abbreviations_without_a_gate_is_rejected():
    with pytest.raises(ConfigError, match="defines no context_gate"):
        ConceptCatalog({"concepts": {"X": {
            "lexicon": ["x"], "abbreviations": ["x"],
            "context_required": ["abbreviations"],
        }}})


def test_a_lab_test_must_convert_its_own_canonical_unit():
    with pytest.raises(ConfigError, match="canonical unit"):
        ConceptCatalog({
            "concepts": {"X": {"lexicon": ["x"]}},
            "lab_tests": {"G": {"canonical_unit": "mg/dL", "conversions": {"mmol/L": 18}}},
        })


def test_a_group_naming_an_unknown_concept_is_rejected():
    with pytest.raises(ConfigError, match="unknown concepts"):
        ConceptCatalog({
            "concepts": {"X": {"lexicon": ["x"]}},
            "concept_groups": {"G": {"members": ["X", "Y"]}},
        })


def test_extraction_config_requires_its_sections():
    with pytest.raises(ConfigError, match="missing section"):
        ExtractionConfig({"assertion": {}})


def test_an_unknown_assertion_class_in_cues_is_rejected(extraction_config):
    import copy

    raw = copy.deepcopy(extraction_config.raw)
    raw["assertion"]["cues"]["speculative"] = ["maybe"]
    with pytest.raises(ConfigError, match="unknown assertion class"):
        ExtractionConfig(raw)


def test_an_unknown_scope_is_rejected(extraction_config):
    import copy

    raw = copy.deepcopy(extraction_config.raw)
    raw["assertion"]["scope"] = "paragraph"
    with pytest.raises(ConfigError, match="sentence|window"):
        ExtractionConfig(raw)


def test_cue_lists_accept_both_the_flat_and_the_directional_form():
    config = ExtractionConfig({
        "assertion": {"cues": {"absent": ["no evidence of"],
                               "uncertain": {"pre": ["possible"], "post": ["is uncertain"]}}},
        "temporality": {}, "values": {}, "labs": {},
    })
    assert config.cue_lists("absent") == (["no evidence of"], [])
    assert config.cue_lists("uncertain") == (["possible"], ["is uncertain"])
    assert config.cue_lists("historical") == ([], [])


def test_config_loading_is_cached_on_content(tmp_path):
    from aelayer import paths

    first = load_configs()
    second = load_configs()
    assert first[0] is second[0], "unchanged config should be served from cache"

    edited = tmp_path / "concepts.yaml"
    body = yaml.safe_load(paths.CONCEPTS_YAML.read_text(encoding="utf-8"))
    body["concepts"]["HYPOGLYCEMIA"]["lexicon"].append("sugar crash")
    edited.write_text(yaml.safe_dump(body), encoding="utf-8")
    third = load_configs(edited, paths.EXTRACTION_YAML)
    assert third[2] != first[2], "a changed config must change the extractor version"
    assert "sugar crash" in third[0].concept("HYPOGLYCEMIA").lexicon
