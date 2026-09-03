"""Configuration: what it declares, and what it refuses to declare."""

from __future__ import annotations

import pytest

from aelayer.catalog import ConceptCatalog, ConfigError, ExtractionConfig
from aelayer.profiles import ProfileError, StudyProfiles


# -- the placeholder notice ---------------------------------------------------


def test_the_catalogue_says_its_terms_are_placeholders(catalog):
    assert "not licensed" in catalog.notice.lower() or \
        "placeholder" in catalog.notice.lower()


def test_the_readme_repeats_the_placeholder_notice():
    from pathlib import Path

    readme = Path(__file__).resolve().parents[1] / "README.md"
    text = readme.read_text(encoding="utf-8").lower()
    assert "illustrative placeholder" in text
    assert "not licensed" in text or "no terminology licence" in text


# -- the concept catalogue ----------------------------------------------------


def test_every_concept_declares_a_code_per_version(catalog):
    for concept in catalog.concepts.values():
        assert concept.codes
        for version in concept.codes:
            assert version in catalog.dictionary_versions


def test_the_target_version_is_declared(catalog):
    assert catalog.target_version in catalog.dictionary_versions


def test_a_concept_absent_from_the_target_exists_on_purpose(catalog):
    missing = [
        c for c in catalog.concepts.values()
        if c.code_in(catalog.target_version) is None
    ]
    assert missing, (
        "no concept is missing from the target version, so the "
        "flagged-for-review path can never be exercised"
    )


def test_a_concept_renamed_between_versions_exists_on_purpose(catalog):
    renamed = [
        c for c in catalog.concepts.values()
        if len({code for code in c.codes.values()}) > 1
    ]
    assert renamed, (
        "no concept is renamed between versions, so the mechanical-remap path "
        "can never be exercised"
    )


def test_an_unknown_concept_raises(catalog):
    with pytest.raises(ConfigError):
        catalog.concept("NOT_A_CONCEPT")


def test_surface_forms_normalize_to_catalogue_values(catalog):
    modifier = catalog.modifier("mucosal_involvement")
    assert modifier.normalize("Oral Ulceration") == "ORAL"
    assert modifier.normalize("oral-ulceration") == "ORAL"
    assert modifier.normalize("something nobody declared") is None


# -- the extraction config ----------------------------------------------------


def test_cue_lists_are_declared_not_hard_coded(configs):
    absent_pre, absent_post = configs.extraction.cue_lists("absent")
    uncertain_pre, uncertain_post = configs.extraction.cue_lists("uncertain")
    assert absent_pre and absent_post
    assert uncertain_pre and uncertain_post


def test_an_unknown_assertion_class_in_cues_is_refused(configs):
    raw = {
        **configs.extraction.raw,
        "assertion": {**configs.extraction.assertion,
                      "cues": {"maybe_sort_of": ["hmm"]}},
    }
    with pytest.raises(ConfigError) as exc:
        ExtractionConfig(raw)
    assert "unknown assertion class" in str(exc.value)


def test_an_unknown_default_assertion_is_refused(configs):
    raw = {
        **configs.extraction.raw,
        "assertion": {**configs.extraction.assertion, "default": "probably"},
    }
    with pytest.raises(ConfigError):
        ExtractionConfig(raw)


def test_a_missing_section_is_refused(configs):
    raw = {k: v for k, v in configs.extraction.raw.items()
           if k != "extractable_modifiers"}
    with pytest.raises(ConfigError) as exc:
        ExtractionConfig(raw)
    assert "missing section" in str(exc.value)


# -- profiles -----------------------------------------------------------------


def test_a_profile_declares_where_every_modifier_lives(profiles):
    for profile in profiles.profiles.values():
        assert profile.homes_for("mucosal_involvement")


def test_a_home_naming_an_unknown_source_kind_is_refused(profiles):
    raw = {
        "modifier_homes": {"bad": {"kind": "telepathy", "variable": "X"}},
        "profiles": {"P": {"study_id": "S"}},
    }
    with pytest.raises(ProfileError) as exc:
        StudyProfiles(raw)
    assert "unknown source kind" in str(exc.value)


def test_a_config_without_profiles_is_refused():
    with pytest.raises(ProfileError):
        StudyProfiles({"modifier_homes": {}})


def test_supportability_is_decided_on_metadata_alone(profiles):
    statuses = {
        profile.supportability("mucosal_involvement")[0]
        for profile in profiles.profiles.values()
    }
    assert statuses == {"supported", "supported_via_extraction", "cannot_ascertain"}


def test_exactly_one_profile_supports_a_silver_standard(profiles):
    both = profiles.evaluation_profiles("mucosal_involvement")
    assert both, "no profile carries the modifier both structurally and in text"


def test_a_second_modifier_is_configured(catalog):
    """So that nothing in the code is specific to mucosal involvement."""
    assert "photosensitivity" in catalog.modifiers


def test_config_versions_are_content_derived(configs):
    assert "+" in configs.normalizer_version
    assert "+" in configs.extractor_version
    assert configs.normalizer_version.startswith("normalize-4")
    assert configs.extractor_version.startswith("extract-4")
