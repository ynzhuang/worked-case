"""Loading a definition: what it must say, and what it may not say."""

from __future__ import annotations

import pytest
import yaml

from aelayer.phenotype.loader import (
    DefinitionCatalog,
    DefinitionError,
    definition_content_hash,
    load_definition,
    validate_condition,
)


@pytest.fixture
def body(definition_v1):
    return definition_v1.model_dump(
        mode="json", exclude={"definition_hash", "source_path"}
    )


def write(tmp_path, body, name="te_symptomatic_hypoglycemia.v1.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return path


def load(tmp_path, body, catalog, **kwargs):
    return load_definition(write(tmp_path, body, **kwargs), catalog)


# -- the happy path ---------------------------------------------------------


def test_a_valid_definition_loads_and_is_stamped(tmp_path, body, catalog):
    definition = load(tmp_path, body, catalog)
    assert definition.definition_hash
    assert definition.source_path.endswith("te_symptomatic_hypoglycemia.v1.yaml")
    assert definition.key == "te_symptomatic_hypoglycemia.v1"


def test_the_hash_follows_the_content_not_the_filename(tmp_path, body, catalog):
    first = load(tmp_path, body, catalog)
    body["description"] = body["description"] + " Edited."
    second = load(tmp_path, body, catalog)
    assert first.definition_hash != second.definition_hash
    assert definition_content_hash(first) != definition_content_hash(second)


# -- what it must say -------------------------------------------------------


def test_a_missing_file_says_so(tmp_path, catalog):
    with pytest.raises(DefinitionError, match="no definition file at"):
        load_definition(tmp_path / "nothing.yaml", catalog)


def test_a_filename_that_disagrees_with_the_content_is_rejected(
    tmp_path, body, catalog
):
    with pytest.raises(DefinitionError, match="does not match declared version"):
        load(tmp_path, body, catalog, name="te_symptomatic_hypoglycemia.v9.yaml")


def test_an_unknown_concept_is_rejected(tmp_path, body, catalog):
    body["concept"]["primary"] = "NOT_A_CONCEPT"
    with pytest.raises(DefinitionError, match="unknown primary concept"):
        load(tmp_path, body, catalog)


def test_a_window_without_an_anchor_is_rejected(tmp_path, body, catalog):
    body["anchor"] = None
    with pytest.raises(DefinitionError, match="no anchor"):
        load(tmp_path, body, catalog)


def test_a_state_no_verdict_covers_is_rejected(tmp_path, body, catalog):
    """A rule that assigns a state nothing maps is a silent dropped cohort."""
    body["case_definition"]["review"] = []
    with pytest.raises(DefinitionError, match="case_definition never places"):
        load(tmp_path, body, catalog)


def test_a_definition_over_records_rather_than_episodes_is_rejected(
    tmp_path, body, catalog
):
    body["operates_on"] = "record"
    with pytest.raises(DefinitionError, match="Input should be 'episode'"):
        load(tmp_path, body, catalog)


# -- the rule language ------------------------------------------------------


def test_an_unknown_predicate_is_rejected(catalog):
    with pytest.raises(DefinitionError, match="unknown"):
        validate_condition({"vibes": True}, catalog, where="r")


def test_a_lab_predicate_naming_an_unknown_test_is_rejected(catalog):
    with pytest.raises(DefinitionError):
        validate_condition(
            {"lab": {"test": "NOPE", "op": "<", "value": 70, "unit": "mg/dL"}},
            catalog, where="r",
        )


def test_a_lab_predicate_with_an_unconvertible_unit_is_rejected(catalog):
    with pytest.raises(DefinitionError):
        validate_condition(
            {"lab": {"test": "GLUCOSE", "op": "<", "value": 70, "unit": "furlongs"}},
            catalog, where="r",
        )


def test_a_symptom_predicate_naming_an_unknown_set_is_rejected(catalog):
    with pytest.raises(DefinitionError):
        validate_condition(
            {"symptoms": {"min_count": 1, "from": ["not_a_set"]}}, catalog, where="r"
        )


def test_an_empty_condition_is_rejected(catalog):
    with pytest.raises(DefinitionError, match="empty condition"):
        validate_condition({}, catalog, where="r")


def test_the_shipped_rules_all_validate(definition_v1, definition_v2, catalog):
    for definition in (definition_v1, definition_v2):
        for rule in definition.evidence_rules:
            validate_condition(rule.when, catalog, where=rule.id)


# -- versions ---------------------------------------------------------------


def test_a_draft_does_not_run_without_an_explicit_opt_in(tmp_path, body, catalog):
    body["status"] = "draft"
    body["version"] = 3
    write(tmp_path, body, name="te_symptomatic_hypoglycemia.v3.yaml")
    catalogue = DefinitionCatalog(tmp_path, catalog)
    with pytest.raises(DefinitionError, match="not reproducible|will not run"):
        catalogue.get("te_symptomatic_hypoglycemia")
    assert catalogue.get("te_symptomatic_hypoglycemia", allow_draft=True).version == 3


def test_asking_for_a_version_that_does_not_exist_says_what_does(
    tmp_path, body, catalog
):
    write(tmp_path, body)
    catalogue = DefinitionCatalog(tmp_path, catalog)
    with pytest.raises(DefinitionError, match=r"available: \[1\]"):
        catalogue.get("te_symptomatic_hypoglycemia", 7)


def test_asking_for_an_unknown_definition_lists_the_known_ones(tmp_path, body, catalog):
    write(tmp_path, body)
    catalogue = DefinitionCatalog(tmp_path, catalog)
    with pytest.raises(DefinitionError, match="known:"):
        catalogue.get("something_else")


def test_the_highest_published_version_wins_over_a_later_draft(
    tmp_path, body, catalog
):
    write(tmp_path, body)
    draft = dict(body, version=2, status="draft")
    write(tmp_path, draft, name="te_symptomatic_hypoglycemia.v2.yaml")
    catalogue = DefinitionCatalog(tmp_path, catalog)
    assert catalogue.get("te_symptomatic_hypoglycemia").version == 1


# -- writing a new version --------------------------------------------------


def test_a_new_version_is_written_and_validated(tmp_path, body, catalog):
    write(tmp_path, body)
    catalogue = DefinitionCatalog(tmp_path, catalog)
    assert catalogue.next_version("te_symptomatic_hypoglycemia") == 2
    target = catalogue.write_candidate(dict(body, version=2, status="draft"))
    assert target.exists()
    written = catalogue.get("te_symptomatic_hypoglycemia", 2, allow_draft=True)
    assert (written.version, written.status) == (2, "draft")


def test_a_candidate_that_would_not_load_is_not_written(tmp_path, body, catalog):
    write(tmp_path, body)
    catalogue = DefinitionCatalog(tmp_path, catalog)
    broken = dict(body, version=2, status="candidate")
    broken["concept"] = dict(broken["concept"], primary="NOT_A_CONCEPT")
    with pytest.raises(DefinitionError):
        catalogue.write_candidate(broken)


def test_a_draft_is_only_replaced_when_asked(tmp_path, body, catalog):
    write(tmp_path, body)
    catalogue = DefinitionCatalog(tmp_path, catalog)
    draft = dict(body, version=2, status="draft")
    catalogue.write_candidate(draft)
    with pytest.raises(DefinitionError, match="Pass overwrite=True"):
        catalogue.write_candidate(draft)
    catalogue.write_candidate(dict(draft, label="Revised"), overwrite=True)
    revised = catalogue.get("te_symptomatic_hypoglycemia", 2, allow_draft=True)
    assert revised.label == "Revised"


def test_a_frozen_definition_is_never_overwritten(tmp_path, body, catalog):
    """A published definition is the record a prior cohort rests on."""
    write(tmp_path, body)
    catalogue = DefinitionCatalog(tmp_path, catalog)
    with pytest.raises(DefinitionError, match="frozen and will not be overwritten"):
        catalogue.write_candidate(dict(body, label="Quietly different"))
