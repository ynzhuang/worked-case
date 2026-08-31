"""Definition loading, validation, versioning and lifecycle."""

from __future__ import annotations

import pytest
import yaml

from aelayer.phenotype.loader import (
    DefinitionCatalog,
    DefinitionError,
    diff_definitions,
    load_definition,
    validate_condition,
)


def write(tmp_path, body, name=None):
    name = name or f"{body['id']}.v{body['version']}.yaml"
    path = tmp_path / name
    path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def minimal():
    return {
        "id": "demo",
        "version": 1,
        "status": "frozen",
        "label": "Demo",
        "concept": {"primary": "HYPOGLYCEMIA"},
        "evidence_rules": [
            {"id": "explicit", "state": "explicit",
             "when": {"coded_term_matches_concept": True}}
        ],
    }


def test_a_valid_definition_loads_and_is_hashed(tmp_path, minimal, catalog):
    definition = load_definition(write(tmp_path, minimal), catalog)
    assert definition.key == "demo.v1"
    assert definition.definition_hash
    assert definition.source_path


def test_the_hash_changes_with_the_content(tmp_path, minimal, catalog):
    first = load_definition(write(tmp_path, minimal), catalog).definition_hash
    minimal["evidence_rules"][0]["when"] = {"has_coded_term": True}
    second = load_definition(write(tmp_path, minimal), catalog).definition_hash
    assert first != second


def test_a_definition_that_fails_validation_does_not_load(tmp_path, minimal, catalog):
    minimal["evidence_rules"][0]["when"] = {"nonsense_predicate": True}
    with pytest.raises(DefinitionError, match="unknown predicate"):
        load_definition(write(tmp_path, minimal), catalog)


def test_an_unknown_concept_is_rejected(tmp_path, minimal, catalog):
    minimal["concept"]["primary"] = "NOT_A_CONCEPT"
    with pytest.raises(DefinitionError, match="unknown primary concept"):
        load_definition(write(tmp_path, minimal), catalog)


def test_an_unknown_lab_test_is_rejected(tmp_path, minimal, catalog):
    minimal["evidence_rules"][0]["when"] = {
        "lab": {"test": "UNOBTAINIUM", "op": "<", "value": 1}
    }
    with pytest.raises(DefinitionError, match="unknown lab test"):
        load_definition(write(tmp_path, minimal), catalog)


def test_an_unconvertible_threshold_unit_is_rejected(tmp_path, minimal, catalog):
    minimal["evidence_rules"][0]["when"] = {
        "lab": {"test": "GLUCOSE", "op": "<", "value": 70, "unit": "furlongs"}
    }
    with pytest.raises(DefinitionError, match="no conversion"):
        load_definition(write(tmp_path, minimal), catalog)


def test_an_unknown_symptom_set_is_rejected(tmp_path, minimal, catalog):
    minimal["evidence_rules"][0]["when"] = {
        "symptoms": {"min_count": 1, "from": ["imaginary_set"]}
    }
    with pytest.raises(DefinitionError, match="unknown symptom sets"):
        load_definition(write(tmp_path, minimal), catalog)


def test_a_state_no_case_definition_mentions_is_rejected(tmp_path, minimal, catalog):
    minimal["evidence_rules"].append(
        {"id": "extra", "state": "supported", "when": {"has_coded_term": True}}
    )
    minimal["case_definition"] = {
        "primary_set": ["explicit"], "review_set": [], "excluded": ["absent", "none"]
    }
    with pytest.raises(DefinitionError, match="never places"):
        load_definition(write(tmp_path, minimal), catalog)


def test_a_window_without_an_anchor_is_rejected(tmp_path, minimal, catalog):
    minimal["window"] = {"unit": "days", "min": 0, "max": 14}
    with pytest.raises(DefinitionError, match="no anchor"):
        load_definition(write(tmp_path, minimal), catalog)


def test_duplicate_rule_ids_are_rejected(tmp_path, minimal, catalog):
    minimal["evidence_rules"].append(dict(minimal["evidence_rules"][0]))
    with pytest.raises(DefinitionError, match="duplicate"):
        load_definition(write(tmp_path, minimal), catalog)


def test_the_filename_must_agree_with_the_declared_version(tmp_path, minimal, catalog):
    path = write(tmp_path, minimal, name="demo.v9.yaml")
    with pytest.raises(DefinitionError, match="does not match declared version"):
        load_definition(path, catalog)


def test_the_loader_refuses_to_overwrite_a_frozen_definition(tmp_path, minimal, catalog):
    write(tmp_path, minimal)
    catalogue = DefinitionCatalog(tmp_path, catalog)
    minimal["label"] = "Quietly changed"
    with pytest.raises(DefinitionError, match="frozen and will not be overwritten"):
        catalogue.write_candidate(minimal)


def test_a_draft_may_be_replaced_only_deliberately(tmp_path, minimal, catalog):
    minimal["status"] = "draft"
    write(tmp_path, minimal)
    catalogue = DefinitionCatalog(tmp_path, catalog)
    with pytest.raises(DefinitionError, match="already exists"):
        catalogue.write_candidate(minimal)
    minimal["label"] = "Revised draft"
    path = catalogue.write_candidate(minimal, overwrite=True)
    assert load_definition(path, catalog).label == "Revised draft"


def test_a_new_version_is_a_new_file(tmp_path, minimal, catalog):
    write(tmp_path, minimal)
    catalogue = DefinitionCatalog(tmp_path, catalog)
    assert catalogue.next_version("demo") == 2
    v2 = dict(minimal, version=2, status="draft", supersedes="demo.v1")
    path = catalogue.write_candidate(v2)
    assert path.name == "demo.v2.yaml"
    assert (tmp_path / "demo.v1.yaml").exists(), "v1 must still be there"


def test_a_draft_will_not_run_in_a_reproducible_run_without_opt_in(
    tmp_path, minimal, catalog
):
    minimal["status"] = "draft"
    write(tmp_path, minimal)
    catalogue = DefinitionCatalog(tmp_path, catalog)
    with pytest.raises(DefinitionError, match="not reproducible|draft"):
        catalogue.get("demo")
    assert catalogue.get("demo", allow_draft=True).version == 1


def test_version_none_selects_the_highest_published_version(pipeline):
    latest = pipeline.definition("te_symptomatic_hypoglycemia")
    assert latest.version == max(
        pipeline.definitions.versions("te_symptomatic_hypoglycemia")
    )
    assert latest.status != "draft"


def test_a_missing_version_reports_what_is_available(pipeline):
    with pytest.raises(DefinitionError, match="available"):
        pipeline.definition("te_symptomatic_hypoglycemia", 99)


def test_the_shipped_v1_and_v2_both_load_and_differ_by_one_threshold(
    definition_v1, definition_v2
):
    changes = {c["path"]: c for c in diff_definitions(definition_v1, definition_v2)}
    threshold = changes["evidence_rules[supported].when.all[0].lab.value"]
    assert threshold["from"] == 70 and threshold["to"] == 54
    assert definition_v1.definition_hash != definition_v2.definition_hash
    assert definition_v2.supersedes == "te_symptomatic_hypoglycemia.v1"


def test_diffing_matches_rules_by_id_not_position(definition_v1):
    reordered = definition_v1.model_copy(deep=True)
    reordered.evidence_rules = list(reversed(reordered.evidence_rules))
    assert diff_definitions(definition_v1, reordered) == []


def test_nested_combinators_validate(catalog):
    validate_condition(
        {"all": [{"any": [{"has_coded_term": True},
                          {"not": {"rescue_treatment": True}}]}]},
        catalog, where="test",
    )


def test_an_empty_condition_is_rejected(catalog):
    with pytest.raises(DefinitionError, match="empty condition"):
        validate_condition({}, catalog, where="test")
