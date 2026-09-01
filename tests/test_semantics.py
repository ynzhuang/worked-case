"""The governing layer: what a blank means, per study, per field."""

from __future__ import annotations

import pytest
import yaml

from aelayer.semantics import CollectionSemantics, SemanticsError


def build(**studies) -> CollectionSemantics:
    return CollectionSemantics({"studies": studies})


# -- blanks -----------------------------------------------------------------


def test_a_field_the_crf_never_carried_is_not_collected_by_protocol():
    semantics = build(ST={"collected_fields": ["verbatim_term", "severity"]})
    study = semantics.for_study("ST")
    assert study.state_for_blank("action_taken") == "not_collected_by_protocol"
    assert study.state_for_blank("severity") == "unknown"


def test_an_explicit_declaration_beats_the_collected_set():
    semantics = build(ST={
        "collected_fields": ["severity"],
        "blank_means": {"severity": "intentionally_blank"},
    })
    assert semantics.for_study("ST").state_for_blank("severity") == "intentionally_blank"


def test_a_study_declaring_no_collected_set_falls_back_to_the_default():
    semantics = CollectionSemantics(
        {"defaults": {"blank_means": "pending_ongoing"}, "studies": {"ST": {}}}
    )
    assert semantics.for_study("ST").state_for_blank("anything") == "pending_ongoing"


def test_collects_is_false_for_both_ways_a_value_never_arrives():
    """Different facts, same consequence: no value will appear in that column."""
    semantics = build(ST={
        "collected_fields": ["severity", "outcome"],
        "blank_means": {"outcome": "intentionally_blank"},
    })
    study = semantics.for_study("ST")
    assert study.collects("severity")
    assert not study.collects("outcome")          # collected, but never filled
    assert not study.collects("action_taken")     # not on the CRF at all
    assert study.state_for_blank("outcome") != study.state_for_blank("action_taken")


# -- the study must be declared --------------------------------------------


def test_an_undeclared_study_cannot_be_read():
    semantics = build(ST={})
    with pytest.raises(SemanticsError, match="no collection semantics"):
        semantics.for_study("OTHER")


def test_an_unknown_collection_state_is_rejected():
    with pytest.raises(SemanticsError, match="is not a collection state"):
        CollectionSemantics({"studies": {"ST": {"blank_means": {"severity": "maybe"}}}})


def test_a_study_referencing_an_undefined_linked_form_is_rejected():
    with pytest.raises(SemanticsError, match="undefined linked forms"):
        CollectionSemantics({"studies": {"ST": {"linked_forms": ["nope"]}}})


def test_config_without_studies_is_rejected():
    with pytest.raises(SemanticsError, match="must define `studies`"):
        CollectionSemantics({"defaults": {}})


# -- gates ------------------------------------------------------------------


def test_a_gated_field_is_not_applicable_when_its_parent_answered_no():
    semantics = CollectionSemantics({
        "defaults": {
            "gated_fields": {
                "seriousness_criteria": {
                    "gate": "seriousness", "when_gate_false": "not_applicable_gated",
                }
            },
            "gate_values": {
                "seriousness": {"from_field": "AESER", "true_when_in": ["Y"]}
            },
        },
        "studies": {"ST": {}},
    })
    study = semantics.for_study("ST")
    assert study.gate_answer("seriousness", {"AESER": "N"}) is False
    assert study.gate_for("seriousness_criteria").resolve(False) == "not_applicable_gated"


def test_a_gate_whose_own_field_is_blank_has_no_answer():
    """An unanswered gate does not make its children inapplicable."""
    semantics = CollectionSemantics({
        "defaults": {
            "gate_values": {
                "seriousness": {"from_field": "AESER", "true_when_in": ["Y"]}
            }
        },
        "studies": {"ST": {}},
    })
    study = semantics.for_study("ST")
    assert study.gate_answer("seriousness", {"AESER": ""}) is None
    assert study.gate_answer("seriousness", {}) is None


def test_a_true_gate_leaves_its_children_alone():
    from aelayer.semantics import GateSpec

    spec = GateSpec("seriousness_criteria", "seriousness", "not_applicable_gated")
    assert spec.resolve(True) is None
    assert spec.resolve(None) is None


# -- codelists --------------------------------------------------------------


def test_a_concept_the_codelist_cannot_express_is_not_representable():
    """The weaker claim is the honest one: no nearest-code substitution."""
    semantics = build(ST={
        "restricted_codelists": {
            "action_taken": {
                "permissible": ["none", "drug_withdrawn"],
                "absent_concepts": ["dose_reduced", "drug_interrupted"],
            }
        }
    })
    codelist = semantics.for_study("ST").codelist_for("action_taken")
    assert codelist.resolve("drug_withdrawn") == ("drug_withdrawn", "collected")
    assert codelist.resolve("dose_reduced") == (None, "not_representable")
    assert codelist.resolve(None) == (None, "unknown")


def test_a_concept_outside_the_codelist_entirely_is_also_not_representable():
    semantics = build(ST={
        "restricted_codelists": {"action_taken": {"permissible": ["none"]}}
    })
    codelist = semantics.for_study("ST").codelist_for("action_taken")
    assert codelist.resolve("something_new") == (None, "not_representable")


# -- the shipped configuration ---------------------------------------------


def test_every_shipped_study_declares_what_its_blanks_mean(semantics):
    assert len(semantics.study_ids()) == 6
    for study_id in semantics.study_ids():
        study = semantics.for_study(study_id)
        assert study.representation
        assert study.dictionary_version
        assert study.collected_fields, f"{study_id} declares no collected fields"


def test_the_six_studies_differ_in_representation(semantics):
    representations = {
        semantics.for_study(s).representation for s in semantics.study_ids()
    }
    assert representations == {"V-A", "V-B", "V-C", "V-D", "V-E", "V-F"}


def test_at_least_one_study_cannot_express_a_dose_reduction(semantics):
    restricted = [
        s for s in semantics.study_ids()
        if (cl := semantics.for_study(s).codelist_for("action_taken"))
        and cl.resolve("dose_reduced")[1] == "not_representable"
    ]
    assert restricted, "no study exercises the not_representable path"


def test_the_shipped_config_is_the_one_that_is_loaded(configs):
    raw = yaml.safe_load(configs.semantics.source_path.read_text(encoding="utf-8"))
    assert sorted(raw["studies"]) == configs.semantics.study_ids()
