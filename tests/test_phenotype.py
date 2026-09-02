"""Definitions and the four-verdict evaluator."""

from __future__ import annotations

import datetime as _dt

import pytest
import yaml

from aelayer.models import Attribute, CanonicalAEEpisode, Span
from aelayer.phenotype import DefinitionError, PhenotypeEvaluator, load_definition


def span(field: str = "location") -> Span:
    return Span(doc_id="AE:R1", start=0, end=5, field=field, extracted_value="CHEST")


def episode(**kwargs) -> CanonicalAEEpisode:
    base = dict(
        episode_id="E1", study_id="STUDY-P1", subject_id="S1", profile="P1_structured",
        standardized_concept="RASH",
        coded_events=["Rash"],
        dictionary_versions=["25.1"],
        episode_start=Attribute[_dt.date].direct(
            _dt.date(2024, 2, 6), "AESTDTC", [span("onset")]
        ),
        location=Attribute[str].direct("CHEST", "AELOC", [span()]),
        onset_offset_days=Attribute[int](
            value=5, availability="collected", method="normalized", source="derived",
            source_variable="EX.first_exposure",
        ),
    )
    base.update(kwargs)
    return CanonicalAEEpisode(**base)


@pytest.fixture
def evaluator(definition_v1, catalog):
    return PhenotypeEvaluator(definition_v1, catalog)


# -- the four verdicts ------------------------------------------------------


def test_everything_satisfied_is_a_case(evaluator):
    verdict = evaluator.evaluate_episode(episode())
    assert verdict.verdict == "case"
    assert all(f.satisfied for f in verdict.findings)


def test_a_present_attribute_that_fails_is_not_a_case(evaluator):
    verdict = evaluator.evaluate_episode(
        episode(location=Attribute[str].direct("ARM", "AELOC", [span()]))
    )
    assert verdict.verdict == "not_case"
    assert "not one of" in verdict.reason


def test_an_unavailable_attribute_is_not_ascertainable(evaluator):
    """Not a negative: nobody can evaluate the rule, reviewer included."""
    verdict = evaluator.evaluate_episode(
        episode(location=Attribute[str].unavailable(
            "not_collected_by_protocol",
            note="P3_prespecified records location nowhere",
        ))
    )
    assert verdict.verdict == "not_ascertainable"
    assert verdict.deciding_attribute == "location"
    assert "never collected" in verdict.reason


def test_an_onset_whose_anchor_will_not_resolve_is_a_review(evaluator):
    """A start date with no resolvable anchor is something a person can settle."""
    verdict = evaluator.evaluate_episode(
        episode(onset_offset_days=Attribute[int].unavailable(
            "unknown", note="no exposure record for this subject"
        ))
    )
    assert verdict.verdict == "review"
    assert verdict.deciding_attribute == "onset"
    assert "could not be resolved" in verdict.reason


def test_an_episode_with_no_start_date_at_all_is_not_ascertainable(evaluator):
    """Nobody can place an event that has no date anywhere in the record."""
    verdict = evaluator.evaluate_episode(episode(
        episode_start=Attribute[_dt.date].unavailable("not_collected_by_protocol"),
        onset_offset_days=Attribute[int].unavailable("not_collected_by_protocol"),
    ))
    assert verdict.verdict == "not_ascertainable"
    assert verdict.deciding_attribute == "onset"


def test_a_definite_negative_outranks_an_unascertainable_one(evaluator):
    """Knowing the rash was on the arm settles it, whatever else is missing."""
    verdict = evaluator.evaluate_episode(episode(
        location=Attribute[str].direct("ARM", "AELOC", [span()]),
        onset_offset_days=Attribute[int].unavailable("unknown"),
    ))
    assert verdict.verdict == "not_case"


def test_an_onset_outside_the_window_is_not_a_case(evaluator):
    verdict = evaluator.evaluate_episode(episode(
        onset_offset_days=Attribute[int](
            value=40, availability="collected", method="normalized",
            source="derived", source_variable="EX.first_exposure",
        )
    ))
    assert verdict.verdict == "not_case"
    assert "outside" in verdict.reason


def test_a_low_confidence_value_is_routed_to_review(evaluator):
    verdict = evaluator.evaluate_episode(episode(
        location=Attribute[str].extracted(
            "CHEST", "AETERM", [span()], confidence=0.3
        )
    ))
    assert verdict.verdict == "review"
    assert "confidence" in verdict.reason


def test_another_concept_is_simply_not_a_case(evaluator):
    verdict = evaluator.evaluate_episode(
        episode(standardized_concept="NAUSEA", coded_events=["Nausea"])
    )
    assert verdict.verdict == "not_case"
    assert "is not" in verdict.reason


def test_a_discovery_candidate_cannot_become_a_case(evaluator):
    verdict = evaluator.evaluate_episode(episode(candidate=True))
    assert verdict.verdict == "not_case"
    assert "adjudication" in verdict.reason


# -- routes -----------------------------------------------------------------


def test_the_rule_accepts_every_route_by_design(evaluator):
    for attribute in (
        Attribute[str].direct("CHEST", "AELOC", [span()]),
        Attribute[str].normalized("CHEST", "SUPPAE.RASHSITE", evidence=[span()]),
        Attribute[str].extracted("CHEST", "AETERM", [span()], confidence=0.9),
    ):
        verdict = evaluator.evaluate_episode(episode(location=attribute))
        assert verdict.verdict == "case", attribute.method


def test_a_narrower_definition_refuses_the_text_route(definition_v2, catalog):
    """v2 is a different scientific claim, not a bug fix."""
    evaluator = PhenotypeEvaluator(definition_v2, catalog)
    verdict = evaluator.evaluate_episode(episode(
        location=Attribute[str].extracted("CHEST", "AETERM", [span()], confidence=0.9)
    ))
    assert verdict.verdict == "not_ascertainable"
    assert "does not accept" in verdict.reason


def test_the_verdict_carries_the_route_it_used(evaluator):
    assignments = evaluator.evaluate([episode()])
    assert assignments[0].attribute_methods == {"location": "direct",
                                                "onset": "normalized"}
    assert assignments[0].attribute_sources["location"] == "AELOC"


def test_a_subject_verdict_takes_the_strongest_claim(evaluator):
    episodes = [
        episode(episode_id="E1"),
        episode(episode_id="E2",
                location=Attribute[str].direct("ARM", "AELOC", [span()])),
    ]
    assert evaluator.evaluate_subjects(episodes) == {"S1": "case"}


def test_an_unascertainable_subject_is_not_a_negative(evaluator):
    episodes = [
        episode(episode_id="E1",
                location=Attribute[str].direct("ARM", "AELOC", [span()])),
        episode(episode_id="E2",
                location=Attribute[str].unavailable("not_collected_by_protocol")),
    ]
    assert evaluator.evaluate_subjects(episodes) == {"S1": "not_ascertainable"}


# -- loading and validation -------------------------------------------------


@pytest.fixture
def body(definition_v1):
    return definition_v1.model_dump(
        mode="json", exclude={"definition_hash", "source_path"}, by_alias=True
    )


def write(tmp_path, body, name="te_truncal_rash.v1.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return path


def test_a_valid_definition_loads_and_is_stamped(tmp_path, body, catalog):
    definition = load_definition(write(tmp_path, body), catalog)
    assert definition.definition_hash
    assert definition.key == "te_truncal_rash.v1"


def test_a_requirement_naming_an_unknown_attribute_is_rejected(
    tmp_path, body, catalog
):
    body["required_attributes"][0]["name"] = "vibes"
    with pytest.raises(DefinitionError, match="not an attribute an episode carries"):
        load_definition(write(tmp_path, body), catalog)


def test_a_requirement_allowing_a_value_outside_the_catalogue_is_rejected(
    tmp_path, body, catalog
):
    body["required_attributes"][0]["in"] = ["CHEST", "ELBOW_PIT"]
    with pytest.raises(DefinitionError, match="not in the location catalogue"):
        load_definition(write(tmp_path, body), catalog)


def test_a_requirement_accepting_no_method_is_rejected(tmp_path, body, catalog):
    body["required_attributes"][0]["accept_methods"] = []
    with pytest.raises(DefinitionError, match="accepts no method"):
        load_definition(write(tmp_path, body), catalog)


def test_a_requirement_that_tests_nothing_is_rejected(tmp_path, body, catalog):
    body["required_attributes"][0].pop("in")
    with pytest.raises(DefinitionError, match="tests nothing"):
        load_definition(write(tmp_path, body), catalog)


def test_a_window_without_an_anchor_is_rejected(tmp_path, body, catalog):
    body["anchor"] = None
    with pytest.raises(DefinitionError, match="declares no anchor"):
        load_definition(write(tmp_path, body), catalog)


def test_the_verdicts_block_must_match_what_the_evaluator_returns(
    tmp_path, body, catalog
):
    body["verdicts"].pop("review")
    with pytest.raises(DefinitionError):
        load_definition(write(tmp_path, body), catalog)


def test_the_shipped_definitions_declare_all_four_verdicts(definition_v1, definition_v2):
    for definition in (definition_v1, definition_v2):
        assert definition.verdicts is not None
        assert set(definition.verdicts.model_dump()) == {
            "case", "not_case", "not_ascertainable", "review"
        }


def test_a_frozen_definition_is_never_overwritten(tmp_path, body, catalog):
    from aelayer.phenotype.loader import DefinitionCatalog

    write(tmp_path, body)
    catalogue = DefinitionCatalog(tmp_path, catalog)
    with pytest.raises(DefinitionError, match="frozen and will not be overwritten"):
        catalogue.write_candidate(dict(body, label="Quietly different"))


def test_a_draft_never_becomes_the_default(tmp_path, body, catalog):
    """A draft's content can change under the same version number."""
    from aelayer.phenotype.loader import DefinitionCatalog

    write(tmp_path, body)
    write(tmp_path, dict(body, version=3, status="draft"),
          "te_truncal_rash.v3.yaml")
    catalogue = DefinitionCatalog(tmp_path, catalog)
    assert catalogue.get("te_truncal_rash").version == 1
    with pytest.raises(DefinitionError, match="is a draft"):
        catalogue.get("te_truncal_rash", 3)
    assert catalogue.get("te_truncal_rash", 3, allow_draft=True).version == 3
