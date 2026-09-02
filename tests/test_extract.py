"""The model path: finding modifiers in language and normalizing them."""

from __future__ import annotations

import pytest

from aelayer.extract.backends import ExtractionRequest, LLMBackend, RulesBackend, select_backend
from aelayer.extract.modifiers import ModifierExtractor


@pytest.fixture(scope="module")
def extractor(catalog, configs):
    return ModifierExtractor(catalog, configs.extraction)


@pytest.fixture(scope="module")
def backend(catalog, configs):
    return RulesBackend(catalog, configs.extraction, configs.extractor_version)


# -- anchoring --------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("rash on the chest", "CHEST"),
    ("skin rash over the anterior chest", "CHEST"),
    ("maculopapular rash affecting the abdomen", "ABDOMEN"),
    ("rash involving the lower back", "BACK"),
    ("chest rash", "CHEST"),
    ("eruption over the periumbilical area", "ABDOMEN"),
])
def test_a_site_joined_to_the_event_is_found(extractor, text, expected):
    hit = extractor.best(text, "location", "RASH")
    assert hit is not None, text
    assert hit.value == expected


def test_a_site_belonging_to_another_event_is_not_attached(extractor):
    """"rash resolved; history of eczema on the back" is not a truncal rash."""
    assert extractor.best(
        "rash resolved; history of eczema on the back", "location", "RASH"
    ) is None


def test_a_site_in_another_sentence_is_not_attached(extractor):
    assert extractor.best(
        "Rash noted. The patient has a tattoo on the back.", "location", "RASH"
    ) is None


def test_a_connector_only_counts_when_nothing_else_intervenes(extractor):
    hits = {h.value: h for h in extractor.find("rash on the chest and the back",
                                               "location", "RASH")}
    assert hits["CHEST"].confidence > hits["BACK"].confidence


def test_confidence_reflects_how_the_site_was_attached(extractor):
    joined = extractor.best("rash on the chest", "location", "RASH")
    adjacent = extractor.best("chest rash", "location", "RASH")
    assert joined.confidence > adjacent.confidence
    assert "joined to the event" in joined.rule


def test_a_comment_gets_its_own_confidence(extractor):
    """A comment is written about this record, so it supports more than prose."""
    prose = extractor.best(
        "Site clarification: rash noted, the chest was affected.", "location", "RASH"
    )
    comment = extractor.best(
        "Site clarification: rash noted, the chest was affected.", "location",
        "RASH", "comment",
    )
    assert comment.confidence >= prose.confidence


# -- abstention -------------------------------------------------------------


def test_a_term_with_no_site_yields_nothing(extractor):
    assert extractor.best("rash", "location", "RASH") is None
    assert extractor.best("Skin disorder", "location", "RASH") is None


def test_a_word_no_lexicon_carries_yields_nothing(extractor):
    """Abstaining is correct: inventing a catalogue value would be the defect."""
    for text in ("rash over the torso", "rash on the midriff",
                 "rash affecting the shoulder blade area"):
        assert extractor.best(text, "location", "RASH") is None, text


def test_two_equally_supported_values_yield_nothing(extractor):
    hits = extractor.find("rash on the chest, rash on the back", "location", "RASH")
    values = {h.value for h in hits if h.confidence == max(x.confidence for x in hits)}
    if len(values) > 1:
        assert extractor.best("rash on the chest, rash on the back",
                              "location", "RASH") is None


# -- patterns and qualities -------------------------------------------------


def test_a_pattern_is_normalized_to_the_catalogue(extractor):
    hit = extractor.best("morbilliform rash on the chest", "pattern", "RASH")
    assert hit.value == "MACULOPAPULAR"


def test_quality_descriptors_are_found_but_not_normalized(extractor):
    hits = extractor.qualities("itchy spreading rash on the chest")
    assert {h.value for h in hits} == {"itchy", "spreading"}
    assert all(h.attribute == "quality" for h in hits)


# -- the backend contract ---------------------------------------------------


def test_the_backend_returns_attributes_that_validate(backend):
    result = backend.extract(ExtractionRequest(
        doc_id="AE:R1:AETERM", text="rash on the chest",
        attributes=("location", "pattern"), concept_id="RASH",
    ))
    assert result.values["location"].value == "CHEST"
    assert result.values["location"].method == "extracted"
    assert result.values["location"].evidence
    assert "pattern" in result.abstained


def test_a_span_points_at_the_characters_it_claims(backend):
    text = "maculopapular rash over the lower back"
    result = backend.extract(ExtractionRequest(
        doc_id="D1", text=text, attributes=("location",), concept_id="RASH",
    ))
    span = result.values["location"].evidence[0]
    assert text[span.start:span.end].lower() == span.text.lower()
    assert span.extracted_value == "BACK"


def test_the_backend_stamps_the_versions_that_produced_the_value(backend, configs):
    result = backend.extract(ExtractionRequest(
        doc_id="D1", text="rash on the chest", attributes=("location",),
        concept_id="RASH",
    ))
    attribute = result.values["location"]
    assert attribute.extractor_version == configs.extractor_version
    assert attribute.prompt_version


def test_the_backend_reports_abstention_rather_than_guessing(backend):
    result = backend.extract(ExtractionRequest(
        doc_id="D1", text="rash", attributes=("location",), concept_id="RASH",
    ))
    assert result.values == {}
    assert result.abstained == ["location"]


# -- backend selection ------------------------------------------------------


def test_without_credentials_the_model_path_is_the_offline_baseline(
    catalog, configs, monkeypatch
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    backend, notes = select_backend(
        catalog, configs.extraction, configs.extractor_version, "auto"
    )
    assert isinstance(backend, RulesBackend)
    assert any("offline rules baseline" in note for note in notes)


def test_asking_for_an_llm_without_credentials_says_it_degraded(
    catalog, configs, monkeypatch
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    backend, notes = select_backend(
        catalog, configs.extraction, configs.extractor_version, "llm"
    )
    assert isinstance(backend, RulesBackend)
    assert any("degraded to the offline rules baseline" in n for n in notes)


def test_the_llm_backend_validates_its_output_against_the_catalogue(catalog, configs):
    llm = LLMBackend(catalog, configs.extraction, configs.extractor_version)
    request = ExtractionRequest(
        doc_id="D1", text="rash on the chest", attributes=("location",),
    )
    assert llm._validate("location", {"value": "ELBOW_PIT", "start": 0, "end": 4},
                         request) is None
    assert llm._validate("location", {"value": "CHEST", "start": 99, "end": 200},
                         request) is None
    good = llm._validate(
        "location", {"value": "anterior chest", "start": 12, "end": 17}, request
    )
    assert good is not None and good.value == "CHEST"
