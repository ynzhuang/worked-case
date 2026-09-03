"""Both backends meet the same contract, and the LLM one is validated hardest.

None of these tests needs credentials: the validation is the point, and it
happens in this repository regardless of what produced the payload.
"""

from __future__ import annotations

import pytest

from aelayer.extract.backends import (
    ExtractionRequest, LLMBackend, RulesBackend, select_backend,
)

MODIFIER = "mucosal_involvement"
TEXT = "rash with oral ulceration"


@pytest.fixture
def llm(configs):
    return LLMBackend(configs.catalog, configs.extraction, "extract-4.0.0-test")


def _request():
    return ExtractionRequest(
        doc_id="AE:R1:AETERM", text=TEXT, modifiers=(MODIFIER,),
        source_kind="reported_term", source_variable="AETERM",
    )


def _payload(**overrides):
    body = {"assertion": "present", "value": "ORAL", "start": 10,
            "end": len(TEXT)}
    body.update(overrides)
    return body


def test_a_valid_response_is_accepted(llm):
    attribute = llm._validate(MODIFIER, _payload(), _request())
    assert attribute is not None
    assert attribute.assertion == "present"
    assert attribute.value == "ORAL"
    assert attribute.method == "extracted"
    assert attribute.evidence


def test_an_assertion_outside_the_three_is_rejected(llm):
    assert llm._validate(MODIFIER, _payload(assertion="probably"), _request()) is None


def test_a_value_outside_the_catalogue_is_rejected_whole(llm):
    """Not rounded off to the nearest declared value."""
    assert llm._validate(MODIFIER, _payload(value="SOMEWHERE"), _request()) is None


def test_a_value_is_optional_but_an_assertion_is_not(llm):
    attribute = llm._validate(MODIFIER, _payload(value=None), _request())
    assert attribute is not None
    assert attribute.value is None
    assert attribute.assertion == "present"


@pytest.mark.parametrize(
    "span", [{"start": -1}, {"end": 9999}, {"start": 20, "end": 5},
             {"start": "x"}],
)
def test_a_span_that_does_not_land_in_the_text_is_rejected(llm, span):
    body = {**_payload(), **span}
    assert llm._validate(MODIFIER, body, _request()) is None


def test_an_answer_with_no_span_at_all_is_rejected(llm):
    """A value with no text behind it cannot be checked by anyone."""
    body = {k: v for k, v in _payload().items() if k != "start"}
    assert llm._validate(MODIFIER, body, _request()) is None


def test_a_surface_form_is_normalized_before_it_leaves_the_backend(llm):
    attribute = llm._validate(
        MODIFIER, _payload(value="oral ulceration"), _request()
    )
    assert attribute is not None
    assert attribute.value == "ORAL"


def test_the_versions_travel_with_every_value(llm):
    attribute = llm._validate(MODIFIER, _payload(), _request())
    assert attribute.versions["extractor"] == "extract-4.0.0-test"
    assert attribute.versions["prompt"] == LLMBackend.prompt_version
    assert attribute.versions["backend"] == "llm"


def test_an_unusable_response_abstains_rather_than_guessing(llm, monkeypatch):
    monkeypatch.setattr(llm, "_call", lambda request: None)
    result = llm.extract(_request())
    assert result.values == {}
    assert result.abstained == [MODIFIER]
    assert any("no value was invented" in note for note in result.notes)


def test_a_null_answer_is_an_abstention_not_a_failure(llm, monkeypatch):
    monkeypatch.setattr(llm, "_call", lambda request: {MODIFIER: None})
    result = llm.extract(_request())
    assert result.abstained == [MODIFIER]
    assert not result.notes


def test_an_invalid_answer_is_discarded_and_reported(llm, monkeypatch):
    monkeypatch.setattr(
        llm, "_call", lambda request: {MODIFIER: _payload(value="SOMEWHERE")}
    )
    result = llm.extract(_request())
    assert result.abstained == [MODIFIER]
    assert any("discarded rather than accepted" in n for n in result.notes)


def test_the_prompt_tells_the_model_that_silence_and_no_are_different():
    assert "Saying nothing and saying no are different answers" in LLMBackend.SYSTEM
    assert "Do not infer, do not guess" in LLMBackend.SYSTEM


def test_the_prompt_offers_only_declared_values(llm):
    vocabulary = llm.vocabulary((MODIFIER,))
    assert "ORAL" in vocabulary
    assert MODIFIER in vocabulary


# -- backend selection --------------------------------------------------------


def test_the_default_is_the_offline_rules_baseline(configs, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    backend, notes = select_backend(configs.catalog, configs.extraction, "v", "auto")
    assert isinstance(backend, RulesBackend)
    assert any("not a trained clinical NLP model" in n for n in notes)


def test_asking_for_an_llm_without_credentials_degrades_and_says_so(configs,
                                                                    monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    backend, notes = select_backend(configs.catalog, configs.extraction, "v", "llm")
    assert isinstance(backend, RulesBackend)
    assert any("degraded to the offline rules baseline" in n for n in notes)


def test_an_available_llm_backend_says_the_run_is_not_reproducible(configs,
                                                                   monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    backend, notes = select_backend(configs.catalog, configs.extraction, "v", "llm")
    assert isinstance(backend, LLMBackend)
    assert any("not bit-reproducible" in n for n in notes)


def test_the_rules_backend_stamps_its_versions(configs):
    backend = RulesBackend(configs.catalog, configs.extraction, "extract-4.0.0-test")
    result = backend.extract(_request())
    attribute = result.values[MODIFIER]
    assert attribute.versions["extractor"] == "extract-4.0.0-test"
    assert attribute.versions["backend"] == "rules"
    assert attribute.versions["prompt"] == RulesBackend.prompt_version
