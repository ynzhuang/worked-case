"""Extraction backends, behind one interface.

The default backend is the deterministic lexicon-and-scope extractor, so
everything runs offline. An LLM backend is optional and swappable; both meet the
same contract:

1. the output validates against the modifier catalogue, or it is rejected
2. every extracted value carries a span into the source text
3. the answer is an **assertion** — present, absent or uncertain — and never a
   bare value, because "the source said no" and "the source said nothing" are
   different answers and the layer must be able to tell them apart
4. **abstention is correct behaviour**: where the text does not support an
   answer, the answer is no answer, reported as a rate rather than counted as a
   failure
5. the extractor, model and prompt versions are stamped on every value

With no network the LLM backend is unavailable, and the engine degrades to the
rules backend and says so in its notes.

A request may only ever concern ``language_variation``. Coded-concept variation
and terminology-version variation are resolved by a concept set and a
mechanical map; ``guards.py`` refuses a request that claims otherwise.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field as _dc_field
from typing import Any, Protocol

from ..catalog import ConceptCatalog, ExtractionConfig
from ..models import ASSERTIONS, Attribute, Span
from .mentions import MentionFinder

#: The only normalization mechanism a backend is ever asked about.
LANGUAGE_VARIATION = "language_variation"


@dataclass(frozen=True)
class ExtractionRequest:
    """A question for the model path.

    It carries text and nothing else. A value the CRF already settled is not a
    question, and ``guards.py`` refuses a request that names one.
    """

    doc_id: str
    text: str
    modifiers: tuple[str, ...]
    concept_id: str | None = None
    source_kind: str = "reported_term"
    source_variable: str = "AETERM"
    mechanism: str = LANGUAGE_VARIATION


@dataclass
class ExtractionResult:
    """What one request produced, including what it declined to answer."""

    values: dict[str, Attribute[str]] = _dc_field(default_factory=dict)
    abstained: list[str] = _dc_field(default_factory=list)
    notes: list[str] = _dc_field(default_factory=list)


class Backend(Protocol):
    name: str
    model_version: str | None
    prompt_version: str | None

    def extract(self, request: ExtractionRequest) -> ExtractionResult: ...


class RulesBackend:
    """Lexicon and scope rules. Deterministic, offline, and honest about it."""

    name = "rules"
    model_version = None
    prompt_version = "rules-4.0.0"

    def __init__(
        self, catalog: ConceptCatalog, config: ExtractionConfig,
        extractor_version: str = "",
    ):
        self.catalog = catalog
        self.config = config
        self.extractor_version = extractor_version
        self.finder = MentionFinder(catalog, config)

    def versions(self) -> dict[str, str]:
        return {
            k: v for k, v in {
                "extractor": self.extractor_version,
                "prompt": self.prompt_version,
                "backend": self.name,
            }.items() if v
        }

    def extract(self, request: ExtractionRequest) -> ExtractionResult:
        result = ExtractionResult()
        for modifier in request.modifiers:
            mention = self.finder.best(
                request.text, modifier, request.concept_id, request.source_kind
            )
            if mention is None:
                result.abstained.append(modifier)
                continue
            result.values[modifier] = Attribute[str].extracted(
                mention.assertion,  # type: ignore[arg-type]
                request.source_variable,
                [mention.span(request.doc_id, modifier)],
                value=mention.value,
                source=request.source_kind,  # type: ignore[arg-type]
                confidence=mention.confidence,
                versions=self.versions(),
                note=f"{mention.surface!r}: {mention.rule}",
            )
        return result


class LLMBackend:
    """Optional. Same contract, and the same refusal to guess.

    Kept deliberately small: the point of the interface is that a model can be
    swapped in without any other module learning about it, and that its output
    is validated exactly as strictly as the rules backend's.
    """

    name = "llm"
    prompt_version = "extract-prompt-4"

    SYSTEM = (
        "You read one adverse event record and report what its text says "
        "about the requested modifiers.\n\n"
        "Return one JSON object and nothing else. For each requested modifier "
        "return either an answer or null.\n\n"
        "An answer has an `assertion`, which is one of:\n"
        "  present   - the text says the modifier was there\n"
        "  absent    - the text says it was looked for and was not there\n"
        "  uncertain - the text hedges and does not settle it\n"
        "It also has the exact character offsets of the text that supports it, "
        "and optionally a `value` from the permitted list.\n\n"
        "Returning null is correct whenever the text does not address the "
        "modifier at all. Saying nothing and saying no are different answers "
        "and must never be conflated. Do not infer, do not guess, and do not "
        "use knowledge outside the text provided.\n\n"
        'Shape: {{"mucosal_involvement": {{"assertion": "absent", '
        '"value": null, "start": 12, "end": 41}}, "photosensitivity": null}}\n\n'
        "Permitted values:\n{vocabulary}"
    )

    def __init__(
        self, catalog: ConceptCatalog, config: ExtractionConfig,
        extractor_version: str = "", model: str | None = None,
    ):
        self.catalog = catalog
        self.config = config
        self.extractor_version = extractor_version
        self.model_version = model or os.environ.get("AELAYER_MODEL", "")
        self._client = None

    @staticmethod
    def available() -> bool:
        """Only with a key present. Absent one, the engine degrades and says so."""
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def versions(self) -> dict[str, str]:
        return {
            k: v for k, v in {
                "extractor": self.extractor_version,
                "prompt": self.prompt_version,
                "model": self.model_version,
                "backend": self.name,
            }.items() if v
        }

    def vocabulary(self, modifiers: tuple[str, ...]) -> str:
        lines = []
        for modifier in modifiers:
            if modifier in self.catalog.modifiers:
                catalogue = self.catalog.modifier(modifier)
                lines.append(f"{modifier}: {', '.join(catalogue.value_ids())}")
        return "\n".join(lines)

    def extract(self, request: ExtractionRequest) -> ExtractionResult:
        result = ExtractionResult()
        payload = self._call(request)
        if payload is None:
            result.notes.append(
                "the LLM backend returned nothing usable; no value was invented "
                "and every requested modifier is reported as abstained"
            )
            result.abstained.extend(request.modifiers)
            return result

        for modifier in request.modifiers:
            body = payload.get(modifier)
            if not isinstance(body, dict) or body.get("assertion") in (None, ""):
                result.abstained.append(modifier)
                continue
            value = self._validate(modifier, body, request)
            if value is None:
                result.abstained.append(modifier)
                result.notes.append(
                    f"{modifier}: the response did not validate against the "
                    f"catalogue or its span did not match the text, so it was "
                    f"discarded rather than accepted"
                )
                continue
            result.values[modifier] = value
        return result

    def _validate(
        self, modifier: str, body: dict[str, Any], request: ExtractionRequest
    ) -> Attribute[str] | None:
        assertion = str(body.get("assertion") or "").strip().lower()
        if assertion not in ASSERTIONS:
            return None
        catalogue = self.catalog.modifiers.get(modifier)
        raw = body.get("value")
        value: str | None = None
        if raw not in (None, ""):
            text = str(raw).strip()
            value = (
                text if catalogue and text in catalogue.values
                else (catalogue.normalize(text) if catalogue else None)
            )
            if value is None:
                # A value outside the declared space is not a near miss to be
                # rounded off. The response is discarded whole.
                return None
        try:
            start, end = int(body["start"]), int(body["end"])
        except (KeyError, TypeError, ValueError):
            return None
        if not (0 <= start < end <= len(request.text)):
            return None
        span = Span(
            doc_id=request.doc_id, start=start, end=end, field=modifier,
            extracted_value=value or assertion, text=request.text[start:end],
            kind="text",
        )
        return Attribute[str].extracted(
            assertion,  # type: ignore[arg-type]
            request.source_variable, [span], value=value,
            source=request.source_kind,  # type: ignore[arg-type]
            confidence=float(body.get("confidence") or 0.8),
            versions=self.versions(),
            note="returned by the LLM backend and validated against the catalogue",
        )

    def _call(self, request: ExtractionRequest) -> dict[str, Any] | None:  # pragma: no cover
        try:
            if self._client is None:
                import anthropic

                self._client = anthropic.Anthropic()
            response = self._client.messages.create(
                model=self.model_version or "claude-sonnet-5",
                max_tokens=512,
                system=self.SYSTEM.format(
                    vocabulary=self.vocabulary(request.modifiers)
                ),
                messages=[{"role": "user", "content": request.text}],
            )
            return json.loads(response.content[0].text)
        except Exception:
            return None


def select_backend(
    catalog: ConceptCatalog, config: ExtractionConfig, extractor_version: str,
    preference: str = "auto",
) -> tuple[Backend, list[str]]:
    """Pick a backend and say plainly what was picked and why."""
    notes: list[str] = []
    if preference == "llm":
        if LLMBackend.available():
            return LLMBackend(catalog, config, extractor_version), [
                "the LLM backend is in use; extraction is not bit-reproducible "
                "and the manifest records the model and prompt versions rather "
                "than guaranteeing the output"
            ]
        notes.append(
            "an LLM backend was requested but no credentials are present; the "
            "model path degraded to the offline rules baseline"
        )
    elif preference == "auto" and LLMBackend.available():
        return LLMBackend(catalog, config, extractor_version), [
            "an LLM backend was available and selected automatically"
        ]
    notes.append(
        "the model path is the offline rules baseline: lexicons and cue scoping "
        "from config/, not a trained clinical NLP model"
    )
    return RulesBackend(catalog, config, extractor_version), notes
