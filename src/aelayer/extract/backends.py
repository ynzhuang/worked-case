"""Extraction backends, behind one interface.

The default backend is the deterministic lexicon-and-scope extractor, so
everything runs offline. An LLM backend is optional and swappable; both meet the
same contract:

1. the output validates against the attribute schema, or it is rejected
2. every extracted value carries a span into the source text
3. **abstention is correct behaviour** — where the text does not support a
   value, the answer is no value, and that is reported as a rate rather than
   counted as a failure
4. values normalize to the concept catalogue before they leave the backend
5. the model and prompt versions are stamped on every record the path touches

With no network, the LLM backend is unavailable and the engine degrades to the
rules backend and says so in its notes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field as _dc_field
from typing import Any, Protocol

from ..catalog import ConceptCatalog, ExtractionConfig
from ..models import Attribute, Span
from .modifiers import ModifierExtractor


@dataclass(frozen=True)
class ExtractionRequest:
    """A question for the model path.

    It carries text and nothing else. A value the CRF already settled is not a
    question, and ``guards.py`` refuses a request that names one.
    """

    doc_id: str
    text: str
    attributes: tuple[str, ...]
    concept_id: str | None = None
    source_kind: str = "reported_term"
    source_variable: str = "AETERM"


@dataclass
class ExtractionResult:
    values: dict[str, Attribute[str]] = _dc_field(default_factory=dict)
    abstained: list[str] = _dc_field(default_factory=list)
    qualities: list[str] = _dc_field(default_factory=list)
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
    prompt_version = "rules-3.0.0"

    def __init__(
        self, catalog: ConceptCatalog, config: ExtractionConfig,
        extractor_version: str = "",
    ):
        self.catalog = catalog
        self.config = config
        self.extractor_version = extractor_version
        self.modifiers = ModifierExtractor(catalog, config)

    def extract(self, request: ExtractionRequest) -> ExtractionResult:
        result = ExtractionResult()
        for attribute in request.attributes:
            if attribute == "quality":
                result.qualities = sorted(
                    {h.value for h in self.modifiers.qualities(request.text)}
                )
                continue
            hit = self.modifiers.best(
                request.text, attribute, request.concept_id, request.source_kind
            )
            if hit is None:
                result.abstained.append(attribute)
                continue
            result.values[attribute] = Attribute[str].extracted(
                hit.value,
                request.source_variable,
                [hit.span(request.doc_id, attribute)],
                source=request.source_kind,  # type: ignore[arg-type]
                confidence=hit.confidence,
                extractor_version=self.extractor_version,
                prompt_version=self.prompt_version,
                note=f"{hit.surface!r} {hit.rule}",
            )
        return result


class LLMBackend:
    """Optional. Same contract, and the same refusal to guess.

    Kept deliberately small: the point of the interface is that a model can be
    swapped in without any other module learning about it, and that its output
    is validated exactly as strictly as the rules backend's.
    """

    name = "llm"
    prompt_version = "extract-prompt-3"

    SYSTEM = (
        "You extract clinical modifiers from an adverse event record.\n\n"
        "Return one JSON object and nothing else. For each requested attribute "
        "return either a value with the exact character offsets of the text "
        "that supports it, or null.\n\n"
        "Returning null is correct whenever the text does not state the "
        "attribute. Do not infer, do not guess, and do not use knowledge "
        "outside the text provided.\n\n"
        'Shape: {{"location": {{"value": "CHEST", "start": 12, "end": 17}}, '
        '"pattern": null}}\n\n'
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

    def vocabulary(self, attributes: tuple[str, ...]) -> str:
        lines = []
        for attribute in attributes:
            if attribute in self.catalog.attributes:
                catalogue = self.catalog.attribute(attribute)
                lines.append(f"{attribute}: {', '.join(catalogue.value_ids())}")
        return "\n".join(lines)

    def extract(self, request: ExtractionRequest) -> ExtractionResult:
        result = ExtractionResult()
        payload = self._call(request)
        if payload is None:
            result.notes.append(
                "the LLM backend returned nothing usable; no value was invented "
                "and every requested attribute is reported as abstained"
            )
            result.abstained.extend(request.attributes)
            return result

        for attribute in request.attributes:
            body = payload.get(attribute)
            if not isinstance(body, dict) or body.get("value") in (None, ""):
                result.abstained.append(attribute)
                continue
            value = self._validate(attribute, body, request)
            if value is None:
                result.abstained.append(attribute)
                result.notes.append(
                    f"{attribute}: the response did not validate against the "
                    f"catalogue or its span did not match the text, so it was "
                    f"discarded rather than accepted"
                )
                continue
            result.values[attribute] = value
        return result

    def _validate(
        self, attribute: str, body: dict[str, Any], request: ExtractionRequest
    ) -> Attribute[str] | None:
        catalogue = self.catalog.attributes.get(attribute)
        raw = str(body["value"]).strip()
        value = raw if catalogue and raw in catalogue.values else (
            catalogue.normalize(raw) if catalogue else None
        )
        if value is None:
            return None
        try:
            start, end = int(body["start"]), int(body["end"])
        except (KeyError, TypeError, ValueError):
            return None
        if not (0 <= start < end <= len(request.text)):
            return None
        span = Span(
            doc_id=request.doc_id, start=start, end=end, field=attribute,
            extracted_value=value, text=request.text[start:end], kind="text",
        )
        return Attribute[str].extracted(
            value, request.source_variable, [span],
            source=request.source_kind,  # type: ignore[arg-type]
            confidence=float(body.get("confidence") or 0.8),
            extractor_version=self.extractor_version,
            model_version=self.model_version or None,
            prompt_version=self.prompt_version,
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
                    vocabulary=self.vocabulary(request.attributes)
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
        "the model path is the offline rules baseline: lexicons and scope "
        "rules from config/, not a trained clinical NLP model"
    )
    return RulesBackend(catalog, config, extractor_version), notes
