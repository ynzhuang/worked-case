"""Backends for the model path, and the contract every one of them meets.

1. Output validates against the target schema, or it is rejected outright.
2. Every populated value carries at least one span into the source text.
3. **Abstention is a valid answer.** A field the text does not support comes
   back null with ``collection_state: unknown``. A guess is a defect.
4. ``model_version`` and ``prompt_version`` are stamped on everything the path
   touches.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field as _dc_field
from typing import Any, Protocol

from ..catalog import ConceptCatalog, ExtractionConfig
from ..guards import ModelRequest
from ..models import Span

PROMPT_VERSION = "extract-prompt-1"


@dataclass
class ExtractedValue:
    """One value a backend proposes, with the span that supports it."""

    field: str
    value: Any
    spans: list[Span] = _dc_field(default_factory=list)
    confidence: float | None = None
    note: str = ""

    def is_grounded(self) -> bool:
        """A populated value without a span is rejected, not accepted warily."""
        return self.value is None or bool(self.spans)


@dataclass
class ExtractionResult:
    values: list[ExtractedValue] = _dc_field(default_factory=list)
    abstained: list[str] = _dc_field(default_factory=list)
    backend: str = ""
    model_version: str | None = None
    prompt_version: str = PROMPT_VERSION
    notes: list[str] = _dc_field(default_factory=list)

    def by_field(self) -> dict[str, ExtractedValue]:
        return {v.field: v for v in self.values}


class Backend(Protocol):
    name: str
    model_version: str | None

    def available(self) -> bool: ...

    def extract(self, request: ModelRequest) -> ExtractionResult: ...


# --------------------------------------------------------------------------
# Rules backend — a local clinical NLP baseline
# --------------------------------------------------------------------------


class RulesBackend:
    """Lexicons, ConText-style assertion scoping and pattern matching.

    A baseline, not a trained clinical NLP model, and described as such
    everywhere it appears.  It runs offline, which is what makes the default
    code path work with the network disconnected.
    """

    name = "rules"
    model_version = None

    def __init__(self, catalog: ConceptCatalog, config: ExtractionConfig):
        from .assertion import AssertionClassifier
        from .concepts import ConceptMatcher
        from .temporal import TemporalExtractor
        from .values import ValueExtractor

        self.catalog = catalog
        self.config = config
        self.matcher = ConceptMatcher(catalog, config)
        self.assertions = AssertionClassifier(config)
        self.values = ValueExtractor(catalog, config)
        self.temporal = TemporalExtractor(config)

    def available(self) -> bool:
        return True

    def extract(self, request: ModelRequest) -> ExtractionResult:
        from .text import split_sentences

        text = request.text
        sentences = split_sentences(text)
        result = ExtractionResult(backend=self.name, prompt_version=request.prompt_version)
        wanted = set(request.requested_fields)

        def span(field: str, start: int, end: int, value: Any) -> Span:
            return Span(
                doc_id=request.doc_id, start=start, end=end, field=field,
                extracted_value="" if value is None else str(value),
                text=text[start:end], kind="text",
            )

        if "symptoms" in wanted:
            found = []
            for match in self.matcher.find_symptoms(text):
                assertion = self.assertions.classify(
                    text, match.start, match.end, sentences
                )
                # A denied symptom is evidence against, not weak evidence for.
                if assertion.assertion == "present":
                    found.append(
                        {
                            "symptom": match.symptom,
                            "span": span("symptoms", match.start, match.end,
                                         match.symptom),
                        }
                    )
            denial = self.values.rescue_treatment(text, field="symptoms_absent")
            if found:
                result.values.append(
                    ExtractedValue(
                        "symptoms", found,
                        spans=[f["span"] for f in found],
                        confidence=self.config.confidence_for("lexicon_exact", 0.95),
                    )
                )
                result.values.append(
                    ExtractedValue(
                        "symptoms_assessed", True,
                        spans=[f["span"] for f in found],
                    )
                )
            elif denial is not None:
                # The source looked for symptoms and found none. That is a
                # finding; an unmentioned symptom list is not.
                result.values.append(
                    ExtractedValue(
                        "symptoms", [],
                        spans=[span("symptoms", denial.start, denial.end, "none")],
                        note="symptoms were explicitly denied in the narrative",
                    )
                )
                result.values.append(
                    ExtractedValue(
                        "symptoms_assessed", True,
                        spans=[span("symptoms_assessed", denial.start, denial.end,
                                    "true")],
                    )
                )
            else:
                result.abstained.append("symptoms")
                result.abstained.append("symptoms_assessed")

        if "labs" in wanted:
            hits = [h for h in self.values.find_labs(text) if not h.implausible]
            if hits:
                result.values.append(
                    ExtractedValue(
                        "labs",
                        [
                            {
                                "test": h.test, "value": h.value, "unit": h.unit,
                                "canonical_value": h.canonical_value,
                                "canonical_unit": h.canonical_unit,
                                "span": span("labs", h.start, h.end,
                                             f"{h.test}={h.value}{h.unit}"),
                            }
                            for h in hits
                        ],
                        spans=[span("labs", h.start, h.end, h.test) for h in hits],
                        confidence=max(h.confidence for h in hits),
                    )
                )
            else:
                result.abstained.append("labs")

        if "standardized_concept" in wanted:
            # Only an asserted mention supports a concept. A narrative that
            # mentions hypoglycemia in order to rule it out is evidence
            # against, and standardizing on it would invert the finding.
            proposed = None
            for mention in self.matcher.find_concepts(text, sentences):
                verdict = self.assertions.classify(
                    text, mention.start, mention.end, sentences
                )
                if verdict.assertion == "present":
                    proposed = (mention, verdict)
                    break
            if proposed is None:
                result.abstained.append("standardized_concept")
            else:
                mention, _verdict = proposed
                result.values.append(
                    ExtractedValue(
                        "standardized_concept", mention.concept_id,
                        spans=[span("standardized_concept", mention.start,
                                    mention.end, mention.concept_id)],
                        confidence=mention.confidence,
                        note=f"from an asserted verbatim mention ({mention.kind})",
                    )
                )

        if "assertion" in wanted:
            mentions = self.matcher.find_concepts(text, sentences)
            if mentions:
                first = mentions[0]
                verdict = self.assertions.classify(
                    text, first.start, first.end, sentences
                )
                anchor = (
                    (verdict.cue_start, verdict.cue_end)
                    if verdict.cue_start is not None
                    else (first.start, first.end)
                )
                result.values.append(
                    ExtractedValue(
                        "assertion", verdict.assertion,
                        spans=[span("assertion", anchor[0], anchor[1],
                                    verdict.assertion)],
                        confidence=verdict.confidence, note=verdict.rule,
                    )
                )
            else:
                result.abstained.append("assertion")

        for field in ("severity", "relatedness", "action_taken", "outcome"):
            if field not in wanted:
                continue
            hit = self.values.single_value(text, self._cue_field(field))
            if hit is None:
                result.abstained.append(field)
                continue
            value = self._map_value(field, hit.value)
            if value is None:
                result.abstained.append(field)
                continue
            result.values.append(
                ExtractedValue(
                    field, value,
                    spans=[span(field, hit.start, hit.end, value)],
                    confidence=hit.confidence,
                )
            )

        if "symptoms_assessed" in wanted and "symptoms" not in wanted:
            result.abstained.append("symptoms_assessed")

        if "coded_term" in wanted:
            # The model path never invents a dictionary term. Where the study
            # did not code the event, it stays uncoded; the narrative can
            # support a concept, but not a coded value that was never assigned.
            result.abstained.append("coded_term")

        for field in ("onset_datetime", "end_datetime"):
            if field in wanted:
                result.abstained.append(field)

        return result

    @staticmethod
    def _cue_field(field: str) -> str:
        return field

    @staticmethod
    def _map_value(field: str, value: str) -> str | None:
        """Map an extraction-config cue value onto the canonical codelist."""
        mapping = {
            "action_taken": {
                "dose_reduced": "dose_reduced",
                "dose_interrupted": "drug_interrupted",
                "drug_withdrawn": "drug_withdrawn",
                "none": "dose_not_changed",
                "unknown": "unknown",
            },
            "outcome": {
                "resolved": "recovered", "resolving": "recovering",
                "not_resolved": "not_recovered", "fatal": "fatal",
                "unknown": "unknown",
            },
        }
        if field in mapping:
            return mapping[field].get(value)
        return value


# --------------------------------------------------------------------------
# LLM backend — optional
# --------------------------------------------------------------------------

_LLM_SYSTEM = """You extract clinical facts from an adverse event narrative.

Return a single JSON object and nothing else. No prose, no markdown fence.

For each requested field, return either a value with the exact character offsets
in the source text that support it, or null.

{{"fields": {{"<name>": {{"value": <value|null>,
                          "spans": [{{"start": <int>, "end": <int>}}],
                          "confidence": <0..1>}}}}}}

Rules you must follow:
- If the text does not support a field, return null for it. Abstaining is
  correct; guessing is an error.
- Every non-null value must have at least one span whose offsets locate the
  supporting text.
- Do not infer a value from clinical plausibility. Only report what the text
  says.

Requested fields: {fields}
"""


class LLMBackend:
    """Optional. Used only when an API key is present.

    Its output is schema-validated and rejected on failure rather than
    repaired: a repaired extraction is one nobody checked.
    """

    name = "llm"

    def __init__(self, model: str | None = None):
        self.model_version = model or os.environ.get(
            "AELAYER_MODEL", "claude-sonnet-5"
        )

    def available(self) -> bool:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return False
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return True

    def extract(self, request: ModelRequest) -> ExtractionResult:  # pragma: no cover
        import anthropic

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=self.model_version,
            max_tokens=2048,
            system=_LLM_SYSTEM.format(fields=list(request.requested_fields)),
            messages=[{"role": "user", "content": request.text}],
        )
        raw = "".join(
            block.text for block in response.content
            if getattr(block, "type", "") == "text"
        ).strip()

        result = ExtractionResult(
            backend=self.name, model_version=self.model_version,
            prompt_version=request.prompt_version,
        )
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            result.notes.append(
                f"model returned invalid JSON and was rejected: {exc}"
            )
            result.abstained.extend(request.requested_fields)
            return result

        for field, body in (payload.get("fields") or {}).items():
            if field not in request.requested_fields:
                result.notes.append(
                    f"model returned unrequested field {field!r}; discarded"
                )
                continue
            value = body.get("value") if isinstance(body, dict) else None
            if value is None:
                result.abstained.append(field)
                continue
            spans = []
            for raw_span in (body.get("spans") or []):
                try:
                    start, end = int(raw_span["start"]), int(raw_span["end"])
                except (KeyError, TypeError, ValueError):
                    continue
                if 0 <= start < end <= len(request.text):
                    spans.append(
                        Span(
                            doc_id=request.doc_id, start=start, end=end,
                            field=field, extracted_value=str(value),
                            text=request.text[start:end], kind="text",
                        )
                    )
            if not spans:
                # Ungrounded values are rejected, not accepted with a caveat.
                result.notes.append(
                    f"model returned {field!r} with no resolvable span; rejected"
                )
                result.abstained.append(field)
                continue
            result.values.append(
                ExtractedValue(field, value, spans=spans,
                               confidence=body.get("confidence"))
            )
        return result


def select_backend(
    catalog: ConceptCatalog, config: ExtractionConfig, prefer: str = "auto"
) -> tuple[Backend, list[str]]:
    """Pick a backend, and say plainly which one and why.

    With no API key the LLM backend is unavailable and the rules baseline runs
    instead.  The choice is stamped on every record and reported in the run
    manifest, because "which system produced this value" is part of the value.
    """
    notes: list[str] = []
    if prefer in ("llm", "auto"):
        llm = LLMBackend()
        if llm.available():
            return llm, ["model path: LLM backend"]
        if prefer == "llm":
            notes.append(
                "the LLM backend was requested but no API key is available; "
                "the model path degraded to the offline rules baseline"
            )
        else:
            notes.append(
                "no LLM backend is configured; the model path is the offline "
                "rules baseline (lexicons and cue scoping, not a trained model)"
            )
    return RulesBackend(catalog, config), notes
