"""Question -> ``PhenotypeQuerySpec``.

Two backends:

**Deterministic** (default, offline)
    Template and slot matching against the definition catalogue.  No network,
    no model, no hidden state.

**LLM** (optional, only when an API key is present)
    The model emits the spec as JSON and nothing else.  Its output is
    schema-validated and rejected on failure.  It never computes a statistic
    and never names a case.

Where the question leaves a rule underdetermined, both backends return a
``Clarification`` naming the specific ambiguity and its effect, rather than
choosing a default.  Silently picking a default is how a definitional choice
becomes invisible, and an invisible definitional choice is the failure this
system exists to prevent.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Sequence

from ..models import (
    ASSERTION_VALUES,
    EVIDENCE_STATE_VALUES,
    Clarification,
    PhenotypeDefinition,
    PhenotypeQuerySpec,
)
from ..pipeline import Pipeline

_STUDY_RE = re.compile(r"\bSTUDY[-_ ]?(\d+)\b", re.IGNORECASE)
_WINDOW_RE = re.compile(
    r"\bwithin\s+(?P<n>\d+)\s+(?P<unit>day|days|week|weeks)\b", re.IGNORECASE
)
_THRESHOLD_RE = re.compile(
    r"(?:below|under|less than|<)\s*(?P<value>\d{2,3}(?:\.\d)?)\s*"
    r"(?P<unit>mg/dl|mmol/l)?",
    re.IGNORECASE,
)

#: Words that mean one thing in a protocol and another in conversation.
#: "severe" is an intensity; "serious" is a regulatory category. Treating them
#: as one graded scale is the single most consequential mistake available here,
#: so the agent refuses to guess which the questioner meant.
_SEVERITY_WORDS = {"severe", "severity", "mild", "moderate"}
_SERIOUSNESS_WORDS = {
    "serious", "seriousness", "hospitalised", "hospitalized", "hospitalisation",
    "life-threatening", "life threatening", "fatal", "death",
}

_REVIEW_WORDS = {"review", "adjudication", "adjudicate", "possible", "borderline"}


@dataclass
class CompileResult:
    spec: PhenotypeQuerySpec | None
    clarification: Clarification | None
    matched_definition: PhenotypeDefinition | None = None
    trace: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.trace = self.trace or []

    @property
    def needs_clarification(self) -> bool:
        return self.clarification is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "needs_clarification": self.needs_clarification,
            "spec": self.spec.model_dump(mode="json") if self.spec else None,
            "clarification": (
                self.clarification.model_dump(mode="json")
                if self.clarification
                else None
            ),
            "definition": (
                f"{self.matched_definition.id}.v{self.matched_definition.version}"
                if self.matched_definition
                else None
            ),
            "trace": self.trace,
        }


# --------------------------------------------------------------------------
# Deterministic backend
# --------------------------------------------------------------------------


def _tokenise(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _score_definition(definition: PhenotypeDefinition, tokens: set[str]) -> int:
    """Overlap between the question and a definition's identifying words."""
    vocabulary = _tokenise(
        f"{definition.id} {definition.label} {definition.description} "
        f"{definition.concept.primary}"
    )
    stop = {"the", "of", "a", "in", "and", "with", "within", "for", "to", "days"}
    return len(tokens & (vocabulary - stop))


def compile_deterministic(
    question: str, pipeline: Pipeline, *, allow_draft: bool = False
) -> CompileResult:
    trace: list[str] = []
    tokens = _tokenise(question)
    definitions = [
        d for d in pipeline.definitions.all()
        if d.status != "draft" or allow_draft
    ]
    if not definitions:
        return CompileResult(
            None,
            Clarification(
                question=question,
                ambiguity="No runnable phenotype definition is available.",
                effect="There is nothing to evaluate the question against.",
                options=["Add a frozen definition to config/phenotypes/"],
            ),
            trace=["no definitions in catalogue"],
        )

    # Match on the definition *id*, represented by its latest runnable version.
    # Which version runs is a lifecycle question, not a text-similarity one: the
    # newest published version of the matched definition is the answer, whatever
    # the prose overlap of an older one happens to be.
    latest_by_id: dict[str, PhenotypeDefinition] = {}
    for definition in definitions:
        current = latest_by_id.get(definition.id)
        if current is None or definition.version > current.version:
            latest_by_id[definition.id] = definition

    scored = sorted(
        ((_score_definition(d, tokens), d) for d in latest_by_id.values()),
        key=lambda pair: (-pair[0], pair[1].id),
    )
    best_score = scored[0][0]
    trace.append(
        "definition match scores: "
        + ", ".join(f"{d.id} (latest v{d.version})={score}" for score, d in scored)
    )

    if best_score == 0:
        return CompileResult(
            None,
            Clarification(
                question=question,
                ambiguity=(
                    "The question does not name a phenotype this catalogue "
                    "defines."
                ),
                effect=(
                    "Without a definition there is no rule for what makes a "
                    "subject a case, and any count would be an invention."
                ),
                options=[f"{d.key}: {d.label}" for d in definitions],
            ),
            trace=trace,
        )

    top = [definition for score, definition in scored if score == best_score]
    if len(top) > 1:
        return CompileResult(
            None,
            Clarification(
                question=question,
                ambiguity=(
                    "The question matches more than one phenotype definition "
                    f"equally well: {', '.join(d.key for d in top)}."
                ),
                effect=(
                    "Each definition applies different rules and would produce "
                    "a different cohort."
                ),
                options=[f"{d.key}: {d.label}" for d in top],
            ),
            trace=trace,
        )

    definition = top[0]
    trace.append(f"selected {definition.key} ({definition.status})")

    clarification = _detect_ambiguity(question, tokens, definition)
    if clarification is not None:
        return CompileResult(None, clarification, definition, trace)

    studies = sorted(
        {f"STUDY-{int(m.group(1)):02d}" for m in _STUDY_RE.finditer(question)}
    )
    known = set(pipeline.store.studies())
    unknown = [s for s in studies if s not in known]
    if unknown:
        return CompileResult(
            None,
            Clarification(
                question=question,
                ambiguity=f"The question names studies not in this snapshot: {unknown}.",
                effect="Counts would silently omit the studies that were asked for.",
                options=sorted(known),
            ),
            definition,
            trace,
        )
    if studies:
        trace.append(f"restricted to studies {studies}")

    window = None
    if definition.window is not None:
        window = (definition.window.min, definition.window.max)
    match = _WINDOW_RE.search(question)
    if match:
        magnitude = int(match.group("n"))
        days = magnitude * (7 if match.group("unit").startswith("week") else 1)
        if definition.window is not None and days != definition.window.max:
            return CompileResult(
                None,
                Clarification(
                    question=question,
                    ambiguity=(
                        f"The question asks for a {days}-day window, but "
                        f"{definition.key} is defined with a window of "
                        f"{definition.window.min}-{definition.window.max} days."
                    ),
                    effect=(
                        "Changing the window changes which subjects are cases. "
                        "That is a change to the case definition, so it needs a "
                        "new definition version rather than a query parameter, "
                        "or the question should be run against the definition "
                        "as written."
                    ),
                    options=[
                        f"Run {definition.key} as written "
                        f"({definition.window.min}-{definition.window.max} days)",
                        f"Create v{pipeline.definitions.next_version(definition.id)} "
                        f"with max={days} and run that",
                    ],
                ),
                definition,
                trace,
            )

    threshold = _THRESHOLD_RE.search(question)
    if threshold is not None:
        declared = _declared_lab_threshold(definition)
        asked = float(threshold.group("value"))
        if declared is not None and abs(declared[0] - asked) > 1e-9:
            return CompileResult(
                None,
                Clarification(
                    question=question,
                    ambiguity=(
                        f"The question asks for a threshold of {asked:g}"
                        f"{threshold.group('unit') or ''}, but {definition.key} "
                        f"uses {declared[0]:g} {declared[1]}."
                    ),
                    effect=(
                        "The threshold decides which subjects reach the "
                        "`supported` state, and moving it moves the case count. "
                        "It is part of the definition, not of the query."
                    ),
                    options=[
                        f"Run {definition.key} as written ({declared[0]:g} {declared[1]})",
                        f"Create v{pipeline.definitions.next_version(definition.id)} "
                        f"with the threshold at {asked:g} and run that",
                    ],
                ),
                definition,
                trace,
            )

    evidence_state = list(definition.case_definition.primary_set)
    if tokens & _REVIEW_WORDS:
        evidence_state = sorted(
            set(evidence_state) | set(definition.case_definition.review_set)
        )
        trace.append("question asks about review as well as primary cases")

    spec = PhenotypeQuerySpec(
        question=question,
        definition_id=definition.id,
        definition_version=definition.version,
        studies=studies or sorted(known),
        concept=definition.concept.primary,
        assertion=list(definition.assertion.require),
        evidence_state=evidence_state,
        window=window,
        anchor=definition.anchor.event if definition.anchor else None,
        retrieval_mode="lexical",
        top_k=20,
        notes=[
            f"Compiled deterministically against {definition.key}, "
            f"status {definition.status}.",
            "Counts follow the definition as written; the agent applied no "
            "default of its own.",
        ],
        backend="deterministic",
    )
    return CompileResult(spec, None, definition, trace)


def _declared_lab_threshold(
    definition: PhenotypeDefinition,
) -> tuple[float, str] | None:
    """The first lab threshold the definition declares, for comparison."""
    def walk(condition: Any) -> tuple[float, str] | None:
        if isinstance(condition, list):
            for item in condition:
                found = walk(item)
                if found:
                    return found
            return None
        if not isinstance(condition, dict):
            return None
        for key, body in condition.items():
            if key == "lab" and isinstance(body, dict):
                return float(body["value"]), str(body.get("unit") or "")
            found = walk(body)
            if found:
                return found
        return None

    for rule in definition.evidence_rules:
        found = walk(rule.when)
        if found:
            return found
    return None


def _detect_ambiguity(
    question: str, tokens: set[str], definition: PhenotypeDefinition
) -> Clarification | None:
    """Ambiguities that change the answer and that the question does not settle."""
    lowered = question.lower()
    mentions_severity = bool(tokens & _SEVERITY_WORDS)
    mentions_seriousness = any(word in lowered for word in _SERIOUSNESS_WORDS)

    if mentions_severity and mentions_seriousness:
        return Clarification(
            question=question,
            ambiguity=(
                "The question uses both severity language and seriousness "
                "language, and they are different fields."
            ),
            effect=(
                "Severity is the intensity of the event; seriousness is a "
                "regulatory category defined by outcome. A mild event can be "
                "serious and a severe event can be non-serious, so the two "
                "filters select different subjects. Collapsing them would "
                "produce a count that answers neither question."
            ),
            options=[
                "Filter on severity (mild / moderate / severe)",
                "Filter on seriousness (death, life-threatening, "
                "hospitalisation, disability, congenital anomaly, other "
                "medically important)",
                "Report both as separate counts",
            ],
        )

    if "serious" in lowered and not mentions_seriousness:  # pragma: no cover
        return None

    if mentions_severity and not mentions_seriousness:
        # "severe hypoglycemia" is itself a term of art in diabetes trials,
        # where it means requiring third-party assistance rather than an
        # intensity grade. The definition does not encode that, so say so.
        if "severe" in lowered and "assistance" not in lowered:
            return Clarification(
                question=question,
                ambiguity=(
                    "'Severe' is underdetermined here. It can mean the "
                    "recorded intensity grade, or the diabetes-trial term of "
                    "art for an episode requiring third-party assistance."
                ),
                effect=(
                    f"{definition.key} does not filter on severity at all, so "
                    f"the word changes the cohort only if it is added as a "
                    f"filter — and the two readings select different subjects."
                ),
                options=[
                    "Intensity grade recorded as severe (AESEV / narrative grading)",
                    "Episodes requiring third-party assistance "
                    "(not encoded in this definition; needs a new version)",
                    f"Run {definition.key} as written, with no severity filter",
                ],
            )
    return None


# --------------------------------------------------------------------------
# LLM backend (optional)
# --------------------------------------------------------------------------

_LLM_SYSTEM = """You translate a clinical research question into a query specification.

You emit a single JSON object and nothing else. No prose, no markdown fence.

You must not compute any statistic, count any subject, or decide whether any
patient is a case. Those are done by validated code after a human approves your
specification.

Fields:
  definition_id      (string) one of the definition ids offered
  definition_version (integer)
  studies            (array of study ids; empty means all)
  assertion          (array from: {assertions})
  evidence_state     (array from: {states})
  window             ([min_days, max_days] or null)
  anchor             (string or null)
  notes              (array of short strings)

If the question leaves a rule underdetermined, emit instead:
  {{"clarification": {{"ambiguity": "...", "effect": "...", "options": ["..."]}}}}
"""


def compile_with_llm(
    question: str, pipeline: Pipeline, *, model: str | None = None
) -> CompileResult:
    """Compile via an LLM, then validate its output against the schema.

    The model's only job is to fill a form.  Anything it returns that does not
    validate is rejected outright rather than repaired, because a silently
    repaired spec is a spec nobody approved.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return CompileResult(
            None,
            Clarification(
                question=question,
                ambiguity="The LLM backend was requested but no API key is set.",
                effect=(
                    "No spec can be compiled by that backend. The deterministic "
                    "backend is offline and always available."
                ),
                options=["Use the deterministic backend", "Set ANTHROPIC_API_KEY"],
            ),
            trace=["ANTHROPIC_API_KEY not set"],
        )
    try:
        import anthropic
    except ImportError:
        return CompileResult(
            None,
            Clarification(
                question=question,
                ambiguity="The `anthropic` package is not installed.",
                effect="The LLM backend cannot run; the deterministic one can.",
                options=["Use the deterministic backend", "pip install anthropic"],
            ),
            trace=["anthropic package missing"],
        )

    definitions = [d for d in pipeline.definitions.all() if d.status != "draft"]
    catalogue = [
        {
            "id": d.id, "version": d.version, "label": d.label,
            "concept": d.concept.primary,
            "window": [d.window.min, d.window.max] if d.window else None,
            "anchor": d.anchor.event if d.anchor else None,
        }
        for d in definitions
    ]
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model or "claude-sonnet-5",
        max_tokens=1024,
        system=_LLM_SYSTEM.format(
            assertions=list(ASSERTION_VALUES), states=list(EVIDENCE_STATE_VALUES)
        ),
        messages=[
            {
                "role": "user",
                "content": json.dumps(
                    {"question": question, "definitions": catalogue}, indent=2
                ),
            }
        ],
    )
    raw = "".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    ).strip()

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return CompileResult(
            None,
            Clarification(
                question=question,
                ambiguity=f"The model did not return valid JSON: {exc}.",
                effect="No spec was produced; nothing was executed.",
                options=["Retry", "Use the deterministic backend"],
            ),
            trace=[f"invalid JSON from model: {raw[:200]}"],
        )

    if "clarification" in payload:
        body = payload["clarification"]
        return CompileResult(
            None,
            Clarification(
                question=question,
                ambiguity=str(body.get("ambiguity", "unspecified")),
                effect=str(body.get("effect", "unspecified")),
                options=[str(o) for o in body.get("options", [])],
            ),
            trace=["model requested clarification"],
        )

    payload.setdefault("question", question)
    payload["backend"] = "llm"
    payload.pop("concept", None)
    if isinstance(payload.get("window"), list) and len(payload["window"]) == 2:
        payload["window"] = (payload["window"][0], payload["window"][1])
    try:
        spec = PhenotypeQuerySpec.model_validate(payload)
    except Exception as exc:
        return CompileResult(
            None,
            Clarification(
                question=question,
                ambiguity=f"The model's specification failed schema validation: {exc}.",
                effect=(
                    "The spec was rejected rather than repaired. A repaired "
                    "spec is one nobody approved."
                ),
                options=["Retry", "Use the deterministic backend"],
            ),
            trace=[f"schema rejection: {exc}"],
        )

    try:
        definition = pipeline.definition(spec.definition_id, spec.definition_version)
    except Exception as exc:
        return CompileResult(
            None,
            Clarification(
                question=question,
                ambiguity=f"The model named a definition that does not exist: {exc}.",
                effect="Nothing was executed.",
                options=[d.key for d in definitions],
            ),
            trace=["model named an unknown definition"],
        )

    spec.concept = definition.concept.primary
    spec.notes = list(spec.notes) + [
        "Compiled by an LLM backend and schema-validated. The model produced "
        "this specification only; it computed nothing and named no cases.",
    ]
    return CompileResult(spec, None, definition, ["compiled by LLM backend"])


def compile_question(
    question: str,
    pipeline: Pipeline,
    *,
    backend: str = "deterministic",
    allow_draft: bool = False,
    model: str | None = None,
) -> CompileResult:
    if backend == "llm":
        return compile_with_llm(question, pipeline, model=model)
    return compile_deterministic(question, pipeline, allow_draft=allow_draft)
