"""Compiling a question into an inspectable specification.

The agent chooses *which* rule to run. It never decides what a case is, never
computes a number, and never resolves an ambiguity by picking a default — where
the question leaves a rule underdetermined it says which rule and stops.

Two backends, one contract. The deterministic one is the default and runs
offline; the LLM one emits the same specification object and is validated
exactly as strictly.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from ..models import Clarification, PhenotypeDefinition, PhenotypeQuerySpec
from ..pipeline import Pipeline

_STUDY_RE = re.compile(r"\bSTUDY-(P\d+)\b", re.IGNORECASE)
_WINDOW_RE = re.compile(
    r"\bwithin\s+(?P<n>\d+)\s+(?P<unit>day|days|week|weeks)\b", re.IGNORECASE
)

#: Words that mean one thing in a protocol and another in conversation.
#: "Severe" is an intensity grade; "serious" is a regulatory category defined by
#: outcome. Treating them as one graded scale is the most consequential mistake
#: available here, so the agent refuses to guess which was meant.
_SEVERITY_WORDS = {"severe", "severity", "mild", "moderate"}
_SERIOUSNESS_WORDS = {
    "serious", "seriousness", "hospitalised", "hospitalized", "hospitalisation",
    "life-threatening", "life threatening", "fatal", "death",
}

#: Site words that are not catalogue values. "Truncal" is a region, and which
#: values it covers is a catalogue decision the questioner may not share.
_REGION_WORDS = {"truncal", "trunk", "torso"}


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
                if self.clarification else None
            ),
            "definition": (
                self.matched_definition.key if self.matched_definition else None
            ),
            "trace": self.trace,
        }


def _tokenise(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _score(definition: PhenotypeDefinition, tokens: set[str]) -> int:
    haystack = _tokenise(
        f"{definition.id} {definition.label} {definition.description} "
        f"{definition.concept.primary}"
    )
    return len(tokens & haystack)


def compile_deterministic(
    question: str, pipeline: Pipeline, *, allow_draft: bool = False
) -> CompileResult:
    trace: list[str] = []
    tokens = _tokenise(question)
    lowered = question.lower()
    definitions = [
        d for d in pipeline.definitions.all() if d.status != "draft" or allow_draft
    ]
    if not definitions:
        return CompileResult(None, Clarification(
            question=question,
            ambiguity="No runnable phenotype definition is available.",
            effect="There is nothing to evaluate the question against.",
            options=["Add a frozen definition to config/phenotypes/"],
        ), trace=["no definitions in catalogue"])

    latest: dict[str, PhenotypeDefinition] = {}
    for definition in definitions:
        current = latest.get(definition.id)
        if current is None or definition.version > current.version:
            latest[definition.id] = definition

    scored = sorted(
        ((_score(d, tokens), d) for d in latest.values()),
        key=lambda pair: (-pair[0], pair[1].id),
    )
    trace.append(
        "definition match scores: "
        + ", ".join(f"{d.id} (latest v{d.version})={s}" for s, d in scored)
    )
    best = scored[0][0]
    if best == 0:
        return CompileResult(None, Clarification(
            question=question,
            ambiguity="The question does not name a phenotype this catalogue defines.",
            effect=(
                "Without a definition there is no rule for what makes a subject "
                "a case, and any count would be an invention."
            ),
            options=[f"{d.key}: {d.label}" for d in definitions],
        ), trace=trace)

    top = [d for s, d in scored if s == best]
    if len(top) > 1:
        return CompileResult(None, Clarification(
            question=question,
            ambiguity=(
                "The question matches more than one definition equally well: "
                f"{', '.join(d.key for d in top)}."
            ),
            effect="Each applies different rules and would produce a different cohort.",
            options=[f"{d.key}: {d.label}" for d in top],
        ), trace=trace)

    definition = top[0]
    trace.append(f"selected {definition.key} ({definition.status})")

    clarification = _ambiguity(question, tokens, lowered, definition, pipeline)
    if clarification is not None:
        return CompileResult(None, clarification, definition, trace)

    studies = sorted(
        {f"STUDY-{m.group(1).upper()}" for m in _STUDY_RE.finditer(question)}
    )
    known = set(pipeline.store.studies())
    unknown = [s for s in studies if s not in known]
    if unknown:
        return CompileResult(None, Clarification(
            question=question,
            ambiguity=f"The question names studies not in this snapshot: {unknown}.",
            effect="Counts would silently omit the studies that were asked for.",
            options=sorted(known),
        ), definition, trace)
    if studies:
        trace.append(f"restricted to studies {studies}")

    onset = definition.requirement("onset")
    window = (
        (onset.window.min, onset.window.max)
        if onset and onset.window else None
    )
    match = _WINDOW_RE.search(question)
    if match and window is not None:
        days = int(match.group("n")) * (
            7 if match.group("unit").startswith("week") else 1
        )
        if days != window[1]:
            return CompileResult(None, Clarification(
                question=question,
                ambiguity=(
                    f"The question asks for a {days}-day window, but "
                    f"{definition.key} is defined with {window[0]}-{window[1]} days."
                ),
                effect=(
                    "Changing the window changes which subjects are cases. That "
                    "is a change to the case definition, so it needs a new "
                    "definition version rather than a query parameter."
                ),
                options=[
                    f"Run {definition.key} as written ({window[0]}-{window[1]} days)",
                    f"Create v{pipeline.definitions.next_version(definition.id)} "
                    f"with max={days} and run that",
                ],
            ), definition, trace)

    verdicts = ["case"]
    if "ascertain" in lowered or "unascertainable" in lowered:
        verdicts.append("not_ascertainable")
        trace.append("question asks about ascertainability as well as cases")

    accept = sorted(
        {m for r in definition.required_attributes for m in r.accept_methods}
    )
    spec = PhenotypeQuerySpec(
        question=question,
        definition_id=definition.id,
        definition_version=definition.version,
        studies=studies or sorted(known),
        concept=definition.concept.primary,
        window=window,
        anchor=definition.anchor.event if definition.anchor else None,
        verdicts=verdicts,  # type: ignore[arg-type]
        accept_methods=accept,  # type: ignore[arg-type]
        retrieval_mode="precise",
        top_k=20,
        notes=[
            f"Compiled deterministically against {definition.key}, status "
            f"{definition.status}.",
            "Counts follow the definition as written; the agent applied no "
            "default of its own.",
            f"The definition accepts evidence by {accept}; every number says "
            f"which route supplied it.",
            "Episodes that cannot be ascertained are reported separately and "
            "are neither cases nor negatives.",
        ],
        backend="deterministic",
    )
    return CompileResult(spec, None, definition, trace)


def _ambiguity(
    question: str, tokens: set[str], lowered: str,
    definition: PhenotypeDefinition, pipeline: Pipeline,
) -> Clarification | None:
    """Ambiguities that change the answer and that the question does not settle."""
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
                "serious and a severe event non-serious, so the two filters "
                "select different subjects and one count answers neither."
            ),
            options=[
                "Filter on severity (mild / moderate / severe)",
                "Filter on seriousness (death, life-threatening, "
                "hospitalisation, disability, congenital anomaly, other "
                "medically important)",
                "Report both as separate counts",
            ],
        )

    location = definition.requirement("location")
    if location and location.allowed and (tokens & _REGION_WORDS):
        catalogue = pipeline.catalog.attribute("location")
        declared = sorted(set(location.allowed))
        trunk = sorted(catalogue.regions.get("trunk", ()))
        if declared != trunk:
            return Clarification(
                question=question,
                ambiguity=(
                    f"'Truncal' is a region, and this definition names specific "
                    f"sites: {declared}. The catalogue's trunk region is {trunk}."
                ),
                effect=(
                    "Which sites count is part of the case definition, not of "
                    "the query, and the two lists select different subjects."
                ),
                options=[
                    f"Run {definition.key} as written ({declared})",
                    f"Create v{pipeline.definitions.next_version(definition.id)} "
                    f"covering the catalogue's trunk region ({trunk})",
                ],
            )
    return None


# --------------------------------------------------------------------------
# LLM backend (optional)
# --------------------------------------------------------------------------

_LLM_SYSTEM = """You translate a clinical research question into a query specification.

Emit a single JSON object and nothing else. No prose, no markdown fence.

You must not compute any statistic, count any subject, or decide whether any
patient is a case. Those are done by code that was written before the question
was asked.

Fields:
  definition_id      (string) one of the definition ids offered
  definition_version (integer)
  studies            (array of study ids; empty means all)
  verdicts           (array from: case, not_case, not_ascertainable, review)
  notes              (array of short strings)

If the question leaves a rule underdetermined, emit instead:
  {{"clarification": {{"ambiguity": "...", "effect": "...", "options": ["..."]}}}}
"""


def compile_with_llm(
    question: str, pipeline: Pipeline, *, allow_draft: bool = False
) -> CompileResult:  # pragma: no cover - requires credentials
    if not os.environ.get("ANTHROPIC_API_KEY"):
        result = compile_deterministic(question, pipeline, allow_draft=allow_draft)
        result.trace.append(
            "an LLM backend was requested but no credentials are present; the "
            "question was compiled deterministically instead"
        )
        return result
    try:
        import anthropic

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=os.environ.get("AELAYER_MODEL", "claude-sonnet-5"),
            max_tokens=800,
            system=_LLM_SYSTEM,
            messages=[{"role": "user", "content": question}],
        )
        payload = json.loads(response.content[0].text)
    except Exception as exc:
        result = compile_deterministic(question, pipeline, allow_draft=allow_draft)
        result.trace.append(f"the LLM backend failed ({exc}); compiled deterministically")
        return result

    if "clarification" in payload:
        body = payload["clarification"]
        return CompileResult(None, Clarification(question=question, **body))
    spec = PhenotypeQuerySpec.model_validate({**payload, "question": question,
                                              "backend": "llm"})
    definition = pipeline.definition(spec.definition_id, spec.definition_version)
    return CompileResult(spec, None, definition, ["compiled by the LLM backend"])


def compile_question(
    question: str, pipeline: Pipeline, *, backend: str = "deterministic",
    allow_draft: bool = False,
) -> CompileResult:
    if backend == "llm":
        return compile_with_llm(question, pipeline, allow_draft=allow_draft)
    return compile_deterministic(question, pipeline, allow_draft=allow_draft)
