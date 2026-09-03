"""Compiling a question into an inspectable specification.

The agent chooses *which* rule to run. It never decides what a case is, never
computes a number, and never resolves an ambiguity by picking a default.

Two things follow, and both are the point of this module:

**It binds a definition version.** The compiled specification names an id, a
version and a content hash. It carries no window, no threshold and no assertion
of its own, because those live in the frozen file. A question that implies
different parameters does not become a query option — it becomes a conflict,
and the options offered are "run the definition as written" or "write a new
version", never "override it for this run".

**It reports a conflict rather than a default.** Where a question leaves a rule
underdetermined, the compiler says which rule, why the choice changes the
answer, and what the alternatives are. Producing a number under a silently
chosen reading is the failure this exists to prevent.

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

from ..models import Conflict, PhenotypeDefinition, QuerySpec
from ..pipeline import Pipeline

_STUDY_RE = re.compile(r"\bSTUDY-([A-Z0-9]+)\b", re.IGNORECASE)
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

#: Phrasings that could mean "the source recorded an absence" or "the source
#: does not say they had it". Those are different populations and the second one
#: is not a population at all, so the compiler refuses to pick.
_NEGATIVE_PHRASES = (
    "without", "did not have", "didn't have", "no mucosal", "negative for",
    "absence of", "free of", "not involving",
)


@dataclass
class CompileResult:
    spec: QuerySpec | None
    conflict: Conflict | None
    matched_definition: PhenotypeDefinition | None = None
    trace: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.trace = self.trace or []

    @property
    def needs_clarification(self) -> bool:
        return self.conflict is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "needs_clarification": self.needs_clarification,
            "spec": self.spec.model_dump(mode="json") if self.spec else None,
            "conflict": (
                self.conflict.model_dump(mode="json") if self.conflict else None
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
        + " ".join(definition.concept_set.include)
        + " " + " ".join(m.name for m in definition.modifiers)
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
        return CompileResult(None, Conflict(
            question=question,
            conflict="No runnable phenotype definition is available.",
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
        return CompileResult(None, Conflict(
            question=question,
            conflict=(
                "The question does not name a phenotype this catalogue defines."
            ),
            effect=(
                "Without a definition there is no rule for what makes a record "
                "a case, and any count would be an invention."
            ),
            options=[f"{d.key}: {d.label}" for d in definitions],
        ), trace=trace)

    top = [d for s, d in scored if s == best]
    if len(top) > 1:
        return CompileResult(None, Conflict(
            question=question,
            conflict=(
                "The question matches more than one definition equally well: "
                f"{', '.join(d.key for d in top)}."
            ),
            effect=(
                "Each applies different rules and would produce a different "
                "cohort."
            ),
            options=[f"{d.key}: {d.label}" for d in top],
        ), trace=trace)

    definition = top[0]
    trace.append(
        f"bound {definition.key} ({definition.status}, "
        f"{definition.definition_hash[:12]})"
    )

    conflict = _conflict(question, tokens, lowered, definition, pipeline)
    if conflict is not None:
        return CompileResult(None, conflict, definition, trace)

    studies = sorted(
        {f"STUDY-{m.group(1).upper()}" for m in _STUDY_RE.finditer(question)}
    )
    known = set(pipeline.store.studies())
    unknown = [s for s in studies if s not in known]
    if unknown:
        return CompileResult(None, Conflict(
            question=question,
            conflict=f"The question names studies not in this snapshot: {unknown}.",
            bound_definition=definition.key,
            effect="Counts would silently omit the studies that were asked for.",
            options=sorted(known),
        ), definition, trace)
    if studies:
        trace.append(f"restricted to studies {studies}")

    verdicts = ["case"]
    if "ascertain" in lowered or "denominator" in lowered:
        verdicts.append("not_ascertainable")
        trace.append("question asks about ascertainability as well as cases")
    if "rate" in lowered or "incidence" in lowered or "denominator" in lowered:
        if "non_case" not in verdicts:
            verdicts.append("non_case")
        trace.append(
            "question asks for a rate, so non_case records are in scope: "
            "a rate needs a denominator and a denominator needs evaluated "
            "negatives"
        )

    accept = definition.accepted_methods()
    spec = QuerySpec(
        question=question,
        definition_id=definition.id,
        definition_version=definition.version,
        definition_hash=definition.definition_hash,
        studies=studies or sorted(known),
        verdicts=verdicts,  # type: ignore[arg-type]
        accept_methods=accept,  # type: ignore[arg-type]
        notes=[
            f"Bound to {definition.key} ({definition.definition_hash[:12]}), "
            f"status {definition.status}. Its parameters come from the frozen "
            f"file, not from the question.",
            "Counts follow the definition as written; the agent applied no "
            "default of its own.",
            f"The definition accepts evidence by {accept}; every number says "
            f"which route supplied it.",
            "Records that cannot be ascertained are reported separately and "
            "are neither cases nor negatives; the ascertainable fraction is "
            "reported beside every rate.",
        ],
        backend="deterministic",
    )
    return CompileResult(spec, None, definition, trace)


def _conflict(
    question: str, tokens: set[str], lowered: str,
    definition: PhenotypeDefinition, pipeline: Pipeline,
) -> Conflict | None:
    """Conflicts that change the answer and that the question does not settle."""
    if bool(tokens & _SEVERITY_WORDS) and any(
        word in lowered for word in _SERIOUSNESS_WORDS
    ):
        return Conflict(
            question=question,
            conflict=(
                "The question uses both severity language and seriousness "
                "language, and they are different fields."
            ),
            bound_definition=definition.key,
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

    # The conflict this whole system is built around.
    if any(phrase in lowered for phrase in _NEGATIVE_PHRASES) and definition.modifiers:
        modifier = definition.modifiers[0].name
        return Conflict(
            question=question,
            conflict=(
                f"The question asks about subjects who did not have "
                f"{modifier}, and that phrase has two readings this system "
                f"keeps deliberately apart."
            ),
            bound_definition=definition.key,
            effect=(
                "A documented negative means somebody examined the patient and "
                "recorded an absence; that subject is a non_case and belongs in "
                "the denominator. Silence means nobody recorded anything, and "
                "that subject belongs in neither the numerator nor the "
                "denominator. The two readings give different rates, and one of "
                "them is not a rate at all."
            ),
            options=[
                f"Subjects with an observed absence of {modifier} "
                f"(assertion=absent) — evaluated negatives",
                f"Subjects for whom {modifier} was never recorded "
                f"(availability=not_collected or unresolved) — not "
                f"ascertainable, reported separately",
                "Both, as separate counts with the ascertainable fraction",
            ],
        )

    match = _WINDOW_RE.search(question)
    if match and definition.temporal is not None:
        days = int(match.group("n")) * (
            7 if match.group("unit").startswith("week") else 1
        )
        window = (definition.temporal.minimum, definition.temporal.maximum)
        if days != window[1]:
            return Conflict(
                question=question,
                conflict=(
                    f"The question asks for a {days}-day window, but "
                    f"{definition.key} is defined with {window[0]}-{window[1]} "
                    f"days."
                ),
                bound_definition=definition.key,
                effect=(
                    "Changing the window changes which subjects are cases. That "
                    "is a change to the case definition, so it needs a new "
                    "definition version rather than a query parameter — and the "
                    "agent will not override a bound version to accommodate a "
                    "question."
                ),
                options=[
                    f"Run {definition.key} as written ({window[0]}-{window[1]} days)",
                    f"Create v{pipeline.definitions.next_version(definition.id)} "
                    f"with max={days} and run that",
                ],
            )

    grade_match = re.search(r"\bgrade\s*(?:>=|≥|of at least\s*)?(\d)\b", lowered)
    if grade_match and definition.grade is not None:
        asked = int(grade_match.group(1))
        if asked != definition.grade.minimum:
            return Conflict(
                question=question,
                conflict=(
                    f"The question asks for grade {asked}, but "
                    f"{definition.key} is defined at a minimum of "
                    f"{definition.grade.minimum}."
                ),
                bound_definition=definition.key,
                effect=(
                    "A grade threshold is part of what makes a case, not a "
                    "filter applied afterwards."
                ),
                options=[
                    f"Run {definition.key} as written (grade >= "
                    f"{definition.grade.minimum})",
                    f"Create v{pipeline.definitions.next_version(definition.id)} "
                    f"with minimum={asked} and run that",
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

You select an existing definition version. You never invent phenotype
parameters: no windows, no thresholds, no assertions. If the question implies
parameters different from the definition's, emit a conflict instead.

Fields:
  definition_id      (string) one of the definition ids offered
  definition_version (integer)
  studies            (array of study ids; empty means all)
  verdicts           (array from: case, non_case, not_ascertainable, review)
  notes              (array of short strings)

If the question leaves a rule underdetermined, emit instead:
  {{"conflict": {{"conflict": "...", "effect": "...", "options": ["..."]}}}}
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
        result.trace.append(
            f"the LLM backend failed ({exc}); compiled deterministically"
        )
        return result

    if "conflict" in payload:
        return CompileResult(
            None, Conflict(question=question, **payload["conflict"])
        )
    definition = pipeline.definition(
        payload["definition_id"], payload.get("definition_version")
    )
    # The version the backend named is looked up and its hash stamped here, so
    # what runs is the file on disk rather than whatever the model described.
    spec = QuerySpec.model_validate({
        **payload,
        "question": question,
        "definition_version": definition.version,
        "definition_hash": definition.definition_hash,
        "accept_methods": definition.accepted_methods(),
        "backend": "llm",
    })
    return CompileResult(spec, None, definition, ["compiled by the LLM backend"])


def compile_question(
    question: str, pipeline: Pipeline, *, backend: str = "deterministic",
    allow_draft: bool = False,
) -> CompileResult:
    if backend == "llm":
        return compile_with_llm(question, pipeline, allow_draft=allow_draft)
    return compile_deterministic(question, pipeline, allow_draft=allow_draft)
