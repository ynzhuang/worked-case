"""The value ablation: does reading narrative text actually buy anything?

This is the experiment that can falsify the proposal, and it is built as a
first-class command rather than a chart in a report, because a proposal whose
falsification test is optional is not a proposal.

Three cumulative stages, each a distinct engineering investment:

1. **structured** — coded concepts, structured qualifiers, linked forms, and
   the cross-domain derivation. Everything deterministic. No model anywhere.
2. **+ reported term** — the model path reads the investigator's own words on
   the AE record itself. One short field per record: the cheap version.
3. **+ comments** — the model path also reads linked comment records. More
   documents, more plumbing, more surface area for error.

Each stage is scored against the generator's gold verdicts, and — this is the
part that matters — the increments are counted in **correctly** ascertained
cases. A stage that finds forty more cases of which thirty are wrong has made
the cohort worse while making the headline number bigger. "Correctly" is doing
the work in that sentence, so the report states precision on the *added* cases
separately from precision overall.

The output states a decision, not just numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from typing import Any, Iterable, Sequence

from .catalog import Configs, ExtractionConfig
from .extract import ExtractionEngine
from .ingest import TrialStore
from .models import CaseAssignment, PhenotypeDefinition
from .normalize import normalize_store
from .normalize.records import RecordNormalizer
from .phenotype import evaluate_definition

#: The stages, in the order they are added. Each entry names the text sources
#: the model path may read at that stage; stage one reads none.
STAGES: tuple[tuple[str, str, tuple[str, ...], str], ...] = (
    (
        "structured",
        "Structured fields only",
        (),
        "coded concepts, structured qualifiers, linked forms and the "
        "cross-domain onset derivation. No model is run at all.",
    ),
    (
        "reported_term",
        "+ the investigator's own words",
        ("reported_term",),
        "the model path reads AETERM where the structured route left the "
        "modifier unresolved.",
    ),
    (
        "comments",
        "+ linked comment records",
        ("reported_term", "comment"),
        "the model path also reads comment records linked to the AE row.",
    ),
)

#: What counts as a material gain. Declared here, before the numbers are seen,
#: so the decision cannot be talked into existence afterwards.
MATERIALITY = {
    # An absolute floor: five extra correct cases is noise on a corpus this
    # size, whatever the percentage says.
    "min_added_correct_cases": 10,
    # A relative floor, against the stage below.
    "min_relative_gain": 0.15,
    # And the added cases have to be right. A stage that buys volume by
    # guessing has not bought anything.
    "min_precision_on_added": 0.80,
}

DECISION_NOTE = (
    "The decision below is about correctly ascertained cases, not about "
    "extraction accuracy. An extractor can score well on its own metrics and "
    "still change no verdict, because the modifier it recovers was already "
    "recorded structurally, or the record fails another criterion anyway. "
    "Those numbers answer different questions and are reported separately."
)


@dataclass
class StageResult:
    """One stage: what it found, and how much of that was right."""

    stage_id: str
    label: str
    readable_sources: tuple[str, ...]
    description: str
    assignments: list[CaseAssignment] = _dc_field(default_factory=list)
    gold_field: str = "verdict_stage_comments"
    gold: dict[str, str] = _dc_field(default_factory=dict)

    # -- populations --------------------------------------------------------

    @property
    def cases(self) -> set[str]:
        return {a.record_id for a in self.assignments if a.verdict == "case"}

    @property
    def ascertained(self) -> set[str]:
        return {a.record_id for a in self.assignments if a.ascertained}

    def correct_cases(self) -> set[str]:
        """Cases this stage called that the gold verdict agrees are cases."""
        return {r for r in self.cases if self.gold.get(r) == "case"}

    def false_cases(self) -> set[str]:
        return {r for r in self.cases if self.gold.get(r) not in (None, "case")}

    def metrics(self) -> dict[str, Any]:
        gold_cases = {r for r, v in self.gold.items() if v == "case"}
        scoped = {a.record_id for a in self.assignments}
        gold_cases &= scoped
        correct = self.correct_cases()
        return {
            "stage": self.stage_id,
            "label": self.label,
            "readable_sources": list(self.readable_sources),
            "n_evaluated": len(self.assignments),
            "n_ascertained": len(self.ascertained),
            "ascertainable_fraction": (
                round(len(self.ascertained) / len(self.assignments), 4)
                if self.assignments else 0.0
            ),
            "n_case": len(self.cases),
            "n_case_correct": len(correct),
            "n_case_incorrect": len(self.false_cases()),
            "precision": (
                round(len(correct) / len(self.cases), 4) if self.cases else 0.0
            ),
            "recall": (
                round(len(correct) / len(gold_cases), 4) if gold_cases else 0.0
            ),
        }


@dataclass
class Increment:
    """What one stage added over the stage below it, and whether it mattered."""

    from_stage: str
    to_stage: str
    added_cases: int
    added_correct: int
    added_incorrect: int
    precision_on_added: float
    relative_gain: float
    added_ascertained: int
    material: bool
    decision: str
    reasons: list[str] = _dc_field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_stage": self.from_stage,
            "to_stage": self.to_stage,
            "added_cases": self.added_cases,
            "added_correct_cases": self.added_correct,
            "added_incorrect_cases": self.added_incorrect,
            "precision_on_added": self.precision_on_added,
            "relative_gain_in_correct_cases": self.relative_gain,
            "added_ascertained_records": self.added_ascertained,
            "material": self.material,
            "decision": self.decision,
            "reasons": list(self.reasons),
        }


@dataclass
class AblationReport:
    definition: PhenotypeDefinition
    stages: list[StageResult] = _dc_field(default_factory=list)
    criteria: dict[str, Any] = _dc_field(default_factory=lambda: dict(MATERIALITY))

    def increments(self) -> list[Increment]:
        return [
            self._increment(self.stages[i - 1], self.stages[i])
            for i in range(1, len(self.stages))
        ]

    def _increment(self, lower: StageResult, upper: StageResult) -> Increment:
        added = upper.cases - lower.cases
        added_correct = {r for r in added if upper.gold.get(r) == "case"}
        added_incorrect = added - added_correct
        base_correct = len(lower.correct_cases())
        precision = round(len(added_correct) / len(added), 4) if added else 0.0
        relative = (
            round(len(added_correct) / base_correct, 4) if base_correct else
            (float("inf") if added_correct else 0.0)
        )

        reasons: list[str] = []
        enough_absolute = len(added_correct) >= self.criteria["min_added_correct_cases"]
        enough_relative = relative >= self.criteria["min_relative_gain"]
        clean_enough = (
            precision >= self.criteria["min_precision_on_added"] if added else False
        )
        reasons.append(
            f"{len(added_correct)} correctly ascertained cases added "
            f"({'>=' if enough_absolute else '<'} the declared floor of "
            f"{self.criteria['min_added_correct_cases']})"
        )
        reasons.append(
            f"a {relative:.1%} relative gain over the {base_correct} correct "
            f"cases the previous stage found "
            f"({'>=' if enough_relative else '<'} the declared floor of "
            f"{self.criteria['min_relative_gain']:.0%})"
        )
        reasons.append(
            f"precision on the added cases is {precision:.1%} "
            f"({'>=' if clean_enough else '<'} the declared floor of "
            f"{self.criteria['min_precision_on_added']:.0%}); "
            f"{len(added_incorrect)} of the added cases are wrong"
        )
        material = enough_absolute and enough_relative and clean_enough

        if material:
            decision = (
                f"ADOPT. Stage {upper.stage_id!r} is worth building: it adds "
                f"{len(added_correct)} correctly ascertained cases the previous "
                f"stage could not reach, at {precision:.0%} precision on those "
                f"additions."
            )
        elif not added:
            decision = (
                f"DO NOT ADOPT. Stage {upper.stage_id!r} changed no verdict at "
                f"all. Whatever it recovered was already recorded structurally, "
                f"or the records fail another criterion regardless."
            )
        elif not clean_enough:
            decision = (
                f"DO NOT ADOPT AS IT STANDS. Stage {upper.stage_id!r} adds "
                f"{len(added)} cases, but only {len(added_correct)} of them are "
                f"right ({precision:.0%}). Volume bought by guessing is a worse "
                f"cohort, not a bigger one."
            )
        else:
            decision = (
                f"DO NOT ADOPT. Stage {upper.stage_id!r} adds only "
                f"{len(added_correct)} correctly ascertained cases "
                f"({relative:.1%} over the previous stage), below the gain "
                f"declared worth the engineering and review cost."
            )
        return Increment(
            from_stage=lower.stage_id,
            to_stage=upper.stage_id,
            added_cases=len(added),
            added_correct=len(added_correct),
            added_incorrect=len(added_incorrect),
            precision_on_added=precision,
            relative_gain=relative,
            added_ascertained=len(upper.ascertained - lower.ascertained),
            material=material,
            decision=decision,
            reasons=reasons,
        )

    def headline(self) -> str:
        """The one sentence a reader takes away.

        Anchored on the second stage, because that is the one the proposal
        rests on: everything before it is deterministic plumbing nobody
        disputes, and everything after it is a larger version of the same bet.
        """
        increments = self.increments()
        if not increments:
            return "No increment to judge: the ablation ran a single stage."
        first = increments[0]
        return first.decision

    def to_dict(self) -> dict[str, Any]:
        return {
            "definition": self.definition.key,
            "definition_hash": self.definition.definition_hash,
            "materiality_criteria": self.criteria,
            "note": DECISION_NOTE,
            "stages": [s.metrics() for s in self.stages],
            "increments": [i.to_dict() for i in self.increments()],
            "decision": self.headline(),
        }


# --------------------------------------------------------------------------
# Running it
# --------------------------------------------------------------------------


def _configs_reading(configs: Configs, sources: Sequence[str]) -> Configs:
    """The same configuration with the model path restricted to ``sources``.

    A copy, never a mutation: the stages run in one process and a stage that
    edited shared config would silently contaminate the next one.
    """
    raw = {**configs.extraction.raw, "readable_sources": list(sources)}
    restricted = ExtractionConfig(raw, configs.extraction.source_path)
    return Configs(
        catalog=configs.catalog,
        extraction=restricted,
        profiles=configs.profiles,
        extractor_version=configs.extractor_version,
        normalizer_version=configs.normalizer_version,
    )


def run_ablation(
    definition: PhenotypeDefinition,
    store: TrialStore,
    configs: Configs,
    backend_preference: str = "auto",
    gold_fields: dict[str, str] | None = None,
) -> AblationReport:
    """Run every stage over the same snapshot and the same frozen definition."""
    gold_by_stage = gold_fields or {
        "structured": "verdict_stage_structured",
        "reported_term": "verdict_stage_reported_term",
        "comments": "verdict_stage_comments",
    }
    gold_rows = store.gold()
    report = AblationReport(definition=definition)

    for stage_id, label, sources, description in STAGES:
        staged = _configs_reading(configs, sources)
        records = normalize_store(store, staged)
        if sources:
            records = ExtractionEngine.build(
                staged, store, backend_preference
            ).enrich_all(records)

        normalizer = RecordNormalizer(staged, store)
        totals: dict[str, float] = {}
        if definition.cumulative_exposure is not None:
            for record in records:
                if record.onset.observed:
                    total, _why = normalizer.cumulative_exposure(
                        record.subject_id, record.onset.value
                    )
                    if total is not None:
                        totals[record.record_id] = total

        result = evaluate_definition(definition, records, configs.catalog, totals)
        field = gold_by_stage[stage_id]
        report.stages.append(StageResult(
            stage_id=stage_id,
            label=label,
            readable_sources=tuple(sources),
            description=description,
            assignments=result.assignments,
            gold_field=field,
            gold={
                f"{row['study_id']}:{row['source_record_id']}": row[field]
                for row in gold_rows if field in row
            },
        ))
    return report


def format_ablation(report: AblationReport) -> str:
    """The ablation as text, ending in a decision rather than a table."""
    lines: list[str] = []
    lines.append(f"Value ablation — {report.definition.key}")
    lines.append(f"  definition {report.definition.definition_hash[:16]}")
    lines.append("")
    header = (
        f"  {'stage':16s} {'eval':>5} {'asc':>5} {'asc.f':>6} {'case':>5} "
        f"{'ok':>4} {'bad':>4} {'prec':>6} {'recall':>7}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for metrics in (s.metrics() for s in report.stages):
        lines.append(
            f"  {metrics['stage']:16s} {metrics['n_evaluated']:5d} "
            f"{metrics['n_ascertained']:5d} "
            f"{metrics['ascertainable_fraction']:6.3f} {metrics['n_case']:5d} "
            f"{metrics['n_case_correct']:4d} {metrics['n_case_incorrect']:4d} "
            f"{metrics['precision']:6.3f} {metrics['recall']:7.3f}"
        )
    lines.append("")
    for stage in report.stages:
        lines.append(f"  {stage.stage_id}: {stage.description}")
    lines.append("")
    for increment in report.increments():
        lines.append(f"  {increment.from_stage} -> {increment.to_stage}")
        for reason in increment.reasons:
            lines.append(f"    - {reason}")
        lines.append(f"    {increment.decision}")
        lines.append("")
    lines.append("  " + DECISION_NOTE)
    lines.append("")
    lines.append(f"  DECISION: {report.headline()}")
    return "\n".join(lines)
