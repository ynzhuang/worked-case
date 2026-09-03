"""Executing a phenotype definition against normalized records.

Four verdicts, and the fourth is the one that makes the other three honest:

``case``
    every criterion satisfied
``non_case``
    a criterion was *evaluated* and failed — somebody looked and the answer was
    no
``review``
    the evidence exists but does not settle the question (the source hedges, or
    the confidence is below the declared threshold)
``not_ascertainable``
    the evidence needed to decide was never collected in a form this study can
    answer from

Without an explicit ``non_case`` you cannot state a denominator: a study that
never asks about the modifier and a study that asks and records "no" would land
in the same bucket, and the resulting rate would compare CRFs rather than
patients. The evaluator therefore never turns silence into a negative, and
never turns a negative into silence.

The rule is route-agnostic: it names a modifier and the assertion it wants, and
it does not know or care which study variable supplied the evidence. That is
what makes one definition executable across studies that collect the same fact
in five different places.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field as _dc_field
from typing import Any, Iterable, Sequence

from ..catalog import ConceptCatalog
from ..models import (
    ASCERTAINED,
    Attribute,
    CanonicalAERecord,
    CaseAssignment,
    CriterionFinding,
    Denominator,
    PhenotypeDefinition,
    Span,
)


@dataclass
class EvaluationResult:
    """Every assignment, plus the denominators they imply."""

    definition: PhenotypeDefinition
    assignments: list[CaseAssignment] = _dc_field(default_factory=list)
    notes: list[str] = _dc_field(default_factory=list)

    # -- summaries ----------------------------------------------------------

    def verdicts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for assignment in self.assignments:
            counts[assignment.verdict] = counts.get(assignment.verdict, 0) + 1
        return dict(sorted(counts.items()))

    def denominators(self) -> list[Denominator]:
        """One per study, because the ascertainable fraction is a study fact."""
        buckets: dict[str, Denominator] = {}
        for assignment in self.assignments:
            bucket = buckets.get(assignment.study_id)
            if bucket is None:
                bucket = Denominator(
                    study_id=assignment.study_id, profile=assignment.profile
                )
                buckets[assignment.study_id] = bucket
            bucket.n_total += 1
            setattr(
                bucket, f"n_{assignment.verdict}",
                getattr(bucket, f"n_{assignment.verdict}") + 1,
            )
        return [buckets[k] for k in sorted(buckets)]

    def overall(self) -> Denominator:
        total = Denominator(study_id="ALL", profile="")
        for bucket in self.denominators():
            total.n_total += bucket.n_total
            total.n_case += bucket.n_case
            total.n_non_case += bucket.n_non_case
            total.n_review += bucket.n_review
            total.n_not_ascertainable += bucket.n_not_ascertainable
        return total

    def cases(self) -> list[CaseAssignment]:
        return [a for a in self.assignments if a.verdict == "case"]

    def ascertained(self) -> list[CaseAssignment]:
        return [a for a in self.assignments if a.verdict in ASCERTAINED]

    def by_record(self) -> dict[str, CaseAssignment]:
        return {a.record_id: a for a in self.assignments}

    def text_dependent_cases(self) -> list[CaseAssignment]:
        """Cases that exist only because the model path read prose.

        This is the number the value ablation turns into a decision.
        """
        return [a for a in self.cases() if a.used_text_extraction]


class PhenotypeEvaluator:
    """Runs one definition. Holds no state between records."""

    def __init__(
        self, definition: PhenotypeDefinition,
        catalog: ConceptCatalog | None = None,
        exposure_totals: dict[str, float] | None = None,
    ):
        self.definition = definition
        self.catalog = catalog
        #: `subject_id -> cumulative dose before onset`, supplied by the
        #: normalizer for definitions that need it.
        self.exposure_totals = exposure_totals or {}

    # -- one record ---------------------------------------------------------

    def evaluate(self, record: CanonicalAERecord) -> CaseAssignment | None:
        """One assignment, or ``None`` when the record is out of scope.

        A record whose coded concept is not in the set is not a `non_case`: it
        was never a candidate, and counting it as an evaluated negative would
        inflate every denominator with unrelated events.
        """
        if not self._in_concept_set(record):
            return None

        findings = [
            *self._concept_finding(record),
            *(self._modifier_finding(record, r) for r in self.definition.modifiers),
        ]
        if self.definition.temporal is not None:
            findings.append(self._temporal_finding(record))
        if self.definition.grade is not None:
            findings.append(self._grade_finding(record))
        if self.definition.cumulative_exposure is not None:
            findings.append(self._exposure_finding(record))

        verdict, deciding, reason = self._decide(findings)
        return CaseAssignment(
            record_id=record.record_id,
            subject_id=record.subject_id,
            study_id=record.study_id,
            profile=record.profile,
            verdict=verdict,
            deciding_criterion=deciding,
            reason=reason,
            findings=findings,
            evidence_spans=[s for f in findings for s in f.spans],
            attribute_sources={
                f.name: f.source_variable for f in findings if f.source_variable
            },
            attribute_methods={
                f.name: f.method for f in findings if f.method
            },
            definition_id=self.definition.id,
            definition_version=self.definition.version,
            definition_hash=self.definition.definition_hash,
        )

    # -- criteria -----------------------------------------------------------

    def _in_concept_set(self, record: CanonicalAERecord) -> bool:
        concept = record.concept_id
        if concept is None:
            return False
        include = set(self.definition.concept_set.include)
        if self.catalog is not None:
            expanded: set[str] = set()
            for name in include:
                if name in self.catalog.concept_groups:
                    expanded.update(self.catalog.expand_group(name))
                else:
                    expanded.add(name)
            include = expanded
        return concept in include and concept not in set(
            self.definition.concept_set.exclude
        )

    def _concept_finding(self, record: CanonicalAERecord) -> list[CriterionFinding]:
        """The coded event, and how its dictionary version reconciled.

        Membership is decided by concept, not by string, so a record whose code
        has no mechanical mapping to the target version still qualifies — and
        the finding says so rather than dropping it.
        """
        coded = record.coded_event
        if coded is None:
            return []
        note = ""
        if coded.reconciliation == "flagged_for_review":
            note = (
                f"; {coded.code!r} has no mechanical mapping to "
                f"{self.definition.concept_set.dictionary_target}, so it is "
                f"flagged for review rather than recoded — the record still "
                f"qualifies by concept"
            )
        elif coded.reconciliation == "remapped_mechanically":
            note = (
                f"; {coded.code!r} ({coded.dictionary_version}) remaps "
                f"mechanically to {coded.reconciled_to!r}, and the original is "
                f"preserved"
            )
        return [CriterionFinding(
            name="concept",
            satisfied=True,
            verdict="case",
            assertion="present",
            availability="observed",
            value=coded.concept_id,
            method="direct",
            source="structured_standard",
            source_variable="AEDECOD",
            reason=(
                f"coded {coded.code!r} under {coded.dictionary_version}, "
                f"concept {coded.concept_id}, which the concept set includes"
                f"{note}"
            ),
            spans=[Span(
                doc_id=f"AE:{record.source_record_id}:AEDECOD", start=0,
                end=len(coded.code), field="concept",
                extracted_value=coded.concept_id or coded.code,
                text=coded.code, kind="structured",
            )],
        )]

    def _modifier_finding(
        self, record: CanonicalAERecord, requirement
    ) -> CriterionFinding:
        """The heart of it: assertion and availability read as separate facts."""
        attribute: Attribute[Any] | None = record.modifiers.get(requirement.name)
        policy = self.definition.ascertainability
        base = dict(
            name=requirement.name,
            source_variable=(attribute.source_variable if attribute else None),
            source=(attribute.source if attribute else None),
            method=(attribute.method if attribute else None),
            confidence=(attribute.confidence if attribute else None),
            spans=list(attribute.evidence) if attribute else [],
            value=(attribute.value if attribute else None),
            assertion=(attribute.assertion if attribute else None),
            availability=(attribute.availability if attribute else "unresolved"),
        )

        # 1 · nothing was said. Not a negative — nobody looked.
        if attribute is None or not attribute.observed:
            availability = attribute.availability if attribute else "unresolved"
            return CriterionFinding(**{**base, **dict(
                satisfied=False,
                verdict=requirement.on_unavailable,
                reason=(
                    f"{requirement.name} is {availability}: the source says "
                    f"nothing, which is not the same as saying no. This record "
                    f"is {requirement.on_unavailable}, not a non_case"
                    + (f" ({attribute.note})" if attribute and attribute.note else "")
                ),
            )})

        # 2 · the route it arrived by is not one this definition accepts.
        if attribute.method not in requirement.accept_methods:
            return CriterionFinding(**{**base, **dict(
                satisfied=False,
                verdict=policy.missing_required_modifier,
                reason=(
                    f"{requirement.name} was read by method "
                    f"{attribute.method!r}, which this definition does not "
                    f"accept ({requirement.accept_methods}); the evidence "
                    f"exists but this version declines to use it"
                ),
            )})
        if (
            requirement.accept_sources is not None
            and attribute.source not in requirement.accept_sources
        ):
            return CriterionFinding(**{**base, **dict(
                satisfied=False,
                verdict=policy.missing_required_modifier,
                reason=(
                    f"{requirement.name} came from {attribute.source!r}, which "
                    f"this definition does not accept "
                    f"({requirement.accept_sources})"
                ),
            )})

        # 3 · an extracted value with no span cannot be checked by anyone.
        evidence = self.definition.evidence_policy
        if (
            attribute.method == "extracted"
            and evidence.extracted_requires_span
            and not attribute.evidence
        ):
            return CriterionFinding(**{**base, **dict(
                satisfied=False,
                verdict="review",
                reason=(
                    f"{requirement.name} was extracted but carries no span, so "
                    f"no reader can check it"
                ),
            )})
        if (
            attribute.method == "extracted"
            and attribute.confidence is not None
            and attribute.confidence < evidence.min_confidence
        ):
            return CriterionFinding(**{**base, **dict(
                satisfied=False,
                verdict=evidence.below_threshold,
                reason=(
                    f"{requirement.name} was extracted at confidence "
                    f"{attribute.confidence:.2f}, below the declared threshold "
                    f"{evidence.min_confidence:.2f}"
                ),
            )})

        # 4 · the source addressed it. Now, what did it say?
        if attribute.assertion == requirement.require_assertion:
            return CriterionFinding(**{**base, **dict(
                satisfied=True,
                verdict="case",
                reason=(
                    f"{requirement.name} is {attribute.assertion} "
                    f"{'(' + str(attribute.value) + ') ' if attribute.value else ''}"
                    f"via {attribute.method} from {attribute.source_variable}"
                ),
            )})
        if attribute.assertion == "uncertain":
            return CriterionFinding(**{**base, **dict(
                satisfied=False,
                verdict=policy.uncertain_assertion,
                reason=(
                    f"{requirement.name}: the source addressed it and hedged, "
                    f"so it is neither a case nor an evaluated negative"
                ),
            )})
        return CriterionFinding(**{**base, **dict(
            satisfied=False,
            verdict="non_case",
            reason=(
                f"{requirement.name} is {attribute.assertion} via "
                f"{attribute.method} from {attribute.source_variable}: this is "
                f"a documented negative, so the subject is a non_case and "
                f"belongs in the denominator"
            ),
        )})

    def _temporal_finding(self, record: CanonicalAERecord) -> CriterionFinding:
        rule = self.definition.temporal
        relation = record.exposure_relation
        base = dict(
            name="temporal",
            source_variable=relation.source_variable,
            source=relation.source,
            method=relation.method,
            spans=list(relation.evidence),
            value=relation.value,
            assertion=relation.assertion,
            availability=relation.availability,
        )
        if not relation.observed or relation.value is None:
            return CriterionFinding(**{**base, **dict(
                satisfied=False,
                verdict=rule.on_unresolved,
                reason=(
                    f"the offset from {rule.anchor} to onset is "
                    f"{relation.availability}"
                    + (f" ({relation.note})" if relation.note else "")
                    + "; without it the window cannot be evaluated"
                ),
            )})
        offset = int(relation.value)
        inside = rule.contains(offset)
        return CriterionFinding(**{**base, **dict(
            satisfied=inside,
            verdict="case" if inside else "non_case",
            reason=(
                f"onset is {offset} days after {rule.anchor}, "
                f"{'inside' if inside else 'outside'} the declared window "
                f"[{rule.minimum}, {rule.maximum}] days"
            ),
        )})

    def _grade_finding(self, record: CanonicalAERecord) -> CriterionFinding:
        rule = self.definition.grade
        attribute = record.attribute(rule.attribute)
        base = dict(
            name=rule.attribute,
            source_variable=(attribute.source_variable if attribute else None),
            source=(attribute.source if attribute else None),
            method=(attribute.method if attribute else None),
            spans=list(attribute.evidence) if attribute else [],
            value=(attribute.value if attribute else None),
            assertion=(attribute.assertion if attribute else None),
            availability=(attribute.availability if attribute else "unresolved"),
        )
        if attribute is None or not attribute.observed or attribute.value is None:
            return CriterionFinding(**{**base, **dict(
                satisfied=False,
                verdict=rule.on_unavailable,
                reason=(
                    f"{rule.attribute} is "
                    f"{attribute.availability if attribute else 'unresolved'}, "
                    f"so the grade threshold cannot be evaluated"
                ),
            )})
        satisfied = int(attribute.value) >= rule.minimum
        return CriterionFinding(**{**base, **dict(
            satisfied=satisfied,
            verdict="case" if satisfied else "non_case",
            reason=(
                f"{rule.attribute} {attribute.value} is "
                f"{'at or above' if satisfied else 'below'} the declared "
                f"minimum {rule.minimum}"
            ),
        )})

    def _exposure_finding(self, record: CanonicalAERecord) -> CriterionFinding:
        rule = self.definition.cumulative_exposure
        total = self.exposure_totals.get(record.record_id)
        base = dict(
            name="cumulative_exposure",
            source_variable="EX",
            source="cross_domain",
            method="derived",
            value=total,
            spans=[],
        )
        if total is None:
            return CriterionFinding(**{**base, **dict(
                satisfied=False,
                verdict=rule.on_unresolved,
                availability="unresolved",
                reason=(
                    "cumulative exposure before onset could not be computed "
                    "from EX, so the threshold cannot be evaluated"
                ),
            )})
        satisfied = total >= rule.minimum
        return CriterionFinding(**{**base, **dict(
            satisfied=satisfied,
            verdict="case" if satisfied else "non_case",
            assertion="present",
            availability="observed",
            spans=[Span(
                doc_id=f"EX:{record.subject_id}", start=0, end=0,
                field="cumulative_exposure",
                extracted_value=f"{total:g} {rule.unit}",
                text=f"sum of EX doses before onset = {total:g} {rule.unit}",
                kind="derived",
            )],
            reason=(
                f"cumulative exposure before onset is {total:g} {rule.unit}, "
                f"{'at or above' if satisfied else 'below'} the declared "
                f"minimum {rule.minimum:g} {rule.unit}"
            ),
        )})

    # -- combining ----------------------------------------------------------

    #: Worst-first. `not_ascertainable` outranks `review`: a question the study
    #: cannot answer is not a question a human reviewer can resolve either, and
    #: sending it to the review queue would waste the queue.
    PRECEDENCE = ("not_ascertainable", "review", "non_case", "case")

    def _decide(
        self, findings: Sequence[CriterionFinding]
    ) -> tuple[str, str | None, str]:
        for verdict in self.PRECEDENCE:
            deciding = next(
                (f for f in findings if f.verdict == verdict and not f.satisfied),
                None,
            )
            if deciding is not None:
                return verdict, deciding.name, deciding.reason
        satisfied = [f.name for f in findings if f.satisfied]
        return "case", None, (
            f"every criterion is satisfied: {', '.join(satisfied)}"
        )

    # -- many records -------------------------------------------------------

    def evaluate_all(
        self, records: Iterable[CanonicalAERecord]
    ) -> EvaluationResult:
        result = EvaluationResult(definition=self.definition)
        for record in records:
            assignment = self.evaluate(record)
            if assignment is not None:
                result.assignments.append(assignment)
        return result


def evaluate_definition(
    definition: PhenotypeDefinition,
    records: Iterable[CanonicalAERecord],
    catalog: ConceptCatalog | None = None,
    exposure_totals: dict[str, float] | None = None,
) -> EvaluationResult:
    return PhenotypeEvaluator(definition, catalog, exposure_totals).evaluate_all(records)


def denominator_table(result: EvaluationResult) -> list[dict[str, Any]]:
    rows = [d.to_dict() for d in result.denominators()]
    rows.append({**result.overall().to_dict(), "study_id": "ALL"})
    return rows


def cases_by_subject(result: EvaluationResult) -> dict[str, list[CaseAssignment]]:
    by_subject: dict[str, list[CaseAssignment]] = defaultdict(list)
    for assignment in result.cases():
        by_subject[assignment.subject_id].append(assignment)
    return dict(by_subject)
