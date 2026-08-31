"""Evaluate a phenotype definition over event objects.

Input: a set of ``EventObject``s and one ``PhenotypeDefinition``.
Output: one ``CaseAssignment`` per subject.

The order of work for each event is fixed, and each step can only remove an
event from consideration or refine its state:

1. **Assertion policy.**  Excluded classes drop out here.  A documented absence
   becomes state ``absent`` rather than disappearing, because a count of
   documented absences is a real number a reviewer will ask for.
2. **Window.**  The offset is measured against the anchor the definition names,
   under the index rule it names.  An unresolvable onset is routed by
   ``on_unresolved_onset``; the extractor never made that decision.
3. **Evidence rules.**  Ordered; the first match assigns the state.
4. **Case definition.**  The state maps to a verdict, downgraded to ``review``
   where step 1 or 2 routed it there.

Every assignment names the rule that decided it.  When a clinician disputes a
case, that is the first question asked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from ..anchors import AnchorResolver
from ..catalog import ConceptCatalog
from ..models import (
    EVIDENCE_STATE_RANK,
    CaseAssignment,
    EventObject,
    PhenotypeDefinition,
    Span,
)

_VERDICT_RANK = {"excluded": 0, "review": 1, "case": 2}

#: Concept-match kinds that count as a verbatim mention in text.
_MENTION_KINDS = {"lexicon", "lexicon_fuzzy", "abbreviation"}

_LAB_OPS = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


@dataclass
class EventVerdict:
    """What the definition made of one event object."""

    event: EventObject
    state: str
    route: str          # normal | review | excluded
    verdict: str        # case | review | excluded
    rule_id: str | None
    reason: str
    offset_days: int | None = None
    spans: list[Span] = field(default_factory=list)

    @property
    def verdict_key(self) -> tuple[int, int]:
        """Sort key for picking a subject's strongest event.

        Verdict dominates, then evidence state, so a `case` on `supported`
        outranks a `review` on `explicit`.
        """
        return (_VERDICT_RANK[self.verdict], EVIDENCE_STATE_RANK[self.state])


class PhenotypeEvaluator:
    def __init__(
        self,
        definition: PhenotypeDefinition,
        catalog: ConceptCatalog,
        anchor_resolver: AnchorResolver | None = None,
    ):
        self.definition = definition
        self.catalog = catalog
        self.resolver = anchor_resolver
        self.concept_ids = self._concept_ids()

    def _concept_ids(self) -> set[str]:
        """Which concepts this definition considers.

        A group is an explicit membership list from the catalogue.  Nothing is
        inferred by walking a hierarchy.
        """
        ids = {self.definition.concept.primary}
        if self.definition.concept.group:
            ids.update(self.catalog.expand_group(self.definition.concept.group))
        return ids

    # -- per event ----------------------------------------------------------

    def evaluate_event(self, event: EventObject) -> EventVerdict:
        policy = self.definition.assertion

        if event.assertion in policy.exclude:
            state = "absent" if event.assertion == "absent" else "none"
            return self._verdict(
                event, state, "excluded", "assertion.exclude",
                f"assertion '{event.assertion}' is excluded by the definition's "
                f"assertion policy",
            )

        route = "normal"
        routing_reason = ""
        if event.assertion in policy.route_to_review:
            route = "review"
            routing_reason = (
                f"assertion '{event.assertion}' is routed to review by the "
                f"definition's assertion policy"
            )
        elif event.assertion not in policy.require:
            return self._verdict(
                event, "none", "excluded", "assertion.require",
                f"assertion '{event.assertion}' is not in the required set "
                f"{policy.require}",
            )

        offset = None
        if self.definition.window is not None:
            offset, resolved, detail = self.resolve_offset(event)
            if not resolved:
                action = self.definition.window.on_unresolved_onset
                if action == "exclude":
                    return self._verdict(
                        event, "none", "excluded", "window.on_unresolved_onset",
                        f"onset could not be resolved ({detail}); the definition "
                        f"excludes unresolved onsets",
                    )
                if action == "review":
                    route = "review"
                    routing_reason = (
                        f"onset could not be resolved ({detail}); the definition "
                        f"routes unresolved onsets to review"
                    )
            elif not self.definition.window.contains(offset):
                return self._verdict(
                    event, "none", "excluded", "window",
                    f"onset {offset} days from {self.definition.anchor.event} is "
                    f"outside the window "
                    f"[{self.definition.window.min}, {self.definition.window.max}]",
                    offset=offset,
                )

        for rule in self.definition.evidence_rules:
            matched, spans, explanation = self._match(rule.when, event)
            if matched:
                reason = (
                    f"rule '{rule.id}' assigned state '{rule.state}' because "
                    f"{explanation}"
                )
                if routing_reason:
                    reason = f"{reason}; routed to review because {routing_reason}"
                return self._verdict(
                    event, rule.state, route, rule.id, reason, offset=offset, spans=spans
                )

        reason = "no evidence rule matched this event"
        if routing_reason:
            reason = f"{reason}; {routing_reason}"
        return self._verdict(event, "none", route, None, reason, offset=offset)

    def _verdict(self, event, state, route, rule_id, reason, offset=None, spans=None):
        """Map an evidence state and a routing decision onto a verdict.

        Routing is more specific than the state-to-verdict mapping and wins
        over it.  When a definition says ``route_to_review: [uncertain]`` it is
        asking for those events to reach a human, and that is true whether or
        not an evidence rule also fired: an uncertain mention with no
        corroboration is precisely what adjudication exists for.  The remainder
        is then reported as a separate count rather than discarded.
        """
        resolved = self.definition.case_definition.verdict_for(state)
        if route == "excluded":
            resolved = "excluded"
        elif route == "review":
            resolved = "review"
        # The reason always names the deciding rule, including the policy
        # pseudo-rules (`assertion.exclude`, `window`, ...). When a clinician
        # disputes a verdict, the first question is which rule fired, and the
        # answer has to be in the row rather than inferred from a nearby field.
        if rule_id and f"'{rule_id}'" not in reason:
            reason = f"rule '{rule_id}': {reason}"
        return EventVerdict(
            event=event, state=state, route=route, verdict=resolved,
            rule_id=rule_id, reason=reason, offset_days=offset,
            spans=list(spans or []),
        )

    # -- window -------------------------------------------------------------

    def resolve_offset(self, event: EventObject) -> tuple[int | None, bool, str]:
        """Days from the definition's anchor to the event's onset.

        Recomputed from the onset date wherever one exists, because the
        definition's own ``index_rule`` decides which anchor occurrence counts
        and that may differ from the convention the extractor used.
        """
        anchor = self.definition.anchor
        if anchor is None:
            return None, True, "no anchor required"

        if event.onset_date is not None and self.resolver is not None:
            hit = self.resolver.resolve(
                event.subject_id, anchor.event,
                index_rule=anchor.index_rule, onset_date=event.onset_date,
            )
            if hit is not None:
                return (event.onset_date - hit.date).days, True, (
                    f"onset {event.onset_date.isoformat()} against {anchor.event} "
                    f"on {hit.date.isoformat()} ({anchor.index_rule})"
                )
            return None, False, (
                f"no {anchor.event} occurrence found in {anchor.source_domain} "
                f"for this subject"
            )

        if event.onset_offset_days is not None and event.anchor_event == anchor.event:
            return event.onset_offset_days, True, (
                f"offset carried on the event object, anchored to {anchor.event}"
            )

        if event.onset_offset_days is not None:
            return None, False, (
                f"offset is relative to '{event.anchor_event}', not the "
                f"definition's anchor '{anchor.event}'"
            )
        return None, False, "no onset date and no resolvable offset"

    # -- rule language ------------------------------------------------------

    def _match(
        self, condition: Any, event: EventObject
    ) -> tuple[bool, list[Span], str]:
        """Evaluate a ``when`` block, returning the spans that satisfied it."""
        if isinstance(condition, list):
            spans: list[Span] = []
            parts: list[str] = []
            for item in condition:
                ok, item_spans, explanation = self._match(item, event)
                if not ok:
                    return False, [], explanation
                spans.extend(item_spans)
                parts.append(explanation)
            return True, spans, " and ".join(parts)

        results: list[tuple[bool, list[Span], str]] = []
        for key, body in condition.items():
            results.append(self._match_predicate(key, body, event))
        for ok, _spans, explanation in results:
            if not ok:
                return False, [], explanation
        spans = [s for _ok, item_spans, _e in results for s in item_spans]
        return True, spans, " and ".join(e for _o, _s, e in results)

    def _match_predicate(
        self, key: str, body: Any, event: EventObject
    ) -> tuple[bool, list[Span], str]:
        if key == "all":
            return self._match(body, event)
        if key == "any":
            failures = []
            for item in body:
                ok, spans, explanation = self._match(item, event)
                if ok:
                    return True, spans, explanation
                failures.append(explanation)
            return False, [], f"none of ({'; '.join(failures)})"
        if key == "not":
            ok, _spans, explanation = self._match(body, event)
            return (not ok), [], f"not ({explanation})"

        if key == "coded_term_matches_concept":
            matches = "coded_term" in event.concept_match_kinds
            spans = event.spans_for("coded_term") if matches else []
            described = (
                f"the coded term {event.coded_term!r} is a catalogue term for "
                f"{event.concept_id}"
                if matches
                else f"the coded term {event.coded_term!r} is not a catalogue "
                     f"term for {event.concept_id}"
            )
            return (matches is bool(body)), spans, described

        if key == "has_coded_term":
            present = event.coded_term is not None
            return (present is bool(body)), event.spans_for("coded_term"), (
                f"coded term {'present' if present else 'absent'}"
            )

        if key == "lexicon_match":
            wanted = body.get("assertion")
            wanted_list = (
                [wanted] if isinstance(wanted, str) else list(wanted or [])
            )
            has_mention = bool(set(event.concept_match_kinds) & _MENTION_KINDS)
            assertion_ok = not wanted_list or event.assertion in wanted_list
            ok = has_mention and assertion_ok
            spans = event.spans_for("concept_id") if ok else []
            if not has_mention:
                described = "there is no verbatim mention of the concept in text"
            elif not assertion_ok:
                described = (
                    f"the mention's assertion is '{event.assertion}', not "
                    f"{wanted_list}"
                )
            else:
                described = (
                    f"a verbatim mention of the concept is asserted "
                    f"'{event.assertion}'"
                )
            return ok, spans, described

        if key == "lab":
            return self._match_lab(body, event)

        if key == "symptoms":
            return self._match_symptoms(body, event)

        if key == "onset_offset_days":
            offset = event.onset_offset_days
            if offset is None:
                return False, [], "no onset offset on the event"
            low, high = body.get("min"), body.get("max")
            ok = (low is None or offset >= low) and (high is None or offset <= high)
            return ok, event.spans_for("onset_offset_days"), (
                f"onset offset {offset} days is "
                f"{'within' if ok else 'outside'} [{low}, {high}]"
            )

        if key == "rescue_treatment":
            ok = event.rescue_treatment is bool(body)
            return ok, event.spans_for("rescue_treatment"), (
                f"rescue treatment {'was' if event.rescue_treatment else 'was not'} given"
            )

        # Remaining enumerated fields: membership in a list of allowed values.
        value = getattr(event, key, None)
        allowed = body if isinstance(body, list) else [body]
        if isinstance(value, list):
            hit = sorted(set(value) & set(allowed))
            return bool(hit), event.spans_for(key), (
                f"{key} includes {hit}" if hit else f"{key} {value} does not include any of {allowed}"
            )
        ok = value in allowed
        return ok, event.spans_for(key), (
            f"{key} is {value!r}" + ("" if ok else f", not one of {allowed}")
        )

    def _match_lab(self, body: dict, event: EventObject) -> tuple[bool, list[Span], str]:
        test_id = body["test"]
        operator = _LAB_OPS[body["op"]]
        threshold = float(body["value"])
        unit = body.get("unit")
        lab_test = self.catalog.lab_tests.get(test_id)

        # The threshold is expressed in some unit; convert it once into the
        # catalogue's canonical unit and compare canonical to canonical. This is
        # the whole reason unit conversion exists: a study reporting mmol/L must
        # not be silently misclassified against a mg/dL threshold.
        if unit and lab_test is not None:
            converted = lab_test.to_canonical(threshold, unit)
            if converted is None:
                return False, [], (
                    f"threshold unit {unit!r} has no conversion for {test_id}"
                )
            threshold = converted
        for lab in event.labs:
            if lab.test != test_id or lab.canonical_value is None:
                continue
            if operator(lab.canonical_value, threshold):
                canonical_unit = lab_test.canonical_unit if lab_test else ""
                return True, [lab.span], (
                    f"{test_id} {lab.value} {lab.unit} "
                    f"(= {lab.canonical_value} {canonical_unit}) "
                    f"{body['op']} {threshold} {canonical_unit}"
                )
        if not any(lab.test == test_id for lab in event.labs):
            return False, [], f"no {test_id} value on the event"
        return False, [], (
            f"no {test_id} value satisfies {body['op']} {body['value']} "
            f"{unit or ''}".strip()
        )

    def _match_symptoms(
        self, body: dict, event: EventObject
    ) -> tuple[bool, list[Span], str]:
        min_count = int(body.get("min_count", 1))
        if "from" in body:
            qualifying = self.catalog.symptoms_in_sets(list(body["from"]))
            label = f"symptom sets {sorted(body['from'])}"
        else:
            qualifying = set(body.get("any_of") or [])
            label = f"symptoms {sorted(qualifying)}"
        matched = [s for s in event.symptoms if s.symptom in qualifying]
        names = sorted({s.symptom for s in matched})
        ok = len(names) >= min_count
        return ok, [s.span for s in matched] if ok else [], (
            f"{len(names)} qualifying symptom(s) from {label}"
            + (f": {names}" if names else "")
        )

    # -- subject level ------------------------------------------------------

    def evaluate(
        self,
        events: Iterable[EventObject],
        subjects: Iterable[tuple[str, str]] | None = None,
    ) -> list[CaseAssignment]:
        """One assignment per subject.

        ``subjects`` is the full cohort as ``(subject_id, study_id)`` pairs.
        Supplying it matters: a subject with no qualifying event object is still
        part of the denominator and still gets a row saying so.
        """
        by_subject: dict[str, list[EventObject]] = {}
        study_of: dict[str, str] = {}
        for event in events:
            if event.concept_id not in self.concept_ids:
                continue
            by_subject.setdefault(event.subject_id, []).append(event)
            study_of.setdefault(event.subject_id, event.study_id)

        cohort: dict[str, str] = {}
        if subjects is not None:
            cohort.update({s: study for s, study in subjects})
        cohort.update(
            {s: study_of[s] for s in by_subject if s not in cohort}
        )

        assignments: list[CaseAssignment] = []
        for subject_id in sorted(cohort):
            subject_events = sorted(
                by_subject.get(subject_id, []), key=lambda e: e.event_id
            )
            assignments.append(
                self._assign(subject_id, cohort[subject_id], subject_events)
            )
        return assignments

    def _assign(
        self, subject_id: str, study_id: str, events: list[EventObject]
    ) -> CaseAssignment:
        definition = self.definition
        if not events:
            return CaseAssignment(
                subject_id=subject_id,
                study_id=study_id,
                verdict="excluded",
                evidence_state="none",
                matched_rule_id=None,
                reason=(
                    f"no event object for concept "
                    f"{sorted(self.concept_ids)} on this subject"
                ),
                contributing_event_ids=[],
                evidence_spans=[],
                definition_id=definition.id,
                definition_version=definition.version,
                definition_hash=definition.definition_hash,
            )

        verdicts = [self.evaluate_event(event) for event in events]
        best = max(verdicts, key=lambda v: (v.verdict_key, v.event.event_id))
        # Every event that reached the same state contributes, so the row shows
        # the full basis rather than a single arbitrary record.
        contributing = [
            v for v in verdicts
            if v.verdict == best.verdict and v.state == best.state
        ]
        spans = [span for v in contributing for span in v.spans]
        seen: set[tuple] = set()
        unique_spans = []
        for span in spans:
            if span.key() not in seen:
                seen.add(span.key())
                unique_spans.append(span)

        return CaseAssignment(
            subject_id=subject_id,
            study_id=study_id,
            verdict=best.verdict,  # type: ignore[arg-type]
            evidence_state=best.state,  # type: ignore[arg-type]
            matched_rule_id=best.rule_id,
            reason=best.reason,
            contributing_event_ids=sorted(v.event.event_id for v in contributing),
            evidence_spans=sorted(
                unique_spans, key=lambda s: (s.field, s.doc_id, s.start, s.end)
            ),
            definition_id=definition.id,
            definition_version=definition.version,
            definition_hash=definition.definition_hash,
        )


def evaluate_definition(
    events: Iterable[EventObject],
    definition: PhenotypeDefinition,
    catalog: ConceptCatalog,
    anchor_resolver: AnchorResolver | None = None,
    subjects: Iterable[tuple[str, str]] | None = None,
) -> list[CaseAssignment]:
    evaluator = PhenotypeEvaluator(definition, catalog, anchor_resolver)
    return evaluator.evaluate(events, subjects)
