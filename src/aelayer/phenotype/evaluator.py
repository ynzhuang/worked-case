"""Evaluate a phenotype definition over episodes.

Input: derived ``CanonicalAEEpisode`` objects and one ``PhenotypeDefinition``.
Output: one ``CaseAssignment`` per episode, each naming the rule that decided it.

Three things this evaluator does that a naive one would not.

**It distinguishes a failed test from an untestable one.**  A glucose value the
study measured and that came back at 90 mg/dL fails the ``supported`` rule on
the evidence.  A study that never measured glucose fails it for want of
evidence.  Those are different findings, and the definition's ``missingness``
policy decides what happens to each — the evaluator never assumes the second is
the first.

**It refuses to read a blank as absence unless the definition says so.**  And
the definition cannot say so for ``not_collected_by_protocol`` or
``not_applicable_gated``; the loader rejects that.

**It carries linkage uncertainty forward.**  An episode assembled under a rule
the reconciler flagged is routed by ``episode.on_review_required`` rather than
counted as though the linkage were certain.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field as _dc_field
from typing import Any, Iterable, Sequence

from ..anchors import AnchorResolver
from ..catalog import ConceptCatalog
from ..models import (
    EVIDENCE_STATE_RANK,
    CanonicalAEEpisode,
    CaseAssignment,
    PhenotypeDefinition,
    Span,
)

_VERDICT_RANK = {"excluded": 0, "review": 1, "case": 2}

_LAB_OPS = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


@dataclass
class PredicateResult:
    """The outcome of one predicate, and why it came out that way."""

    satisfied: bool
    explanation: str
    spans: list[Span] = _dc_field(default_factory=list)
    #: Set when the predicate could not be evaluated on the evidence available,
    #: as opposed to being evaluated and failing.
    unresolved_field: str | None = None
    unresolved_state: str | None = None


@dataclass
class EpisodeVerdict:
    episode: CanonicalAEEpisode
    state: str
    verdict: str
    route: str
    rule_id: str | None
    reason: str
    review_reasons: list[str] = _dc_field(default_factory=list)
    spans: list[Span] = _dc_field(default_factory=list)
    offset_days: int | None = None

    @property
    def rank(self) -> tuple[int, int]:
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
        """Concepts in scope, by explicit catalogue membership only."""
        ids = {self.definition.concept.primary}
        if self.definition.concept.group:
            ids.update(self.catalog.expand_group(self.definition.concept.group))
        return ids

    # -- concept identity ---------------------------------------------------

    def concept_terms(self, episode: CanonicalAEEpisode) -> set[str]:
        """Coded terms that count as this concept for this episode.

        With ``bridge_dictionary_versions`` the union across versions applies,
        so a study coded under an earlier dictionary still matches.  Without
        it, only the terms that episode's own dictionary version carried are
        eligible — which is how you measure what bridging is worth.
        """
        concept = self.catalog.concept(self.definition.concept.primary)
        if self.definition.concept.bridge_dictionary_versions:
            terms = set(concept.all_coded_terms())
        else:
            terms = set()
            for version in episode.dictionary_versions or [None]:
                terms.update(concept.coded_terms_for_version(version))
        return {t.strip().casefold() for t in terms}

    # -- window -------------------------------------------------------------

    def resolve_offset(
        self, episode: CanonicalAEEpisode
    ) -> tuple[int | None, bool, str]:
        anchor = self.definition.anchor
        if anchor is None:
            return None, True, "no anchor required"
        start = episode.episode_start.value
        if start is None:
            return None, False, (
                f"episode start is {episode.episode_start.collection_state}, so "
                f"the window cannot be evaluated"
            )
        if self.resolver is None:
            return None, False, "no exposure data available to resolve the anchor"
        hit = self.resolver.resolve(
            episode.subject_id, anchor.event,
            index_rule=anchor.index_rule, onset_date=start.date(),
        )
        if hit is None:
            return None, False, (
                f"no {anchor.event} occurrence in {anchor.source_domain} for "
                f"this subject"
            )
        return (start.date() - hit.date).days, True, (
            f"episode start {start.date().isoformat()} against {anchor.event} "
            f"on {hit.date.isoformat()} ({anchor.index_rule})"
        )

    # -- per episode --------------------------------------------------------

    def evaluate_episode(self, episode: CanonicalAEEpisode) -> EpisodeVerdict:
        route = "normal"
        review_reasons: list[str] = []

        # A discovery candidate has not been adjudicated and may not enter a
        # cohort on its own. It is surfaced, never counted.
        if episode.candidate:
            return self._verdict(
                episode, "none", "excluded", "candidate",
                "the episode is an unadjudicated discovery candidate and "
                "cannot enter a cohort without adjudication or a new "
                "definition version",
                review_reasons,
            )

        if episode.standardized_concept not in self.concept_ids:
            # Not this definition's concept at all. Only an episode whose
            # concept could not be standardized is worth flagging.
            if episode.standardized_concept is None and episode.verbatim_terms:
                review_reasons.append(
                    "the episode has no standardized concept, so it could not "
                    "be tested against this definition"
                )
            return self._verdict(
                episode, "none", "excluded", "concept",
                f"episode concept {episode.standardized_concept!r} is not "
                f"{sorted(self.concept_ids)}",
                review_reasons,
            )

        # Linkage uncertainty travels with the episode.
        policy = self.definition.episode
        if episode.linkage_review_required:
            route = "review"
            review_reasons.append(
                f"episode linkage was flagged for review: {episode.linkage_note}"
            )
        elif episode.linkage_confidence < policy.require_linkage_confidence:
            route = "review"
            review_reasons.append(
                f"linkage confidence {episode.linkage_confidence:.2f} is below "
                f"the definition's threshold "
                f"{policy.require_linkage_confidence:.2f} "
                f"(rule: {episode.linkage_rule})"
            )

        offset = None
        if self.definition.window is not None:
            offset, resolved, detail = self.resolve_offset(episode)
            if not resolved:
                action = self.definition.window.on_unresolved_onset
                if action == "exclude":
                    return self._verdict(
                        episode, "none", "excluded", "window.on_unresolved_onset",
                        f"onset could not be resolved ({detail}); the definition "
                        f"excludes unresolved onsets", review_reasons,
                    )
                if action == "review":
                    route = "review"
                    review_reasons.append(
                        f"onset could not be resolved ({detail}); the definition "
                        f"routes unresolved onsets to review"
                    )
            elif not self.definition.window.contains(offset):
                return self._verdict(
                    episode, "none", "excluded", "window",
                    f"onset {offset} days from {self.definition.anchor.event} is "
                    f"outside [{self.definition.window.min}, "
                    f"{self.definition.window.max}]",
                    review_reasons, offset=offset,
                )

        for rule in self.definition.evidence_rules:
            result = self._match(rule.when, episode)
            if result.satisfied:
                reason = (
                    f"rule {rule.id!r} assigned state {rule.state!r} because "
                    f"{result.explanation}"
                )
                return self._verdict(
                    episode, rule.state, None, rule.id, reason, review_reasons,
                    offset=offset, spans=result.spans, route=route,
                )
            # A rule that could not be evaluated for want of evidence is
            # reported as such, and routed if the definition says so.
            if result.unresolved_field and result.unresolved_state:
                if result.unresolved_state in self.definition.missingness.route_to_review:
                    route = "review"
                    review_reasons.append(
                        f"rule {rule.id!r} could not be evaluated: "
                        f"{result.unresolved_field} is "
                        f"{result.unresolved_state} — {result.explanation}"
                    )

        reason = "no evidence rule matched this episode"
        return self._verdict(
            episode, "none", None, None, reason, review_reasons,
            offset=offset, route=route,
        )

    def _verdict(
        self, episode, state, forced_verdict, rule_id, reason, review_reasons,
        offset=None, spans=None, route="normal",
    ) -> EpisodeVerdict:
        """Map a state and a routing decision onto a verdict.

        Routing is more specific than the state-to-verdict mapping and wins over
        it: when a definition routes something to review it is asking for a
        human, and that is true whether or not a rule also fired.
        """
        if forced_verdict is not None:
            verdict = forced_verdict
        else:
            verdict = self.definition.case_definition.verdict_for(state)
            if route == "review":
                verdict = "review"
        if rule_id and f"{rule_id!r}" not in reason:
            reason = f"rule {rule_id!r}: {reason}"
        if review_reasons and verdict == "review":
            reason = f"{reason}; routed to review because {review_reasons[0]}"
        return EpisodeVerdict(
            episode=episode, state=state, verdict=verdict, route=route,
            rule_id=rule_id, reason=reason, review_reasons=list(review_reasons),
            spans=list(spans or []), offset_days=offset,
        )

    # -- rule language ------------------------------------------------------

    def _match(self, condition: Any, episode: CanonicalAEEpisode) -> PredicateResult:
        if isinstance(condition, list):
            spans: list[Span] = []
            parts: list[str] = []
            for item in condition:
                result = self._match(item, episode)
                if not result.satisfied:
                    return result
                spans.extend(result.spans)
                parts.append(result.explanation)
            return PredicateResult(True, " and ".join(parts), spans)

        results = [
            self._predicate(key, body, episode) for key, body in condition.items()
        ]
        for result in results:
            if not result.satisfied:
                return result
        return PredicateResult(
            True,
            " and ".join(r.explanation for r in results),
            [s for r in results for s in r.spans],
        )

    def _predicate(
        self, key: str, body: Any, episode: CanonicalAEEpisode
    ) -> PredicateResult:
        if key == "all":
            return self._match(body, episode)
        if key == "any":
            failures = []
            unresolved = None
            for item in body:
                result = self._match(item, episode)
                if result.satisfied:
                    return result
                failures.append(result.explanation)
                unresolved = unresolved or result
            return PredicateResult(
                False, f"none of ({'; '.join(failures)})",
                unresolved_field=unresolved.unresolved_field if unresolved else None,
                unresolved_state=unresolved.unresolved_state if unresolved else None,
            )
        if key == "not":
            result = self._match(body, episode)
            return PredicateResult(not result.satisfied, f"not ({result.explanation})")

        if key == "coded_term_matches_concept":
            return self._coded_term(body, episode)
        if key == "has_coded_term":
            present = bool(episode.coded_terms)
            return self._stateful(
                episode, "coded_term", present is bool(body),
                f"coded term {'present' if present else 'absent'}",
            )
        if key == "lab":
            return self._lab(body, episode)
        if key == "symptoms":
            return self._symptoms(body, episode)
        if key == "onset_offset_days":
            return self._onset(body, episode)
        if key == "collection_state":
            return self._collection_state(body, episode)
        if key == "linkage_review_required":
            value = episode.linkage_review_required
            return PredicateResult(
                value is bool(body),
                f"linkage review {'is' if value else 'is not'} required",
            )
        if key == "seriousness":
            field = episode.seriousness
            return self._stateful(
                episode, "seriousness", field.value is bool(body),
                f"seriousness is {field.value!r}",
            )
        if key == "peak_severity":
            allowed = body if isinstance(body, list) else [body]
            peak = episode.peak_severity
            return self._stateful(
                episode, "severity", peak in allowed,
                f"peak severity is {peak!r}"
                + ("" if peak in allowed else f", not one of {allowed}"),
            )
        if key == "seriousness_criteria":
            allowed = set(body if isinstance(body, list) else [body])
            present = {c for _when, cs in episode.seriousness_trajectory for c in cs}
            hit = sorted(present & allowed)
            return self._stateful(
                episode, "seriousness", bool(hit),
                f"seriousness criteria {sorted(present)} "
                f"{'include' if hit else 'do not include'} {sorted(allowed)}",
            )

        # Remaining enumerated attributes, read off the episode's Field.
        field = episode.field_for(key)
        if field is None:
            return PredicateResult(False, f"episode has no field {key!r}")
        allowed = body if isinstance(body, list) else [body]
        return self._stateful(
            episode, key, field.value in allowed,
            f"{key} is {field.value!r}"
            + ("" if field.value in allowed else f", not one of {allowed}"),
        )

    def _stateful(
        self, episode: CanonicalAEEpisode, field_name: str, satisfied: bool,
        explanation: str,
    ) -> PredicateResult:
        """Attach the field's collection state to a failed predicate.

        A predicate that failed on collected evidence is a finding.  One that
        failed because the study never collected the field is not, and the
        difference has to reach the definition's missingness policy intact.
        """
        if satisfied:
            return PredicateResult(True, explanation, self._spans_for(episode, field_name))
        state = episode.field_states.get(field_name, "unknown")
        if state == "collected":
            return PredicateResult(False, explanation)
        note = episode.field_notes.get(field_name)
        detail = (
            f"{explanation}; {field_name} is {state}"
            + (f" ({note})" if note else "")
            + (
                "" if state in self.definition.missingness.treat_as_absent
                else ", which is not evidence of absence"
            )
        )
        return PredicateResult(
            False, detail, unresolved_field=field_name, unresolved_state=state,
        )

    @staticmethod
    def _spans_for(episode: CanonicalAEEpisode, field_name: str) -> list[Span]:
        return [s for s in episode.linked_evidence if s.field == field_name]

    def _coded_term(self, body: Any, episode: CanonicalAEEpisode) -> PredicateResult:
        eligible = self.concept_terms(episode)
        found = sorted(
            t for t in episode.coded_terms if t.strip().casefold() in eligible
        )
        matched = bool(found)
        bridged = self.definition.concept.bridge_dictionary_versions
        explanation = (
            f"coded term {found} is a catalogue term for "
            f"{self.definition.concept.primary}"
            if matched else
            f"coded terms {episode.coded_terms} are not catalogue terms for "
            f"{self.definition.concept.primary}"
            + ("" if bridged else " under this episode's dictionary version")
        )
        return self._stateful(
            episode, "coded_term", matched is bool(body), explanation
        )

    def _lab(self, body: dict, episode: CanonicalAEEpisode) -> PredicateResult:
        test = body["test"]
        operator = _LAB_OPS[body["op"]]
        threshold = float(body["value"])
        unit = body.get("unit")
        lab_test = self.catalog.lab_tests.get(test)

        # The threshold is converted into canonical units once, so a study
        # reporting mmol/L is compared like for like rather than misread.
        if unit and lab_test is not None:
            converted = lab_test.to_canonical(threshold, unit)
            if converted is None:
                return PredicateResult(
                    False, f"threshold unit {unit!r} has no conversion for {test}"
                )
            threshold = converted

        values = [l for l in episode.labs if l.test == test and l.canonical_value is not None]
        for lab in values:
            if operator(lab.canonical_value, threshold):
                canonical_unit = lab_test.canonical_unit if lab_test else ""
                return PredicateResult(
                    True,
                    f"{test} {lab.value} {lab.unit} (= {lab.canonical_value} "
                    f"{canonical_unit}) {body['op']} {threshold} {canonical_unit}",
                    [lab.span],
                )
        if values:
            # Measured, and it did not meet the bar. That is a finding.
            return PredicateResult(
                False,
                f"{test} values {[l.canonical_value for l in values]} do not "
                f"satisfy {body['op']} {threshold}",
            )
        return self._stateful(
            episode, f"labs.{test}", False,
            f"no {test} value is available for this episode",
        )

    def _symptoms(self, body: dict, episode: CanonicalAEEpisode) -> PredicateResult:
        minimum = int(body.get("min_count", 1))
        if "from" in body:
            qualifying = self.catalog.symptoms_in_sets(list(body["from"]))
            label = f"symptom sets {sorted(body['from'])}"
        else:
            qualifying = set(body.get("any_of") or [])
            label = f"symptoms {sorted(qualifying)}"
        matched = [
            s for s in episode.symptoms
            if s.symptom in qualifying and s.assertion == "present"
        ]
        names = sorted({s.symptom for s in matched})
        if len(names) >= minimum:
            return PredicateResult(
                True,
                f"{len(names)} qualifying symptom(s) from {label}: {names}",
                [s.span for s in matched],
            )
        return self._stateful(
            episode, "symptoms", False,
            f"{len(names)} qualifying symptom(s) from {label}, fewer than "
            f"the {minimum} required",
        )

    def _onset(self, body: dict, episode: CanonicalAEEpisode) -> PredicateResult:
        offset, resolved, detail = self.resolve_offset(episode)
        if not resolved or offset is None:
            return PredicateResult(
                False, f"onset offset is unresolved: {detail}",
                unresolved_field="onset_datetime",
                unresolved_state=episode.episode_start.collection_state,
            )
        low, high = body.get("min"), body.get("max")
        ok = (low is None or offset >= low) and (high is None or offset <= high)
        return PredicateResult(
            ok, f"onset offset {offset} days is "
                f"{'within' if ok else 'outside'} [{low}, {high}]",
        )

    def _collection_state(
        self, body: dict, episode: CanonicalAEEpisode
    ) -> PredicateResult:
        """Query the collection state of a field directly.

        Lets a definition say, explicitly, what it wants done about a field the
        study could not record — rather than leaving it to a blanket rule.
        """
        field_name = body["field"]
        wanted = body.get("is")
        wanted_list = [wanted] if isinstance(wanted, str) else list(wanted or [])
        state = episode.field_states.get(field_name, "unknown")
        ok = state in wanted_list if wanted_list else True
        return PredicateResult(
            ok, f"collection state of {field_name} is {state!r}"
                + ("" if ok else f", not one of {wanted_list}"),
        )

    # -- cohort -------------------------------------------------------------

    def evaluate(
        self, episodes: Iterable[CanonicalAEEpisode]
    ) -> list[CaseAssignment]:
        """One assignment per episode, in a stable order."""
        assignments: list[CaseAssignment] = []
        for episode in sorted(episodes, key=lambda e: e.episode_id):
            verdict = self.evaluate_episode(episode)
            assignments.append(
                CaseAssignment(
                    episode_id=episode.episode_id,
                    subject_id=episode.subject_id,
                    study_id=episode.study_id,
                    verdict=verdict.verdict,  # type: ignore[arg-type]
                    evidence_state=verdict.state,  # type: ignore[arg-type]
                    matched_rule_id=verdict.rule_id,
                    reason=verdict.reason,
                    source_record_ids=list(episode.source_record_ids),
                    evidence_spans=sorted(
                        verdict.spans, key=lambda s: (s.field, s.doc_id, s.start)
                    ),
                    definition_id=self.definition.id,
                    definition_version=self.definition.version,
                    definition_hash=self.definition.definition_hash,
                    linkage_review_required=episode.linkage_review_required,
                    review_reasons=verdict.review_reasons,
                )
            )
        return assignments

    def evaluate_subjects(
        self, episodes: Iterable[CanonicalAEEpisode]
    ) -> dict[str, str]:
        """Subject-level verdict: the strongest verdict across their episodes."""
        best: dict[str, tuple[int, str]] = {}
        for assignment in self.evaluate(episodes):
            rank = _VERDICT_RANK[assignment.verdict]
            current = best.get(assignment.subject_id)
            if current is None or rank > current[0]:
                best[assignment.subject_id] = (rank, assignment.verdict)
        return {subject: verdict for subject, (_r, verdict) in sorted(best.items())}


def evaluate_definition(
    episodes: Iterable[CanonicalAEEpisode],
    definition: PhenotypeDefinition,
    catalog: ConceptCatalog,
    anchor_resolver: AnchorResolver | None = None,
) -> list[CaseAssignment]:
    return PhenotypeEvaluator(definition, catalog, anchor_resolver).evaluate(episodes)
