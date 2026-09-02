"""Evaluating a definition over episodes, with four verdicts.

The fourth verdict is the point of this module.

``case``
    every required attribute is present and satisfies its rule
``not_case``
    a required attribute is present and *fails* its rule
``not_ascertainable``
    a required attribute is unavailable and unrecoverable — nobody can evaluate
    the rule, not the system and not a reviewer, so this is neither a negative
    nor a review item
``review``
    an attribute is present but weakly supported, or an onset could not be
    resolved against the anchor — a person could settle it

Precedence: a requirement that is present and fails settles the episode as a
negative, whatever else is missing. Knowing the rash was on the arm makes it not
a truncal rash even if the onset date is missing. Only when nothing has failed
does an unavailable requirement make the episode unascertainable.

The evaluator is **route-agnostic where the definition says so**. It never asks
where a value came from except to check it against ``accept_methods``, and it
records the route on every assignment so a reader can see what the cohort
depended on.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from typing import Any, Iterable, Sequence

from ..catalog import ConceptCatalog
from ..models import (
    Attribute,
    AttributeFinding,
    AttributeRequirement,
    CanonicalAEEpisode,
    CaseAssignment,
    PhenotypeDefinition,
    Span,
    Verdict,
)

#: Which verdict wins when requirements disagree. A definite negative outranks
#: an unascertainable one, which outranks a review, which outranks a case.
VERDICT_PRECEDENCE: tuple[str, ...] = (
    "not_case", "not_ascertainable", "review", "case",
)


@dataclass
class EpisodeVerdict:
    episode: CanonicalAEEpisode
    verdict: Verdict
    reason: str
    deciding_attribute: str | None = None
    findings: list[AttributeFinding] = _dc_field(default_factory=list)
    review_reasons: list[str] = _dc_field(default_factory=list)
    spans: list[Span] = _dc_field(default_factory=list)

    def attribute_sources(self) -> dict[str, str]:
        return {
            f.name: f.source_variable for f in self.findings
            if f.satisfied and f.source_variable
        }

    def attribute_methods(self) -> dict[str, str]:
        return {
            f.name: f.method for f in self.findings if f.satisfied and f.method
        }


class PhenotypeEvaluator:
    def __init__(self, definition: PhenotypeDefinition, catalog: ConceptCatalog):
        self.definition = definition
        self.catalog = catalog
        self.concept_ids = self._concept_ids()

    def _concept_ids(self) -> set[str]:
        ids = {self.definition.concept.primary}
        if self.definition.concept.group:
            ids.update(self.catalog.expand_group(self.definition.concept.group))
        return ids

    # -- concept identity ---------------------------------------------------

    def concept_terms(self, episode: CanonicalAEEpisode) -> set[str]:
        """Coded terms that count as this concept for this episode.

        With ``bridge_dictionary_versions`` the union across versions applies,
        so a study coded under an earlier dictionary still matches. Without it,
        only the terms that episode's own version carried are eligible — which
        is how you measure what bridging is worth.
        """
        concept = self.catalog.concept(self.definition.concept.primary)
        if self.definition.concept.bridge_dictionary_versions:
            terms = set(concept.all_coded_terms())
        else:
            terms = set()
            for version in episode.dictionary_versions or [None]:
                terms.update(concept.coded_terms_for_version(version))
        return {t.strip().casefold() for t in terms}

    def in_scope(self, episode: CanonicalAEEpisode) -> tuple[bool, str]:
        if episode.standardized_concept in self.concept_ids:
            if self.definition.concept.bridge_dictionary_versions:
                return True, ""
            eligible = self.concept_terms(episode)
            matched = [
                t for t in episode.coded_events if t.strip().casefold() in eligible
            ]
            if matched:
                return True, ""
            return False, (
                f"coded terms {episode.coded_events} are not terms for "
                f"{self.definition.concept.primary} under this episode's "
                f"dictionary version {episode.dictionary_versions}, and this "
                f"definition does not bridge versions"
            )
        return False, (
            f"episode concept {episode.standardized_concept!r} is not "
            f"{sorted(self.concept_ids)}"
        )

    # -- one requirement ----------------------------------------------------

    def check(
        self, requirement: AttributeRequirement, episode: CanonicalAEEpisode
    ) -> tuple[str, AttributeFinding]:
        """Evaluate one requirement, returning a verdict and what supported it."""
        attribute = self._attribute_for(requirement, episode)
        if attribute is None:
            return "not_ascertainable", AttributeFinding(
                name=requirement.name, satisfied=False,
                reason=f"the episode carries no {requirement.name} attribute",
            )

        finding = AttributeFinding(
            name=requirement.name,
            satisfied=False,
            value=attribute.value,
            method=attribute.method,
            source=attribute.source,
            source_variable=attribute.source_variable,
            availability=attribute.availability,
            spans=list(attribute.evidence),
        )

        if not attribute.populated:
            # For a window there are two different emptinesses. An episode with
            # a start date whose anchor could not be resolved is *unresolved* —
            # a person with the exposure record could settle it. An episode with
            # no start date at all is unavailable, and nobody can.
            if requirement.window is not None and episode.episode_start.populated:
                finding.reason = (
                    f"the episode starts "
                    f"{episode.episode_start.value.isoformat()} but the offset "
                    f"from {requirement.window.anchor} could not be resolved"
                    + (f" ({attribute.note})" if attribute.note else "")
                )
                return requirement.on_unresolved, finding
            finding.reason = self._unavailable_reason(requirement, attribute)
            return requirement.on_unavailable, finding

        if attribute.method not in requirement.accept_methods:
            finding.reason = (
                f"{requirement.name} came by the {attribute.method!r} route from "
                f"{attribute.source_variable}, which this definition does not "
                f"accept (it accepts {requirement.accept_methods})"
            )
            return requirement.on_unavailable, finding

        if (
            requirement.accept_sources
            and attribute.source not in requirement.accept_sources
        ):
            finding.reason = (
                f"{requirement.name} came from {attribute.source!r}, which this "
                f"definition does not accept"
            )
            return requirement.on_unavailable, finding

        if requirement.window is not None:
            return self._window(requirement, episode, attribute, finding)

        if requirement.allowed is not None and attribute.value not in requirement.allowed:
            finding.reason = (
                f"{requirement.name} is {attribute.value!r}, which is not one of "
                f"{requirement.allowed}"
            )
            return "not_case", finding

        if (
            requirement.min_confidence is not None
            and attribute.confidence is not None
            and attribute.confidence < requirement.min_confidence
        ):
            finding.reason = (
                f"{requirement.name} is {attribute.value!r} but its confidence "
                f"{attribute.confidence:.2f} is below the definition's threshold "
                f"{requirement.min_confidence:.2f}"
            )
            return requirement.on_low_confidence, finding

        finding.satisfied = True
        finding.reason = (
            f"{requirement.name} is {attribute.value!r}, taken by the "
            f"{attribute.method} route from {attribute.source_variable}"
        )
        return "case", finding

    def _attribute_for(
        self, requirement: AttributeRequirement, episode: CanonicalAEEpisode
    ) -> Attribute[Any] | None:
        if requirement.name == "onset":
            return episode.onset_offset_days
        return episode.attribute(requirement.name)

    @staticmethod
    def _unavailable_reason(
        requirement: AttributeRequirement, attribute: Attribute[Any]
    ) -> str:
        detail = f" ({attribute.note})" if attribute.note else ""
        if attribute.availability == "not_collected_by_protocol":
            return (
                f"{requirement.name} was never collected by this study's "
                f"protocol and is not recoverable from anywhere{detail}"
            )
        return (
            f"{requirement.name} is {attribute.availability}{detail}, so the "
            f"rule cannot be evaluated on this episode"
        )

    def _window(
        self, requirement: AttributeRequirement, episode: CanonicalAEEpisode,
        attribute: Attribute[Any], finding: AttributeFinding,
    ) -> tuple[str, AttributeFinding]:
        window = requirement.window
        assert window is not None
        offset = attribute.value
        if offset is None:
            finding.reason = "the onset offset could not be resolved"
            return requirement.on_unresolved, finding
        if not window.contains(int(offset)):
            finding.reason = (
                f"onset is {offset} days from {window.anchor}, outside "
                f"[{window.min}, {window.max}]"
            )
            return "not_case", finding
        finding.satisfied = True
        finding.reason = (
            f"onset is {offset} days from {window.anchor}, inside "
            f"[{window.min}, {window.max}]"
        )
        return "case", finding

    # -- one episode --------------------------------------------------------

    def evaluate_episode(self, episode: CanonicalAEEpisode) -> EpisodeVerdict:
        if episode.candidate:
            return EpisodeVerdict(
                episode=episode, verdict="not_case",
                deciding_attribute=None,
                reason=(
                    "the episode is an unadjudicated discovery candidate and "
                    "cannot enter a cohort without adjudication or a new "
                    "definition version"
                ),
            )

        in_scope, why = self.in_scope(episode)
        if not in_scope:
            return EpisodeVerdict(
                episode=episode, verdict="not_case", reason=why,
            )

        review_reasons: list[str] = []
        if episode.linkage_review_required:
            review_reasons.append(
                f"episode linkage was flagged for review: {episode.linkage_note}"
            )
        elif episode.linkage_confidence < self.definition.episode_linkage_confidence:
            review_reasons.append(
                f"linkage confidence {episode.linkage_confidence:.2f} is below "
                f"the definition's threshold "
                f"{self.definition.episode_linkage_confidence:.2f} "
                f"(rule: {episode.linkage_rule})"
            )

        findings: list[AttributeFinding] = []
        outcomes: list[tuple[str, AttributeFinding]] = []
        for requirement in self.definition.required_attributes:
            verdict, finding = self.check(requirement, episode)
            outcomes.append((verdict, finding))
            findings.append(finding)

        verdict = "case"
        deciding: str | None = None
        for candidate in VERDICT_PRECEDENCE:
            matching = [(v, f) for v, f in outcomes if v == candidate]
            if matching:
                verdict = candidate
                deciding = matching[0][1].name if candidate != "case" else None
                break

        if verdict == "case" and review_reasons:
            verdict = self.definition.on_linkage_review
            deciding = "episode_linkage"

        reasons = [f.reason for v, f in outcomes if v == verdict] or [
            f.reason for _v, f in outcomes
        ]
        reason = "; ".join(r for r in reasons if r)
        if verdict == "case":
            reason = "; ".join(f.reason for f in findings if f.satisfied)
        if review_reasons and verdict == "review":
            reason = f"{reason}; {review_reasons[0]}" if reason else review_reasons[0]

        return EpisodeVerdict(
            episode=episode,
            verdict=verdict,  # type: ignore[arg-type]
            deciding_attribute=deciding,
            reason=reason,
            findings=findings,
            review_reasons=review_reasons,
            spans=[s for f in findings if f.satisfied for s in f.spans],
        )

    # -- a cohort -----------------------------------------------------------

    def evaluate(
        self, episodes: Iterable[CanonicalAEEpisode]
    ) -> list[CaseAssignment]:
        assignments: list[CaseAssignment] = []
        for episode in sorted(episodes, key=lambda e: e.episode_id):
            verdict = self.evaluate_episode(episode)
            assignments.append(CaseAssignment(
                episode_id=episode.episode_id,
                subject_id=episode.subject_id,
                study_id=episode.study_id,
                profile=episode.profile,
                verdict=verdict.verdict,
                deciding_attribute=verdict.deciding_attribute,
                reason=verdict.reason,
                findings=verdict.findings,
                source_record_ids=list(episode.source_record_ids),
                evidence_spans=sorted(
                    verdict.spans, key=lambda s: (s.field, s.doc_id, s.start)
                ),
                attribute_sources=verdict.attribute_sources(),
                attribute_methods=verdict.attribute_methods(),
                definition_id=self.definition.id,
                definition_version=self.definition.version,
                definition_hash=self.definition.definition_hash,
                linkage_review_required=episode.linkage_review_required,
                review_reasons=verdict.review_reasons,
            ))
        return assignments

    def evaluate_subjects(
        self, episodes: Iterable[CanonicalAEEpisode]
    ) -> dict[str, str]:
        """Subject-level verdict: the strongest claim across their episodes.

        A subject with one truncal rash is a case whatever else they had; a
        subject whose only rash cannot be ascertained is unascertainable, not a
        negative.
        """
        order = ["not_case", "review", "not_ascertainable", "case"]
        best: dict[str, str] = {}
        for assignment in self.evaluate(episodes):
            current = best.get(assignment.subject_id)
            if current is None or order.index(assignment.verdict) > order.index(current):
                best[assignment.subject_id] = assignment.verdict
        return dict(sorted(best.items()))


def evaluate_definition(
    episodes: Iterable[CanonicalAEEpisode], definition: PhenotypeDefinition,
    catalog: ConceptCatalog,
) -> list[CaseAssignment]:
    return PhenotypeEvaluator(definition, catalog).evaluate(episodes)
