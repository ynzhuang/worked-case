"""Synthetic corpus generator.

Ground truth is sampled first; representations are rendered from it.  That
order is what makes representation invariance measurable at all: the same
clinical episode appears in six studies under six collection conventions, and
the harness can ask whether the pipeline reaches the same conclusion each time.

Nothing here derives from real patients.  Every table row carries a
``SYNTHETIC`` column and every narrative carries a synthetic header.

The variants, per the build spec:

===== ==========================================================================
V-A   one record per episode, everything in structured fields
V-B   split into several records on severity change
V-C   core record plus a linked event form carrying the objective value
V-D   minimal coding; clinical detail only in narrative
V-E   a study whose action codelist lacks the relevant concept
V-F   an earlier dictionary version with different preferred terms
===== ==========================================================================
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from . import paths
from .catalog import ConceptCatalog, load_configs
from .models import EVIDENCE_STATE_RANK, SERIOUSNESS_CRITERIA
from .semantics import CollectionSemantics, StudySemantics

SYNTHETIC_FLAG = "Y"
NARRATIVE_HEADER = (
    "*** SYNTHETIC RECORD - COMPUTER GENERATED - NOT REAL PATIENT DATA ***"
)

TERMINAL_OUTCOMES = {"recovered", "recovered_with_sequelae", "fatal"}

#: Terms a coder may assign when the verbatim does not clearly name the
#: concept. They are real coded terms; they are simply not this concept's.
NON_SPECIFIC_CODED_TERMS = [
    "Malaise", "Asthenia", "Feeling abnormal", "General physical health deterioration",
]


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------


@dataclass
class EpisodeTruth:
    """What actually happened, before any study wrote it down."""

    truth_id: str
    concept: str
    onset_offset_days: int              # from the first dose escalation
    duration_days: int
    severity_steps: list[tuple[int, str]]   # (day offset from onset, severity)
    seriousness: bool
    seriousness_criteria: list[str]
    relatedness: str
    action_taken: str | None
    outcome: str
    glucose_mgdl: float | None
    symptoms: list[str]
    rescue_given: bool
    third_party_assistance: bool
    #: Whether the event was coded to a term that denotes the concept, as
    #: opposed to a non-specific term like "Malaise". This is a property of how
    #: the event was reported and coded, not of the study's conventions, so it
    #: is the same across every rendering of this truth.
    coded_specifically: bool = True
    #: Whether *this study* collects coded terms at all. Representation-
    #: dependent: V-D never does.
    coded_by_study: bool = True
    assertion: str = "present"
    #: A second, distinct episode for the same subject a few days later. For a
    #: concept where recurrence is expected these must stay separate.
    recurrence_gap_days: int | None = None
    note: str = ""

    @property
    def peak_severity(self) -> str:
        order = ["mild", "moderate", "severe"]
        return max((s for _d, s in self.severity_steps), key=order.index)

    @property
    def onset_is_in_window(self) -> bool:
        return 0 <= self.onset_offset_days <= 14

    def true_evidence_state(self, *, as_recorded: bool = False) -> str:
        """The state v1 should assign.

        With ``as_recorded`` false this is representation-independent: it uses
        whether the event was coded *specifically*, not whether this particular
        study collects coded terms at all. That is the reference invariance is
        measured against.

        With ``as_recorded`` true it accounts for the study's conventions, and
        so is what this rendering can actually support.
        """
        if self.concept != "HYPOGLYCEMIA" or self.assertion != "present":
            return "none"
        coded = self.coded_specifically and (self.coded_by_study or not as_recorded)
        has_symptom = bool(self.symptoms)
        low_glucose = self.glucose_mgdl is not None and self.glucose_mgdl < 70
        if coded:
            return "explicit"
        if low_glucose and has_symptom:
            return "supported"
        if has_symptom:
            return "insufficient"
        return "none"

    def true_verdict(self, *, as_recorded: bool = False) -> str:
        """The verdict v1 should reach.

        Representation-independent by default. A study that codes the event and
        one that leaves it to the narrative should land in the same bucket; the
        invariance harness measures whether they do, and reports where they do
        not rather than assuming they will.
        """
        if self.concept != "HYPOGLYCEMIA" or self.assertion != "present":
            return "excluded"
        if not self.onset_is_in_window:
            return "excluded"
        state = self.true_evidence_state(as_recorded=as_recorded)
        if state in ("explicit", "supported"):
            return "case"
        if state == "insufficient":
            return "review"
        return "excluded"


# --------------------------------------------------------------------------
# Surface forms
# --------------------------------------------------------------------------

VERBATIM_TERMS = [
    "low blood sugar episode", "hypoglycaemic event", "symptomatic low glucose",
    "hypo episode", "low glucose reading",
]

SYMPTOM_NOUNS: dict[str, list[str]] = {
    "confusion": ["confusion", "disorientation"],
    "dizziness": ["dizziness", "giddiness"],
    "lightheadedness": ["lightheadedness", "light-headedness"],
    "blurred vision": ["blurred vision", "blurry vision"],
    "difficulty concentrating": ["difficulty concentrating", "poor concentration"],
    "drowsiness": ["drowsiness", "somnolence"],
    "slurred speech": ["slurred speech", "dysarthria"],
    "shakiness": ["shakiness", "the shakes"],
    "tremor": ["tremor", "trembling"],
    "diaphoresis": ["diaphoresis", "profuse sweating"],
    "sweating": ["sweating"],
    "palpitations": ["palpitations", "a racing heart"],
    "pallor": ["pallor", "a pale appearance"],
    "clamminess": ["clamminess"],
    "hunger": ["hunger"],
    "anxiety": ["anxiety"],
}

CONCEPT_SURFACES = {
    "HYPOGLYCEMIA": ["hypoglycaemia", "hypoglycemia", "low blood sugar",
                     "hypoglycaemic episode", "low blood glucose"],
    "HYPERGLYCEMIA": ["hyperglycaemia", "high blood sugar"],
    "NAUSEA": ["nausea", "feeling sick"],
    "HEADACHE": ["headache", "cephalgia"],
    "ANAEMIA": ["anaemia", "low haemoglobin"],
}

ACTION_PHRASES = {
    "dose_not_changed": "No change was made to study drug.",
    "dose_reduced": "The study drug dose was reduced at the next visit.",
    "drug_interrupted": "Study drug was interrupted for 48 hours.",
    "drug_withdrawn": "The subject was permanently withdrawn from study drug.",
    "not_applicable": "No study drug action was applicable.",
    "unknown": "The action taken with study drug was not recorded.",
}

OUTCOME_PHRASES = {
    "recovered": "The event resolved the same day.",
    "recovering": "The event was still resolving at the last assessment.",
    "not_recovered": "The event was ongoing at the end of the reporting period.",
    "recovered_with_sequelae": "The event resolved with residual symptoms.",
    "fatal": "The event had a fatal outcome.",
    "unknown": "The outcome was not recorded.",
}

RELATEDNESS_PHRASES = {
    "not_related": "The investigator considered the event unrelated to study drug.",
    "unlikely": "The investigator judged the event unlikely related to study drug.",
    "possible": "The investigator assessed the event as possibly related to study drug.",
    "probable": "The investigator assessed the event as probably related to study drug.",
    "definite": "The investigator considered the event definitely related to study drug.",
    "unknown": "Causality was not assessed.",
}

CRITERION_PHRASES = {
    "hospitalisation": "The subject was admitted to hospital.",
    "life_threatening": "The episode was considered life threatening.",
    "other_medically_important": "The event was regarded as medically important.",
    "death": "The subject died.",
    "disability": "The event resulted in persistent disability.",
    "congenital_anomaly": "A congenital anomaly was reported.",
}

#: Assertion-bearing sentences. Assertion matters in narrative and discovery,
#: not in a coded AE row: a coded row asserts presence by construction.
NEGATED_ASIDES = [
    "There was no evidence of {concept} at the follow-up visit.",
    "Screening was negative for {concept}.",
    "{concept_cap} was ruled out on review of the glucose log.",
]
HYPOTHETICAL_ASIDES = [
    "The subject was advised to report symptoms of {concept} promptly.",
    "The site was instructed to monitor for {concept} after each dose increase.",
]
HISTORICAL_ASIDES = [
    "Past medical history includes {concept} prior to study entry.",
    "{concept_cap} was documented previously, before screening.",
]


class NarrativeBuilder:
    """Assembles a narrative while recording where key mentions land."""

    def __init__(self, header: str):
        self.header = header
        self._parts: list[str] = []
        self._marks: dict[str, dict[str, Any]] = {}

    @property
    def _cursor(self) -> int:
        return len(self.header) + 1 + sum(len(p) for p in self._parts)

    def sentence(self, text: str) -> None:
        if self._parts:
            self._parts.append(" ")
        self._parts.append(text)

    def marked(self, before: str, mention: str, after: str, label: str) -> None:
        if self._parts:
            self._parts.append(" ")
        start = self._cursor + len(before)
        self._parts.append(before + mention + after)
        self._marks[label] = {"text": mention, "start": start,
                              "end": start + len(mention)}

    def mark(self, label: str) -> dict[str, Any] | None:
        return self._marks.get(label)

    def body(self) -> str:
        return "".join(self._parts)

    def full_text(self) -> str:
        return f"{self.header}\n{self.body()}"


# --------------------------------------------------------------------------
# Generator
# --------------------------------------------------------------------------


@dataclass
class GeneratedCorpus:
    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    narratives: list[dict[str, Any]] = field(default_factory=list)
    truths: list[dict[str, Any]] = field(default_factory=list)
    gold_records: list[dict[str, Any]] = field(default_factory=list)
    gold_episodes: list[dict[str, Any]] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)


class CorpusGenerator:
    def __init__(
        self,
        seed: int = 7,
        n_studies: int = 6,
        invariance_truths: int = 24,
        background_per_study: int = 18,
        catalog: ConceptCatalog | None = None,
        semantics: CollectionSemantics | None = None,
    ):
        self.seed = seed
        self.rng = random.Random(seed)
        if catalog is None or semantics is None:
            catalog, _extraction, semantics, _v = load_configs()
        self.catalog = catalog
        self.semantics = semantics
        all_studies = semantics.study_ids()
        if not 1 <= n_studies <= len(all_studies):
            raise ValueError(f"n_studies must be between 1 and {len(all_studies)}")
        self.study_ids = all_studies[:n_studies]
        self.invariance_truths = invariance_truths
        self.background_per_study = background_per_study
        self._lb_seq: dict[str, int] = {}

    # -- helpers ------------------------------------------------------------

    def _pick(self, seq: list[Any]) -> Any:
        return seq[self.rng.randrange(len(seq))]

    def _symptom_phrase(self, symptoms: list[str]) -> str:
        surfaces = [self._pick(SYMPTOM_NOUNS.get(s, [s])) for s in symptoms]
        if len(surfaces) == 1:
            return surfaces[0]
        return ", ".join(surfaces[:-1]) + " and " + surfaces[-1]

    def _glucose_in(self, unit: str, mgdl: float) -> tuple[float, float]:
        """Return ``(value_as_reported, canonical_mgdl)`` for a study's unit."""
        if unit == "mmol/L":
            reported = round(mgdl / 18.0182, 1)
            return reported, round(reported * 18.0182, 4)
        return float(mgdl), float(mgdl)

    def coded_term_for(self, concept: str, dictionary_version: str) -> str | None:
        """The preferred term this dictionary version uses for the concept."""
        body = self.catalog.concept(concept)
        by_version = body.coded_terms.get("by_dictionary_version") or {}
        if dictionary_version in by_version:
            terms = by_version[dictionary_version].get("pt") or []
            if terms:
                return terms[0]
        terms = body.coded_terms.get("pt") or []
        return terms[0] if terms else None

    # -- truth sampling -----------------------------------------------------

    def sample_truth(self, truth_id: str, *, kind: str = "case") -> EpisodeTruth:
        """Sample one episode of ground truth.

        ``kind`` steers the clinical picture so the corpus covers cases, review
        candidates and clear non-cases rather than only positives.
        """
        symptom_pool = sorted(
            set(self.catalog.symptom_sets["neuroglycopenic"])
            | set(self.catalog.symptom_sets["autonomic"])
        )
        onset = self.rng.randint(0, 14)
        severity_start = self._pick(["mild", "moderate", "severe"])
        steps = [(0, severity_start)]
        if self.rng.random() < 0.35:
            worse = {"mild": "moderate", "moderate": "severe", "severe": "severe"}
            steps.append((self.rng.randint(1, 3), worse[severity_start]))
        serious = severity_start == "severe" and self.rng.random() < 0.5
        criteria = (
            sorted(self.rng.sample(
                ["hospitalisation", "life_threatening", "other_medically_important"], 1
            ))
            if serious else []
        )
        outcome = self._pick(
            ["recovered", "recovered", "recovering", "not_recovered"]
        )
        action = self._pick(
            ["dose_not_changed", "dose_reduced", "drug_interrupted",
             "drug_withdrawn", "dose_reduced"]
        )

        # Some events are coded to a term that denotes the concept; others are
        # coded non-specifically ("Malaise") while the narrative plainly
        # describes hypoglycemia. Both happen, and the second is what the
        # evidence ladder below `explicit` exists to catch.
        coded_specifically = self.rng.random() < 0.55

        if kind == "case":
            glucose = float(self.rng.choice([38, 44, 48, 52, 54, 58, 62, 66, 68]))
            symptoms = sorted(self.rng.sample(symptom_pool, self.rng.choice([1, 2, 2, 3])))
        elif kind == "review":
            # Symptoms with no confirming value: the adjudication set.
            glucose = None
            symptoms = sorted(self.rng.sample(symptom_pool, self.rng.choice([1, 2])))
        elif kind == "out_of_window":
            glucose = float(self.rng.choice([44, 52, 58, 64]))
            symptoms = sorted(self.rng.sample(symptom_pool, 2))
            onset = self.rng.randint(21, 60)
        elif kind == "no_symptoms":
            glucose = float(self.rng.choice([48, 58, 66]))
            symptoms = []
        elif kind == "chronic":
            # A condition that evolves rather than recurs. Successive records
            # are the same episode changing grade, and the catalogue says so
            # via recurrence_expected: false.
            return EpisodeTruth(
                truth_id=truth_id, concept="ANAEMIA",
                onset_offset_days=onset, duration_days=self.rng.randint(6, 20),
                severity_steps=[(0, "mild"), (self.rng.randint(1, 3), "moderate")],
                seriousness=False, seriousness_criteria=[],
                relatedness=self._pick(["possible", "probable"]),
                action_taken=action, outcome=outcome, glucose_mgdl=None,
                symptoms=[], rescue_given=False, third_party_assistance=False,
                coded_specifically=True, note="chronic evolving condition",
            )
        else:  # distractor concept
            return EpisodeTruth(
                truth_id=truth_id,
                concept=self._pick(["HYPERGLYCEMIA", "NAUSEA", "HEADACHE"]),
                onset_offset_days=onset, duration_days=self.rng.randint(1, 6),
                severity_steps=steps, seriousness=serious,
                seriousness_criteria=criteria,
                relatedness=self._pick(["possible", "probable", "unlikely", "not_related"]),
                action_taken=action, outcome=outcome, glucose_mgdl=None,
                symptoms=[], rescue_given=False, third_party_assistance=False,
                note="distractor concept",
            )

        return EpisodeTruth(
            truth_id=truth_id,
            concept="HYPOGLYCEMIA",
            onset_offset_days=onset,
            duration_days=self.rng.randint(1, 5),
            severity_steps=steps,
            seriousness=serious,
            seriousness_criteria=criteria,
            relatedness=self._pick(["possible", "probable", "unlikely", "not_related"]),
            action_taken=action,
            outcome=outcome,
            glucose_mgdl=glucose,
            symptoms=symptoms,
            rescue_given=bool(symptoms) and self.rng.random() < 0.6,
            third_party_assistance=severity_start == "severe" and self.rng.random() < 0.4,
            coded_specifically=coded_specifically,
            recurrence_gap_days=(
                self.rng.randint(2, 5) if self.rng.random() < 0.22 else None
            ),
            note=kind,
        )

    # -- rendering ----------------------------------------------------------

    def render(
        self,
        truth: EpisodeTruth,
        study: StudySemantics,
        subject_id: str,
        anchor_date: _dt.date,
        corpus: GeneratedCorpus,
        episode_index: int = 0,
    ) -> dict[str, Any]:
        """Write one truth into one study's collection conventions."""
        onset = anchor_date + _dt.timedelta(days=truth.onset_offset_days)
        end = onset + _dt.timedelta(days=truth.duration_days)
        variant = study.representation

        # V-B splits a worsening event across records; every other variant is
        # one record per episode.
        recurs = True
        try:
            recurs = self.catalog.concept(truth.concept).recurrence_expected
        except Exception:
            pass
        if len(truth.severity_steps) > 1 and (variant == "V-B" or not recurs):
            # V-B splits a worsening event by convention; a chronic condition
            # is recorded as successive grade changes everywhere.
            steps = truth.severity_steps
        else:
            steps = [(0, truth.peak_severity)]

        record_ids: list[str] = []
        narrative_doc_id = f"{subject_id}-NAR-{episode_index + 1:02d}"

        for step_index, (day_offset, severity) in enumerate(steps):
            step_start = onset + _dt.timedelta(days=day_offset)
            is_last = step_index == len(steps) - 1
            seq = len(record_ids) + 1 + episode_index * 10
            source_record_id = f"{subject_id}-AE-{seq:03d}"
            record_ids.append(source_record_id)

            outcome = truth.outcome if is_last else "not_recovered"
            row = self._ae_row(
                truth=truth, study=study, subject_id=subject_id,
                source_record_id=source_record_id, seq=seq,
                severity=severity, onset=step_start,
                end=end if (is_last and outcome in TERMINAL_OUTCOMES) else None,
                outcome=outcome, is_last=is_last,
                narrative_doc_id=narrative_doc_id if step_index == 0 else None,
                # Half of the split records carry an explicit continuation
                # pointer; the rest must be linked from the study's declared
                # splitting convention, so both paths are exercised.
                continuation_of=(
                    record_ids[step_index - 1]
                    if step_index and self.rng.random() < 0.5
                    else None
                ),
            )
            corpus.tables["ae"].append(row)
            corpus.gold_records.append(
                self._gold_record(truth, study, row, severity, step_start,
                                  end if is_last else None, outcome)
            )

        # V-C puts the objective value on a linked form instead of anywhere else.
        if variant == "V-C" and truth.glucose_mgdl is not None:
            reported, canonical = self._glucose_in(study.glucose_unit, truth.glucose_mgdl)
            corpus.tables["linked_hypo_event"].append({
                "STUDYID": study.study_id, "USUBJID": subject_id,
                "FORMID": "HYPO_EVENT_FORM",
                "LNKID": f"{record_ids[0]}-HEF",
                "AESPID": record_ids[0],
                "GLUCVAL": reported, "GLUCUNIT": study.glucose_unit,
                "SYMPTOMATIC": "Y" if truth.symptoms else "N",
                "THIRDPARTY": "Y" if truth.third_party_assistance else "N",
                "RESCUE": "Y" if truth.rescue_given else "N",
                "HEFDTC": onset.isoformat(),
                "SYNTHETIC": SYNTHETIC_FLAG,
            })

        narrative = self._narrative(truth, study, subject_id, narrative_doc_id, onset)
        corpus.narratives.append(narrative)

        # Routine laboratory surveillance, plus the event value where the study
        # records glucose in the lab domain rather than on a form.
        if truth.glucose_mgdl is not None and variant != "V-C":
            reported, canonical = self._glucose_in(study.glucose_unit, truth.glucose_mgdl)
            corpus.tables["lb"].append(
                self._lb_row(study, subject_id, reported, canonical, onset)
            )

        return {
            "record_ids": record_ids,
            "narrative_doc_id": narrative_doc_id,
            "onset": onset,
            "end": end,
        }

    def _ae_row(
        self, *, truth: EpisodeTruth, study: StudySemantics, subject_id: str,
        source_record_id: str, seq: int, severity: str, onset: _dt.date,
        end: _dt.date | None, outcome: str, is_last: bool,
        narrative_doc_id: str | None, continuation_of: str | None,
    ) -> dict[str, Any]:
        """One AE row, blanked wherever this study did not collect the field."""

        def emit(field_name: str, value: Any) -> Any:
            if not study.collects(field_name):
                return ""
            # A small share of collected fields are simply not answered by the
            # site. That is `unknown` — asked, unanswered, reason unrecorded —
            # and it is distinct from every other kind of blank.
            if field_name in ("relatedness", "outcome") and self.rng.random() < 0.05:
                return ""
            return value

        coded = ""
        if study.collects("coded_term") and truth.coded_by_study:
            coded = (
                self.coded_term_for(truth.concept, study.dictionary_version)
                if truth.coded_specifically
                else self._pick(NON_SPECIFIC_CODED_TERMS)
            ) or ""

        # A restricted codelist cannot express every clinical concept. Where it
        # cannot, the cell is left empty and the semantics layer resolves it to
        # `not_representable` rather than substituting a nearby code.
        action_value = ""
        codelist = study.codelist_for("action_taken")
        if study.collects("action_taken") and truth.action_taken:
            if codelist is None or truth.action_taken in codelist.permissible:
                action_value = truth.action_taken

        serious = truth.seriousness and is_last
        return {
            "STUDYID": study.study_id,
            "USUBJID": subject_id,
            "AESEQ": seq,
            "AESPID": source_record_id,
            "AETERM": emit("verbatim_term", self._pick(VERBATIM_TERMS)
                           if truth.concept == "HYPOGLYCEMIA"
                           else truth.concept.lower()),
            "AEDECOD": coded,
            "AEDICTVER": study.dictionary_version if coded else "",
            "AESTDTC": emit("onset_datetime", onset.isoformat()),
            "AEENDTC": emit("end_datetime", end.isoformat() if end else ""),
            "AESEV": emit("severity", severity),
            "AESER": emit("seriousness", "Y" if serious else "N"),
            "AESCAT": "|".join(truth.seriousness_criteria) if serious else "",
            "AEREL": emit("relatedness", truth.relatedness),
            "AEACN": action_value,
            "AEOUT": emit("outcome", outcome),
            "AELNKID": f"{source_record_id}-HEF" if study.representation == "V-C" else "",
            "AECONTRP": continuation_of or "",
            "DOCID": narrative_doc_id or "",
            "SYNTHETIC": SYNTHETIC_FLAG,
        }

    def _gold_record(
        self, truth: EpisodeTruth, study: StudySemantics, row: dict[str, Any],
        severity: str, onset: _dt.date, end: _dt.date | None, outcome: str,
    ) -> dict[str, Any]:
        """The true field values and true collection states for one record."""
        states: dict[str, str] = {}
        values: dict[str, Any] = {}

        def note(name: str, value: Any, cell: Any) -> None:
            # The answer key asks the resolver what a blank means here rather
            # than restating the rule; if the two could disagree, the metric
            # would be measuring the duplication, not the pipeline.
            values[name] = value if cell not in (None, "") else None
            states[name] = (
                "collected" if cell not in (None, "") else study.state_for_blank(name)
            )

        note("verbatim_term", row["AETERM"] or None, row["AETERM"])
        note("coded_term", row["AEDECOD"] or None, row["AEDECOD"])
        note("severity", severity, row["AESEV"])
        note("relatedness", truth.relatedness, row["AEREL"])
        note("outcome", outcome, row["AEOUT"])
        note("onset_datetime", onset.isoformat(), row["AESTDTC"])

        # End date is gated on the outcome being terminal — as *recorded*, not
        # as it truly was. Where the outcome cell is itself blank, nothing in
        # the record says whether the event has ended, so the end date is
        # `unknown` rather than `pending_ongoing`. The answer key must not
        # claim a state the record cannot support.
        recorded_outcome = row["AEOUT"]
        if row["AEENDTC"]:
            values["end_datetime"] = row["AEENDTC"]
            states["end_datetime"] = "collected"
        elif not study.collects("end_datetime"):
            values["end_datetime"] = None
            states["end_datetime"] = study.state_for_blank("end_datetime")
        elif recorded_outcome and recorded_outcome not in TERMINAL_OUTCOMES:
            values["end_datetime"] = None
            states["end_datetime"] = "pending_ongoing"
        else:
            values["end_datetime"] = None
            states["end_datetime"] = "unknown"

        # Action taken: collected, never asked, or not expressible here.
        codelist = study.codelist_for("action_taken")
        if row["AEACN"]:
            values["action_taken"] = row["AEACN"]
            states["action_taken"] = "collected"
        elif not study.collects("action_taken"):
            values["action_taken"] = None
            states["action_taken"] = study.state_for_blank("action_taken")
        elif (
            truth.action_taken
            and codelist is not None
            and truth.action_taken in codelist.absent_concepts
        ):
            values["action_taken"] = None
            states["action_taken"] = "not_representable"
        else:
            values["action_taken"] = None
            states["action_taken"] = "unknown"

        # Seriousness criteria are gated on the seriousness answer.
        gate = row["AESER"] == "Y"
        for criterion in SERIOUSNESS_CRITERIA:
            name = f"seriousness_criteria.{criterion}"
            if not study.collects("seriousness"):
                states[name] = study.state_for_blank("seriousness")
                values[name] = None
            elif not gate:
                states[name] = "not_applicable_gated"
                values[name] = None
            else:
                states[name] = "collected"
                values[name] = criterion in truth.seriousness_criteria

        # What the narrative states, and therefore what the model path can
        # legitimately recover. Compared against `values`, which is only what
        # the structured cell holds, this is the difference between "the CRF
        # did not record it" and "the record does not say".
        narrated: dict[str, Any] = {}
        detail = study.narrative_detail
        if detail == "rich":
            narrated["severity"] = severity
            narrated["relatedness"] = truth.relatedness
            narrated["outcome"] = outcome
            if truth.action_taken:
                narrated["action_taken"] = self._narrated_action(truth.action_taken)
        elif detail == "standard":
            narrated["outcome"] = outcome

        recoverable = {
            name: (values.get(name) if values.get(name) is not None
                   else narrated.get(name))
            for name in set(values) | set(narrated)
        }
        return {
            "source_record_id": row["AESPID"],
            "study_id": study.study_id,
            "subject_id": row["USUBJID"],
            "truth_id": truth.truth_id,
            "representation": study.representation,
            "values": values,
            "narrated_values": narrated,
            "recoverable_values": recoverable,
            "collection_states": states,
        }

    @staticmethod
    def _narrated_action(action: str) -> str:
        """The action a narrative sentence conveys.

        The prose for a dose reduction reads as a dose reduction wherever it
        appears, including in a study whose codelist cannot express one.
        """
        return action

    def _narrative(
        self, truth: EpisodeTruth, study: StudySemantics, subject_id: str,
        doc_id: str, onset: _dt.date,
    ) -> dict[str, Any]:
        header = (
            f"{NARRATIVE_HEADER}\n"
            f"Study {study.study_id} | Subject {subject_id} | Narrative {doc_id} "
            f"| representation {study.representation} | synthetic"
        )
        builder = NarrativeBuilder(header)
        surface = self._pick(CONCEPT_SURFACES.get(truth.concept, [truth.concept.lower()]))
        detail = study.narrative_detail

        builder.sentence(
            f"Subject {subject_id} in study {study.study_id} was receiving study "
            f"drug per protocol."
        )
        builder.marked(
            "The subject experienced ", surface,
            f" {truth.onset_offset_days} days after the dose escalation.", "concept",
        )

        if truth.symptoms:
            builder.sentence(
                f"The subject reported {self._symptom_phrase(truth.symptoms)}."
            )
        elif truth.concept == "HYPOGLYCEMIA":
            # Stating the negative is what makes "asymptomatic" recoverable.
            # Without it, an empty symptom list is merely undocumented.
            builder.sentence("The subject reported no associated symptoms.")
        # V-C keeps the objective value on the linked form and out of the text;
        # V-D is the study where the narrative is the only place it appears.
        if truth.glucose_mgdl is not None and study.representation != "V-C":
            reported, _canonical = self._glucose_in(study.glucose_unit, truth.glucose_mgdl)
            builder.sentence(
                f"Capillary glucose was {reported} {study.glucose_unit}."
            )
        if truth.rescue_given:
            builder.sentence("Oral glucose gel was administered.")
        if detail == "rich":
            # The study that collected almost nothing structurally puts
            # severity, causality, action and outcome in prose instead.
            builder.sentence(
                f"The event was graded as {truth.peak_severity} in intensity."
            )
            if truth.action_taken:
                builder.sentence(ACTION_PHRASES[truth.action_taken])
            builder.sentence(RELATEDNESS_PHRASES[truth.relatedness])
            builder.sentence(OUTCOME_PHRASES[truth.outcome])
        elif detail == "standard":
            builder.sentence(OUTCOME_PHRASES[truth.outcome])
        for criterion in truth.seriousness_criteria:
            builder.sentence(CRITERION_PHRASES[criterion])

        # Assertion-bearing asides. These are where assertion actually matters:
        # a coded AE row asserts presence by construction, but a narrative can
        # mention a concept in order to rule it out.
        aside_assertion = None
        if self.rng.random() < 0.30:
            aside_assertion = self._pick(["absent", "hypothetical", "historical"])
            pool = {
                "absent": NEGATED_ASIDES,
                "hypothetical": HYPOTHETICAL_ASIDES,
                "historical": HISTORICAL_ASIDES,
            }[aside_assertion]
            other = self._pick(CONCEPT_SURFACES["HYPOGLYCEMIA"])
            template = self._pick(pool)
            builder.sentence(
                template.format(concept=other, concept_cap=other.capitalize())
            )

        return {
            "doc_id": doc_id,
            "study_id": study.study_id,
            "subject_id": subject_id,
            "truth_id": truth.truth_id,
            "header": header,
            "text": builder.body(),
            "concept_mention": builder.mark("concept"),
            "aside_assertion": aside_assertion,
        }

    def _lb_row(
        self, study: StudySemantics, subject_id: str, reported: float,
        canonical: float, when: _dt.date,
    ) -> dict[str, Any]:
        seq = self._lb_seq.get(subject_id, 0) + 1
        self._lb_seq[subject_id] = seq
        return {
            "STUDYID": study.study_id, "USUBJID": subject_id, "LBSEQ": seq,
            "LBTESTCD": "GLUC", "LBTEST": "Glucose", "LBORRES": reported,
            "LBORRESU": study.glucose_unit, "LBSTRESN": canonical,
            "LBSTRESU": "mg/dL", "LBDTC": when.isoformat(),
            "SYNTHETIC": SYNTHETIC_FLAG,
        }

    # -- subject scaffolding ------------------------------------------------

    def _subject(
        self, study: StudySemantics, subject_id: str, corpus: GeneratedCorpus,
        start: _dt.date,
    ) -> _dt.date:
        """Emit DM and EX rows; return the dose-escalation date."""
        corpus.tables["dm"].append({
            "STUDYID": study.study_id, "USUBJID": subject_id,
            "AGE": self.rng.randint(38, 79), "SEX": self._pick(["M", "F"]),
            "ARM": self._pick(["Study drug", "Study drug", "Placebo"]),
            "RFSTDTC": start.isoformat(),
            "COUNTRY": self._pick(["USA", "GBR", "DEU", "JPN", "BRA"]),
            "SYNTHETIC": SYNTHETIC_FLAG,
        })
        dose = self._pick([5.0, 10.0, 20.0])
        escalation = start + _dt.timedelta(days=self.rng.randint(21, 84))
        for seq, (when, amount) in enumerate(
            [(start, dose), (escalation, dose * 2)], start=1
        ):
            corpus.tables["ex"].append({
                "STUDYID": study.study_id, "USUBJID": subject_id, "EXSEQ": seq,
                "EXTRT": "Study drug", "EXDOSE": amount, "EXDOSU": "mg",
                "EXSTDTC": when.isoformat(), "EXENDTC": "",
                "SYNTHETIC": SYNTHETIC_FLAG,
            })
        return escalation

    # -- top level ----------------------------------------------------------

    def generate(self) -> GeneratedCorpus:
        corpus = GeneratedCorpus(tables={
            "dm": [], "ae": [], "ex": [], "lb": [], "linked_hypo_event": []
        })
        studies = [self.semantics.for_study(s) for s in self.study_ids]

        # 1. The invariance cohort: every truth rendered in every study, so the
        #    harness can ask whether representation changes the conclusion.
        for index in range(self.invariance_truths):
            kind = ["case", "case", "case", "review", "out_of_window", "no_symptoms"][
                index % 6
            ]
            truth = self.sample_truth(f"T{index + 1:04d}", kind=kind)
            for study in studies:
                subject_id = f"{study.study_id}-INV-{index + 1:03d}"
                # V-D never codes the term; that is the point of the variant.
                rendered_truth = EpisodeTruth(**{
                    **asdict(truth),
                    "coded_by_study": study.collects("coded_term"),
                })
                start = _dt.date(2019, 1, 7) + _dt.timedelta(
                    days=self.rng.randint(0, 300)
                )
                anchor = self._subject(study, subject_id, corpus, start)
                emitted = self.render(
                    rendered_truth, study, subject_id, anchor, corpus
                )
                corpus.gold_episodes.append(
                    self._gold_episode(truth, rendered_truth, study, subject_id,
                                       emitted, anchor, episode_index=0)
                )
                if truth.recurrence_gap_days is not None:
                    # A second, distinct episode a few days later. For a concept
                    # where recurrence is expected these must stay separate.
                    second = EpisodeTruth(**{
                        **asdict(rendered_truth),
                        "truth_id": f"{truth.truth_id}R",
                        "onset_offset_days": (
                            truth.onset_offset_days + truth.duration_days
                            + truth.recurrence_gap_days
                        ),
                        "recurrence_gap_days": None,
                    })
                    emitted2 = self.render(
                        second, study, subject_id, anchor, corpus, episode_index=1
                    )
                    corpus.gold_episodes.append(
                        self._gold_episode(second, second, study, subject_id,
                                           emitted2, anchor, episode_index=1)
                    )
            corpus.truths.append({
                **asdict(truth),
                "true_evidence_state": truth.true_evidence_state(),
                "true_verdict": truth.true_verdict(),
                "rendered_in": [s.study_id for s in studies],
                "cohort": "invariance",
            })

        # 2. Study-local background subjects, so the corpus is not composed
        #    solely of the invariance cohort.
        counter = 0
        for study in studies:
            for index in range(self.background_per_study):
                counter += 1
                kind = self._pick(
                    ["case", "case", "review", "out_of_window", "no_symptoms",
                     "distractor", "distractor", "chronic"]
                )
                truth = self.sample_truth(f"B{counter:04d}", kind=kind)
                truth.coded_by_study = study.collects("coded_term")
                subject_id = f"{study.study_id}-{index + 1:03d}"
                start = _dt.date(2019, 1, 7) + _dt.timedelta(
                    days=self.rng.randint(0, 300)
                )
                anchor = self._subject(study, subject_id, corpus, start)
                emitted = self.render(truth, study, subject_id, anchor, corpus)
                corpus.gold_episodes.append(
                    self._gold_episode(truth, truth, study, subject_id, emitted,
                                       anchor, episode_index=0)
                )
                corpus.truths.append({
                    **asdict(truth),
                    "true_evidence_state": truth.true_evidence_state(),
                    "true_verdict": truth.true_verdict(),
                    "rendered_in": [study.study_id],
                    "cohort": "background",
                })

        corpus.manifest = self._manifest(studies, corpus)
        return corpus

    def _gold_episode(
        self, base_truth: EpisodeTruth, rendered: EpisodeTruth,
        study: StudySemantics, subject_id: str, emitted: dict[str, Any],
        anchor: _dt.date, episode_index: int,
    ) -> dict[str, Any]:
        return {
            "truth_id": rendered.truth_id,
            "base_truth_id": base_truth.truth_id,
            "study_id": study.study_id,
            "subject_id": subject_id,
            "representation": study.representation,
            "episode_index": episode_index,
            "concept": rendered.concept,
            "source_record_ids": emitted["record_ids"],
            "n_records": len(emitted["record_ids"]),
            "episode_start": emitted["onset"].isoformat(),
            "episode_end": emitted["end"].isoformat(),
            "onset_offset_days": rendered.onset_offset_days,
            "peak_severity": rendered.peak_severity,
            "coded_specifically": rendered.coded_specifically,
            "coded_by_study": rendered.coded_by_study,
            "state_as_recorded": rendered.true_evidence_state(as_recorded=True),
            "verdict_as_recorded": rendered.true_verdict(as_recorded=True),
            "true_evidence_state": base_truth.true_evidence_state(),
            "true_verdict": base_truth.true_verdict(),
            "representation_independent_verdict": base_truth.true_verdict(),
        }

    def _manifest(
        self, studies: list[StudySemantics], corpus: GeneratedCorpus
    ) -> dict[str, Any]:
        return {
            "generator": "aelayer.generate",
            "seed": self.seed,
            "synthetic": True,
            "notice": (
                "All records are computer generated. No real patient data is "
                "present in this repository."
            ),
            "gold_case_definition": "te_symptomatic_hypoglycemia.v1",
            "invariance_truths": self.invariance_truths,
            "studies": {
                s.study_id: {
                    "label": s.label, "representation": s.representation,
                    "dictionary_version": s.dictionary_version,
                    "glucose_unit": s.glucose_unit,
                    "record_splitting": s.record_splitting,
                    "linked_forms": list(s.linked_forms),
                    "narrative_detail": s.narrative_detail,
                    "note": s.note,
                }
                for s in studies
            },
            "counts": {
                "studies": len(studies),
                "subjects": len(corpus.tables["dm"]),
                "ae_records": len(corpus.tables["ae"]),
                "episodes_expected": len(corpus.gold_episodes),
                "narratives": len(corpus.narratives),
            },
        }


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

TABLE_COLUMNS: dict[str, list[str]] = {
    "dm": ["STUDYID", "USUBJID", "AGE", "SEX", "ARM", "RFSTDTC", "COUNTRY",
           "SYNTHETIC"],
    "ae": ["STUDYID", "USUBJID", "AESEQ", "AESPID", "AETERM", "AEDECOD",
           "AEDICTVER", "AESTDTC", "AEENDTC", "AESEV", "AESER", "AESCAT",
           "AEREL", "AEACN", "AEOUT", "AELNKID", "AECONTRP", "DOCID",
           "SYNTHETIC"],
    "ex": ["STUDYID", "USUBJID", "EXSEQ", "EXTRT", "EXDOSE", "EXDOSU",
           "EXSTDTC", "EXENDTC", "SYNTHETIC"],
    "lb": ["STUDYID", "USUBJID", "LBSEQ", "LBTESTCD", "LBTEST", "LBORRES",
           "LBORRESU", "LBSTRESN", "LBSTRESU", "LBDTC", "SYNTHETIC"],
    "linked_hypo_event": ["STUDYID", "USUBJID", "FORMID", "LNKID", "AESPID",
                          "GLUCVAL", "GLUCUNIT", "SYMPTOMATIC", "THIRDPARTY",
                          "RESCUE", "HEFDTC", "SYNTHETIC"],
}


def write_corpus(corpus: GeneratedCorpus, out_dir: str | Path | None = None) -> Path:
    root = Path(out_dir or paths.DATA_DIR)
    root.mkdir(parents=True, exist_ok=True)
    for table, columns in TABLE_COLUMNS.items():
        with (root / f"{table}.csv").open("w", encoding="utf-8", newline="\n") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            for row in corpus.tables.get(table, []):
                writer.writerow({c: row.get(c, "") for c in columns})

    _write_jsonl(root / "narratives.jsonl", corpus.narratives)
    _write_jsonl(root / "truths.jsonl", corpus.truths)
    _write_jsonl(root / "gold_records.jsonl", corpus.gold_records)
    _write_jsonl(root / "gold_episodes.jsonl", corpus.gold_episodes)
    (root / "manifest.json").write_text(
        json.dumps(corpus.manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "README.txt").write_text(
        "SYNTHETIC DATA ONLY\n"
        "===================\n\n"
        "Every file here is computer generated by aelayer.generate. No real\n"
        "patient data, and nothing derived from real patients, is present.\n\n"
        "truths.jsonl holds the sampled ground truth; gold_records.jsonl and\n"
        "gold_episodes.jsonl hold the answer key. All three are read only by\n"
        "the evaluation harness.\n",
        encoding="utf-8",
    )
    return root


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def generate_corpus(
    seed: int = 7,
    n_studies: int = 6,
    out_dir: str | Path | None = None,
    invariance_truths: int = 24,
    background_per_study: int = 18,
) -> tuple[Path, dict[str, Any]]:
    generator = CorpusGenerator(
        seed=seed, n_studies=n_studies, invariance_truths=invariance_truths,
        background_per_study=background_per_study,
    )
    corpus = generator.generate()
    return write_corpus(corpus, out_dir), corpus.manifest
