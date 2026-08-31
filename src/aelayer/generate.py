"""Synthetic corpus generator.

Produces SDTM-shaped tables, one case narrative per adverse event record, and a
gold answer key.  The corpus is what makes evaluation possible: every narrative
is assembled from a known intent, and that intent is written out alongside it.

Nothing here is derived from real patients.  Every table carries a ``SYNTHETIC``
column and every narrative carries a synthetic header.

The gold labels are the generator's *intent*, not the extractor's output.  A
metric computed against them therefore measures whether the pipeline recovers a
signal that was deliberately planted, which is a weaker claim than performance
on real clinical text.  The README says so plainly.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import paths
from .catalog import ConceptCatalog, load_configs
from .models import EVIDENCE_STATE_RANK

SYNTHETIC_FLAG = "Y"
NARRATIVE_HEADER = (
    "*** SYNTHETIC RECORD - COMPUTER GENERATED - NOT REAL PATIENT DATA ***"
)

# --------------------------------------------------------------------------
# Study-level conventions.  Studies differ on purpose: a system that only works
# when every trial reports the same way has not solved the problem it claims to.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StudyConvention:
    study_id: str
    label: str
    glucose_unit: str
    dictionary_version: str
    codes_hypoglycemia: bool
    severe_events_only: bool
    n_subjects: int
    start: _dt.date
    note: str
    #: How completely the AE table itself is populated. Older studies leave
    #: causality, action and outcome to the narrative.
    structured_ae_detail: str = "full"   # full | partial | minimal


STUDY_CONVENTIONS: list[StudyConvention] = [
    StudyConvention(
        "STUDY-01", "Cardiometabolic phase 3, region A", "mg/dL", "MedDRA 26.0",
        True, False, 62, _dt.date(2019, 3, 4),
        "Conventional coding, US units.",
    ),
    StudyConvention(
        "STUDY-02", "Cardiometabolic phase 3, region B", "mmol/L", "MedDRA 24.1",
        True, False, 58, _dt.date(2017, 9, 11),
        "SI units throughout; a threshold rule that ignores units misreads this study.",
    ),
    StudyConvention(
        "STUDY-03", "Phase 2b dose ranging", "mg/dL", "MedDRA 25.0",
        True, True, 44, _dt.date(2020, 6, 15),
        "Protocol collected severe adverse events only; milder events survive "
        "solely as narrative context.",
    ),
    StudyConvention(
        "STUDY-04", "Legacy phase 3 extension", "mmol/L", "MedDRA 21.1",
        False, False, 71, _dt.date(2015, 1, 20),
        "No coded term for hypoglycemia was ever applied in this study; every "
        "event is recoverable only from narrative.",
        structured_ae_detail="minimal",
    ),
    StudyConvention(
        "STUDY-05", "Phase 3 open-label", "mg/dL", "MedDRA 27.0",
        True, False, 55, _dt.date(2021, 11, 2),
        "Recent conventions, mixed phrasing.",
    ),
    StudyConvention(
        "STUDY-06", "Phase 2 crossover", "mmol/L", "MedDRA 23.0",
        True, False, 40, _dt.date(2018, 5, 7),
        "Crossover design; rechallenge language appears more often.",
        structured_ae_detail="partial",
    ),
]

NON_SPECIFIC_CODED_TERMS = [
    "Malaise", "Asthenia", "Feeling abnormal", "General physical health deterioration"
]

# --------------------------------------------------------------------------
# Narrative assembly with offset tracking
# --------------------------------------------------------------------------


class NarrativeBuilder:
    """Assembles a narrative while recording character offsets of key mentions."""

    def __init__(self, header: str):
        self.header = header
        self._parts: list[str] = []
        self._marks: dict[str, dict[str, Any]] = {}

    @property
    def _cursor(self) -> int:
        # Offsets are measured against header + "\n" + body, which is the exact
        # string the extractor sees.
        return len(self.header) + 1 + sum(len(p) for p in self._parts)

    def sentence(self, text: str) -> None:
        if self._parts:
            self._parts.append(" ")
        self._parts.append(text)

    def marked_sentence(self, before: str, mention: str, after: str, label: str) -> None:
        """Add a sentence and record where ``mention`` lands inside it."""
        if self._parts:
            self._parts.append(" ")
        start = self._cursor + len(before)
        self._parts.append(before + mention + after)
        self._marks[label] = {
            "text": mention,
            "start": start,
            "end": start + len(mention),
        }

    def mark(self, label: str) -> dict[str, Any] | None:
        return self._marks.get(label)

    def body(self) -> str:
        return "".join(self._parts)

    def full_text(self) -> str:
        return f"{self.header}\n{self.body()}"


# --------------------------------------------------------------------------
# Surface-form pools
# --------------------------------------------------------------------------

CONCEPT_SURFACES_US = [
    "hypoglycemia", "hypoglycemic episode", "low blood sugar", "low blood glucose",
]
CONCEPT_SURFACES_GB = [
    "hypoglycaemia", "hypoglycaemic episode", "low blood sugar",
]
MISSPELLINGS = [
    "hypoglycemai", "hypoglyceemia", "hypoglycaema", "hypogycemia", "hypoglcyemia",
]

ONSET_PHRASINGS: dict[str, list[str]] = {
    "relative_after": [
        "{n} {unit} after the {anchor}",
        "{n} {unit} following the {anchor}",
        "{n} {unit} post {anchor}",
    ],
    "relative_within": ["within {n} {unit} of the {anchor}"],
    "relative_later": ["{n} {unit} later"],
    "day_after": ["the day after the {anchor}"],
    "study_day": ["on study day {study_day}"],
    "absolute_date": ["on {date_dmy}"],
    "vague": ["several days after the {anchor}", "a few days after the {anchor}"],
    "same_day": ["on the day of the {anchor}"],
}

#: Noun-phrase surfaces, safe after "the subject reported ...". The adjectival
#: variants in concepts.yaml are exercised by a separate sentence template so
#: the narratives stay grammatical while still covering both surface forms.
SYMPTOM_NOUN_FORMS: dict[str, list[str]] = {
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

#: Adjectival surfaces, used only after "the subject was ...".
SYMPTOM_ADJ_FORMS: dict[str, list[str]] = {
    "confusion": ["confused", "disoriented"],
    "dizziness": ["dizzy"],
    "lightheadedness": ["lightheaded", "light-headed"],
    "drowsiness": ["drowsy"],
    "diaphoresis": ["diaphoretic"],
    "sweating": ["sweaty"],
    "shakiness": ["shaky"],
    "tremor": ["tremulous"],
    "pallor": ["pale"],
    "clamminess": ["clammy"],
    "hunger": ["hungry"],
    "anxiety": ["anxious"],
}

ANCHOR_SURFACES = [
    "dose escalation", "dose increase", "uptitration", "dose titration",
]

NUMBER_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
    7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
}

LAB_PHRASES = [
    "Capillary glucose was {value} {unit}.",
    "Fingerstick glucose {value} {unit} was recorded at the time of the event.",
    "Plasma glucose measured {value} {unit}.",
    "A blood glucose of {value} {unit} was documented.",
]

SYMPTOM_PHRASES = [
    "The subject reported {symptoms}.",
    "The subject described {symptoms}.",
    "{symptoms_cap} {was_were} noted by the site nurse.",
    "On examination the subject had {symptoms}.",
]
SYMPTOM_ADJ_PHRASES = [
    "The subject was {adj} at the time of the assessment.",
    "The site nurse recorded that the subject appeared {adj}.",
]

RESCUE_PHRASES = [
    "Oral glucose gel was administered.",
    "The subject was given orange juice as rescue carbohydrate.",
    "Glucose tablets were given by the study nurse.",
    "IV dextrose was administered in the clinic.",
]

ACTION_PHRASES = {
    "dose_reduced": "The dose was reduced at the next scheduled visit.",
    "dose_interrupted": "Study drug was held for 48 hours.",
    "drug_withdrawn": "The subject was permanently discontinued from study drug.",
    "none": "No dose change to study drug was made.",
}

OUTCOME_PHRASES = {
    "resolved": "The event resolved the same day.",
    "resolving": "The event was resolving at the time of the last assessment.",
    "not_resolved": "The event was ongoing at the end of the reporting period.",
    "unknown": "The outcome was not recorded.",
}

RELATEDNESS_PHRASES = {
    "possible": "The investigator assessed the event as possibly related to study drug.",
    "probable": "The investigator assessed the event as probably related to study drug.",
    "not_related": "The investigator considered the event unrelated to study drug.",
    "unlikely": "The investigator judged the event unlikely related to study drug.",
}

RECHALLENGE_PHRASES = {
    "done_recurred": "On rechallenge the event recurred.",
    "done_no_recurrence": "The subject was rechallenged without recurrence.",
    "not_done": "Rechallenge was not performed.",
}

SERIOUSNESS_PHRASES = {
    "hospitalisation": "The subject was admitted to hospital for observation.",
    "life_threatening": "The episode was considered life threatening by the investigator.",
    "other_medically_important": "The event was regarded as medically important.",
}

SEVERITY_WORDS = {"mild": "mild", "moderate": "moderate", "severe": "severe"}

NEGATION_PHRASES = [
    "There was no evidence of {concept} during the visit.",
    "The subject denies {concept}.",
    "Screening was negative for {concept}.",
    "{concept_cap} was ruled out on review of the glucose log.",
]

HYPOTHETICAL_PHRASES = [
    "The subject was advised to report symptoms of {concept} promptly.",
    "The site was instructed to monitor for {concept} after each dose increase.",
    "Carbohydrate was supplied in case of {concept}.",
]

HISTORICAL_PHRASES = [
    "The subject has a history of {concept} prior to study entry.",
    "Past medical history includes {concept}.",
    "{concept_cap} was documented previously, before screening.",
]

FAMILY_PHRASES = [
    "The subject's mother has {concept}.",
    "Family history of {concept} was recorded at screening.",
]

UNCERTAIN_PHRASES = [
    "The presentation was concerning for {concept}, though this could not be confirmed.",
    "Possible {concept} was considered by the investigator.",
    "{concept_cap} cannot be excluded on the available information.",
]

DISTRACTOR_CONCEPTS = {
    "NAUSEA": ("Nausea", ["nausea", "feeling sick"]),
    "HEADACHE": ("Headache", ["headache", "cephalgia"]),
    "HYPERGLYCEMIA": ("Hyperglycaemia", ["hyperglycemia", "high blood sugar"]),
}

#: Distractor concepts whose surface form is itself a catalogued symptom. The
#: narrative genuinely mentions the symptom, so gold has to record it.
DISTRACTOR_SYMPTOMS = {"NAUSEA": ["nausea"]}

# Pattern mix.  Weights are chosen so every pattern appears often enough for a
# per-pattern metric to mean something, while explicit cases stay the majority
# as they are in real coded data.
PATTERN_WEIGHTS: dict[str, int] = {
    "explicit_coded": 14,
    "explicit_verbatim_only": 6,
    "explicit_british": 5,
    "explicit_misspelled": 4,
    "abbrev_gated": 5,
    "abbrev_ungated": 4,
    "lab_symptom": 9,
    "split_sentence": 5,
    "context_rescue": 6,
    "context_action": 5,
    "symptom_only": 5,
    "negated": 8,
    "hypothetical": 5,
    "historical": 5,
    "family_history": 4,
    "uncertain": 5,
    "out_of_window": 5,
    "unresolved_onset": 4,
    "distractor": 8,
}


@dataclass
class Row:
    table: str
    values: dict[str, Any]


@dataclass
class GeneratedCorpus:
    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    narratives: list[dict[str, Any]] = field(default_factory=list)
    gold: list[dict[str, Any]] = field(default_factory=list)
    gold_cases: list[dict[str, Any]] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Generator
# --------------------------------------------------------------------------


class CorpusGenerator:
    def __init__(
        self,
        seed: int = 7,
        n_studies: int = 4,
        catalog: ConceptCatalog | None = None,
        subjects_per_study: int | None = None,
    ):
        if not 1 <= n_studies <= len(STUDY_CONVENTIONS):
            raise ValueError(
                f"n_studies must be between 1 and {len(STUDY_CONVENTIONS)}"
            )
        self.seed = seed
        self.rng = random.Random(seed)
        self.conventions = STUDY_CONVENTIONS[:n_studies]
        self.subjects_per_study = subjects_per_study
        if catalog is None:
            catalog, _, _ = load_configs()
        self.catalog = catalog
        self.symptom_surfaces = catalog.symptom_lexicon
        self.neuro = catalog.symptom_sets["neuroglycopenic"]
        self.autonomic = catalog.symptom_sets["autonomic"]

    # -- helpers ------------------------------------------------------------

    def _pick(self, seq: list[Any]) -> Any:
        return seq[self.rng.randrange(len(seq))]

    def _weighted_pattern(self, convention: StudyConvention) -> str:
        weights = dict(PATTERN_WEIGHTS)
        if convention.severe_events_only:
            # Milder presentations are simply not collected as AE records here,
            # so contextual-only patterns become rarer, not absent.
            for key in ("context_rescue", "context_action", "symptom_only"):
                weights[key] = max(1, weights[key] // 3)
            weights["explicit_coded"] += 6
        if not convention.codes_hypoglycemia:
            weights["explicit_coded"] = 0
            weights["explicit_british"] = 0
            weights["explicit_verbatim_only"] += 10
            weights["lab_symptom"] += 5
        names = sorted(weights)
        population = [n for n in names for _ in range(weights[n])]
        return self._pick(population)

    @staticmethod
    def _ucfirst(text: str) -> str:
        """Capitalise the first character only.

        ``str.capitalize`` lowercases the remainder, which would corrupt an
        embedded date such as ``on 28-Apr-2019``.
        """
        return text[:1].upper() + text[1:] if text else text

    def _set_coded_term(self, intent: dict[str, Any], term: str | None, *, matches: bool) -> None:
        """Record the AE dictionary term and whether it denotes the concept.

        A study may code an event as "Malaise" while the narrative plainly
        describes hypoglycemia. That is a coded term, but it is not a coded
        term *for this concept*, and the `explicit` rule must not fire on it.
        """
        intent["coded_term"] = term
        intent["coded_term_matches_concept"] = bool(term) and matches

    def _concept_coded_term(self, concept_id: str = "HYPOGLYCEMIA") -> str:
        return self._pick(sorted(self.catalog.concept(concept_id).all_coded_terms()))

    def _symptom_phrase(self, symptoms: list[str]) -> str:
        surfaces = [self._pick(SYMPTOM_NOUN_FORMS.get(s, [s])) for s in symptoms]
        if len(surfaces) == 1:
            return surfaces[0]
        return ", ".join(surfaces[:-1]) + " and " + surfaces[-1]

    def _choose_symptoms(self, count: int = 0) -> list[str]:
        count = count or self.rng.choice([1, 1, 2, 2, 3])
        pool = sorted(set(self.neuro) | set(self.autonomic))
        return sorted(self.rng.sample(pool, min(count, len(pool))))

    def _glucose_value(self, unit: str, *, level: str = "normal") -> tuple[float, float]:
        """Return ``(reported_value, canonical_mgdl)`` for a glucose draw.

        ``level`` is ``low`` (a hypoglycaemic value), ``normal`` (a routine
        surveillance draw) or ``high`` (genuinely hyperglycaemic). A
        hyperglycemia narrative reporting 76 mg/dL would be neither realistic
        nor separable from a normal result.
        """
        if level == "low":
            mgdl = self.rng.choice([38, 44, 48, 52, 54, 58, 61, 64, 66, 68])
        elif level == "high":
            mgdl = self.rng.choice([196, 214, 238, 265, 291, 320])
        else:
            mgdl = self.rng.choice([76, 84, 91, 98, 105, 112])
        if unit == "mmol/L":
            return round(mgdl / 18.0182, 1), round(round(mgdl / 18.0182, 1) * 18.0182, 4)
        return float(mgdl), float(mgdl)

    def _onset_phrase(
        self, offset: int, anchor_surface: str, anchor_date: _dt.date, ref_start: _dt.date
    ) -> tuple[str, str, int]:
        """Pick a phrasing for an onset offset.

        Returns ``(phrase, phrasing_id, offset_actually_expressed)``.  Vague
        quantifiers express a range rather than a number: the true offset stays
        in gold, and the mismatch shows up in the per-pattern onset metric
        instead of being hidden.
        """
        onset_date = anchor_date + _dt.timedelta(days=offset)
        if offset == 0:
            # "0 days later" is not something anyone writes.
            options = ["study_day", "absolute_date", "same_day"]
        else:
            options = [
                "relative_after", "relative_within", "relative_later",
                "study_day", "absolute_date",
            ]
        if offset == 1:
            options.append("day_after")
        if 3 <= offset <= 5:
            options.append("vague")
        if offset >= 14 and offset % 7 == 0:
            options.append("relative_after")  # expressed in weeks below
        phrasing = self._pick(options)
        template = self._pick(ONSET_PHRASINGS[phrasing])
        # Weeks where the offset divides evenly, days otherwise. Trials write
        # both, and the extractor has to normalise both to days.
        if offset and offset % 7 == 0 and offset >= 7 and self.rng.random() < 0.4:
            magnitude, unit_word = offset // 7, "week"
        else:
            magnitude, unit_word = offset, "day"
        unit_word = unit_word if magnitude == 1 else unit_word + "s"
        n_text = (
            NUMBER_WORDS[magnitude]
            if magnitude in NUMBER_WORDS and self.rng.random() < 0.45
            else str(magnitude)
        )
        phrase = template.format(
            n=n_text,
            unit=unit_word,
            anchor=anchor_surface,
            study_day=(onset_date - ref_start).days + 1,
            date_dmy=onset_date.strftime("%d-%b-%Y"),
        )
        return phrase, phrasing, offset

    # -- gold state computation --------------------------------------------

    @staticmethod
    def gold_state(intent: dict[str, Any]) -> str:
        """The evidence state v1 should assign, computed from generation intent.

        This mirrors the semantics of ``te_symptomatic_hypoglycemia.v1``.  It is
        deliberately written from the intent side rather than by calling the
        evaluator, so that the harness compares two independent derivations of
        the same answer.
        """
        if intent.get("concept") != "HYPOGLYCEMIA":
            return "none"
        assertion = intent.get("assertion")
        if assertion == "absent":
            return "absent"

        if assertion in ("hypothetical", "historical", "family_history"):
            return "none"
        if assertion == "uncertain":
            # Routed to review by the assertion policy. The `explicit` rule can
            # still fire on a matching coded term, because a coded term is an
            # assertion-independent fact; the lexicon arm requires `present`.
            return "explicit" if intent.get("coded_term_matches_concept") else "none"
        # assertion == present
        if intent.get("coded_term_matches_concept") or intent.get("explicit_mention"):
            return "explicit"
        low_glucose = any(
            lab["canonical_mgdl"] < 70 for lab in intent.get("labs", [])
            if lab["test"] == "GLUCOSE"
        )
        has_symptom = bool(intent.get("symptoms"))
        if low_glucose and has_symptom:
            return "supported"
        if has_symptom and (
            intent.get("rescue_treatment")
            or intent.get("action_taken")
            in ("dose_reduced", "dose_interrupted", "drug_withdrawn")
        ):
            return "possible"
        return "none"

    @staticmethod
    def gold_verdict(intent: dict[str, Any], state: str) -> str:
        """The v1 verdict for one record, from intent."""
        if intent.get("concept") != "HYPOGLYCEMIA":
            return "excluded"
        assertion = intent.get("assertion")
        if assertion in ("absent", "hypothetical", "historical", "family_history"):
            return "excluded"
        if intent.get("onset_offset_days") is None:
            return "review"  # window.on_unresolved_onset: review
        if not (0 <= intent["onset_offset_days"] <= 14):
            return "excluded"
        if assertion == "uncertain":
            return "review"  # assertion.route_to_review
        if state in ("explicit", "supported"):
            return "case"
        if state == "possible":
            return "review"
        return "excluded"

    # -- record builders ----------------------------------------------------

    def _build_record(
        self,
        convention: StudyConvention,
        subject_id: str,
        ae_seq: int,
        anchor_date: _dt.date,
        ref_start: _dt.date,
    ) -> dict[str, Any]:
        """Build one AE record, its narrative, and its gold block."""
        pattern = self._weighted_pattern(convention)
        doc_id = f"{subject_id}-NAR-{ae_seq:02d}"
        header = (
            f"{NARRATIVE_HEADER}\n"
            f"Study {convention.study_id} | Subject {subject_id} | "
            f"Narrative {doc_id} | synthetic"
        )
        builder = NarrativeBuilder(header)
        unit = convention.glucose_unit
        anchor_surface = self._pick(ANCHOR_SURFACES)

        intent: dict[str, Any] = {
            "pattern": pattern,
            "concept": "HYPOGLYCEMIA",
            "assertion": "present",
            "coded_term": None,
            "coded_term_matches_concept": False,
            "coded_term_version": convention.dictionary_version,
            "explicit_mention": False,
            "onset_expressed_in_text": False,
            "onset_in_ae_table": False,
            "suppress_extra_rescue": False,
            "symptoms": [],
            "labs": [],
            "onset_offset_days": None,
            "onset_phrasing": None,
            "anchor_event": "dose_escalation",
            "severity": None,
            "seriousness": [],
            "relatedness": None,
            "action_taken": None,
            "rechallenge": None,
            "rescue_treatment": False,
            "outcome": None,
        }
        extra_rows: list[Row] = []

        builder.sentence(
            f"Subject {subject_id} in study {convention.study_id} was receiving "
            f"study drug per protocol."
        )

        handler = getattr(self, f"_pattern_{pattern}")
        handler(
            builder=builder,
            intent=intent,
            convention=convention,
            subject_id=subject_id,
            anchor_date=anchor_date,
            anchor_surface=anchor_surface,
            ref_start=ref_start,
            unit=unit,
            extra_rows=extra_rows,
        )

        self._add_management(builder, intent, convention)

        state = self.gold_state(intent)
        verdict = self.gold_verdict(intent, state)

        onset_date = (
            anchor_date + _dt.timedelta(days=intent["onset_offset_days"])
            if intent["onset_offset_days"] is not None
            else None
        )
        # The AE table carries the onset date only where the study recorded it.
        table_onset_date = onset_date if intent["onset_in_ae_table"] else None

        ae_row = self._ae_row(
            convention, subject_id, ae_seq, doc_id, intent, table_onset_date
        )
        for lab in intent["labs"]:
            extra_rows.append(
                Row(
                    "lb",
                    self._lb_row(
                        convention, subject_id, len(extra_rows) + 1, lab,
                        onset_date or anchor_date,
                    ),
                )
            )
        if intent["rescue_treatment"]:
            extra_rows.append(
                Row(
                    "cm",
                    {
                        "STUDYID": convention.study_id,
                        "USUBJID": subject_id,
                        "CMSEQ": ae_seq,
                        "CMTRT": self._pick(["Glucose gel", "Dextrose 50%", "Glucagon", "Orange juice"]),
                        "CMINDC": "Rescue for low blood glucose",
                        "CMSTDTC": (onset_date or anchor_date).isoformat(),
                        "SYNTHETIC": SYNTHETIC_FLAG,
                    },
                )
            )

        gold = {
            "doc_id": doc_id,
            "study_id": convention.study_id,
            "subject_id": subject_id,
            "ae_seq": ae_seq,
            "pattern": pattern,
            "concept": intent["concept"],
            "assertion": intent["assertion"] if intent["concept"] else None,
            "coded_term": intent["coded_term"],
            "coded_term_version": intent["coded_term_version"] if intent["coded_term"] else None,
            "symptoms": sorted(intent["symptoms"]),
            "labs": intent["labs"],
            "onset_offset_days": intent["onset_offset_days"],
            "onset_phrasing": intent["onset_phrasing"],
            "onset_date": onset_date.isoformat() if onset_date else None,
            "anchor_event": intent["anchor_event"] if intent["onset_offset_days"] is not None else None,
            "severity": intent["severity"],
            "seriousness": sorted(intent["seriousness"]),
            "relatedness": intent["relatedness"],
            "action_taken": intent["action_taken"],
            "rechallenge": intent["rechallenge"],
            "rescue_treatment": intent["rescue_treatment"],
            "outcome": intent["outcome"],
            "concept_mention": builder.mark("concept"),
            "evidence_state": state,
            "verdict": verdict,
        }

        return {
            "ae_row": ae_row,
            "extra_rows": extra_rows,
            "narrative": {
                "doc_id": doc_id,
                "study_id": convention.study_id,
                "subject_id": subject_id,
                "ae_seq": ae_seq,
                "header": header,
                "text": builder.body(),
            },
            "gold": gold,
        }

    # -- individual patterns ------------------------------------------------

    def _onset(self, builder, intent, anchor_date, anchor_surface, ref_start, *, offset=None):
        offset = offset if offset is not None else self.rng.randint(0, 14)
        phrase, phrasing, _ = self._onset_phrase(offset, anchor_surface, anchor_date, ref_start)
        intent["onset_offset_days"] = offset
        intent["onset_phrasing"] = phrasing
        intent["onset_expressed_in_text"] = True
        # AESTDTC is recorded about half the time. Where it is missing the onset
        # is recoverable only from the narrative, which is the case the temporal
        # extractor exists for; where both are present they must agree.
        intent["onset_in_ae_table"] = self.rng.random() < 0.5
        return phrase

    def _concept_sentence(self, builder, intent, surface, onset_phrase, severity=None):
        prefix = "The subject experienced "
        if severity:
            prefix += f"{SEVERITY_WORDS[severity]} "
            intent["severity"] = severity
        builder.marked_sentence(prefix, surface, f" {onset_phrase}.", "concept")
        intent["explicit_mention"] = True

    def _pattern_explicit_coded(self, builder, intent, convention, subject_id,
                                anchor_date, anchor_surface, ref_start, unit, extra_rows):
        surface = self._pick(CONCEPT_SURFACES_US)
        onset = self._onset(builder, intent, anchor_date, anchor_surface, ref_start)
        severity = self._severity_for(convention)
        self._concept_sentence(builder, intent, surface, onset, severity)
        self._set_coded_term(intent, self._concept_coded_term(), matches=True)
        if self.rng.random() < 0.6:
            self._add_symptoms(builder, intent)
        if self.rng.random() < 0.5:
            self._add_lab(builder, intent, unit, level="low")

    def _pattern_explicit_verbatim_only(self, builder, intent, convention, subject_id,
                                        anchor_date, anchor_surface, ref_start, unit, extra_rows):
        surface = self._pick(CONCEPT_SURFACES_US + CONCEPT_SURFACES_GB)
        onset = self._onset(builder, intent, anchor_date, anchor_surface, ref_start)
        self._concept_sentence(builder, intent, surface, onset, self._severity_for(convention))
        if not convention.codes_hypoglycemia:
            self._set_coded_term(
                intent, self._pick(NON_SPECIFIC_CODED_TERMS), matches=False
            )
        self._add_symptoms(builder, intent)

    def _pattern_explicit_british(self, builder, intent, convention, subject_id,
                                  anchor_date, anchor_surface, ref_start, unit, extra_rows):
        surface = self._pick(CONCEPT_SURFACES_GB)
        onset = self._onset(builder, intent, anchor_date, anchor_surface, ref_start)
        self._concept_sentence(builder, intent, surface, onset, self._severity_for(convention))
        self._set_coded_term(intent, "Hypoglycaemia", matches=True)

    def _pattern_explicit_misspelled(self, builder, intent, convention, subject_id,
                                     anchor_date, anchor_surface, ref_start, unit, extra_rows):
        surface = self._pick(MISSPELLINGS)
        onset = self._onset(builder, intent, anchor_date, anchor_surface, ref_start)
        self._concept_sentence(builder, intent, surface, onset, self._severity_for(convention))
        self._add_symptoms(builder, intent)

    def _pattern_abbrev_gated(self, builder, intent, convention, subject_id,
                              anchor_date, anchor_surface, ref_start, unit, extra_rows):
        onset = self._onset(builder, intent, anchor_date, anchor_surface, ref_start)
        symptoms = self._choose_symptoms(2)
        intent["symptoms"] = symptoms
        value, canonical = self._glucose_value(unit, level="low")
        intent["labs"].append(
            {"test": "GLUCOSE", "value": value, "unit": unit, "canonical_mgdl": canonical}
        )
        # The abbreviation and its context gate sit in the same sentence.
        builder.marked_sentence(
            f"The subject had a symptomatic ",
            "hypo",
            f" {onset}, with {self._symptom_phrase(symptoms)} and a "
            f"capillary glucose of {value} {unit}.",
            "concept",
        )
        intent["explicit_mention"] = True

    def _pattern_abbrev_ungated(self, builder, intent, convention, subject_id,
                                anchor_date, anchor_surface, ref_start, unit, extra_rows):
        # "hypo" with no glucose value and no qualifying symptom in scope. The
        # gate must suppress it: an ungated abbreviation is not an event.
        intent["concept"] = None
        intent["assertion"] = None
        self._set_coded_term(intent, self._pick(NON_SPECIFIC_CODED_TERMS), matches=False)
        intent["coded_term_version"] = convention.dictionary_version
        builder.sentence(
            "The site coordinator noted in passing that the subject uses the word "
            "hypo to describe any unwell feeling."
        )
        builder.sentence("No glucose measurement was taken at that visit.")

    def _pattern_lab_symptom(self, builder, intent, convention, subject_id,
                             anchor_date, anchor_surface, ref_start, unit, extra_rows):
        onset = self._onset(builder, intent, anchor_date, anchor_surface, ref_start)
        symptoms = self._choose_symptoms(2)
        intent["symptoms"] = symptoms
        builder.sentence(
            f"{self._ucfirst(onset)} the subject "
            f"reported {self._symptom_phrase(symptoms)}."
        )
        self._add_lab(builder, intent, unit, level="low")
        if not convention.codes_hypoglycemia or self.rng.random() < 0.5:
            self._set_coded_term(intent, self._pick(NON_SPECIFIC_CODED_TERMS), matches=False)

    def _pattern_split_sentence(self, builder, intent, convention, subject_id,
                                anchor_date, anchor_surface, ref_start, unit, extra_rows):
        onset = self._onset(builder, intent, anchor_date, anchor_surface, ref_start)
        symptoms = self._choose_symptoms(2)
        intent["symptoms"] = symptoms
        builder.sentence(
            f"The subject became unwell {onset}."
        )
        builder.sentence(f"They described {self._symptom_phrase(symptoms)}.")
        self._add_lab(builder, intent, unit, level="low")
        self._set_coded_term(intent, self._pick(NON_SPECIFIC_CODED_TERMS), matches=False)

    def _pattern_context_rescue(self, builder, intent, convention, subject_id,
                                anchor_date, anchor_surface, ref_start, unit, extra_rows):
        onset = self._onset(builder, intent, anchor_date, anchor_surface, ref_start)
        symptoms = self._choose_symptoms(2)
        intent["symptoms"] = symptoms
        builder.sentence(
            f"{self._ucfirst(onset)} the subject "
            f"reported {self._symptom_phrase(symptoms)}."
        )
        builder.sentence(self._pick(RESCUE_PHRASES))
        intent["rescue_treatment"] = True
        self._set_coded_term(intent, self._pick(NON_SPECIFIC_CODED_TERMS), matches=False)

    def _pattern_context_action(self, builder, intent, convention, subject_id,
                                anchor_date, anchor_surface, ref_start, unit, extra_rows):
        onset = self._onset(builder, intent, anchor_date, anchor_surface, ref_start)
        symptoms = self._choose_symptoms(2)
        intent["symptoms"] = symptoms
        builder.sentence(
            f"{self._ucfirst(onset)} the subject "
            f"reported {self._symptom_phrase(symptoms)}."
        )
        action = self._pick(["dose_reduced", "dose_interrupted"])
        builder.sentence(ACTION_PHRASES[action])
        intent["action_taken"] = action
        self._set_coded_term(intent, self._pick(NON_SPECIFIC_CODED_TERMS), matches=False)

    def _pattern_symptom_only(self, builder, intent, convention, subject_id,
                              anchor_date, anchor_surface, ref_start, unit, extra_rows):
        # Symptoms with nothing corroborating: no value, no rescue, no action.
        # v1 assigns `none`. Counting these would manufacture signal.
        onset = self._onset(builder, intent, anchor_date, anchor_surface, ref_start)
        symptoms = self._choose_symptoms(1)
        intent["symptoms"] = symptoms
        builder.sentence(
            f"{self._ucfirst(onset)} the subject "
            f"mentioned {self._symptom_phrase(symptoms)}."
        )
        builder.sentence("No glucose measurement was available.")
        builder.sentence(ACTION_PHRASES["none"])
        self._set_coded_term(intent, self._pick(NON_SPECIFIC_CODED_TERMS), matches=False)
        intent["action_taken"] = "none"
        intent["suppress_extra_rescue"] = True

    def _pattern_negated(self, builder, intent, convention, subject_id,
                         anchor_date, anchor_surface, ref_start, unit, extra_rows):
        surface = self._pick(CONCEPT_SURFACES_US + CONCEPT_SURFACES_GB)
        template = self._pick(NEGATION_PHRASES)
        self._templated_concept(builder, template, surface)
        intent["assertion"] = "absent"
        self._structured_only_onset(intent)
        self._set_coded_term(intent, self._pick(NON_SPECIFIC_CODED_TERMS), matches=False)

    def _pattern_hypothetical(self, builder, intent, convention, subject_id,
                              anchor_date, anchor_surface, ref_start, unit, extra_rows):
        surface = self._pick(CONCEPT_SURFACES_US + CONCEPT_SURFACES_GB)
        self._templated_concept(builder, self._pick(HYPOTHETICAL_PHRASES), surface)
        intent["assertion"] = "hypothetical"
        self._structured_only_onset(intent)
        self._set_coded_term(intent, self._pick(NON_SPECIFIC_CODED_TERMS), matches=False)

    def _pattern_historical(self, builder, intent, convention, subject_id,
                            anchor_date, anchor_surface, ref_start, unit, extra_rows):
        surface = self._pick(CONCEPT_SURFACES_US + CONCEPT_SURFACES_GB)
        self._templated_concept(builder, self._pick(HISTORICAL_PHRASES), surface)
        intent["assertion"] = "historical"
        self._structured_only_onset(intent)
        self._set_coded_term(intent, self._pick(NON_SPECIFIC_CODED_TERMS), matches=False)

    def _pattern_family_history(self, builder, intent, convention, subject_id,
                                anchor_date, anchor_surface, ref_start, unit, extra_rows):
        surface = self._pick(CONCEPT_SURFACES_US + CONCEPT_SURFACES_GB)
        self._templated_concept(builder, self._pick(FAMILY_PHRASES), surface)
        intent["assertion"] = "family_history"
        self._structured_only_onset(intent)
        self._set_coded_term(intent, self._pick(NON_SPECIFIC_CODED_TERMS), matches=False)

    def _pattern_uncertain(self, builder, intent, convention, subject_id,
                           anchor_date, anchor_surface, ref_start, unit, extra_rows):
        surface = self._pick(CONCEPT_SURFACES_US + CONCEPT_SURFACES_GB)
        self._templated_concept(builder, self._pick(UNCERTAIN_PHRASES), surface)
        intent["assertion"] = "uncertain"
        self._structured_only_onset(intent)
        self._add_symptoms(builder, intent)
        self._set_coded_term(intent, self._pick(NON_SPECIFIC_CODED_TERMS), matches=False)

    def _pattern_out_of_window(self, builder, intent, convention, subject_id,
                               anchor_date, anchor_surface, ref_start, unit, extra_rows):
        offset = self.rng.randint(21, 60)
        onset = self._onset(builder, intent, anchor_date, anchor_surface, ref_start, offset=offset)
        surface = self._pick(CONCEPT_SURFACES_US + CONCEPT_SURFACES_GB)
        self._concept_sentence(builder, intent, surface, onset, self._severity_for(convention))
        self._set_coded_term(
            intent,
            "Hypoglycaemia" if convention.codes_hypoglycemia else None,
            matches=True,
        )

    def _pattern_unresolved_onset(self, builder, intent, convention, subject_id,
                                  anchor_date, anchor_surface, ref_start, unit, extra_rows):
        # No temporal expression in the narrative and no AESTDTC in the table.
        # The offset cannot be resolved, and the definition — not the extractor
        # — decides what happens to it.
        surface = self._pick(CONCEPT_SURFACES_US + CONCEPT_SURFACES_GB)
        builder.marked_sentence(
            "The subject experienced ", surface,
            " at an unrecorded time during the treatment period.", "concept",
        )
        intent["explicit_mention"] = True
        intent["onset_offset_days"] = None
        intent["onset_phrasing"] = "none"
        self._set_coded_term(
            intent,
            "Hypoglycaemia" if convention.codes_hypoglycemia else None,
            matches=True,
        )
        self._add_symptoms(builder, intent)

    def _pattern_distractor(self, builder, intent, convention, subject_id,
                            anchor_date, anchor_surface, ref_start, unit, extra_rows):
        concept_id = self._pick(sorted(DISTRACTOR_CONCEPTS))
        coded, surfaces = DISTRACTOR_CONCEPTS[concept_id]
        surface = self._pick(surfaces)
        onset = self._onset(builder, intent, anchor_date, anchor_surface, ref_start)
        intent["concept"] = concept_id
        self._set_coded_term(intent, coded, matches=True)
        builder.marked_sentence("The subject reported ", surface, f" {onset}.", "concept")
        intent["explicit_mention"] = True
        intent["symptoms"] = sorted(
            set(intent["symptoms"]) | set(DISTRACTOR_SYMPTOMS.get(concept_id, []))
        )
        intent["severity"] = self._severity_for(convention)
        if intent["severity"]:
            builder.sentence(
                f"The event was graded as {SEVERITY_WORDS[intent['severity']]} in intensity."
            )
        if concept_id == "HYPERGLYCEMIA":
            self._add_lab(builder, intent, unit, level="high")

    # -- shared narrative fragments ----------------------------------------

    def _templated_concept(self, builder, template: str, surface: str) -> None:
        """Render a template containing ``{concept}`` while tracking its offset."""
        token = "{concept_cap}" if "{concept_cap}" in template else "{concept}"
        mention = surface.capitalize() if token == "{concept_cap}" else surface
        before, _, after = template.partition(token)
        builder.marked_sentence(before, mention, after, "concept")

    def _structured_only_onset(self, intent) -> int | None:
        """An onset present in the AE table but expressed nowhere in the text.

        These records still sit somewhere in time; the window never gets to
        matter because the assertion policy removes them first.  Where the
        table is also silent the onset is genuinely unknown, and gold records
        it as unknown rather than as a miss the extractor could never make good.
        """
        if self.rng.random() < 0.35:
            intent["onset_offset_days"] = self.rng.randint(0, 20)
            intent["onset_phrasing"] = "structured"
            intent["onset_in_ae_table"] = True
        else:
            intent["onset_offset_days"] = None
            intent["onset_phrasing"] = "none"
        return intent["onset_offset_days"]

    def _severity_for(self, convention: StudyConvention) -> str | None:
        if convention.severe_events_only:
            return "severe"
        return self._pick(["mild", "moderate", "severe", "moderate", "mild"])

    def _add_symptoms(self, builder, intent, count: int = 0) -> None:
        symptoms = self._choose_symptoms(count)
        intent["symptoms"] = sorted(set(intent["symptoms"]) | set(symptoms))
        adjectival = [s for s in symptoms if s in SYMPTOM_ADJ_FORMS]
        if len(symptoms) == 1 and adjectival and self.rng.random() < 0.35:
            builder.sentence(
                self._pick(SYMPTOM_ADJ_PHRASES).format(
                    adj=self._pick(SYMPTOM_ADJ_FORMS[adjectival[0]])
                )
            )
            return
        template = self._pick(SYMPTOM_PHRASES)
        phrase = self._symptom_phrase(symptoms)
        builder.sentence(
            template.format(
                symptoms=phrase,
                symptoms_cap=self._ucfirst(phrase),
                was_were="was" if len(symptoms) == 1 else "were",
            )
        )

    def _add_lab(self, builder, intent, unit: str, *, level: str = "normal") -> None:
        value, canonical = self._glucose_value(unit, level=level)
        intent["labs"].append(
            {"test": "GLUCOSE", "value": value, "unit": unit, "canonical_mgdl": canonical}
        )
        builder.sentence(self._pick(LAB_PHRASES).format(value=value, unit=unit))

    def _add_management(self, builder, intent, convention: StudyConvention) -> None:
        """Management, outcome, seriousness and causality sentences.

        Kept in their own sentences so their cue words never fall inside the
        assertion scope of a concept mention.
        """
        if intent["concept"] is None:
            return
        if intent["assertion"] in ("hypothetical", "historical", "family_history", "absent"):
            # These records still carry a coded AE of some kind, but management
            # language would confuse the picture; keep them short.
            builder.sentence("Study drug continued unchanged.")
            intent["action_taken"] = "none"
            intent["outcome"] = "unknown" if self.rng.random() < 0.3 else None
            if intent["outcome"]:
                builder.sentence(OUTCOME_PHRASES["unknown"])
            return

        if intent["action_taken"] is None and self.rng.random() < 0.55:
            action = self._pick(["dose_reduced", "dose_interrupted", "drug_withdrawn", "none"])
            builder.sentence(ACTION_PHRASES[action])
            intent["action_taken"] = action

        if (
            not intent["rescue_treatment"]
            and not intent["suppress_extra_rescue"]
            and intent["concept"] == "HYPOGLYCEMIA"
        ):
            if self.rng.random() < 0.3:
                builder.sentence(self._pick(RESCUE_PHRASES))
                intent["rescue_treatment"] = True

        if intent["severity"] == "severe" and self.rng.random() < 0.55:
            category = self._pick(["hospitalisation", "life_threatening", "other_medically_important"])
            builder.sentence(SERIOUSNESS_PHRASES[category])
            intent["seriousness"] = sorted(set(intent["seriousness"]) | {category})
        elif self.rng.random() < 0.12:
            # A mild event can be serious. Severity and seriousness are separate
            # fields and this corpus contains counterexamples to their conflation.
            builder.sentence(SERIOUSNESS_PHRASES["hospitalisation"])
            intent["seriousness"] = sorted(set(intent["seriousness"]) | {"hospitalisation"})

        if self.rng.random() < 0.6:
            relatedness = self._pick(["possible", "probable", "not_related", "unlikely"])
            builder.sentence(RELATEDNESS_PHRASES[relatedness])
            intent["relatedness"] = relatedness

        if self.rng.random() < 0.22:
            rechallenge = self._pick(["done_recurred", "done_no_recurrence", "not_done"])
            builder.sentence(RECHALLENGE_PHRASES[rechallenge])
            intent["rechallenge"] = rechallenge

        if self.rng.random() < 0.8:
            outcome = self._pick(["resolved", "resolved", "resolving", "not_resolved"])
            builder.sentence(OUTCOME_PHRASES[outcome])
            intent["outcome"] = outcome

    # -- table rows ---------------------------------------------------------

    #: Which AE columns each study populates. What is not in the table is
    #: recoverable only from the narrative, which is the point.
    _STRUCTURED_AE_COLUMNS = {
        "full": {"AESEV", "AESER", "AESCAT", "AEREL", "AEACN", "AEOUT"},
        "partial": {"AESEV", "AESER", "AESCAT", "AEREL"},
        "minimal": {"AESER"},
    }

    def _ae_row(self, convention, subject_id, ae_seq, doc_id, intent, onset_date):
        seriousness = intent["seriousness"]
        populated = self._STRUCTURED_AE_COLUMNS[convention.structured_ae_detail]

        def col(name: str, value):
            return value if name in populated else ""

        return {
            "STUDYID": convention.study_id,
            "USUBJID": subject_id,
            "AESEQ": ae_seq,
            "DOCID": doc_id,
            "AETERM": self._verbatim_for(intent),
            "AEDECOD": intent["coded_term"] or "",
            "AEDICTVER": convention.dictionary_version if intent["coded_term"] else "",
            "AESTDTC": onset_date.isoformat() if onset_date else "",
            "AESEV": col("AESEV", intent["severity"] or ""),
            "AESER": col("AESER", "Y" if seriousness else "N"),
            "AESCAT": col("AESCAT", "|".join(seriousness)),
            "AEREL": col("AEREL", intent["relatedness"] or ""),
            "AEACN": col("AEACN", intent["action_taken"] or ""),
            "AEOUT": col("AEOUT", intent["outcome"] or ""),
            "SYNTHETIC": SYNTHETIC_FLAG,
        }

    def _verbatim_for(self, intent) -> str:
        if intent["concept"] == "HYPOGLYCEMIA":
            return self._pick(
                ["low blood sugar episode", "hypoglycaemic event", "feeling shaky and sweaty",
                 "symptomatic low glucose", "unwell episode"]
            )
        if intent["concept"]:
            return DISTRACTOR_CONCEPTS.get(intent["concept"], (intent["concept"], []))[0].lower()
        return "unwell episode"

    def _lb_row(self, convention, subject_id, seq, lab, when):
        canonical = lab["canonical_mgdl"]
        return {
            "STUDYID": convention.study_id,
            "USUBJID": subject_id,
            "LBSEQ": seq,
            "LBTESTCD": "GLUC",
            "LBTEST": "Glucose",
            "LBORRES": lab["value"],
            "LBORRESU": lab["unit"],
            "LBSTRESN": canonical,
            "LBSTRESU": "mg/dL",
            "LBDTC": when.isoformat(),
            "SYNTHETIC": SYNTHETIC_FLAG,
        }

    # -- top level ----------------------------------------------------------

    def generate(self) -> GeneratedCorpus:
        corpus = GeneratedCorpus(
            tables={"dm": [], "ae": [], "ex": [], "lb": [], "cm": []}
        )
        lb_seq_by_subject: dict[str, int] = {}

        for convention in self.conventions:
            n_subjects = self.subjects_per_study or convention.n_subjects
            for index in range(1, n_subjects + 1):
                subject_id = f"{convention.study_id}-{index:03d}"
                ref_start = convention.start + _dt.timedelta(days=self.rng.randint(0, 240))
                age = self.rng.randint(38, 79)
                sex = self._pick(["M", "F"])
                corpus.tables["dm"].append(
                    {
                        "STUDYID": convention.study_id,
                        "USUBJID": subject_id,
                        "SUBJID": f"{index:03d}",
                        "AGE": age,
                        "SEX": sex,
                        "ARM": self._pick(["Study drug", "Study drug", "Placebo"]),
                        "RFSTDTC": ref_start.isoformat(),
                        "COUNTRY": self._pick(["USA", "GBR", "DEU", "JPN", "BRA"]),
                        "SYNTHETIC": SYNTHETIC_FLAG,
                    }
                )

                # Exposure: an initial dose, then one or two escalations. SDTM
                # represents an escalation as a new record with a higher dose,
                # not as a flag, so that is how the anchor must be found.
                dose = self._pick([5.0, 10.0, 20.0])
                ex_rows = [
                    {
                        "STUDYID": convention.study_id,
                        "USUBJID": subject_id,
                        "EXSEQ": 1,
                        "EXTRT": "Study drug",
                        "EXDOSE": dose,
                        "EXDOSU": "mg",
                        "EXSTDTC": ref_start.isoformat(),
                        "EXENDTC": "",
                        "SYNTHETIC": SYNTHETIC_FLAG,
                    }
                ]
                escalation_date = ref_start + _dt.timedelta(days=self.rng.randint(21, 84))
                ex_rows.append(
                    {
                        "STUDYID": convention.study_id,
                        "USUBJID": subject_id,
                        "EXSEQ": 2,
                        "EXTRT": "Study drug",
                        "EXDOSE": dose * 2,
                        "EXDOSU": "mg",
                        "EXSTDTC": escalation_date.isoformat(),
                        "EXENDTC": "",
                        "SYNTHETIC": SYNTHETIC_FLAG,
                    }
                )
                if self.rng.random() < 0.35:
                    second = escalation_date + _dt.timedelta(days=self.rng.randint(28, 70))
                    ex_rows.append(
                        {
                            "STUDYID": convention.study_id,
                            "USUBJID": subject_id,
                            "EXSEQ": 3,
                            "EXTRT": "Study drug",
                            "EXDOSE": dose * 3,
                            "EXDOSU": "mg",
                            "EXSTDTC": second.isoformat(),
                            "EXENDTC": "",
                            "SYNTHETIC": SYNTHETIC_FLAG,
                        }
                    )
                corpus.tables["ex"].extend(ex_rows)

                n_events = self.rng.choice([0, 1, 1, 1, 2, 2, 3])
                for ae_seq in range(1, n_events + 1):
                    record = self._build_record(
                        convention, subject_id, ae_seq, escalation_date, ref_start
                    )
                    corpus.tables["ae"].append(record["ae_row"])
                    corpus.narratives.append(record["narrative"])
                    corpus.gold.append(record["gold"])
                    for row in record["extra_rows"]:
                        if row.table == "lb":
                            seq = lb_seq_by_subject.get(subject_id, 0) + 1
                            lb_seq_by_subject[subject_id] = seq
                            row.values["LBSEQ"] = seq
                        corpus.tables[row.table].append(row.values)

                # Routine, non-event glucose measurements, so the LB table is not
                # composed exclusively of qualifying values.
                for offset in (7, 28, 56):
                    value, canonical = self._glucose_value(
                        convention.glucose_unit, level="normal"
                    )
                    seq = lb_seq_by_subject.get(subject_id, 0) + 1
                    lb_seq_by_subject[subject_id] = seq
                    corpus.tables["lb"].append(
                        self._lb_row(
                            convention, subject_id, seq,
                            {"test": "GLUCOSE", "value": value, "unit": convention.glucose_unit,
                             "canonical_mgdl": canonical},
                            ref_start + _dt.timedelta(days=offset),
                        )
                    )

        corpus.gold_cases = self._subject_gold(corpus.gold)
        corpus.manifest = {
            "generator": "aelayer.generate",
            "seed": self.seed,
            "synthetic": True,
            "notice": (
                "All records are computer generated. No real patient data is "
                "present in this repository."
            ),
            "gold_case_definition": "te_symptomatic_hypoglycemia.v1",
            "studies": {
                c.study_id: {
                    "label": c.label,
                    "glucose_unit": c.glucose_unit,
                    "dictionary_version": c.dictionary_version,
                    "codes_hypoglycemia": c.codes_hypoglycemia,
                    "severe_events_only": c.severe_events_only,
                    "structured_ae_detail": c.structured_ae_detail,
                    "note": c.note,
                }
                for c in self.conventions
            },
            "counts": {
                "studies": len(self.conventions),
                "subjects": len(corpus.tables["dm"]),
                "ae_records": len(corpus.tables["ae"]),
            },
        }
        return corpus

    @staticmethod
    def _subject_gold(gold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Roll record-level truth up to one gold verdict per subject.

        The same aggregation the evaluator uses: strongest verdict wins, and
        within a verdict the strongest evidence state wins.
        """
        verdict_rank = {"excluded": 0, "review": 1, "case": 2}
        best: dict[str, dict[str, Any]] = {}
        for row in gold_rows:
            subject = row["subject_id"]
            key = (verdict_rank[row["verdict"]], EVIDENCE_STATE_RANK[row["evidence_state"]])
            current = best.get(subject)
            if current is None or key > current["_key"]:
                best[subject] = {
                    "_key": key,
                    "subject_id": subject,
                    "study_id": row["study_id"],
                    "verdict": row["verdict"],
                    "evidence_state": row["evidence_state"],
                    "doc_id": row["doc_id"],
                    "pattern": row["pattern"],
                }
        return [
            {k: v for k, v in entry.items() if k != "_key"}
            for entry in sorted(best.values(), key=lambda e: e["subject_id"])
        ]


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

TABLE_COLUMNS: dict[str, list[str]] = {
    "dm": ["STUDYID", "USUBJID", "SUBJID", "AGE", "SEX", "ARM", "RFSTDTC", "COUNTRY", "SYNTHETIC"],
    "ae": ["STUDYID", "USUBJID", "AESEQ", "DOCID", "AETERM", "AEDECOD", "AEDICTVER",
           "AESTDTC", "AESEV", "AESER", "AESCAT", "AEREL", "AEACN", "AEOUT", "SYNTHETIC"],
    "ex": ["STUDYID", "USUBJID", "EXSEQ", "EXTRT", "EXDOSE", "EXDOSU", "EXSTDTC", "EXENDTC", "SYNTHETIC"],
    "lb": ["STUDYID", "USUBJID", "LBSEQ", "LBTESTCD", "LBTEST", "LBORRES", "LBORRESU",
           "LBSTRESN", "LBSTRESU", "LBDTC", "SYNTHETIC"],
    "cm": ["STUDYID", "USUBJID", "CMSEQ", "CMTRT", "CMINDC", "CMSTDTC", "SYNTHETIC"],
}


def write_corpus(corpus: GeneratedCorpus, out_dir: str | Path | None = None) -> Path:
    root = Path(out_dir or paths.DATA_DIR)
    root.mkdir(parents=True, exist_ok=True)

    for table, columns in TABLE_COLUMNS.items():
        path = root / f"{table}.csv"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            for row in corpus.tables.get(table, []):
                writer.writerow({c: row.get(c, "") for c in columns})

    _write_jsonl(root / "narratives.jsonl", corpus.narratives)
    _write_jsonl(root / "gold.jsonl", corpus.gold)
    _write_jsonl(root / "gold_cases.jsonl", corpus.gold_cases)
    (root / "manifest.json").write_text(
        json.dumps(corpus.manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "README.txt").write_text(
        "SYNTHETIC DATA ONLY\n"
        "===================\n\n"
        "Every file in this directory is computer generated by aelayer.generate.\n"
        "No real patient data, and no data derived from real patients, is present.\n"
        "The gold.jsonl and gold_cases.jsonl files hold the generator's intent and\n"
        "are read only by the evaluation harness.\n",
        encoding="utf-8",
    )
    return root


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def generate_corpus(
    seed: int = 7,
    n_studies: int = 4,
    out_dir: str | Path | None = None,
    subjects_per_study: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    generator = CorpusGenerator(
        seed=seed, n_studies=n_studies, subjects_per_study=subjects_per_study
    )
    corpus = generator.generate()
    root = write_corpus(corpus, out_dir)
    return root, corpus.manifest
