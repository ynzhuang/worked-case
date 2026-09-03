"""The synthetic corpus.

Everything here is computer generated. No real patient data, and nothing
derived from real patient data, is present anywhere in this repository.

The generator samples a ``ClinicalTruth`` — what happened to a patient — and
renders that same truth under seven study profiles. The truth does not change;
where its modifier ends up, and whether it can be read at all, does.

Three of the profiles exist to make a specific distinction testable:

``P_both``
    the modifier in a structured qualifier *and* in the reported term, which is
    the evaluation set the silver standard is built from
``P_negated``
    the reported term states the modifier is **absent** — the only way to prove
    the system tells a documented negative from silence
``P_absent``
    the modifier nowhere, so a qualifying event with qualifying timing still
    cannot be evaluated
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import hashlib
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from . import paths
from .catalog import ConceptCatalog, load_configs
from .profiles import StudyProfile, StudyProfiles

SYNTHETIC_FLAG = "Y"
DOC_HEADER = "*** SYNTHETIC RECORD - COMPUTER GENERATED - NOT REAL PATIENT DATA ***"

CUTANEOUS = ("RASH", "RASH_ERYTHEMATOUS", "RASH_MACULOPAPULAR")
DISTRACTORS = ("PRURITUS", "NAUSEA", "NEUTROPENIA")

#: Phrasings a site writes for mucosal involvement, by catalogue value. Several
#: per value, so normalization is doing real work rather than matching one
#: canned string.
MUCOSAL_PHRASES = {
    "ORAL": ["oral mucosal involvement", "mucosal lesions of the mouth",
             "oral ulceration", "buccal erosions", "stomatitis"],
    "OCULAR": ["ocular involvement", "conjunctival involvement"],
    "GENITAL": ["genital erosions", "genital involvement"],
    "UNSPECIFIED": ["mucosal involvement", "mucous membrane involvement",
                    "mucosal erosions"],
}

#: Phrasings no catalogue value covers. The extractor meets these and abstains,
#: which is the behaviour the abstention rate exists to measure.
UNLEXICONED = ["involvement of the wet surfaces", "erosive changes internally",
               "lining tissue affected"]

NEGATION_TEMPLATES = [
    "{event} without {modifier}",
    "{event}, no {modifier}",
    "{event}; {modifier} was absent",
    "{event}, {modifier} was not present",
]

UNCERTAIN_TEMPLATES = [
    "{event} with possible {modifier}",
    "{event}, query {modifier}",
    "{event}; {modifier} cannot be excluded",
]

PRESENT_TEMPLATES = [
    "{event} with {modifier}",
    "{event} and {modifier}",
    "{event}, {modifier} noted",
]

COMMENT_TEMPLATES = {
    "present": [
        "Investigator comment: {modifier} confirmed at the study visit.",
        "Site clarification: the eruption was accompanied by {modifier}.",
    ],
    "absent": [
        "Investigator comment: no {modifier} was seen at any visit.",
        "Site clarification: the mucosa was examined and spared.",
    ],
    "uncertain": [
        "Investigator comment: possible {modifier}, not confirmed.",
    ],
}

EVENT_WORDS = {
    "RASH": ["rash", "skin rash", "eruption"],
    "RASH_ERYTHEMATOUS": ["erythematous rash", "red rash"],
    "RASH_MACULOPAPULAR": ["maculopapular rash", "morbilliform rash"],
    "PRURITUS": ["pruritus", "itching"],
    "NAUSEA": ["nausea"],
    "NEUTROPENIA": ["neutropenia"],
}


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------


@dataclass
class ClinicalTruth:
    """What happened, before any study wrote it down."""

    truth_id: str
    concept: str
    #: What was actually the case about the modifier. "unknown" means nobody
    #: looked — distinct from "absent", which means somebody looked and found
    #: nothing.
    mucosal: str                   # present | absent | uncertain | unknown
    mucosal_site: str | None       # a catalogue value, when present
    onset_offset_days: int         # from first exposure
    duration_days: int
    severity: str
    grade: int
    seriousness: bool
    seriousness_criteria: list[str]
    relatedness: str
    action: str
    outcome: str
    daily_dose: int
    note: str = ""

    @property
    def onset_in_window(self) -> bool:
        return 0 <= self.onset_offset_days <= 30

    @property
    def cumulative_exposure(self) -> float:
        """Total dose taken before onset, in the study's own units."""
        return float(self.daily_dose * max(self.onset_offset_days, 0))

    def verdict_under_v1(self, mucosal_readable: bool) -> str:
        """The verdict cutaneous_mucosal v1 should reach for this rendering.

        Precedence mirrors the evaluator: a criterion that is present and fails
        settles the record as a negative whatever else is missing. Only when
        nothing has failed does an unreadable modifier make it unascertainable.
        """
        if self.concept not in CUTANEOUS:
            return "non_case"
        if not self.onset_in_window:
            return "non_case"
        if not mucosal_readable:
            return "not_ascertainable"
        if self.mucosal == "present":
            return "case"
        if self.mucosal == "absent":
            return "non_case"
        return "review"          # uncertain, as the definition asks


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


@dataclass
class GeneratedCorpus:
    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    documents: list[dict[str, Any]] = field(default_factory=list)
    truths: list[dict[str, Any]] = field(default_factory=list)
    gold: list[dict[str, Any]] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)


class CorpusGenerator:
    def __init__(
        self,
        seed: int = 7,
        profiles: Iterable[str] | None = None,
        shared_truths: int = 24,
        extra_per_profile: int = 12,
        catalog: ConceptCatalog | None = None,
        study_profiles: StudyProfiles | None = None,
    ):
        configs = load_configs()
        self.catalog = catalog or configs.catalog
        self.profiles = study_profiles or configs.profiles
        self.seed = seed
        self.rng = random.Random(seed)
        self.profile_ids = list(profiles or self.profiles.profile_ids())
        self.shared_truths = shared_truths
        self.extra_per_profile = extra_per_profile

    def _pick(self, options):
        return options[self.rng.randrange(len(options))]

    # -- sampling -----------------------------------------------------------

    def sample_truth(self, truth_id: str, kind: str) -> ClinicalTruth:
        """One clinical truth of a requested kind.

        The kinds exist so the corpus contains each thing the evaluation needs
        to measure, rather than whatever a uniform sample happened to produce.
        """
        concept = "RASH"
        site = None
        if kind == "mucosal_in_window":
            mucosal, offset = "present", self.rng.randint(0, 30)
            site = self._pick(["ORAL", "ORAL", "OCULAR", "UNSPECIFIED", "GENITAL"])
        elif kind == "mucosal_out_of_window":
            mucosal, offset = "present", self.rng.randint(40, 120)
            site = self._pick(["ORAL", "UNSPECIFIED"])
        elif kind == "documented_negative":
            # Somebody looked and found nothing. A non_case, not silence.
            mucosal, offset = "absent", self.rng.randint(0, 30)
        elif kind == "uncertain":
            mucosal, offset = "uncertain", self.rng.randint(0, 30)
            site = "UNSPECIFIED"
        elif kind == "never_examined":
            mucosal, offset = "unknown", self.rng.randint(0, 30)
        elif kind == "distractor":
            concept = self._pick(DISTRACTORS)
            mucosal, offset = "unknown", self.rng.randint(0, 60)
        elif kind == "graded_toxicity":
            # A haematological event, so the second definition has a real
            # denominator rather than a token handful of records. It has no
            # mucosal modifier at all, which is the point: the two definitions
            # select on different criteria and neither is special-cased.
            concept = "NEUTROPENIA"
            mucosal, offset = "unknown", self.rng.randint(5, 170)
        else:
            raise ValueError(f"unknown truth kind {kind!r}")

        serious = self.rng.random() < 0.15
        grade = self.rng.choice([1, 2, 2, 3, 3, 4])
        if kind == "graded_toxicity":
            # Spread across the threshold on purpose: a criterion nothing ever
            # fails is not a criterion.
            grade = self.rng.choice([1, 2, 3, 3, 4, 4])
        return ClinicalTruth(
            truth_id=truth_id,
            concept=concept,
            mucosal=mucosal,
            mucosal_site=site,
            onset_offset_days=offset,
            duration_days=self.rng.randint(2, 28),
            severity=["mild", "moderate", "severe"][min(grade, 3) - 1],
            grade=grade,
            seriousness=serious,
            seriousness_criteria=(
                [self._pick(["hospitalisation", "other_medically_important"])]
                if serious else []
            ),
            relatedness=self._pick(
                ["not_related", "unlikely", "possible", "possible", "probable"]
            ),
            action=self._pick(
                ["dose_not_changed", "dose_not_changed", "dose_reduced",
                 "drug_interrupted", "drug_withdrawn"]
            ),
            outcome=self._pick(["recovered", "recovered", "recovering",
                                "not_recovered"]),
            daily_dose=self._pick([10, 20, 50]),
            note=kind,
        )

    # -- text ---------------------------------------------------------------

    def _modifier_phrase(self, truth: ClinicalTruth) -> tuple[str | None, str | None]:
        """A phrasing for the modifier, and the value it resolves to.

        Sometimes the phrasing is one no catalogue covers, which is how the
        corpus contains text the extractor should decline rather than guess at.
        """
        if truth.mucosal == "unknown":
            return None, None
        if truth.mucosal == "absent":
            return self._pick(MUCOSAL_PHRASES["UNSPECIFIED"]), None
        if self.rng.random() < 0.12:
            return self._pick(UNLEXICONED), None
        site = truth.mucosal_site or "UNSPECIFIED"
        return self._pick(MUCOSAL_PHRASES[site]), site

    def reported_term(
        self, truth: ClinicalTruth, profile: StudyProfile
    ) -> tuple[str, str | None, str | None]:
        """The investigator's own words.

        Returns the text, the assertion it actually states (or None), and the
        modifier value it resolves to.
        """
        event = self._pick(EVENT_WORDS[truth.concept])
        style = profile.reported_term_style
        homes = profile.home_ids("mucosal_involvement")

        if style == "terse" or "reported_term" not in homes:
            return event, None, None

        # The text styles differ in exactly one way, and it is the distinction
        # the whole `assertion` field exists for.
        #
        # `rich`    — the site writes the modifier down when it is there, and
        #             says nothing when it is not. An absence then looks
        #             identical to never having looked, which is the trap.
        # `negated` — the convention is to state the modifier either way, so an
        #             absence is *documented* and the subject belongs in the
        #             denominator as a non_case.
        # `mixed`   — what most real sites do: presence always written down,
        #             absence written down sometimes. Without this style the
        #             one profile a silver standard can be built from would
        #             contain no documented negatives at all, and the class
        #             that decides the denominator would be unmeasurable.
        if truth.mucosal not in ("present", "uncertain"):
            if style == "rich":
                return event, None, None
            if style == "mixed" and self.rng.random() < 0.45:
                return event, None, None

        phrase, value = self._modifier_phrase(truth)
        if phrase is None:
            return event, None, None

        if truth.mucosal == "absent":
            template = self._pick(NEGATION_TEMPLATES)
            return template.format(event=event, modifier=phrase), "absent", None
        if truth.mucosal == "uncertain":
            template = self._pick(UNCERTAIN_TEMPLATES)
            return template.format(event=event, modifier=phrase), "uncertain", value
        template = self._pick(PRESENT_TEMPLATES)
        return template.format(event=event, modifier=phrase), "present", value

    def comment_text(
        self, truth: ClinicalTruth
    ) -> tuple[str | None, str | None, str | None]:
        if truth.mucosal == "unknown":
            return None, None, None
        phrase, value = self._modifier_phrase(truth)
        if phrase is None:
            return None, None, None
        assertion = truth.mucosal
        template = self._pick(COMMENT_TEMPLATES.get(assertion, COMMENT_TEMPLATES["present"]))
        return template.format(modifier=phrase), assertion, (
            value if assertion == "present" else None
        )

    # -- one subject --------------------------------------------------------

    def _subject_rows(
        self, profile: StudyProfile, subject_id: str, corpus: GeneratedCorpus,
        first_dose: _dt.date, truth: ClinicalTruth,
    ) -> None:
        corpus.tables["dm"].append({
            "STUDYID": profile.study_id, "USUBJID": subject_id,
            "AGE": self.rng.randint(21, 82), "SEX": self._pick(["M", "F"]),
            "ARM": self._pick(["Drug 10 mg", "Drug 20 mg", "Drug 50 mg"]),
            "RFSTDTC": first_dose.isoformat(),
            "COUNTRY": self._pick(["USA", "GBR", "DEU", "JPN"]),
            "PROFILE": profile.profile_id, "SYNTHETIC": SYNTHETIC_FLAG,
        })
        # Monthly exposure records at a constant daily dose, so cumulative
        # exposure before onset is a governed computation rather than a guess.
        for index in range(4):
            start = first_dose + _dt.timedelta(days=28 * index)
            corpus.tables["ex"].append({
                "STUDYID": profile.study_id, "USUBJID": subject_id,
                "EXSEQ": index + 1, "EXTRT": "STUDY DRUG",
                "EXDOSE": truth.daily_dose,
                "EXDOSU": profile.conventions.get("dose_unit", "mg"),
                "EXSTDTC": self._date(start, profile),
                "EXENDTC": self._date(start + _dt.timedelta(days=27), profile),
                "SYNTHETIC": SYNTHETIC_FLAG,
            })

    @staticmethod
    def _date(value: _dt.date | None, profile: StudyProfile) -> str:
        if value is None:
            return ""
        if profile.conventions.get("date_style") == "dmy":
            return value.strftime("%d-%b-%Y")
        return value.isoformat()

    def render(
        self, truth: ClinicalTruth, profile: StudyProfile, subject_id: str,
        first_dose: _dt.date, corpus: GeneratedCorpus, cohort: str,
    ) -> dict[str, Any]:
        """Write one truth into one study's tables, and record the answer key."""
        onset = first_dose + _dt.timedelta(days=truth.onset_offset_days)
        end = onset + _dt.timedelta(days=truth.duration_days)
        record_id = f"{subject_id}-AE-01"
        homes = profile.home_ids("mucosal_involvement")

        # Which concept this study codes the event to. A study with several
        # declared codings spreads its records across them deterministically:
        # one clinical situation, several legitimate codes, which is what the
        # concept set and the version reconciliation both exist to handle.
        concept = truth.concept
        if concept == "RASH" and profile.prefer_concept:
            concept = profile.prefer_concept[
                _stable_index(f"{profile.profile_id}|{subject_id}",
                              len(profile.prefer_concept))
            ]
        code = self.catalog.concept(concept).code_in(profile.dictionary_version)
        if code is None:
            # A concept with no code under this study's version stays as it was
            # coded under the version the study actually used.
            fallback = next(iter(self.catalog.concept(concept).codes.items()))
            code = fallback[1]

        term, term_assertion, term_value = self.reported_term(truth, profile)
        comment, comment_assertion, comment_value = (None, None, None)
        if "comment" in homes:
            comment, comment_assertion, comment_value = self.comment_text(truth)

        structured = None
        if profile.structured_home("mucosal_involvement") is not None \
                and truth.mucosal != "unknown":
            structured = {
                "present": "Y", "absent": "N", "uncertain": "U",
            }[truth.mucosal]

        home = profile.structured_home("mucosal_involvement")
        row = {
            "STUDYID": profile.study_id, "USUBJID": subject_id,
            "AESEQ": 1, "AESPID": record_id,
            "AETERM": term,
            "AEDECOD": code if profile.collects_variable("AEDECOD") else "",
            "AEDICTVER": profile.dictionary_version,
            "AEMUCOS": (
                structured or ""
                if home is not None and home.variable == "AEMUCOS" else ""
            ),
            "AESEV": truth.severity if profile.collects_variable("AESEV") else "",
            "AEGRADE": truth.grade if profile.collects_variable("AEGRADE") else "",
            "AESER": "Y" if truth.seriousness else "N",
            "AESCAT": "|".join(truth.seriousness_criteria) if truth.seriousness else "",
            "AEREL": truth.relatedness,
            "AEACN": truth.action,
            "AEOUT": truth.outcome,
            "AESTDTC": self._date(onset, profile),
            "AEENDTC": self._date(end, profile),
            "SYNTHETIC": SYNTHETIC_FLAG,
        }
        corpus.tables["ae"].append(row)

        if home is not None and home.variable == "SC.MUCOSAL" and structured:
            corpus.tables["sc"].append({
                "STUDYID": profile.study_id, "USUBJID": subject_id,
                "IDVAR": "AESPID", "IDVARVAL": record_id,
                "SCTESTCD": "MUCOSAL", "SCTEST": "Mucosal involvement",
                "SCORRES": structured, "SYNTHETIC": SYNTHETIC_FLAG,
            })

        doc_id = None
        if comment:
            doc_id = f"{record_id}-CO"
            corpus.tables["co"].append({
                "STUDYID": profile.study_id, "USUBJID": subject_id,
                "IDVAR": "AESPID", "IDVARVAL": record_id, "COSEQ": 1,
                "COVAL": comment, "DOCID": doc_id, "SYNTHETIC": SYNTHETIC_FLAG,
            })
            corpus.documents.append({
                "doc_id": doc_id, "study_id": profile.study_id,
                "subject_id": subject_id, "source_record_id": record_id,
                "kind": "comment", "header": DOC_HEADER, "text": comment,
            })

        readable = self._readable(
            profile, structured, term_assertion, comment_assertion
        )
        gold = self._gold(
            truth, profile, record_id, subject_id, onset, cohort, structured,
            term_assertion, term_value, comment_assertion, comment_value, readable,
            concept, code,
        )
        corpus.gold.append(gold)
        return gold

    def _readable(
        self, profile: StudyProfile, structured: str | None,
        term_assertion: str | None, comment_assertion: str | None,
    ) -> dict[str, Any]:
        """What each route holds, and whether the modifier can be read at all.

        The routes are kept apart rather than collapsed into one boolean
        because the value ablation adds them one at a time, and each stage
        needs to know what a system limited to that stage could have seen.
        """
        from_structured = structured is not None
        from_term = term_assertion is not None
        from_comment = comment_assertion is not None
        from_text = from_term or from_comment
        return {
            "in_structured": from_structured,
            "in_reported_term": from_term,
            "in_comment": from_comment,
            "in_text": from_text,
            "readable": from_structured or from_text,
            "text_only": from_text and not from_structured,
        }

    def _gold(
        self, truth: ClinicalTruth, profile: StudyProfile, record_id: str,
        subject_id: str, onset: _dt.date, cohort: str, structured: str | None,
        term_assertion: str | None, term_value: str | None,
        comment_assertion: str | None, comment_value: str | None,
        readable: dict[str, Any], coded_concept: str, coded_code: str,
    ) -> dict[str, Any]:
        text_assertion = term_assertion or comment_assertion
        return {
            "source_record_id": record_id,
            "subject_id": subject_id,
            "study_id": profile.study_id,
            "profile": profile.profile_id,
            "truth_id": truth.truth_id,
            "cohort": cohort,
            "concept": truth.concept,
            "coded_concept": coded_concept,
            "coded_code": coded_code,
            "coded_dictionary_version": profile.dictionary_version,
            "onset": onset.isoformat(),
            "onset_offset_days": truth.onset_offset_days,
            "grade": truth.grade,
            "cumulative_exposure": truth.cumulative_exposure,
            # What was true, what each route holds, and what this rendering
            # therefore supports. Kept apart on purpose: the difference between
            # them is what the whole prototype is about.
            "true_assertion": None if truth.mucosal == "unknown" else truth.mucosal,
            "true_value": truth.mucosal_site,
            "true_availability": (
                "observed" if readable["readable"]
                else ("not_collected"
                      if not profile.collects_modifier("mucosal_involvement")
                      else "unresolved")
            ),
            "structured_assertion": {
                "Y": "present", "N": "absent", "U": "uncertain",
            }.get(structured or "", None),
            # Kept apart because the reported-term style governs AETERM only.
            # A comment record documenting an absence is legitimate whatever
            # the site's terse-or-rich habit with the term itself.
            "term_assertion": term_assertion,
            "comment_assertion": comment_assertion,
            "text_assertion": text_assertion,
            "text_value": term_value or comment_value,
            **readable,
            "true_verdict": truth.verdict_under_v1(readable["readable"]),
            # What each cumulative ablation stage could reach. Stage 1 sees
            # structured variables only; stage 2 adds the reported term; stage
            # 3 adds linked comment records.
            "verdict_stage_structured": truth.verdict_under_v1(
                readable["in_structured"]
            ),
            "verdict_stage_reported_term": truth.verdict_under_v1(
                readable["in_structured"] or readable["in_reported_term"]
            ),
            "verdict_stage_comments": truth.verdict_under_v1(readable["readable"]),
            "verdict_if_readable": truth.verdict_under_v1(True),
        }

    # -- the whole corpus ---------------------------------------------------

    def generate(self) -> GeneratedCorpus:
        corpus = GeneratedCorpus(
            tables={"dm": [], "ex": [], "ae": [], "sc": [], "co": []}
        )
        # Weighted so the corpus contains each thing the evaluation needs to
        # measure, rather than whatever a uniform sample happened to produce.
        # `graded_toxicity` gives the second phenotype definition a denominator
        # worth reporting.
        kinds = ["mucosal_in_window", "mucosal_in_window", "documented_negative",
                 "mucosal_out_of_window", "uncertain", "never_examined",
                 "graded_toxicity", "distractor"]
        shared = [
            self.sample_truth(f"T{index + 1:04d}", kinds[index % len(kinds)])
            for index in range(self.shared_truths)
        ]
        profiles = [self.profiles.profile(p) for p in self.profile_ids]

        for profile in profiles:
            for index, truth in enumerate(shared):
                subject_id = f"{profile.study_id}-SH-{index + 1:03d}"
                first_dose = _dt.date(2022, 2, 1) + _dt.timedelta(
                    days=self.rng.randint(0, 180)
                )
                self._subject_rows(profile, subject_id, corpus, first_dose, truth)
                self.render(truth, profile, subject_id, first_dose, corpus, "shared")

            for counter in range(1, self.extra_per_profile + 1):
                truth = self.sample_truth(
                    f"{profile.profile_id}-B{counter:03d}", self._pick(kinds)
                )
                subject_id = f"{profile.study_id}-BG-{counter:03d}"
                first_dose = _dt.date(2022, 1, 5) + _dt.timedelta(
                    days=self.rng.randint(0, 240)
                )
                self._subject_rows(profile, subject_id, corpus, first_dose, truth)
                self.render(truth, profile, subject_id, first_dose, corpus,
                            "background")
                corpus.truths.append(
                    self._truth_row(truth, [profile.profile_id], "background")
                )

        for truth in shared:
            corpus.truths.append(
                self._truth_row(truth, list(self.profile_ids), "shared")
            )
        corpus.manifest = self._manifest(profiles, corpus)
        return corpus

    def _truth_row(
        self, truth: ClinicalTruth, rendered_in: list[str], cohort: str
    ) -> dict[str, Any]:
        return {
            **asdict(truth),
            "onset_in_window": truth.onset_in_window,
            "cumulative_exposure": truth.cumulative_exposure,
            "verdict_if_readable": truth.verdict_under_v1(True),
            "rendered_in": rendered_in,
            "cohort": cohort,
        }

    def _manifest(
        self, profiles: list[StudyProfile], corpus: GeneratedCorpus
    ) -> dict[str, Any]:
        return {
            "generator": "aelayer.generate",
            "seed": self.seed,
            "synthetic": True,
            "notice": (
                "All records are computer generated. No real patient data is "
                "present in this repository."
            ),
            "gold_definition": "cutaneous_mucosal.v1",
            "shared_truths": self.shared_truths,
            "profiles": {
                p.profile_id: {
                    "study_id": p.study_id,
                    "label": p.label,
                    "reported_term_style": p.reported_term_style,
                    "modifier_homes": p.home_ids("mucosal_involvement"),
                    "dictionary_version": p.dictionary_version,
                    "prefer_concept": p.prefer_concept,
                    "note": p.note,
                }
                for p in profiles
            },
            "counts": {
                "profiles": len(profiles),
                "subjects": len(corpus.tables["dm"]),
                "ae_records": len(corpus.tables["ae"]),
                "linked_form_records": len(corpus.tables["sc"]),
                "comments": len(corpus.tables["co"]),
            },
        }


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

TABLE_COLUMNS: dict[str, list[str]] = {
    "dm": ["STUDYID", "USUBJID", "AGE", "SEX", "ARM", "RFSTDTC", "COUNTRY",
           "PROFILE", "SYNTHETIC"],
    "ex": ["STUDYID", "USUBJID", "EXSEQ", "EXTRT", "EXDOSE", "EXDOSU",
           "EXSTDTC", "EXENDTC", "SYNTHETIC"],
    "ae": ["STUDYID", "USUBJID", "AESEQ", "AESPID", "AETERM", "AEDECOD",
           "AEDICTVER", "AEMUCOS", "AESEV", "AEGRADE", "AESER", "AESCAT",
           "AEREL", "AEACN", "AEOUT", "AESTDTC", "AEENDTC", "SYNTHETIC"],
    "sc": ["STUDYID", "USUBJID", "IDVAR", "IDVARVAL", "SCTESTCD", "SCTEST",
           "SCORRES", "SYNTHETIC"],
    "co": ["STUDYID", "USUBJID", "IDVAR", "IDVARVAL", "COSEQ", "COVAL", "DOCID",
           "SYNTHETIC"],
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

    _write_jsonl(root / "documents.jsonl", corpus.documents)
    _write_jsonl(root / "truths.jsonl", corpus.truths)
    _write_jsonl(root / "gold.jsonl", corpus.gold)
    (root / "manifest.json").write_text(
        json.dumps(corpus.manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "README.txt").write_text(
        "SYNTHETIC DATA ONLY\n"
        "===================\n\n"
        "Every file here is computer generated by aelayer.generate. No real\n"
        "patient data, and nothing derived from real patients, is present.\n\n"
        "truths.jsonl holds the sampled ground truth and gold.jsonl the answer\n"
        "key. Both are read only by the evaluation harness.\n",
        encoding="utf-8",
    )
    return root


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True, ensure_ascii=False, default=str) + "\n"
            )


def generate_corpus(
    seed: int = 7,
    profiles: Iterable[str] | None = None,
    out_dir: str | Path | None = None,
    shared_truths: int = 24,
    extra_per_profile: int = 12,
) -> tuple[Path, dict[str, Any]]:
    generator = CorpusGenerator(
        seed=seed, profiles=profiles, shared_truths=shared_truths,
        extra_per_profile=extra_per_profile,
    )
    corpus = generator.generate()
    return write_corpus(corpus, out_dir), corpus.manifest


def _stable_index(key: str, modulus: int) -> int:
    """A deterministic spread that does not move when the RNG stream shifts."""
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulus
