"""The synthetic corpus.

Everything here is computer generated. No real patient data, and nothing
derived from real patient data, is present anywhere in this repository.

The generator samples a ``ClinicalTruth`` — what happened to a patient — and
then renders that same truth under each study profile. The truth does not
change; where its attributes end up does. That is the entire subject of the
prototype, and it is why the gold labels distinguish three different things:

``true_location``
    what was actually the case
``availability``
    whether this rendering could record it at all
``true_verdict``
    what the phenotype should conclude *from this rendering*, which is
    ``not_ascertainable`` wherever the location had nowhere to live

A fraction of events are genuinely generalised. Those are recorded as
``GENERALISED`` and are a real negative — which is what makes "no truncal
location" and "no location recorded" separable in the answer key.
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
from .profiles import StudyProfile, StudyProfiles

SYNTHETIC_FLAG = "Y"
DOC_HEADER = "*** SYNTHETIC RECORD - COMPUTER GENERATED - NOT REAL PATIENT DATA ***"

#: Truncal locations are the ones te_truncal_rash asks about.
TRUNCAL = ("CHEST", "ABDOMEN", "BACK")
NON_TRUNCAL = ("ARM", "LEG", "FACE")

#: Distractor concepts, so the corpus is not all rash.
DISTRACTORS = ("NAUSEA", "HEADACHE", "ANAEMIA", "PRURITUS", "URTICARIA")

#: Coded terms the generator assigns, by concept. None of them carries a site.
CODED_TERMS = {
    "RASH": ["Rash", "Skin rash", "Eruption"],
    "URTICARIA": ["Urticaria", "Hives"],
    "PRURITUS": ["Pruritus", "Itching"],
    "NAUSEA": ["Nausea"],
    "HEADACHE": ["Headache"],
    "ANAEMIA": ["Anaemia", "Anemia"],
}

#: What a site writes when the CRF gives it a free-text term box. Several
#: phrasings per location, so normalization is doing real work in the silver
#: standard rather than matching one canned string.
LOCATION_PHRASES = {
    "CHEST": ["chest", "anterior chest", "chest wall"],
    "ABDOMEN": ["abdomen", "periumbilical area", "flank"],
    "BACK": ["back", "upper back", "lower back"],
    "ARM": ["arm", "forearm", "upper arm"],
    "LEG": ["leg", "thigh", "calf"],
    "FACE": ["face", "cheek", "forehead"],
    "GENERALISED": ["generalised", "widespread", "diffuse"],
}

#: Adjectival forms only: these qualify the event term ("maculopapular rash"),
#: so a plural noun would not read as English.
PATTERN_PHRASES = {
    "MACULAR": ["macular"],
    "PAPULAR": ["papular"],
    "MACULOPAPULAR": ["maculopapular", "morbilliform"],
    "URTICARIAL": ["urticarial"],
    "VESICULAR": ["vesicular"],
}

#: A study whose CRF offers a fixed list of terms and nothing else. None of
#: these can carry a site, which is the point.
PRESPECIFIED_TERMS = ["Rash", "Skin disorder", "Dermatitis", "Skin reaction"]

#: Real reported terms contain site words nobody wrote into a lexicon. These
#: are deliberately absent from concepts.yaml, so the extractor meets them and
#: abstains — which is the behaviour the abstention rate is there to measure.
UNLEXICONED_SITES = {
    "CHEST": ["torso", "upper trunk"],
    "ABDOMEN": ["midriff", "tummy"],
    "BACK": ["shoulder blade area", "dorsum"],
    "ARM": ["limb", "extremity"],
    "LEG": ["lower limb", "extremity"],
    "FACE": ["visage", "head and neck"],
    "GENERALISED": ["multiple sites", "several areas"],
}

#: How a rich reported term can differ from what the structured qualifier says.
#: Real records disagree with themselves, and a silver standard with no
#: disagreements demonstrates nothing about adjudication.
TEXT_STATED = "stated"
TEXT_OMITTED = "omitted"
TEXT_UNLEXICONED = "unlexiconed"
TEXT_DISCREPANT = "discrepant"

CONNECTORS = ["on", "over", "affecting", "involving"]

#: Sites that read as a region rather than a named part, so they take no
#: article: "rash affecting widespread areas" is not English, "generalised rash"
#: is. Keeping the generator's prose grammatical matters because the extractor
#: reads it.
NO_ARTICLE = {"generalised", "generalized", "widespread", "diffuse"}


def site_phrase(connector: str, site: str) -> str:
    """Join a connector to a site phrase without producing "over the the chest"."""
    if site in NO_ARTICLE:
        return f"{connector} {site} areas"
    return f"{connector} the {site}"

#: US and UK spellings of the same words, chosen per profile.
SPELLINGS = {
    "us": {"generalised": "generalized", "oedema": "edema", "anaemia": "anemia"},
    "uk": {},
}


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------


@dataclass
class ClinicalTruth:
    """What happened, before any study wrote it down."""

    truth_id: str
    concept: str
    location: str | None            # a catalogue value, or None for "no site"
    pattern: str | None
    onset_offset_days: int          # from first exposure
    duration_days: int
    severity_steps: list[tuple[int, str]]
    seriousness: bool
    seriousness_criteria: list[str]
    relatedness: str
    action_taken: str
    outcome: str
    quality: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def peak_severity(self) -> str:
        order = ["mild", "moderate", "severe"]
        return max((s for _d, s in self.severity_steps), key=order.index)

    @property
    def onset_in_window(self) -> bool:
        return 0 <= self.onset_offset_days <= 14

    @property
    def location_is_truncal(self) -> bool:
        return self.location in TRUNCAL

    def verdict_under(self, location_available: bool) -> str:
        """The verdict te_truncal_rash v1 should reach for this rendering.

        Precedence, mirroring the evaluator: a requirement that is present and
        fails settles the case as a negative, whatever else is missing. Only
        when nothing has failed does an unavailable requirement make the
        episode unascertainable.
        """
        if self.concept != "RASH":
            return "not_case"
        if not self.onset_in_window:
            return "not_case"
        if location_available:
            return "case" if self.location_is_truncal else "not_case"
        return "not_ascertainable"


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


@dataclass
class GeneratedCorpus:
    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    documents: list[dict[str, Any]] = field(default_factory=list)
    truths: list[dict[str, Any]] = field(default_factory=list)
    gold_records: list[dict[str, Any]] = field(default_factory=list)
    gold_episodes: list[dict[str, Any]] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)


class CorpusGenerator:
    def __init__(
        self,
        seed: int = 7,
        profiles: Iterable[str] | None = None,
        shared_truths: int = 24,
        extra_per_profile: int = 14,
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

    # -- sampling -----------------------------------------------------------

    def _pick(self, options):
        return options[self.rng.randrange(len(options))]

    def sample_truth(self, truth_id: str, kind: str) -> ClinicalTruth:
        """One clinical truth of a requested kind.

        The kinds exist so the corpus contains each thing the evaluation needs
        to measure, rather than whatever a uniform sample happened to produce.
        """
        concept = "RASH"
        pattern = self._pick(sorted(PATTERN_PHRASES))
        if kind == "truncal_in_window":
            location, offset = self._pick(TRUNCAL), self.rng.randint(0, 14)
        elif kind == "truncal_out_of_window":
            location, offset = self._pick(TRUNCAL), self.rng.randint(20, 90)
        elif kind == "non_truncal":
            location, offset = self._pick(NON_TRUNCAL), self.rng.randint(0, 14)
        elif kind == "generalised":
            # A real negative: the site was recorded, and it is not truncal.
            location, offset = "GENERALISED", self.rng.randint(0, 14)
        elif kind == "distractor":
            concept = self._pick(DISTRACTORS)
            location = self._pick(TRUNCAL) if concept in ("URTICARIA", "PRURITUS") else None
            offset = self.rng.randint(0, 30)
            pattern = None
        else:
            raise ValueError(f"unknown truth kind {kind!r}")

        serious = self.rng.random() < 0.12
        return ClinicalTruth(
            truth_id=truth_id,
            concept=concept,
            location=location,
            pattern=pattern,
            onset_offset_days=offset,
            duration_days=self.rng.randint(1, 21),
            severity_steps=self._severity_steps(),
            seriousness=serious,
            seriousness_criteria=(
                [self._pick(["hospitalisation", "other_medically_important"])]
                if serious else []
            ),
            relatedness=self._pick(
                ["not_related", "unlikely", "possible", "possible", "probable"]
            ),
            action_taken=self._pick(
                ["dose_not_changed", "dose_not_changed", "dose_reduced",
                 "drug_interrupted", "drug_withdrawn"]
            ),
            outcome=self._pick(
                ["recovered", "recovered", "recovering", "not_recovered"]
            ),
            quality=(
                [self._pick(["itchy", "painful", "spreading", "confluent"])]
                if self.rng.random() < 0.5 else []
            ),
            note=kind,
        )

    def _severity_steps(self) -> list[tuple[int, str]]:
        first = self._pick(["mild", "mild", "moderate", "severe"])
        if self.rng.random() < 0.3 and first != "severe":
            worse = "severe" if first == "moderate" else "moderate"
            return [(0, first), (self.rng.randint(1, 5), worse)]
        return [(0, first)]

    # -- text ---------------------------------------------------------------

    def _spell(self, text: str, profile: StudyProfile) -> str:
        for uk, us in SPELLINGS.get(profile.conventions.get("spelling", "uk"), {}).items():
            text = text.replace(uk, us)
        return text

    def _text_site(self, truth: ClinicalTruth) -> tuple[str | None, str | None, str]:
        """What a free-text field says about the site, and what it resolves to.

        Four outcomes, in the proportions a real corpus has them: the site
        stated plainly; the site omitted; the site written in words no lexicon
        carries; and the site disagreeing with the study's own structured
        qualifier. The last is seeded only *within* a verdict class — a truncal
        site is confused with another truncal site — so that disagreements show
        up in the silver numbers without contaminating the phenotype numbers.
        That is a deliberate corpus design choice, and the README says so.
        """
        if truth.location is None:
            return None, None, TEXT_OMITTED
        draw = self.rng.random()
        if draw < 0.12:
            return None, None, TEXT_OMITTED
        if draw < 0.22:
            return self._pick(UNLEXICONED_SITES[truth.location]), None, TEXT_UNLEXICONED
        if draw < 0.30:
            pool = (
                [v for v in TRUNCAL if v != truth.location]
                if truth.location in TRUNCAL
                else [v for v in NON_TRUNCAL if v != truth.location]
            )
            if pool:
                other = self._pick(pool)
                return self._pick(LOCATION_PHRASES[other]), other, TEXT_DISCREPANT
        return self._pick(LOCATION_PHRASES[truth.location]), truth.location, TEXT_STATED

    def reported_term(
        self, truth: ClinicalTruth, profile: StudyProfile
    ) -> tuple[str, str | None, str]:
        """What the investigator typed into the term box.

        Returns the text, the location it actually states (or None), and which
        of the four text outcomes produced it. Under a rich style the phrase
        carries the site, because that is what a site writes when the coded term
        cannot. Under a terse style it does not, and the site has to live
        somewhere else or nowhere.
        """
        style = profile.reported_term_style
        base = {
            "RASH": "rash", "URTICARIA": "urticaria", "PRURITUS": "pruritus",
            "NAUSEA": "nausea", "HEADACHE": "headache", "ANAEMIA": "anaemia",
        }[truth.concept]

        if style == "prespecified":
            term = (
                self._pick(PRESPECIFIED_TERMS) if truth.concept == "RASH"
                else base.capitalize()
            )
            return term, None, TEXT_OMITTED
        if style == "terse" or truth.location is None:
            return self._spell(base, profile), None, TEXT_OMITTED

        site, resolved, kind = self._text_site(truth)
        parts = []
        if truth.pattern and self.rng.random() < 0.5:
            parts.append(self._pick(PATTERN_PHRASES[truth.pattern]))
        # "skin rash" is a phrasing sites use; "skin pruritus" is not.
        if truth.concept == "RASH" and self.rng.random() < 0.4:
            parts.append("skin " + base)
        else:
            parts.append(base)
        phrase = " ".join(parts)
        if site is None:
            return self._spell(phrase, profile), None, kind
        connector = self._pick(CONNECTORS)
        return (
            self._spell(f"{phrase} {site_phrase(connector, site)}", profile),
            resolved,
            kind,
        )

    def comment_text(
        self, truth: ClinicalTruth, profile: StudyProfile
    ) -> tuple[str, str | None, str]:
        """A comment written about one AE record, and what site it states."""
        site, resolved, kind = self._text_site(truth)
        if site is None:
            body = self._pick([
                "Investigator comment: event reviewed, no change to study drug.",
                "Site clarification: the event was documented at the study visit.",
            ])
            return self._spell(body, profile), None, kind
        connector = self._pick(["on", "over", "involving"])
        lead = self._pick([
            "Site clarification: rash noted {where}.",
            "Investigator comment: eruption {where}, no other areas involved.",
            "Additional detail: the rash was {where} at the study visit.",
        ])
        body = lead.format(where=site_phrase(connector, site))
        if truth.quality:
            body += f" Described as {self._pick(truth.quality)}."
        return self._spell(body, profile), resolved, kind

    # -- one subject --------------------------------------------------------

    def _subject_rows(
        self, profile: StudyProfile, subject_id: str, corpus: GeneratedCorpus,
        first_dose: _dt.date,
    ) -> _dt.date:
        corpus.tables["dm"].append({
            "STUDYID": profile.study_id, "USUBJID": subject_id,
            "AGE": self.rng.randint(24, 78),
            "SEX": self._pick(["M", "F"]),
            "ARM": self._pick(["Drug 10 mg", "Drug 20 mg", "Placebo"]),
            "RFSTDTC": first_dose.isoformat(),
            "COUNTRY": self._pick(["USA", "GBR", "DEU", "JPN"]),
            "PROFILE": profile.profile_id,
            "SYNTHETIC": SYNTHETIC_FLAG,
        })
        dose = self._pick([10, 20])
        for index in range(self.rng.randint(2, 4)):
            start = first_dose + _dt.timedelta(days=28 * index)
            if index and self.rng.random() < 0.5:
                dose *= 2
            corpus.tables["ex"].append({
                "STUDYID": profile.study_id, "USUBJID": subject_id,
                "EXSEQ": index + 1, "EXTRT": "STUDY DRUG", "EXDOSE": dose,
                "EXDOSU": "mg", "EXSTDTC": start.isoformat(),
                "EXENDTC": (start + _dt.timedelta(days=27)).isoformat(),
                "SYNTHETIC": SYNTHETIC_FLAG,
            })
        return first_dose

    def render(
        self, truth: ClinicalTruth, profile: StudyProfile, subject_id: str,
        first_dose: _dt.date, corpus: GeneratedCorpus, event_index: int,
    ) -> dict[str, Any]:
        """Write one truth into one study's tables, and record the answer key."""
        onset = first_dose + _dt.timedelta(days=truth.onset_offset_days)
        end = onset + _dt.timedelta(days=truth.duration_days)
        split = profile.splits_on_severity_change() and len(truth.severity_steps) > 1
        # A study that does not split records the whole event as one row, which
        # starts when the event started and carries the worst severity reached.
        # Taking the last step's day offset here would move the onset, and the
        # onset is the one thing every window in the corpus is measured from.
        steps = truth.severity_steps if split else [(0, truth.peak_severity)]

        # The reported term is written once per event, not once per record: a
        # study that splits on severity change repeats the investigator's own
        # words across the rows it splits into.
        term, text_location, text_kind = self.reported_term(truth, profile)
        comment_location, comment_kind = None, TEXT_OMITTED

        record_ids: list[str] = []
        previous_id: str | None = None
        for step_index, (day_offset, severity) in enumerate(steps):
            record_id = f"{subject_id}-AE-{event_index:02d}{step_index + 1}"
            record_start = onset + _dt.timedelta(days=day_offset)
            record_end = (
                end if step_index == len(steps) - 1
                else onset + _dt.timedelta(days=steps[step_index + 1][0])
            )
            terminal = truth.outcome in ("recovered", "recovered_with_sequelae", "fatal")
            row = self._ae_row(
                truth, profile, subject_id, record_id, severity, term,
                record_start,
                record_end if terminal or step_index < len(steps) - 1 else None,
                previous_id,
            )
            corpus.tables["ae"].append(row)
            comment_location, comment_kind = self._supplemental(
                truth, profile, subject_id, record_id, corpus
            )
            in_text = self._text_location(
                profile, text_location, text_kind, comment_location, comment_kind
            )
            gold = self._gold_record(
                truth, profile, row, severity, record_start, in_text,
            )
            corpus.gold_records.append(gold)
            record_ids.append(record_id)
            previous_id = record_id

        in_text = self._text_location(
            profile, text_location, text_kind, comment_location, comment_kind
        )
        return {
            "record_ids": record_ids, "onset": onset, "end": end,
            "text_location": in_text["text_location"],
            "text_outcome": in_text["text_outcome"],
            "location_available": self._location_available(truth, profile, in_text),
        }

    def _ae_row(
        self, truth: ClinicalTruth, profile: StudyProfile, subject_id: str,
        record_id: str, severity: str, term: str, start: _dt.date,
        end: _dt.date | None, continuation_of: str | None,
    ) -> dict[str, Any]:
        coded = self._pick(CODED_TERMS[truth.concept])
        homes = profile.home_kinds("location")
        pattern_homes = profile.home_kinds("pattern")

        def cell(variable: str, value: Any) -> Any:
            return value if profile.collects_variable(variable) else ""

        location_cell = ""
        if "AELOC" in homes and truth.location:
            location_cell = truth.location
        pattern_cell = ""
        if "AEPATT" in pattern_homes and truth.pattern:
            pattern_cell = truth.pattern

        return {
            "STUDYID": profile.study_id,
            "USUBJID": subject_id,
            "AESEQ": int(record_id.rsplit("-", 1)[-1]),
            "AESPID": record_id,
            "AETERM": term,
            "AEDECOD": cell("AEDECOD", coded),
            "AEDICTVER": profile.dictionary_version,
            "AELOC": cell("AELOC", location_cell),
            "AEPATT": cell("AEPATT", pattern_cell),
            "AESEV": cell("AESEV", severity),
            "AESER": cell("AESER", "Y" if truth.seriousness else "N"),
            "AESCAT": "|".join(truth.seriousness_criteria) if truth.seriousness else "",
            "AEREL": cell("AEREL", truth.relatedness),
            "AEACN": cell("AEACN", truth.action_taken),
            "AEOUT": cell("AEOUT", truth.outcome),
            "AESTDTC": cell("AESTDTC", self._date(start, profile)),
            "AEENDTC": cell("AEENDTC", self._date(end, profile) if end else ""),
            "AECONTRP": continuation_of or "",
            "SYNTHETIC": SYNTHETIC_FLAG,
        }

    @staticmethod
    def _date(value: _dt.date | None, profile: StudyProfile) -> str:
        if value is None:
            return ""
        if profile.conventions.get("date_style") == "dmy":
            return value.strftime("%d-%b-%Y")
        return value.isoformat()

    def _supplemental(
        self, truth: ClinicalTruth, profile: StudyProfile, subject_id: str,
        record_id: str, corpus: GeneratedCorpus,
    ) -> tuple[str | None, str]:
        homes = profile.home_kinds("location")
        if "sponsor_variable" in homes and truth.location:
            corpus.tables["suppae"].append({
                "STUDYID": profile.study_id, "RDOMAIN": "AE",
                "USUBJID": subject_id, "IDVAR": "AESPID", "IDVARVAL": record_id,
                "QNAM": profile.sponsor_variable_name or "RASHSITE",
                "QLABEL": "Rash site",
                "QVAL": profile.sponsor_code_for(truth.location) or "",
                "SYNTHETIC": SYNTHETIC_FLAG,
            })
        if "comment" not in homes or truth.location is None:
            return None, TEXT_OMITTED
        doc_id = f"{record_id}-CO"
        text, resolved, kind = self.comment_text(truth, profile)
        corpus.tables["co"].append({
            "STUDYID": profile.study_id, "RDOMAIN": "AE",
            "USUBJID": subject_id, "IDVAR": "AESPID", "IDVARVAL": record_id,
            "COSEQ": 1, "COVAL": text, "DOCID": doc_id,
            "SYNTHETIC": SYNTHETIC_FLAG,
        })
        corpus.documents.append({
            "doc_id": doc_id, "study_id": profile.study_id,
            "subject_id": subject_id, "source_record_id": record_id,
            "kind": "comment", "header": DOC_HEADER, "text": text,
        })
        return resolved, kind

    # -- the answer key -----------------------------------------------------

    def _text_location(
        self, profile: StudyProfile, term_location: str | None, term_kind: str,
        comment_location: str | None, comment_kind: str,
    ) -> dict[str, Any]:
        """What the free text of this record actually states about the site.

        Only the homes this profile uses count: a rich reported term in a study
        whose location home is a structured variable is still text a reader
        could use, but it is not where the study keeps the answer.
        """
        homes = profile.home_kinds("location")
        if "comment" in homes:
            return {"text_location": comment_location, "text_outcome": comment_kind}
        if "reported_term" in homes:
            return {"text_location": term_location, "text_outcome": term_kind}
        return {"text_location": None, "text_outcome": TEXT_OMITTED}

    def _structured_location(
        self, truth: ClinicalTruth, profile: StudyProfile
    ) -> str | None:
        homes = profile.home_kinds("location")
        if truth.location is None:
            return None
        return truth.location if {"AELOC", "sponsor_variable"} & set(homes) else None

    def _location_available(
        self, truth: ClinicalTruth, profile: StudyProfile, in_text: dict[str, Any]
    ) -> bool:
        """Could this rendering carry the location *and* did it?

        Not the same question as whether the profile collects it. A study whose
        only home is the reported term did not record a site if the
        investigator did not write one, and a phrase no lexicon carries is a
        site nobody downstream can read.
        """
        if truth.location is None or not profile.collects_attribute("location"):
            return False
        if self._structured_location(truth, profile) is not None:
            return True
        return in_text["text_location"] is not None

    def _availability(
        self, truth: ClinicalTruth, profile: StudyProfile, attribute: str,
        in_text: dict[str, Any] | None = None,
    ) -> str:
        value = truth.location if attribute == "location" else truth.pattern
        if not profile.collects_attribute(attribute):
            return "not_collected_by_protocol"
        if value is None:
            return "unknown"
        if attribute == "location" and in_text is not None:
            return "collected" if self._location_available(truth, profile, in_text) \
                else "unknown"
        return "collected"

    def _gold_record(
        self, truth: ClinicalTruth, profile: StudyProfile, row: dict[str, Any],
        severity: str, start: _dt.date, in_text: dict[str, Any],
    ) -> dict[str, Any]:
        homes = profile.home_kinds("location")
        structured = self._structured_location(truth, profile)
        return {
            "source_record_id": row["AESPID"],
            "study_id": profile.study_id,
            "profile": profile.profile_id,
            "truth_id": truth.truth_id,
            "concept": truth.concept,
            "true_location": truth.location,
            "true_pattern": truth.pattern,
            "location_homes": homes,
            "pattern_homes": profile.home_kinds("pattern"),
            # Where the location can be read from in *this* record, which is
            # what the silver standard masks and what the extractor must find.
            # What each route holds for this record. The silver standard
            # compares the last two against each other, and they are allowed to
            # disagree — that is why it is a silver standard.
            "structured_location": structured,
            "text_location": in_text["text_location"],
            "text_outcome": in_text["text_outcome"],
            "location_in_structured": structured is not None,
            "location_in_text": in_text["text_location"] is not None,
            "availability": {
                "location": self._availability(truth, profile, "location", in_text),
                "pattern": self._availability(truth, profile, "pattern"),
            },
            "reported_term": row["AETERM"],
            "severity": severity,
            "onset": start.isoformat(),
            "onset_offset_days": truth.onset_offset_days,
        }

    # -- the whole corpus ---------------------------------------------------

    def generate(self) -> GeneratedCorpus:
        corpus = GeneratedCorpus(
            tables={"dm": [], "ex": [], "ae": [], "suppae": [], "co": []}
        )
        kinds = ["truncal_in_window", "truncal_in_window", "truncal_out_of_window",
                 "non_truncal", "generalised", "distractor"]

        # The shared cohort: one truth, rendered under every profile. This is
        # what representation invariance is measured over.
        shared = [
            self.sample_truth(f"T{index + 1:04d}", kinds[index % len(kinds)])
            for index in range(self.shared_truths)
        ]

        profiles = [self.profiles.profile(pid) for pid in self.profile_ids]
        for profile in profiles:
            for index, truth in enumerate(shared):
                subject_id = f"{profile.study_id}-SH-{index + 1:03d}"
                first_dose = _dt.date(2021, 3, 1) + _dt.timedelta(
                    days=self.rng.randint(0, 200)
                )
                self._subject_rows(profile, subject_id, corpus, first_dose)
                emitted = self.render(truth, profile, subject_id, first_dose, corpus, 1)
                corpus.gold_episodes.append(
                    self._gold_episode(truth, profile, subject_id, emitted, "shared")
                )

            counter = 0
            for _ in range(self.extra_per_profile):
                counter += 1
                truth = self.sample_truth(
                    f"{profile.profile_id}-B{counter:03d}", self._pick(kinds)
                )
                subject_id = f"{profile.study_id}-BG-{counter:03d}"
                first_dose = _dt.date(2021, 1, 4) + _dt.timedelta(
                    days=self.rng.randint(0, 300)
                )
                self._subject_rows(profile, subject_id, corpus, first_dose)
                emitted = self.render(truth, profile, subject_id, first_dose, corpus, 1)
                corpus.gold_episodes.append(
                    self._gold_episode(truth, profile, subject_id, emitted, "background")
                )
                corpus.truths.append(self._truth_row(truth, [profile.profile_id],
                                                     "background"))

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
            "peak_severity": truth.peak_severity,
            "onset_in_window": truth.onset_in_window,
            "location_is_truncal": truth.location_is_truncal,
            "verdict_if_location_available": truth.verdict_under(True),
            "rendered_in": rendered_in,
            "cohort": cohort,
        }

    def _gold_episode(
        self, truth: ClinicalTruth, profile: StudyProfile, subject_id: str,
        emitted: dict[str, Any], cohort: str,
    ) -> dict[str, Any]:
        available = emitted["location_available"]
        structured = self._structured_location(truth, profile)
        return {
            "truth_id": truth.truth_id,
            "study_id": profile.study_id,
            "profile": profile.profile_id,
            "subject_id": subject_id,
            "cohort": cohort,
            "concept": truth.concept,
            "source_record_ids": emitted["record_ids"],
            "n_records": len(emitted["record_ids"]),
            "episode_start": emitted["onset"].isoformat(),
            "episode_end": emitted["end"].isoformat(),
            "onset_offset_days": truth.onset_offset_days,
            "true_location": truth.location,
            "true_pattern": truth.pattern,
            "location_available": available,
            "structured_location": structured,
            "text_location": emitted["text_location"],
            "text_outcome": emitted["text_outcome"],
            "location_in_structured": structured is not None,
            "location_in_text": emitted["text_location"] is not None,
            # The route the location can *only* come from, which is what the
            # value ablation counts.
            "text_only": structured is None and emitted["text_location"] is not None,
            # What this rendering supports, and what the truth was regardless.
            "true_verdict": truth.verdict_under(available),
            "verdict_if_location_available": truth.verdict_under(True),
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
            "gold_case_definition": "te_truncal_rash.v1",
            "shared_truths": self.shared_truths,
            "profiles": {
                p.profile_id: {
                    "study_id": p.study_id,
                    "label": p.label,
                    "reported_term_style": p.reported_term_style,
                    "location_home": p.home_kinds("location"),
                    "pattern_home": p.home_kinds("pattern"),
                    "dictionary_version": p.dictionary_version,
                    "sponsor_variable": p.sponsor_variable_name,
                    "conventions": p.conventions,
                    "note": p.note,
                }
                for p in profiles
            },
            "counts": {
                "profiles": len(profiles),
                "subjects": len(corpus.tables["dm"]),
                "ae_records": len(corpus.tables["ae"]),
                "suppae_records": len(corpus.tables["suppae"]),
                "comments": len(corpus.tables["co"]),
                "episodes_expected": len(corpus.gold_episodes),
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
           "AEDICTVER", "AELOC", "AEPATT", "AESEV", "AESER", "AESCAT", "AEREL",
           "AEACN", "AEOUT", "AESTDTC", "AEENDTC", "AECONTRP", "SYNTHETIC"],
    "suppae": ["STUDYID", "RDOMAIN", "USUBJID", "IDVAR", "IDVARVAL", "QNAM",
               "QLABEL", "QVAL", "SYNTHETIC"],
    "co": ["STUDYID", "RDOMAIN", "USUBJID", "IDVAR", "IDVARVAL", "COSEQ",
           "COVAL", "DOCID", "SYNTHETIC"],
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
            handle.write(
                json.dumps(row, sort_keys=True, ensure_ascii=False, default=str) + "\n"
            )


def generate_corpus(
    seed: int = 7,
    profiles: Iterable[str] | None = None,
    out_dir: str | Path | None = None,
    shared_truths: int = 24,
    extra_per_profile: int = 14,
) -> tuple[Path, dict[str, Any]]:
    generator = CorpusGenerator(
        seed=seed, profiles=profiles, shared_truths=shared_truths,
        extra_per_profile=extra_per_profile,
    )
    corpus = generator.generate()
    return write_corpus(corpus, out_dir), corpus.manifest
