"""Core data models.

The two artifacts this system keeps deliberately apart:

``EventObject``
    What happened to this patient.  Per-record, extracted, evidence-bearing,
    stamped with the extractor version that produced it.  Produced by
    ``aelayer.extract``.  It carries *no* evidence state and *no* case verdict.

``PhenotypeDefinition``
    For this scientific question, which event objects make this patient a case.
    A declarative, versioned rule over event objects.  Configuration, not code.
    Loaded by ``aelayer.phenotype.loader``.

Severity and seriousness are separate fields and are never collapsed.  Severity
is the intensity of the event.  Seriousness is a regulatory category defined by
outcome.  A mild event can be serious; a severe event can be non-serious.
"""

from __future__ import annotations

import datetime as _dt
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --------------------------------------------------------------------------
# Enumerations (as Literals so they serialise as plain strings)
# --------------------------------------------------------------------------

Assertion = Literal[
    "present",
    "absent",
    "hypothetical",
    "historical",
    "family_history",
    "uncertain",
]
ASSERTION_VALUES: tuple[str, ...] = (
    "present",
    "absent",
    "hypothetical",
    "historical",
    "family_history",
    "uncertain",
)

Severity = Literal["mild", "moderate", "severe"]
SEVERITY_VALUES: tuple[str, ...] = ("mild", "moderate", "severe")

Seriousness = Literal[
    "death",
    "life_threatening",
    "hospitalisation",
    "disability",
    "congenital_anomaly",
    "other_medically_important",
]
SERIOUSNESS_VALUES: tuple[str, ...] = (
    "death",
    "life_threatening",
    "hospitalisation",
    "disability",
    "congenital_anomaly",
    "other_medically_important",
)

Relatedness = Literal[
    "not_related", "unlikely", "possible", "probable", "definite", "unknown"
]
RELATEDNESS_VALUES: tuple[str, ...] = (
    "not_related",
    "unlikely",
    "possible",
    "probable",
    "definite",
    "unknown",
)

ActionTaken = Literal[
    "dose_reduced", "dose_interrupted", "drug_withdrawn", "none", "unknown"
]
ACTION_TAKEN_VALUES: tuple[str, ...] = (
    "dose_reduced",
    "dose_interrupted",
    "drug_withdrawn",
    "none",
    "unknown",
)

Rechallenge = Literal["not_done", "done_recurred", "done_no_recurrence"]
RECHALLENGE_VALUES: tuple[str, ...] = (
    "not_done",
    "done_recurred",
    "done_no_recurrence",
)

Outcome = Literal["resolved", "resolving", "not_resolved", "fatal", "unknown"]
OUTCOME_VALUES: tuple[str, ...] = (
    "resolved",
    "resolving",
    "not_resolved",
    "fatal",
    "unknown",
)

EvidenceState = Literal["explicit", "supported", "possible", "absent", "none"]
EVIDENCE_STATE_VALUES: tuple[str, ...] = (
    "explicit",
    "supported",
    "possible",
    "absent",
    "none",
)

#: Ordering used when a subject has several qualifying events and the strongest
#: state must win.  Higher is stronger.
EVIDENCE_STATE_RANK: dict[str, int] = {
    "none": 0,
    "absent": 1,
    "possible": 2,
    "supported": 3,
    "explicit": 4,
}

Verdict = Literal["case", "review", "excluded"]

DefinitionStatus = Literal["draft", "frozen", "superseded"]


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


class Span(BaseModel):
    """A character range in a source document backing one extracted value.

    Every non-null field on an ``EventObject`` must be backed by at least one
    span.  A derived value without provenance is a bug, not a degraded result.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    field: str = Field(description="The EventObject field this span supports.")
    extracted_value: str = Field(
        description="The value derived from this span, rendered as text."
    )
    text: str = Field(
        default="", description="The literal source substring, for display."
    )

    @model_validator(mode="after")
    def _check_range(self) -> "Span":
        if self.end < self.start:
            raise ValueError(f"span end {self.end} precedes start {self.start}")
        return self

    def key(self) -> tuple[str, int, int, str, str]:
        return (self.doc_id, self.start, self.end, self.field, self.extracted_value)


class LabValue(BaseModel):
    """A laboratory result, kept in both reported and canonical units.

    Trials report glucose in mg/dL or mmol/L depending on region.  A threshold
    rule that ignores this silently misclassifies entire studies, so the
    canonical value is computed once, at extraction, and carried alongside the
    value as reported.
    """

    model_config = ConfigDict(extra="forbid")

    test: str = Field(description="Catalogue key, e.g. GLUCOSE.")
    value: float
    unit: str = Field(description="Unit as reported in the source.")
    canonical_value: float | None = Field(
        default=None, description="Value converted to the catalogue's canonical unit."
    )
    canonical_unit: str | None = None
    collection_date: _dt.date | None = None
    span: Span


class SymptomMention(BaseModel):
    """A normalised symptom concept with its own span."""

    model_config = ConfigDict(extra="forbid")

    symptom: str
    span: Span


# --------------------------------------------------------------------------
# Event object
# --------------------------------------------------------------------------


class EventObject(BaseModel):
    """One record per (patient, candidate clinical concept, occurrence).

    Produced by the extraction engine.  It reports what the text and the
    structured tables say.  It deliberately does not assign an evidence state
    and does not decide whether the subject is a case; that is the phenotype
    definition's job.
    """

    model_config = ConfigDict(extra="forbid")

    event_id: str
    subject_id: str
    study_id: str
    doc_id: str
    source_record_id: str | None = Field(
        default=None, description="AE table record this event was derived from."
    )

    concept_id: str = Field(description="Reference into the concept catalogue.")
    coded_term: str | None = Field(
        default=None, description="Existing MedDRA PT where one exists, verbatim."
    )
    coded_term_version: str | None = Field(
        default=None, description="Dictionary version under which coding occurred."
    )
    verbatim_term: str | None = None

    assertion: Assertion = "present"
    #: How this concept came to be identified on this record: any of
    #: ``lexicon``, ``lexicon_fuzzy``, ``abbreviation``, ``coded_term``,
    #: ``contextual``. A rule that asks for a verbatim mention needs to be able
    #: to tell a mention from an inference drawn off surrounding evidence.
    concept_match_kinds: list[str] = Field(default_factory=list)
    symptoms: list[SymptomMention] = Field(default_factory=list)
    labs: list[LabValue] = Field(default_factory=list)

    onset_date: _dt.date | None = None
    onset_offset_days: int | None = None
    anchor_event: str | None = Field(
        default=None, description="The exposure event the offset is relative to."
    )
    anchor_date: _dt.date | None = None

    severity: Severity | None = Field(
        default=None, description="Intensity of the event. Never seriousness."
    )
    seriousness: list[Seriousness] = Field(
        default_factory=list,
        description="Regulatory categories. Never collapsed into severity.",
    )
    relatedness: Relatedness | None = None
    action_taken: ActionTaken | None = None
    rechallenge: Rechallenge | None = None
    rescue_treatment: bool = False
    outcome: Outcome | None = None

    evidence: list[Span] = Field(default_factory=list)
    confidence: dict[str, float] = Field(
        default_factory=dict, description="Extractor confidence, per field."
    )
    extractor_version: str = ""

    # -- provenance helpers -------------------------------------------------

    #: Fields that must be backed by a span whenever they hold a value.
    PROVENANCE_FIELDS: tuple[str, ...] = (
        "concept_id",
        "coded_term",
        "assertion",
        "onset_date",
        "onset_offset_days",
        "severity",
        "seriousness",
        "relatedness",
        "action_taken",
        "rechallenge",
        "rescue_treatment",
        "outcome",
    )

    def spans_for(self, field: str) -> list[Span]:
        return [s for s in self.evidence if s.field == field]

    def missing_provenance(self) -> list[str]:
        """Return the names of populated fields that carry no span.

        Empty list means every value in this object traces to source text or a
        source table row.  Anything else is a defect.
        """
        missing: list[str] = []
        for name in self.PROVENANCE_FIELDS:
            value = getattr(self, name)
            populated = value not in (None, [], False, "")
            if name == "assertion":
                populated = True  # always set, always needs a cue or default span
            if populated and not self.spans_for(name):
                missing.append(name)
        for sym in self.symptoms:
            if sym.span is None:  # pragma: no cover - schema forbids
                missing.append("symptoms")
        return missing

    def has_full_provenance(self) -> bool:
        return not self.missing_provenance()


# --------------------------------------------------------------------------
# Phenotype definition
# --------------------------------------------------------------------------


class ConceptSelector(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary: str
    include_coded_terms: bool = True
    include_lexicon: bool = True
    group: str | None = Field(
        default=None,
        description=(
            "Named concept group from concepts.yaml. Grouping above term level "
            "is always an explicit list; no hierarchy is walked as subsumption."
        ),
    )


class AssertionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    require: list[Assertion] = Field(default_factory=lambda: ["present"])
    route_to_review: list[Assertion] = Field(default_factory=list)
    exclude: list[Assertion] = Field(default_factory=list)

    @model_validator(mode="after")
    def _no_overlap(self) -> "AssertionPolicy":
        seen: dict[str, str] = {}
        for bucket, values in (
            ("require", self.require),
            ("route_to_review", self.route_to_review),
            ("exclude", self.exclude),
        ):
            for value in values:
                if value in seen:
                    raise ValueError(
                        f"assertion {value!r} appears in both {seen[value]} and {bucket}"
                    )
                seen[value] = bucket
        return self


class AnchorSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event: str
    source_domain: str = "EX"
    index_rule: Literal["first_occurrence", "most_recent_before_onset"] = (
        "first_occurrence"
    )


class WindowSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unit: Literal["days"] = "days"
    min: int = 0
    max: int = 14
    on_unresolved_onset: Literal["case", "review", "exclude"] = "review"

    @model_validator(mode="after")
    def _ordered(self) -> "WindowSpec":
        if self.max < self.min:
            raise ValueError(f"window max {self.max} precedes min {self.min}")
        return self

    def contains(self, offset: int) -> bool:
        return self.min <= offset <= self.max


class EvidenceRule(BaseModel):
    """One ordered rule: when ``when`` matches an event, assign ``state``."""

    model_config = ConfigDict(extra="forbid")

    id: str
    state: EvidenceState
    when: dict[str, Any]
    description: str | None = None


class CaseDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary_set: list[EvidenceState] = Field(
        default_factory=lambda: ["explicit", "supported"]
    )
    review_set: list[EvidenceState] = Field(default_factory=lambda: ["possible"])
    excluded: list[EvidenceState] = Field(default_factory=lambda: ["absent", "none"])

    @model_validator(mode="after")
    def _no_overlap(self) -> "CaseDefinition":
        seen: dict[str, str] = {}
        for bucket, values in (
            ("primary_set", self.primary_set),
            ("review_set", self.review_set),
            ("excluded", self.excluded),
        ):
            for value in values:
                if value in seen:
                    raise ValueError(
                        f"evidence state {value!r} appears in both "
                        f"{seen[value]} and {bucket}"
                    )
                seen[value] = bucket
        return self

    def verdict_for(self, state: str) -> Verdict:
        if state in self.primary_set:
            return "case"
        if state in self.review_set:
            return "review"
        return "excluded"


class ReportingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    counts_by_state: bool = True
    require_evidence_span: bool = True
    report_review_set_separately: bool = True


class PhenotypeDefinition(BaseModel):
    """A versioned scientific artifact with its own lifecycle.

    Changing what qualifies as a case creates a new version; it never rewrites
    the cohort a prior analysis was built on.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    version: int = Field(ge=1)
    status: DefinitionStatus = "draft"
    label: str
    description: str = ""
    supersedes: str | None = None
    authors: list[str] = Field(default_factory=list)
    created: _dt.date | None = None

    concept: ConceptSelector
    assertion: AssertionPolicy = Field(default_factory=AssertionPolicy)
    anchor: AnchorSpec | None = None
    window: WindowSpec | None = None
    evidence_rules: list[EvidenceRule]
    case_definition: CaseDefinition = Field(default_factory=CaseDefinition)
    reporting: ReportingSpec = Field(default_factory=ReportingSpec)

    # Populated by the loader, never read from the YAML body.
    definition_hash: str = ""
    source_path: str = ""

    @field_validator("evidence_rules")
    @classmethod
    def _rules_non_empty_and_unique(
        cls, rules: list[EvidenceRule]
    ) -> list[EvidenceRule]:
        if not rules:
            raise ValueError("a phenotype definition needs at least one evidence rule")
        ids = [r.id for r in rules]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate evidence rule ids: {sorted(dupes)}")
        return rules

    @property
    def key(self) -> str:
        return f"{self.id}.v{self.version}"


# --------------------------------------------------------------------------
# Evaluation output
# --------------------------------------------------------------------------


class CaseAssignment(BaseModel):
    """One row per subject.

    ``reason`` is not a nicety.  When a clinician disputes a case, the first
    question is which rule fired and on what evidence, and the answer must be in
    the row.
    """

    model_config = ConfigDict(extra="forbid")

    subject_id: str
    study_id: str
    verdict: Verdict
    evidence_state: EvidenceState
    matched_rule_id: str | None = None
    reason: str
    contributing_event_ids: list[str] = Field(default_factory=list)
    evidence_spans: list[Span] = Field(default_factory=list)
    definition_id: str
    definition_version: int
    definition_hash: str


class PhenotypeQuerySpec(BaseModel):
    """The compiled, inspectable plan the agent proposes before executing."""

    model_config = ConfigDict(extra="forbid")

    question: str
    definition_id: str
    definition_version: int
    studies: list[str] = Field(default_factory=list)
    concept: str | None = None
    assertion: list[Assertion] = Field(default_factory=lambda: ["present"])
    evidence_state: list[EvidenceState] = Field(default_factory=list)
    window: tuple[int, int] | None = None
    anchor: str | None = None
    retrieval_mode: Literal["lexical", "dense", "hybrid"] = "lexical"
    top_k: int = 20
    notes: list[str] = Field(default_factory=list)
    backend: Literal["deterministic", "llm"] = "deterministic"


class Clarification(BaseModel):
    """Returned instead of a spec when the question leaves a rule undetermined."""

    model_config = ConfigDict(extra="forbid")

    question: str
    ambiguity: str = Field(description="The specific thing that is underdetermined.")
    effect: str = Field(description="What changes in the result depending on it.")
    options: list[str] = Field(default_factory=list)


class RunManifest(BaseModel):
    """The reproducibility record for one evaluation."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    created_at: str
    spec: dict[str, Any]
    extractor_version: str
    definition_id: str
    definition_version: int
    definition_hash: str
    definition_status: DefinitionStatus
    snapshot_id: str
    deterministic: bool = True
    nondeterministic_paths: list[str] = Field(default_factory=list)
    counts_by_state: dict[str, int] = Field(default_factory=dict)
    counts_by_verdict: dict[str, int] = Field(default_factory=dict)
    assignments: list[CaseAssignment] = Field(default_factory=list)
    results_hash: str = ""
    limitations: list[str] = Field(default_factory=list)
