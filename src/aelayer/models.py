"""Core data models for the evidence layer.

Three ideas carry most of the weight here.

**A blank is not a value.**  Every clinical field is a ``Field[T]``: a value
*and* a ``collection_state``.  A field left empty because the protocol never
collected it, because a parent gate was answered No, because the event is still
ongoing, or because the study's codelist had no permissible value for the
concept, are four different facts.  Flattening them to ``None`` throws away the
only information that says whether absence is evidence of anything.

**Two levels, and the lower one is never destroyed.**  A
``CanonicalAERecord`` is source-faithful: one per source CRF record, never
merged, never overwritten.  A ``CanonicalAEEpisode`` is derived above it and is
purely additive.  Collapsing grade-change rows into one row is irreversible;
deriving an episode view alongside the records is not.

**Coded and verbatim both survive.**  The coded term gives comparability across
studies; the verbatim preserves what coding compressed.  Neither replaces the
other, so both are stored, along with the dictionary version in force.
"""

from __future__ import annotations

import datetime as _dt
from typing import Annotated, Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, model_validator
from pydantic import Field as _PydanticField

T = TypeVar("T")

# --------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------

Severity = Literal["mild", "moderate", "severe"]
SEVERITY_VALUES: tuple[str, ...] = ("mild", "moderate", "severe")

#: The regulatory seriousness criteria. Seriousness itself is a separate gate.
SERIOUSNESS_CRITERIA: tuple[str, ...] = (
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
    "not_related", "unlikely", "possible", "probable", "definite", "unknown",
)

ActionTaken = Literal[
    "dose_not_changed",
    "dose_reduced",
    "drug_interrupted",
    "drug_withdrawn",
    "not_applicable",
    "unknown",
]
ACTION_TAKEN_VALUES: tuple[str, ...] = (
    "dose_not_changed", "dose_reduced", "drug_interrupted", "drug_withdrawn",
    "not_applicable", "unknown",
)

Outcome = Literal[
    "recovered", "recovering", "not_recovered", "recovered_with_sequelae",
    "fatal", "unknown",
]
OUTCOME_VALUES: tuple[str, ...] = (
    "recovered", "recovering", "not_recovered", "recovered_with_sequelae",
    "fatal", "unknown",
)

Assertion = Literal[
    "present", "absent", "hypothetical", "historical", "family_history", "uncertain"
]
ASSERTION_VALUES: tuple[str, ...] = (
    "present", "absent", "hypothetical", "historical", "family_history", "uncertain",
)

#: What a blank in a source field actually means. These are not interchangeable
#: and the phenotype definition decides how each is treated.
CollectionState = Literal[
    "collected",
    "not_collected_by_protocol",   # the CRF never asked
    "not_applicable_gated",        # a parent gate was answered No
    "pending_ongoing",             # the event has not ended yet
    "intentionally_blank",         # protocol instructed the site to leave it blank
    "not_representable",           # the concept exists but the codelist cannot express it
    "unknown",                     # asked, not answered, reason unrecorded
]
COLLECTION_STATES: tuple[str, ...] = (
    "collected", "not_collected_by_protocol", "not_applicable_gated",
    "pending_ongoing", "intentionally_blank", "not_representable", "unknown",
)

#: States that are never, on their own, evidence that something did not happen.
#: A field the protocol never collected says nothing about the patient.
NOT_EVIDENCE_OF_ABSENCE: frozenset[str] = frozenset(
    {"not_collected_by_protocol", "not_applicable_gated", "unknown",
     "pending_ongoing", "not_representable", "intentionally_blank"}
)

FieldSource = Literal["structured", "text", "derived"]

EvidenceState = Literal["explicit", "supported", "insufficient", "absent", "none"]
EVIDENCE_STATE_VALUES: tuple[str, ...] = (
    "explicit", "supported", "insufficient", "absent", "none",
)
EVIDENCE_STATE_RANK: dict[str, int] = {
    "none": 0, "absent": 1, "insufficient": 2, "supported": 3, "explicit": 4,
}

Verdict = Literal["case", "review", "excluded"]
DefinitionStatus = Literal["draft", "frozen", "superseded"]


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


class Span(BaseModel):
    """A pointer back to the exact evidence a value came from.

    For text, a character range in a document.  For a structured CRF field, the
    record and column it was read from, rendered so the pointer resolves to
    something a person can check.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str
    start: int = _PydanticField(default=0, ge=0)
    end: int = _PydanticField(default=0, ge=0)
    field: str
    extracted_value: str = ""
    text: str = ""
    kind: Literal["text", "structured", "derived"] = "text"

    @model_validator(mode="after")
    def _ordered(self) -> "Span":
        if self.end < self.start:
            raise ValueError(f"span end {self.end} precedes start {self.start}")
        return self

    def key(self) -> tuple[str, int, int, str, str]:
        return (self.doc_id, self.start, self.end, self.field, self.extracted_value)


# --------------------------------------------------------------------------
# Field
# --------------------------------------------------------------------------


class Field(BaseModel, Generic[T]):
    """A clinical value together with why it looks the way it does.

    ``value`` alone cannot distinguish "the site said No" from "the CRF never
    asked".  ``collection_state`` carries that, and downstream rules consult it
    rather than guessing from a null.
    """

    model_config = ConfigDict(extra="forbid")

    value: T | None = None
    collection_state: CollectionState = "unknown"
    source: FieldSource = "structured"
    spans: list[Span] = _PydanticField(default_factory=list)
    confidence: float | None = None
    note: str = ""

    @property
    def populated(self) -> bool:
        return self.value is not None

    @property
    def is_evidence_of_absence(self) -> bool:
        """Can a false or empty reading here be taken as "it did not happen"?

        Only a value the study actually collected can carry that weight.
        """
        return self.collection_state == "collected"

    def has_provenance(self) -> bool:
        """A populated value must point at where it came from."""
        return (not self.populated) or bool(self.spans)

    @classmethod
    def collected(
        cls, value: T, spans: list[Span] | None = None, *,
        source: FieldSource = "structured", confidence: float | None = None,
    ) -> "Field[T]":
        return cls(
            value=value, collection_state="collected", source=source,
            spans=list(spans or []), confidence=confidence,
        )

    @classmethod
    def missing(
        cls, state: CollectionState, *, note: str = "",
        spans: list[Span] | None = None, source: FieldSource = "structured",
    ) -> "Field[T]":
        """An empty field that says why it is empty.

        Abstention is a valid answer and must be recorded as one; a guess is a
        defect.
        """
        if state == "collected":
            raise ValueError("a missing field cannot be in state 'collected'")
        return cls(
            value=None, collection_state=state, source=source,
            spans=list(spans or []), note=note,
        )


class LabValue(BaseModel):
    """A laboratory result in both reported and canonical units.

    Trials report glucose in mg/dL or mmol/L by region.  A threshold applied to
    an unconverted value misclassifies an entire study in silence.
    """

    model_config = ConfigDict(extra="forbid")

    test: str
    value: float
    unit: str
    canonical_value: float | None = None
    canonical_unit: str | None = None
    collection_datetime: _dt.datetime | None = None
    source: FieldSource = "structured"
    span: Span


class SymptomMention(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symptom: str
    span: Span
    assertion: Assertion = "present"


# --------------------------------------------------------------------------
# Level 1 — the source-faithful record
# --------------------------------------------------------------------------


class CanonicalAERecord(BaseModel):
    """One per source CRF record. Never merged, never overwritten.

    This is the grain the study actually collected.  Everything above it is
    derived and can be recomputed; this cannot, so it is never edited in place.
    """

    model_config = ConfigDict(extra="forbid")

    record_id: str
    study_id: str
    subject_id: str
    source_record_id: str
    source_form_id: str = "AE"

    verbatim_term: Field[str] = _PydanticField(default_factory=Field)
    coded_term: Field[str] = _PydanticField(default_factory=Field)
    dictionary: str | None = None
    dictionary_version: str | None = None
    standardized_concept: str | None = _PydanticField(
        default=None,
        description="Catalogue concept the coded term maps to, by explicit "
                    "membership. Null when no catalogue term matches.",
    )

    onset_datetime: Field[_dt.datetime] = _PydanticField(default_factory=Field)
    end_datetime: Field[_dt.datetime] = _PydanticField(default_factory=Field)

    severity: Field[str] = _PydanticField(default_factory=Field)
    seriousness: Field[bool] = _PydanticField(default_factory=Field)
    seriousness_criteria: dict[str, Field[bool]] = _PydanticField(default_factory=dict)

    relatedness: Field[str] = _PydanticField(default_factory=Field)
    alternative_attribution: list[str] = _PydanticField(default_factory=list)
    action_taken: Field[str] = _PydanticField(default_factory=Field)
    outcome: Field[str] = _PydanticField(default_factory=Field)

    #: Clinical detail that lives on linked event forms or in narrative.
    symptoms: list[SymptomMention] = _PydanticField(default_factory=list)
    labs: list[LabValue] = _PydanticField(default_factory=list)
    assertion: Field[str] = _PydanticField(default_factory=Field)

    linked_form_ids: list[str] = _PydanticField(default_factory=list)
    narrative_doc_id: str | None = None
    continuation_of: str | None = _PydanticField(
        default=None,
        description="Source record this one explicitly continues, where the "
                    "CRF records that. Never inferred.",
    )

    evidence: list[Span] = _PydanticField(default_factory=list)
    normalizer_version: str = ""
    extractor_version: str = ""
    model_version: str | None = None
    prompt_version: str | None = None

    #: Field names that are `Field[...]` instances on this model.
    CLINICAL_FIELDS: tuple[str, ...] = (
        "verbatim_term", "coded_term", "onset_datetime", "end_datetime",
        "severity", "seriousness", "relatedness", "action_taken", "outcome",
        "assertion",
    )

    def fields(self) -> dict[str, Field[Any]]:
        """Every ``Field`` on the record, including the criteria vector."""
        out: dict[str, Field[Any]] = {
            name: getattr(self, name) for name in self.CLINICAL_FIELDS
        }
        for criterion, field in self.seriousness_criteria.items():
            out[f"seriousness_criteria.{criterion}"] = field
        return out

    def missing_provenance(self) -> list[str]:
        """Populated fields that carry no span. Every one is a defect."""
        missing = [
            name for name, field in self.fields().items() if not field.has_provenance()
        ]
        missing.extend(
            f"symptoms[{s.symptom}]" for s in self.symptoms if not s.span.doc_id
        )
        missing.extend(f"labs[{l.test}]" for l in self.labs if not l.span.doc_id)
        return sorted(missing)

    def has_full_provenance(self) -> bool:
        return not self.missing_provenance()

    def collection_states(self) -> dict[str, str]:
        return {name: field.collection_state for name, field in self.fields().items()}


# --------------------------------------------------------------------------
# Level 2 — the derived episode
# --------------------------------------------------------------------------


LinkageRule = Literal[
    "single_record",
    "explicit_continuation",      # the CRF declares this record continues that one
    "declared_convention",        # the study declares it splits on severity change
    "temporal_overlap",           # the intervals actually overlap
    "gap_within_tolerance",       # close enough, for a concept that does not recur
    "recurrence_split",           # kept apart because the concept recurs
    "model_proposed",             # proposed where deterministic evidence ran out
]


class CanonicalAEEpisode(BaseModel):
    """A clinical episode derived over one or more records.

    Derivation is additive: the records it was built from are untouched and
    remain the authority.  Where the linkage rule cannot decide,
    ``linkage_review_required`` is set and the episode is reported rather than
    silently resolved.
    """

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    study_id: str
    subject_id: str
    standardized_concept: str | None = None

    episode_start: Field[_dt.datetime] = _PydanticField(default_factory=Field)
    episode_end: Field[_dt.datetime] = _PydanticField(default_factory=Field)
    source_record_ids: list[str] = _PydanticField(default_factory=list)

    severity_trajectory: list[tuple[_dt.datetime | None, str]] = _PydanticField(
        default_factory=list
    )
    seriousness_trajectory: list[tuple[_dt.datetime | None, list[str]]] = _PydanticField(
        default_factory=list
    )

    relatedness: Field[str] = _PydanticField(default_factory=Field)
    action_history: list[tuple[_dt.datetime | None, str]] = _PydanticField(
        default_factory=list
    )
    outcome: Field[str] = _PydanticField(default_factory=Field)
    seriousness: Field[bool] = _PydanticField(default_factory=Field)

    symptoms: list[SymptomMention] = _PydanticField(default_factory=list)
    labs: list[LabValue] = _PydanticField(default_factory=list)
    coded_terms: list[str] = _PydanticField(default_factory=list)
    verbatim_terms: list[str] = _PydanticField(default_factory=list)
    dictionary_versions: list[str] = _PydanticField(default_factory=list)
    assertions: list[str] = _PydanticField(default_factory=list)

    onset_offset_days: Field[int] = _PydanticField(default_factory=Field)
    anchor_event: str | None = None
    anchor_datetime: _dt.datetime | None = None

    linked_evidence: list[Span] = _PydanticField(default_factory=list)
    linkage_rule: LinkageRule = "single_record"
    linkage_confidence: float = 1.0
    linkage_review_required: bool = False
    linkage_note: str = ""

    #: Collection state per derived field, summarised across the chain. A rule
    #: that fails on a field consults this to learn *why* it failed: a value
    #: the study collected and that did not meet the bar is a different finding
    #: from a value the study never collected.
    field_states: dict[str, str] = _PydanticField(default_factory=dict)
    field_notes: dict[str, str] = _PydanticField(default_factory=dict)

    episode_provenance: dict[str, Any] = _PydanticField(default_factory=dict)

    #: Discovery results are candidates and may not enter a cohort directly.
    candidate: bool = False

    @property
    def peak_severity(self) -> str | None:
        ranked = [s for _when, s in self.severity_trajectory if s in SEVERITY_VALUES]
        if not ranked:
            return None
        return max(ranked, key=lambda s: SEVERITY_VALUES.index(s))

    def field_for(self, name: str) -> Field[Any] | None:
        value = getattr(self, name, None)
        return value if isinstance(value, Field) else None


# --------------------------------------------------------------------------
# Phenotype definition
# --------------------------------------------------------------------------


class ConceptSelector(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary: str
    group: str | None = None
    bridge_dictionary_versions: bool = True
    include_coded_terms: bool = True
    include_lexicon: bool = True


class EvidenceRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    state: EvidenceState
    when: dict[str, Any]
    description: str | None = None


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


class MissingnessPolicy(BaseModel):
    """How the definition reads a field that is empty.

    ``not_collected_by_protocol`` and ``not_applicable_gated`` are never
    evidence of absence, and the loader refuses a definition that says they are.
    """

    model_config = ConfigDict(extra="forbid")

    treat_as_absent: list[CollectionState] = _PydanticField(default_factory=list)
    route_to_review: list[CollectionState] = _PydanticField(
        default_factory=lambda: ["pending_ongoing", "unknown", "not_representable"]
    )

    @model_validator(mode="after")
    def _absence_must_be_evidenced(self) -> "MissingnessPolicy":
        forbidden = {"not_collected_by_protocol", "not_applicable_gated"}
        offending = sorted(set(self.treat_as_absent) & forbidden)
        if offending:
            raise ValueError(
                f"{offending} cannot be treated as evidence of absence: a field "
                f"the protocol never collected, or that a gate made "
                f"inapplicable, says nothing about the patient"
            )
        overlap = sorted(set(self.treat_as_absent) & set(self.route_to_review))
        if overlap:
            raise ValueError(
                f"collection states {overlap} are both treated as absent and "
                f"routed to review"
            )
        return self


class EpisodePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    require_linkage_confidence: float = _PydanticField(default=0.8, ge=0.0, le=1.0)
    on_review_required: Literal["case", "review", "exclude"] = "review"


class CaseDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary: list[EvidenceState] = _PydanticField(
        default_factory=lambda: ["explicit", "supported"]
    )
    review: list[EvidenceState] = _PydanticField(
        default_factory=lambda: ["insufficient"]
    )
    excluded: list[EvidenceState] = _PydanticField(
        default_factory=lambda: ["absent", "none"]
    )

    @model_validator(mode="after")
    def _no_overlap(self) -> "CaseDefinition":
        seen: dict[str, str] = {}
        for bucket, values in (
            ("primary", self.primary), ("review", self.review),
            ("excluded", self.excluded),
        ):
            for value in values:
                if value in seen:
                    raise ValueError(
                        f"evidence state {value!r} is in both {seen[value]} and {bucket}"
                    )
                seen[value] = bucket
        return self

    def verdict_for(self, state: str) -> Verdict:
        if state in self.primary:
            return "case"
        if state in self.review:
            return "review"
        return "excluded"


class ReportingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    counts_by_state: bool = True
    require_evidence_span: bool = True
    report_review_set_separately: bool = True


class PhenotypeDefinition(BaseModel):
    """A versioned scientific artifact, evaluated over episodes."""

    model_config = ConfigDict(extra="forbid")

    id: str
    version: int = _PydanticField(ge=1)
    status: DefinitionStatus = "draft"
    label: str
    description: str = ""
    operates_on: Literal["episode"] = "episode"
    supersedes: str | None = None
    authors: list[str] = _PydanticField(default_factory=list)
    created: _dt.date | None = None

    concept: ConceptSelector
    evidence_rules: list[EvidenceRule]
    anchor: AnchorSpec | None = None
    window: WindowSpec | None = None
    missingness: MissingnessPolicy = _PydanticField(default_factory=MissingnessPolicy)
    episode: EpisodePolicy = _PydanticField(default_factory=EpisodePolicy)
    case_definition: CaseDefinition = _PydanticField(default_factory=CaseDefinition)
    reporting: ReportingSpec = _PydanticField(default_factory=ReportingSpec)

    definition_hash: str = ""
    source_path: str = ""

    @model_validator(mode="after")
    def _rules_are_usable(self) -> "PhenotypeDefinition":
        if not self.evidence_rules:
            raise ValueError("a definition needs at least one evidence rule")
        ids = [r.id for r in self.evidence_rules]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate evidence rule ids: {duplicates}")
        return self

    @property
    def key(self) -> str:
        return f"{self.id}.v{self.version}"


# --------------------------------------------------------------------------
# Assignment, manifest, trace
# --------------------------------------------------------------------------


class CaseAssignment(BaseModel):
    """One row per episode.

    ``reason`` names the rule that decided.  When a clinician disputes a case,
    that is the first question asked, and the answer has to be in the row.
    """

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    subject_id: str
    study_id: str
    verdict: Verdict
    evidence_state: EvidenceState
    matched_rule_id: str | None = None
    reason: str
    source_record_ids: list[str] = _PydanticField(default_factory=list)
    evidence_spans: list[Span] = _PydanticField(default_factory=list)
    definition_id: str
    definition_version: int
    definition_hash: str
    linkage_review_required: bool = False
    review_reasons: list[str] = _PydanticField(default_factory=list)


class Manifest(BaseModel):
    """The record of one governed execution.

    A pointer, never a copy: duplicating result payloads here would create a
    second, uncontrolled result store with its own drift problem.
    """

    model_config = ConfigDict(extra="forbid")

    manifest_id: str
    created_at: str
    actor: str = "unknown"

    question: str = ""
    specification: dict[str, Any] = _PydanticField(default_factory=dict)

    phenotype_definition_id: str
    phenotype_definition_version: int
    definition_hash: str
    definition_status: DefinitionStatus = "frozen"

    cohort_specification: dict[str, Any] = _PydanticField(default_factory=dict)
    data_snapshot_id: str
    terminology_versions: dict[str, str] = _PydanticField(default_factory=dict)
    normalizer_version: str = ""
    extractor_version: str = ""
    model_version: str | None = None
    prompt_version: str | None = None

    analysis_method: str = "phenotype_evaluation"
    parameters: dict[str, Any] = _PydanticField(default_factory=dict)
    validation_status: Literal[
        "unvalidated", "internally_validated", "externally_validated"
    ] = "unvalidated"

    output_pointer: str = ""
    results_hash: str = ""
    counts_by_state: dict[str, int] = _PydanticField(default_factory=dict)
    counts_by_verdict: dict[str, int] = _PydanticField(default_factory=dict)
    deterministic: bool = True
    nondeterministic_paths: list[str] = _PydanticField(default_factory=list)
    limitations: list[str] = _PydanticField(default_factory=list)


class TraceLink(BaseModel):
    """One hop in the chain from a reported number back to source text."""

    model_config = ConfigDict(extra="forbid")

    level: Literal[
        "number", "analysis", "cohort", "definition", "episode", "record", "span"
    ]
    identifier: str
    detail: str = ""
    payload: dict[str, Any] = _PydanticField(default_factory=dict)


class Trace(BaseModel):
    """The full chain behind a reported number.

    A number that cannot be traced end to end is a failing test, not a caveat.
    """

    model_config = ConfigDict(extra="forbid")

    number: float | int
    label: str
    complete: bool
    broken_at: str | None = None
    links: list[TraceLink] = _PydanticField(default_factory=list)

    def levels(self) -> list[str]:
        return [link.level for link in self.links]


class PhenotypeQuerySpec(BaseModel):
    """The compiled, inspectable plan an agent execution runs."""

    model_config = ConfigDict(extra="forbid")

    question: str
    definition_id: str
    definition_version: int
    studies: list[str] = _PydanticField(default_factory=list)
    concept: str | None = None
    window: tuple[int, int] | None = None
    anchor: str | None = None
    evidence_state: list[EvidenceState] = _PydanticField(default_factory=list)
    assertion: list[Assertion] = _PydanticField(default_factory=list)
    retrieval_mode: Literal["precise", "discovery", "hybrid"] = "precise"
    top_k: int = 20
    notes: list[str] = _PydanticField(default_factory=list)
    backend: Literal["deterministic", "llm"] = "deterministic"


class Clarification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    ambiguity: str
    effect: str
    options: list[str] = _PydanticField(default_factory=list)
