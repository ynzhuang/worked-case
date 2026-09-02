"""Core data models for the evidence layer.

Three ideas carry most of the weight.

**The same clinical fact arrives by different routes, and the route is part of
the fact.**  Every clinical value is an ``Attribute[T]``: the value, the kind of
source it came from, the *named variable* it came from, and the ``method`` that
produced it — ``direct`` from a standard structured field, ``normalized`` from a
sponsor variable or codelist mapping, or ``extracted`` from language.  A
phenotype rule can then be route-agnostic on purpose, while every number it
produces still says which route supplied the evidence.

**A blank is not a value.**  ``availability`` says which kind of empty: never
collected by the protocol, made inapplicable by a gate, still pending, not
representable in the study's codelist, or simply unknown.  Flattening those to
``None`` throws away the only thing that says whether absence is evidence.

**Two levels, and the lower one is never destroyed.**  A ``CanonicalAERecord``
is source-faithful: one per source record, never merged, never overwritten.
Episodes and trajectories derive above it and can be recomputed.

There is deliberately no ``inferred`` method.  A value the system worked out for
itself, with nothing in the source to point at, is not an attribute of a
patient; the enum has no way to express it and a test asserts the word appears
nowhere in the codebase.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, model_validator
from pydantic import Field as _PydanticField

T = TypeVar("T")

# --------------------------------------------------------------------------
# Provenance vocabulary
# --------------------------------------------------------------------------

#: How a value was produced. There is no fourth member: see the module
#: docstring on why a value the system worked out for itself is not one.
Method = Literal["direct", "normalized", "extracted"]
METHODS: tuple[str, ...] = ("direct", "normalized", "extracted")

#: What kind of place the value came from.
SourceKind = Literal[
    "structured_standard",   # a standard domain variable, e.g. AELOC
    "structured_sponsor",    # a sponsor-defined supplemental variable
    "reported_term",         # the investigator's own words, e.g. AETERM
    "comment",               # a comment record pointing at the AE record
    "linked_form",           # a separate form linked to the AE record
    "derived",               # computed above the records, e.g. an episode field
]
SOURCE_KINDS: tuple[str, ...] = (
    "structured_standard", "structured_sponsor", "reported_term", "comment",
    "linked_form", "derived",
)

#: Which kind of empty a blank is. These are not interchangeable.
Availability = Literal[
    "collected",
    "not_collected_by_protocol",   # the CRF never asked
    "not_applicable_gated",        # a parent gate was answered No
    "pending_ongoing",             # the event has not ended yet
    "not_representable",           # the concept exists, the codelist cannot express it
    "unknown",                     # asked, not answered, or nothing in the text
]
AVAILABILITY_VALUES: tuple[str, ...] = (
    "collected", "not_collected_by_protocol", "not_applicable_gated",
    "pending_ongoing", "not_representable", "unknown",
)

#: Availabilities that are never, on their own, evidence that something did not
#: happen. Everything except `collected`.
NOT_EVIDENCE_OF_ABSENCE: frozenset[str] = frozenset(
    v for v in AVAILABILITY_VALUES if v != "collected"
)

#: Source kinds that the deterministic path owns outright. A value from one of
#: these is settled, and `guards.py` refuses to send it to a model.
STRUCTURED_SOURCES: frozenset[str] = frozenset(
    {"structured_standard", "structured_sponsor"}
)


# --------------------------------------------------------------------------
# Clinical vocabulary
# --------------------------------------------------------------------------

Severity = Literal["mild", "moderate", "severe"]
SEVERITY_VALUES: tuple[str, ...] = ("mild", "moderate", "severe")

SERIOUSNESS_CRITERIA: tuple[str, ...] = (
    "death", "life_threatening", "hospitalisation", "disability",
    "congenital_anomaly", "other_medically_important",
)

RELATEDNESS_VALUES: tuple[str, ...] = (
    "not_related", "unlikely", "possible", "probable", "definite", "unknown",
)

ACTION_TAKEN_VALUES: tuple[str, ...] = (
    "dose_not_changed", "dose_reduced", "drug_interrupted", "drug_withdrawn",
    "not_applicable", "unknown",
)

OUTCOME_VALUES: tuple[str, ...] = (
    "recovered", "recovering", "not_recovered", "recovered_with_sequelae",
    "fatal", "unknown",
)

#: Four verdicts. `not_ascertainable` is the one that earns its place: a
#: required modifier that nobody recorded is not a negative finding and not a
#: review item, because no reviewer can resolve it either.
Verdict = Literal["case", "not_case", "not_ascertainable", "review"]
VERDICTS: tuple[str, ...] = ("case", "not_case", "not_ascertainable", "review")

DefinitionStatus = Literal["draft", "frozen", "superseded"]


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


class Span(BaseModel):
    """A pointer back to the exact evidence a value came from.

    For text, a character range in a document. For a structured variable, the
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
# Attribute
# --------------------------------------------------------------------------


class Attribute(BaseModel, Generic[T]):
    """One clinical value, with the route that produced it.

    The invariants below are enforced here rather than checked downstream,
    because every one of them is a claim the rest of the system relies on:

    - an extracted value must point at the text it came from
    - a ``direct`` value must come from a standard structured variable, and
      nothing but the deterministic path may set it
    - a value only exists where the attribute was actually available
    """

    model_config = ConfigDict(extra="forbid")

    value: T | None = None
    source: SourceKind | None = None
    #: The variable this came from, by its real name: "AELOC", "AETERM",
    #: "SUPPAE.RASHSITE", "CO.COVAL". Two studies can carry the same fact under
    #: different names, and a reader has to be able to see which.
    source_variable: str | None = None
    method: Method | None = None
    evidence: list[Span] = _PydanticField(default_factory=list)
    availability: Availability = "unknown"
    confidence: float | None = None
    #: Set only where a model path produced the value.
    extractor_version: str | None = None
    model_version: str | None = None
    prompt_version: str | None = None
    note: str = ""
    #: What the deterministic path left behind, where a later route filled the
    #: attribute. Recovering a location from text does not make the CRF column
    #: collected, and both facts are worth keeping.
    prior_availability: Availability | None = None

    @model_validator(mode="after")
    def _invariants(self) -> "Attribute[T]":
        if self.method == "extracted" and not self.evidence:
            raise ValueError(
                "an extracted value must carry at least one span: a value with "
                "no text behind it cannot be checked by anyone"
            )
        if self.method == "direct" and self.source != "structured_standard":
            raise ValueError(
                f"method 'direct' means a standard structured variable, but "
                f"source is {self.source!r}"
            )
        if self.availability != "collected" and self.value is not None:
            raise ValueError(
                f"availability {self.availability!r} carries a value "
                f"{self.value!r}; only a collected attribute has one"
            )
        if self.availability == "collected" and self.value is None:
            raise ValueError(
                "availability 'collected' with no value: say which kind of "
                "empty it is instead"
            )
        if self.extractor_version and self.method != "extracted":
            raise ValueError(
                f"extractor_version is set on a {self.method!r} value; only the "
                f"model path stamps it"
            )
        return self

    # -- reading ------------------------------------------------------------

    @property
    def populated(self) -> bool:
        return self.value is not None

    @property
    def is_evidence_of_absence(self) -> bool:
        """Can an empty reading here be taken as "it did not happen"?"""
        return self.availability == "collected"

    @property
    def structured_availability(self) -> Availability:
        """How the deterministic path left this attribute."""
        return self.prior_availability or self.availability

    @property
    def from_text(self) -> bool:
        return self.method == "extracted"

    def has_provenance(self) -> bool:
        return (not self.populated) or bool(self.evidence)

    def describe_route(self) -> str:
        if not self.populated:
            return f"{self.availability}"
        return f"{self.value!r} via {self.method} from {self.source_variable}"

    # -- constructing -------------------------------------------------------

    @classmethod
    def direct(
        cls, value: T, variable: str, evidence: list[Span] | None = None,
    ) -> "Attribute[T]":
        """Read straight from a standard structured variable."""
        return cls(
            value=value, source="structured_standard", source_variable=variable,
            method="direct", evidence=list(evidence or []), availability="collected",
        )

    @classmethod
    def normalized(
        cls, value: T, variable: str, source: SourceKind = "structured_sponsor",
        evidence: list[Span] | None = None, note: str = "",
    ) -> "Attribute[T]":
        """Mapped deterministically — a sponsor codelist, a unit, a spelling."""
        return cls(
            value=value, source=source, source_variable=variable,
            method="normalized", evidence=list(evidence or []),
            availability="collected", note=note,
        )

    @classmethod
    def extracted(
        cls, value: T, variable: str, evidence: list[Span],
        source: SourceKind = "reported_term", confidence: float | None = None,
        extractor_version: str | None = None, note: str = "",
        prior_availability: Availability | None = None,
        model_version: str | None = None, prompt_version: str | None = None,
    ) -> "Attribute[T]":
        """Read out of language, with the span that supports it."""
        return cls(
            value=value, source=source, source_variable=variable,
            method="extracted", evidence=list(evidence), availability="collected",
            confidence=confidence, extractor_version=extractor_version, note=note,
            prior_availability=prior_availability, model_version=model_version,
            prompt_version=prompt_version,
        )

    @classmethod
    def unavailable(
        cls, availability: Availability, *, variable: str | None = None,
        source: SourceKind | None = None, note: str = "",
        prior_availability: Availability | None = None,
    ) -> "Attribute[T]":
        """An empty attribute that says which kind of empty it is.

        Abstention is a valid answer and is recorded as one; a guess is a defect.
        """
        if availability == "collected":
            raise ValueError("an unavailable attribute cannot be 'collected'")
        return cls(
            value=None, source=source, source_variable=variable,
            availability=availability, note=note,
            prior_availability=prior_availability,
        )


class Modifier(BaseModel):
    """A modifier mention found in text, before it is promoted to an attribute."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["location", "laterality", "pattern", "quality"]
    value: str
    surface: str
    span: Span
    confidence: float = 0.0
    normalized_from: str = ""


# --------------------------------------------------------------------------
# Level 1 — the source-faithful record
# --------------------------------------------------------------------------


class CanonicalAERecord(BaseModel):
    """One per source record. Never merged, never overwritten."""

    model_config = ConfigDict(extra="forbid")

    record_id: str
    study_id: str
    subject_id: str
    source_record_id: str
    source_form_id: str = "AE"
    profile: str = ""

    coded_event: Attribute[str] = _PydanticField(default_factory=Attribute)
    reported_term: Attribute[str] = _PydanticField(default_factory=Attribute)
    dictionary: str | None = None
    dictionary_version: str | None = None
    standardized_concept: str | None = _PydanticField(
        default=None,
        description="Catalogue concept for this record, by explicit membership.",
    )

    location: Attribute[str] = _PydanticField(default_factory=Attribute)
    laterality: Attribute[str] = _PydanticField(default_factory=Attribute)
    pattern: Attribute[str] = _PydanticField(default_factory=Attribute)

    severity: Attribute[str] = _PydanticField(default_factory=Attribute)
    seriousness: Attribute[bool] = _PydanticField(default_factory=Attribute)
    seriousness_criteria: dict[str, Attribute[bool]] = _PydanticField(
        default_factory=dict
    )
    relatedness: Attribute[str] = _PydanticField(default_factory=Attribute)
    action_taken: Attribute[str] = _PydanticField(default_factory=Attribute)
    outcome: Attribute[str] = _PydanticField(default_factory=Attribute)
    onset: Attribute[_dt.date] = _PydanticField(default_factory=Attribute)
    end: Attribute[_dt.date] = _PydanticField(default_factory=Attribute)

    comment_doc_id: str | None = None
    linked_form_ids: list[str] = _PydanticField(default_factory=list)
    continuation_of: str | None = None

    modifiers: list[Modifier] = _PydanticField(default_factory=list)
    normalizer_version: str = ""
    extractor_version: str = ""

    #: Attribute names that are `Attribute[...]` instances on this model.
    ATTRIBUTES: tuple[str, ...] = (
        "coded_event", "reported_term", "location", "laterality", "pattern",
        "severity", "seriousness", "relatedness", "action_taken", "outcome",
        "onset", "end",
    )

    def attributes(self) -> dict[str, Attribute[Any]]:
        """Every attribute on the record, including the criteria vector."""
        out: dict[str, Attribute[Any]] = {
            name: getattr(self, name) for name in self.ATTRIBUTES
        }
        for criterion, attribute in self.seriousness_criteria.items():
            out[f"seriousness_criteria.{criterion}"] = attribute
        return out

    def attribute(self, name: str) -> Attribute[Any] | None:
        if name.startswith("seriousness_criteria."):
            return self.seriousness_criteria.get(name.split(".", 1)[1])
        value = getattr(self, name, None)
        return value if isinstance(value, Attribute) else None

    def availabilities(self) -> dict[str, str]:
        return {n: a.availability for n, a in self.attributes().items()}

    def sources(self) -> dict[str, str | None]:
        return {
            n: a.source_variable for n, a in self.attributes().items() if a.populated
        }

    def missing_provenance(self) -> list[str]:
        """Populated attributes that carry no span. Every one is a defect."""
        return sorted(
            name for name, attribute in self.attributes().items()
            if not attribute.has_provenance()
        )

    def has_full_provenance(self) -> bool:
        return not self.missing_provenance()

    def spans(self) -> list[Span]:
        seen: dict[tuple, Span] = {}
        for attribute in self.attributes().values():
            for span in attribute.evidence:
                seen.setdefault(span.key(), span)
        return sorted(seen.values(), key=lambda s: (s.field, s.doc_id, s.start))


# --------------------------------------------------------------------------
# Level 2 — derived
# --------------------------------------------------------------------------

LinkageRule = Literal[
    "single_record", "explicit_continuation", "declared_convention",
    "temporal_overlap", "gap_within_tolerance", "recurrence_split",
]


class CanonicalAEEpisode(BaseModel):
    """A clinical episode derived over one or more records.

    Derivation is additive: the records it was built from are untouched and
    remain the authority. Where the linkage rule cannot decide, the episode is
    flagged rather than silently resolved.
    """

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    study_id: str
    subject_id: str
    profile: str = ""
    standardized_concept: str | None = None

    episode_start: Attribute[_dt.date] = _PydanticField(default_factory=Attribute)
    episode_end: Attribute[_dt.date] = _PydanticField(default_factory=Attribute)
    source_record_ids: list[str] = _PydanticField(default_factory=list)

    location: Attribute[str] = _PydanticField(default_factory=Attribute)
    laterality: Attribute[str] = _PydanticField(default_factory=Attribute)
    pattern: Attribute[str] = _PydanticField(default_factory=Attribute)
    severity: Attribute[str] = _PydanticField(default_factory=Attribute)
    seriousness: Attribute[bool] = _PydanticField(default_factory=Attribute)
    relatedness: Attribute[str] = _PydanticField(default_factory=Attribute)
    outcome: Attribute[str] = _PydanticField(default_factory=Attribute)
    action_taken: Attribute[str] = _PydanticField(default_factory=Attribute)

    coded_events: list[str] = _PydanticField(default_factory=list)
    reported_terms: list[str] = _PydanticField(default_factory=list)
    dictionary_versions: list[str] = _PydanticField(default_factory=list)
    severity_trajectory: list[tuple[_dt.date | None, str]] = _PydanticField(
        default_factory=list
    )

    onset_offset_days: Attribute[int] = _PydanticField(default_factory=Attribute)
    anchor_event: str | None = None
    anchor_date: _dt.date | None = None

    linked_evidence: list[Span] = _PydanticField(default_factory=list)
    linkage_rule: LinkageRule = "single_record"
    linkage_confidence: float = 1.0
    linkage_review_required: bool = False
    linkage_note: str = ""

    episode_provenance: dict[str, Any] = _PydanticField(default_factory=dict)

    #: Discovery results are candidates and may not enter a cohort directly.
    candidate: bool = False

    EPISODE_ATTRIBUTES: tuple[str, ...] = (
        "episode_start", "episode_end", "location", "laterality", "pattern",
        "severity", "seriousness", "relatedness", "outcome", "action_taken",
        "onset_offset_days",
    )

    def attributes(self) -> dict[str, Attribute[Any]]:
        return {name: getattr(self, name) for name in self.EPISODE_ATTRIBUTES}

    def attribute(self, name: str) -> Attribute[Any] | None:
        value = getattr(self, name, None)
        return value if isinstance(value, Attribute) else None

    def availabilities(self) -> dict[str, str]:
        return {n: a.availability for n, a in self.attributes().items()}

    @property
    def peak_severity(self) -> str | None:
        ranked = [s for _when, s in self.severity_trajectory if s in SEVERITY_VALUES]
        if not ranked:
            return None
        return max(ranked, key=lambda s: SEVERITY_VALUES.index(s))

    def attribute_sources(self) -> dict[str, str]:
        """Which named variable supplied each populated attribute."""
        return {
            name: attribute.source_variable or "?"
            for name, attribute in self.attributes().items()
            if attribute.populated and attribute.source_variable
        }


# --------------------------------------------------------------------------
# Trajectory
# --------------------------------------------------------------------------


class TrajectoryEvent(BaseModel):
    """One dated thing that happened to a subject, in a comparable shape."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["exposure", "episode"]
    identifier: str
    date: _dt.date
    label: str = ""
    detail: dict[str, Any] = _PydanticField(default_factory=dict)
    offset_days: int | None = None


class Trajectory(BaseModel):
    """A subject's exposures and episodes in one ordered sequence.

    Deliberately not a progression model. It is the ordered structure the
    phenotype window and the "time since exposure" question both need, and
    nothing more.
    """

    model_config = ConfigDict(extra="forbid")

    subject_id: str
    study_id: str
    profile: str = ""
    anchor_event: str | None = None
    anchor_date: _dt.date | None = None
    events: list[TrajectoryEvent] = _PydanticField(default_factory=list)

    def exposures(self) -> list[TrajectoryEvent]:
        return [e for e in self.events if e.kind == "exposure"]

    def episodes(self) -> list[TrajectoryEvent]:
        return [e for e in self.events if e.kind == "episode"]

    def offset_of(self, identifier: str) -> int | None:
        for event in self.events:
            if event.identifier == identifier:
                return event.offset_days
        return None


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


class AnchorSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event: str = "first_exposure"
    source_domain: str = "EX"
    index_rule: Literal["first_occurrence", "most_recent_before_onset"] = (
        "first_occurrence"
    )


class WindowSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unit: Literal["days"] = "days"
    min: int = 0
    max: int = 14
    anchor: str = "first_exposure"

    @model_validator(mode="after")
    def _ordered(self) -> "WindowSpec":
        if self.max < self.min:
            raise ValueError(f"window max {self.max} precedes min {self.min}")
        return self

    def contains(self, offset: int) -> bool:
        return self.min <= offset <= self.max


class AttributeRequirement(BaseModel):
    """One attribute a definition requires, and what to do when it is missing.

    ``accept_methods`` is the heart of it. A rule that lists all three methods
    is saying: I do not care whether the location came from ``AELOC`` or from
    the investigator's own words, only that it is present and points at
    something. A rule that lists only ``direct`` is a different, narrower
    scientific claim, and the difference is now visible in the file.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str
    allowed: list[str] | None = _PydanticField(default=None, alias="in")
    accept_methods: list[Method] = _PydanticField(
        default_factory=lambda: ["direct", "normalized", "extracted"]
    )
    accept_sources: list[SourceKind] | None = None
    min_confidence: float | None = None
    window: WindowSpec | None = None
    on_unavailable: Literal["not_ascertainable", "review", "not_case"] = (
        "not_ascertainable"
    )
    on_unresolved: Literal["review", "not_ascertainable", "not_case"] = "review"
    on_low_confidence: Literal["review", "not_ascertainable", "not_case"] = "review"
    description: str = ""

    @model_validator(mode="after")
    def _has_a_test(self) -> "AttributeRequirement":
        if self.allowed is None and self.window is None:
            raise ValueError(
                f"requirement {self.name!r} tests nothing: give it `in` or a window"
            )
        if not self.accept_methods:
            raise ValueError(
                f"requirement {self.name!r} accepts no method, so nothing can "
                f"ever satisfy it"
            )
        return self


class VerdictSpec(BaseModel):
    """What each verdict means, stated in the definition file.

    The evaluator implements these semantics; the block exists so the file is
    readable on its own, and a test asserts its keys are exactly the four
    verdicts the code can return.
    """

    model_config = ConfigDict(extra="forbid")

    case: str
    not_case: str
    not_ascertainable: str
    review: str


class ReportingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    counts_by_verdict: bool = True
    counts_by_attribute_source: bool = True
    require_evidence_span_for_extracted: bool = True
    report_not_ascertainable_separately: bool = True


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
    required_attributes: list[AttributeRequirement]
    anchor: AnchorSpec | None = None
    episode_linkage_confidence: float = _PydanticField(default=0.8, ge=0.0, le=1.0)
    on_linkage_review: Literal["case", "review", "not_ascertainable"] = "review"
    verdicts: VerdictSpec | None = None
    reporting: ReportingSpec = _PydanticField(default_factory=ReportingSpec)

    definition_hash: str = ""
    source_path: str = ""

    @model_validator(mode="after")
    def _requirements_are_usable(self) -> "PhenotypeDefinition":
        if not self.required_attributes:
            raise ValueError("a definition needs at least one required attribute")
        names = [r.name for r in self.required_attributes]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate required attributes: {duplicates}")
        return self

    @property
    def key(self) -> str:
        return f"{self.id}.v{self.version}"

    def requirement(self, name: str) -> AttributeRequirement | None:
        return next((r for r in self.required_attributes if r.name == name), None)


# --------------------------------------------------------------------------
# Assignment, manifest, trace
# --------------------------------------------------------------------------


class AttributeFinding(BaseModel):
    """How one required attribute came out, for one episode."""

    model_config = ConfigDict(extra="forbid")

    name: str
    satisfied: bool
    value: Any = None
    method: Method | None = None
    source: SourceKind | None = None
    source_variable: str | None = None
    availability: Availability = "unknown"
    reason: str = ""
    spans: list[Span] = _PydanticField(default_factory=list)


class CaseAssignment(BaseModel):
    """One row per episode.

    ``reason`` names what decided it. When a clinician disputes a case, that is
    the first question asked, and the answer has to be in the row.
    """

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    subject_id: str
    study_id: str
    profile: str = ""
    verdict: Verdict
    deciding_attribute: str | None = None
    reason: str
    findings: list[AttributeFinding] = _PydanticField(default_factory=list)
    source_record_ids: list[str] = _PydanticField(default_factory=list)
    evidence_spans: list[Span] = _PydanticField(default_factory=list)
    attribute_sources: dict[str, str] = _PydanticField(default_factory=dict)
    attribute_methods: dict[str, str] = _PydanticField(default_factory=dict)
    definition_id: str
    definition_version: int
    definition_hash: str
    linkage_review_required: bool = False
    review_reasons: list[str] = _PydanticField(default_factory=list)

    @property
    def used_text_extraction(self) -> bool:
        return "extracted" in self.attribute_methods.values()


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
    counts_by_verdict: dict[str, int] = _PydanticField(default_factory=dict)
    #: Which routes supplied the evidence behind this cohort. A later reader can
    #: see that the cohort depended on text extraction rather than having to
    #: re-derive it.
    attribute_sources: dict[str, int] = _PydanticField(default_factory=dict)
    attribute_methods: dict[str, int] = _PydanticField(default_factory=dict)
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
    verdicts: list[Verdict] = _PydanticField(default_factory=lambda: ["case"])
    accept_methods: list[Method] = _PydanticField(default_factory=list)
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
