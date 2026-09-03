"""Core data models for the evidence layer.

Four ideas carry the weight.

**Assertion and availability are orthogonal, and merging them is the error this
whole layer exists to prevent.** ``assertion`` is what the source *says* —
present, absent, uncertain. ``availability`` is whether the source says anything
at all — observed, not collected, not applicable, pending, unresolved. "The
investigator recorded no mucosal involvement" and "nobody was ever asked" look
identical once they are both a null, and every incidence estimate downstream
inherits that confusion.

**The route is part of the fact.** Every value carries the kind of source it
came from, the named variable, and the ``method`` that produced it — ``direct``
from a structured field, ``derived`` by governed computation across domains, or
``extracted`` from language. A phenotype rule can then be route-agnostic on
purpose while every number still says which route supplied it.

**Nothing overwrites a coded value.** ``Rash`` and ``Rash erythematous`` are
both legitimate codings; a concept set decides which qualify. There is no code
path that edits a coded field, and a test asserts it.

**There is no ``inferred`` method.** A value the system worked out for itself,
with nothing in the source to point at, is not an attribute of a patient. The
enum cannot express it and a test asserts the word appears nowhere.
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

#: What the source says about the attribute.
Assertion = Literal["present", "absent", "uncertain"]
ASSERTIONS: tuple[str, ...] = ("present", "absent", "uncertain")

#: Whether the source says anything at all. Never merged with `assertion`.
Availability = Literal[
    "observed",         # the source addressed it, and `assertion` says what it found
    "not_collected",    # the study never asked
    "not_applicable",   # a parent gate made the question inapplicable
    "pending",          # the answer does not exist yet
    "unresolved",       # asked, or askable, and nothing settled it
]
AVAILABILITIES: tuple[str, ...] = (
    "observed", "not_collected", "not_applicable", "pending", "unresolved",
)

#: How a value was produced. There is no fourth member: see the module
#: docstring on why a value the system worked out for itself is not one.
Method = Literal["direct", "derived", "extracted"]
METHODS: tuple[str, ...] = ("direct", "derived", "extracted")

#: What kind of place the value came from.
SourceKind = Literal[
    "structured_standard",   # a standard domain variable, e.g. AELOC
    "structured_sponsor",    # a sponsor-defined supplemental qualifier
    "linked_form",           # a separate form linked to the AE record
    "reported_term",         # the investigator's own words, e.g. AETERM
    "comment",               # a comment record pointing at the AE record
    "cross_domain",          # computed across domains, e.g. AE onset against EX
]
SOURCE_KINDS: tuple[str, ...] = (
    "structured_standard", "structured_sponsor", "linked_form", "reported_term",
    "comment", "cross_domain",
)

#: Source kinds the deterministic path owns outright. A value from one of these
#: is settled, and `guards.py` refuses to send it to a model.
STRUCTURED_SOURCES: frozenset[str] = frozenset(
    {"structured_standard", "structured_sponsor", "linked_form"}
)

#: Availabilities that carry no assertion. Exactly the complement of `observed`.
SILENT: frozenset[str] = frozenset(a for a in AVAILABILITIES if a != "observed")


# --------------------------------------------------------------------------
# Clinical vocabulary
# --------------------------------------------------------------------------

SEVERITY_VALUES: tuple[str, ...] = ("mild", "moderate", "severe")
GRADES: tuple[int, ...] = (1, 2, 3, 4, 5)

SERIOUSNESS_CRITERIA: tuple[str, ...] = (
    "death", "life_threatening", "hospitalisation", "disability",
    "congenital_anomaly", "other_medically_important",
)

RELATEDNESS_VALUES: tuple[str, ...] = (
    "not_related", "unlikely", "possible", "probable", "definite", "unknown",
)

ACTION_VALUES: tuple[str, ...] = (
    "dose_not_changed", "dose_reduced", "drug_interrupted", "drug_withdrawn",
    "not_applicable", "unknown",
)

OUTCOME_VALUES: tuple[str, ...] = (
    "recovered", "recovering", "not_recovered", "recovered_with_sequelae",
    "fatal", "unknown",
)

#: Four verdicts. `non_case` exists so a denominator can be stated at all;
#: `not_ascertainable` exists because an event nobody can evaluate is neither a
#: case nor a negative, and folding it into either one biases the estimate.
Verdict = Literal["case", "non_case", "review", "not_ascertainable"]
VERDICTS: tuple[str, ...] = ("case", "non_case", "review", "not_ascertainable")

#: Verdicts that put a subject in the denominator.
ASCERTAINED: frozenset[str] = frozenset({"case", "non_case"})

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
    """One clinical value, what the source asserted, and how it got here.

    The invariants below are enforced here rather than checked downstream,
    because every one of them is a claim the rest of the system relies on.
    """

    model_config = ConfigDict(extra="forbid")

    value: T | None = None
    #: What the source says. None exactly when the source says nothing.
    assertion: Assertion | None = None
    availability: Availability = "unresolved"
    source: SourceKind | None = None
    #: The variable this came from, by its real name: "AELOC", "AETERM",
    #: "SUPPAE.MUCOSAL", "CO.COVAL", "AE+EX". Two studies can carry the same
    #: fact under different names, and a reader must be able to see which.
    source_variable: str | None = None
    method: Method | None = None
    evidence: list[Span] = _PydanticField(default_factory=list)
    confidence: float | None = None
    #: Dictionary, extractor, prompt — whichever produced this value.
    versions: dict[str, str] = _PydanticField(default_factory=dict)
    note: str = ""
    #: What the deterministic path left behind, where a later route filled the
    #: attribute. Recovering a modifier from prose does not make the CRF column
    #: collected, and both facts are worth keeping.
    prior_availability: Availability | None = None

    @model_validator(mode="after")
    def _invariants(self) -> "Attribute[T]":
        if self.availability == "observed":
            if self.assertion is None:
                raise ValueError(
                    "availability 'observed' means the source said something, so "
                    "it must carry an assertion"
                )
        elif self.assertion is not None:
            raise ValueError(
                f"availability {self.availability!r} means the source is silent, "
                f"so it cannot assert {self.assertion!r}; assertion and "
                f"availability are orthogonal and must never be merged"
            )
        if self.method == "extracted" and not self.evidence:
            raise ValueError(
                "an extracted value must carry at least one span: a value with "
                "no text behind it cannot be checked by anyone"
            )
        if self.method == "direct" and self.source not in STRUCTURED_SOURCES:
            raise ValueError(
                f"method 'direct' means a structured variable, but source is "
                f"{self.source!r}"
            )
        if self.method == "derived" and self.source != "cross_domain":
            raise ValueError(
                f"method 'derived' means a governed computation across domains, "
                f"but source is {self.source!r}"
            )
        if self.value is not None and self.availability != "observed":
            raise ValueError(
                f"availability {self.availability!r} carries a value "
                f"{self.value!r}; only an observed attribute has one"
            )
        return self

    # -- reading ------------------------------------------------------------

    @property
    def observed(self) -> bool:
        return self.availability == "observed"

    @property
    def present(self) -> bool:
        """The source addressed it and found it."""
        return self.availability == "observed" and self.assertion == "present"

    @property
    def documented_negative(self) -> bool:
        """The source addressed it and found it absent.

        Different from silence in every way that matters: a documented negative
        puts the subject in the denominator, and silence does not.
        """
        return self.availability == "observed" and self.assertion == "absent"

    @property
    def silent(self) -> bool:
        return self.availability in SILENT

    @property
    def structured_availability(self) -> Availability:
        """How the deterministic path left this attribute."""
        return self.prior_availability or self.availability

    @property
    def from_text(self) -> bool:
        return self.method == "extracted"

    def has_provenance(self) -> bool:
        return (not self.observed) or bool(self.evidence)

    def describe_route(self) -> str:
        if not self.observed:
            return self.availability
        return (
            f"{self.assertion}"
            + (f"={self.value!r}" if self.value is not None else "")
            + f" via {self.method} from {self.source_variable}"
        )

    # -- constructing -------------------------------------------------------

    @classmethod
    def direct(
        cls, assertion: Assertion, variable: str, evidence: list[Span] | None = None,
        value: T | None = None, source: SourceKind = "structured_standard",
        versions: dict[str, str] | None = None, note: str = "",
    ) -> "Attribute[T]":
        """Read straight from a structured variable."""
        return cls(
            value=value, assertion=assertion, availability="observed",
            source=source, source_variable=variable, method="direct",
            evidence=list(evidence or []), versions=dict(versions or {}), note=note,
        )

    @classmethod
    def derived(
        cls, value: T, variable: str, evidence: list[Span] | None = None,
        assertion: Assertion = "present", note: str = "",
    ) -> "Attribute[T]":
        """Computed across domains by governed code, never by model reasoning."""
        return cls(
            value=value, assertion=assertion, availability="observed",
            source="cross_domain", source_variable=variable, method="derived",
            evidence=list(evidence or []), note=note,
        )

    @classmethod
    def extracted(
        cls, assertion: Assertion, variable: str, evidence: list[Span],
        value: T | None = None, source: SourceKind = "reported_term",
        confidence: float | None = None, versions: dict[str, str] | None = None,
        note: str = "", prior_availability: Availability | None = None,
    ) -> "Attribute[T]":
        """Read out of language, with the span that supports it."""
        return cls(
            value=value, assertion=assertion, availability="observed",
            source=source, source_variable=variable, method="extracted",
            evidence=list(evidence), confidence=confidence,
            versions=dict(versions or {}), note=note,
            prior_availability=prior_availability,
        )

    @classmethod
    def silent_because(
        cls, availability: Availability, *, variable: str | None = None,
        source: SourceKind | None = None, note: str = "",
        prior_availability: Availability | None = None,
    ) -> "Attribute[T]":
        """An attribute the source says nothing about, and why.

        Abstention is a valid answer and is recorded as one; a guess is a defect.
        """
        if availability == "observed":
            raise ValueError(
                "'observed' means the source said something: use direct(), "
                "derived() or extracted() and state the assertion"
            )
        return cls(
            value=None, assertion=None, availability=availability, source=source,
            source_variable=variable, note=note,
            prior_availability=prior_availability,
        )


class CodedTerm(BaseModel):
    """A coded value exactly as the study recorded it, plus any reconciliation.

    The original is never modified. Where cross-study analysis needs a common
    dictionary version, ``reconciled_to`` records what the code maps to under
    the target version and ``reconciliation`` says how that was decided —
    mechanically, or flagged for a human. A model never recodes anything.
    """

    model_config = ConfigDict(extra="forbid")

    code: str
    dictionary: str = ""
    dictionary_version: str = ""
    concept_id: str | None = None
    reconciled_to: str | None = None
    reconciled_version: str | None = None
    reconciliation: Literal[
        "unchanged", "remapped_mechanically", "flagged_for_review", "not_attempted"
    ] = "not_attempted"
    note: str = ""

    @property
    def effective_code(self) -> str:
        """The code to compare against a concept set. Never a rewrite."""
        return self.reconciled_to or self.code


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


class CanonicalAERecord(BaseModel):
    """One per source record. Immutable, never merged, never overwritten.

    This is the grain the study actually collected. Episode derivation, where a
    study evidences one, happens above it and is optional; the source-record
    grain is what everything traces back to.
    """

    model_config = ConfigDict(extra="forbid")

    record_id: str
    study_id: str
    subject_id: str
    source_record_id: str
    source_form_id: str = "AE"
    profile: str = ""

    coded_event: CodedTerm | None = None
    reported_term: Attribute[str] = _PydanticField(default_factory=Attribute)
    concept_id: str | None = None

    #: Configured clinical modifiers, keyed by name. The worked example uses
    #: `mucosal_involvement`; the model does not know that name.
    modifiers: dict[str, Attribute[str]] = _PydanticField(default_factory=dict)

    onset: Attribute[_dt.date] = _PydanticField(default_factory=Attribute)
    end: Attribute[_dt.date] = _PydanticField(default_factory=Attribute)
    #: Days from the anchor exposure to onset. Cross-domain, method `derived`.
    exposure_relation: Attribute[int] = _PydanticField(default_factory=Attribute)

    severity: Attribute[str] = _PydanticField(default_factory=Attribute)
    grade: Attribute[int] = _PydanticField(default_factory=Attribute)
    seriousness: Attribute[bool] = _PydanticField(default_factory=Attribute)
    seriousness_criteria: dict[str, Attribute[bool]] = _PydanticField(
        default_factory=dict
    )
    relatedness: Attribute[str] = _PydanticField(default_factory=Attribute)
    action: Attribute[str] = _PydanticField(default_factory=Attribute)
    outcome: Attribute[str] = _PydanticField(default_factory=Attribute)

    comment_doc_id: str | None = None
    linked_form_ids: list[str] = _PydanticField(default_factory=list)

    normalizer_version: str = ""
    extractor_version: str = ""

    SCALAR_ATTRIBUTES: tuple[str, ...] = (
        "reported_term", "onset", "end", "exposure_relation", "severity",
        "grade", "seriousness", "relatedness", "action", "outcome",
    )

    def attributes(self) -> dict[str, Attribute[Any]]:
        """Every attribute on the record, modifiers and criteria included."""
        out: dict[str, Attribute[Any]] = {
            name: getattr(self, name) for name in self.SCALAR_ATTRIBUTES
        }
        out.update(self.modifiers)
        for criterion, attribute in self.seriousness_criteria.items():
            out[f"seriousness_criteria.{criterion}"] = attribute
        return out

    def attribute(self, name: str) -> Attribute[Any] | None:
        if name in self.modifiers:
            return self.modifiers[name]
        if name.startswith("seriousness_criteria."):
            return self.seriousness_criteria.get(name.split(".", 1)[1])
        value = getattr(self, name, None)
        return value if isinstance(value, Attribute) else None

    def availabilities(self) -> dict[str, str]:
        return {n: a.availability for n, a in self.attributes().items()}

    def assertions(self) -> dict[str, str | None]:
        return {n: a.assertion for n, a in self.attributes().items()}

    def missing_provenance(self) -> list[str]:
        """Observed attributes that carry no span. Every one is a defect."""
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
# Phenotype definition
# --------------------------------------------------------------------------


class ConceptSet(BaseModel):
    """Which coded concepts qualify, and under which dictionary version.

    Listing both ``RASH`` and ``RASH_ERYTHEMATOUS`` is the point: they are
    different legitimate codings of the same clinical situation, and the
    definition decides they both qualify rather than anything merging them.
    """

    model_config = ConfigDict(extra="forbid")

    include: list[str]
    exclude: list[str] = _PydanticField(default_factory=list)
    dictionary_target: str | None = None

    @model_validator(mode="after")
    def _not_empty(self) -> "ConceptSet":
        if not self.include:
            raise ValueError("a concept set that includes nothing selects nobody")
        overlap = sorted(set(self.include) & set(self.exclude))
        if overlap:
            raise ValueError(f"concepts {overlap} are both included and excluded")
        return self


class ModifierRequirement(BaseModel):
    """One modifier a definition requires, and what to do when it is missing."""

    model_config = ConfigDict(extra="forbid")

    name: str
    require_assertion: Assertion = "present"
    #: Route-agnostic by design: the rule does not know or care which study
    #: field supplied the evidence, only that it is there and points at
    #: something.
    accept_methods: list[Method] = _PydanticField(
        default_factory=lambda: ["direct", "extracted"]
    )
    accept_sources: list[SourceKind] | None = None
    on_unavailable: Literal["not_ascertainable", "review", "non_case"] = (
        "not_ascertainable"
    )
    description: str = ""

    @model_validator(mode="after")
    def _usable(self) -> "ModifierRequirement":
        if not self.accept_methods:
            raise ValueError(
                f"modifier {self.name!r} accepts no method, so nothing can ever "
                f"satisfy it"
            )
        return self


class TemporalRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    anchor: str = "first_exposure"
    window: dict[str, Any] = _PydanticField(
        default_factory=lambda: {"min": 0, "max": 30, "unit": "days"}
    )
    on_unresolved: Literal["review", "not_ascertainable", "non_case"] = (
        "not_ascertainable"
    )

    @property
    def minimum(self) -> int:
        return int(self.window.get("min", 0))

    @property
    def maximum(self) -> int:
        return int(self.window.get("max", 30))

    @model_validator(mode="after")
    def _ordered(self) -> "TemporalRule":
        if self.maximum < self.minimum:
            raise ValueError(
                f"window max {self.maximum} precedes min {self.minimum}"
            )
        if self.window.get("unit", "days") != "days":
            raise ValueError("only day windows are supported")
        return self

    def contains(self, offset: int) -> bool:
        return self.minimum <= offset <= self.maximum


class EvidencePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    extracted_requires_span: bool = True
    min_confidence: float = _PydanticField(default=0.7, ge=0.0, le=1.0)
    below_threshold: Literal["review", "not_ascertainable", "non_case"] = "review"


class AscertainabilityPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    missing_required_modifier: Literal["not_ascertainable", "review", "non_case"] = (
        "not_ascertainable"
    )
    uncertain_assertion: Literal["review", "not_ascertainable", "non_case"] = "review"


class GradeRule(BaseModel):
    """A graded-toxicity criterion, for the second shipped definition."""

    model_config = ConfigDict(extra="forbid")

    attribute: str = "grade"
    minimum: int = _PydanticField(default=3, ge=1, le=5)
    on_unavailable: Literal["not_ascertainable", "review", "non_case"] = (
        "not_ascertainable"
    )


class CumulativeExposureRule(BaseModel):
    """Total exposure before onset, in the study's own dose units."""

    model_config = ConfigDict(extra="forbid")

    minimum: float = 0.0
    unit: str = "mg"
    on_unresolved: Literal["review", "not_ascertainable", "non_case"] = (
        "not_ascertainable"
    )


class PhenotypeDefinition(BaseModel):
    """A versioned scientific artifact. Frozen versions are never edited."""

    model_config = ConfigDict(extra="forbid")

    id: str
    version: int = _PydanticField(ge=1)
    status: DefinitionStatus = "draft"
    label: str
    description: str = ""
    supersedes: str | None = None
    authors: list[str] = _PydanticField(default_factory=list)
    created: _dt.date | None = None

    concept_set: ConceptSet
    modifiers: list[ModifierRequirement] = _PydanticField(default_factory=list)
    temporal: TemporalRule | None = None
    grade: GradeRule | None = None
    cumulative_exposure: CumulativeExposureRule | None = None
    evidence_policy: EvidencePolicy = _PydanticField(default_factory=EvidencePolicy)
    ascertainability: AscertainabilityPolicy = _PydanticField(
        default_factory=AscertainabilityPolicy
    )
    verdicts: list[Verdict] = _PydanticField(
        default_factory=lambda: list(VERDICTS)
    )

    definition_hash: str = ""
    source_path: str = ""

    @model_validator(mode="after")
    def _usable(self) -> "PhenotypeDefinition":
        if not (self.modifiers or self.temporal or self.grade
                or self.cumulative_exposure):
            raise ValueError(
                "a definition needs at least one criterion beyond its concept "
                "set, or it selects every event of that concept"
            )
        names = [m.name for m in self.modifiers]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate modifier requirements: {duplicates}")
        if set(self.verdicts) != set(VERDICTS):
            raise ValueError(
                f"the verdicts block declares {sorted(self.verdicts)}, but the "
                f"evaluator returns {sorted(VERDICTS)}"
            )
        return self

    @property
    def key(self) -> str:
        return f"{self.id}.v{self.version}"

    def modifier(self, name: str) -> ModifierRequirement | None:
        return next((m for m in self.modifiers if m.name == name), None)

    def accepted_methods(self) -> list[str]:
        return sorted({m for r in self.modifiers for m in r.accept_methods})


# --------------------------------------------------------------------------
# Assignment and denominators
# --------------------------------------------------------------------------


class CriterionFinding(BaseModel):
    """How one criterion came out, for one record."""

    model_config = ConfigDict(extra="forbid")

    name: str
    satisfied: bool
    verdict: Verdict
    assertion: Assertion | None = None
    availability: Availability = "unresolved"
    value: Any = None
    method: Method | None = None
    source: SourceKind | None = None
    source_variable: str | None = None
    confidence: float | None = None
    reason: str = ""
    spans: list[Span] = _PydanticField(default_factory=list)


class CaseAssignment(BaseModel):
    """One row per source record.

    ``reason`` names what decided it. When a clinician disputes a verdict, that
    is the first question asked, and the answer has to be in the row.
    """

    model_config = ConfigDict(extra="forbid")

    record_id: str
    subject_id: str
    study_id: str
    profile: str = ""
    verdict: Verdict
    deciding_criterion: str | None = None
    reason: str
    findings: list[CriterionFinding] = _PydanticField(default_factory=list)
    evidence_spans: list[Span] = _PydanticField(default_factory=list)
    attribute_sources: dict[str, str] = _PydanticField(default_factory=dict)
    attribute_methods: dict[str, str] = _PydanticField(default_factory=dict)
    definition_id: str
    definition_version: int
    definition_hash: str

    @property
    def ascertained(self) -> bool:
        return self.verdict in ASCERTAINED

    @property
    def used_text_extraction(self) -> bool:
        return "extracted" in self.attribute_methods.values()


class Denominator(BaseModel):
    """What a study contributes, and how much of it is answerable at all.

    The ascertainable fraction is reported as a study characteristic beside
    every incidence figure. Silently dropping unascertainable subjects makes the
    denominator vary with collection convention, which is the quiet way an
    estimate becomes a comparison of CRFs rather than of patients.
    """

    model_config = ConfigDict(extra="forbid")

    study_id: str
    profile: str = ""
    n_total: int = 0
    n_case: int = 0
    n_non_case: int = 0
    n_review: int = 0
    n_not_ascertainable: int = 0

    @property
    def n_ascertainable(self) -> int:
        return self.n_case + self.n_non_case

    @property
    def ascertainable_fraction(self) -> float:
        return round(self.n_ascertainable / self.n_total, 4) if self.n_total else 0.0

    @property
    def incidence(self) -> float | None:
        """Within the ascertainable population, or None if there is none."""
        if not self.n_ascertainable:
            return None
        return round(self.n_case / self.n_ascertainable, 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "profile": self.profile,
            "n_total": self.n_total,
            "n_case": self.n_case,
            "n_non_case": self.n_non_case,
            "n_review": self.n_review,
            "n_not_ascertainable": self.n_not_ascertainable,
            "n_ascertainable": self.n_ascertainable,
            "ascertainable_fraction": self.ascertainable_fraction,
            "incidence_within_ascertainable": self.incidence,
        }


DENOMINATOR_NOTE = (
    "Incidence is computed within the ascertainable population (case + "
    "non_case). Subjects whose required evidence was never collected enter "
    "neither the numerator nor the denominator, and the ascertainable fraction "
    "is reported beside every rate as a study characteristic: dropping them "
    "silently would make the denominator vary with collection convention "
    "rather than with the patients."
)


# --------------------------------------------------------------------------
# Manifest, trace, agent
# --------------------------------------------------------------------------


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

    definition_id: str
    definition_version: int
    definition_hash: str
    definition_status: DefinitionStatus = "frozen"

    study_scope: list[str] = _PydanticField(default_factory=list)
    cohort_specification: dict[str, Any] = _PydanticField(default_factory=dict)
    data_snapshot_id: str
    dictionary_versions: dict[str, str] = _PydanticField(default_factory=dict)
    normalizer_version: str = ""
    extractor_version: str = ""
    model_version: str | None = None
    prompt_version: str | None = None

    method_parameters: dict[str, Any] = _PydanticField(default_factory=dict)
    validation_status: Literal[
        "unvalidated", "internally_validated", "externally_validated"
    ] = "unvalidated"

    output_pointer: str = ""
    results_hash: str = ""
    counts_by_verdict: dict[str, int] = _PydanticField(default_factory=dict)
    denominators: list[dict[str, Any]] = _PydanticField(default_factory=list)
    #: Which routes supplied the evidence behind this cohort, so a later reader
    #: can see it depended on text extraction without re-deriving it.
    attribute_sources: dict[str, int] = _PydanticField(default_factory=dict)
    attribute_methods: dict[str, int] = _PydanticField(default_factory=dict)
    deterministic: bool = True
    nondeterministic_paths: list[str] = _PydanticField(default_factory=list)
    limitations: list[str] = _PydanticField(default_factory=list)


class TraceLink(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: Literal[
        "result", "analysis", "cohort", "definition", "attribute", "record", "span"
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


class QuerySpec(BaseModel):
    """The compiled, inspectable plan an agent execution runs.

    It **binds** a definition version. The agent never invents phenotype
    parameters: a question implying a different window is a conflict to be
    raised, not a parameter to override.
    """

    model_config = ConfigDict(extra="forbid")

    question: str
    definition_id: str
    definition_version: int
    definition_hash: str = ""
    studies: list[str] = _PydanticField(default_factory=list)
    verdicts: list[Verdict] = _PydanticField(default_factory=lambda: ["case"])
    accept_methods: list[Method] = _PydanticField(default_factory=list)
    notes: list[str] = _PydanticField(default_factory=list)
    backend: Literal["deterministic", "llm"] = "deterministic"


class Conflict(BaseModel):
    """A question that cannot be run against the definition it names."""

    model_config = ConfigDict(extra="forbid")

    question: str
    conflict: str
    bound_definition: str | None = None
    effect: str
    options: list[str] = _PydanticField(default_factory=list)


class Supportability(BaseModel):
    """Whether a study can answer the question at all, decided on metadata.

    Run before any patient-level query. A study that cannot ascertain the
    required modifier is worth knowing about before a cohort is built, not
    after.
    """

    model_config = ConfigDict(extra="forbid")

    study_id: str
    profile: str = ""
    status: Literal["supported", "supported_via_extraction", "cannot_ascertain"]
    reason: str
    modifier_homes: dict[str, list[str]] = _PydanticField(default_factory=dict)
