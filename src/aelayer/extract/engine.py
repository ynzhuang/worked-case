"""Orchestration of the model path.

The engine asks the backend about exactly the modifiers a record left
unresolved, in exactly the places the study's profile says they live. It never
asks about a value the deterministic path already settled — ``guards.py``
refuses that, and a test asserts it over the whole corpus.

Two counts come out of every run and both are reported rather than hidden:

``recovered``
    modifiers the text settled that the structured data did not
``abstained``
    modifiers the text was read for and did not settle

An abstention is a valid answer. It leaves the attribute silent, with a note
saying the text was read and supported nothing — which is a different statement
from "nobody looked", and the note keeps both facts on the row.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from typing import Any, Iterable, Sequence

from ..catalog import Configs
from ..guards import askable_modifiers, assert_model_path_permitted
from ..ingest import TrialStore
from ..models import CanonicalAERecord
from ..profiles import StudyProfile
from .backends import Backend, ExtractionRequest, select_backend


@dataclass
class ExtractionStats:
    """What the model path did, as numbers the report can print."""

    requests: int = 0
    recovered: int = 0
    abstained: int = 0
    by_assertion: dict[str, int] = _dc_field(default_factory=dict)

    @property
    def abstention_rate(self) -> float:
        asked = self.recovered + self.abstained
        return 0.0 if not asked else self.abstained / asked

    def to_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "recovered": self.recovered,
            "abstained": self.abstained,
            "abstention_rate": round(self.abstention_rate, 4),
            "by_assertion": dict(sorted(self.by_assertion.items())),
        }


@dataclass
class ExtractionEngine:
    configs: Configs
    store: TrialStore
    backend: Backend
    notes: list[str] = _dc_field(default_factory=list)
    stats: ExtractionStats = _dc_field(default_factory=ExtractionStats)

    @classmethod
    def build(
        cls, configs: Configs, store: TrialStore, preference: str = "auto"
    ) -> "ExtractionEngine":
        backend, notes = select_backend(
            configs.catalog, configs.extraction, configs.extractor_version,
            preference,
        )
        return cls(configs=configs, store=store, backend=backend, notes=notes)

    # -- where the text is --------------------------------------------------

    def sources_for(
        self, record: CanonicalAERecord, profile: StudyProfile
    ) -> list[tuple[str, str, str, str]]:
        """Every readable text source for a record.

        Returns ``(doc_id, text, source_kind, source_variable)``. Only sources
        the extraction config declares readable appear here, and a structured
        kind cannot even be declared there.
        """
        readable = set(self.configs.extraction.readable_sources)
        found: list[tuple[str, str, str, str]] = []
        if "reported_term" in readable and record.reported_term.observed:
            found.append((
                f"AE:{record.source_record_id}:AETERM",
                str(record.reported_term.value), "reported_term", "AETERM",
            ))
        if "comment" in readable:
            for document in self.store.documents_of(record.source_record_id):
                found.append((
                    document.doc_id, document.full_text, "comment", "CO.COVAL",
                ))
        return found

    # -- one record ---------------------------------------------------------

    def enrich(self, record: CanonicalAERecord) -> CanonicalAERecord:
        """Fill what the deterministic path left open, and nothing else."""
        profile = self.configs.profiles.for_study(record.study_id)
        askable = askable_modifiers(record, profile, self.configs.extraction)
        record.extractor_version = self.configs.extractor_version
        if not askable:
            return record

        for doc_id, text, source_kind, variable in self.sources_for(record, profile):
            remaining = tuple(
                name for name in askable
                if not self._settled(record, name)
            )
            if not remaining:
                break
            request = ExtractionRequest(
                doc_id=doc_id, text=text, modifiers=remaining,
                # Deliberately not `record.concept_id`. Coding and writing are
                # separate acts: a record coded `Rash erythematous` is very
                # often written up as "skin rash", and requiring the prose to
                # echo the code would silently drop those records. The anchor
                # rule is still enforced — a mention must sit in a sentence
                # that names *some* event — it is just not narrowed to the
                # coding somebody chose afterwards.
                concept_id=None,
                source_kind=source_kind, source_variable=variable,
            )
            # The boundary is enforced here, on every request, not asserted once
            # in a test and trusted thereafter.
            assert_model_path_permitted(request, record, profile)
            self.stats.requests += 1
            result = self.backend.extract(request)

            for modifier, value in result.values.items():
                current = record.modifiers.get(modifier)
                record.modifiers[modifier] = value.model_copy(update={
                    "prior_availability": current.availability if current else None,
                })
                self.stats.recovered += 1
                self.stats.by_assertion[value.assertion or "?"] = (
                    self.stats.by_assertion.get(value.assertion or "?", 0) + 1
                )
            for modifier in result.abstained:
                self.stats.abstained += 1
                current = record.modifiers.get(modifier)
                if current is None or current.observed:
                    continue
                # Both facts survive: what the structured side already said
                # about this modifier, and that the text was read and supported
                # nothing. Replacing the first with the second would make an
                # availability and its explanation contradict each other on the
                # same row.
                record.modifiers[modifier] = current.model_copy(update={
                    "note": (
                        f"{modifier} is {current.availability}"
                        + (f" ({current.note})" if current.note else "")
                        + f"; the model path was then asked about it in "
                          f"{variable} and abstained — nothing in the text "
                          f"settles it"
                    ),
                })
            self.notes.extend(n for n in result.notes if n not in self.notes)

        return record

    @staticmethod
    def _settled(record: CanonicalAERecord, modifier: str) -> bool:
        current = record.modifiers.get(modifier)
        return bool(current and current.observed)

    def enrich_all(
        self, records: Iterable[CanonicalAERecord]
    ) -> list[CanonicalAERecord]:
        return [self.enrich(record) for record in records]

    def versions(self) -> dict[str, Any]:
        return {
            "extraction_backend": self.backend.name,
            "model_version": getattr(self.backend, "model_version", None),
            "prompt_version": getattr(self.backend, "prompt_version", None),
        }


def enrich_records(
    records: Sequence[CanonicalAERecord], configs: Configs, store: TrialStore,
    preference: str = "auto",
) -> list[CanonicalAERecord]:
    return ExtractionEngine.build(configs, store, preference).enrich_all(records)
