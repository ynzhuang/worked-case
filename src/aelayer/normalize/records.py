"""The deterministic path: source rows into canonical records.

This path reads structured variables only. Where a study put an attribute in a
standard variable it is read ``direct``; where it put it in a sponsor variable
it is read ``normalized`` through the declared mapping. Where the study put it
in language, this path deliberately leaves the attribute unresolved and says so,
because guessing at prose is not the deterministic path's job.

Nothing here consults a model, and nothing here reads the answer key.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

from ..catalog import ConceptCatalog, Configs
from ..ingest import TrialStore
from ..models import (
    SERIOUSNESS_CRITERIA,
    Attribute,
    CanonicalAERecord,
    Span,
)
from ..profiles import AttributeHome, StudyProfile
from .values import coerce_bool, coerce_date, coerce_enum, is_blank, resolve_sponsor_value


def structured_span(record_id: str, variable: str, value: Any, domain: str = "AE") -> Span:
    """A pointer at the variable a value was read from.

    Rendered so it resolves to something a person can check: the record, the
    column, and what was in it.
    """
    text = f"{variable}={value}"
    return Span(
        doc_id=f"{domain}:{record_id}",
        start=0,
        end=len(text),
        field=variable,
        extracted_value=str(value),
        text=text,
        kind="structured",
    )


class RecordNormalizer:
    """One source row, read through its study's profile."""

    def __init__(self, configs: Configs, store: TrialStore):
        self.configs = configs
        self.catalog: ConceptCatalog = configs.catalog
        self.profiles = configs.profiles
        self.store = store

    # -- one record ---------------------------------------------------------

    def normalize(self, row: dict[str, Any]) -> CanonicalAERecord:
        study_id = str(row["STUDYID"])
        profile = self.profiles.for_study(study_id)
        record_id = str(row["AESPID"])

        coded = self._enum_free(row, "AEDECOD", profile, record_id)
        reported = self._enum_free(row, "AETERM", profile, record_id)
        concept = self.catalog.concept_for_coded_term(coded.value)

        record = CanonicalAERecord(
            record_id=f"{study_id}:{record_id}",
            study_id=study_id,
            subject_id=str(row["USUBJID"]),
            source_record_id=record_id,
            profile=profile.profile_id,
            coded_event=coded,
            reported_term=reported,
            dictionary=profile.dictionary or None,
            dictionary_version=str(row.get("AEDICTVER") or "") or None,
            standardized_concept=concept,
            location=self._modifier(row, profile, record_id, "location"),
            pattern=self._modifier(row, profile, record_id, "pattern"),
            laterality=Attribute[str].unavailable(
                "not_collected_by_protocol" if not profile.collects_variable("AELAT")
                else "unknown",
                note="no study in this corpus collects laterality structurally",
            ),
            severity=self._enum(row, "AESEV", "severity", profile, record_id),
            relatedness=self._enum(row, "AEREL", "relatedness", profile, record_id),
            action_taken=self._enum(row, "AEACN", "action_taken", profile, record_id),
            outcome=self._enum(row, "AEOUT", "outcome", profile, record_id),
            seriousness=self._bool(row, "AESER", profile, record_id),
            seriousness_criteria=self._criteria(row, profile, record_id),
            onset=self._date(row, "AESTDTC", profile, record_id),
            end=self._end(row, profile, record_id),
            comment_doc_id=self._comment_doc(record_id),
            continuation_of=str(row.get("AECONTRP") or "") or None,
            normalizer_version=self.configs.normalizer_version,
        )
        return record

    # -- attributes ---------------------------------------------------------

    def _enum_free(
        self, row: dict[str, Any], variable: str, profile: StudyProfile,
        record_id: str,
    ) -> Attribute[str]:
        """A free-text structured variable: the coded term, the reported term."""
        cell = row.get(variable)
        if is_blank(cell):
            return Attribute[str].unavailable(
                profile.availability_for_blank(variable),
                variable=variable, source="structured_standard",
            )
        return Attribute[str].direct(
            str(cell).strip(), variable, [structured_span(record_id, variable, cell)]
        )

    def _enum(
        self, row: dict[str, Any], variable: str, attribute: str,
        profile: StudyProfile, record_id: str,
    ) -> Attribute[str]:
        value, availability, note = coerce_enum(
            attribute, row.get(variable), profile, variable
        )
        if value is None:
            return Attribute[str].unavailable(
                availability, variable=variable, source="structured_standard",
                note=note,
            )
        return Attribute[str].direct(
            value, variable, [structured_span(record_id, variable, row[variable])]
        )

    def _bool(
        self, row: dict[str, Any], variable: str, profile: StudyProfile,
        record_id: str,
    ) -> Attribute[bool]:
        value, availability, note = coerce_bool(row.get(variable), profile, variable)
        if value is None:
            return Attribute[bool].unavailable(
                availability, variable=variable, source="structured_standard",
                note=note,
            )
        return Attribute[bool].direct(
            value, variable, [structured_span(record_id, variable, row[variable])]
        )

    def _date(
        self, row: dict[str, Any], variable: str, profile: StudyProfile,
        record_id: str,
    ) -> Attribute[_dt.date]:
        value, availability, note = coerce_date(row.get(variable), profile, variable)
        if value is None:
            return Attribute[_dt.date].unavailable(
                availability, variable=variable, source="structured_standard",
                note=note,
            )
        return Attribute[_dt.date].direct(
            value, variable, [structured_span(record_id, variable, row[variable])]
        )

    def _end(
        self, row: dict[str, Any], profile: StudyProfile, record_id: str
    ) -> Attribute[_dt.date]:
        """The end date, gated on the outcome as *recorded*.

        An event whose recorded outcome is not terminal has no end date yet;
        that is `pending_ongoing`, and it is a different fact from a date the
        study failed to record.
        """
        attribute = self._date(row, "AEENDTC", profile, record_id)
        if attribute.populated:
            return attribute
        outcome = str(row.get("AEOUT") or "").strip().lower()
        if outcome in ("not recovered", "not_recovered", "ongoing", "recovering",
                       "resolving"):
            return Attribute[_dt.date].unavailable(
                "pending_ongoing", variable="AEENDTC", source="structured_standard",
                note=f"the recorded outcome is {outcome!r}, so no end date exists yet",
            )
        return attribute

    def _criteria(
        self, row: dict[str, Any], profile: StudyProfile, record_id: str
    ) -> dict[str, Attribute[bool]]:
        """The seriousness criteria vector, gated on the seriousness answer.

        Where seriousness is No, the criteria are not unanswered — they do not
        apply, and saying so is different from saying nobody filled them in.
        """
        answer = profile.gate_answer("seriousness", row)
        gate = profile.gate_for("seriousness_criteria")
        recorded = {
            c.strip() for c in str(row.get("AESCAT") or "").split("|") if c.strip()
        }
        out: dict[str, Attribute[bool]] = {}
        for criterion in SERIOUSNESS_CRITERIA:
            if answer is False and gate is not None:
                out[criterion] = Attribute[bool].unavailable(
                    gate.when_gate_false, variable="AESCAT",
                    source="structured_standard",
                    note="the seriousness gate was answered No, so this "
                         "criterion was never applicable",
                )
            elif answer is None:
                out[criterion] = Attribute[bool].unavailable(
                    profile.availability_for_blank("AESER"), variable="AESCAT",
                    source="structured_standard",
                    note="the seriousness gate itself is unanswered",
                )
            else:
                out[criterion] = Attribute[bool].direct(
                    criterion in recorded, "AESCAT",
                    [structured_span(record_id, "AESCAT", row.get("AESCAT") or "")],
                )
        return out

    # -- the attribute this prototype is about -----------------------------

    def _modifier(
        self, row: dict[str, Any], profile: StudyProfile, record_id: str,
        attribute: str,
    ) -> Attribute[str]:
        """Resolve ``location`` or ``pattern`` from wherever this study keeps it.

        The deterministic path can settle a structured home. A text home is left
        unresolved *with a note naming the home*, which is what tells the model
        path there is a question worth asking.
        """
        homes = profile.homes_for(attribute)
        if not homes or all(home.is_nowhere for home in homes):
            return Attribute[str].unavailable(
                "not_collected_by_protocol",
                note=f"{profile.profile_id} records {attribute} nowhere: it is "
                     f"not recoverable from this study",
            )

        for home in homes:
            if home.kind == "structured_standard":
                resolved = self._standard_home(row, home, profile, record_id, attribute)
                if resolved is not None:
                    return resolved
            elif home.kind == "structured_sponsor":
                resolved = self._sponsor_home(profile, record_id, attribute)
                if resolved is not None:
                    return resolved

        text_home = next((h for h in homes if h.is_text), None)
        if text_home is not None:
            return Attribute[str].unavailable(
                "unknown", variable=text_home.variable, source=text_home.kind,
                note=f"{attribute} lives in {text_home.variable} in "
                     f"{profile.profile_id}; the deterministic path cannot read "
                     f"prose, so the model path is asked",
            )
        # A structured home that this record left empty. The variable is named
        # as the study names it — a sponsor qualifier is "SUPPAE.RASHSITE", not
        # "SUPPAE" — or the blank would be read as a column the study does not
        # have rather than one it left empty.
        home = homes[0]
        variable = home.variable or attribute
        if home.kind == "structured_sponsor" and profile.sponsor_variable_name:
            variable = f"SUPPAE.{profile.sponsor_variable_name}"
        return Attribute[str].unavailable(
            profile.availability_for_blank(variable),
            variable=variable, source=home.kind,
            note=f"{variable} is empty on this record",
        )

    def _standard_home(
        self, row: dict[str, Any], home: AttributeHome, profile: StudyProfile,
        record_id: str, attribute: str,
    ) -> Attribute[str] | None:
        variable = home.variable or ""
        cell = row.get(variable)
        if is_blank(cell):
            return None
        raw = str(cell).strip()
        catalogue = self.catalog.attribute(attribute)
        if raw in catalogue.values:
            return Attribute[str].direct(
                raw, variable, [structured_span(record_id, variable, raw)]
            )
        normalized = catalogue.normalize(raw)
        if normalized is None:
            return Attribute[str].unavailable(
                "not_representable", variable=variable, source=home.kind,
                note=f"{variable} holds {raw!r}, which is not a value in the "
                     f"{attribute} catalogue and was not coerced to one",
            )
        return Attribute[str].normalized(
            normalized, variable, source="structured_standard",
            evidence=[structured_span(record_id, variable, raw)],
            note=f"{raw!r} normalized to {normalized}",
        )

    def _sponsor_home(
        self, profile: StudyProfile, record_id: str, attribute: str
    ) -> Attribute[str] | None:
        rows = self.store.supplemental_for(record_id)
        value, availability, note, variable = resolve_sponsor_value(
            profile, rows, attribute
        )
        if value is None:
            if availability == "unknown" and not rows:
                return None
            return Attribute[str].unavailable(
                availability, variable=variable, source="structured_sponsor",
                note=note,
            )
        return Attribute[str].normalized(
            value, variable or "SUPPAE",
            evidence=[structured_span(record_id, variable or "SUPPAE", value,
                                      domain="SUPPAE")],
            note=note,
        )

    def _comment_doc(self, record_id: str) -> str | None:
        documents = self.store.documents_of(record_id)
        return documents[0].doc_id if documents else None


def normalize_store(store: TrialStore, configs: Configs) -> list[CanonicalAERecord]:
    """Every AE row in the snapshot, in a stable order."""
    normalizer = RecordNormalizer(configs, store)
    return [normalizer.normalize(row) for row in store.ae_rows()]
