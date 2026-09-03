"""The deterministic path: source rows into canonical records.

This path reads structured variables only, and does three things a naive
mapper would not.

**It never rewrites a coded value.** The coded term is preserved exactly as the
study recorded it, together with its dictionary version. Reconciliation to a
target version is an *additional* field with an outcome — unchanged, remapped
mechanically, or flagged for review — and a model is nowhere near it.

**It keeps assertion and availability apart.** A structured qualifier reading
"N" is an observed absence. An empty one is silence, and which kind of silence
depends on the study's profile.

**It derives across domains under governed code.** ``AE.onset − EX.first_exposure``
is a computation, not a reading and not a model's reasoning, and it is stamped
``method="derived"`` so a reader can tell.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

from ..anchors import AnchorResolver, parse_date
from ..catalog import ConceptCatalog, Configs
from ..ingest import TrialStore
from ..models import (
    SERIOUSNESS_CRITERIA,
    Attribute,
    CanonicalAERecord,
    CodedTerm,
    Span,
)
from ..profiles import ModifierHome, StudyProfile
from .values import (
    coerce_bool,
    coerce_date,
    coerce_enum,
    coerce_int,
    coerce_tristate,
    is_blank,
)


def structured_span(
    record_id: str, variable: str, value: Any, domain: str = "AE"
) -> Span:
    """A pointer at the variable a value was read from."""
    text = f"{variable}={value}"
    return Span(
        doc_id=f"{domain}:{record_id}", start=0, end=len(text), field=variable,
        extracted_value=str(value), text=text, kind="structured",
    )


def derived_span(record_id: str, detail: str) -> Span:
    return Span(
        doc_id=f"DERIVED:{record_id}", start=0, end=len(detail), field="exposure_relation",
        extracted_value=detail, text=detail, kind="derived",
    )


class RecordNormalizer:
    """One source row, read through its study's profile."""

    def __init__(
        self, configs: Configs, store: TrialStore,
        anchor_resolver: AnchorResolver | None = None,
    ):
        self.configs = configs
        self.catalog: ConceptCatalog = configs.catalog
        self.profiles = configs.profiles
        self.store = store
        # Built from the snapshot when the caller does not inject one. The
        # exposure rows are right there; leaving every offset unresolved
        # because a caller forgot an argument is exactly the silent failure
        # this layer exists to prevent.
        self.resolver = anchor_resolver or store.anchor_resolver(
            configs.extraction.anchors
        )
        self.anchor = configs.extraction.default_anchor or "first_exposure"

    # -- one record ---------------------------------------------------------

    def normalize(self, row: dict[str, Any]) -> CanonicalAERecord:
        study_id = str(row["STUDYID"])
        profile = self.profiles.for_study(study_id)
        record_id = str(row["AESPID"])

        coded = self._coded_term(row, profile)
        onset = self._date(row, "AESTDTC", profile, record_id)

        record = CanonicalAERecord(
            record_id=f"{study_id}:{record_id}",
            study_id=study_id,
            subject_id=str(row["USUBJID"]),
            source_record_id=record_id,
            profile=profile.profile_id,
            coded_event=coded,
            concept_id=coded.concept_id if coded else None,
            reported_term=self._free_text(row, "AETERM", profile, record_id),
            modifiers=self._modifiers(row, profile, record_id),
            onset=onset,
            end=self._date(row, "AEENDTC", profile, record_id),
            exposure_relation=self._exposure_relation(
                str(row["USUBJID"]), record_id, onset
            ),
            severity=self._enum(row, "AESEV", "severity", profile, record_id),
            grade=self._grade(row, profile, record_id),
            seriousness=self._bool(row, "AESER", profile, record_id),
            seriousness_criteria=self._criteria(row, profile, record_id),
            relatedness=self._enum(row, "AEREL", "relatedness", profile, record_id),
            action=self._enum(row, "AEACN", "action", profile, record_id),
            outcome=self._enum(row, "AEOUT", "outcome", profile, record_id),
            comment_doc_id=self._comment_doc(record_id),
            linked_form_ids=[
                str(r.get("SCTESTCD")) for r in self.store.linked_form_rows(record_id)
            ],
            normalizer_version=self.configs.normalizer_version,
        )
        return record

    # -- coded values, never rewritten -------------------------------------

    def _coded_term(
        self, row: dict[str, Any], profile: StudyProfile
    ) -> CodedTerm | None:
        code = row.get("AEDECOD")
        if is_blank(code):
            return None
        version = str(row.get("AEDICTVER") or profile.dictionary_version or "")
        target = self.catalog.target_version
        reconciled = self.catalog.reconcile(str(code).strip(), version, target)
        return CodedTerm(
            code=str(code).strip(),
            dictionary=profile.dictionary or self.catalog.dictionary_name,
            dictionary_version=version,
            concept_id=reconciled.concept_id,
            reconciled_to=reconciled.reconciled_to,
            reconciled_version=target,
            reconciliation=reconciled.outcome,
            note=reconciled.note,
        )

    # -- attributes ---------------------------------------------------------

    def _free_text(
        self, row: dict[str, Any], variable: str, profile: StudyProfile,
        record_id: str,
    ) -> Attribute[str]:
        cell = row.get(variable)
        if is_blank(cell):
            return Attribute[str].silent_because(
                profile.availability_for_silence(variable),
                variable=variable, source="structured_standard",
            )
        return Attribute[str].direct(
            "present", variable,
            [structured_span(record_id, variable, cell)], value=str(cell).strip(),
        )

    def _enum(
        self, row: dict[str, Any], variable: str, attribute: str,
        profile: StudyProfile, record_id: str,
    ) -> Attribute[str]:
        value, availability, note = coerce_enum(
            attribute, row.get(variable), profile, variable
        )
        if value is None:
            return Attribute[str].silent_because(
                availability, variable=variable, source="structured_standard",
                note=note,
            )
        return Attribute[str].direct(
            "present", variable,
            [structured_span(record_id, variable, row[variable])], value=value,
        )

    def _grade(
        self, row: dict[str, Any], profile: StudyProfile, record_id: str
    ) -> Attribute[int]:
        value, availability, note = coerce_int(row.get("AEGRADE"), profile, "AEGRADE")
        if value is None:
            return Attribute[int].silent_because(
                availability, variable="AEGRADE", source="structured_standard",
                note=note,
            )
        return Attribute[int].direct(
            "present", "AEGRADE",
            [structured_span(record_id, "AEGRADE", value)], value=value,
        )

    def _bool(
        self, row: dict[str, Any], variable: str, profile: StudyProfile,
        record_id: str,
    ) -> Attribute[bool]:
        value, availability, note = coerce_bool(row.get(variable), profile, variable)
        if value is None:
            return Attribute[bool].silent_because(
                availability, variable=variable, source="structured_standard",
                note=note,
            )
        return Attribute[bool].direct(
            "present" if value else "absent", variable,
            [structured_span(record_id, variable, row[variable])], value=value,
        )

    def _date(
        self, row: dict[str, Any], variable: str, profile: StudyProfile,
        record_id: str,
    ) -> Attribute[_dt.date]:
        value, availability, note = coerce_date(row.get(variable), profile, variable)
        if value is None:
            return Attribute[_dt.date].silent_because(
                availability, variable=variable, source="structured_standard",
                note=note,
            )
        return Attribute[_dt.date].direct(
            "present", variable,
            [structured_span(record_id, variable, row[variable])], value=value,
        )

    def _criteria(
        self, row: dict[str, Any], profile: StudyProfile, record_id: str
    ) -> dict[str, Attribute[bool]]:
        """The seriousness criteria vector, gated on the seriousness answer.

        Where seriousness is No the criteria do not apply — which is different
        from nobody having filled them in.
        """
        answer = profile.gate_answer("seriousness", row)
        gate = profile.gate_for("seriousness_criteria")
        recorded = {
            c.strip() for c in str(row.get("AESCAT") or "").split("|") if c.strip()
        }
        out: dict[str, Attribute[bool]] = {}
        for criterion in SERIOUSNESS_CRITERIA:
            if answer is False and gate is not None:
                out[criterion] = Attribute[bool].silent_because(
                    gate.when_gate_false, variable="AESCAT",
                    source="structured_standard",
                    note="the seriousness gate was answered No, so this criterion "
                         "was never applicable",
                )
            elif answer is None:
                out[criterion] = Attribute[bool].silent_because(
                    profile.availability_for_silence("AESER"), variable="AESCAT",
                    source="structured_standard",
                    note="the seriousness gate itself is unanswered",
                )
            else:
                present = criterion in recorded
                out[criterion] = Attribute[bool].direct(
                    "present" if present else "absent", "AESCAT",
                    [structured_span(record_id, "AESCAT", row.get("AESCAT") or "")],
                    value=present,
                )
        return out

    # -- the modifiers this prototype is about -----------------------------

    def _modifiers(
        self, row: dict[str, Any], profile: StudyProfile, record_id: str
    ) -> dict[str, Attribute[str]]:
        """Resolve each configured modifier from wherever this study keeps it.

        A structured home is settled here. A text home is left unresolved *with
        a note naming the home*, which is what tells the model path there is a
        question worth asking.
        """
        out: dict[str, Attribute[str]] = {}
        for name in self.catalog.modifiers:
            homes = profile.homes_for(name)
            if not homes or all(home.is_nowhere for home in homes):
                out[name] = Attribute[str].silent_because(
                    "not_collected",
                    note=f"{profile.profile_id} records {name} nowhere: it is not "
                         f"recoverable from this study",
                )
                continue
            resolved: Attribute[str] | None = None
            for home in homes:
                if home.is_structured:
                    resolved = self._structured_modifier(
                        row, home, profile, record_id, name
                    )
                    if resolved is not None:
                        break
            if resolved is not None:
                out[name] = resolved
                continue
            text_home = next((h for h in homes if h.is_text), None)
            if text_home is not None:
                out[name] = Attribute[str].silent_because(
                    "unresolved", variable=text_home.variable, source=text_home.kind,
                    note=f"{name} lives in {text_home.variable} in "
                         f"{profile.profile_id}; the deterministic path cannot read "
                         f"prose, so the model path is asked",
                )
            else:
                home = homes[0]
                out[name] = Attribute[str].silent_because(
                    profile.availability_for_silence(home.variable or name),
                    variable=home.variable, source=home.kind,
                    note=f"{home.variable} is empty on this record",
                )
        return out

    def _structured_modifier(
        self, row: dict[str, Any], home: ModifierHome, profile: StudyProfile,
        record_id: str, name: str,
    ) -> Attribute[str] | None:
        variable = home.variable or ""
        if home.kind == "linked_form":
            rows = self.store.linked_form_rows(record_id)
            testcd = variable.split(".", 1)[-1]
            matching = [r for r in rows if str(r.get("SCTESTCD")) == testcd]
            if not matching:
                return None
            cell = matching[0].get("SCORRES")
            domain = "SC"
        else:
            cell = row.get(variable)
            domain = "AE"
        if is_blank(cell):
            return None
        assertion, availability, note = coerce_tristate(cell, profile, variable)
        if assertion is None:
            return Attribute[str].silent_because(
                availability, variable=variable, source=home.kind, note=note,
            )
        return Attribute[str].direct(
            assertion, variable,
            [structured_span(record_id, variable, cell, domain=domain)],
            source=home.kind,
        )

    # -- cross-domain -------------------------------------------------------

    def _exposure_relation(
        self, subject_id: str, record_id: str, onset: Attribute[_dt.date]
    ) -> Attribute[int]:
        """Days from the anchor exposure to onset.

        This attribute exists in no single field: it is AE against EX, computed
        by governed code. Never model reasoning, and stamped `derived` so a
        reader can tell the difference.
        """
        if self.resolver is None:
            return Attribute[int].silent_because(
                "unresolved", variable="AE+EX",
                note="no exposure data was available to resolve the anchor",
            )
        if not onset.observed or onset.value is None:
            return Attribute[int].silent_because(
                "unresolved", variable="AE+EX",
                note=f"the onset date is {onset.availability}, so no offset exists",
            )
        hit = self.resolver.resolve(subject_id, self.anchor, onset_date=onset.value)
        if hit is None:
            return Attribute[int].silent_because(
                "unresolved", variable="AE+EX",
                note=f"no {self.anchor} occurrence in this subject's exposure record",
            )
        offset = (onset.value - hit.date).days
        detail = (
            f"AE.AESTDTC {onset.value.isoformat()} - EX.{self.anchor} "
            f"{hit.date.isoformat()} = {offset} days"
        )
        return Attribute[int].derived(
            offset, "AE+EX", [derived_span(record_id, detail)], note=detail
        )

    def _comment_doc(self, record_id: str) -> str | None:
        documents = self.store.documents_of(record_id)
        return documents[0].doc_id if documents else None

    def cumulative_exposure(
        self, subject_id: str, onset: _dt.date | None
    ) -> tuple[float | None, str]:
        """Total dose taken strictly before onset, from the EX records.

        Governed computation, used by the graded-toxicity definition. Returns
        None where the exposure record cannot support it rather than a zero a
        threshold would silently pass or fail.
        """
        if onset is None:
            return None, "no onset date, so no exposure can be accumulated"
        total = 0.0
        counted = 0
        for row in self.store.subject_rows(subject_id, "ex"):
            start = parse_date(row.get("EXSTDTC"))
            if start is None or start >= onset:
                continue
            end = parse_date(row.get("EXENDTC")) or onset
            days = (min(end, onset) - start).days
            dose = row.get("EXDOSE")
            if dose in (None, "") or days <= 0:
                continue
            total += float(dose) * days
            counted += 1
        if not counted:
            return None, f"no exposure record starts before {onset.isoformat()}"
        return total, (
            f"{counted} exposure record(s) before {onset.isoformat()} summed to "
            f"{total:g} dose-days"
        )


def normalize_store(
    store: TrialStore, configs: Configs,
    anchor_resolver: AnchorResolver | None = None,
) -> list[CanonicalAERecord]:
    """Every AE row in the snapshot, in a stable order."""
    normalizer = RecordNormalizer(configs, store, anchor_resolver)
    return [normalizer.normalize(row) for row in store.ae_rows()]
