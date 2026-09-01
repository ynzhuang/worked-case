"""Value extraction: labs and units, severity, seriousness, action, outcome.

Two rules govern everything in this module.

**Severity and seriousness never touch each other.**  Severity is the intensity
of the event; seriousness is a regulatory category defined by outcome.  A mild
event can be serious and a severe event can be non-serious, so the cue lists are
separate, the fields are separate, and neither writes to the other.

**Units are converted once, at extraction.**  Trials report glucose in mg/dL or
mmol/L depending on region.  A threshold rule applied to an unconverted value
silently misclassifies an entire study, so every lab value carries both what was
reported and its canonical equivalent.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from typing import Any

from ..anchors import parse_date
from ..catalog import ConceptCatalog, ExtractionConfig
from .assertion import _cue_pattern




@dataclass(frozen=True)
class CueHit:
    field: str
    value: str
    cue: str
    start: int
    end: int
    confidence: float


@dataclass(frozen=True)
class LabHit:
    test: str
    value: float
    unit: str
    canonical_value: float | None
    canonical_unit: str
    start: int
    end: int
    surface: str
    collection_date: _dt.date | None
    source: str
    confidence: float
    implausible: bool = False


class ValueExtractor:
    def __init__(self, catalog: ConceptCatalog, config: ExtractionConfig):
        self.catalog = catalog
        self.config = config
        self.values = config.values
        self.labs_config = config.labs
        self._lab_patterns = self._build_lab_patterns()

    # -- enumerated fields --------------------------------------------------

    def allowed_values(self, field: str) -> tuple[str, ...]:
        """The cue vocabulary configured for a field.

        Taken from ``extraction.yaml`` rather than from the model enums: the
        cue space is a property of the config, and the backend maps it onto the
        canonical codelist afterwards.
        """
        return tuple(sorted((self.values.get(field) or {})))

    def find_cues(self, text: str, field: str) -> list[CueHit]:
        """All cue hits for one enumerated field, most specific first.

        Where several values' cues match, the longest cue wins: "not resolved"
        must beat the "resolved" it contains.
        """
        cue_map = self.values.get(field) or {}
        hits: list[CueHit] = []
        for value in sorted(cue_map):
            if not isinstance(cue_map[value], list):
                continue
            for cue in cue_map[value]:
                for match in _cue_pattern(cue).finditer(text):
                    hits.append(
                        CueHit(
                            field=field,
                            value=value,
                            cue=match.group(0),
                            start=match.start(),
                            end=match.end(),
                            confidence=self.config.confidence_for("assertion_cue", 0.9),
                        )
                    )
        return self._drop_shadowed(hits)

    @staticmethod
    def _drop_shadowed(hits: list[CueHit]) -> list[CueHit]:
        """Remove hits whose span is contained in a longer hit for the same field."""
        ordered = sorted(hits, key=lambda h: (h.start, -(h.end - h.start)))
        kept: list[CueHit] = []
        for hit in ordered:
            if any(
                k.start <= hit.start and hit.end <= k.end and k is not hit
                for k in ordered
                if (k.end - k.start) > (hit.end - hit.start)
            ):
                continue
            kept.append(hit)
        return kept

    def single_value(self, text: str, field: str) -> CueHit | None:
        """The single best value for a scalar field, or None.

        The longest cue wins; ties go to the earliest mention, so the result
        does not depend on dictionary ordering.
        """
        hits = self.find_cues(text, field)
        if not hits:
            return None
        return sorted(hits, key=lambda h: (-(h.end - h.start), h.start))[0]

    def multi_value(self, text: str, field: str) -> list[CueHit]:
        """All distinct values for a list-valued field, e.g. seriousness."""
        seen: dict[str, CueHit] = {}
        for hit in self.find_cues(text, field):
            if hit.value not in seen:
                seen[hit.value] = hit
        return [seen[value] for value in sorted(seen)]

    def rescue_treatment(self, text: str, field: str = "rescue_treatment") -> CueHit | None:
        """First hit from a named `cues:` list in extraction config."""
        cues = (self.values.get(field) or {}).get("cues") or []
        for cue in cues:
            match = _cue_pattern(cue).search(text)
            if match:
                return CueHit(
                    field, "true", match.group(0),
                    match.start(), match.end(),
                    self.config.confidence_for("assertion_cue", 0.9),
                )
        return None

    # -- laboratory values --------------------------------------------------

    def _build_lab_patterns(self) -> dict[str, re.Pattern[str]]:
        template = self.labs_config.get("value_pattern", "")
        patterns: dict[str, re.Pattern[str]] = {}
        for test_id, lab in self.catalog.lab_tests.items():
            # Longest name first so "capillary glucose" is not shadowed by
            # "glucose".
            names = sorted(lab.names, key=len, reverse=True)
            alternation = "|".join(re.escape(n) for n in names)
            body = template.replace("{test_names}", alternation)
            patterns[test_id] = re.compile(body, re.IGNORECASE)
        return patterns

    def find_labs(self, text: str) -> list[LabHit]:
        """Laboratory values written in the narrative, converted to canonical units."""
        hits: list[LabHit] = []
        claimed: list[tuple[int, int]] = []
        for test_id in sorted(self._lab_patterns):
            lab = self.catalog.lab_tests[test_id]
            for match in self._lab_patterns[test_id].finditer(text):
                span = (match.start(), match.end())
                if any(s < span[1] and span[0] < e for s, e in claimed):
                    continue
                try:
                    value = float(match.group("value"))
                except (TypeError, ValueError):
                    continue
                unit = (match.group("unit") or "").strip()
                inferred = False
                if not unit:
                    unit = self._infer_unit(test_id, value) or ""
                    inferred = bool(unit)
                if not unit and self.labs_config.get("require_unit_or_inference", True):
                    # A bare number whose magnitude is ambiguous between units is
                    # not a usable value. Dropping it beats guessing wrong.
                    continue
                canonical = lab.to_canonical(value, unit) if unit else None
                if canonical is not None and not lab.plausible(canonical):
                    hits.append(
                        LabHit(
                            test_id, value, unit, canonical, lab.canonical_unit,
                            span[0], span[1], match.group(0), None, "narrative",
                            0.2, implausible=True,
                        )
                    )
                    claimed.append(span)
                    continue
                claimed.append(span)
                hits.append(
                    LabHit(
                        test=test_id,
                        value=value,
                        unit=unit,
                        canonical_value=canonical,
                        canonical_unit=lab.canonical_unit,
                        start=span[0],
                        end=span[1],
                        surface=match.group(0),
                        collection_date=None,
                        source="narrative",
                        confidence=self.config.confidence_for(
                            "lab_inferred_unit" if inferred else "lab_with_unit", 0.9
                        ),
                    )
                )
        return sorted(hits, key=lambda h: (h.start, h.end))

    def _infer_unit(self, test_id: str, value: float) -> str | None:
        """Infer a missing unit from magnitude, conservatively.

        Ranges that overlap between units yield nothing at all, because a wrong
        unit is far worse than a missing value.
        """
        rules = (self.labs_config.get("default_units") or {}).get(test_id) or []
        for rule in rules:
            low = rule.get("min")
            high = rule.get("max")
            if low is not None and value < low:
                continue
            if high is not None and value > high:
                continue
            return rule.get("unit")
        return None

    def _unused_labs_from_lb(
        self,
        rows: list[dict[str, Any]],
        onset_date: _dt.date | None,
        *,
        window_days: int = 0,
    ) -> list[LabHit]:
        """Structured lab results collected on the day of the event.

        Same day by default. A wider window picks up routine surveillance draws
        and results belonging to a neighbouring event on the same subject, and
        offers them as corroboration for this one. Confirmation of a
        hypoglycaemic episode is a same-visit measurement; anything looser is
        borrowing evidence.
        """
        if onset_date is None:
            return []
        code_map = {"GLUC": "GLUCOSE", "HBA1C": "HBA1C"}
        hits: list[LabHit] = []
        for row in rows:
            test_id = code_map.get(str(row.get("LBTESTCD") or "").upper())
            if test_id is None or test_id not in self.catalog.lab_tests:
                continue
            collected = parse_date(row.get("LBDTC"))
            if collected is None or abs((collected - onset_date).days) > window_days:
                continue
            lab = self.catalog.lab_tests[test_id]
            try:
                value = float(row.get("LBORRES"))
            except (TypeError, ValueError):
                continue
            unit = str(row.get("LBORRESU") or "").strip()
            canonical = lab.to_canonical(value, unit) if unit else None
            if canonical is None:
                continue
            rendered = (
                f"LB {row.get('LBTEST') or test_id} {value} {unit} "
                f"on {collected.isoformat()}"
            )
            hits.append(
                LabHit(
                    test=test_id,
                    value=value,
                    unit=unit,
                    canonical_value=canonical,
                    canonical_unit=lab.canonical_unit,
                    start=0,
                    end=len(rendered),
                    surface=rendered,
                    collection_date=collected,
                    source=f"LB:{row.get('USUBJID')}:{row.get('LBSEQ')}",
                    confidence=self.config.confidence_for("lab_with_unit", 0.95),
                )
            )
        return sorted(hits, key=lambda h: (h.source, h.start))
