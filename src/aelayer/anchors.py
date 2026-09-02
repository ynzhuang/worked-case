"""Anchor resolution against the structured exposure record.

A phenotype window is measured from an event — first exposure, a dose
escalation — and that event lives in the EX domain rather than being written
anywhere as such. This module is the single place that decides which EX record
*is* the anchor, so the episode reconciler and the phenotype evaluator can never
disagree about it.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class AnchorHit:
    """One resolved anchor occurrence."""

    event: str
    date: _dt.date
    domain: str
    record_id: str
    detail: str


def parse_date(value: Any) -> _dt.date | None:
    """Parse an ISO-8601 date, tolerating partial SDTM dates and empties."""
    if value in (None, "", "NA"):
        return None
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S", "%d-%b-%Y", "%d/%b/%Y"):
        try:
            return _dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


class AnchorResolver:
    """Locates anchor events in a subject's exposure records.

    Rules are declared in ``extraction.yaml`` under ``anchors``:

    ``first_record``
        the earliest exposure record — study drug initiation
    ``dose_increase``
        the first record whose dose exceeds the running maximum, i.e. an
        escalation, which SDTM represents as a new EX row rather than a flag
    ``dose_decrease``
        the mirror case, a reduction
    """

    def __init__(self, anchor_config: dict[str, Any], exposures_by_subject: dict[str, list[dict]]):
        self.config = anchor_config or {}
        self.exposures = exposures_by_subject

    def known_events(self) -> list[str]:
        return sorted(self.config)

    def occurrences(self, subject_id: str, event: str) -> list[AnchorHit]:
        """Every occurrence of an anchor event for a subject, in date order."""
        spec = self.config.get(event)
        if spec is None:
            return []
        rows = self._sorted_rows(subject_id, spec.get("date_field", "EXSTDTC"))
        rule = spec.get("rule", "first_record")
        domain = spec.get("domain", "EX")
        date_field = spec.get("date_field", "EXSTDTC")
        hits: list[AnchorHit] = []

        if rule == "first_record":
            if rows:
                row, when = rows[0]
                hits.append(
                    AnchorHit(
                        event=event,
                        date=when,
                        domain=domain,
                        record_id=self._record_id(row, domain),
                        detail=f"first {domain} record ({date_field}={when.isoformat()})",
                    )
                )
            return hits

        if rule in ("dose_increase", "dose_decrease"):
            previous: float | None = None
            for row, when in rows:
                dose = _as_float(row.get("EXDOSE"))
                if dose is None:
                    continue
                if previous is not None:
                    increased = dose > previous
                    if (rule == "dose_increase" and increased) or (
                        rule == "dose_decrease" and dose < previous
                    ):
                        direction = "increase" if increased else "decrease"
                        hits.append(
                            AnchorHit(
                                event=event,
                                date=when,
                                domain=domain,
                                record_id=self._record_id(row, domain),
                                detail=(
                                    f"{domain} dose {direction} "
                                    f"{previous:g}->{dose:g} {row.get('EXDOSU', '')}"
                                    f" on {when.isoformat()}"
                                ).strip(),
                            )
                        )
                previous = dose if previous is None else max(previous, dose) if rule == "dose_increase" else dose
            return hits

        raise ValueError(f"unknown anchor rule {rule!r} for event {event!r}")

    def resolve(
        self,
        subject_id: str,
        event: str,
        *,
        index_rule: Literal["first_occurrence", "most_recent_before_onset"] = "first_occurrence",
        onset_date: _dt.date | None = None,
    ) -> AnchorHit | None:
        """Pick the single anchor occurrence a window should be measured from."""
        hits = self.occurrences(subject_id, event)
        if not hits:
            return None
        if index_rule == "first_occurrence":
            return hits[0]
        if index_rule == "most_recent_before_onset":
            if onset_date is None:
                return hits[0]
            prior = [h for h in hits if h.date <= onset_date]
            return prior[-1] if prior else None
        raise ValueError(f"unknown index_rule {index_rule!r}")

    # -- internals ----------------------------------------------------------

    def _sorted_rows(self, subject_id: str, date_field: str) -> list[tuple[dict, _dt.date]]:
        rows = []
        for row in self.exposures.get(subject_id, []):
            when = parse_date(row.get(date_field))
            if when is not None:
                rows.append((row, when))
        # Sort by date then sequence, so ties resolve deterministically.
        rows.sort(key=lambda rw: (rw[1], _as_float(rw[0].get("EXSEQ")) or 0.0))
        return rows

    @staticmethod
    def _record_id(row: dict, domain: str) -> str:
        seq = row.get(f"{domain}SEQ") or row.get("SEQ") or "?"
        return f"{domain}:{row.get('USUBJID', '?')}:{seq}"


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
