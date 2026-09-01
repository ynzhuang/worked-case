"""Load SDTM-shaped tables, narratives and gold labels into an in-memory store.

The corpus on disk is:

``dm.csv`` ``ae.csv`` ``ex.csv`` ``lb.csv`` ``cm.csv``
    SDTM-shaped tables, one row per record.
``narratives.jsonl``
    one case narrative per AE record, keyed by ``doc_id``.
``gold.jsonl``
    the generator's ground truth for each AE record.  Never read by the
    extractor or the evaluator; only by the evaluation harness.
``manifest.json``
    generator seed and study-level conventions.

Gold labels are deliberately loaded through a separate accessor so that no
extraction or evaluation code path can reach them by accident.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from . import paths
from .anchors import AnchorResolver, parse_date
from .hashing import snapshot_id as compute_snapshot_id

TABLES = ("dm", "ae", "ex", "lb", "linked_hypo_event")

#: Columns whose values are numeric when present.  Kept narrow on purpose: SDTM
#: character columns stay characters.
_NUMERIC_COLUMNS = {"EXDOSE", "LBSTRESN", "AESEQ", "EXSEQ", "LBSEQ", "AGE", "GLUCVAL"}


class IngestError(RuntimeError):
    pass


@dataclass
class Narrative:
    doc_id: str
    study_id: str
    subject_id: str
    ae_seq: int | None
    text: str
    header: str = ""

    @property
    def full_text(self) -> str:
        """Header plus body, which is what span offsets are measured against."""
        return f"{self.header}\n{self.text}" if self.header else self.text


@dataclass
class TrialStore:
    """Everything ingested from one data snapshot."""

    root: Path
    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    narratives: dict[str, Narrative] = field(default_factory=dict)
    snapshot_id: str = ""
    manifest: dict[str, Any] = field(default_factory=dict)

    # -- derived indices ----------------------------------------------------

    def __post_init__(self) -> None:
        self._by_subject: dict[str, dict[str, list[dict]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for name, rows in self.tables.items():
            for row in rows:
                subject = row.get("USUBJID")
                if subject:
                    self._by_subject[subject][name].append(row)

    def rows(self, table: str) -> list[dict[str, Any]]:
        return self.tables.get(table, [])

    def subject_rows(self, subject_id: str, table: str) -> list[dict[str, Any]]:
        return self._by_subject.get(subject_id, {}).get(table, [])

    def subjects(self) -> list[str]:
        return sorted(self._by_subject)

    def studies(self) -> list[str]:
        return sorted({row["STUDYID"] for row in self.rows("dm") if row.get("STUDYID")})

    def subjects_in_study(self, study_id: str) -> list[str]:
        return sorted(
            row["USUBJID"] for row in self.rows("dm") if row.get("STUDYID") == study_id
        )

    def study_of(self, subject_id: str) -> str | None:
        for row in self.subject_rows(subject_id, "dm"):
            return row.get("STUDYID")
        return None

    def exposures_by_subject(self) -> dict[str, list[dict[str, Any]]]:
        return {s: self.subject_rows(s, "ex") for s in self.subjects()}

    def anchor_resolver(self, anchor_config: dict[str, Any]) -> AnchorResolver:
        return AnchorResolver(anchor_config, self.exposures_by_subject())

    def narratives_for(self, subject_id: str) -> list[Narrative]:
        return sorted(
            (n for n in self.narratives.values() if n.subject_id == subject_id),
            key=lambda n: n.doc_id,
        )

    def iter_ae_with_narrative(self) -> Iterator[tuple[dict[str, Any], Narrative | None]]:
        """AE records in a stable order, paired with their narrative."""
        for row in sorted(
            self.rows("ae"),
            key=lambda r: (str(r.get("STUDYID")), str(r.get("USUBJID")),
                           int(r.get("AESEQ") or 0)),
        ):
            yield row, self.narratives.get(str(row.get("DOCID") or ""))

    def linked_rows(self, source_record_id: str) -> list[dict[str, Any]]:
        return [
            row for row in self.rows("linked_hypo_event")
            if str(row.get("AESPID")) == source_record_id
        ]

    # -- ground truth -------------------------------------------------------
    #
    # Reached only through these accessors, and only by the evaluation harness.
    # Nothing in normalization, extraction, reconciliation or evaluation may
    # touch them: an answer key readable from the pipeline is not an answer key.

    def _jsonl(self, name: str) -> list[dict[str, Any]]:
        path = self.root / name
        return [json.loads(line) for line in _iter_lines(path)] if path.exists() else []

    def gold_records(self) -> list[dict[str, Any]]:
        """True field values and collection states, one entry per source record."""
        return self._jsonl("gold_records.jsonl")

    def gold_episodes(self) -> list[dict[str, Any]]:
        """True episode boundaries and phenotype classification."""
        return self._jsonl("gold_episodes.jsonl")

    def truths(self) -> list[dict[str, Any]]:
        """The sampled ground truth, before any study wrote it down."""
        return self._jsonl("truths.jsonl")

    def gold_records_by_id(self) -> dict[str, dict[str, Any]]:
        return {g["source_record_id"]: g for g in self.gold_records()}

    def study_conventions(self, study_id: str) -> dict[str, Any]:
        return (self.manifest.get("studies") or {}).get(study_id, {})

    def summary(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "studies": len(self.studies()),
            "subjects": len(self.subjects()),
            "ae_records": len(self.rows("ae")),
            "narratives": len(self.narratives),
            "lb_records": len(self.rows("lb")),
            "ex_records": len(self.rows("ex")),
            "linked_form_records": len(self.rows("linked_hypo_event")),
        }


def _iter_lines(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield line


def _coerce(column: str, value: str) -> Any:
    if value == "":
        return None
    if column in _NUMERIC_COLUMNS:
        try:
            number = float(value)
        except ValueError:
            return value
        return int(number) if number.is_integer() else number
    return value


def load_table(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            {column: _coerce(column, value) for column, value in row.items()}
            for row in reader
        ]


def load_store(data_dir: str | Path | None = None) -> TrialStore:
    """Ingest a data snapshot and stamp its content hash."""
    root = Path(data_dir or paths.DATA_DIR)
    if not root.exists():
        raise IngestError(
            f"no data at {root}. Run `aelayer generate` first — this repository "
            f"generates its own corpus and ships no patient data."
        )

    tables: dict[str, list[dict[str, Any]]] = {}
    for name in TABLES:
        path = root / f"{name}.csv"
        if not path.exists():
            if name in ("dm", "ae"):
                raise IngestError(f"required table {name}.csv missing from {root}")
            tables[name] = []
            continue
        tables[name] = load_table(path)

    narratives: dict[str, Narrative] = {}
    narrative_path = root / "narratives.jsonl"
    if narrative_path.exists():
        for line in _iter_lines(narrative_path):
            payload = json.loads(line)
            narrative = Narrative(
                doc_id=payload["doc_id"],
                study_id=payload["study_id"],
                subject_id=payload["subject_id"],
                ae_seq=payload.get("ae_seq"),
                text=payload["text"],
                header=payload.get("header", ""),
            )
            narratives[narrative.doc_id] = narrative

    manifest: dict[str, Any] = {}
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # The snapshot hash covers the input tables and narratives, never the gold
    # labels: gold is evaluation scaffolding, not input data, and including it
    # would make a run id change when only the answer key changed.
    inputs = [root / f"{n}.csv" for n in TABLES] + [narrative_path, manifest_path]
    # The answer key is evaluation scaffolding, not input data: a run id must
    # not change when only the answer key changes.
    payload = []
    from .hashing import hash_file

    for path in inputs:
        if path.exists():
            payload.append((path.name, hash_file(path, length=0)))
    from .hashing import hash_payload

    store = TrialStore(
        root=root,
        tables=tables,
        narratives=narratives,
        snapshot_id=hash_payload(payload),
        manifest=manifest,
    )
    return store


__all__ = [
    "IngestError",
    "Narrative",
    "TrialStore",
    "load_store",
    "load_table",
    "parse_date",
    "compute_snapshot_id",
]
