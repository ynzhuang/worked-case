"""Load the tables, documents and gold labels of one snapshot into memory.

The corpus on disk is:

``dm.csv`` ``ae.csv`` ``ex.csv``
    the standard domains, one row per record
``sc.csv``
    a linked findings form, keyed back to an AE record by ``IDVAR``/``IDVARVAL``
    — the same clinical fact recorded on a different form
``co.csv``
    comment records pointing at an AE record
``documents.jsonl``
    the free text of each comment, keyed by ``doc_id``, which is what span
    offsets are measured against
``truths.jsonl`` ``gold.jsonl``
    the generator's answer key

Gold labels are loaded through separate accessors so no extraction or
evaluation code path can reach them by accident.
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

TABLES = ("dm", "ae", "ex", "sc", "co")

#: Columns whose values are numeric when present.  Kept narrow on purpose: SDTM
#: character columns stay characters.
_NUMERIC_COLUMNS = {"EXDOSE", "AESEQ", "EXSEQ", "COSEQ", "AGE", "AEGRADE"}


class IngestError(RuntimeError):
    pass


@dataclass
class Document:
    """One piece of free text, with the record it belongs to."""

    doc_id: str
    study_id: str
    subject_id: str
    source_record_id: str
    text: str
    kind: str = "comment"
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
    documents: dict[str, Document] = field(default_factory=dict)
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

    def documents_for(self, subject_id: str) -> list[Document]:
        return sorted(
            (d for d in self.documents.values() if d.subject_id == subject_id),
            key=lambda d: d.doc_id,
        )

    def ae_rows(self) -> list[dict[str, Any]]:
        """AE records in a stable order."""
        return sorted(
            self.rows("ae"),
            key=lambda r: (str(r.get("STUDYID")), str(r.get("USUBJID")),
                           str(r.get("AESPID"))),
        )

    def linked_form_rows(self, source_record_id: str) -> list[dict[str, Any]]:
        """Linked-form findings attached to one AE record."""
        return [
            row for row in self.rows("sc")
            if str(row.get("IDVARVAL")) == source_record_id
        ]

    def comments_for(self, source_record_id: str) -> list[dict[str, Any]]:
        return [
            row for row in self.rows("co")
            if str(row.get("IDVARVAL")) == source_record_id
        ]

    def documents_of(self, source_record_id: str) -> list[Document]:
        return sorted(
            (d for d in self.documents.values()
             if d.source_record_id == source_record_id),
            key=lambda d: d.doc_id,
        )

    # -- ground truth -------------------------------------------------------
    #
    # Reached only through these accessors, and only by the evaluation harness.
    # Nothing in normalization, extraction, reconciliation or evaluation may
    # touch them: an answer key readable from the pipeline is not an answer key.

    def _jsonl(self, name: str) -> list[dict[str, Any]]:
        path = self.root / name
        return [json.loads(line) for line in _iter_lines(path)] if path.exists() else []

    def gold(self) -> list[dict[str, Any]]:
        """True assertion, availability and verdict, one entry per record."""
        return self._jsonl("gold.jsonl")

    def truths(self) -> list[dict[str, Any]]:
        """The sampled ground truth, before any study wrote it down."""
        return self._jsonl("truths.jsonl")

    def gold_by_record(self) -> dict[str, dict[str, Any]]:
        return {g["source_record_id"]: g for g in self.gold()}

    def profile_of(self, study_id: str) -> str | None:
        for profile_id, body in (self.manifest.get("profiles") or {}).items():
            if body.get("study_id") == study_id:
                return profile_id
        return None

    def summary(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "studies": len(self.studies()),
            "subjects": len(self.subjects()),
            "ae_records": len(self.rows("ae")),
            "documents": len(self.documents),
            "linked_form_records": len(self.rows("sc")),
            "comment_records": len(self.rows("co")),
            "ex_records": len(self.rows("ex")),
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

    documents: dict[str, Document] = {}
    document_path = root / "documents.jsonl"
    if document_path.exists():
        for line in _iter_lines(document_path):
            payload = json.loads(line)
            document = Document(
                doc_id=payload["doc_id"],
                study_id=payload["study_id"],
                subject_id=payload["subject_id"],
                source_record_id=payload.get("source_record_id", ""),
                text=payload["text"],
                kind=payload.get("kind", "comment"),
                header=payload.get("header", ""),
            )
            documents[document.doc_id] = document

    manifest: dict[str, Any] = {}
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # The snapshot hash covers the input tables and narratives, never the gold
    # labels: gold is evaluation scaffolding, not input data, and including it
    # would make a run id change when only the answer key changed.
    inputs = [root / f"{n}.csv" for n in TABLES] + [document_path, manifest_path]
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
        documents=documents,
        snapshot_id=hash_payload(payload),
        manifest=manifest,
    )
    return store


__all__ = [
    "Document",
    "IngestError",
    "TrialStore",
    "load_store",
    "load_table",
    "parse_date",
    "compute_snapshot_id",
]
