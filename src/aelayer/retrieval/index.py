"""The retrieval index.

SQLite with FTS5 over free text, plus structured columns for every attribute a
cohort query filters on. Two tables, and the separation is the point:

``records``
    adjudicated units at the source-record grain. A precise query runs here,
    over normalized values, and never consults an embedding to decide cohort
    membership.
``mentions``
    places in a document where something was named. Discovery runs here, and
    everything it returns is a candidate.

A mention is not a record. Somewhere in a narrative saying "mucosal
involvement" is not the same claim as an adjudicated attribute on an event, and
merging the two tables would make that difference disappear at exactly the
moment it matters.

The index records the extractor and normalizer versions it was built with, so a
query can tell whether it is reading a stale index rather than silently
returning last week's answer.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..models import CanonicalAERecord, CaseAssignment

#: Bumped whenever the columns change. An index built under an older version is
#: dropped and rebuilt rather than queried against the wrong grain.
SCHEMA_VERSION = 4

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS records (
    record_id             TEXT PRIMARY KEY,
    source_record_id      TEXT,
    study_id              TEXT,
    subject_id            TEXT,
    profile               TEXT,
    concept               TEXT,
    code                  TEXT,
    dictionary_version    TEXT,
    reconciliation        TEXT,
    modifier              TEXT,
    modifier_assertion    TEXT,
    modifier_availability TEXT,
    modifier_value        TEXT,
    modifier_method       TEXT,
    modifier_source       TEXT,
    modifier_confidence   REAL,
    severity              TEXT,
    grade                 INTEGER,
    exposure_offset_days  INTEGER,
    onset                 TEXT,
    candidate             INTEGER,
    verdict               TEXT,
    definition_id         TEXT,
    definition_version    INTEGER,
    reported_term         TEXT,
    payload               TEXT
);

CREATE TABLE IF NOT EXISTS mentions (
    mention_id       TEXT PRIMARY KEY,
    doc_id           TEXT,
    study_id         TEXT,
    subject_id       TEXT,
    source_record_id TEXT,
    profile          TEXT,
    modifier         TEXT,
    assertion        TEXT,
    value            TEXT,
    surface          TEXT,
    sentence         TEXT,
    source_kind      TEXT,
    source_variable  TEXT,
    confidence       REAL,
    rule             TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS text_index
    USING fts5(doc_id UNINDEXED, kind UNINDEXED, body);

CREATE INDEX IF NOT EXISTS records_by_concept ON records(concept);
CREATE INDEX IF NOT EXISTS records_by_profile ON records(profile);
CREATE INDEX IF NOT EXISTS records_by_assertion ON records(modifier_assertion);
CREATE INDEX IF NOT EXISTS mentions_by_modifier ON mentions(modifier);
"""


@dataclass
class IndexMeta:
    extractor_version: str
    normalizer_version: str
    record_count: int
    mention_count: int

    def stale_against(self, extractor: str, normalizer: str) -> list[str]:
        drift: list[str] = []
        if self.extractor_version != extractor:
            drift.append(
                f"index built with extractor {self.extractor_version!r}, now "
                f"{extractor!r}"
            )
        if self.normalizer_version != normalizer:
            drift.append(
                f"index built with normalizer {self.normalizer_version!r}, now "
                f"{normalizer!r}"
            )
        return drift


class RecordIndex:
    """The queryable index. Thread-safe enough for a single-process server."""

    def __init__(self, path: str | Path | None = None,
                 modifier: str = "mucosal_involvement"):
        self.path = Path(path) if path else None
        self.modifier = modifier
        self._lock = threading.RLock()
        self.connection = sqlite3.connect(
            str(self.path) if self.path else ":memory:",
            check_same_thread=False,
        )
        self.connection.row_factory = sqlite3.Row
        with self._lock:
            self._migrate()
            self.connection.executescript(SCHEMA)
            self.connection.commit()

    def _migrate(self) -> None:
        """Drop an index built under a different schema rather than limp on.

        A stale index that half-answers is worse than no index: the query
        surface would silently return columns from a grain that no longer
        exists. Rebuilding is cheap; a wrong cohort is not.
        """
        existing = {
            row["name"] for row in self.connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
            )
        }
        current = self.connection.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone() if "meta" in existing else None
        if existing and (
            current is None or json.loads(current["value"]) != SCHEMA_VERSION
        ):
            for name in existing:
                self.connection.execute(f"DROP TABLE IF EXISTS {name}")
            self.connection.commit()

    # -- lifecycle ----------------------------------------------------------

    def clear(self) -> None:
        with self._lock:
            for table in ("records", "mentions", "text_index", "meta"):
                self.connection.execute(f"DELETE FROM {table}")
            self.connection.commit()

    def set_meta(self, **values: Any) -> None:
        with self._lock:
            for key, value in values.items():
                self.connection.execute(
                    "INSERT OR REPLACE INTO meta VALUES (?, ?)",
                    (key, json.dumps(value)),
                )
            self.connection.commit()

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.connection.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return default if row is None else json.loads(row["value"])

    def meta(self) -> IndexMeta:
        return IndexMeta(
            extractor_version=self.get_meta("extractor_version", ""),
            normalizer_version=self.get_meta("normalizer_version", ""),
            record_count=self._count("records"),
            mention_count=self._count("mentions"),
        )

    def _count(self, table: str) -> int:
        return int(
            self.connection.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
        )

    # -- loading ------------------------------------------------------------

    def add_records(self, records: Iterable[CanonicalAERecord]) -> int:
        added = 0
        with self._lock:
            for record in records:
                modifier = record.modifiers.get(self.modifier)
                coded = record.coded_event
                self.connection.execute(
                    "INSERT OR REPLACE INTO records VALUES "
                    "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        record.record_id, record.source_record_id, record.study_id,
                        record.subject_id, record.profile, record.concept_id,
                        coded.code if coded else None,
                        coded.dictionary_version if coded else None,
                        coded.reconciliation if coded else None,
                        self.modifier,
                        modifier.assertion if modifier else None,
                        modifier.availability if modifier else "unresolved",
                        modifier.value if modifier else None,
                        modifier.method if modifier else None,
                        modifier.source_variable if modifier else None,
                        modifier.confidence if modifier else None,
                        record.severity.value, record.grade.value,
                        record.exposure_relation.value,
                        record.onset.value.isoformat() if record.onset.value else None,
                        0, None, None, None,
                        record.reported_term.value or "",
                        record.model_dump_json(),
                    ),
                )
                added += 1
            self.connection.commit()
        return added

    def add_verdicts(self, assignments: Sequence[CaseAssignment]) -> int:
        """Attach verdicts to indexed records.

        A verdict is a property of (record, definition version); the index
        holds the most recent one written, and the columns name which.
        """
        with self._lock:
            for assignment in assignments:
                self.connection.execute(
                    "UPDATE records SET verdict = ?, definition_id = ?, "
                    "definition_version = ? WHERE record_id = ?",
                    (assignment.verdict, assignment.definition_id,
                     assignment.definition_version, assignment.record_id),
                )
            self.connection.commit()
        return len(assignments)

    def add_mentions(self, mentions: Iterable[dict[str, Any]]) -> int:
        added = 0
        with self._lock:
            for mention in mentions:
                self.connection.execute(
                    "INSERT OR REPLACE INTO mentions VALUES "
                    "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        mention["mention_id"], mention["doc_id"],
                        mention["study_id"], mention["subject_id"],
                        mention["source_record_id"], mention["profile"],
                        mention["modifier"], mention["assertion"],
                        mention.get("value"), mention["surface"],
                        mention["sentence"], mention["source_kind"],
                        mention["source_variable"], mention["confidence"],
                        mention["rule"],
                    ),
                )
                added += 1
            self.connection.commit()
        return added

    def add_documents(self, documents: Iterable[Any]) -> int:
        added = 0
        with self._lock:
            for document in documents:
                self.connection.execute(
                    "INSERT INTO text_index (doc_id, kind, body) VALUES (?, ?, ?)",
                    (document.doc_id, document.kind, document.full_text),
                )
                added += 1
            self.connection.commit()
        return added

    # -- reading ------------------------------------------------------------

    def query(self, sql: str, parameters: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self.connection.execute(sql, parameters).fetchall())

    def records(self) -> list[CanonicalAERecord]:
        return [
            CanonicalAERecord.model_validate_json(row["payload"])
            for row in self.query(
                "SELECT payload FROM records ORDER BY record_id"
            )
        ]

    def studies(self) -> list[str]:
        return [
            row["study_id"] for row in self.query(
                "SELECT DISTINCT study_id FROM records ORDER BY study_id"
            )
        ]


def build_index(
    path: str | Path | None,
    store: Any,
    records: Sequence[CanonicalAERecord],
    extractor_version: str,
    normalizer_version: str,
    mentions: Sequence[dict[str, Any]] | None = None,
    modifier: str = "mucosal_involvement",
) -> RecordIndex:
    index = RecordIndex(path, modifier=modifier)
    index.clear()
    index.add_records(records)
    index.add_documents(store.documents.values())
    index.add_mentions(mentions or [])
    index.set_meta(
        schema_version=SCHEMA_VERSION,
        extractor_version=extractor_version,
        normalizer_version=normalizer_version,
        snapshot_id=store.snapshot_id,
        modifier=modifier,
    )
    return index


#: Kept so existing imports keep working while the grain changed underneath.
EpisodeIndex = RecordIndex
