"""SQLite store: FTS5 over narrative text plus structured event columns.

The structured columns are the point.  Assertion is a column, not a hope about
what an embedding encodes, so "hypoglycemia" and "no evidence of hypoglycemia"
are separable by a predicate rather than by similarity.

Evidence state lives in its own table keyed by definition id and version,
because a state is assigned by a definition and a different version can assign a
different one to the same event.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..hashing import canonical_json
from ..ingest import TrialStore
from ..models import CaseAssignment, EventObject

SCHEMA_VERSION = 3

_SCHEMA = """
CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE documents (
    doc_id     TEXT PRIMARY KEY,
    study_id   TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    header     TEXT NOT NULL DEFAULT '',
    text       TEXT NOT NULL
);

-- Lexical index over narrative text. `content=` keeps a single copy of the
-- text in `documents` and lets FTS index it in place.
CREATE VIRTUAL TABLE documents_fts USING fts5(
    text,
    content='documents',
    content_rowid='rowid',
    tokenize='unicode61'
);

CREATE TABLE events (
    event_id          TEXT PRIMARY KEY,
    subject_id        TEXT NOT NULL,
    study_id          TEXT NOT NULL,
    doc_id            TEXT NOT NULL,
    concept_id        TEXT NOT NULL,
    coded_term        TEXT,
    coded_term_version TEXT,
    assertion         TEXT NOT NULL,
    onset_date        TEXT,
    onset_offset_days INTEGER,
    anchor_event      TEXT,
    severity          TEXT,
    seriousness       TEXT NOT NULL DEFAULT '',
    relatedness       TEXT,
    action_taken      TEXT,
    rechallenge       TEXT,
    rescue_treatment  INTEGER NOT NULL DEFAULT 0,
    outcome           TEXT,
    symptoms          TEXT NOT NULL DEFAULT '',
    min_glucose_mgdl  REAL,
    extractor_version TEXT NOT NULL,
    payload           TEXT NOT NULL
);

CREATE INDEX events_concept   ON events(concept_id);
CREATE INDEX events_assertion ON events(assertion);
CREATE INDEX events_study     ON events(study_id);
CREATE INDEX events_subject   ON events(subject_id);
CREATE INDEX events_doc       ON events(doc_id);

CREATE TABLE event_states (
    event_id           TEXT NOT NULL,
    definition_id      TEXT NOT NULL,
    definition_version INTEGER NOT NULL,
    definition_hash    TEXT NOT NULL,
    evidence_state     TEXT NOT NULL,
    verdict            TEXT NOT NULL,
    matched_rule_id    TEXT,
    PRIMARY KEY (event_id, definition_id, definition_version)
);

CREATE TABLE assignments (
    subject_id         TEXT NOT NULL,
    study_id           TEXT NOT NULL,
    definition_id      TEXT NOT NULL,
    definition_version INTEGER NOT NULL,
    definition_hash    TEXT NOT NULL,
    verdict            TEXT NOT NULL,
    evidence_state     TEXT NOT NULL,
    matched_rule_id    TEXT,
    reason             TEXT NOT NULL,
    payload            TEXT NOT NULL,
    PRIMARY KEY (subject_id, definition_id, definition_version)
);
"""


@dataclass(frozen=True)
class IndexMeta:
    schema_version: int
    snapshot_id: str
    extractor_version: str
    document_count: int
    event_count: int

    def matches(self, snapshot_id: str, extractor_version: str) -> bool:
        """Is this index still valid for the given data and extractor?"""
        return (
            self.schema_version == SCHEMA_VERSION
            and self.snapshot_id == snapshot_id
            and self.extractor_version == extractor_version
        )


class EventIndex:
    """Read/write access to the store."""

    def __init__(self, connection: sqlite3.Connection, path: Path | None = None):
        self.connection = connection
        self.connection.row_factory = sqlite3.Row
        self.path = path

    # -- lifecycle ----------------------------------------------------------

    @classmethod
    def create(cls, path: str | Path | None) -> "EventIndex":
        target = Path(path) if path else None
        if target is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                target.unlink()
        connection = sqlite3.connect(str(target) if target else ":memory:")
        connection.executescript(_SCHEMA)
        return cls(connection, target)

    @classmethod
    def open(cls, path: str | Path) -> "EventIndex":
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(
                f"no store at {target}. Run `aelayer extract` to build one."
            )
        return cls(sqlite3.connect(str(target)), target)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "EventIndex":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- population ---------------------------------------------------------

    def populate(
        self, store: TrialStore, events: Iterable[EventObject], extractor_version: str
    ) -> None:
        cursor = self.connection.cursor()
        for narrative in sorted(store.narratives.values(), key=lambda n: n.doc_id):
            cursor.execute(
                "INSERT INTO documents(doc_id, study_id, subject_id, header, text) "
                "VALUES (?,?,?,?,?)",
                (
                    narrative.doc_id, narrative.study_id, narrative.subject_id,
                    narrative.header, narrative.full_text,
                ),
            )
        cursor.execute(
            "INSERT INTO documents_fts(rowid, text) SELECT rowid, text FROM documents"
        )

        event_list = list(events)
        for event in event_list:
            cursor.execute(
                """INSERT INTO events(
                    event_id, subject_id, study_id, doc_id, concept_id, coded_term,
                    coded_term_version, assertion, onset_date, onset_offset_days,
                    anchor_event, severity, seriousness, relatedness, action_taken,
                    rechallenge, rescue_treatment, outcome, symptoms,
                    min_glucose_mgdl, extractor_version, payload
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    event.event_id, event.subject_id, event.study_id, event.doc_id,
                    event.concept_id, event.coded_term, event.coded_term_version,
                    event.assertion,
                    event.onset_date.isoformat() if event.onset_date else None,
                    event.onset_offset_days, event.anchor_event, event.severity,
                    "|".join(event.seriousness), event.relatedness,
                    event.action_taken, event.rechallenge,
                    1 if event.rescue_treatment else 0, event.outcome,
                    "|".join(sorted({s.symptom for s in event.symptoms})),
                    _min_glucose(event),
                    event.extractor_version,
                    event.model_dump_json(),
                ),
            )
        cursor.executemany(
            "INSERT INTO meta(key, value) VALUES (?,?)",
            [
                ("schema_version", str(SCHEMA_VERSION)),
                ("snapshot_id", store.snapshot_id),
                ("extractor_version", extractor_version),
                ("document_count", str(len(store.narratives))),
                ("event_count", str(len(event_list))),
            ],
        )
        self.connection.commit()

    def record_assignments(
        self,
        assignments: Iterable[CaseAssignment],
        event_states: Iterable[tuple[str, str, str]] | None = None,
    ) -> None:
        """Store one definition's verdicts, replacing any previous run of it."""
        cursor = self.connection.cursor()
        rows = list(assignments)
        if rows:
            first = rows[0]
            cursor.execute(
                "DELETE FROM assignments WHERE definition_id=? AND definition_version=?",
                (first.definition_id, first.definition_version),
            )
            cursor.execute(
                "DELETE FROM event_states WHERE definition_id=? AND definition_version=?",
                (first.definition_id, first.definition_version),
            )
        for assignment in rows:
            cursor.execute(
                """INSERT INTO assignments(
                    subject_id, study_id, definition_id, definition_version,
                    definition_hash, verdict, evidence_state, matched_rule_id,
                    reason, payload
                ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    assignment.subject_id, assignment.study_id,
                    assignment.definition_id, assignment.definition_version,
                    assignment.definition_hash, assignment.verdict,
                    assignment.evidence_state, assignment.matched_rule_id,
                    assignment.reason, assignment.model_dump_json(),
                ),
            )
            for event_id in assignment.contributing_event_ids:
                cursor.execute(
                    """INSERT OR REPLACE INTO event_states(
                        event_id, definition_id, definition_version, definition_hash,
                        evidence_state, verdict, matched_rule_id
                    ) VALUES (?,?,?,?,?,?,?)""",
                    (
                        event_id, assignment.definition_id,
                        assignment.definition_version, assignment.definition_hash,
                        assignment.evidence_state, assignment.verdict,
                        assignment.matched_rule_id,
                    ),
                )
        self.connection.commit()

    # -- reading ------------------------------------------------------------

    def meta(self) -> IndexMeta:
        rows = {
            row["key"]: row["value"]
            for row in self.connection.execute("SELECT key, value FROM meta")
        }
        return IndexMeta(
            schema_version=int(rows.get("schema_version", 0)),
            snapshot_id=rows.get("snapshot_id", ""),
            extractor_version=rows.get("extractor_version", ""),
            document_count=int(rows.get("document_count", 0)),
            event_count=int(rows.get("event_count", 0)),
        )

    def events(self) -> list[EventObject]:
        return [
            EventObject.model_validate_json(row["payload"])
            for row in self.connection.execute(
                "SELECT payload FROM events ORDER BY event_id"
            )
        ]

    def assignments(
        self, definition_id: str, definition_version: int
    ) -> list[CaseAssignment]:
        return [
            CaseAssignment.model_validate_json(row["payload"])
            for row in self.connection.execute(
                "SELECT payload FROM assignments WHERE definition_id=? AND "
                "definition_version=? ORDER BY subject_id",
                (definition_id, definition_version),
            )
        ]

    def document(self, doc_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM documents WHERE doc_id=?", (doc_id,)
        ).fetchone()
        return dict(row) if row else None

    def studies(self) -> list[str]:
        return [
            row[0]
            for row in self.connection.execute(
                "SELECT DISTINCT study_id FROM documents ORDER BY study_id"
            )
        ]


def _min_glucose(event: EventObject) -> float | None:
    """Lowest canonical glucose on the event, for cheap threshold filtering."""
    values = [
        lab.canonical_value
        for lab in event.labs
        if lab.test == "GLUCOSE" and lab.canonical_value is not None
    ]
    return min(values) if values else None


def build_index(
    path: str | Path | None,
    store: TrialStore,
    events: Iterable[EventObject],
    extractor_version: str,
) -> EventIndex:
    index = EventIndex.create(path)
    index.populate(store, events, extractor_version)
    return index
