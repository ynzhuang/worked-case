"""SQLite store: FTS5 over narrative text plus structured episode columns.

The structured columns are the point.  Assertion, temporality and provenance are
predicates on columns, not hopes about what an embedding encoded, so
"hypoglycemia" and "no evidence of hypoglycemia" are separable by a WHERE clause.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..ingest import TrialStore
from ..models import CanonicalAEEpisode, CaseAssignment

SCHEMA_VERSION = 5

_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE documents (
    doc_id     TEXT PRIMARY KEY,
    study_id   TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    header     TEXT NOT NULL DEFAULT '',
    text       TEXT NOT NULL
);
CREATE VIRTUAL TABLE documents_fts USING fts5(
    text, content='documents', content_rowid='rowid', tokenize='unicode61'
);

CREATE TABLE episodes (
    episode_id            TEXT PRIMARY KEY,
    subject_id            TEXT NOT NULL,
    study_id              TEXT NOT NULL,
    representation        TEXT NOT NULL DEFAULT '',
    standardized_concept  TEXT,
    coded_terms           TEXT NOT NULL DEFAULT '',
    verbatim_terms        TEXT NOT NULL DEFAULT '',
    dictionary_versions   TEXT NOT NULL DEFAULT '',
    assertions            TEXT NOT NULL DEFAULT '',
    episode_start         TEXT,
    episode_end           TEXT,
    onset_offset_days     INTEGER,
    anchor_event          TEXT,
    peak_severity         TEXT,
    seriousness           INTEGER,
    relatedness           TEXT,
    outcome               TEXT,
    symptoms              TEXT NOT NULL DEFAULT '',
    min_glucose_mgdl      REAL,
    record_count          INTEGER NOT NULL DEFAULT 1,
    source_record_ids     TEXT NOT NULL DEFAULT '',
    doc_ids               TEXT NOT NULL DEFAULT '',
    linkage_rule          TEXT NOT NULL DEFAULT '',
    linkage_confidence    REAL NOT NULL DEFAULT 1.0,
    linkage_review        INTEGER NOT NULL DEFAULT 0,
    provenance_paths      TEXT NOT NULL DEFAULT '',
    candidate             INTEGER NOT NULL DEFAULT 0,
    payload               TEXT NOT NULL
);
CREATE INDEX episodes_concept ON episodes(standardized_concept);
CREATE INDEX episodes_study   ON episodes(study_id);
CREATE INDEX episodes_subject ON episodes(subject_id);

-- Concept mentions in narrative text, each with its assertion.
--
-- This is where assertion actually matters. A coded AE row asserts presence by
-- construction; a narrative can name a concept in order to rule it out, and a
-- discovery search that cannot tell the two apart returns documented absences
-- as though they were events.
CREATE TABLE mentions (
    mention_id  TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL,
    study_id    TEXT NOT NULL,
    subject_id  TEXT NOT NULL,
    concept_id  TEXT NOT NULL,
    assertion   TEXT NOT NULL,
    match_kind  TEXT NOT NULL DEFAULT '',
    start       INTEGER NOT NULL,
    end         INTEGER NOT NULL,
    surface     TEXT NOT NULL DEFAULT '',
    sentence    TEXT NOT NULL DEFAULT '',
    cue         TEXT
);
CREATE INDEX mentions_concept   ON mentions(concept_id);
CREATE INDEX mentions_assertion ON mentions(assertion);
CREATE INDEX mentions_doc       ON mentions(doc_id);

CREATE TABLE episode_states (
    episode_id         TEXT NOT NULL,
    definition_id      TEXT NOT NULL,
    definition_version INTEGER NOT NULL,
    definition_hash    TEXT NOT NULL,
    evidence_state     TEXT NOT NULL,
    verdict            TEXT NOT NULL,
    matched_rule_id    TEXT,
    PRIMARY KEY (episode_id, definition_id, definition_version)
);

CREATE TABLE assignments (
    episode_id         TEXT NOT NULL,
    subject_id         TEXT NOT NULL,
    study_id           TEXT NOT NULL,
    definition_id      TEXT NOT NULL,
    definition_version INTEGER NOT NULL,
    verdict            TEXT NOT NULL,
    evidence_state     TEXT NOT NULL,
    reason             TEXT NOT NULL,
    payload            TEXT NOT NULL,
    PRIMARY KEY (episode_id, definition_id, definition_version)
);
"""


@dataclass(frozen=True)
class IndexMeta:
    schema_version: int
    snapshot_id: str
    extractor_version: str
    normalizer_version: str
    document_count: int
    episode_count: int
    mention_count: int = 0

    def matches(
        self, snapshot_id: str, extractor_version: str, normalizer_version: str
    ) -> bool:
        return (
            self.schema_version == SCHEMA_VERSION
            and self.snapshot_id == snapshot_id
            and self.extractor_version == extractor_version
            and self.normalizer_version == normalizer_version
        )


class EpisodeIndex:
    def __init__(self, connection: sqlite3.Connection, path: Path | None = None):
        self.connection = connection
        self.connection.row_factory = sqlite3.Row
        # The API serves sync endpoints from a threadpool, so connections are
        # opened cross-thread and every statement runs under this lock.
        self._lock = threading.RLock()
        self.path = path

    @contextlib.contextmanager
    def transaction(self):
        with self._lock:
            cursor = self.connection.cursor()
            try:
                yield cursor
                self.connection.commit()
            finally:
                cursor.close()

    def query(self, sql: str, parameters: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self.connection.execute(sql, parameters))

    @classmethod
    def create(cls, path: str | Path | None) -> "EpisodeIndex":
        target = Path(path) if path else None
        if target is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                target.unlink()
        connection = sqlite3.connect(
            str(target) if target else ":memory:", check_same_thread=False
        )
        connection.executescript(_SCHEMA)
        return cls(connection, target)

    @classmethod
    def open(cls, path: str | Path) -> "EpisodeIndex":
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(
                f"no store at {target}. Run `aelayer extract` to build one."
            )
        return cls(sqlite3.connect(str(target), check_same_thread=False), target)

    def close(self) -> None:
        with self._lock:
            self.connection.close()

    # -- population ---------------------------------------------------------

    def populate(
        self,
        store: TrialStore,
        episodes: Iterable[CanonicalAEEpisode],
        extractor_version: str,
        normalizer_version: str,
        mentions: Iterable[dict[str, Any]] = (),
    ) -> None:
        episode_list = list(episodes)
        mention_list = list(mentions)
        with self.transaction() as cursor:
            for narrative in sorted(store.narratives.values(), key=lambda n: n.doc_id):
                cursor.execute(
                    "INSERT INTO documents(doc_id, study_id, subject_id, header, text)"
                    " VALUES (?,?,?,?,?)",
                    (narrative.doc_id, narrative.study_id, narrative.subject_id,
                     narrative.header, narrative.full_text),
                )
            cursor.execute(
                "INSERT INTO documents_fts(rowid, text) "
                "SELECT rowid, text FROM documents"
            )
            doc_by_record = {
                str(row.get("AESPID")): str(row.get("DOCID") or "")
                for row in store.rows("ae")
            }
            for episode in episode_list:
                glucose = [
                    l.canonical_value for l in episode.labs
                    if l.test == "GLUCOSE" and l.canonical_value is not None
                ]
                docs = sorted(
                    {doc_by_record.get(r, "") for r in episode.source_record_ids} - {""}
                )
                cursor.execute(
                    """INSERT INTO episodes(
                        episode_id, subject_id, study_id, representation,
                        standardized_concept, coded_terms, verbatim_terms,
                        dictionary_versions, assertions, episode_start,
                        episode_end, onset_offset_days, anchor_event,
                        peak_severity, seriousness, relatedness, outcome,
                        symptoms, min_glucose_mgdl, record_count,
                        source_record_ids, doc_ids, linkage_rule,
                        linkage_confidence, linkage_review, provenance_paths,
                        candidate, payload
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        episode.episode_id, episode.subject_id, episode.study_id,
                        str(episode.episode_provenance.get("representation_hint") or ""),
                        episode.standardized_concept,
                        "|".join(episode.coded_terms),
                        "|".join(episode.verbatim_terms),
                        "|".join(episode.dictionary_versions),
                        "|".join(episode.assertions),
                        episode.episode_start.value.isoformat()
                        if episode.episode_start.value else None,
                        episode.episode_end.value.isoformat()
                        if episode.episode_end.value else None,
                        episode.onset_offset_days.value,
                        episode.anchor_event,
                        episode.peak_severity,
                        1 if episode.seriousness.value else 0,
                        episode.relatedness.value, episode.outcome.value,
                        "|".join(sorted({s.symptom for s in episode.symptoms})),
                        min(glucose) if glucose else None,
                        len(episode.source_record_ids),
                        "|".join(episode.source_record_ids),
                        "|".join(docs),
                        episode.linkage_rule, episode.linkage_confidence,
                        1 if episode.linkage_review_required else 0,
                        "|".join(sorted({s.kind for s in episode.linked_evidence})),
                        1 if episode.candidate else 0,
                        episode.model_dump_json(),
                    ),
                )
            for mention in mention_list:
                cursor.execute(
                    """INSERT OR REPLACE INTO mentions(
                        mention_id, doc_id, study_id, subject_id, concept_id,
                        assertion, match_kind, start, end, surface, sentence, cue
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        mention["mention_id"], mention["doc_id"],
                        mention["study_id"], mention["subject_id"],
                        mention["concept_id"], mention["assertion"],
                        mention.get("match_kind", ""), mention["start"],
                        mention["end"], mention.get("surface", ""),
                        mention.get("sentence", ""), mention.get("cue"),
                    ),
                )
            cursor.executemany(
                "INSERT INTO meta(key, value) VALUES (?,?)",
                [
                    ("schema_version", str(SCHEMA_VERSION)),
                    ("snapshot_id", store.snapshot_id),
                    ("extractor_version", extractor_version),
                    ("normalizer_version", normalizer_version),
                    ("document_count", str(len(store.narratives))),
                    ("episode_count", str(len(episode_list))),
                    ("mention_count", str(len(mention_list))),
                ],
            )

    def record_assignments(self, assignments: Iterable[CaseAssignment]) -> None:
        rows = list(assignments)
        if not rows:
            return
        first = rows[0]
        with self.transaction() as cursor:
            for table in ("assignments", "episode_states"):
                cursor.execute(
                    f"DELETE FROM {table} WHERE definition_id=? AND "
                    f"definition_version=?",
                    (first.definition_id, first.definition_version),
                )
            for assignment in rows:
                cursor.execute(
                    """INSERT INTO assignments(
                        episode_id, subject_id, study_id, definition_id,
                        definition_version, verdict, evidence_state, reason, payload
                    ) VALUES (?,?,?,?,?,?,?,?,?)""",
                    (
                        assignment.episode_id, assignment.subject_id,
                        assignment.study_id, assignment.definition_id,
                        assignment.definition_version, assignment.verdict,
                        assignment.evidence_state, assignment.reason,
                        assignment.model_dump_json(),
                    ),
                )
                cursor.execute(
                    """INSERT OR REPLACE INTO episode_states(
                        episode_id, definition_id, definition_version,
                        definition_hash, evidence_state, verdict, matched_rule_id
                    ) VALUES (?,?,?,?,?,?,?)""",
                    (
                        assignment.episode_id, assignment.definition_id,
                        assignment.definition_version, assignment.definition_hash,
                        assignment.evidence_state, assignment.verdict,
                        assignment.matched_rule_id,
                    ),
                )

    # -- reading ------------------------------------------------------------

    def meta(self) -> IndexMeta:
        rows = {r["key"]: r["value"] for r in self.query("SELECT key, value FROM meta")}
        return IndexMeta(
            schema_version=int(rows.get("schema_version", 0)),
            snapshot_id=rows.get("snapshot_id", ""),
            extractor_version=rows.get("extractor_version", ""),
            normalizer_version=rows.get("normalizer_version", ""),
            document_count=int(rows.get("document_count", 0)),
            episode_count=int(rows.get("episode_count", 0)),
            mention_count=int(rows.get("mention_count", 0)),
        )

    def episodes(self) -> list[CanonicalAEEpisode]:
        return [
            CanonicalAEEpisode.model_validate_json(row["payload"])
            for row in self.query("SELECT payload FROM episodes ORDER BY episode_id")
        ]

    def assignments(
        self, definition_id: str, definition_version: int
    ) -> list[CaseAssignment]:
        return [
            CaseAssignment.model_validate_json(row["payload"])
            for row in self.query(
                "SELECT payload FROM assignments WHERE definition_id=? AND "
                "definition_version=? ORDER BY episode_id",
                (definition_id, definition_version),
            )
        ]

    def studies(self) -> list[str]:
        return [
            r[0] for r in self.query(
                "SELECT DISTINCT study_id FROM episodes ORDER BY study_id"
            )
        ]

    def document(self, doc_id: str) -> dict[str, Any] | None:
        rows = self.query("SELECT * FROM documents WHERE doc_id=?", (doc_id,))
        return dict(rows[0]) if rows else None


def build_index(
    path: str | Path | None,
    store: TrialStore,
    episodes: Iterable[CanonicalAEEpisode],
    extractor_version: str,
    normalizer_version: str,
    mentions: Iterable[dict[str, Any]] = (),
) -> EpisodeIndex:
    index = EpisodeIndex.create(path)
    index.populate(
        store, episodes, extractor_version, normalizer_version, mentions
    )
    return index
