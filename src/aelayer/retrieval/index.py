"""The retrieval index.

SQLite with FTS5 over free text, plus structured columns for every attribute a
cohort query filters on. Two tables, and the separation is the point:

``episodes``
    adjudicated units. A precise query runs here, over normalized attribute
    values, and never consults an embedding for cohort membership.
``mentions``
    places in a document where something was named. Discovery runs here, and
    everything it returns is a candidate.

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

from ..models import CanonicalAEEpisode, CaseAssignment

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS episodes (
    episode_id            TEXT PRIMARY KEY,
    study_id              TEXT,
    subject_id            TEXT,
    profile               TEXT,
    concept               TEXT,
    location              TEXT,
    location_method       TEXT,
    location_source       TEXT,
    location_confidence   REAL,
    pattern               TEXT,
    severity              TEXT,
    onset_offset_days     INTEGER,
    episode_start         TEXT,
    record_count          INTEGER,
    linkage_rule          TEXT,
    linkage_confidence    REAL,
    linkage_review        INTEGER,
    candidate             INTEGER,
    verdict               TEXT,
    definition_id         TEXT,
    definition_version    INTEGER,
    reported_terms        TEXT,
    payload               TEXT
);

CREATE TABLE IF NOT EXISTS mentions (
    mention_id       TEXT PRIMARY KEY,
    doc_id           TEXT,
    study_id         TEXT,
    subject_id       TEXT,
    source_record_id TEXT,
    profile          TEXT,
    attribute        TEXT,
    value            TEXT,
    surface          TEXT,
    source_kind      TEXT,
    source_variable  TEXT,
    confidence       REAL,
    normalized       INTEGER,
    sentence         TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS text_index USING fts5(
    mention_id UNINDEXED,
    surface,
    sentence,
    tokenize = 'porter'
);

CREATE INDEX IF NOT EXISTS episodes_by_concept ON episodes(concept);
CREATE INDEX IF NOT EXISTS episodes_by_profile ON episodes(profile);
CREATE INDEX IF NOT EXISTS mentions_by_attribute ON mentions(attribute);
"""


@dataclass(frozen=True)
class IndexMeta:
    extractor_version: str
    normalizer_version: str
    snapshot_id: str
    episode_count: int
    mention_count: int


class EpisodeIndex:
    """Read/write access to one index."""

    def __init__(self, path: Path | None):
        self.path = Path(path) if path else None
        self._local = threading.local()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection.executescript(SCHEMA)
        self.connection.commit()

    @property
    def connection(self) -> sqlite3.Connection:
        """One connection per thread: SQLite objects are not shareable."""
        existing = getattr(self._local, "connection", None)
        if existing is None:
            existing = sqlite3.connect(
                str(self.path) if self.path else ":memory:",
                check_same_thread=False,
            )
            existing.row_factory = sqlite3.Row
            self._local.connection = existing
        return existing

    # -- writing ------------------------------------------------------------

    def clear(self) -> None:
        connection = self.connection
        for table in ("episodes", "mentions", "text_index", "meta"):
            connection.execute(f"DELETE FROM {table}")
        connection.commit()

    def set_meta(self, **values: Any) -> None:
        connection = self.connection
        for key, value in values.items():
            connection.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                (key, json.dumps(value, default=str)),
            )
        connection.commit()

    def meta(self) -> IndexMeta:
        rows = {
            row["key"]: json.loads(row["value"])
            for row in self.connection.execute("SELECT key, value FROM meta")
        }
        return IndexMeta(
            extractor_version=rows.get("extractor_version", ""),
            normalizer_version=rows.get("normalizer_version", ""),
            snapshot_id=rows.get("snapshot_id", ""),
            episode_count=self._count("episodes"),
            mention_count=self._count("mentions"),
        )

    def _count(self, table: str) -> int:
        return int(
            self.connection.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
        )

    def add_episodes(self, episodes: Iterable[CanonicalAEEpisode]) -> int:
        connection = self.connection
        count = 0
        for episode in episodes:
            location = episode.location
            connection.execute(
                """
                INSERT OR REPLACE INTO episodes VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    episode.episode_id, episode.study_id, episode.subject_id,
                    episode.profile, episode.standardized_concept,
                    location.value, location.method, location.source_variable,
                    location.confidence,
                    episode.pattern.value, episode.severity.value,
                    episode.onset_offset_days.value,
                    episode.episode_start.value.isoformat()
                    if episode.episode_start.value else None,
                    len(episode.source_record_ids),
                    episode.linkage_rule, episode.linkage_confidence,
                    int(episode.linkage_review_required), int(episode.candidate),
                    None, None, None,
                    " | ".join(episode.reported_terms),
                    episode.model_dump_json(),
                ),
            )
            count += 1
        connection.commit()
        return count

    def record_assignments(self, assignments: Sequence[CaseAssignment]) -> int:
        """Attach verdicts to indexed episodes.

        A verdict is a property of (episode, definition version); the index
        stores the most recent one written and names the version, so a filter on
        `verdict` can never silently mean a different definition.
        """
        connection = self.connection
        for assignment in assignments:
            connection.execute(
                "UPDATE episodes SET verdict = ?, definition_id = ?, "
                "definition_version = ? WHERE episode_id = ?",
                (assignment.verdict, assignment.definition_id,
                 assignment.definition_version, assignment.episode_id),
            )
        connection.commit()
        return len(assignments)

    def add_mentions(self, mentions: Iterable[dict[str, Any]]) -> int:
        connection = self.connection
        count = 0
        for mention in mentions:
            connection.execute(
                "INSERT OR REPLACE INTO mentions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    mention["mention_id"], mention["doc_id"], mention["study_id"],
                    mention["subject_id"], mention.get("source_record_id"),
                    mention.get("profile"), mention["attribute"], mention["value"],
                    mention["surface"], mention.get("source_kind"),
                    mention.get("source_variable"), mention.get("confidence"),
                    int(bool(mention.get("normalized", True))),
                    mention.get("sentence", ""),
                ),
            )
            connection.execute(
                "INSERT INTO text_index(mention_id, surface, sentence) VALUES (?,?,?)",
                (mention["mention_id"], mention["surface"],
                 mention.get("sentence", "")),
            )
            count += 1
        connection.commit()
        return count

    # -- reading ------------------------------------------------------------

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        return list(self.connection.execute(sql, tuple(params)))

    def episodes(self) -> list[CanonicalAEEpisode]:
        return [
            CanonicalAEEpisode.model_validate_json(row["payload"])
            for row in self.query("SELECT payload FROM episodes ORDER BY episode_id")
        ]

    def studies(self) -> list[str]:
        return [
            row["study_id"]
            for row in self.query(
                "SELECT DISTINCT study_id FROM episodes ORDER BY study_id"
            )
        ]

    def is_stale(self, extractor_version: str, normalizer_version: str) -> bool:
        meta = self.meta()
        return (
            meta.extractor_version != extractor_version
            or meta.normalizer_version != normalizer_version
        )


def build_index(
    path: Path | None,
    store: Any,
    episodes: Sequence[CanonicalAEEpisode],
    extractor_version: str,
    normalizer_version: str,
    mentions: Sequence[dict[str, Any]] = (),
) -> EpisodeIndex:
    index = EpisodeIndex(path)
    index.clear()
    index.add_episodes(episodes)
    index.add_mentions(mentions)
    index.set_meta(
        extractor_version=extractor_version,
        normalizer_version=normalizer_version,
        snapshot_id=getattr(store, "snapshot_id", ""),
    )
    return index
