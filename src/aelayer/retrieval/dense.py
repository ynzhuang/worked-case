"""Optional dense retrieval.

Dense retrieval is strictly optional and degrades to lexical when no local model
is present, because the default code path must run with the network cable
pulled.

Where it is enabled, assertion filtering still happens as a structured
predicate.  Similarity is used to order candidates, never to decide whether a
record documents an occurrence or an absence — that distinction is not
reliably recoverable from an embedding, and getting it wrong is the failure
mode this whole system exists to prevent.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def _load_model() -> Any | None:
    """Load a local sentence embedding model if one is available offline.

    Never downloads.  ``AELAYER_DENSE_MODEL`` must point at a model already on
    disk; absent that, dense retrieval stays unavailable.
    """
    path = os.environ.get("AELAYER_DENSE_MODEL")
    if not path or not os.path.isdir(path):
        return None
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    try:
        return SentenceTransformer(path, local_files_only=True)
    except Exception:  # pragma: no cover - depends on local model state
        return None


def dense_backend_available() -> bool:
    return _load_model() is not None


def rerank(records: list, query: str, index: Any) -> list:  # pragma: no cover
    """Reorder lexical candidates by embedding similarity to the query."""
    model = _load_model()
    if model is None or not records:
        return records
    import numpy as np

    query_vector = model.encode([query], normalize_embeddings=True)[0]
    snippets = [r.snippet or "" for r in records]
    matrix = model.encode(snippets, normalize_embeddings=True)
    scores = np.asarray(matrix) @ np.asarray(query_vector)
    for record, score in zip(records, scores):
        record.score = float(score)
    return sorted(records, key=lambda r: (-r.score, r.event_id))
