"""Content hashing.

Three hashes are stamped on every output row and every run manifest:

``extractor_version``
    code version plus extraction and concept config
``definition_hash``
    content hash of the phenotype definition file
``snapshot_id``
    hash of the input data

All of them are content-derived and stable across processes and machines, so
the same inputs always produce the same identifiers.  Nothing time-varying is
ever folded into a hash.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

#: Bumped by hand when extractor behaviour changes in a way config cannot
#: express.  Folded into ``extractor_version``.
EXTRACTOR_CODE_VERSION = "extract-4.0.0"

#: The deterministic path has its own version, separate from the model path,
#: because a normalizer change and an extractor change have different blast
#: radii and a result needs to say which one moved.
NORMALIZER_CODE_VERSION = "normalize-4.0.0"

_HASH_LEN = 16


def canonical_json(payload: Any) -> str:
    """Serialise deterministically: sorted keys, no incidental whitespace."""
    return json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
    )


def hash_payload(payload: Any, *, length: int = _HASH_LEN) -> str:
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return digest[:length] if length else digest


def hash_text(text: str, *, length: int = _HASH_LEN) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[:length] if length else digest


def hash_file(path: str | Path, *, length: int = _HASH_LEN) -> str:
    """Hash a file's bytes, normalising line endings so the hash is portable."""
    raw = Path(path).read_bytes().replace(b"\r\n", b"\n")
    digest = hashlib.sha256(raw).hexdigest()
    return digest[:length] if length else digest


def extractor_version(
    concepts_path: str | Path, extraction_path: str | Path
) -> str:
    """Hash of (code version, extraction.yaml, concepts.yaml)."""
    payload = {
        "code": EXTRACTOR_CODE_VERSION,
        "concepts": hash_file(concepts_path, length=0),
        "extraction": hash_file(extraction_path, length=0),
    }
    return f"{EXTRACTOR_CODE_VERSION}+{hash_payload(payload, length=12)}"


def normalizer_version(
    concepts_path: str | Path, profiles_path: str | Path
) -> str:
    """Hash of (code version, concepts.yaml, study_profiles.yaml)."""
    payload = {
        "code": NORMALIZER_CODE_VERSION,
        "concepts": hash_file(concepts_path, length=0),
        "profiles": hash_file(profiles_path, length=0),
    }
    return f"{NORMALIZER_CODE_VERSION}+{hash_payload(payload, length=12)}"


def snapshot_id(data_dir: str | Path, *, length: int = _HASH_LEN) -> str:
    """Hash of every input file in a data directory, in sorted path order."""
    root = Path(data_dir)
    parts: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.name.startswith("."):
            parts.append((str(path.relative_to(root)), hash_file(path, length=0)))
    return hash_payload(parts, length=length)
