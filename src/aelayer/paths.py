"""Default filesystem locations.

Every path is resolved relative to the repository root so the CLI, the API and
the tests agree without any of them depending on the working directory.
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    override = os.environ.get("AELAYER_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[2]


ROOT = repo_root()
CONFIG_DIR = ROOT / "config"
CONCEPTS_YAML = CONFIG_DIR / "concepts.yaml"
EXTRACTION_YAML = CONFIG_DIR / "extraction.yaml"
PHENOTYPE_DIR = CONFIG_DIR / "phenotypes"
DATA_DIR = ROOT / "data" / "synthetic"
STORE_DB = ROOT / "store.db"
RUNS_DIR = ROOT / "runs"
REPORTS_DIR = ROOT / "reports"
UI_DIR = ROOT / "ui"
