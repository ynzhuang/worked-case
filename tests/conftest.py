"""Shared fixtures.

A small corpus is generated once per session into a temporary directory, so the
tests exercise the real pipeline end to end without depending on whatever
``data/synthetic`` happens to contain.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aelayer.catalog import load_configs  # noqa: E402
from aelayer.generate import generate_corpus  # noqa: E402
from aelayer.ingest import load_store  # noqa: E402
from aelayer.pipeline import Pipeline  # noqa: E402

SEED = 11
MODIFIER = "mucosal_involvement"


@pytest.fixture(scope="session")
def corpus_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("corpus")
    generate_corpus(seed=SEED, out_dir=root, shared_truths=16, extra_per_profile=8)
    return root


@pytest.fixture(scope="session")
def configs():
    return load_configs()


@pytest.fixture(scope="session")
def catalog(configs):
    return configs.catalog


@pytest.fixture(scope="session")
def profiles(configs):
    return configs.profiles


@pytest.fixture(scope="session")
def store(corpus_dir):
    return load_store(corpus_dir)


@pytest.fixture(scope="session")
def pipeline(corpus_dir, tmp_path_factory) -> Pipeline:
    store_path = tmp_path_factory.mktemp("store") / "store.db"
    return Pipeline.load(corpus_dir, store_path=store_path)


@pytest.fixture(scope="session")
def records(pipeline):
    """Records with the model path run. The primary grain."""
    return pipeline.records()


@pytest.fixture(scope="session")
def structured_records(pipeline):
    """The same records with the model path never run."""
    return pipeline.structured_only_records()


@pytest.fixture(scope="session")
def gold(store):
    return store.gold_by_record()


@pytest.fixture(scope="session")
def definition_v1(pipeline):
    """The conservative cut: structured evidence only."""
    return pipeline.definition("cutaneous_mucosal", 1)


@pytest.fixture(scope="session")
def definition_v2(pipeline):
    """Its successor, which also accepts evidence read out of prose."""
    return pipeline.definition("cutaneous_mucosal", 2)


@pytest.fixture(scope="session")
def graded(pipeline):
    """A second, structurally different definition, shipped as configuration."""
    return pipeline.definition("graded_toxicity", 1)


@pytest.fixture(scope="session")
def result(pipeline, definition_v2):
    return pipeline.evaluate(definition_v2)


@pytest.fixture(scope="session")
def assignments(result):
    return result.assignments


@pytest.fixture(scope="session")
def index(pipeline, assignments):
    idx = pipeline.index()
    idx.add_verdicts(assignments)
    return idx


@pytest.fixture(scope="session")
def client(pipeline):
    """A test client pinned to the session's temporary corpus.

    The API caches its pipeline behind an ``lru_cache``; the fixture replaces
    the cached loader wholesale rather than warming it, so no test can reach
    whatever ``data/synthetic`` happens to contain.
    """
    from fastapi.testclient import TestClient

    from aelayer import api as api_module

    original = api_module._pipeline_singleton
    api_module._pipeline_singleton = lambda: pipeline
    try:
        yield TestClient(api_module.app)
    finally:
        api_module._pipeline_singleton = original
