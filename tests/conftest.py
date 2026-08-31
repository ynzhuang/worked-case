"""Shared fixtures.

A small corpus is generated once per session into a temporary directory, so the
tests exercise the real pipeline end to end without depending on whatever the
repository's ``data/synthetic`` happens to contain.
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


@pytest.fixture(scope="session")
def corpus_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("corpus")
    generate_corpus(seed=SEED, n_studies=3, out_dir=root, subjects_per_study=20)
    return root


@pytest.fixture(scope="session")
def configs():
    return load_configs()


@pytest.fixture(scope="session")
def catalog(configs):
    return configs[0]


@pytest.fixture(scope="session")
def extraction_config(configs):
    return configs[1]


@pytest.fixture(scope="session")
def store(corpus_dir):
    return load_store(corpus_dir)


@pytest.fixture(scope="session")
def pipeline(corpus_dir, tmp_path_factory) -> Pipeline:
    store_path = tmp_path_factory.mktemp("store") / "store.db"
    return Pipeline.load(corpus_dir, store_path=store_path)


@pytest.fixture(scope="session")
def events(pipeline):
    return pipeline.events()


@pytest.fixture(scope="session")
def definition_v1(pipeline):
    return pipeline.definition("te_symptomatic_hypoglycemia", 1)


@pytest.fixture(scope="session")
def definition_v2(pipeline):
    return pipeline.definition("te_symptomatic_hypoglycemia", 2)


@pytest.fixture(scope="session")
def index(pipeline, definition_v1):
    idx = pipeline.index()
    idx.record_assignments(pipeline.evaluate(definition_v1))
    return idx


@pytest.fixture(scope="session")
def gold(store):
    return {row["doc_id"]: row for row in store.gold()}
