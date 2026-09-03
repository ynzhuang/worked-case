"""Traceability, including what happens when the chain breaks.

A number that cannot be traced end to end is a failing test, not a caveat — so
the incomplete case has to be representable, and it has to name where it broke.
"""

from __future__ import annotations

import pytest

from aelayer.agent.trace import render_trace, trace_number
from aelayer.runs import execute


@pytest.fixture(scope="module")
def run(pipeline, definition_v2):
    return execute(pipeline, definition_v2, save=False)


def test_a_complete_chain_reaches_source_text(run, records):
    manifest, assignments = run
    chain = trace_number(
        number=1, label="case count", manifest=manifest,
        assignments=assignments, records=records,
    )
    assert chain.complete
    assert chain.broken_at is None
    levels = chain.levels()
    assert levels.index("result") < levels.index("analysis")
    assert levels.index("analysis") < levels.index("cohort")
    assert levels.index("cohort") < levels.index("definition")
    assert "span" in levels


def test_each_hop_names_the_route_the_value_came_by(run, records):
    manifest, assignments = run
    chain = trace_number(
        number=1, label="case count", manifest=manifest,
        assignments=assignments, records=records,
    )
    attributes = [link for link in chain.links if link.level == "attribute"]
    assert attributes
    for link in attributes:
        assert link.payload.get("method")
        assert link.payload.get("source_variable")


def test_a_missing_record_breaks_the_chain_and_says_where(run):
    manifest, assignments = run
    chain = trace_number(
        number=1, label="case count", manifest=manifest,
        assignments=assignments, records=[],
    )
    assert not chain.complete
    assert chain.broken_at == "record"


def test_a_verdict_nothing_reached_breaks_at_the_record_level(run, records):
    manifest, assignments = run
    chain = trace_number(
        number=0, label="nothing", manifest=manifest, assignments=[],
        records=records, verdict="case",
    )
    assert not chain.complete
    assert chain.broken_at == "record"


def test_a_record_without_spans_breaks_at_the_span_level(run, records):
    manifest, assignments = run
    stripped = []
    for record in records:
        copy = record.model_copy(deep=True)
        for name, attribute in copy.attributes().items():
            attribute.evidence.clear()
        stripped.append(copy)
    chain = trace_number(
        number=1, label="case count", manifest=manifest,
        assignments=assignments, records=stripped,
    )
    assert not chain.complete
    assert chain.broken_at == "span"


def test_the_rendering_says_so_when_the_chain_is_incomplete(run):
    manifest, assignments = run
    chain = trace_number(
        number=1, label="case count", manifest=manifest,
        assignments=assignments, records=[],
    )
    text = render_trace(chain)
    assert "INCOMPLETE" in text
    assert "not reportable" in text


def test_the_rendering_indents_by_level(run, records):
    manifest, assignments = run
    text = render_trace(trace_number(
        number=1, label="case count", manifest=manifest,
        assignments=assignments, records=records,
    ))
    lines = text.splitlines()
    result_line = next(l for l in lines if l.startswith("result"))
    span_line = next(l for l in lines if l.strip().startswith("span"))
    assert len(span_line) - len(span_line.lstrip()) > \
        len(result_line) - len(result_line.lstrip())
