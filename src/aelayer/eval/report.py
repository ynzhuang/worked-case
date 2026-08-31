"""Render the evaluation results as markdown.

Kept separate from the harness so that adding a table never risks changing a
number.
"""

from __future__ import annotations

from typing import Any

from .harness import DISCLAIMER


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(c) for c in row) + " |")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if value is None:
        return "-"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value) or "-"
    return str(value)


def _prf_rows(section: dict[str, dict[str, Any]]) -> list[list[Any]]:
    """Rows for a P/R/F1 table, omitting slots with no gold instances."""
    return [
        [
            name, d["precision"], d["recall"], d["f1"],
            d["support"], d["tp"], d["fp"], d["fn"],
        ]
        for name, d in section.items()
        if d["support"] or d["predicted"]
    ]


def render_markdown(results: dict[str, Any]) -> str:
    out: list[str] = []
    add = out.append

    definition = results["definition"]
    corpus = results["corpus"]

    add("# Adverse event evidence layer — evaluation report")
    add("")
    add(f"> {DISCLAIMER}")
    add("")
    add(_table(
        ["", "value"],
        [
            ["generated", results["generated_at"]],
            ["definition", f"`{definition['id']}` v{definition['version']} ({definition['status']})"],
            ["definition hash", f"`{definition['hash']}`"],
            ["extractor version", f"`{results['extractor_version']}`"],
            ["data snapshot", f"`{results['snapshot_id']}`"],
            ["studies / subjects", f"{corpus['studies']} / {corpus['subjects']}"],
            ["AE records / narratives", f"{corpus['ae_records']} / {corpus['narratives']}"],
        ],
    ))
    add("")

    # -- extraction ---------------------------------------------------------
    extraction = results["extraction"]
    add("## 1. Extraction")
    add("")
    add(
        f"{extraction['event_count']} event objects from "
        f"{extraction['gold_record_count']} gold records."
    )
    violations = extraction["provenance_violations"]
    if violations:
        add("")
        add(
            f"**{len(violations)} event object(s) carry a populated field with no "
            f"span.** Every one of these is a defect:"
        )
        for violation in violations[:10]:
            add(f"- `{violation}`")
    else:
        add("")
        add(
            "Every populated field on every event object traces to at least one "
            "span. No provenance violations."
        )
    add("")

    add("### Concept detection")
    add("")
    add(_table(
        ["precision", "recall", "f1", "gold", "tp", "fp", "fn"],
        [[
            extraction["concept_detection"]["precision"],
            extraction["concept_detection"]["recall"],
            extraction["concept_detection"]["f1"],
            extraction["concept_detection"]["support"],
            extraction["concept_detection"]["tp"],
            extraction["concept_detection"]["fp"],
            extraction["concept_detection"]["fn"],
        ]],
    ))
    add("")

    add("### Per field")
    add("")
    add(_table(
        ["field", "precision", "recall", "f1", "gold", "tp", "fp", "fn"],
        _prf_rows(extraction["overall"]),
    ))
    add("")

    add("### Assertion classification")
    add("")
    matrix = extraction.get("assertion_confusion_matrix")
    if matrix is not None:
        add(matrix.to_markdown())
        add("")
        add(f"Accuracy: **{matrix.accuracy:.3f}** over {matrix.total} classified mentions.")
        add("")
    add(_table(
        ["assertion", "precision", "recall", "f1", "gold", "tp", "fp", "fn"],
        _prf_rows(extraction["assertion_per_label"]),
    ))
    add("")

    add("### Onset offset, by narrative phrasing")
    add("")
    add(
        "Vague quantifiers (\"several days after\") are mapped to a single value "
        "by config. Exact-match accuracy on those is expected to be lower, and "
        "breaking it out here is what keeps that assumption visible."
    )
    add("")
    add(_table(
        ["phrasing", "precision", "recall", "f1", "gold", "tp", "fp", "fn"],
        _prf_rows(extraction["onset_by_phrasing"]),
    ))
    add("")

    add("### By assertion class (selected fields)")
    add("")
    interesting = ["assertion", "symptoms", "onset_offset_days", "severity"]
    rows = []
    for assertion, fields in extraction["by_assertion"].items():
        for name in interesting:
            if name in fields and fields[name]["support"]:
                d = fields[name]
                rows.append([assertion, name, d["precision"], d["recall"], d["f1"], d["support"]])
    add(_table(["assertion", "field", "precision", "recall", "f1", "gold"], rows))
    add("")

    add("### By narrative pattern")
    add("")
    rows = []
    for pattern, fields in extraction["by_pattern"].items():
        assertion = fields.get("assertion", {})
        onset = fields.get("onset_offset_days", {})
        symptoms = fields.get("symptoms", {})
        rows.append([
            pattern,
            assertion.get("f1", 0.0), assertion.get("support", 0),
            onset.get("f1", 0.0), onset.get("support", 0),
            symptoms.get("f1", 0.0), symptoms.get("support", 0),
        ])
    add(_table(
        ["pattern", "assertion F1", "n", "onset F1", "n", "symptoms F1", "n"], rows
    ))
    add("")

    # -- phenotype ----------------------------------------------------------
    phenotype = results["phenotype"]
    add("## 2. Phenotype")
    add("")
    if not phenotype.get("matches_gold_definition", True):
        add(f"> **{phenotype['comparability_note']}**")
        add("")
    pooled = phenotype["pooled"]
    add(
        f"Pooled over {phenotype['subjects']} subjects: PPV **{pooled['ppv']:.3f}**, "
        f"sensitivity **{pooled['sensitivity']:.3f}** for the `case` verdict "
        f"({pooled['tp']} true positives, {pooled['fp']} false positives, "
        f"{pooled['fn']} false negatives)."
    )
    add("")
    add("### Per study")
    add("")
    add(_table(
        ["study", "PPV", "sensitivity", "F1", "gold cases", "predicted cases", "tp", "fp", "fn"],
        [
            [study, d["ppv"], d["sensitivity"], d["f1"], d["gold_cases"],
             d["predicted_cases"], d["tp"], d["fp"], d["fn"]]
            for study, d in phenotype["per_study"].items()
        ],
    ))
    add("")
    add("### Verdict confusion")
    add("")
    verdict_matrix = phenotype.get("verdict_confusion_matrix")
    if verdict_matrix is not None:
        add(verdict_matrix.to_markdown())
        add("")
    add("### Evidence state confusion")
    add("")
    state_matrix = phenotype.get("state_confusion_matrix")
    if state_matrix is not None:
        add(state_matrix.to_markdown())
        add("")
    add(
        f"The review set holds **{phenotype['review_set_size']}** subjects. It is "
        f"reported here as a separate count and is not folded into the case "
        f"count in either direction."
    )
    add("")

    # -- retrieval ----------------------------------------------------------
    retrieval = results["retrieval"]
    add("## 3. Retrieval")
    add("")
    add(
        f"Concept `{retrieval['concept']}` expands to "
        f"{len(retrieval['expanded_terms'])} surface forms from the catalogue "
        f"(synonyms and coded terms; no hierarchy is walked)."
    )
    add("")
    add("### Assertion filter on vs off")
    add("")
    on = retrieval["assertion_filter_on"]
    off = retrieval["assertion_filter_off"]
    add(_table(
        ["", "assertion=present", "no assertion filter"],
        [
            ["records returned", on["returned"], off["returned"]],
            ["distinct documents", on["distinct_documents"], off["distinct_documents"]],
            ["records asserting absence", on["records_with_assertion_absent"],
             off["records_with_assertion_absent"]],
            ["**negation false positive rate**",
             on["negation_false_positive_rate"], off["negation_false_positive_rate"]],
            ["gold-negated documents returned",
             on["gold_negated_documents_returned"], off["gold_negated_documents_returned"]],
            ["MRR", on["mrr"], off["mrr"]],
            ["precision@10", on["precision@10"], off["precision@10"]],
            ["precision@50", on["precision@50"], off["precision@50"]],
            ["recall@50 (ceiling)",
             f"{on['recall@50']:.3f} ({on['ceiling@50']:.3f})",
             f"{off['recall@50']:.3f} ({off['ceiling@50']:.3f})"],
        ],
    ))
    add("")
    add(
        "That contrast is what a structured assertion column buys. With the "
        "filter on, no record documenting an absence of the concept can be "
        "returned, whatever the lexical scorer made of the text."
    )
    add("")
    add("### Per-study queries")
    add("")
    add(
        "Recall@k is bounded above by k divided by the number of relevant "
        "documents: with 63 relevant records, recall@5 cannot exceed 0.079 "
        "however perfect the ranking. The ceiling is shown alongside, and "
        "precision@k is the figure that is not capped."
    )
    add("")
    add(_table(
        ["study", "relevant", "returned", "MRR", "P@10", "P@50",
         "recall@50", "ceiling@50"],
        [
            [q["study"], q["relevant"], q["returned"], q["mrr"],
             q["precision@10"], q["precision@50"],
             q["recall@50"], q["ceiling@50"]]
            for q in retrieval["per_study"]
        ],
    ))
    add("")
    add(f"Mean MRR across per-study queries: **{retrieval['mean_mrr_per_study']:.3f}**.")
    if retrieval["notes"]:
        add("")
        for note in retrieval["notes"]:
            add(f"- {note}")
    add("")

    # -- stability ----------------------------------------------------------
    stability = results["stability"]
    add("## 4. Stability")
    add("")
    add(
        f"Extraction and evaluation were repeated {stability['repeats']} times on "
        f"the same corpus, config and definition."
    )
    add("")
    add(_table(
        ["check", "stable", "value"],
        [
            ["extraction output hash", stability["extraction_stable"],
             f"`{stability['extraction_hashes'][0]}`"],
            ["run id", stability["run_id_stable"], f"`{stability['run_ids'][0]}`"],
            ["results hash", stability["results_stable"],
             f"`{stability['result_hashes'][0]}`"],
        ],
    ))
    add("")

    # -- sensitivity --------------------------------------------------------
    add("## 5. Definition sensitivity")
    add("")
    add(
        "Each sweep re-runs case assignment with one rule parameter changed and "
        "nothing else. The variants are derived in memory from the frozen "
        "definition; a sweep explores what a change would do, it does not "
        "publish a new version."
    )
    add("")
    for sweep in results["sensitivity"]["sweeps"]:
        add(f"### {sweep['parameter']}")
        add("")
        if sweep["note"]:
            add(f"{sweep['note']}")
            add("")
        add(_table(
            ["value", "case", "review", "excluded"],
            [[r["value"], r["case"], r["review"], r["excluded"]] for r in sweep["rows"]],
        ))
        low, high = sweep["case_count_range"]
        add("")
        add(
            f"Case count moves between **{low}** and **{high}** across this "
            f"sweep — a {high - low}-subject swing driven entirely by a "
            f"configuration value."
        )
        add("")

    add("---")
    add("")
    add(
        "Every figure above is reproducible from a clean checkout with "
        "`make eval`. The run id and results hash are content-derived, so an "
        "identical corpus, config and definition always produce the same "
        "numbers."
    )
    add("")
    return "\n".join(out)
