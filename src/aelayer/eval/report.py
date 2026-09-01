"""Render evaluation results as markdown.

Separate from the harness so that adding a table can never change a number.
"""

from __future__ import annotations

from typing import Any

from .harness import DISCLAIMER, INVARIANCE_CAVEAT


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}"
    if value is None:
        return "-"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value) or "-"
    return str(value)


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(c) for c in row) + " |")
    return "\n".join(lines)


def _prf_rows(section: dict[str, dict[str, Any]]) -> list[list[Any]]:
    return [
        [name, d["precision"], d["recall"], d["f1"], d["support"],
         d["tp"], d["fp"], d["fn"]]
        for name, d in section.items()
        if d["support"] or d.get("predicted")
    ]


def render_markdown(results: dict[str, Any]) -> str:
    out: list[str] = []
    add = out.append

    definition = results["definition"]
    versions = results["versions"]
    corpus = results["corpus"]

    add("# Adverse event evidence layer — evaluation report")
    add("")
    add(f"> {DISCLAIMER}")
    add("")
    add(_table(["", "value"], [
        ["generated", results["generated_at"]],
        ["definition", f"`{definition['id']}` v{definition['version']} ({definition['status']})"],
        ["definition hash", f"`{definition['hash']}`"],
        ["normalizer", f"`{versions['normalizer_version']}`"],
        ["extractor", f"`{versions['extractor_version']}`"],
        ["extraction backend", f"`{versions['extraction_backend']}`"],
        ["data snapshot", f"`{versions['snapshot_id']}`"],
        ["studies / subjects", f"{corpus['studies']} / {corpus['subjects']}"],
        ["source records / episodes", f"{corpus['records']} / {corpus['episodes']}"],
    ]))
    add("")

    # ---------------------------------------------------------------- L1
    layer1 = results["layer1"]
    add("## Layer 1 — clinical validity")
    add("")
    add(
        "Does the pipeline recover what the source says? Field-level agreement "
        "against the gold record values, over "
        f"{layer1['records']} source records."
    )
    add("")
    violations = layer1["provenance_violations"]
    if violations:
        add(f"**{len(violations)} record(s) carry a populated field with no span.** "
            "Each is a defect:")
        for entry in violations[:10]:
            add(f"- `{entry}`")
    else:
        add("Every populated field on every record traces to at least one span. "
            "No provenance violations.")
    add("")
    add("### Per field")
    add("")
    add(_table(["field", "precision", "recall", "f1", "gold", "tp", "fp", "fn"],
               _prf_rows(layer1["overall"])))
    add("")

    add("### By source path")
    add("")
    add(
        "`structured` is the deterministic path; `text` is the model path. The "
        "model path is asked only about fields the deterministic path left "
        "unresolved, so these are different populations, not a head-to-head."
    )
    add("")
    rows = []
    for path, fields in layer1["by_source_path"].items():
        for name, body in fields.items():
            if body["support"]:
                rows.append([path, name, body["precision"], body["recall"],
                             body["f1"], body["support"]])
    add(_table(["path", "field", "precision", "recall", "f1", "gold"], rows))
    add("")

    add("### Collection-state classification")
    add("")
    add("A blank is not a value. This matrix is whether the pipeline reads each "
        "blank for what it is.")
    add("")
    matrix = layer1.get("collection_state_matrix")
    if matrix is not None:
        add(matrix.to_markdown())
        add("")
        add(f"Accuracy: **{matrix.accuracy:.3f}** over {matrix.total} field readings.")
    add("")

    add("### Abstention quality")
    add("")
    abstention = layer1["abstention"]
    add(
        "Scored only where the model path was asked. Abstaining when the text "
        "does not support a value is correct behaviour; guessing is a defect."
    )
    add("")
    add(_table(["outcome", "count"], [
        ["correctly abstained", abstention["correct_abstention"]],
        ["wrongly abstained (value was recoverable)", abstention["wrong_abstention"]],
        ["answered correctly", abstention["correct_answer"]],
        ["answered wrongly", abstention["wrong_answer"]],
        ["**abstention precision**", abstention["abstention_precision"]],
        ["**answer precision**", abstention["answer_precision"]],
    ]))
    add("")

    # ---------------------------------------------------------------- L2
    layer2 = results["layer2"]
    add("## Layer 2 — episode reconciliation")
    add("")
    add(
        f"{layer2['gold_episodes']} true episodes; {layer2['derived_episodes']} "
        f"derived. Boundary agreement **{layer2['boundary_agreement']:.3f}**."
    )
    add("")
    add(_table(["", "count", "rate"], [
        ["exact boundary matches", layer2["exact_boundary_matches"],
         layer2["boundary_agreement"]],
        ["over-merged episodes", layer2["over_merge"], layer2["over_merge_rate"]],
        ["over-split episodes", layer2["over_split"], layer2["over_split_rate"]],
        ["flagged for linkage review", layer2["flagged_for_review"], "-"],
    ]))
    add("")
    add("### Split behaviour by recurrence expectation")
    add("")
    add(
        "The default linkage rule is wrong for recurrent conditions, which is "
        "why the catalogue declares `recurrence_expected` per concept. These "
        "two rows are the reason it is declared rather than assumed."
    )
    add("")
    add(_table(["concepts", "episodes", "over-split", "rate"], [
        ["recurrence expected", layer2["recurrence_expected"]["episodes"],
         layer2["recurrence_expected"]["over_split"],
         layer2["recurrence_expected"]["over_split_rate"]],
        ["recurrence not expected", layer2["recurrence_not_expected"]["episodes"],
         layer2["recurrence_not_expected"]["over_split"],
         layer2["recurrence_not_expected"]["over_split_rate"]],
    ]))
    add("")
    add(_table(["linkage rule", "episodes"],
               [[k, v] for k, v in layer2["linkage_rules"].items()]))
    if layer2["mismatches_by_representation"]:
        add("")
        add("Boundary mismatches by representation: "
            + ", ".join(f"{k} ({v})" for k, v in
                        layer2["mismatches_by_representation"].items()))
    add("")

    # ---------------------------------------------------------------- L3
    layer3 = results["layer3"]
    pooled = layer3["pooled"]
    add("## Layer 3 — phenotype")
    add("")
    add(
        f"Over {layer3['episodes']} episodes under `{layer3['evaluated_definition']}`: "
        f"PPV **{pooled['ppv']:.3f}**, sensitivity **{pooled['sensitivity']:.3f}** "
        f"for the `case` verdict ({pooled['tp']} true positives, {pooled['fp']} "
        f"false positives, {pooled['fn']} false negatives)."
    )
    add("")
    add(_table(["", "value"], [
        ["false negatives routed to review by linkage uncertainty",
         pooled["false_negatives_from_linkage_review"]],
        ["false negatives from anything else", pooled["false_negatives_other"]],
        ["sensitivity excluding the declined",
         pooled["sensitivity_excluding_declined"]],
    ]))
    add("")
    add(
        "A case the system routed to review because the episode boundary was a "
        "judgement call is a different kind of miss from one it got wrong. The "
        "first is the system declining to assert something it cannot settle, "
        "which is the behaviour the definition asks for; only the second is an "
        "accuracy problem."
    )
    add("")
    add("### Per study")
    add("")
    add(_table(["study", "PPV", "sensitivity", "F1", "gold cases", "predicted", "tp", "fp", "fn"],
               [[s, d["ppv"], d["sensitivity"], d["f1"], d["gold_cases"],
                 d["predicted_cases"], d["tp"], d["fp"], d["fn"]]
                for s, d in layer3["per_study"].items()]))
    add("")
    add("### Verdict confusion")
    add("")
    verdict_matrix = layer3.get("verdict_matrix")
    if verdict_matrix is not None:
        add(verdict_matrix.to_markdown())
    add("")
    add(f"The review set holds **{layer3['review_set_size']}** episodes and is "
        f"reported as its own count, never folded into the case count.")
    add("")
    add("### Cross-study transportability")
    add("")
    transport = layer3["transportability"]
    add(f"{transport['note']}")
    add("")
    add(_table(["set", "studies", "PPV", "sensitivity", "gold cases"], [
        ["development", transport["development_studies"],
         transport["development"]["ppv"], transport["development"]["sensitivity"],
         transport["development"]["gold_cases"]],
        ["held out", transport["held_out_studies"],
         transport["held_out"]["ppv"], transport["held_out"]["sensitivity"],
         transport["held_out"]["gold_cases"]],
        ["**drop**", "-", transport["ppv_drop"], transport["sensitivity_drop"], "-"],
    ]))
    add("")

    # ------------------------------------------------------- invariance
    invariance = results["invariance"]
    add("## Stress test — representation invariance")
    add("")
    add(f"> **{INVARIANCE_CAVEAT}**")
    add("")
    add(
        f"{invariance['truths_compared']} sampled truths, each rendered under "
        f"{len(invariance['representations'])} collection conventions "
        f"({', '.join(invariance['representations'])})."
    )
    add("")
    add(_table(["", "rate"], [
        ["verdict agreement across representations", invariance["verdict_agreement"]],
        ["evidence-state agreement across representations",
         invariance["state_agreement"]],
        ["discordant truths", invariance["discordant_count"]],
    ]))
    add("")
    if invariance["discordance_by_representation"]:
        add("Which representation departs from the majority:")
        add("")
        add(_table(["representation", "departures"],
                   [[k, sum(v.values())] for k, v in
                    invariance["discordance_by_representation"].items()]))
        add("")
    if invariance["discordant"]:
        add("Examples:")
        add("")
        rows = []
        for entry in invariance["discordant"][:8]:
            verdicts = entry["verdicts"]
            odd = [r for r, v in verdicts.items() if v != entry["majority"]]
            rows.append([entry["truth_id"], entry["reference_verdict"],
                         entry["majority"], ", ".join(odd),
                         ", ".join(f"{r}={verdicts[r]}" for r in odd)])
        add(_table(["truth", "reference", "majority", "departs", "verdict"], rows))
        add("")
    add(
        "A lower state agreement than verdict agreement is expected and is not "
        "a fault: a study that codes the event reaches `explicit` while a study "
        "that leaves it to narrative reaches `supported`, by different rules, "
        "on the same patient. The verdict is what a cohort is built from."
    )
    add("")

    # ---------------------------------------------------------- retrieval
    retrieval = results.get("retrieval") or {}
    if retrieval.get("available"):
        add("## Retrieval")
        add("")
        precise = retrieval["precise"]
        add("### Precise cohort path")
        add("")
        add(_table(["", "value"], [
            ["cohort size under the definition", precise["cohort_size"]],
            ["returned by the precise path", precise["returned"]],
            ["exact match", precise["exact_match"]],
            ["candidates returned", precise["candidates_returned"]],
            ["usable as a cohort", precise["usable_as_cohort"]],
        ]))
        add("")
        add("### Discovery path — assertion filter on vs off")
        add("")
        add(
            "This is where assertion matters. A coded AE row asserts presence by "
            "construction; a narrative can name a concept in order to rule it "
            "out, and a discovery search that cannot tell them apart returns "
            "documented absences as though they were events."
        )
        add("")
        on = retrieval["assertion_filter_on"]
        off = retrieval["assertion_filter_off"]
        add(_table(["", "assertion=present", "no assertion filter"], [
            ["mentions returned", on["returned"], off["returned"]],
            ["mentions asserting absence", on["mentions_asserting_absence"],
             off["mentions_asserting_absence"]],
            ["**negation false positive rate**",
             on["negation_false_positive_rate"], off["negation_false_positive_rate"]],
            ["MRR", on["mrr"], off["mrr"]],
            ["precision@10", on["precision@10"], off["precision@10"]],
            ["recall@50 (ceiling)",
             f"{on['recall@50']:.3f} ({on['ceiling@50']:.3f})",
             f"{off['recall@50']:.3f} ({off['ceiling@50']:.3f})"],
        ]))
        add("")
        add(
            "Recall@k is bounded by k divided by the number of relevant "
            "mentions, so the ceiling is shown alongside; precision@k is the "
            "figure that is not capped."
        )
        add("")
        add(_table(["study", "relevant", "returned", "MRR", "P@10", "recall@50", "ceiling@50"],
                   [[q["study"], q["relevant"], q["returned"], q["mrr"],
                     q["precision@10"], q["recall@50"], q["ceiling@50"]]
                    for q in retrieval["per_study"]]))
        add("")

    # ----------------------------------------------------- reproducibility
    repro = results["reproducibility"]
    add("## Reproducibility")
    add("")
    add(_table(["check", "stable", "value"], [
        ["normalization output hash", repro["normalization_stable"],
         f"`{repro['record_hashes'][0]}`"],
        ["manifest id", repro["manifest_id_stable"], f"`{repro['manifest_ids'][0]}`"],
        ["results hash", repro["results_stable"], f"`{repro['result_hashes'][0]}`"],
    ]))
    add("")
    add("---")
    add("")
    add(
        "Every figure above is reproducible from a clean checkout with "
        "`make eval`. Manifest ids and results hashes are content-derived, so "
        "identical inputs always produce identical numbers."
    )
    add("")
    return "\n".join(out)
