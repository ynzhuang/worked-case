"""Rendering the evaluation as markdown.

Order matters: the disclaimer comes before any number, the silver standard is
labelled as silver everywhere it appears, and the not-ascertainable rate is a
column of its own rather than something folded into a denominator.
"""

from __future__ import annotations

from typing import Any

from ..silver import SILVER_CAVEAT
from .harness import ABLATION_NOTE, DISCLAIMER, INVARIANCE_CAVEAT


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value) or "—"
    if isinstance(value, dict):
        return ", ".join(f"{k}={v}" for k, v in value.items()) or "—"
    return "—" if value is None else str(value)


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_(nothing to report)_"
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "---|" * len(headers),
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(cell) for cell in row) + " |")
    return "\n".join(lines)


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
        ["definition", f"`{definition['id']}` v{definition['version']} "
                       f"({definition['status']})"],
        ["definition hash", f"`{definition['hash']}`"],
        ["normalizer", f"`{versions['normalizer_version']}`"],
        ["extractor", f"`{versions['extractor_version']}`"],
        ["extraction backend", f"`{versions['extraction_backend']}`"],
        ["data snapshot", f"`{versions['snapshot_id']}`"],
        ["studies / subjects", f"{corpus['studies']} / {corpus['subjects']}"],
        ["source records / episodes", f"{corpus['records']} / {corpus['episodes']}"],
    ]))
    add("")

    # ------------------------------------------------------------- silver
    silver = results["silver"]
    overall = silver["overall"]
    add("## Silver standard — extraction against the study's own structured field")
    add("")
    add(f"> {SILVER_CAVEAT}")
    add("")
    add(
        f"Attribute: **{silver['attribute']}**. Eligible profiles: "
        f"{', '.join(silver['profiles'])} — the ones that record the attribute "
        f"in a structured variable *and* in the investigator's own words. The "
        f"structured value is masked from the extractor and used only as the "
        f"comparator."
    )
    add("")
    add(_table(
        ["precision", "recall", "f1", "coverage", "abstention rate",
         "normalized agreement", "eligible", "answered"],
        [[overall["precision"], overall["recall"], overall["f1"],
          overall["coverage"], overall["abstention_rate"],
          overall["normalized_agreement"], overall["eligible_records"],
          overall["answered"]]],
    ))
    add("")
    add(
        "Coverage and abstention are reported beside precision on purpose: a "
        "precision reached by answering three times in a hundred is not a "
        "useful extractor, and the pair is what makes that visible."
    )
    add("")
    add("### By reported-term style")
    add("")
    add(_table(
        ["style", "precision", "recall", "coverage", "abstention", "answered"],
        [[style, body["precision"], body["recall"], body["coverage"],
          body["abstention_rate"], body["answered"]]
         for style, body in silver["by_reported_term_style"].items()],
    ))
    add("")

    # ---------------------------------------------------------- phenotype
    phenotype = results["phenotype"]
    pooled = phenotype["pooled"]
    add("## Phenotype")
    add("")
    add(
        f"`{phenotype['evaluated_definition']}` over {phenotype['episodes']} "
        f"episodes, {phenotype['matched_to_gold']} of them matched to a gold "
        f"label."
    )
    add("")
    add(_table(
        ["PPV", "sensitivity", "F1", "gold cases", "predicted cases",
         "not-ascertainable rate", "not-ascertainable agreement"],
        [[pooled["ppv"], pooled["sensitivity"], pooled["f1"],
          pooled["gold_cases"], pooled["predicted_cases"],
          pooled["not_ascertainable_rate"], pooled["not_ascertainable_agreement"]]],
    ))
    add("")
    add(
        "**not_ascertainable is not a negative.** It counts episodes where a "
        "required attribute was never recorded and cannot be recovered — no "
        "reviewer can settle them either. Reporting them inside the negatives "
        "would understate the cohort's uncertainty; reporting them as review "
        "items would send work to a human who has nothing to work with."
    )
    add("")
    add("### By profile")
    add("")
    add(_table(
        ["profile", "episodes", "PPV", "sensitivity", "cases",
         "not-ascertainable", "rate"],
        [[profile, body["episodes"], body["ppv"], body["sensitivity"],
          body["predicted_cases"], body["not_ascertainable_predicted"],
          body["not_ascertainable_rate"]]
         for profile, body in phenotype["per_profile"].items()],
    ))
    add("")
    add("### Where the evidence came from")
    add("")
    add(
        "Counted over cases only. A cohort that depended on text extraction "
        "says so here, rather than leaving a later reader to work it out."
    )
    add("")
    add(_table(
        ["route", "attributes satisfied"],
        [[method, count] for method, count in phenotype["attribute_methods"].items()],
    ))
    add("")
    add(_table(
        ["source variable", "attributes satisfied"],
        [[variable, count] for variable, count in phenotype["attribute_sources"].items()],
    ))
    add("")

    # ----------------------------------------------------------- ablation
    ablation = results["ablation"]
    add("## Value ablation — what text recovery is worth")
    add("")
    add(f"{ABLATION_NOTE}")
    add("")
    add(_table(
        ["structured only", "with text", "only findable through text",
         "fraction", "not-ascertainable resolved by text"],
        [[ablation["cases_structured_only"], ablation["cases_with_text"],
          ablation["cases_only_findable_through_text"],
          ablation["fraction_only_findable_through_text"],
          ablation["not_ascertainable_resolved_by_text"]]],
    ))
    add("")
    add(
        f"**{ablation['cases_only_findable_through_text']} of "
        f"{ablation['cases_with_text']} qualifying events "
        f"({ablation['fraction_only_findable_through_text']:.1%}) are findable "
        f"only through text.** Without the extraction layer they are not "
        f"negatives — they are unascertainable, and "
        f"{ablation['not_ascertainable_resolved_by_text']} of them move out of "
        f"that bucket when text is read."
    )
    add("")
    add(_table(
        ["profile", "cases found only through text"],
        [[profile, count] for profile, count in ablation["by_profile"].items()],
    ))
    add("")

    # ------------------------------------------------------- availability
    availability = results["availability"]
    add("## Availability — does the system tell the kinds of missing apart?")
    add("")
    add(
        "Confusing \"no location recorded\" with \"no location\" is the failure "
        "that quietly biases every downstream estimate, so it is scored "
        "directly rather than left to a single accuracy number."
    )
    add("")
    matrix = availability.get("matrix")
    if matrix is not None:
        add(matrix.to_markdown())
        add("")
    add(_table(
        ["accuracy", "collected read as missing", "missing read as collected",
         "not-collected read as unknown"],
        [[availability["accuracy"], availability["collected_read_as_missing"],
          availability["missing_read_as_collected"],
          availability["not_collected_read_as_unknown"]]],
    ))
    add("")

    # ---------------------------------------------------------- transport
    transport = results["transport"]
    add("## Transportability — whole studies held out")
    add("")
    add(f"{transport['note']}")
    add("")
    add(_table(
        ["side", "profiles", "episodes", "PPV", "sensitivity",
         "not-ascertainable rate"],
        [
            ["development", transport["development_profiles"],
             transport["development"]["episodes"], transport["development"]["ppv"],
             transport["development"]["sensitivity"],
             transport["development"]["not_ascertainable_rate"]],
            ["held out", transport["held_out_profiles"],
             transport["held_out"]["episodes"], transport["held_out"]["ppv"],
             transport["held_out"]["sensitivity"],
             transport["held_out"]["not_ascertainable_rate"]],
        ],
    ))
    add("")
    add(
        f"Sensitivity drop **{transport['sensitivity_drop']:+.3f}**, PPV drop "
        f"**{transport['ppv_drop']:+.3f}**, change in the not-ascertainable "
        f"rate **{transport['not_ascertainable_rate_change']:+.3f}**. "
        f"{transport['not_fitted']}"
    )
    add("")

    # --------------------------------------------------------- invariance
    invariance = results["invariance"]
    add("## Representation invariance")
    add("")
    add(f"> {INVARIANCE_CAVEAT}")
    add("")
    add(
        f"{invariance['truths_compared']} clinical truths rendered under more "
        f"than one profile. Raw agreement across every rendering is "
        f"**{invariance['raw_agreement']:.3f}**; agreement across the renderings "
        f"that could actually record the attribute is "
        f"**{invariance['agreement_where_evidence_supports_it']:.3f}** over "
        f"{invariance['truths_with_supporting_evidence']} truths."
    )
    add("")
    add(
        "The two numbers are different questions. Raw agreement is low by "
        "construction: a study that never collected the location returns "
        "not_ascertainable, and it is *right* to. Counting that as a "
        "disagreement would be scoring the system for the study's collection "
        "decision."
    )
    add("")
    add(_table(
        ["profile", "verdicts over the shared truths"],
        [[profile, body]
         for profile, body in invariance["verdicts_by_profile"].items()],
    ))
    add("")

    # ----------------------------------------------------- reproducibility
    repro = results["reproducibility"]
    add("## Reproducibility")
    add("")
    add(_table(
        ["manifest id stable", "results stable", "normalization stable", "repeats"],
        [[repro["manifest_id_stable"], repro["results_stable"],
          repro["normalization_stable"], repro["repeats"]]],
    ))
    add("")
    return "\n".join(out) + "\n"
