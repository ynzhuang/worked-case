"""Rendering the evaluation as markdown.

Order matters. The disclaimer comes before any number. The value ablation's
decision and the silver standard's two caveats appear near the top, because
they are what the whole prototype exists to produce and burying them under
tables would misrepresent the result. The not-ascertainable rate and the
ascertainable fraction get columns of their own rather than being folded into
a denominator.
"""

from __future__ import annotations

from typing import Any

from ..ablation import DECISION_NOTE
from ..models import DENOMINATOR_NOTE
from ..silver import SILVER_CAVEATS
from .harness import DISCLAIMER, INVARIANCE_CAVEAT, SILENCE_CAVEAT


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}"
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
        ["dictionary target", f"`{versions['dictionary_target']}`"],
        ["data snapshot", f"`{versions['snapshot_id']}`"],
        ["records", corpus["records"]],
        ["studies", corpus["studies"]],
        ["subjects", corpus["subjects"]],
    ]))
    add("")

    # ---------------------------------------------------------------- decision
    ablation = results["ablation"]
    add("## The decision")
    add("")
    add(f"**{ablation['decision']}**")
    add("")
    add(_table(
        ["stage", "evaluated", "ascertained", "asc. fraction", "cases",
         "correct", "wrong", "precision", "recall"],
        [[
            s["stage"], s["n_evaluated"], s["n_ascertained"],
            s["ascertainable_fraction"], s["n_case"], s["n_case_correct"],
            s["n_case_incorrect"], s["precision"], s["recall"],
        ] for s in ablation["stages"]],
    ))
    add("")
    for increment in ablation["increments"]:
        add(f"**{increment['from_stage']} → {increment['to_stage']}**")
        add("")
        for reason in increment["reasons"]:
            add(f"- {reason}")
        add("")
        add(f"> {increment['decision']}")
        add("")
    add(f"_{DECISION_NOTE}_")
    add("")
    add("Materiality thresholds were declared before the numbers were seen:")
    add("")
    add(_table(["criterion", "value"], [
        [k, v] for k, v in sorted(ablation["materiality_criteria"].items())
    ]))
    add("")

    # ------------------------------------------------------------------ silver
    silver = results["silver"]
    add("## Silver standard")
    add("")
    for caveat in results.get("silver_caveats", SILVER_CAVEATS):
        add(f"> {caveat}")
        add("")
    add(_table(["", "value"], [
        [k, v] for k, v in silver["overall"].items()
    ]))
    add("")
    add("### By assertion class")
    add("")
    add(_table(
        ["assertion", "n", "answered", "correct", "recall", "precision"],
        [[k, v["n"], v["answered"], v["correct"], v["recall"], v["precision"]]
         for k, v in silver["by_assertion"].items()],
    ))
    add("")
    add("### Calibration")
    add("")
    calibration = silver["calibration"]
    add(_table(["", "value"], [
        ["Brier score", calibration["brier_score"]],
        ["expected calibration error", calibration["expected_calibration_error"]],
    ]))
    add("")
    add(_table(
        ["confidence bin", "n", "mean confidence", "observed accuracy", "gap"],
        [[r["bin"], r["n"], r["mean_confidence"], r["observed_accuracy"], r["gap"]]
         for r in calibration["reliability"]],
    ))
    add("")
    add(f"_{calibration['note']}_")
    add("")
    add(f"Adjudication queue: **{silver.get('adjudication_queue_size', 0)}** rows, "
        f"including a random sample of agreements so the comparator's own error "
        f"rate can be estimated rather than assumed.")
    add("")

    # -------------------------------------------------------------- phenotype
    phenotype = results["phenotype"]
    add("## Phenotype")
    add("")
    add(_table(["", "value"], [
        [k, v] for k, v in phenotype["pooled"].items() if k != "counts"
    ]))
    add("")
    add("### Verdicts (gold ↓ / assigned →)")
    add("")
    add(phenotype["verdict_matrix"].to_markdown())
    add("")
    add("### Per study")
    add("")
    add(_table(
        ["study", "records", "PPV", "sensitivity", "not-ascertainable rate"],
        [[study, m["records"], m["ppv"], m["sensitivity"],
          m["not_ascertainable_rate"]]
         for study, m in phenotype["per_study"].items()],
    ))
    add("")
    add(f"Cases whose evidence came out of prose: "
        f"**{phenotype['cases_depending_on_text']}**. "
        f"Routes behind the cases: {_fmt(phenotype['attribute_methods'])}.")
    add("")

    # ------------------------------------------------------------ denominators
    denominators = results["denominators"]
    add("## Denominators")
    add("")
    add(f"> {DENOMINATOR_NOTE}")
    add("")
    add(_table(
        ["study", "profile", "total", "case", "non_case", "review",
         "not ascertainable", "ascertainable fraction", "incidence"],
        [[d["study_id"], d["profile"], d["n_total"], d["n_case"],
          d["n_non_case"], d["n_review"], d["n_not_ascertainable"],
          d["ascertainable_fraction"], d["incidence_within_ascertainable"]]
         for d in [*denominators["per_study"], denominators["overall"]]],
    ))
    add("")

    # ------------------------------------------- silence vs documented negative
    assertion = results["assertion"]
    add("## A documented negative is not silence")
    add("")
    add(f"> {assertion['caveat']}")
    add("")
    add(assertion["assertion_matrix"].to_markdown("said ↓ / read as →"))
    add("")
    add(_table(["", "value"], [
        ["accuracy", assertion["accuracy"]],
        ["documented negatives recovered",
         assertion["documented_negatives_recovered"]],
        ["documented negatives read as something else",
         assertion["documented_negatives_read_as_something_else"]],
        ["silence read as an assertion",
         assertion["silence_read_as_an_assertion"]],
    ]))
    add("")

    # --------------------------------------------------------------- transport
    transport = results["transport"]
    add("## Transportability")
    add("")
    add(f"> {transport['note']}")
    add("")
    add(f"> {transport['holdout_character']}")
    add("")
    add(_table(["", "development", "held out"], [
        ["profiles", transport["development_profiles"],
         transport["held_out_profiles"]],
        ["records", transport["development"]["n"], transport["held_out"]["n"]],
        ["PPV", transport["development"]["ppv"], transport["held_out"]["ppv"]],
        ["sensitivity", transport["development"]["sensitivity"],
         transport["held_out"]["sensitivity"]],
        ["not-ascertainable rate",
         transport["development"]["not_ascertainable_rate"],
         transport["held_out"]["not_ascertainable_rate"]],
    ]))
    add("")
    add(f"Sensitivity drop: **{_fmt(transport['sensitivity_drop'])}**; "
        f"PPV drop: **{_fmt(transport['ppv_drop'])}**.")
    add("")
    add(f"_{transport['not_fitted']}_")
    add("")

    # --------------------------------------------------------------- invariance
    invariance = results["invariance"]
    add("## Representation invariance")
    add("")
    add(f"> {INVARIANCE_CAVEAT}")
    add("")
    add(_table(["", "value"], [
        ["truths compared", invariance["truths_compared"]],
        ["raw agreement", invariance["raw_agreement"]],
        ["agreement where evidence supports it",
         invariance["agreement_where_evidence_supports_it"]],
        ["discordant", invariance["discordant_count"]],
    ]))
    add("")

    # ----------------------------------------------------------- reproducibility
    reproducibility = results["reproducibility"]
    add("## Reproducibility")
    add("")
    add(_table(["", "value"], [
        ["repeats", reproducibility["repeats"]],
        ["manifest id stable", reproducibility["manifest_id_stable"]],
        ["results stable", reproducibility["results_stable"]],
        ["normalization stable", reproducibility["normalization_stable"]],
        ["results hash", f"`{reproducibility['result_hashes'][0]}`"],
    ]))
    add("")
    return "\n".join(out) + "\n"
