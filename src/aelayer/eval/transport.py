"""Transportability: hold out whole studies, never rows.

The deployment failure mode for this layer is **protocol shift**, not unseen
wording. A random row split leaks every profile's collection conventions into
both sides and produces a number that says nothing about what happens when the
layer meets a study whose CRF was designed differently. Row splits are
therefore disallowed here, not merely discouraged.

So the holdout is by study, and the report says which studies were held out and
what the drop was. Nothing in this pipeline is fitted to data, so the gap
measures how much harder the held-out conventions are, not overfitting — and
that is stated in the output rather than left for a reader to assume either way.
"""

from __future__ import annotations

import collections
from typing import Any, Sequence

from ..models import VERDICTS, PhenotypeDefinition
from ..pipeline import Pipeline
from .metrics import ConfusionMatrix

#: The profiles the extraction lexicon and cue lists were written against.
#: Named here rather than inferred, so a reader can check the claim against the
#: config history instead of taking it on trust.
DEVELOPMENT_PROFILES: tuple[str, ...] = ("P_structured", "P_text", "P_both")

NOTE = (
    "Whole studies are held out, never rows. A random row split would leak "
    "every profile's collection conventions into both sides and measure "
    "nothing about protocol shift, which is the way this layer actually fails "
    "when it meets a new study."
)

NOT_FITTED = (
    "Nothing in this pipeline is fitted to data, so the gap between "
    "development and held-out studies measures how much harder the held-out "
    "collection conventions are, not overfitting to the development set."
)

HOLDOUT_CHARACTER = (
    "The held-out profiles differ from the development ones in kind, not "
    "degree: one documents negatives in prose, one codes under a superseded "
    "dictionary version, one keeps the modifier only in a linked comment, and "
    "one does not collect it at all. That last study cannot be scored above "
    "chance by any extractor, and its not-ascertainable rate is the honest "
    "result rather than a failure to report."
)


def case_metrics(matrix: ConfusionMatrix) -> dict[str, Any]:
    """Case-level PPV and sensitivity, with the unascertainable rate beside them.

    A record nobody can evaluate is neither a hit nor a miss. Folding it into
    either would overstate both the numerator and what the denominator means,
    so it is reported as its own rate.
    """
    per_label = matrix.per_label()
    case = per_label.get("case")
    predicted_unascertainable = sum(
        count for (_gold, predicted), count in matrix.counts.items()
        if predicted == "not_ascertainable"
    )
    return {
        "n": matrix.total,
        "ppv": round(case.precision, 4) if case else 0.0,
        "sensitivity": round(case.recall, 4) if case else 0.0,
        "f1": round(case.f1, 4) if case else 0.0,
        "accuracy": round(matrix.accuracy, 4),
        "not_ascertainable_rate": (
            round(predicted_unascertainable / matrix.total, 4)
            if matrix.total else 0.0
        ),
        "counts": case.to_dict() if case else {},
    }


def transportability(
    pipeline: Pipeline,
    definition: PhenotypeDefinition,
    holdout: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evaluate on development studies and on held-out studies separately."""
    profiles = set(pipeline.configs.profiles.profile_ids())
    held_out = set(holdout) if holdout else profiles - set(DEVELOPMENT_PROFILES)
    unknown = sorted(held_out - profiles)
    if unknown:
        raise ValueError(
            f"cannot hold out {unknown}: no such profile. Known profiles: "
            f"{sorted(profiles)}"
        )
    development = profiles - held_out
    if not development or not held_out:
        raise ValueError(
            "a transportability split needs at least one profile on each side; "
            f"got development={sorted(development)}, held_out={sorted(held_out)}"
        )

    assignments = pipeline.assignments(definition)
    gold = pipeline.store.gold_by_record()

    matrices = {
        "development": ConfusionMatrix(labels=list(VERDICTS)),
        "held_out": ConfusionMatrix(labels=list(VERDICTS)),
    }
    per_profile: dict[str, ConfusionMatrix] = {}
    for assignment in assignments:
        truth = gold.get(assignment.record_id.split(":", 1)[-1])
        if truth is None:
            continue
        side = "held_out" if assignment.profile in held_out else "development"
        matrices[side].add(truth["true_verdict"], assignment.verdict)
        per_profile.setdefault(
            assignment.profile, ConfusionMatrix(labels=list(VERDICTS))
        ).add(truth["true_verdict"], assignment.verdict)

    development_result = case_metrics(matrices["development"])
    held_out_result = case_metrics(matrices["held_out"])
    return {
        "note": NOTE,
        "not_fitted": NOT_FITTED,
        "holdout_character": HOLDOUT_CHARACTER,
        "split": "whole_study",
        "row_splits": "disallowed",
        "development_profiles": sorted(development),
        "held_out_profiles": sorted(held_out),
        "development": development_result,
        "held_out": held_out_result,
        "sensitivity_drop": round(
            development_result["sensitivity"] - held_out_result["sensitivity"], 4
        ),
        "ppv_drop": round(
            development_result["ppv"] - held_out_result["ppv"], 4
        ),
        "not_ascertainable_rate_change": round(
            held_out_result["not_ascertainable_rate"]
            - development_result["not_ascertainable_rate"], 4
        ),
        "per_profile": {p: case_metrics(m) for p, m in sorted(per_profile.items())},
        "verdicts_by_profile": {
            profile: dict(sorted(collections.Counter(
                a.verdict for a in assignments if a.profile == profile
            ).items()))
            for profile in sorted(profiles)
        },
    }
