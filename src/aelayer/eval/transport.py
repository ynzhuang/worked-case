"""Transportability: hold out whole studies, never rows.

The deployment failure mode for this layer is **protocol shift**, not unseen
wording. A random row split leaks every profile's conventions into both sides
and produces a number that says nothing about what happens when the layer meets
a study whose CRF was designed differently.

So the holdout is by study, and the report says which studies were held out and
what the drop was. Nothing here is fitted to data, so the gap measures how much
harder the held-out conventions are, not overfitting — and that is stated in the
output rather than left for a reader to assume either way.
"""

from __future__ import annotations

import collections
from typing import Any, Sequence

from ..models import VERDICTS, PhenotypeDefinition
from ..pipeline import Pipeline
from .metrics import ConfusionMatrix

#: The profiles whose conventions the extraction lexicon and scope rules were
#: written against. Named here rather than inferred, so a reader can check the
#: claim against the config history.
DEVELOPMENT_PROFILES: tuple[str, ...] = ("P1_structured", "P2_text", "P3_prespecified")

NOTE = (
    "Whole studies are held out, never rows. A random row split would leak "
    "every profile's collection conventions into both sides and measure "
    "nothing about protocol shift, which is the way this layer actually fails "
    "when it meets a new study."
)

NOT_FITTED = (
    "Nothing in this pipeline is fitted to data, so the gap between development "
    "and held-out studies measures how much harder the held-out collection "
    "conventions are, not overfitting to the development set."
)


def transportability(
    pipeline: Pipeline,
    definition: PhenotypeDefinition,
    holdout: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evaluate on development profiles and on held-out profiles separately."""
    from .harness import _case_metrics

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

    episodes = pipeline.episodes()
    assignments = pipeline.evaluate(definition, episodes=episodes)

    gold_by_record: dict[str, dict[str, Any]] = {}
    for entry in pipeline.store.gold_episodes():
        for record_id in entry["source_record_ids"]:
            gold_by_record[record_id] = entry

    matrices = {
        "development": ConfusionMatrix(labels=list(VERDICTS)),
        "held_out": ConfusionMatrix(labels=list(VERDICTS)),
    }
    per_profile: dict[str, ConfusionMatrix] = {}
    for assignment in assignments:
        truth = next(
            (gold_by_record[r] for r in assignment.source_record_ids
             if r in gold_by_record),
            None,
        )
        if truth is None:
            continue
        side = "held_out" if assignment.profile in held_out else "development"
        matrices[side].add(truth["true_verdict"], assignment.verdict)
        per_profile.setdefault(
            assignment.profile, ConfusionMatrix(labels=list(VERDICTS))
        ).add(truth["true_verdict"], assignment.verdict)

    development_result = _case_metrics(matrices["development"])
    held_out_result = _case_metrics(matrices["held_out"])
    return {
        "note": NOTE,
        "not_fitted": NOT_FITTED,
        "split": "whole_study",
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
        "per_profile": {p: _case_metrics(m) for p, m in sorted(per_profile.items())},
        "verdicts_by_profile": {
            profile: dict(sorted(collections.Counter(
                a.verdict for a in assignments if a.profile == profile
            ).items()))
            for profile in sorted(profiles)
        },
    }
