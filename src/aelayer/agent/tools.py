"""The registered services an execution may call.

Five, and nothing else.  Each is validated code invoked with a schema-checked
specification; none of them is written or modified by a model.  Anything the
agent reports has to have come through one of these.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ..models import CaseAssignment, PhenotypeQuerySpec
from ..pipeline import Pipeline

SERVICES = (
    "cohort_evidence",
    "exposure_and_covariates",
    "statistical_analysis",
    "genetics_handoff",
    "summarise",
)


@dataclass
class AgentServices:
    pipeline: Pipeline

    # -- cohort and evidence ------------------------------------------------

    def cohort_evidence(self, spec: PhenotypeQuerySpec) -> dict[str, Any]:
        """The cohort a definition claims, with the evidence behind it."""
        definition = self.pipeline.definition(
            spec.definition_id, spec.definition_version
        )
        assignments = self.pipeline.evaluate(definition, spec.studies or None)
        return {
            "definition": {
                "id": definition.id, "version": definition.version,
                "status": definition.status, "hash": definition.definition_hash,
                "label": definition.label,
            },
            "assignments": [a.model_dump(mode="json") for a in assignments],
            "episodes": len(assignments),
        }

    # -- exposure and covariates -------------------------------------------

    def exposure_and_covariates(
        self, spec: PhenotypeQuerySpec
    ) -> dict[str, Any]:
        """Exposure and baseline covariates for the cohort's subjects."""
        subjects = [s for s, _study in self.pipeline.cohort(spec.studies or None)]
        exposures = self.pipeline.store.exposures_by_subject()
        demographics = {
            row["USUBJID"]: row for row in self.pipeline.store.rows("dm")
        }
        return {
            "subjects": len(subjects),
            "with_exposure": sum(1 for s in subjects if exposures.get(s)),
            "arms": _counts(demographics, subjects, "ARM"),
            "sex": _counts(demographics, subjects, "SEX"),
            "countries": _counts(demographics, subjects, "COUNTRY"),
        }

    # -- statistics ---------------------------------------------------------

    def statistical_analysis(
        self, assignments: Sequence[dict[str, Any]], spec: PhenotypeQuerySpec
    ) -> dict[str, Any]:
        """Counts and incidence proportions. Descriptive only.

        No inferential test is offered: this layer establishes cohorts, and a
        comparative claim needs a design the definition does not encode.
        """
        rows = list(assignments)
        subjects = {r["subject_id"] for r in rows}
        cases = {r["subject_id"] for r in rows if r["verdict"] == "case"}
        review = {r["subject_id"] for r in rows if r["verdict"] == "review"}
        by_study: dict[str, dict[str, Any]] = {}
        for row in rows:
            entry = by_study.setdefault(
                row["study_id"], {"subjects": set(), "cases": set(), "review": set()}
            )
            entry["subjects"].add(row["subject_id"])
            if row["verdict"] == "case":
                entry["cases"].add(row["subject_id"])
            if row["verdict"] == "review":
                entry["review"].add(row["subject_id"])
        return {
            "method": "descriptive counts and incidence proportion",
            "subjects": len(subjects),
            "case_subjects": len(cases),
            "review_subjects": len(review),
            "incidence_proportion": round(len(cases) / len(subjects), 4)
            if subjects else 0.0,
            "per_study": {
                study: {
                    "subjects": len(v["subjects"]),
                    "cases": len(v["cases"]),
                    "review": len(v["review"]),
                    "incidence_proportion": round(
                        len(v["cases"]) / len(v["subjects"]), 4
                    ) if v["subjects"] else 0.0,
                }
                for study, v in sorted(by_study.items())
            },
            "caveat": (
                "Incidence proportions here are descriptive. The review set is "
                "reported separately and is neither counted as cases nor "
                "discarded."
            ),
        }

    # -- handoff ------------------------------------------------------------

    def genetics_handoff(
        self, assignments: Sequence[dict[str, Any]]
    ) -> dict[str, Any]:
        """A case-control export shaped for an existing pipeline.

        Genetics and omics analysis is out of scope; producing a file another
        pipeline can consume is not.
        """
        rows = [
            {
                "subject_id": r["subject_id"],
                "study_id": r["study_id"],
                "status": (
                    1 if r["verdict"] == "case"
                    else (0 if r["verdict"] == "excluded" else None)
                ),
                "verdict": r["verdict"],
                "definition": f"{r['definition_id']}.v{r['definition_version']}",
                "definition_hash": r["definition_hash"],
            }
            for r in assignments
        ]
        return {
            "format": "case_control_v1",
            "rows": rows,
            "note": (
                "status is null for review-set subjects: they are neither cases "
                "nor controls until adjudicated, and coding them either way "
                "would put an unadjudicated judgement into someone else's "
                "analysis."
            ),
        }

    # -- summarise ----------------------------------------------------------

    def summarise(
        self, assignments: Sequence[dict[str, Any]], spec: PhenotypeQuerySpec
    ) -> dict[str, Any]:
        by_verdict: dict[str, int] = {}
        by_state: dict[str, int] = {}
        by_rule: dict[str, int] = {}
        per_study: dict[str, dict[str, int]] = {}
        flagged = 0
        for row in assignments:
            by_verdict[row["verdict"]] = by_verdict.get(row["verdict"], 0) + 1
            by_state[row["evidence_state"]] = by_state.get(row["evidence_state"], 0) + 1
            rule = row.get("matched_rule_id") or "(no rule matched)"
            by_rule[rule] = by_rule.get(rule, 0) + 1
            study = per_study.setdefault(row["study_id"], {})
            study[row["verdict"]] = study.get(row["verdict"], 0) + 1
            if row.get("linkage_review_required"):
                flagged += 1
        return {
            "episodes": len(assignments),
            "counts_by_verdict": dict(sorted(by_verdict.items())),
            "counts_by_state": dict(sorted(by_state.items())),
            "counts_by_rule": dict(sorted(by_rule.items())),
            "per_study": {s: dict(sorted(v.items())) for s, v in sorted(per_study.items())},
            "primary_case_count": by_verdict.get("case", 0),
            "review_set_count": by_verdict.get("review", 0),
            "linkage_flagged": flagged,
            "studies": list(spec.studies),
        }

    # -- dispatch -----------------------------------------------------------

    def call(self, name: str, **kwargs: Any) -> Any:
        if name not in SERVICES:
            raise ValueError(
                f"{name!r} is not a registered service. An execution may call "
                f"only: {list(SERVICES)}"
            )
        return getattr(self, name)(**kwargs)


def _counts(
    demographics: dict[str, dict[str, Any]], subjects: Sequence[str], column: str
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for subject in subjects:
        value = str((demographics.get(subject) or {}).get(column) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
