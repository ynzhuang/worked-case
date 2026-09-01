"""Retrieval metrics for both paths.

The precise path is measured on whether it returns the cohort the definition
claims.  The discovery path is measured on ranking quality and on the negation
false positive rate with the assertion filter on and off — which is where that
contrast belongs, because a coded AE row asserts presence by construction and a
narrative does not.
"""

from __future__ import annotations

from typing import Any

from .metrics import precision_at_k, recall_at_k, recall_ceiling_at_k, reciprocal_rank

_KS = (5, 10, 20, 50)


def evaluate_retrieval(pipeline, definition) -> dict[str, Any]:
    from ..retrieval.query import discover, retrieve

    index = pipeline.index()
    catalog = pipeline.catalog
    concept = definition.concept.primary

    # -- precise path -------------------------------------------------------
    evaluated = pipeline.evaluate(definition)
    index.record_assignments(evaluated)
    assignments = {a.episode_id: a for a in evaluated}
    cohort_truth = {i for i, a in assignments.items() if a.verdict == "case"}
    precise = retrieve(
        index, catalog, concept=concept, verdict=["case"],
        definition_id=definition.id, definition_version=definition.version,
        mode="precise", top_k=2000,
    )
    returned = {r.episode_id for r in precise.records}
    precise_view = {
        "returned": len(returned),
        "cohort_size": len(cohort_truth),
        "exact_match": returned == cohort_truth,
        "missing": sorted(cohort_truth - returned)[:5],
        "extra": sorted(returned - cohort_truth)[:5],
        "candidates_returned": precise.candidates_excluded,
        "usable_as_cohort": precise.to_dict()["usable_as_cohort"],
    }

    # -- discovery path -----------------------------------------------------
    relevant = {
        row["mention_id"]
        for row in index.query(
            "SELECT mention_id FROM mentions WHERE concept_id=? AND assertion='present'",
            (concept,),
        )
    }

    def view(result) -> dict[str, Any]:
        ranked = [m.mention_id for m in result.mentions]
        return {
            "returned": len(ranked),
            "mentions_asserting_absence": result.negation_false_positives,
            "negation_false_positive_rate": round(
                result.negation_false_positive_rate, 4
            ),
            "mrr": round(reciprocal_rank(ranked, relevant), 4),
            **{f"recall@{k}": round(recall_at_k(ranked, relevant, k), 4) for k in _KS},
            **{f"ceiling@{k}": round(recall_ceiling_at_k(relevant, k), 4) for k in _KS},
            **{
                f"precision@{k}": round(precision_at_k(ranked, relevant, k), 4)
                for k in _KS
            },
        }

    with_filter = discover(
        index, catalog, concept=concept, assertion=["present"], top_k=2000
    )
    without_filter = discover(index, catalog, concept=concept, top_k=2000)

    per_study = []
    for study in index.studies():
        study_relevant = {
            row["mention_id"]
            for row in index.query(
                "SELECT mention_id FROM mentions WHERE concept_id=? AND "
                "assertion='present' AND study_id=?",
                (concept, study),
            )
        }
        result = discover(
            index, catalog, concept=concept, assertion=["present"],
            studies=[study], top_k=2000,
        )
        ranked = [m.mention_id for m in result.mentions]
        per_study.append({
            "study": study,
            "relevant": len(study_relevant),
            "returned": len(ranked),
            "mrr": round(reciprocal_rank(ranked, study_relevant), 4),
            "precision@10": round(precision_at_k(ranked, study_relevant, 10), 4),
            "recall@50": round(recall_at_k(ranked, study_relevant, 50), 4),
            "ceiling@50": round(recall_ceiling_at_k(study_relevant, 50), 4),
        })

    return {
        "available": True,
        "concept": concept,
        "expanded_terms": with_filter.expanded_terms,
        "indexed_mentions": index.meta().mention_count,
        "relevant_mentions": len(relevant),
        "precise": precise_view,
        "assertion_filter_on": view(with_filter),
        "assertion_filter_off": view(without_filter),
        "per_study": per_study,
        "notes": with_filter.notes,
    }
