"""Metric primitives.

Small and explicit on purpose: a number in the report should be traceable to a
count of true positives, false positives and false negatives that a reader can
reconstruct.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


@dataclass
class PRF:
    """Precision, recall and F1 with the counts they came from."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, tp: int = 0, fp: int = 0, fn: int = 0) -> None:
        self.tp += tp
        self.fp += fp
        self.fn += fn

    @property
    def support(self) -> int:
        """Gold instances: what recall is measured against."""
        return self.tp + self.fn

    @property
    def predicted(self) -> int:
        return self.tp + self.fp

    @property
    def precision(self) -> float:
        return self.tp / self.predicted if self.predicted else 0.0

    @property
    def recall(self) -> float:
        return self.tp / self.support if self.support else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return (2 * p * r / (p + r)) if (p + r) else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "support": self.support,
            "predicted": self.predicted,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


def prf_from_counts(tp: int, fp: int, fn: int) -> PRF:
    return PRF(tp=tp, fp=fp, fn=fn)


def set_prf(gold: Iterable[Any], predicted: Iterable[Any]) -> tuple[int, int, int]:
    """TP/FP/FN for two sets, as used for symptoms and seriousness."""
    gold_set, predicted_set = set(gold), set(predicted)
    tp = len(gold_set & predicted_set)
    return tp, len(predicted_set - gold_set), len(gold_set - predicted_set)


def scalar_prf(gold: Any, predicted: Any) -> tuple[int, int, int]:
    """TP/FP/FN for a single-valued slot.

    A populated but wrong prediction counts once as a false positive and once
    as a false negative, which is the standard slot-filling convention and
    keeps precision and recall from flattering a confidently wrong extractor.
    """
    gold_present = gold not in (None, "", [])
    predicted_present = predicted not in (None, "", [])
    if gold_present and predicted_present:
        return (1, 0, 0) if gold == predicted else (0, 1, 1)
    if predicted_present:
        return 0, 1, 0
    if gold_present:
        return 0, 0, 1
    return 0, 0, 0


@dataclass
class ConfusionMatrix:
    labels: list[str]
    counts: dict[tuple[str, str], int] = field(default_factory=dict)

    def add(self, gold: str, predicted: str) -> None:
        for label in (gold, predicted):
            if label not in self.labels:
                self.labels.append(label)
        key = (gold, predicted)
        self.counts[key] = self.counts.get(key, 0) + 1

    def get(self, gold: str, predicted: str) -> int:
        return self.counts.get((gold, predicted), 0)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def correct(self) -> int:
        return sum(v for (g, p), v in self.counts.items() if g == p)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def per_label(self) -> dict[str, PRF]:
        out: dict[str, PRF] = {label: PRF() for label in self.labels}
        for (gold, predicted), count in self.counts.items():
            if gold == predicted:
                out[gold].add(tp=count)
            else:
                out[predicted].add(fp=count)
                out[gold].add(fn=count)
        return out

    def to_markdown(self, title: str = "gold \\ predicted") -> str:
        labels = sorted(self.labels)
        header = f"| {title} | " + " | ".join(labels) + " | total |"
        divider = "|" + "---|" * (len(labels) + 2)
        lines = [header, divider]
        for gold in labels:
            row_total = sum(self.get(gold, p) for p in labels)
            cells = " | ".join(str(self.get(gold, p)) for p in labels)
            lines.append(f"| **{gold}** | {cells} | {row_total} |")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": sorted(self.labels),
            "accuracy": round(self.accuracy, 4),
            "total": self.total,
            "counts": {f"{g}->{p}": c for (g, p), c in sorted(self.counts.items())},
        }


def reciprocal_rank(ranked_ids: Sequence[str], relevant: set[str]) -> float:
    for position, identifier in enumerate(ranked_ids, start=1):
        if identifier in relevant:
            return 1.0 / position
    return 0.0


def recall_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def recall_ceiling_at_k(relevant: set[str], k: int) -> float:
    """The best recall@k achievable when there are more relevant items than k.

    With 63 relevant documents, recall@5 cannot exceed 0.079 however perfect the
    ranking is. Reporting the raw figure without its ceiling invites reading a
    ranking that is exactly right as one that is mostly wrong.
    """
    if not relevant:
        return 0.0
    return min(k, len(relevant)) / len(relevant)


def precision_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    """Proportion of the top k that is relevant. Not capped by the relevant count."""
    top = ranked_ids[:k]
    if not top:
        return 0.0
    return len([i for i in top if i in relevant]) / len(top)
