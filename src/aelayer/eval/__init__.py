"""Evaluation: what the layer measurably does, and what it cannot claim."""

from .harness import EvaluationHarness, case_metrics, run_evaluation
from .metrics import PRF, ConfusionMatrix
from .transport import transportability

__all__ = [
    "PRF", "ConfusionMatrix", "EvaluationHarness", "case_metrics",
    "run_evaluation", "transportability",
]
