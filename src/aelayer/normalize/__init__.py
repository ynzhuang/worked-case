"""The deterministic path.

Controlled values, dates, units and gating logic.  No model runs here, ever.
What this path resolves, the model path is never asked about; the boundary is
enforced by ``aelayer.guards``, not by convention.
"""

from .records import RecordNormalizer, normalize_store

__all__ = ["RecordNormalizer", "normalize_store"]
