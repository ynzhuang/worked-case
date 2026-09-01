"""The model path.

Schema-constrained, span-grounded, and permitted to abstain.  It is asked only
about fields the deterministic path left unresolved; ``aelayer.guards`` enforces
that.

Two backends.  ``rules`` is a local clinical NLP baseline of lexicons and
ConText-style cue scoping — deterministic, offline, and not a trained model.
``llm`` is optional and used only when an API key is present.  With the network
disconnected the layer runs the rules backend and the run manifest says which
one produced the values.
"""

from .engine import ExtractionEngine, extract_records

__all__ = ["ExtractionEngine", "extract_records"]
