"""Extraction: free text and structured tables in, event objects out.

This is a configurable rule and lexicon baseline.  It is deterministic, runs
offline, and is not a trained clinical NLP model.  Nothing here assigns an
evidence state or decides whether a subject is a case: the extractor reports
what the text and the tables say, and the phenotype definition interprets it.
"""

from .engine import ExtractionEngine, extract_corpus

__all__ = ["ExtractionEngine", "extract_corpus"]
