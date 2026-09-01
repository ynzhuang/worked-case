"""The program knowledge layer.

Not a third search index.  It is a registry of governed executions plus one
executable comparison, and its honest lifecycle is forward capture: it accrues
from executions and is empty on day one.
"""

from .registry import KnowledgeRegistry, ScopeRequired
from .diff import DefinitionComparison, diff_definitions

__all__ = [
    "DefinitionComparison",
    "KnowledgeRegistry",
    "ScopeRequired",
    "diff_definitions",
]
