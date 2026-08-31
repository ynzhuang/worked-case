"""Specification-first agent.

The agent compiles a question into an inspectable ``PhenotypeQuerySpec`` and
stops.  Execution requires explicit approval.  It never computes a statistic
itself and never names a case: it calls the tools in ``tools.py`` and nothing
else.
"""

from .compile import compile_question
from .run import AgentSession, EvidencePackage
from .tools import TOOLS, AgentTools

__all__ = [
    "AgentSession",
    "AgentTools",
    "EvidencePackage",
    "TOOLS",
    "compile_question",
]
