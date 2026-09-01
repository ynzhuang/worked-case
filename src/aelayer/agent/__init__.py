"""Traceable agent.

``compile`` produces an inspectable specification.  ``execute`` calls only
registered services.  ``trace`` follows any reported number back to the text a
site wrote — which is the requirement, rather than an approval click on a plan
nobody can independently evaluate.
"""

from .compile import compile_question
from .run import AgentSession, EvidencePackage
from .tools import SERVICES, AgentServices
from .trace import render_trace, trace_number

__all__ = [
    "AgentServices", "AgentSession", "EvidencePackage", "SERVICES",
    "compile_question", "render_trace", "trace_number",
]
