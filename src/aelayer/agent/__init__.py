"""The agent: compile a question, run typed tools, return a traceable package."""

from .compile import CompileResult, compile_question
from .run import (
    AgentSession, ClarificationRequired, ConflictUnresolved, EvidencePackage,
)
from .tools import REGISTRY, SERVICES, AgentServices, ToolError
from .trace import render_trace, trace_number

__all__ = [
    "AgentServices", "AgentSession", "ClarificationRequired",
    "CompileResult", "ConflictUnresolved", "EvidencePackage", "REGISTRY",
    "SERVICES", "ToolError", "compile_question", "render_trace", "trace_number",
]
