"""Recovery agent: the ReAct loop that maps pattern roles onto codebase entities.

- `prompt.py` — the system prompt.
- `tools.py`  — tool schemas (Thought-augmented) and their executors;
  `build_tools(disabled=...)` withholds tools for an ablation arm.
- `agent.py`  — `RecoveryAgent`, the Thought → Action → Observation loop.

`prompt.SYSTEM` is the canonical prompt and is never rewritten for an ablation:
an arm passes its own copy to `RecoveryAgent(system=...)`, kept as a file under
`src/secpat_recovery/ablation/prompts/`.
"""

from __future__ import annotations

from .agent import AgentError, MaxStepsExceededError, RecoveryAgent
from .prompt import SYSTEM
from .tools import ABLATABLE_TOOLS, RAW_TOOLS, TOOLS, build_tools

__all__ = [
    "RecoveryAgent",
    "AgentError",
    "MaxStepsExceededError",
    "SYSTEM",
    "RAW_TOOLS",
    "TOOLS",
    "ABLATABLE_TOOLS",
    "build_tools",
]
