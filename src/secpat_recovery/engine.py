from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .interaction import detect_action_sequence
from .kb.builder import build_kb, kb_source, load_kb
from .llm.base import LLMProvider
from .recovery_agent import RecoveryAgent
from .repo_access import RepoAccess

logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    max_files: int = 150
    max_file_bytes: int = 65536
    extra_extensions: set[str] | None = None
    kb_dir: Path = field(default_factory=lambda: Path(".kb"))
    rebuild_kb: bool = False
    #: Whether stage 3 runs at all. The order is read out of the call graph the
    #: Joern query service exports; there is no second analysis to choose between,
    #: so this is a yes/no rather than a name.
    detect_order: bool = True
    #: Override the detected source language (the Joern frontend is chosen by it).
    language: str | None = None
    #: Tool-ablation: agent tools to withhold from this run (see
    #: ``recovery_agent.tools.ABLATABLE_TOOLS``). Empty for a normal run. The KB
    #: is still built when ``search`` is withheld — stage 3 and the scorer read
    #: ``struct.json`` from it, and only the agent's access is being ablated.
    disabled_tools: tuple[str, ...] = ()
    #: System prompt to run the agent on. ``None`` means the canonical
    #: ``recovery_agent.prompt.SYSTEM``, which is what every normal run uses and
    #: what no ablation rewrites. An arm passes the text of its own copy — see
    #: ``src/secpat_recovery/ablation/prompts/`` — so the prompt a run used is a file
    #: that can be read and diffed rather than a string assembled at runtime.
    system_prompt: str | None = None
    #: Where ``system_prompt`` came from, for the transcript header.
    system_prompt_source: str = ""


class PatternRecoveryEngine:
    def __init__(self, provider: LLMProvider, config: EngineConfig | None = None) -> None:
        self._provider = provider
        self._config = config or EngineConfig()

    def recover(self, codebase_path: Path, pattern_text: str) -> tuple[dict, list[dict]]:
        """Stages 1 and 2: KB → role mapping.

        Returns ``(result, transcript)``. ``result`` carries the agent's
        ``role_mappings`` and ``run_stats``; stage 3 is :meth:`derive_order`, run
        separately so the caller can save this result first — a stage-3 error
        (``KotlinCitation`` raises by design) must not discard a completed
        agent run.
        """
        cfg = self._config
        kb_dir = cfg.kb_dir

        # Rebuild if forced, or if the cached KB was built from a different
        # codebase (the KB dir is shared, so a stale index must not be reused).
        stale = kb_dir.exists() and kb_source(kb_dir) != str(codebase_path.resolve())
        if (cfg.rebuild_kb or stale) and kb_dir.exists():
            shutil.rmtree(kb_dir)

        if not kb_dir.exists():
            build_kb(codebase_path, kb_dir)

        kb = load_kb(kb_dir)
        # Raw-file access backs the agent's grep/read_file tools: the KB answers
        # "what is semantically similar", RepoAccess answers "where does this
        # exact string appear" and "show me the real file".
        agent = RecoveryAgent(
            self._provider,
            kb,
            repo=RepoAccess(codebase_path),
            disabled_tools=cfg.disabled_tools,
            system=cfg.system_prompt,
            system_source=cfg.system_prompt_source,
        )
        result = agent.map(pattern_text)

        # What the run cost in the agent's own unit. Every report that compares
        # arms or providers wants it, and deriving it from run.log afterwards
        # only works while the logs are kept — so it goes in the result itself.
        # `usage` is the provider's own billed token counts summed over the
        # steps (None when it reports none); reasoning tokens are part of
        # output_tokens and also broken out. (Checklist P1 asks for wall-clock
        # here too.)
        result["run_stats"] = {
            "steps": agent.steps,
            "max_steps": RecoveryAgent.MAX_STEPS,
            "tools": list(agent.tool_names),
            "disabled_tools": list(cfg.disabled_tools),
            "model": self._provider.model_id,
            "reasoning_effort": getattr(self._provider, "_reasoning_effort", None),
            "usage": agent.usage,
        }
        return result, agent.transcript

    def derive_order(self, codebase_path: Path, result: dict, pattern: dict | None = None) -> dict:
        """Stage 3: the interaction order of the roles in ``result``.

        ``pattern`` is the parsed ``04_pattern.json``, when the run has one — the
        role vocabulary orders the walk and the behaviour walkthrough picks its
        entry role. It is optional for the same reason a raw ``.mmd`` pattern is
        still accepted: without it the order is derived from the call graph
        alone, which is weaker but not wrong.

        A derivation failure comes back as an envelope with ``available: false``
        (see ``detect_action_sequence``); a contradiction between the mapping and
        the code raises.
        """
        return detect_action_sequence(
            codebase_path,
            result.get("role_mappings", []),
            pattern,
            detect=self._config.detect_order,
            language=self._config.language,
        )
