"""RecoveryAgent — the strict Thought → Action → Observation loop."""

from __future__ import annotations

import sys
from collections.abc import Iterable

from ..kb.builder import KBIndex
from ..llm.base import USAGE_KEYS, LLMProvider
from ..repo_access import RepoAccess
from .prompt import SYSTEM
from .tools import (
    build_tools,
    exec_grep,
    exec_list_dir,
    exec_read_file,
    exec_search,
    fmt_input,
    split_thought,
    submission_problems,
)

# ── Exceptions ────────────────────────────────────────────────────────────────


class AgentError(Exception):
    pass


class MaxStepsExceededError(Exception):
    pass


# ── RecoveryAgent ─────────────────────────────────────────────────────────────


class RecoveryAgent:
    MAX_STEPS = 50

    def __init__(
        self,
        provider: LLMProvider,
        kb: KBIndex,
        repo: RepoAccess | None = None,
        disabled_tools: Iterable[str] = (),
        system: str | None = None,
        system_source: str = "",
    ) -> None:
        self._provider = provider
        self._kb = kb
        # Raw-file access backing the grep/read_file tools. Without it those two
        # tools report themselves unavailable rather than failing the run.
        self._repo = repo
        # Tool ablation: the disabled tools are removed from the schemas sent to
        # the provider. `build_tools` rejects an unknown name, so a typo fails
        # here rather than yielding a silent control arm.
        self._disabled: tuple[str, ...] = tuple(
            n.strip() for n in disabled_tools if n and n.strip()
        )
        self._tools = build_tools(self._disabled)
        self._tool_names: tuple[str, ...] = tuple(t["name"] for t in self._tools)
        # The prompt is a *given*, not something derived from the tool list. The
        # canonical `SYSTEM` in prompt.py is the one every normal run has always
        # used and is never rewritten here; an ablation arm supplies its own
        # copy — `src/secpat_recovery/ablation/prompts/<arm>.md`, the same prompt with
        # the missing tool edited out — so the text each arm ran on is a file on
        # disk that can be read and diffed, not a string assembled at runtime.
        self._system = system if system is not None else SYSTEM
        self._system_source = system_source or "prompt.SYSTEM"
        if self._disabled:
            print(
                f"[RecoveryAgent] Ablation: {', '.join(self._disabled)} disabled "
                f"(tools: {', '.join(self._tool_names)}; prompt: {self._system_source})",
                file=sys.stderr,
            )
        # Structured, ordered record of the whole run (prompt, thoughts, tool
        # calls, observations, final mapping). Populated by map(); rendered to
        # the run transcript written alongside recovery.json.
        self.transcript: list[dict] = []
        #: ReAct steps the last run took, `submit_mapping` included — the agent's
        #: unit of work, and what an ablation moves when it moves no score. Read
        #: off the transcript rather than counted separately so the two can never
        #: disagree.
        self.steps: int = 0
        #: Tokens the last run billed, summed over its steps (``USAGE_KEYS``), or
        #: None when the provider reported none. Each step's own counts are in
        #: the transcript.
        self.usage: dict[str, int] | None = None

    @property
    def tool_names(self) -> tuple[str, ...]:
        """The tools this run was given, `submit_mapping` included."""
        return self._tool_names

    def map(self, pattern_text: str) -> dict:
        kb_size = self._kb.count()
        # The first user message is the pattern and nothing else — `load_pattern`
        # has already framed it with the description of its own JSON shape, and
        # every instruction (including the KB size) lives in the system prompt.
        task = pattern_text.strip()
        messages: list[dict] = [{"role": "user", "content": task}]

        self.transcript = [
            {
                "type": "prompt",
                "model": self._provider.model_id,
                "kb_size": kb_size,
                # Which tools this run had is part of what produced it: an
                # ablated transcript read without them looks like an agent that
                # inexplicably never grepped.
                "tools": list(self._tool_names),
                "disabled_tools": list(self._disabled),
                "system_source": self._system_source,
                "system": self._system,
                "task": task,
            }
        ]

        print(
            f"[RecoveryAgent] Starting ({self._provider.model_id}, KB size={kb_size})",
            file=sys.stderr,
        )

        self.usage = None
        for step in range(self.MAX_STEPS):
            # parallel_tools=False: strict ReAct is one Action per Thought, so the
            # provider is asked for a single tool call per turn.
            resp = self._provider.tool_response(
                self._system, messages, self._tools, parallel_tools=False
            )
            step_no = step + 1
            if resp.usage is not None:
                self.usage = {
                    k: (self.usage or {}).get(k, 0) + resp.usage.get(k, 0) for k in USAGE_KEYS
                }

            # Append assistant turn to message history
            assistant_content: list[dict] = []
            if resp.text:
                assistant_content.append({"type": "text", "text": resp.text})
            for tu in resp.tool_uses:
                assistant_content.append(
                    {"type": "tool_use", "id": tu.id, "name": tu.name, "input": tu.input}
                )
            messages.append({"role": "assistant", "content": assistant_content})

            if not resp.tool_uses:
                raise AgentError(
                    "Agent produced no tool calls. Expected search, read_file, or submit_mapping."
                )

            # --- THOUGHT (schema-required; free text is only a fallback) ---
            action, extras = resp.tool_uses[0], resp.tool_uses[1:]
            thought, args = split_thought(action)
            if not thought:
                thought = resp.text.strip()

            print(
                f"[Agent] Step {step_no} thought: {thought[:120] or '(none)'}",
                file=sys.stderr,
            )

            # --- DONE --- (only on a submission the schema accepts; an
            # incomplete one is answered as an Observation below, and the
            # model submits again)
            problems = submission_problems(args) if action.name == "submit_mapping" else []
            if action.name == "submit_mapping" and not problems:
                self.steps = step_no
                print(f"[Agent] Done after {step_no} steps.", file=sys.stderr)
                self.transcript.append(
                    {
                        "type": "submit",
                        "step": step_no,
                        "thought": thought,
                        "input": args,
                        "usage": resp.usage,
                    }
                )
                return args

            # --- ACTION ---
            print(
                f"[Agent] Step {step_no} action: {action.name}({fmt_input(action)})",
                file=sys.stderr,
            )

            # --- OBSERVATION ---
            if problems:
                print(
                    f"[Agent] Step {step_no}: submit_mapping rejected — "
                    f"{len(problems)} required field gap(s)",
                    file=sys.stderr,
                )
                observation = (
                    "[REJECTED] submit_mapping is incomplete; the run has not ended. "
                    "Every entry needs all of patternRole, status, entity, file, "
                    'lines and notes (use "" for a missing role\'s entity/file/'
                    "lines). Call submit_mapping again with the complete mapping.\n"
                    + "\n".join(f"- {p}" for p in problems)
                )
            else:
                observation = self._exec_tool(action.name, args)
            if not thought:
                observation += (
                    "\n\n[REMINDER] You issued this action without a `thought`. "
                    "Every action must carry the reasoning that led to it — supply "
                    "`thought` on your next call."
                )

            self.transcript.append(
                {
                    "type": "step",
                    "step": step_no,
                    "thought": thought,
                    "tool": action.name,
                    "input": args,
                    "observation": observation,
                    "usage": resp.usage,
                }
            )

            tool_results: list[dict] = [
                {
                    "type": "tool_result",
                    "tool_use_id": action.id,
                    "content": f"Observation:\n{observation}",
                }
            ]
            # A tool_result is required for every tool_use block, so any extra
            # call the provider emitted despite parallel_tools=False is answered
            # but not executed — one Action per step, no silent drops.
            for extra in extras:
                print(
                    f"[Agent] Step {step_no}: skipped extra action {extra.name!r}",
                    file=sys.stderr,
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": extra.id,
                        "content": (
                            "Observation:\n[SKIPPED] This loop executes exactly one "
                            "action per step. Only your first tool call "
                            f"({action.name}) was run. Read its observation, then "
                            "reason again and issue this action next step if you "
                            "still need it."
                        ),
                    }
                )

            messages.append({"role": "user", "content": tool_results})

        raise MaxStepsExceededError(
            f"Agent did not call submit_mapping within {self.MAX_STEPS} steps."
        )

    def _exec_tool(self, name: str, args: dict) -> str:
        """Run one Action. `args` has the ReAct `thought` already stripped out."""
        if name in self._disabled:
            # Unreachable through a conforming provider — the tool is not in the
            # schemas it was sent — but an ablation must not quietly execute the
            # capability it is measuring the absence of.
            return (
                f"[ERROR] {name!r} is not available in this run. Available tools: "
                f"{', '.join(self._tool_names)}."
            )
        if name == "search":
            # The tool set goes in too: a search observation points at grep and
            # read_file, and an arm without one of those must not be told to use
            # it mid-run — the same leak the arm's prompt closes at step 0.
            return exec_search(self._kb, args.get("query", ""), enabled=self._tool_names)
        if name == "grep":
            return exec_grep(self._repo, args)
        if name == "read_file":
            return exec_read_file(self._repo, args)
        if name == "list_dir":
            return exec_list_dir(self._repo, args)
        return f"[ERROR] Unknown tool: {name!r}"
