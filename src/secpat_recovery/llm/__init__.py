from __future__ import annotations

import os

from .base import LLMProvider, ToolResponse, ToolUse

# Self-hosted model families, each selectable as its own `--provider`, mapped to
# the model tag used when `--model` is not given. Adding a model to the study is
# a row here — the CLI's provider choices are read straight off this map. Kept in
# this module (not `self_hosted`) so listing the names costs no import.
SELF_HOSTED_MODELS: dict[str, str] = {
    "qwen": "Qwen/Qwen3.6-35B-A3B",
    "devstral": "devstral:24b",
    "gpt-oss": "gpt-oss:20b",
    "deepseek": "deepseek-ai/DeepSeek-V4-Flash-0731",
    "glm": "zai-org/GLM-5.3-Flash",
}

PROVIDERS: list[str] = ["claude", "openai", "muse", *SELF_HOSTED_MODELS]

#: Reasoning effort of every `--provider openai` call. gpt-6.1-sol has no "none"
#: (its lowest is "low"), so every OpenAI run reasons; this fixes how much.
OPENAI_REASONING_EFFORT = "medium"

#: Reasoning effort of every `--provider glm` call. GLM keeps its thinking, so
#: this is fixed here rather than read from LLM_REASONING_EFFORT, whose "none"
#: is the other self-hosted models' setting. The GLM endpoint has no "medium".
GLM_REASONING_EFFORT = "low"


def make_agent(
    name: str,
    model: str | None = None,
    thinking: bool = False,
    effort: str = "high",
    max_tokens: int | None = None,
) -> LLMProvider:
    """Build the LLM agent (provider) that backs a reasoning/vision task.

    `max_tokens` raises the per-response output cap. Every provider here keeps
    its own default, sized for a ReAct step's few hundred tokens; a one-shot
    caller that asks for a whole mapping in a single response needs room for
    all of it, and on a thinking model for the reasoning that bills as output
    alongside it. Omitted, each provider's own default stands.
    """
    cap = {"max_tokens": max_tokens} if max_tokens else {}
    if name == "claude":
        from .claude import ClaudeProvider

        return ClaudeProvider(
            model=model or "claude-opus-4-8", thinking=thinking, effort=effort, **cap
        )
    if name == "openai":
        from .openai import OpenAIProvider

        # gpt-6.1-sol for both text/reasoning and image (vision) tasks, at a
        # fixed OPENAI_REASONING_EFFORT. LLM_REASONING_EFFORT is the self-hosted
        # models' knob (its "none" turns Qwen's thinking off) and is not read
        # here: gpt-6.1-sol rejects "none", and the effort a run used must not
        # depend on a setting meant for another provider.
        return OpenAIProvider(
            model=model or "gpt-6.1-sol",
            vision_model="gpt-6.1-sol",
            reasoning_effort=OPENAI_REASONING_EFFORT,
            **cap,
        )
    if name == "muse":
        from .muse import MuseProvider

        # Meta's Muse on api.meta.ai's Responses API, streamed, at the fixed
        # sampling settings in `muse.py`. MUSE_API_KEY authenticates it.
        return MuseProvider(model=model or "muse-spark-1.3-contributor", **cap)
    if name in SELF_HOSTED_MODELS:
        # A self-hosted model family, served on Modal by default. `--model`
        # overrides the family's default tag; LLM_BASE_URL points at another
        # serving runtime (e.g. Ollama on localhost). DeepSeek and GLM each have
        # their own copy of the provider, holding their own endpoint.
        if name == "deepseek":
            from .deepseek import DeepSeekProvider as provider_cls
        elif name == "glm":
            from .glm import GlmProvider as provider_cls
        else:
            from .self_hosted import SelfHostedProvider as provider_cls

        num_ctx = os.getenv("LLM_NUM_CTX")
        return provider_cls(
            model=model or SELF_HOSTED_MODELS[name],
            num_ctx=int(num_ctx) if num_ctx else None,
            reasoning_effort=(
                GLM_REASONING_EFFORT if name == "glm" else os.getenv("LLM_REASONING_EFFORT") or None
            ),
            **cap,
        )
    choices = ", ".join(repr(n) for n in PROVIDERS)
    raise ValueError(f"Unknown provider: {name!r}. Choose one of {choices}.")


__all__ = [
    "PROVIDERS",
    "SELF_HOSTED_MODELS",
    "LLMProvider",
    "ToolResponse",
    "ToolUse",
    "make_agent",
]
