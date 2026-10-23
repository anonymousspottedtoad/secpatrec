from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolUse:
    id: str
    name: str
    input: dict[str, Any]


#: The token counts a provider reports for one call, under one set of names:
#: ``input_tokens`` (the whole prompt, cached part included),
#: ``cached_input_tokens``, ``output_tokens`` (reasoning included) and
#: ``reasoning_tokens``. Read off the response's own ``usage`` — the billed
#: numbers, not an estimate.
USAGE_KEYS = ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens")


@dataclass
class ToolResponse:
    text: str
    tool_uses: list[ToolUse] = field(default_factory=list)
    stop_reason: str = "end_turn"
    #: ``USAGE_KEYS`` → count for this call; None when the provider sent none.
    usage: dict[str, int] | None = None


class LLMProvider(ABC):
    #: ``USAGE_KEYS`` → count for the last ``response()`` call; None until one
    #: is made, or when the provider sent none. ``response()`` returns bare text,
    #: so its billed usage is kept here for callers that report it (the baseline).
    last_usage: dict[str, int] | None = None

    @abstractmethod
    def response(self, system_prompt: str, user_message: str) -> str:
        """Send a single system + user turn; return assistant text."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Human-readable model identifier (text/reasoning calls)."""

    @property
    def vision_model_id(self) -> str:
        """Model identifier used for image (vision) calls; defaults to model_id."""
        return self.model_id

    def vision_response(
        self,
        system_prompt: str,
        user_message: str,
        images: list[dict] | None = None,
    ) -> str:
        """Single system + user turn that may include images.

        images: list of {"data": base64_str, "media_type": "image/png"} blocks.
        Providers without vision should fall back to text only.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support vision")

    def tool_response(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        *,
        parallel_tools: bool = True,
    ) -> ToolResponse:
        """Multi-turn call with tool use.

        Messages follow the Anthropic content-block format:
          user:      {"role": "user", "content": str | list[block]}
          assistant: {"role": "assistant", "content": list[text_block | tool_use_block]}
          tool_result turn: {"role": "user", "content": list[tool_result_block]}

        tools: list of Anthropic-style tool dicts with "name", "description",
        and "input_schema" (JSON Schema object).

        parallel_tools: when False, the provider is asked to emit at most one
        tool call per turn. `RecoveryAgent` sets this to keep its ReAct loop to
        exactly one Action per Thought.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support tool use")
