from __future__ import annotations

import os

import anthropic

from .base import LLMProvider, ToolResponse, ToolUse


def _text_of(msg) -> str:
    """Concatenate the text blocks of a message, skipping thinking blocks."""
    return "".join(b.text for b in msg.content if b.type == "text")


def _usage(u) -> dict[str, int]:
    """A Messages API ``usage`` block under the ``USAGE_KEYS`` names."""
    cached = (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0)
    # Anthropic reports the uncached prompt alone; thinking is billed
    # inside output_tokens and not broken out, so reasoning is 0 here.
    return {
        "input_tokens": u.input_tokens + cached,
        "cached_input_tokens": u.cache_read_input_tokens or 0,
        "output_tokens": u.output_tokens,
        "reasoning_tokens": 0,
    }


class ClaudeProvider(LLMProvider):
    def __init__(
        self,
        model: str = "claude-opus-4-8",
        max_tokens: int = 16000,
        thinking: bool = False,
        effort: str = "high",
    ):
        self._model = model
        # Adaptive thinking (Opus 4.8 / 4.7 / Sonnet 5): the on-mode is
        # {"type": "adaptive"}; budget_tokens is rejected with a 400. Depth is
        # controlled by output_config.effort (low | medium | high | xhigh | max).
        self._thinking = thinking
        self._effort = effort
        self._max_tokens = max_tokens
        # A key not scoped to a workspace is refused (400) unless every request
        # names one; ANTHROPIC_WORKSPACE_ID supplies it as the header the API asks for.
        workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()
        self._client = anthropic.Anthropic(
            default_headers={"anthropic-workspace-id": workspace} if workspace else None
        )

    @property
    def model_id(self) -> str:
        return self._model

    def _create(self, **kwargs):
        if self._thinking:
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["output_config"] = {"effort": self._effort}
        return self._client.messages.create(
            model=self._model, max_tokens=self._max_tokens, **kwargs
        )

    def response(self, system_prompt: str, user_message: str) -> str:
        msg = self._create(
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        self.last_usage = _usage(msg.usage)
        return _text_of(msg)

    def vision_response(
        self,
        system_prompt: str,
        user_message: str,
        images: list[dict] | None = None,
    ) -> str:
        content: list[dict] = [{"type": "text", "text": user_message}]
        for img in images or []:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img["media_type"],
                        "data": img["data"],
                    },
                }
            )
        msg = self._create(
            system=system_prompt,
            messages=[{"role": "user", "content": content}],  # type: ignore[arg-type]
        )
        return _text_of(msg)

    def tool_response(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        *,
        parallel_tools: bool = True,
    ) -> ToolResponse:
        extra: dict = {}
        if not parallel_tools:
            extra["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
        resp = self._create(
            system=system,
            tools=tools,  # type: ignore[arg-type]
            messages=messages,  # type: ignore[arg-type]
            **extra,
        )
        text_parts = [b.text for b in resp.content if b.type == "text"]
        tool_uses = [
            ToolUse(id=b.id, name=b.name, input=b.input)
            for b in resp.content
            if b.type == "tool_use"
        ]
        return ToolResponse(
            text="\n".join(text_parts),
            tool_uses=tool_uses,
            stop_reason=resp.stop_reason or "end_turn",
            usage=_usage(resp.usage),
        )
