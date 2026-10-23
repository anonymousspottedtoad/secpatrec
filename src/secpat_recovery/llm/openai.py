from __future__ import annotations

import json

import openai as _openai

from .base import LLMProvider, ToolResponse, ToolUse
from .rate_limit import retry_on_rate_limit


def _to_oai_messages(system: str, messages: list[dict]) -> list[dict]:
    """Convert Anthropic-style message list to OpenAI chat format."""
    oai: list[dict] = [{"role": "system", "content": system}]
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            oai.append({"role": role, "content": content})
            continue

        # content is a list of blocks
        if role == "user":
            tool_results = [b for b in content if b.get("type") == "tool_result"]
            text_parts = [b for b in content if b.get("type") == "text"]
            # tool_result blocks each become a "tool" role message
            for tr in tool_results:
                result_content = tr["content"]
                if not isinstance(result_content, str):
                    result_content = json.dumps(result_content)
                oai.append(
                    {
                        "role": "tool",
                        "tool_call_id": tr["tool_use_id"],
                        "content": result_content,
                    }
                )
            if text_parts:
                oai.append(
                    {
                        "role": "user",
                        "content": "\n".join(p["text"] for p in text_parts),
                    }
                )

        elif role == "assistant":
            text_parts = [b for b in content if b.get("type") == "text"]
            tool_uses = [b for b in content if b.get("type") == "tool_use"]
            msg_dict: dict = {
                "role": "assistant",
                "content": "\n".join(p["text"] for p in text_parts) if text_parts else None,
            }
            if tool_uses:
                msg_dict["tool_calls"] = [
                    {
                        "id": tu["id"],
                        "type": "function",
                        "function": {
                            "name": tu["name"],
                            "arguments": json.dumps(tu["input"]),
                        },
                    }
                    for tu in tool_uses
                ]
            oai.append(msg_dict)

    return oai


def _to_oai_tools(tools: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


def _to_responses_tools(tools: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t["input_schema"],
            "strict": False,  # the same schema Chat Completions took, unvalidated
        }
        for t in tools
    ]


def _responses_usage(u) -> dict[str, int] | None:
    """A Responses API ``usage`` block under the ``USAGE_KEYS`` names."""
    if u is None:
        return None
    return {
        "input_tokens": u.input_tokens or 0,
        "cached_input_tokens": getattr(u.input_tokens_details, "cached_tokens", 0) or 0,
        "output_tokens": u.output_tokens or 0,
        "reasoning_tokens": getattr(u.output_tokens_details, "reasoning_tokens", 0) or 0,
    }


def _chat_usage(u) -> dict[str, int] | None:
    """A Chat Completions ``usage`` block under the ``USAGE_KEYS`` names."""
    if u is None:
        return None
    return {
        "input_tokens": u.prompt_tokens or 0,
        "cached_input_tokens": getattr(u.prompt_tokens_details, "cached_tokens", 0) or 0,
        "output_tokens": u.completion_tokens or 0,
        # OpenAI nests it under completion_tokens_details; vLLM's GLM
        # server leaves that null and reports it at the top level.
        "reasoning_tokens": getattr(u.completion_tokens_details, "reasoning_tokens", 0)
        or getattr(u, "reasoning_tokens", 0)
        or 0,
    }


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        model: str = "gpt-4o",
        max_tokens: int = 8192,
        vision_model: str | None = None,
        reasoning_effort: str | None = None,
    ):
        self._model = model
        # Image (vision) calls may use a stronger model than text tasks.
        self._vision_model = vision_model or model
        self._max_tokens = max_tokens
        # Forwarded on every call when set (reasoning models only).
        self._reasoning_effort = reasoning_effort
        # call_id → the encrypted reasoning items that produced that call.
        self._reasoning_by_call: dict[str, list[dict]] = {}
        self._client = _openai.OpenAI()

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def vision_model_id(self) -> str:
        return self._vision_model

    def _create(self, *, model: str, messages, tools=None, **kwargs):
        """One chat completion. Requests go out at full speed; a rate limit is
        waited out and the request reissued, so a long run survives a 429
        instead of dying on it."""
        return retry_on_rate_limit(
            lambda: self._client.chat.completions.create(
                model=model,
                max_completion_tokens=self._max_tokens,
                messages=messages,
                **({"tools": tools} if tools else {}),
                **({"reasoning_effort": self._reasoning_effort} if self._reasoning_effort else {}),
                **kwargs,
            )
        )

    def response(self, system_prompt: str, user_message: str) -> str:
        resp = self._create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
        )
        self.last_usage = _chat_usage(resp.usage)
        return resp.choices[0].message.content

    def vision_response(
        self,
        system_prompt: str,
        user_message: str,
        images: list[dict] | None = None,
    ) -> str:
        content: list[dict] = [{"type": "text", "text": user_message}]
        for img in images or []:
            data_uri = f"data:{img['media_type']};base64,{img['data']}"
            content.append({"type": "image_url", "image_url": {"url": data_uri}})
        resp = self._create(
            model=self._vision_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
        )
        return resp.choices[0].message.content

    def tool_response(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        *,
        parallel_tools: bool = True,
    ) -> ToolResponse:
        """One ReAct turn on the Responses API.

        Not Chat Completions: there, gpt-6-sol and gpt-6.1-sol refuse function
        tools unless ``reasoning_effort`` is ``"none"``, and gpt-6.1-sol does not
        accept ``"none"`` at all. The request is stateless (``store=False``); each
        turn's reasoning comes back encrypted and is replayed before the tool call
        it led to, so the model keeps its reasoning across steps.
        """
        extra: dict = {}
        if not parallel_tools:
            extra["parallel_tool_calls"] = False
        if self._reasoning_effort:
            extra["reasoning"] = {"effort": self._reasoning_effort}
        resp = self._responses_create(
            model=self._model,
            instructions=system,
            input=self._to_responses_input(messages),
            tools=_to_responses_tools(tools),
            max_output_tokens=self._max_tokens,
            **extra,
        )
        # Only a reasoning item that carries its encrypted content can be sent
        # back on a stateless request; anything else is dropped, not replayed.
        reasoning = [
            o.model_dump(exclude_none=True)
            for o in resp.output
            if o.type == "reasoning" and getattr(o, "encrypted_content", None)
        ]
        tool_uses: list[ToolUse] = []
        for o in resp.output:
            if o.type == "function_call":
                tool_uses.append(ToolUse(id=o.call_id, name=o.name, input=json.loads(o.arguments)))
                self._reasoning_by_call[o.call_id] = reasoning
                reasoning = []  # replayed once, before the first call it led to
        return ToolResponse(
            text=resp.output_text or "",
            tool_uses=tool_uses,
            stop_reason=(
                "max_tokens"
                if resp.status == "incomplete"
                else "tool_use"
                if tool_uses
                else "end_turn"
            ),
            usage=_responses_usage(resp.usage),
        )

    def _responses_create(self, **kwargs):
        """One Responses API call, stateless, its reasoning returned encrypted
        so the next turn can replay it."""
        return retry_on_rate_limit(
            lambda: self._client.responses.create(
                store=False, include=["reasoning.encrypted_content"], **kwargs
            )
        )

    def _to_responses_input(self, messages: list[dict]) -> list[dict]:
        """Anthropic-style history → Responses input items, reasoning replayed."""
        items: list[dict] = []
        for msg in messages:
            role, content = msg["role"], msg["content"]
            if isinstance(content, str):
                items.append({"role": role, "content": content})
                continue
            for b in content:
                if b.get("type") == "text":
                    items.append({"role": role, "content": b["text"]})
                elif b.get("type") == "tool_use":
                    items.extend(self._reasoning_by_call.get(b["id"], []))
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": b["id"],
                            "name": b["name"],
                            "arguments": json.dumps(b["input"]),
                        }
                    )
                elif b.get("type") == "tool_result":
                    out = b["content"]
                    items.append(
                        {
                            "type": "function_call_output",
                            "call_id": b["tool_use_id"],
                            "output": out if isinstance(out, str) else json.dumps(out),
                        }
                    )
        return items

    def _chat_tool_response(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        *,
        parallel_tools: bool = True,
    ) -> ToolResponse:
        """One ReAct turn on Chat Completions — the self-hosted servers' API."""
        extra: dict = {}
        if not parallel_tools:
            extra["parallel_tool_calls"] = False
        resp = self._create(
            model=self._model,
            messages=_to_oai_messages(system, messages),
            tools=_to_oai_tools(tools),
            **extra,
        )
        choice = resp.choices[0]
        text = choice.message.content or ""
        tool_uses: list[ToolUse] = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_uses.append(
                    ToolUse(
                        id=tc.id,
                        name=tc.function.name,
                        input=json.loads(tc.function.arguments),
                    )
                )
        return ToolResponse(
            text=text,
            tool_uses=tool_uses,
            stop_reason=choice.finish_reason or "end_turn",
            usage=_chat_usage(resp.usage),
        )
