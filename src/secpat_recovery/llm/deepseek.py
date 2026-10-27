"""DeepSeek V4 Flash on its own Modal endpoint — `--provider deepseek`.

DeepSeek and GLM each have their own copy of this module, holding their own
endpoint; auth (`MODAL_KEY`/`MODAL_SECRET`), the request shape and the overrides
below are the same. The serving runtime (a Modal-hosted vLLM server in the
paper's runs; locally Ollama, llama.cpp's `llama-server`, vLLM, LM Studio)
exposes an OpenAI-compatible `/v1/chat/completions` endpoint, tool calls
included, so this is not a new wire protocol, only a different `base_url`
(`LLM_BASE_URL`). The provider subclasses `OpenAIProvider` and inherits the
Anthropic↔OpenAI message/tool translation unchanged.

`SELF_HOSTED_MODELS` (in `llm/__init__.py`) holds the model's default tag;
`--model` overrides it, and `model_id` — recorded in every transcript header —
always names the concrete tag that ran. The tag must match what the server
reports at `/v1/models`.

What genuinely differs from the hosted OpenAI API, and is handled here:

* **Auth is the server's own, not OpenAI's.** A local server ignores the key
  entirely (the SDK still refuses to construct a client without one, so a
  placeholder is supplied); the Modal endpoint sits behind Modal's proxy auth,
  a token id/secret pair checked at the proxy before vLLM sees the request.
  Joined by a dot (`<id>.<secret>`) the pair *is* a bearer token, so it rides
  in `api_key` and needs no custom headers — set `MODAL_KEY`/`MODAL_SECRET`
  (or `MODAL_PROXY_TOKEN_ID`/`_SECRET`). Without them Modal answers
  `401 {"error":"proxy auth required"}` before the request reaches the model.
* **`max_tokens`, not `max_completion_tokens`.** The newer field is an OpenAI
  addition; local servers generally only implement the original.
* **JSON mode is optional.** `response_format={"type": "json_object"}` is
  unsupported by some builds, so the first refusal downgrades the call to a
  plain completion for the rest of the run rather than failing it.
* **Reasoning models burn the step budget thinking.** Hidden reasoning before
  every tool call comes on top of a `thought` the schema already requires, so
  the loop pays for its reasoning twice. `reasoning_effort=` is forwarded when
  set (Ollama ignores the native `think` field on this endpoint).
* **Context length must be raised explicitly.** Ollama truncates to 4096 tokens
  by default; a 50-step ReAct transcript blows past that long before
  `submit_mapping`, and truncation is silent — the run just degrades. Set
  `OLLAMA_CONTEXT_LENGTH=32768` (or a Modelfile `PARAMETER num_ctx`) before
  serving; `num_ctx=` here forwards the same request in-band for servers that
  honour it. It is an Ollama option: on vLLM the context window is fixed at
  launch (`--max-model-len`), so leave `LLM_NUM_CTX` unset against Modal — vLLM
  rejects unknown `extra_body` keys.
"""

from __future__ import annotations

import os

import openai as _openai

from .openai import OpenAIProvider
from .rate_limit import retry_on_rate_limit

# Default serving endpoint: a local vLLM server. Set
# LLM_BASE_URL to serve locally instead — Ollama :11434/v1, llama.cpp :8080/v1,
# vLLM :8000/v1, LM Studio :1234/v1.
DEFAULT_BASE_URL = "http://localhost:8000/v1"

LOCAL_BASE_URL = "http://localhost:11434/v1"


def _modal_proxy_key() -> str | None:
    """Modal's proxy-auth token pair, in the form the OpenAI SDK can carry.

    Modal fronts a web endpoint with its own auth, checked at the proxy before
    vLLM sees the request. It accepts the token id and secret either as the
    `Modal-Key`/`Modal-Secret` header pair or joined by a dot as a single
    bearer token — the latter is what an OpenAI client passes as `api_key`, so
    no custom headers are needed.
    """
    key = os.getenv("MODAL_KEY") or os.getenv("MODAL_PROXY_TOKEN_ID")
    secret = os.getenv("MODAL_SECRET") or os.getenv("MODAL_PROXY_TOKEN_SECRET")
    return f"{key}.{secret}" if key and secret else None


class DeepSeekProvider(OpenAIProvider):
    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        max_tokens: int = 8192,
        vision_model: str | None = None,
        num_ctx: int | None = None,
        reasoning_effort: str | None = None,
        api_key: str | None = None,
    ):
        self._model = model
        self._vision_model = vision_model or model
        self._max_tokens = max_tokens
        self._num_ctx = num_ctx
        self._reasoning_effort = reasoning_effort
        self._json_mode = True
        self._base_url = base_url or os.getenv("LLM_BASE_URL") or DEFAULT_BASE_URL
        self._client = _openai.OpenAI(
            base_url=self._base_url,
            api_key=(api_key or os.getenv("LLM_API_KEY") or _modal_proxy_key() or "self-hosted"),
            # A cold Modal container loads the weights before answering; the
            # SDK's 10-minute default is generous enough for that, but a long
            # ReAct turn on a big model can still outrun the default retries.
            timeout=float(os.getenv("LLM_TIMEOUT_S", "600")),
        )

    # The servers speak Chat Completions, not the Responses API OpenAI's own
    # models are driven through.
    tool_response = OpenAIProvider._chat_tool_response

    @property
    def base_url(self) -> str:
        return self._base_url

    def _create(self, *, model: str, messages, tools=None, **kwargs):
        """One chat completion against the local server.

        `retry_on_rate_limit` is kept for its second job: it also retries a
        dropped connection, which is what the server does while a large model
        is still being loaded into memory.
        """
        kwargs.pop("max_completion_tokens", None)
        if not self._json_mode:
            kwargs.pop("response_format", None)
        extra = dict(kwargs.pop("extra_body", None) or {})
        if self._reasoning_effort is not None:
            # Not a named parameter of the SDK's `create`, so it travels in the
            # request body rather than as a keyword.
            extra["reasoning_effort"] = self._reasoning_effort
        if self._num_ctx is not None:
            extra.setdefault("options", {}).setdefault("num_ctx", self._num_ctx)
        if extra:
            kwargs["extra_body"] = extra

        def call():
            return self._client.chat.completions.create(
                model=model,
                max_tokens=self._max_tokens,
                messages=messages,
                **({"tools": tools} if tools else {}),
                **kwargs,
            )

        try:
            return retry_on_rate_limit(call)
        except _openai.BadRequestError:
            # Most likely the server rejected one of the optional extras. Drop
            # them (for good) and retry once; anything else re-raises on the
            # second go.
            if "response_format" not in kwargs and "reasoning_effort" not in extra:
                raise
            self._json_mode = False
            self._reasoning_effort = None
            kwargs.pop("response_format", None)
            extra.pop("reasoning_effort", None)
            kwargs["extra_body"] = extra
            if not extra:
                kwargs.pop("extra_body")
            return retry_on_rate_limit(call)
