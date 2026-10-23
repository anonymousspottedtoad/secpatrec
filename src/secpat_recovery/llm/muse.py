"""Meta's Muse models, served from `api.meta.ai`.

The endpoint is OpenAI's Responses API (`POST /v1/responses`), so
`MuseProvider` subclasses `OpenAIProvider` and keeps its Anthropic↔Responses
message/tool translation; only the client and the request shape change:

* **Auth and base URL are Meta's.** The key is read from `MUSE_API_KEY`;
  `MUSE_BASE_URL` overrides `https://api.meta.ai/v1`.
* **Every call streams.** The request goes out with `stream: true` and the
  final `response.completed` (or `response.incomplete`) event's response is
  what the caller gets — the same object a non-streaming call returns, so the
  parsing downstream is unchanged.
* **Sampling is fixed:** `temperature=1`, `top_p=1`, `max_output_tokens=32000`,
  reasoning effort `low`, matching the configuration the model is studied
  at. (`medium` reasoned ~1,000 tokens a step at ~27 tokens/s — ~40s a step.)
* **A transient `server_error` is retried.** The endpoint sometimes fails a
  stream with "The model failed to generate a response." and the identical
  request succeeds moments later; one such failure must not end a 50-step run,
  so it is backed off and reissued (`_SERVER_ERROR_ATTEMPTS`). Any other
  failure code raises at once.
* **No Chat Completions.** The plain-text and vision calls, which
  `OpenAIProvider` sends to `/chat/completions`, go to `/responses` as well.
* **Nothing is stored server-side and no reasoning is replayed.** `store` and
  `include` are not sent; a reasoning item is replayed only when it carries
  encrypted content, which these responses do not ask for.
"""

from __future__ import annotations

import os
import random
import sys
import time

import openai as _openai

from .openai import OpenAIProvider, _responses_usage
from .rate_limit import retry_on_rate_limit

DEFAULT_BASE_URL = "https://api.meta.ai/v1"

# The request settings every Muse call goes out with.
TEMPERATURE = 1.0
TOP_P = 1.0
MAX_OUTPUT_TOKENS = 32000
REASONING_EFFORT = "low"

# Terminal events of a streamed response; each carries the full response.
_FINAL_EVENTS = {"response.completed", "response.incomplete"}

# Attempts at a call whose stream fails with `server_error`, and the cap on one
# backoff wait between them.
_SERVER_ERROR_ATTEMPTS = 5
_MAX_BACKOFF_S = 60.0


class MuseServerError(RuntimeError):
    """A stream that ended in `response.failed` with code `server_error`."""


class MuseProvider(OpenAIProvider):
    def __init__(
        self,
        model: str = "muse-spark-1.3-contributor",
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = MAX_OUTPUT_TOKENS,
        vision_model: str | None = None,
        reasoning_effort: str | None = REASONING_EFFORT,
    ):
        self._model = model
        self._vision_model = vision_model or model
        self._max_tokens = max_tokens
        self._reasoning_effort = reasoning_effort
        self._reasoning_by_call: dict[str, list[dict]] = {}
        self._base_url = base_url or os.getenv("MUSE_BASE_URL") or DEFAULT_BASE_URL
        self._client = _openai.OpenAI(
            base_url=self._base_url,
            api_key=api_key or os.environ["MUSE_API_KEY"],
            # A streamed medium-effort turn can reason for minutes before the
            # final event arrives.
            timeout=float(os.getenv("LLM_TIMEOUT_S", "600")),
        )

    @property
    def base_url(self) -> str:
        return self._base_url

    def _responses_create(self, **kwargs):
        """One streamed Responses call; returns the final response object."""

        def call():
            final = None
            stream = self._client.responses.create(
                stream=True, temperature=TEMPERATURE, top_p=TOP_P, **kwargs
            )
            for event in stream:
                if event.type in _FINAL_EVENTS:
                    final = event.response
                elif event.type == "response.failed":
                    err = event.response.error
                    cls = MuseServerError if err and err.code == "server_error" else RuntimeError
                    raise cls(f"Muse response failed: {err}")
                elif event.type == "error":
                    raise RuntimeError(f"Muse stream error: {event.message}")
            if final is None:
                raise RuntimeError("Muse stream ended without a final response")
            return final

        for attempt in range(1, _SERVER_ERROR_ATTEMPTS + 1):
            try:
                return retry_on_rate_limit(call)
            except MuseServerError as err:
                if attempt == _SERVER_ERROR_ATTEMPTS:
                    raise
                delay = min(_MAX_BACKOFF_S, 2.0**attempt) + random.uniform(0.0, 0.5)
                print(
                    f"[ServerError] {err} (attempt {attempt}/{_SERVER_ERROR_ATTEMPTS}) "
                    f"— waiting {delay:.1f}s, then retrying",
                    file=sys.stderr,
                )
                time.sleep(delay)
        raise AssertionError("unreachable")

    def _text_response(self, model: str, system_prompt: str, content) -> str:
        extra = {"reasoning": {"effort": self._reasoning_effort}} if self._reasoning_effort else {}
        resp = self._responses_create(
            model=model,
            instructions=system_prompt,
            input=[{"role": "user", "content": content}],
            max_output_tokens=self._max_tokens,
            **extra,
        )
        self.last_usage = _responses_usage(resp.usage)
        return resp.output_text or ""

    def response(self, system_prompt: str, user_message: str) -> str:
        self.last_usage = None
        return self._text_response(self._model, system_prompt, user_message)

    def vision_response(
        self,
        system_prompt: str,
        user_message: str,
        images: list[dict] | None = None,
    ) -> str:
        content: list[dict] = [{"type": "input_text", "text": user_message}]
        for img in images or []:
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{img['media_type']};base64,{img['data']}",
                }
            )
        return self._text_response(self._vision_model, system_prompt, content)
