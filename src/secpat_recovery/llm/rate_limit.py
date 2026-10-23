"""Wait out provider rate limits and continue.

A ReAct run resends its whole transcript each turn, so a long recovery run will
sometimes outrun the account's tokens-per-minute quota:

    429 ... Rate limit reached for gpt-4o ... on tokens per min (TPM):
    Limit 30000, Used 20425, Requested 11738. Please try again in 4.326s.

That is not a failure, only a "not yet": the window refills on its own. So the
request is simply reissued once the window has passed, and the run carries on.
The provider states how long to wait in the error itself, which is better
information than any backoff curve we could invent — the wait comes from that
hint when present, and falls back to exponential backoff when it is absent.
"""

from __future__ import annotations

import random
import re
import sys
import time

# Attempts before giving up. Generous: each wait is short and a run that dies
# 15 steps in has thrown away far more than the sleep would have cost.
_ATTEMPTS = 8

# Cap on a single fallback wait, when the error carries no hint.
_MAX_BACKOFF_S = 60.0

# "Please try again in 4.326s" / "in 1m2.5s" — the provider's own advice.
_RETRY_HINT = re.compile(r"try again in ((?:\d+m)?[\d.]+)s", re.I)


def _hinted_delay(err: Exception) -> float | None:
    """Seconds the provider asked us to wait, or None if it did not say."""
    m = _RETRY_HINT.search(str(err))
    if not m:
        return None
    raw = m.group(1)
    if "m" in raw:
        mins, _, secs = raw.partition("m")
        return float(mins) * 60 + float(secs or 0)
    return float(raw)


def retry_on_rate_limit(call, *, attempts: int = _ATTEMPTS, verbose: bool = True):
    """Run `call()`, waiting out any rate limit and retrying until it goes through."""
    from openai import APIConnectionError, RateLimitError

    for attempt in range(1, attempts + 1):
        try:
            return call()
        except (RateLimitError, APIConnectionError) as err:
            if attempt == attempts:
                raise
            # Wait the window out. A little padding past the stated time, since
            # the hint is computed server-side before the response reaches us.
            delay = _hinted_delay(err)
            if delay is None:
                delay = min(_MAX_BACKOFF_S, 2.0**attempt)
            delay += 1.0 + random.uniform(0.0, 0.5)
            if verbose:
                print(
                    f"[RateLimit] {type(err).__name__} (attempt {attempt}/{attempts}) "
                    f"— waiting {delay:.1f}s, then continuing",
                    file=sys.stderr,
                )
            time.sleep(delay)
    raise AssertionError("unreachable")


__all__ = ["retry_on_rate_limit"]
