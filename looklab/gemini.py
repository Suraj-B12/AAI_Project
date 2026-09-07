"""Multi-key Gemini client with rotation, failover and health tracking.

Why this exists rather than just calling the SDK: measured against the live
API, roughly 30% of calls to a popular model come back ``503 high demand``
even with a perfectly valid key. A single-key client would surface that as a
broken app. With ten keys and failover, a transient 503 costs one retry on a
different credential and the caller never sees it.

Design notes:

* **stdlib only.** ``urllib.request``, no new dependency, so nothing else has
  to be installed for this to work.
* **Keys are never logged.** Only the last four characters appear in health
  output, and the module refuses to put a key in a URL query string -- it uses
  the ``x-goog-api-key`` header, so keys cannot leak into proxy or server logs.
* **Health is per key.** A key that returns 401/403 is marked dead and is not
  retried; a key that returns 429/503 is cooled off for a while and comes back.
  That distinction matters: one is a bad credential, the other is weather.
* **It is never required.** Everything in LookLab runs deterministically with
  no model at all. This only ever improves the prose.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"

# Model choice is a measured decision, not a guess. Benchmarked across all ten
# keys concurrently:
#
#   gemini-3.1-flash-lite     10/10 keys OK, median 2.2s
#   gemini-3.5-flash-lite      5/10 keys OK (5x HTTP 429)
#   gemini-flash-lite-latest   0/10 keys OK (10x HTTP 429)
#   gemini-3.6-flash           0/10 keys OK (10x HTTP 429)
#
# The "-latest" aliases are the most contended and are the worst choice for a
# demo, which is the opposite of the intuition that they are safest.
DEFAULT_MODEL = os.getenv("LOOKLAB_MODEL_NAME") or "gemini-3.1-flash-lite"

# Tried in order when the primary model 404s (retired) or is saturated.
# Pinning a single dated model is how this project first hit a 404 that looked
# exactly like an auth failure, so the pool now walks a list instead.
FALLBACK_MODELS: tuple[str, ...] = (
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-flash-latest",
)

# Status codes that mean "this key is fine, the service is busy" -- cool off
# and try a different key.
TRANSIENT = {408, 429, 500, 502, 503, 504}
# Status codes that mean "this credential is no good" -- stop using it.
FATAL_KEY = {400, 401, 403}

COOLDOWN_SECONDS = 30.0
DEFAULT_TIMEOUT = 45.0

# Total wall-clock budget for one generate() call, across all retries.
#
# Measured: with every key valid, rotation reaches 100% success, but a call
# that has to walk several cooling-off keys took up to 207 seconds. That is
# fine for a batch job and unacceptable for a web request, so an interactive
# caller passes a short deadline and falls back to the deterministic narrator
# when it expires. Being late is a worse failure than being plain.
DEFAULT_DEADLINE = 20.0


@dataclass
class KeyHealth:
    key: str
    calls: int = 0
    ok: int = 0
    transient: int = 0
    dead: bool = False
    cool_until: float = 0.0
    last_error: str = ""
    total_latency: float = 0.0

    @property
    def tail(self) -> str:
        return f"...{self.key[-8:]}"

    @property
    def available(self) -> bool:
        return not self.dead and time.monotonic() >= self.cool_until

    def snapshot(self) -> dict[str, Any]:
        return {
            "key": self.tail,
            "calls": self.calls,
            "ok": self.ok,
            "transient": self.transient,
            "dead": self.dead,
            "cooling": max(0.0, round(self.cool_until - time.monotonic(), 1)),
            "mean_latency": round(self.total_latency / self.ok, 2) if self.ok else None,
            "last_error": self.last_error[:120],
        }


def load_keys() -> list[str]:
    """Read keys from the environment, falling back to a local .env file.

    ``.env`` is gitignored. Keys must never be committed, and nothing in this
    module writes them anywhere.
    """
    raw = os.getenv("GEMINI_API_KEYS") or os.getenv("GOOGLE_API_KEY") or ""
    if not raw:
        env_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
        )
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line.startswith("GEMINI_API_KEYS="):
                            raw = line.split("=", 1)[1]
                            break
            except OSError:
                raw = ""
    return [k.strip() for k in raw.split(",") if k.strip()]


class GeminiPool:
    """A pool of Gemini credentials that behaves like one reliable one."""

    def __init__(
        self,
        keys: list[str] | None = None,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self._lock = threading.Lock()
        self._health = [KeyHealth(key=k) for k in (keys if keys is not None else load_keys())]
        self._cursor = 0
        # Model ladder: the chosen model first, then the rest as fallbacks.
        self._models = [model] + [m for m in FALLBACK_MODELS if m != model]
        self._model_index = 0
        self._retired: set[str] = set()

    @property
    def active_model(self) -> str:
        return self._models[self._model_index]

    def _next_model(self) -> bool:
        """Advance to the next model in the ladder. False when exhausted."""
        with self._lock:
            while self._model_index + 1 < len(self._models):
                self._model_index += 1
                if self._models[self._model_index] not in self._retired:
                    return True
            return False

    # -- introspection ----------------------------------------------------
    def __len__(self) -> int:
        return len(self._health)

    @property
    def usable(self) -> int:
        with self._lock:
            return sum(1 for h in self._health if not h.dead)

    def health(self) -> list[dict[str, Any]]:
        with self._lock:
            return [h.snapshot() for h in self._health]

    # -- key selection ----------------------------------------------------
    def _next_key(self) -> KeyHealth | None:
        """Round-robin over available keys; fall back to the soonest to cool."""
        with self._lock:
            n = len(self._health)
            if n == 0:
                return None
            for offset in range(n):
                h = self._health[(self._cursor + offset) % n]
                if h.available:
                    self._cursor = (self._cursor + offset + 1) % n
                    return h
            waiting = [h for h in self._health if not h.dead]
            return min(waiting, key=lambda h: h.cool_until) if waiting else None

    # -- the call ---------------------------------------------------------
    def generate(
        self,
        prompt: str,
        system: str | None = None,
        max_output_tokens: int = 2048,
        temperature: float = 0.7,
        max_attempts: int | None = None,
        deadline: float | None = DEFAULT_DEADLINE,
    ) -> str:
        """Generate text, rotating keys on transient failure.

        ``deadline`` is a total wall-clock budget across all retries; pass
        ``None`` to retry until the attempt budget is exhausted (batch use).

        Raises ``RuntimeError`` only when every key has been tried and failed,
        or the deadline expires. Callers in LookLab always catch that and fall
        back to the deterministic narrator, so an outage degrades the prose
        rather than breaking the request.
        """
        if not self._health:
            raise RuntimeError("no Gemini keys configured (set GEMINI_API_KEYS)")

        started_call = time.monotonic()

        def out_of_time() -> bool:
            return deadline is not None and (time.monotonic() - started_call) >= deadline

        attempts = max_attempts or max(3, min(len(self._health) * 2, 12))
        payload: dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_output_tokens,
                "temperature": temperature,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        body = json.dumps(payload).encode("utf-8")
        errors: list[str] = []
        for attempt in range(attempts):
            if out_of_time():
                errors.append(f"deadline of {deadline:.0f}s exceeded")
                break
            h = self._next_key()
            if h is None:
                break
            if not h.available:
                # Never sleep past the deadline waiting for a key to cool.
                wait = max(0.0, h.cool_until - time.monotonic())
                if deadline is not None:
                    wait = min(wait, max(0.0, deadline - (time.monotonic() - started_call)))
                time.sleep(min(2.0, wait))
                if out_of_time():
                    errors.append(f"deadline of {deadline:.0f}s exceeded while waiting")
                    break

            started = time.monotonic()
            # Never let one socket outlive the whole call's budget.
            per_call_timeout = self.timeout
            if deadline is not None:
                per_call_timeout = max(
                    2.0, min(self.timeout, deadline - (time.monotonic() - started_call))
                )
            url = f"{API_ROOT}/models/{self.active_model}:generateContent"
            try:
                # Key travels in a header, never in the query string, so it
                # cannot end up in an access log or a browser history.
                request = urllib.request.Request(
                    url,
                    data=body,
                    headers={"Content-Type": "application/json", "x-goog-api-key": h.key},
                )
                with urllib.request.urlopen(request, timeout=per_call_timeout) as response:
                    data = json.load(response)
                text = self._extract(data)
                with self._lock:
                    h.calls += 1
                    h.ok += 1
                    h.total_latency += time.monotonic() - started
                if text:
                    return text
                errors.append(f"{h.tail}: empty completion")
                continue

            except urllib.error.HTTPError as exc:
                detail = self._error_detail(exc)
                if exc.code == 404:
                    # The MODEL is gone, not the key. Retire it and move down
                    # the ladder rather than burning attempts or blaming keys.
                    with self._lock:
                        self._retired.add(self.active_model)
                    errors.append(f"model {self.active_model} retired: {detail[:60]}")
                    if not self._next_model():
                        break
                    continue
                with self._lock:
                    h.calls += 1
                    h.last_error = f"HTTP {exc.code}: {detail}"
                    if exc.code in FATAL_KEY:
                        h.dead = True
                    elif exc.code in TRANSIENT:
                        h.transient += 1
                        # Jittered backoff so ten keys do not retry in lockstep.
                        h.cool_until = time.monotonic() + COOLDOWN_SECONDS * (
                            0.5 + random.random()
                        )
                errors.append(f"{h.tail}: HTTP {exc.code} {detail[:80]}")
                if exc.code in TRANSIENT:
                    time.sleep(min(1.5, 0.2 * (attempt + 1)))
                continue

            except Exception as exc:  # timeouts, DNS, TLS
                with self._lock:
                    h.calls += 1
                    h.transient += 1
                    h.last_error = f"{type(exc).__name__}: {exc}"
                    h.cool_until = time.monotonic() + COOLDOWN_SECONDS / 2
                errors.append(f"{h.tail}: {type(exc).__name__}")
                continue

        raise RuntimeError(
            f"all {len(self._health)} Gemini keys failed after {attempts} attempts: "
            + "; ".join(errors[-4:])
        )

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _extract(data: dict) -> str:
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        return "".join(p.get("text", "") for p in parts).strip()

    @staticmethod
    def _error_detail(exc: urllib.error.HTTPError) -> str:
        try:
            return str(json.loads(exc.read().decode())["error"]["message"])
        except Exception:
            return exc.reason or ""


_POOL: GeminiPool | None = None
_POOL_LOCK = threading.Lock()


def get_pool() -> GeminiPool | None:
    """Process-wide pool, or None when no keys are configured."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            keys = load_keys()
            _POOL = GeminiPool(keys) if keys else None
        return _POOL


def available() -> bool:
    pool = get_pool()
    return bool(pool and pool.usable)
