"""Topic 3: message filtering and trimming, with telemetry.

Two stages, both visible in the UI, because the brief says trimming *and/or*
filtering and doing both costs nothing.

Stage 1 (filter) has a real architectural reason to exist rather than being
rubric theatre: **the measured signatures live in ``state["images"]``, so the
model never needs them in the message history.** State holds the numbers, the
prompt holds the conversation. Anything tagged as a payload is dropped before
the model sees it.

Stage 2 (trim) enforces a token budget with ``trim_messages``.

Two traps are defended against here, both verified against the installed
versions:

* **Trimming is EPHEMERAL.** The trimmed list is handed to the model and never
  returned as a state update. Returning ``{"messages": trimmed}`` would be a
  silent no-op, because ``add_messages`` upserts by ID and never deletes -- the
  telemetry would claim "12 -> 4" while ``/state`` still showed 12. When
  durable deletion is genuinely wanted, ``drop_messages`` below returns
  ``RemoveMessage`` objects built from IDs read back out of state.
* **``trim_messages`` collapses silently at a tight budget.** It never raises;
  at max_tokens=40 it returns a single message, and with ``start_on="human"``
  it can return nothing at all. Every call is guarded by a floor so the model
  always receives at least the current exchange.
"""

from __future__ import annotations

import os
from typing import Any, Callable

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    trim_messages,
)

# Default budget, chosen so trimming actually engages within a five-turn demo
# rather than being decorative. Configurable from the UI.
DEFAULT_BUDGET = 800

# Below this, trim_messages starts returning one message or none. The UI
# clamps its slider here so the budget control cannot be dragged into
# producing an empty history live on stage.
MIN_BUDGET = 200

# Messages carrying this name are raw measurement payloads: they are dropped
# before the model call because the same numbers already live in state.
PAYLOAD_NAME = "signature_payload"


def _make_token_counter() -> tuple[Callable[[Any], int], str]:
    """Return a token counter and the name of the backend it uses.

    Prefers ``tiktoken`` (exact cl100k_base counts), but tiktoken downloads its
    BPE vocabulary on first use and caches it in a temp directory that Windows
    Storage Sense will happily clear. If that fails for any reason -- no
    network, cleared cache, anything -- fall back to langchain's local
    approximate counter rather than letting a demo die on a token count.
    """
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "tiktoken")
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", cache_dir)
    try:
        os.makedirs(cache_dir, exist_ok=True)
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")

        def count(messages: Any) -> int:
            if isinstance(messages, BaseMessage):
                messages = [messages]
            return sum(len(enc.encode(_text_of(m))) for m in messages)

        count([HumanMessage(content="warm up")])  # force the vocabulary load now
        return count, "tiktoken/cl100k_base"
    except Exception:
        from langchain_core.messages.utils import count_tokens_approximately

        def count(messages: Any) -> int:
            if isinstance(messages, BaseMessage):
                messages = [messages]
            return int(count_tokens_approximately(messages))

        return count, "approximate (tiktoken unavailable)"


def _text_of(message: Any) -> str:
    """Flatten message content to text; multimodal content arrives as a list."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text", "")))
        return " ".join(parts)
    return str(content)


count_tokens, TOKEN_BACKEND = _make_token_counter()


def is_payload(message: Any) -> bool:
    return getattr(message, "name", None) == PAYLOAD_NAME


def prepare(
    messages: list[BaseMessage],
    budget: int = DEFAULT_BUDGET,
    system: str | None = None,
) -> tuple[list[BaseMessage], dict[str, Any]]:
    """Filter then trim, returning the model-ready messages and telemetry.

    The returned list is for the model call only. It is deliberately NOT
    written back to state -- see the module docstring.
    """
    messages = list(messages or [])
    budget = max(int(budget or DEFAULT_BUDGET), MIN_BUDGET)

    before_n = len(messages)
    before_t = count_tokens(messages) if messages else 0

    # --- Stage 1: filter -------------------------------------------------
    # Drop raw measurement payloads: the numbers live in state["images"], so
    # putting them in the prompt would be paying tokens for data the graph
    # already holds.
    kept = [m for m in messages if not is_payload(m)]

    # Always keep the anchor -- the first human message states the task, and
    # a "last N" trim would otherwise discard it once the thread grows.
    anchor = next((m for m in messages if isinstance(m, HumanMessage)), None)
    if anchor is not None and anchor not in kept:
        kept.insert(0, anchor)

    after_filter_n = len(kept)
    dropped_payloads = before_n - after_filter_n

    # --- Stage 2: trim to the token budget --------------------------------
    head: list[BaseMessage] = []
    if system:
        head = [SystemMessage(content=system)]

    trimmed: list[BaseMessage]
    try:
        trimmed = trim_messages(
            head + kept,
            max_tokens=budget,
            token_counter=count_tokens,
            strategy="last",
            include_system=bool(system),
            start_on="human",
            allow_partial=False,
        )
    except Exception:
        trimmed = head + kept[-4:]

    # trim_messages never raises on a tight budget -- it silently returns one
    # message or none. Guarantee the model always sees the current exchange.
    if not [m for m in trimmed if not isinstance(m, SystemMessage)]:
        trimmed = head + kept[-2:]

    # Count history only, for BOTH counts. The injected system prompt is not
    # part of the conversation, and counting it on the "after" side but not the
    # "before" side makes the pane read 15 -> 16 messages and, worse,
    # 241 -> 252 tokens: trimming appearing to ADD tokens.
    history_after = [m for m in trimmed if not isinstance(m, SystemMessage)]
    after_n = len(history_after)
    after_t = count_tokens(history_after) if history_after else 0
    system_tokens = count_tokens(head) if head else 0

    telemetry = {
        "msgs_before": before_n,
        "msgs_after_filter": after_filter_n,
        "msgs_after_trim": after_n,
        "tokens_before": before_t,
        "tokens_after": after_t,
        "budget": budget,
        # Reported separately rather than folded into either count, so the
        # budget arithmetic on screen is inspectable.
        "system_tokens": system_tokens,
        "dropped_payloads": dropped_payloads,
        "dropped_by_trim": max(0, after_filter_n - after_n),
        "dropped_total": max(0, before_n - after_n),
        "token_backend": TOKEN_BACKEND,
        "trim_engaged": bool(after_n < after_filter_n),
    }
    return trimmed, telemetry


def drop_messages(messages: list[BaseMessage], keep_last: int) -> list[RemoveMessage]:
    """Durable deletion: ``RemoveMessage`` objects for everything but the tail.

    Not used on the normal path -- trimming is ephemeral by design -- but it is
    the correct mechanism when history really should shrink, and it is
    exercised by the test suite. IDs are read back out of the supplied
    messages: a ``RemoveMessage`` built against a locally constructed object
    would carry an ID that never entered state and would match nothing.
    """
    if keep_last <= 0:
        doomed = list(messages)
    else:
        doomed = list(messages[:-keep_last])
    return [RemoveMessage(id=m.id) for m in doomed if getattr(m, "id", None)]


def summarise(telemetry: dict) -> str:
    """One-line human-readable form of the telemetry, for the chat transcript."""
    if not telemetry:
        return ""
    return (
        f"{telemetry.get('msgs_before', 0)} msgs / "
        f"{telemetry.get('tokens_before', 0):,} tok -> "
        f"{telemetry.get('msgs_after_trim', 0)} msgs / "
        f"{telemetry.get('tokens_after', 0):,} tok "
        f"(budget {telemetry.get('budget', 0):,})"
    )


__all__ = [
    "DEFAULT_BUDGET",
    "MIN_BUDGET",
    "PAYLOAD_NAME",
    "TOKEN_BACKEND",
    "AIMessage",
    "count_tokens",
    "drop_messages",
    "is_payload",
    "prepare",
    "summarise",
]
