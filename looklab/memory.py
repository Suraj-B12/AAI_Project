"""Topic 4: the long-term profile store, and extraction into it.

The checkpointer and the store do different jobs and it is worth being able to
state the difference precisely:

* the **checkpointer** persists graph state per ``thread_id``. Short-term
  memory, scoped to one conversation.
* the **store** is namespaced by ``user_id``, read at graph entry and written
  at graph exit regardless of which thread is running. Long-term memory,
  scoped to a person.

Extraction is regex-first and deterministic, for the same reason the router
is: it works with no model, no network and no API key, so Topic 4 demos even
if every model provider is unreachable.

The design document proposed a ``JsonStore(InMemoryStore)`` subclass that
overrides ``put()`` to persist. That does not work: ``BaseStore.put``
delegates to ``self.batch([PutOp(...)])``, so writes arriving through batch or
async paths bypass the override entirely. The profile would look correct for a
whole session and then vanish on restart -- failing in exactly the scenario
the subclass was written to prevent. This module uses the shipped
``SqliteStore`` instead (see ``persistence.py``).
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage

PROFILE_NAMESPACE = "profiles"

# Ordered, most-specific-first. Each pattern captures one field.
EXTRACTION_PATTERNS: list[tuple[str, str]] = [
    # Name: "I'm Suraj", "my name is Suraj", "call me Suraj".
    # Longest alternatives first -- regex alternation takes the first branch
    # that matches, so "my name'?s" placed before "my name is" would match
    # "my name" and then capture the word "is" as the name.
    (
        "name",
        r"\b(?:my name is|my name'?s|i am|i'?m|call me)\s+([A-Z][a-z]+|[a-z]{2,15})\b",
    ),
    # Camera: "I shoot on a Fuji X-T4", "shooting with a Sony A7III".
    # The trailing (?!...) stops the capture running into a following clause.
    (
        "camera",
        r"\bi\s+shoot(?:ing)?\s+(?:on|with|using)?\s*(?:an?\s+)?"
        r"((?:(?!\b(?:and|but|because|so|which|that)\b)[\w\-]+)"
        r"(?:\s+(?:(?!\b(?:and|but|because|so|which|that)\b)[\w\-]+)){0,2})",
    ),
    # Preference: "I like warm golden looks", "I prefer muted matte"
    (
        "preferred_look",
        r"\bi\s+(?:really\s+)?(?:like|love|prefer|am into|gravitate towards?)\s+"
        r"([\w\- ]{3,45}?)(?:\s*(?:looks?|grades?|tones?|edits?|styles?))?"
        r"(?=[.,;!?]|\s+(?:and|but|because|so)\b|$)",
    ),
    # Dislike: "I hate heavy teal shadows", "I don't like crushed blacks"
    (
        "dislikes",
        r"\bi\s+(?:really\s+)?(?:hate|dislike|don'?t like|can'?t stand|avoid)\s+"
        r"([\w\- ]{3,45}?)(?:\s*(?:looks?|grades?|tones?|edits?|styles?))?"
        r"(?=[.,;!?]|\s+(?:and|but|because|so)\b|$)",
    ),
    # Editor: "I use Lightroom", "I edit in Capture One"
    ("editor", r"\bi\s+(?:use|edit in|edit with|work in)\s+([A-Za-z][\w\- ]{2,25}?)(?=[.,;!?]|$)"),
]

# Words that are never a name, however well they fit the pattern.
_NAME_STOPWORDS = {
    "not", "just", "really", "trying", "going", "looking", "working", "using",
    "the", "a", "an", "sure", "here", "back", "done", "new", "sorry", "good",
    "fine", "ok", "okay", "still", "always", "never", "about", "into", "on",
}

_MAX_VALUE_LEN = 60


_TRAILING_JUNK = re.compile(
    r"\s+(?:and|but|because|so|then|with|which|that|too|also|as well)$", re.IGNORECASE
)


def _clean(value: str) -> str:
    value = re.sub(r"\s+", " ", (value or "")).strip(" .,;:!?-'\"")
    # A capture can run past its clause -- "a Fuji X-T4 and I like..." yields
    # "fuji x-t4 and". Strip dangling conjunctions until none remain.
    previous = None
    while previous != value:
        previous = value
        value = _TRAILING_JUNK.sub("", value).strip(" .,;:!?-")
    return value[:_MAX_VALUE_LEN]


def extract_profile_facts(text: str) -> dict[str, str]:
    """Pull profile facts out of a single user utterance. Deterministic."""
    if not text:
        return {}
    found: dict[str, str] = {}
    lowered = text.lower()

    for key, pattern in EXTRACTION_PATTERNS:
        # Names are matched case-sensitively first so "I'm Suraj" beats "I'm not".
        source = text if key == "name" else lowered
        match = re.search(pattern, source, flags=re.IGNORECASE)
        if not match:
            continue
        value = _clean(match.group(1))
        if not value:
            continue
        if key == "name":
            if value.lower() in _NAME_STOPWORDS or len(value) < 2:
                continue
            value = value[:1].upper() + value[1:]
        if key == "camera" and value.lower() in _NAME_STOPWORDS:
            continue
        found[key] = value
    return found


def last_human_text(messages: list) -> str:
    """The most recent human message's text.

    Deliberately not ``messages[-2]``: that is only the human turn if the
    responder wrote exactly one message, and any two-message turn would make
    the extractor learn the assistant's own words into long-term memory.
    """
    for message in reversed(messages or []):
        if isinstance(message, HumanMessage):
            from .context import _text_of

            return _text_of(message)
    return ""


def load_profile(store: Any, user_id: str) -> dict[str, Any]:
    """Read every fact stored for a user. Namespace: ``("profiles", user_id)``."""
    if store is None or not user_id:
        return {}
    try:
        items = store.search((PROFILE_NAMESPACE, str(user_id)))
    except Exception:
        return {}
    profile: dict[str, Any] = {}
    for item in items or []:
        value = item.value
        profile[item.key] = value.get("value") if isinstance(value, dict) else value
    return profile


def save_profile_facts(store: Any, user_id: str, facts: dict[str, str]) -> list[str]:
    """Write extracted facts; returns the keys actually written."""
    if store is None or not user_id or not facts:
        return []
    namespace = (PROFILE_NAMESPACE, str(user_id))
    written: list[str] = []
    for key, value in facts.items():
        try:
            store.put(namespace, key, {"value": value})
            written.append(key)
        except Exception:
            continue
    return written


def describe_profile(profile: dict[str, Any]) -> str:
    """A short prose summary, injected into the system prompt each turn."""
    if not profile:
        return ""
    order = ("name", "camera", "editor", "preferred_look", "dislikes")
    labels = {
        "name": "their name is",
        "camera": "they shoot on",
        "editor": "they edit in",
        "preferred_look": "they prefer",
        "dislikes": "they dislike",
    }
    parts = [
        f"{labels[key]} {profile[key]}"
        for key in order
        if profile.get(key)
    ]
    parts += [f"{k} is {v}" for k, v in profile.items() if k not in order and v]
    return "; ".join(parts)
