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
    # Feelings and states: "I'm confused" stored the name "Confused", and the
    # app then greeted them as Confused in every later reply.
    "confused", "lost", "stuck", "tired", "curious", "interested", "happy",
    "sad", "annoyed", "frustrated", "unsure", "certain", "wondering",
    "asking", "hoping", "guessing", "afraid", "worried", "keen", "ready",
    "back", "here", "there", "close", "done", "finished", "learning",
    "beginner", "new", "amateur", "pro", "professional", "student",
}

_MAX_VALUE_LEN = 60

# Things that follow "I shoot" but are not cameras. Without this,
# "I shoot weddings on a Nikon Z6" stores the camera as "weddings on a".
_NOT_A_CAMERA = {
    "weddings", "portraits", "landscapes", "events", "film", "digital", "raw",
    "jpeg", "a lot", "everything", "mostly", "professionally", "people",
    "products", "sports", "street", "nature", "wildlife", "concerts",
    # Formats and modes, not cameras: "I shoot log" stored camera="log", and
    # every later reply then told the user they shoot on a "log".
    "log", "slog", "s-log", "clog", "c-log", "vlog", "v-log", "hlg", "flat",
    "raw", "jpeg", "jpg", "video", "stills", "handheld", "manual", "auto",
    "wide", "tele", "macro", "prime", "zoom", "anamorphic",
}

# An explicit instruction to remember something. These are checked before the
# implicit patterns and stored verbatim, because when a user says "remember
# that I hate crushed blacks" they have told you exactly what to keep -- there
# is nothing to infer, and guessing a category would lose the point.
EXPLICIT_SAVE_PATTERNS: list[str] = [
    r"\b(?:please\s+)?remember\s+(?:that\s+|this[:,]?\s+|me\s+)?(.+)",
    r"\b(?:save|store|keep)\s+(?:this|that|it)?\s*(?:to|in|into)?\s*"
    r"(?:your\s+)?(?:long[- ]?term\s+)?memory[:,]?\s*(.+)",
    r"\b(?:save|store|note|keep)\s+(?:this|that)[:,]\s*(.+)",
    r"\bnote\s+(?:that\s+|down\s+)(.+)",
    r"\bmake\s+a\s+note\s+(?:that\s+|of\s+)?(.+)",
    r"\bdon'?t\s+forget\s+(?:that\s+)?(.+)",
    r"\bkeep\s+in\s+mind\s+(?:that\s+)?(.+)",
]

# How many free-form notes to keep per user. Long-term memory that grows
# without limit is a storage leak, and the oldest note is the least likely to
# still be true.
MAX_NOTES = 10
_MAX_NOTE_LEN = 240


_TRAILING_JUNK = re.compile(
    r"\s+(?:and|but|because|so|then|with|which|that|too|also|as well"
    r"|on|in|at|of|for|to|a|an|the|using)$",
    re.IGNORECASE,
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


# Words that can follow "remember" without being the thing to remember.
# "I shoot on a Sony a6700, remember that too" puts the fact BEFORE the verb,
# and naively capturing what follows stored the note "too".
_FILLER_AFTER_TRIGGER = {
    "that", "this", "it", "too", "also", "as well", "that too", "this too",
    "them", "these", "those", "please", "ok", "okay", "yeah", "yes",
    "that as well", "this as well", "for me", "for later", "next time",
}

# A trailing imperative with nothing after it: "..., remember." / "... remember?"
_TRAILING_TRIGGER = re.compile(
    r"[,;.\s]*(?:and\s+)?(?:please\s+)?"
    r"(?:remember|note|save|store|keep)(?:\s+(?:that|this|it|too|also|as\s+well)){0,3}"
    r"\s*[.!?]*\s*$",
    re.IGNORECASE,
)


# A question is not a fact to store. "don't forget what a LUT does" is asking,
# not telling, and storing it produced "Saved - I will remember what a LUT
# does" followed by no answer at all.
_LOOKS_LIKE_QUESTION = re.compile(
    r"^\s*(?:what|why|how|when|where|which|who|whose|can|could|should"
    r"|would|is|are|was|were|do|does|did|will|shall|may|might)\b",
    re.IGNORECASE,
)


def _is_filler(note: str) -> bool:
    return note.strip().lower().strip(" .,;:!?") in _FILLER_AFTER_TRIGGER


def _is_question(note: str) -> bool:
    note = note.strip()
    return bool(note.endswith("?") or _LOOKS_LIKE_QUESTION.match(note))


# Asking the app to forget. Memory a user cannot remove is not memory they
# control, and the app previously had a write path and a read path but no way
# back out -- "delete my profile" was even caught by the recall cues and
# answered by listing the profile it was being asked to erase.
FORGET_ALL_PATTERNS = (
    r"\b(?:forget|delete|clear|wipe|erase|reset|remove)\s+(?:my\s+|the\s+|all\s+(?:my\s+)?)?"
    r"(?:profile|memory|memories|data|everything|it all|all of it|what you know)",
    r"\bforget\s+(?:me|everything|it all|all of it)\b",
    r"\bstart\s+(?:over|fresh|again)\s+(?:with\s+)?(?:my\s+)?(?:profile|memory)\b",
)

# Forgetting one thing: "forget my camera", "forget that I like warm tones".
FORGET_ONE_PATTERNS = (
    r"\b(?:forget|delete|remove|drop|unset)\s+(?:that\s+|my\s+|the\s+)?(.+)",
    r"\bi\s+(?:never|didn'?t)\s+(?:said|say|told you)\s+(.+)",
    r"\bthat'?s\s+(?:not\s+right|wrong)\s*[,.]?\s*(?:i\s+)?(.+)",
)

# Which stored field a phrase refers to, so "forget my camera" clears `camera`.
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "name": ("name", "my name", "who i am"),
    "camera": ("camera", "my camera", "body", "gear"),
    "editor": ("editor", "software", "app"),
    "preferred_look": ("preference", "preferences", "favourite", "favorite",
                       "preferred look", "what i like", "the look i like", "likes"),
    "dislikes": ("dislike", "dislikes", "what i hate", "what i dislike"),
    "notes": ("notes", "note"),
}


def forget_request(text: str) -> tuple[str, str | None] | None:
    """Is this asking the app to forget something?

    Returns ``("all", None)`` to clear the profile, ``("field", name)`` to
    clear one field, ``("note", phrase)`` to drop a matching note, or None.
    """
    if not text:
        return None
    for pattern in FORGET_ALL_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return ("all", None)

    for pattern in FORGET_ONE_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        target = _clean(match.group(1)).lower()
        if not target:
            continue
        for field, aliases in FIELD_ALIASES.items():
            if any(re.search(rf"(?<!\w){re.escape(a)}(?!\w)", target) for a in aliases):
                return ("field", field)
        return ("note", target)
    return None


def preview_forget(profile: dict, what: str, target: str | None) -> dict:
    """The profile as it will look after ``forget``, without touching the store.

    ``respond`` runs before ``save_profile`` in the graph, so at narration time
    the deletion has not happened yet. Confirming "done" while listing the
    thing that was just deleted is worse than not confirming at all, so the
    narrator is handed this projection instead of the current state.
    """
    profile = dict(profile or {})
    if what == "all":
        return {}
    if what == "field":
        profile.pop(str(target), None)
        return profile

    notes = profile.get("notes")
    if isinstance(notes, list) and notes and target:
        keep = [n for n in notes if target not in str(n).lower()]
        if len(keep) != len(notes):
            if keep:
                profile["notes"] = keep
            else:
                profile.pop("notes", None)
            return profile
    for key, value in list(profile.items()):
        if key != "notes" and target and target in str(value).lower():
            profile.pop(key, None)
            break
    return profile


def forget(store: Any, user_id: str, what: str, target: str | None) -> list[str]:
    """Remove stored facts. Returns what was actually removed."""
    if store is None or not user_id:
        return []
    namespace = (PROFILE_NAMESPACE, str(user_id))
    profile = load_profile(store, user_id)
    removed: list[str] = []

    def drop(key: str) -> None:
        try:
            store.delete(namespace, key)
            removed.append(key)
        except Exception:
            pass

    if what == "all":
        for key in list(profile):
            drop(key)
        return removed

    if what == "field":
        if profile.get(target):
            drop(str(target))
        return removed

    # A note: keep the ones that do not mention the phrase.
    notes = profile.get("notes")
    if isinstance(notes, list) and notes and target:
        keep = [n for n in notes if target not in str(n).lower()]
        if len(keep) != len(notes):
            try:
                if keep:
                    store.put(namespace, "notes", {"value": keep})
                else:
                    store.delete(namespace, "notes")
                removed.append("notes")
            except Exception:
                pass
    # A phrase that names no field and matches no note: try the field whose
    # stored VALUE contains it, so "forget that I like warm golden" works.
    if not removed:
        for key, value in profile.items():
            if key != "notes" and target and target in str(value).lower():
                drop(key)
                break
    return removed


def explicit_save_request(text: str) -> str | None:
    """The thing the user explicitly asked to be remembered, if any.

    Handles both word orders, because people use them interchangeably:

        "remember that I shoot on a Sony a6700"   -> the fact FOLLOWS the verb
        "I shoot on a Sony a6700, remember that"  -> the fact PRECEDES it

    The second form used to capture the word after the verb, so
    "..., remember that too" stored the note "too" and dropped the camera
    entirely. Returns None when this is not a save request, or when there is
    genuinely nothing to save ("remember that" on its own).
    """
    if not text:
        return None

    def clean(value: str) -> str:
        return re.sub(r"\s+", " ", value or "").strip(" .,;:!?-\"'")

    for pattern in EXPLICIT_SAVE_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        note = clean(match.group(1))
        if _is_question(note):
            # "remember what a LUT is" is a request for an answer, not a fact.
            return None
        if len(note) >= 3 and not _is_filler(note):
            return note[:_MAX_NOTE_LEN]
        # What followed the verb was filler, so the fact is what came before it.
        before = clean(_TRAILING_TRIGGER.sub("", text[: match.start()]))
        if len(before) >= 3:
            return before[:_MAX_NOTE_LEN]
        # "remember that" with nothing on either side -- nothing to store.
        return None

    # A trailing imperative the patterns above cannot match, because they all
    # require something after the verb: "I shoot on a Sony a6700, remember."
    trailing = _TRAILING_TRIGGER.search(text)
    if trailing and trailing.start() > 0:
        before = clean(text[: trailing.start()])
        if len(before) >= 3:
            return before[:_MAX_NOTE_LEN]
    return None


def extract_profile_facts(text: str) -> dict[str, str]:
    """Pull profile facts out of a single user utterance. Deterministic.

    An explicit "remember that ..." is handled first and its content is
    scanned for known fields, so "remember that I shoot on a Nikon Z6" both
    stores the sentence and fills in the camera field.
    """
    if not text:
        return {}
    found: dict[str, str] = {}

    explicit = explicit_save_request(text)
    if explicit:
        # Re-run the field patterns against the remembered clause only, so
        # "remember that I like warm tones" sets preferred_look rather than
        # matching something earlier in the sentence.
        found.update(_scan_fields(explicit))
        if not _note_is_redundant(explicit, found):
            found["_note"] = explicit
        return found

    found.update(_scan_fields(text))
    return found


# Words a note can contain without adding anything the structured fields do
# not already record. "im john" alongside name="John" is not a second fact.
_NOTE_FILLER = {
    "i", "im", "i'm", "am", "is", "are", "my", "me", "a", "an", "the", "on",
    "in", "with", "using", "use", "uses", "shoot", "shooting", "shoots",
    "edit", "editing", "edits", "like", "likes", "love", "prefer", "prefers",
    "hate", "hates", "dislike", "dislikes", "avoid", "name", "called", "and",
    "that", "this", "it", "to", "of", "for", "look", "looks", "tone", "tones",
    "grade", "grades", "style", "styles", "camera",
}


def _note_is_redundant(note: str, fields: dict[str, str]) -> bool:
    """Would storing this note just repeat the structured fields?

    "remember me, im john" already becomes name="John"; keeping "im john" as a
    separate note as well makes the recall listing read like the app is
    confused about what it knows.
    """
    if not fields:
        return False
    known = " ".join(str(v) for k, v in fields.items() if k != "_note").lower()
    words = [w for w in re.findall(r"[\w'-]+", note.lower()) if w not in _NOTE_FILLER]
    if not words:
        return True
    return all(word in known for word in words)


def _scan_fields(text: str) -> dict[str, str]:
    """Run the field patterns over one piece of text."""
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
        if key == "camera":
            head = value.lower().split()[0] if value.split() else ""
            if value.lower() in _NAME_STOPWORDS or head in _NOT_A_CAMERA:
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
    """Write extracted facts; returns the keys actually written.

    ``_note`` is treated separately: free-form things the user explicitly asked
    to be remembered are appended to a capped list rather than overwriting one
    another, because "remember I hate teal" and "remember I shoot weddings"
    are both worth keeping.
    """
    if store is None or not user_id or not facts:
        return []
    namespace = (PROFILE_NAMESPACE, str(user_id))
    written: list[str] = []

    for key, value in facts.items():
        if key == "_note":
            if _append_note(store, namespace, str(value)):
                written.append("notes")
            continue
        try:
            store.put(namespace, key, {"value": value})
            written.append(key)
        except Exception:
            continue
    return written


def _append_note(store: Any, namespace: tuple, note: str) -> bool:
    """Append to the capped note list. Duplicates are ignored."""
    try:
        existing = store.get(namespace, "notes")
        notes = list((existing.value or {}).get("value") or []) if existing else []
    except Exception:
        notes = []
    if any(note.lower() == str(n).lower() for n in notes):
        return True  # already known; treat as saved rather than duplicating
    notes.append(note)
    # Keep the most recent MAX_NOTES: an unbounded list is a storage leak.
    notes = notes[-MAX_NOTES:]
    try:
        store.put(namespace, "notes", {"value": notes})
        return True
    except Exception:
        return False


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
    notes = profile.get("notes")
    if isinstance(notes, list) and notes:
        parts.append("they asked you to remember: " + "; ".join(str(n) for n in notes))
    parts += [
        f"{k} is {v}"
        for k, v in profile.items()
        if k not in order and k != "notes" and v
    ]
    return "; ".join(parts)
