"""The colour-science glossary, and deciding what a free-form message wants.

Two jobs, both deterministic and both working with no model and no network:

1. **Look a term up.** ``knowledge.json`` holds fifteen definitions scraped
   from Wikipedia with their source URLs, revisions and CC BY-SA licence. If a
   message mentions one, the app can answer from a cited source instead of
   guessing or -- as it used to -- reciting a canned blurb.

2. **Classify the message.** The chat branch previously had three outcomes and
   everything else fell through to the same sentence: measured, fourteen out of
   fourteen ordinary questions got an identical non-answer. Classifying first
   means the app can tell the difference between a question it can answer from
   its own data, one that needs a model, and one it should decline.

Everything here is keyword and pattern matching. No model is consulted to
decide what kind of message something is, for the same reason the router does
not use one: the app has to behave sensibly with no API key at all.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Any

KNOWLEDGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge.json")


@lru_cache(maxsize=1)
def load_knowledge() -> dict[str, Any]:
    if not os.path.exists(KNOWLEDGE_PATH):
        return {"meta": {}, "terms": []}
    try:
        with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {"meta": {}, "terms": []}


# Extra ways people say the terms in knowledge.json. The glossary is titled
# formally ("CIELAB", "colour appearance model"); photographers type "lab
# colour" and "why does it look different on my phone".
ALIASES: dict[str, tuple[str, ...]] = {
    "CIELAB": ("cielab", "lab colour", "lab color", "l*a*b", "lab space"),
    "colour grading": ("colour grading", "color grading", "grading", "grade", "colour grade"),
    "white balance": ("white balance", "wb", "colour cast", "color cast", "cast"),
    "colour temperature": ("colour temperature", "color temperature", "kelvin",
                           "warm or cool", "temperature slider"),
    "chroma": ("chroma", "saturation", "vibrance", "colourfulness", "colorfulness"),
    "hue": ("hue", "hue shift", "hue wheel"),
    "circular mean": ("circular mean", "average hue", "averaging hues"),
    "split toning": ("split tone", "split toning", "split-tone", "cross process"),
    "gamma correction": ("gamma", "gamma correction", "linear light", "srgb curve"),
    "sRGB": ("srgb", "colour space", "color space", "colour profile", "color profile",
             "adobe rgb", "display p3", "rec709", "rec 709", "rec.709"),
    "tone curve": ("tone curve", "curves", "contrast curve", "s-curve", "s curve"),
    "clipping": ("clipping", "clipped", "blown", "blown out", "crushed", "banding"),
    "dynamic range": ("dynamic range", "latitude", "highlight recovery"),
    "colour appearance model": ("colour appearance", "color appearance", "perceptual",
                                "why does it look different"),
    "curse of dimensionality": ("curse of dimensionality", "high dimensional",
                                "too many features"),
}


def lookup(text: str, limit: int = 2) -> list[dict[str, Any]]:
    """Glossary entries whose term or alias appears in ``text``.

    Longest alias first, so "colour grading" is not beaten by "grade", and a
    match must fall on a word boundary so "cast" does not fire inside
    "broadcast".
    """
    if not text:
        return []
    lowered = text.lower()
    terms = {t["term"]: t for t in load_knowledge().get("terms", [])}

    scored: list[tuple[int, dict[str, Any]]] = []
    for term, entry in terms.items():
        candidates = (term.lower(),) + ALIASES.get(term, ())
        best = 0
        for alias in candidates:
            if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", lowered):
                best = max(best, len(alias))
        if best:
            scored.append((best, entry))

    scored.sort(key=lambda pair: -pair[0])
    return [entry for _, entry in scored[:limit]]


# --------------------------------------------------------------------------
# Message classification
# --------------------------------------------------------------------------

# Vocabulary that puts a question inside this app's subject area even when no
# glossary term is named: "what log can I shoot", "is my monitor calibrated".
DOMAIN_WORDS = (
    "colour", "color", "grade", "grading", "grading", "lut", "log", "slog", "s-log",
    "clog", "c-log", "v-log", "vlog", "hlg", "raw", "jpeg", "jpg", "tiff", "dng",
    "lightroom", "photoshop", "capture one", "davinci", "resolve", "premiere",
    "camera", "lens", "sensor", "iso", "aperture", "shutter", "exposure",
    "highlight", "shadow", "midtone", "black point", "white point", "histogram",
    "tone", "tonal", "contrast", "saturation", "vibrance", "hue", "tint",
    "temperature", "kelvin", "white balance", "monitor", "calibrat", "profile",
    "srgb", "rec709", "rec 709", "adobe rgb", "p3", "gamut", "bit depth",
    "photo", "photograph", "image", "edit", "editing", "preset", "film",
    "skin tone", "sky", "portrait", "landscape", "print", "paper", "export",
)

# Small talk. Answered briefly and warmly rather than with a capabilities dump.
SOCIAL_PATTERNS = (
    r"^\s*(hi|hey|hello|yo|hiya|howdy|good (morning|afternoon|evening))\b",
    r"^\s*(thanks|thank you|ta|cheers|nice|great|cool|awesome|perfect|lovely)\b",
    r"^\s*(bye|goodbye|see you|later|cya)\b",
    r"^\s*(ok|okay|sure|right|got it|alright|fine)\s*[.!]?\s*$",
)

# Questions about the assistant itself. Worth answering plainly rather than
# letting a model improvise a personality.
META_PATTERNS = (
    r"\bare you (an? )?(ai|bot|robot|human|real|chatgpt|gpt|gemini|llm)\b",
    r"\bare you (hallucinat|making (this|it) up|guessing|sure|certain)",
    r"\b(do|are) you (actually )?(know|understand|measure|see)\b.*\?",
    r"\bwhat (model|ai|llm) (are you|do you use)\b",
    r"\bwho (made|built|created) you\b",
    r"\bhow do you (know|work|do that)\b",
)

# Questions about the reference library itself, answerable from kb.json.
LIBRARY_PATTERNS = (
    r"\bhow many looks?\b",
    r"\bwhat looks?\b.*\b(have|know|support|offer)\b",
    r"\b(list|show me)\b.*\blooks?\b",
    r"\bwhat (styles?|presets?)\b.*\b(have|know|support)\b",
    r"\byour (library|looks|catalogue|catalog)\b",
)

QUESTION_HINTS = ("?", "what", "why", "how", "when", "which", "can ", "could ",
                  "should ", "is ", "are ", "does ", "do ", "explain", "tell me")


def _matches(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def classify_message(text: str) -> str:
    """What kind of free-form message is this?

    Returns one of: ``social``, ``meta``, ``library``, ``glossary``,
    ``domain_question``, ``out_of_scope``, ``unclear``.

    Order matters. Social and meta are checked first because "thanks, are you
    sure?" should not be answered with a definition of chroma.
    """
    text = (text or "").strip()
    if not text:
        return "unclear"

    if _matches(text, SOCIAL_PATTERNS) and len(text.split()) <= 6:
        return "social"
    if _matches(text, META_PATTERNS):
        return "meta"
    if _matches(text, LIBRARY_PATTERNS):
        return "library"

    lowered = text.lower()
    looks_like_question = any(h in lowered for h in QUESTION_HINTS)

    if lookup(text):
        return "glossary"
    if any(word in lowered for word in DOMAIN_WORDS):
        return "domain_question" if looks_like_question else "domain_statement"
    if looks_like_question:
        return "out_of_scope"
    return "unclear"


def render_entry(entry: dict[str, Any]) -> str:
    """One glossary entry as markdown, with its attribution."""
    return (
        f"**{entry['term']}** — {entry['summary']}\n\n"
        f"*Why it matters here:* {entry['why_it_matters']}\n\n"
        f"[Source: {entry['source_title']}]({entry['source_url']}) "
        f"({entry.get('licence', 'CC BY-SA 4.0')})"
    )


def look_labels() -> list[str]:
    """Labels of every reference look, for answering library questions."""
    from . import kb as kb_module

    return [look["label"] for look in kb_module.looks()]
