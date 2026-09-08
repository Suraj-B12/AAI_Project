"""Retrieval: look things up before answering, so answers can cite a source.

Why this exists
---------------
The app previously had two ways to answer a free-form question: a fifteen-term
local glossary, or a language model answering unaided. The second was marked
"not measured", which is honest, but honest-and-unverifiable is still weaker
than checkable. This module adds a middle path -- fetch real sources, hand
them to the model, and require the answer to come from them.

That is the difference between "the model believes X" and "this source says X,
here is the link". It does not make hallucination impossible; it makes it
*visible*, because a claim with no supporting passage has nowhere to hide.

Why Wikipedia specifically
--------------------------
It needs no API key, has a documented search endpoint, is stable, and is
already the project's glossary source, so the licensing story is unchanged
(CC BY-SA 4.0, attributed). No commercial search API is used because every one
of them needs a key, and a feature that only works when a key is present is
not a feature this project is willing to depend on.

The honest limitation, stated here rather than discovered later: Wikipedia
covers colour science well and specific camera specifications poorly. A
question like "what log profile does the a6700 shoot" will usually retrieve
nothing useful, and the app says so instead of dressing a guess in citations.
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

SEARCH_API = "https://en.wikipedia.org/w/api.php"
SUMMARY_API = "https://en.wikipedia.org/api/rest_v1/page/summary/"

USER_AGENT = (
    "LookLab/1.0 (student colour-grading project; "
    "https://github.com/Suraj-B12/AAI_Project; surajbrajesh@gmail.com)"
)

# Retrieval must never make a chat turn feel broken. If Wikipedia is slow, the
# app gives up and falls back rather than making the user wait.
TIMEOUT = 6.0
TOTAL_DEADLINE = 10.0
MAX_PASSAGE_CHARS = 900

# Words that make a search worse rather than better -- they are about the
# conversation, not the subject.
_STOP = {
    "what", "whats", "what's", "why", "how", "when", "which", "who", "can",
    "could", "should", "would", "is", "are", "was", "were", "do", "does",
    "did", "the", "a", "an", "my", "me", "i", "you", "your", "it", "its",
    "to", "of", "in", "on", "for", "and", "or", "but", "with", "about",
    "tell", "explain", "please", "help", "know", "mean", "means", "difference",
    "between", "vs", "versus", "any", "some", "this", "that", "there",
}

_cache: dict[str, list[dict[str, Any]]] = {}
_cache_lock = threading.Lock()
CACHE_MAX = 128


def _get(url: str, timeout: float = TIMEOUT) -> bytes | None:
    try:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


# Vocabulary that marks a passage as being about this subject at all. A
# retrieved article must contain at least one of these or it is discarded --
# without this gate, "what does a LUT do?" retrieved *Lut Desert* and *Lot in
# Islam*, and "why is my photo green?" retrieved the Windows XP wallpaper.
# Citing those would be worse than citing nothing: it dresses an irrelevant
# passage in the authority of a source link.
DOMAIN_VOCAB = (
    "colour", "color", "photograph", "photography", "camera", "image", "video",
    "film", "lens", "exposure", "gamma", "luminance", "brightness", "hue",
    "saturation", "chroma", "contrast", "pixel", "rgb", "cmyk", "srgb",
    "sensor", "lut", "lookup table", "grading", "grade", "tonal", "tone",
    "white balance", "kelvin", "dynamic range", "codec", "encoding", "display",
    "monitor", "print", "cinema", "footage", "shutter", "aperture", "iso",
    "light", "spectral", "perceptual", "gamut", "calibration",
)

# Appended to short or ambiguous queries so the search lands in this subject.
# "lut" alone is a desert in Iran; "lut colour video" is a lookup table.
DOMAIN_HINT = "colour photography video"


# Technical names in this field are written glued together in chat and spaced
# apart in an encyclopaedia: "rec709" is the article "Rec. 709". Measured, the
# glued form returns NOTHING -- not bad results, zero results -- while the
# split form returns Log profile, Rec. 2100, Hybrid log-gamma and Transfer
# functions in imaging, which is exactly the right cluster. The letter run must
# be at least two characters so a camera body like "a6700" is left alone.
_GLUED = re.compile(r"(?<![\w])([a-z]{2,})(\d{3,})(?![\w])")


def _keywords(text: str) -> list[str]:
    """Content words, with glued technical names split apart."""
    lowered = _GLUED.sub(r"\1 \2", (text or "").lower())
    words = re.findall(r"[\w'-]+", lowered)
    return [w for w in words if w not in _STOP and len(w) > 1]


def plain_query(text: str) -> str:
    """The question stripped to its content words, with no subject hint."""
    return " ".join(_keywords(text)[:8]) or (text or "").strip()[:60]


def build_query(text: str) -> str:
    """Turn a chat message into a search query.

    Strips the conversational scaffolding so "what's the difference between log
    and rec709" searches for "log rec 709" rather than for the word
    "difference", and adds a subject hint when what is left is short enough to
    be ambiguous.
    """
    kept = _keywords(text)
    query = plain_query(text)

    # Three words or fewer is not enough to disambiguate. Note this does NOT
    # skip the hint just because a domain word is present: "lut" is a domain
    # word AND a desert in Iran, and treating it as already-scoped is exactly
    # how the search ended up in Iran.
    if len(kept) <= 3:
        query = f"{query} {DOMAIN_HINT}".strip()
    return query


# How many DISTINCT domain words a passage must contain to count as on-topic.
#
# One is not enough, and the reason is instructive: "lut" is itself a domain
# word, so *Lut Desert* passed a one-hit gate on its title alone. A genuine
# article about colour or imaging uses several of these words in its opening
# paragraph; an article about a salt flat in Iran uses exactly one, by
# coincidence of spelling.
MIN_DOMAIN_HITS = 2


def domain_hits(text: str) -> set[str]:
    """Distinct domain words appearing in ``text``, matched on word boundaries."""
    lowered = (text or "").lower()
    return {
        word
        for word in DOMAIN_VOCAB
        if re.search(rf"(?<!\w){re.escape(word)}(?!\w)", lowered)
    }


def is_relevant(entry: dict[str, Any]) -> bool:
    """Does this passage actually concern colour, imaging or photography?"""
    haystack = f"{entry.get('title', '')} {entry.get('passage', '')}"
    return len(domain_hits(haystack)) >= MIN_DOMAIN_HITS


def search(query: str, limit: int = 3) -> list[str]:
    """Article titles matching a query. Empty list on any failure."""
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": str(limit),
            "format": "json",
            "formatversion": "2",
        }
    )
    raw = _get(f"{SEARCH_API}?{params}")
    if not raw:
        return []
    try:
        data = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []
    return [hit["title"] for hit in data.get("query", {}).get("search", [])]


def summarise(title: str) -> dict[str, Any] | None:
    """The lead paragraph of an article, with its URL and revision."""
    raw = _get(SUMMARY_API + urllib.parse.quote(title, safe=""))
    if not raw:
        return None
    try:
        data = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    extract = (data.get("extract") or "").strip()
    if not extract:
        return None
    # A disambiguation page is a list of links, not an answer.
    if data.get("type") == "disambiguation":
        return None
    return {
        "title": data.get("title", title),
        "passage": extract[:MAX_PASSAGE_CHARS],
        "url": (data.get("content_urls", {}).get("desktop", {}) or {}).get(
            "page", f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title)}"
        ),
        "revision": data.get("revision"),
        "licence": "CC BY-SA 4.0",
    }


def candidate_titles(text: str, limit: int = 6) -> list[str]:
    """Article titles from both phrasings of the question, best first.

    The subject hint that rescues "lut" also drags other questions away from
    the article that answers them: hinted, "how does white balance work"
    returns *Color photography*; unhinted it returns *Color balance*, which is
    the actual answer. Neither phrasing dominates, so both are searched and
    the results merged. The relevance gate downstream is what makes this safe
    -- the unhinted query's junk (*Lut Desert*, *Work-life balance*) is
    discarded there rather than being trusted here.
    """
    hinted = build_query(text)
    plain = plain_query(text)
    lists = [search(q, limit=limit) for q in dict.fromkeys([hinted, plain]) if q]

    # Interleaved, not concatenated. Concatenating lets the hinted query's
    # results fill the caller's quota before the plain query is reached, which
    # is the same as not running it: "how does white balance work" kept
    # returning *Color photography* while *Color balance* sat unread at the top
    # of the other list.
    ordered: list[str] = []
    seen: set[str] = set()
    for rank in range(max((len(x) for x in lists), default=0)):
        for titles in lists:
            if rank < len(titles) and titles[rank] not in seen:
                seen.add(titles[rank])
                ordered.append(titles[rank])
    return ordered


def retrieve(text: str, limit: int = 3) -> list[dict[str, Any]]:
    """Sources relevant to a question. Cached, deadline-bounded, never raises.

    Returns an empty list when nothing useful is found, which the caller must
    treat as "I could not verify this" rather than quietly answering anyway.
    """
    query = build_query(text)
    if not query:
        return []

    with _cache_lock:
        if query in _cache:
            return _cache[query]

    started = time.monotonic()
    # Over-fetch, because the relevance gate discards some.
    titles = candidate_titles(text, limit=limit + 3)
    sources: list[dict[str, Any]] = []
    for title in titles:
        if time.monotonic() - started > TOTAL_DEADLINE or len(sources) >= limit:
            break
        entry = summarise(title)
        if entry and is_relevant(entry):
            sources.append(entry)

    with _cache_lock:
        if len(_cache) >= CACHE_MAX:
            _cache.clear()
        _cache[query] = sources
    return sources


def format_sources(sources: list[dict[str, Any]]) -> str:
    """The passages, as they are handed to the model."""
    return "\n\n".join(
        f"[{i}] {s['title']}\n{s['passage']}" for i, s in enumerate(sources, 1)
    )


def format_citations(sources: list[dict[str, Any]]) -> str:
    """The reading list shown to the user underneath an answer."""
    lines = [f"[{i}] [{s['title']}]({s['url']})" for i, s in enumerate(sources, 1)]
    return "**Sources:** " + " · ".join(lines) + "  \n*Wikipedia, CC BY-SA 4.0.*"


def reset_cache() -> None:
    with _cache_lock:
        _cache.clear()
