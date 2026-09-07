"""Narration. Deterministic by default; a real model is strictly optional.

The design goal is that the app still does real work if the model is removed
entirely. Routing, colour analysis, matching, the slider rules and profile
extraction are all deterministic -- the model only ever narrates numbers that
were computed before it was called.

``DemoChatModel`` is therefore not a stub or a fallback for a missing feature.
It is the default narrator, it needs no API key and no network, and every
sentence it emits is built from values the graph already holds. If
``LOOKLAB_MODEL`` names a provider and its key is present, a real model is
used for the prose instead -- but the same computed facts are handed to it,
and nothing downstream changes.
"""

from __future__ import annotations

import os
import threading
from typing import Any

from langchain_core.messages import AIMessage


def _fmt_amount(amount: Any) -> str:
    if isinstance(amount, (int, float)):
        return f"{amount:+.0f}"
    return str(amount)


def _bullet_steps(steps: list[dict]) -> str:
    """One line per step: what to move, then why, then the numbers.

    The numbers go last and in smaller words on purpose. A photographer needs
    to know which slider and which direction; the measurement is there to be
    checked, not to be waded through.
    """
    lines = []
    for i, step in enumerate(steps, 1):
        head = f"{i}. **{step['slider']} {_fmt_amount(step['amount'])}**"
        why = step.get("why") or ""
        detail = step.get("detail") or ""
        line = f"{head}\n   {why}"
        if detail:
            line += f"\n   _({detail})_"
        lines.append(line)
    return "\n".join(lines)


def _plain_distance(distance: float) -> str:
    """Turn the internal distance into something a person can act on."""
    if distance <= 0.35:
        return "very close already"
    if distance <= 0.8:
        return "fairly close"
    if distance <= 1.5:
        return "some way off"
    return "a long way off"


class DemoChatModel:
    """Deterministic narrator. Same inputs always produce the same words.

    Written to be read by a photographer, not by a colour scientist. Numbers
    appear where they help you check the advice, and the measurements are
    described in ordinary words -- "the dark areas lean blue" rather than
    "shadow hue 271 degrees, chroma 14.8".
    """

    name = "DemoChatModel (deterministic, no API key required)"
    is_deterministic = True

    def narrate(self, intent: str, facts: dict) -> str:
        handler = getattr(self, f"_say_{intent}", None)
        if handler is None:
            return self._say_chat(facts)
        return handler(facts)

    # -- shared -------------------------------------------------------------
    @staticmethod
    def _describe(sig: dict) -> str:
        """A short plain-language read of what an image looks like."""
        if not sig:
            return ""
        bits = []
        b = float(sig.get("b_global", 0.0))
        if b > 16:
            bits.append("warm overall")
        elif b < 2:
            bits.append("cool overall")
        else:
            bits.append("fairly neutral in warmth")

        contrast = float(sig.get("contrast", 0.0))
        if contrast > 78:
            bits.append("high contrast")
        elif contrast < 45:
            bits.append("low contrast, quite flat")
        else:
            bits.append("moderate contrast")

        chroma = float(sig.get("chroma", 0.0))
        if chroma > 32:
            bits.append("strong colour")
        elif chroma < 12:
            bits.append("muted colour")

        return ", ".join(bits)

    @staticmethod
    def _hsl_line(sig: dict) -> str:
        """Which colour families actually occupy the frame."""
        hsl = sig.get("hsl") or {}
        present = sorted(
            ((v.get("coverage", 0.0), k) for k, v in hsl.items() if v.get("coverage", 0) >= 0.05),
            reverse=True,
        )
        if not present:
            return ""
        named = ", ".join(f"{name} ({cov * 100:.0f}%)" for cov, name in present[:4])
        return f"Colour families in frame: {named}."

    # -- identify -----------------------------------------------------------
    def _say_identify(self, facts: dict) -> str:
        matches = facts.get("matches") or []
        if not matches:
            return (
                "I could not measure anything yet. Upload a photo, or just name a "
                "look you want to know about \u2014 try \"what is teal and orange?\""
            )
        best = matches[0]

        if best.get("from_profile"):
            profile = facts.get("profile") or {}
            known = []
            if profile.get("camera"):
                known.append(f"you shoot on a {profile['camera']}")
            if profile.get("preferred_look"):
                known.append(f"you like {profile['preferred_look']}")
            if profile.get("dislikes"):
                known.append(f"you would rather avoid {profile['dislikes']}")
            lead = (
                f"Going on what you have told me before \u2014 {', and '.join(known)} \u2014 "
                if known
                else "Based on what I have saved about you, "
            )
            lines = [lead + "here is what I would try:", ""]
            for match in matches:
                lines.append(f"- **{match['label']}** \u2014 {match['notes']}")
            lines.append(
                f"\nSay \"how do I get the {matches[0]['label'].lower()} look?\" "
                f"and I will give you the exact sliders."
            )
            return "\n".join(lines)

        sig = facts.get("signature") or {}
        lines = []
        if best.get("confident"):
            lines.append(
                f"This is closest to **{best['label']}**. "
                f"{best['notes']}"
            )
        else:
            lines.append(
                f"**I am not confident about this one.** The nearest thing in my "
                f"library is *{best['label']}*, but it is not a clean match, so I would "
                f"rather say so than guess."
            )

        if sig:
            read = self._describe(sig)
            if read:
                lines.append(f"\nWhat I see: {read}.")
            shadow_c = float(sig.get("shadow_C", 0.0))
            if shadow_c > 5:
                lines.append(
                    f"The dark areas lean {_colour_word(float(sig.get('shadow_hue', 0.0)))}, "
                    f"which is usually the deliberate part of a look."
                )
            high_c = float(sig.get("high_C", 0.0))
            if high_c > 5:
                lines.append(
                    f"The bright areas lean "
                    f"{_colour_word(float(sig.get('high_hue', 0.0)))}."
                )
            hsl = self._hsl_line(sig)
            if hsl:
                lines.append(hsl)
            coh = float(sig.get("shadow_coh", 0.0))
            if coh and coh < 0.35:
                lines.append(
                    "Worth knowing: the colours in the shadows point in lots of "
                    "different directions, so that is probably the subject rather "
                    "than a grade."
                )

        if len(matches) > 1:
            others = ", ".join(m["label"] for m in matches[1:])
            lines.append(f"\nOther possibilities: {others}.")
        lines.append(
            f"\nWant the recipe? Ask \"how do I get the {best['label'].lower()} look?\""
        )
        return "\n".join(lines)

    # -- achieve ------------------------------------------------------------
    def _say_achieve(self, facts: dict) -> str:
        steps = facts.get("steps") or []
        distance = facts.get("distance")
        previous = facts.get("previous_distance")
        target_name = facts.get("target_name") or "the reference"

        if not steps and distance is not None:
            return (
                f"You are there. Your photo now matches {target_name} closely enough "
                f"that any further change would be guesswork rather than improvement."
            )
        if not steps:
            return (
                "I need something to aim at. Either upload the photo you want to "
                "match, or name a look \u2014 for example \"how do I get the warm "
                "golden look?\""
            )

        head = f"Here is how to move your photo toward {target_name}. Work top to bottom:"
        body = _bullet_steps(steps)
        tail = []
        if distance is not None:
            if previous is not None:
                moved = previous - distance
                if moved > 0.01:
                    tail.append(
                        f"\nYou are closer than last time \u2014 {_plain_distance(distance)} now, "
                        f"down from {_plain_distance(previous)}."
                    )
                else:
                    tail.append(
                        f"\nStill {_plain_distance(distance)}. That happens \u2014 sliders "
                        f"interact, so a change can move one thing and shift another."
                    )
            else:
                tail.append(f"\nRight now you are {_plain_distance(distance)}.")
            tail.append(
                "Apply these, upload the result, and I will measure again and give "
                "you smaller corrections. It usually takes two or three rounds."
            )
        return "\n".join([head, "", body, *tail])

    # -- critique -----------------------------------------------------------
    def _say_critique(self, facts: dict) -> str:
        faults = facts.get("faults") or []
        notes = facts.get("taste") or []
        sig = facts.get("signature") or {}
        lines = []

        if not sig:
            return (
                "Upload the edit you want me to look at and I will check it for "
                "blown highlights, crushed shadows, colour casts and over-saturation "
                "\u2014 and compare it against what you have told me you like."
            )

        read = self._describe(sig)
        if read:
            lines.append(f"Overall this reads as {read}.")

        if not faults:
            lines.append("\nTechnically it is clean. No clipping, no cast, sensible contrast.")
        else:
            lines.append(f"\n**{len(faults)} thing{'s' if len(faults) != 1 else ''} I would fix:**")
            for fault in faults:
                lines.append(
                    f"\n- **{fault['issue']}** \u2014 {fault['detail']}.\n"
                    f"  What to do: {fault['fix']}."
                )

        hsl = self._hsl_line(sig)
        if hsl:
            lines.append(f"\n{hsl}")

        if notes:
            lines.append("\n**Against what you have told me you like:**")
            for note in notes:
                lines.append(f"- {note['note']}")
        return "\n".join(lines)

    # -- chat ---------------------------------------------------------------
    def _say_chat(self, facts: dict) -> str:
        profile = facts.get("profile") or {}
        text = (facts.get("user_text") or "").lower()
        saved = facts.get("saved_facts") or {}

        greeting = f"Hi {profile['name']}. " if profile.get("name") else ""

        if saved:
            pretty = "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in saved.items())
            return (
                f"{greeting}Saved. I will remember that \u2014 {pretty}.\n\n"
                f"It stays with your profile, so it applies in every conversation, "
                f"not just this one."
            )

        if any(k in text for k in ("what can you do", "help", "how does this work",
                                   "what is this", "how do i use")):
            return greeting + (
                "I measure the actual colour in a photo and turn it into Lightroom "
                "slider moves. Three things I can do:\n\n"
                "**1. Identify** \u2014 \"what look is this?\" Upload a photo and I will "
                "tell you which style it is closest to and what makes it that.\n\n"
                "**2. Recreate** \u2014 \"how do I get the teal and orange look?\" I give you "
                "ordered slider steps: Basic panel, Colour Grading wheels, and the "
                "Colour Mixer (HSL) per colour. Upload your result and I will refine it.\n\n"
                "**3. Critique** \u2014 \"what is wrong with my edit?\" I check clipping, "
                "contrast, colour casts and saturation.\n\n"
                "You can also tell me things to remember \u2014 your camera, the looks you "
                "like \u2014 and I will use them in future conversations.\n\n"
                "No AI image model is involved. Every number comes from measuring pixels."
            )

        known = []
        if profile.get("camera"):
            known.append(f"you shoot on a {profile['camera']}")
        if profile.get("preferred_look"):
            known.append(f"you like {profile['preferred_look']}")
        if profile.get("dislikes"):
            known.append(f"you avoid {profile['dislikes']}")
        if profile.get("editor"):
            known.append(f"you edit in {profile['editor']}")

        if known:
            return (
                greeting
                + "Got it \u2014 "
                + ", and ".join(known)
                + ". I will keep that in mind.\n\nAsk me to identify a look, recreate "
                "one, or check an edit."
            )
        return (
            greeting
            + "I am a colour-grading assistant. Ask me what a look is, how to get "
            "one, or what is wrong with an edit \u2014 you do not need to upload "
            "anything to start. Try: *how do I get the warm golden look?*"
        )


def _colour_word(lab_hue: float) -> str:
    """Nearest everyday colour word for a measured hue angle."""
    from .rules import LAB_HUE
    from .color import circ_dist

    best, best_gap = "neutral", 1e9
    for name, angle in LAB_HUE.items():
        gap = circ_dist(lab_hue, angle)
        if gap < best_gap:
            best, best_gap = name, gap
    return best


class GeminiNarrator:
    """Rewrites the deterministic draft with Gemini, over a pool of keys.

    The facts are still computed deterministically and are passed in as a
    finished draft; the model only changes the voice. That is the whole point
    of the design -- the model narrates numbers it did not produce and cannot
    change, so a hallucinated slider value is structurally impossible.

    Any failure at all -- no keys, every key rate-limited, the deadline
    expiring, an empty completion -- returns the deterministic draft unchanged.
    """

    SYSTEM = (
        "You are LookLab, a colour-grading assistant talking to a photographer. "
        "Rewrite the draft below in a warmer, more natural voice.\n"
        "HARD RULES:\n"
        "1. Never change, add, remove or round any number, slider name, hue "
        "angle, distance or measurement. They are computed, not suggested.\n"
        "2. Keep every markdown heading, list item and bold marker.\n"
        "3. Keep it the same length or shorter. No preamble, no sign-off.\n"
        "4. Do not invent advice that is not in the draft.\n"
        "Return only the rewritten text."
    )

    def __init__(self, pool: Any, deadline: float = 12.0) -> None:
        self._pool = pool
        self._deadline = deadline
        self._demo = DemoChatModel()
        self.name = f"gemini/{getattr(pool, 'model', '?')} ({len(pool)} keys, pooled)"
        self.is_deterministic = False
        self.rewrites = 0
        self.fallbacks = 0

    def narrate(self, intent: str, facts: dict) -> str:
        draft = self._demo.narrate(intent, facts)
        try:
            text = self._pool.generate(
                draft,
                system=self.SYSTEM,
                max_output_tokens=1400,
                temperature=0.6,
                deadline=self._deadline,
            )
        except Exception:
            self.fallbacks += 1
            return draft
        text = (text or "").strip()
        if not text or len(text) > 4 * len(draft) + 400:
            # A wildly longer answer means it ignored the brief and started
            # inventing. The draft is the trustworthy artefact; keep it.
            self.fallbacks += 1
            return draft
        self.rewrites += 1
        return text


class ProviderChatModel:
    """Optional wrapper around a real chat model, used only for prose.

    The computed facts are still produced deterministically and are passed in;
    the model rewrites the deterministic draft. If anything at all goes wrong,
    the deterministic text is returned unchanged.
    """

    def __init__(self, model: Any, name: str) -> None:
        self._model = model
        self.name = name
        self.is_deterministic = False
        self._demo = DemoChatModel()

    def narrate(self, intent: str, facts: dict) -> str:
        draft = self._demo.narrate(intent, facts)
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            reply = self._model.invoke(
                [
                    SystemMessage(
                        content=(
                            "You are LookLab, a colour-grading assistant. Rewrite the draft "
                            "below in a warmer, more conversational voice. You must not change, "
                            "add, or remove any number, slider name, or measurement. Keep it "
                            "roughly the same length and keep the markdown structure."
                        )
                    ),
                    HumanMessage(content=draft),
                ]
            )
            text = getattr(reply, "content", "")
            return text.strip() or draft
        except Exception:
            return draft


_NARRATOR: Any = None
_NARRATOR_KEY: str | None = None
_NARRATOR_LOCK = threading.Lock()


def get_narrator() -> Any:
    """Return the configured narrator, built once per process.

    Cached deliberately. Building a fresh narrator per turn would rebuild the
    Gemini pool on every request -- discarding its per-key health, so a
    credential that just rate-limited would be retried immediately -- and would
    make the rewrite/fallback counters on /models read zero forever, because
    the object being inspected was never the one that did the work.

    The cache key is the provider configuration, so changing ``LOOKLAB_MODEL``
    in a test still takes effect.
    """
    global _NARRATOR, _NARRATOR_KEY
    cache_key = "|".join(
        (os.getenv(name) or "") for name in ("LOOKLAB_MODEL", "LOOKLAB_MODEL_NAME", "GEMINI_API_KEYS")
    )
    with _NARRATOR_LOCK:
        if _NARRATOR is not None and _NARRATOR_KEY == cache_key:
            return _NARRATOR
        _NARRATOR = _build_narrator()
        _NARRATOR_KEY = cache_key
        return _NARRATOR


def _build_narrator() -> Any:
    """Construct the narrator. Deterministic unless a provider is configured.

    Set ``LOOKLAB_MODEL=google`` with ``GEMINI_API_KEYS``, or
    ``LOOKLAB_MODEL=ollama`` with Ollama running locally, to use a real model.
    Anything missing falls back to ``DemoChatModel`` silently -- a missing key
    must never be able to break a demo.
    """
    provider = (os.getenv("LOOKLAB_MODEL") or "").strip().lower()
    if not provider or provider in {"demo", "none", "off"}:
        return DemoChatModel()

    try:
        if provider in {"google", "gemini"}:
            # Prefer the pooled multi-key client: it rotates across every
            # configured credential and absorbs the transient 503s that the
            # live API returns for roughly a third of calls.
            from .gemini import get_pool

            pool = get_pool()
            if pool is not None and pool.usable:
                return GeminiNarrator(pool)
            return DemoChatModel()
        if provider == "ollama":
            from langchain_ollama import ChatOllama

            return ProviderChatModel(
                ChatOllama(model=os.getenv("LOOKLAB_MODEL_NAME", "llama3.2")),
                f"ollama/{os.getenv('LOOKLAB_MODEL_NAME', 'llama3.2')}",
            )
    except Exception:
        return DemoChatModel()
    return DemoChatModel()


__all__ = ["AIMessage", "DemoChatModel", "ProviderChatModel", "get_narrator"]
