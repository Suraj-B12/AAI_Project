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
from typing import Any

from langchain_core.messages import AIMessage


def _fmt_amount(amount: Any) -> str:
    if isinstance(amount, (int, float)):
        return f"{amount:+.0f}"
    return str(amount)


def _bullet_steps(steps: list[dict]) -> str:
    lines = []
    for i, step in enumerate(steps, 1):
        lines.append(f"{i}. **{step['slider']} {_fmt_amount(step['amount'])}** - {step['why']}")
    return "\n".join(lines)


class DemoChatModel:
    """Deterministic narrator. Same inputs always produce the same words."""

    name = "DemoChatModel (deterministic, no API key required)"
    is_deterministic = True

    def narrate(self, intent: str, facts: dict) -> str:
        handler = getattr(self, f"_say_{intent}", None)
        if handler is None:
            return self._say_chat(facts)
        return handler(facts)

    # -- identify -----------------------------------------------------------
    def _say_identify(self, facts: dict) -> str:
        matches = facts.get("matches") or []
        if not matches:
            return (
                "I could not measure a signature for that. Upload a reference image, "
                "or name a look from the library and I will describe it."
            )
        best = matches[0]

        # Recommended from the long-term profile rather than measured. This is
        # the answer to "what should I try?" in a thread that has never been
        # told anything -- the payoff for storing taste across conversations.
        if best.get("from_profile"):
            profile = facts.get("profile") or {}
            known = []
            if profile.get("camera"):
                known.append(f"you shoot on a {profile['camera']}")
            if profile.get("preferred_look"):
                known.append(f"you lean toward {profile['preferred_look']}")
            if profile.get("dislikes"):
                known.append(f"you'd rather avoid {profile['dislikes']}")
            lead = (
                f"Going on what you've told me before -- {', and '.join(known)} -- "
                if known
                else "Based on your saved profile, "
            )
            lines = [lead + "these are the closest fits in the library:", ""]
            for match in matches:
                lines.append(f"- **{match['label']}** - {match['notes']}")
            lines.append(
                "\nSay \"how do I get the "
                f"{matches[0]['label'].lower()} look?\" and I'll give you the slider steps."
            )
            return "\n".join(lines)
        sig = facts.get("signature") or {}
        lines = []
        if best.get("confident"):
            lines.append(
                f"That reads closest to **{best['label']}** "
                f"(distance {best['distance']:.2f}, confidence {best['confidence']:.0%})."
            )
        else:
            lines.append(
                f"**No confident match.** The nearest entry is *{best['label']}* at "
                f"distance {best['distance']:.2f}, which is past my threshold - so I would "
                f"rather say I do not know than force one."
            )
        if best.get("notes"):
            lines.append(f"_{best['notes']}_")

        if sig:
            lines.append(
                f"\nWhat I measured: shadow hue **{sig.get('shadow_hue', 0):.0f}deg** "
                f"(chroma {sig.get('shadow_C', 0):.1f}, coherence {sig.get('shadow_coh', 0):.2f}), "
                f"highlight hue **{sig.get('high_hue', 0):.0f}deg**, "
                f"contrast **{sig.get('contrast', 0):.0f}** L\\*, "
                f"overall a\\* {sig.get('a_global', 0):+.1f} / b\\* {sig.get('b_global', 0):+.1f}."
            )
            coh = float(sig.get("shadow_coh", 0.0))
            if coh < 0.35:
                lines.append(
                    f"Note the low shadow coherence ({coh:.2f}) - the shadow hues largely "
                    f"cancel out, which usually means a colourful subject rather than a grade."
                )
        if len(matches) > 1:
            others = ", ".join(f"{m['label']} ({m['distance']:.2f})" for m in matches[1:])
            lines.append(f"\nRunners-up: {others}.")
        return "\n".join(lines)

    # -- achieve ------------------------------------------------------------
    def _say_achieve(self, facts: dict) -> str:
        steps = facts.get("steps") or []
        distance = facts.get("distance")
        previous = facts.get("previous_distance")
        target_name = facts.get("target_name") or "the reference"

        if not steps and distance is not None:
            return (
                f"You are within tolerance of {target_name} - measured distance "
                f"**{distance:.3f}**, below my convergence threshold. Nothing left worth changing."
            )
        if not steps:
            return (
                "I need something to compare against. Upload a reference image, or name a "
                "look from the library - try *warm golden* or *teal and orange*."
            )

        head = f"To move toward {target_name}, in this order:"
        body = _bullet_steps(steps)
        tail = []
        if distance is not None:
            if previous is not None:
                delta = previous - distance
                arrow = "down from" if delta > 0 else "up from"
                tail.append(
                    f"\nDistance to target: **{distance:.3f}** ({arrow} {previous:.3f})."
                )
            else:
                tail.append(f"\nDistance to target: **{distance:.3f}**.")
            tail.append(
                "Apply these, re-upload, and I will re-measure. The slider mapping is "
                "approximate, so the loop is what makes it converge."
            )
        return "\n".join([head, "", body, *tail])

    # -- critique -----------------------------------------------------------
    def _say_critique(self, facts: dict) -> str:
        faults = facts.get("faults") or []
        notes = facts.get("taste") or []
        sig = facts.get("signature") or {}
        lines = []

        if not sig:
            # No image at all. Saying "technically this is clean" here would be
            # asserting something about a frame that was never measured.
            return (
                "I have nothing to look at yet - upload the edit you want critiqued and I "
                "will measure it. I check clipping, tonal range, colour casts and "
                "saturation, and compare the result against what you have told me you like."
            )

        if not faults:
            lines.append("Technically this is clean - no clipping, no cast, tonal range is sane.")
        else:
            lines.append(f"**{len(faults)} thing{'s' if len(faults) != 1 else ''} I would fix:**")
            for fault in faults:
                lines.append(
                    f"- **{fault['issue']}** ({fault['severity']}) - {fault['detail']}. "
                    f"Try: {fault['fix']}."
                )
        if sig:
            lines.append(
                f"\nMeasured: contrast **{sig.get('contrast', 0):.0f}** L\\*, "
                f"mean chroma **{sig.get('chroma', 0):.1f}**, "
                f"clipping {float(sig.get('clip_black', 0)) * 100:.1f}% black / "
                f"{float(sig.get('clip_white', 0)) * 100:.1f}% white."
            )
        if notes:
            lines.append("\n**Against what you have told me you like:**")
            for note in notes:
                lines.append(f"- {note['note']}")
        return "\n".join(lines)

    # -- chat ---------------------------------------------------------------
    def _say_chat(self, facts: dict) -> str:
        profile = facts.get("profile") or {}
        text = (facts.get("user_text") or "").lower()

        greeting = ""
        if profile.get("name"):
            greeting = f"Hi {profile['name']}. "

        if any(k in text for k in ("what can you do", "help", "how does this work", "what is this")):
            body = (
                "I measure colour in CIELAB and turn the difference between two images into "
                "Lightroom slider moves. Three things I can do:\n\n"
                "1. **Identify** - \"what look is this?\" - I match a signature against a "
                "15-look reference library.\n"
                "2. **Achieve** - \"how do I get the teal and orange look?\" - I compute the "
                "delta and give you ordered slider steps, then refine as you re-upload.\n"
                "3. **Critique** - \"what's wrong with my edit?\" - I check clipping, contrast, "
                "casts and saturation, and compare against your stated taste.\n\n"
                "You can do all three by name with no uploads at all. "
                "No vision model is involved - every number is measured."
            )
            return greeting + body

        known = []
        if profile.get("camera"):
            known.append(f"you shoot on a {profile['camera']}")
        if profile.get("preferred_look"):
            known.append(f"you lean toward {profile['preferred_look']}")
        if profile.get("dislikes"):
            known.append(f"you dislike {profile['dislikes']}")
        if profile.get("editor"):
            known.append(f"you edit in {profile['editor']}")

        if known:
            return (
                greeting
                + "Noted - "
                + ", and ".join(known)
                + ". I will keep that in mind. Ask me to identify a look, help you achieve one, "
                "or critique an edit."
            )
        return (
            greeting
            + "I am a colour-grading assistant. Ask me what a look is, how to achieve one, or "
            "what is wrong with an edit. Tell me your camera and what you like and I will "
            "remember it across conversations."
        )


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


def get_narrator() -> Any:
    """Return the configured narrator. Deterministic unless explicitly enabled.

    Set ``LOOKLAB_MODEL=google`` plus ``GOOGLE_API_KEY``, or
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
