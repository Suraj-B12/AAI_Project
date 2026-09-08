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
import re
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
            answered = self._general_fallback(
                facts,
                "If you want a look measured rather than explained, upload a photo "
                "and ask what look it is.",
            )
            if answered:
                return answered
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
            answered = self._general_fallback(
                facts,
                "If you want sliders rather than an explanation, name a look \u2014 for "
                "example *how do I get the warm golden look?*",
            )
            if answered:
                return answered
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
            answered = self._general_fallback(
                facts,
                "If you want this checked on an actual photo, upload it and I will "
                "measure it.",
            )
            if answered:
                return answered
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
    # -- recall: answer a question about what is stored ---------------------
    LABELS = {
        "name": "Your name",
        "camera": "You shoot on",
        "editor": "You edit in",
        "preferred_look": "You like",
        "dislikes": "You would rather avoid",
    }

    def _say_forget(self, request, profile: dict) -> str:
        """Confirm a deletion by showing what is left.

        ``profile`` here is the state AFTER the removal, because save_profile
        re-reads the store. Listing the remainder is the confirmation worth
        giving: "done" is a claim, the remaining list is evidence.
        """
        what, target = request
        if what == "all":
            return (
                "Cleared. I have deleted everything I had saved about you \u2014 name, "
                "camera, preferences and any notes.\n\nNothing is kept. Tell me "
                "something new whenever you like and I will start again."
            )

        label = {
            "camera": "your camera",
            "name": "your name",
            "editor": "which editor you use",
            "preferred_look": "what you said you like",
            "dislikes": "what you said you dislike",
            "notes": "that note",
        }.get(str(target), f"\u201c{target}\u201d")

        if not profile:
            return (
                f"Done \u2014 I have forgotten {label}, and there is nothing else left "
                f"in your profile either."
            )

        listing = "\n".join(
            line for line in self._say_recall(profile).splitlines()
            if line.lstrip().startswith("-")
        )
        if not listing:
            return f"Done \u2014 I have forgotten {label}. Nothing else is saved."
        return f"Done \u2014 I have forgotten {label}.\n\nStill saved:\n\n{listing}"

    def _say_recall(self, profile: dict) -> str:
        """List exactly what is in long-term memory. No inference, no padding.

        This is a question about stored data, so the answer is the stored data.
        Anything added here would be the app claiming to know something it does
        not.
        """
        if not profile:
            return (
                "Nothing yet — this is the first thing you have told me, or your "
                "profile was cleared.\n\n"
                "Tell me something and I will keep it: your camera, the looks you "
                "like, anything you want avoided. Try "
                "*\"remember that I shoot on a Fuji X-T4\"*."
            )

        lines = ["Here is everything I have saved about you:", ""]
        for key, label in self.LABELS.items():
            if profile.get(key):
                lines.append(f"- **{label}:** {profile[key]}")

        notes = profile.get("notes")
        if isinstance(notes, list) and notes:
            lines.append("- **You asked me to remember:**")
            for note in notes:
                lines.append(f"    - {note}")

        extra = [
            k for k in profile
            if k not in self.LABELS and k != "notes" and profile.get(k)
        ]
        for key in extra:
            lines.append(f"- **{key.replace('_', ' ')}:** {profile[key]}")

        lines.append(
            "\nThis is stored against your user id, not this conversation, so it "
            "applies everywhere. Say *\"remember that ...\"* to add to it."
        )
        return "\n".join(lines)

    def _say_chat(self, facts: dict) -> str:
        """The chat branch: everything that is not identify, achieve or critique.

        This used to have three outcomes and a catch-all, and the catch-all
        swallowed everything: measured, fourteen out of fourteen ordinary
        questions came back with the same sentence. A user who told the app
        their camera and then asked four reasonable questions got the same
        canned recital of their own profile four times.

        The message is now classified first (see ``knowledge.classify_message``)
        and each class gets a real answer. Nothing here needs a model; the
        model, when configured, only widens what can be answered.
        """
        from .knowledge import classify_message, look_labels, lookup, render_entry

        profile = facts.get("profile") or {}
        text = (facts.get("user_text") or "").strip()
        saved = facts.get("saved_facts") or {}
        greeting = f"Hi {profile['name']}. " if profile.get("name") else ""

        if facts.get("forget_request"):
            return self._say_forget(facts["forget_request"], profile)

        if facts.get("is_recall"):
            return self._say_recall(profile)

        # A message can both tell us something and ask something: "I use
        # Lightroom, what can you do?". Acknowledging the fact and stopping
        # answers only half of it, so the acknowledgement becomes a one-line
        # prefix and the question is answered underneath.
        asks_too = bool(re.search(r"\?|^\s*(what|how|why|which|can|does|is|are)\b",
                                  text, flags=re.IGNORECASE))

        # Acknowledge ONLY on a turn that actually learned something. This
        # branch used to fire whenever a profile existed, which is why every
        # later question was answered by reciting the profile back.
        if saved and not asks_too:
            explicit = "remembered" in saved
            fields = {k: v for k, v in saved.items() if k != "remembered"}
            pretty = ", ".join(f"{k.replace('_', ' ')} is {v}" for k, v in fields.items())
            if explicit:
                lead = f"{greeting}Saved — I will remember that {saved['remembered']}."
                if pretty:
                    lead += f" (Filed under {pretty}.)"
            else:
                lead = f"{greeting}Noted — {pretty}." if pretty else f"{greeting}Noted."
            return (
                f"{lead}\n\nThis is kept with your profile, so it applies in every "
                f"conversation, not just this one. Ask *\"what do you know about me?\"* "
                f"to see everything saved."
            )

        lowered = text.lower()

        # A short prefix noting what was learned, prepended to whatever answer
        # follows, so a message that both states a fact and asks a question
        # gets both.
        noted = ""
        if saved and asks_too:
            fields = {k: v for k, v in saved.items() if k != "remembered"}
            if fields:
                noted = (
                    "*(Noted — "
                    + ", ".join(f"{k.replace('_', ' ')} is {v}" for k, v in fields.items())
                    + ", saved to your profile.)*\n\n"
                )
            elif saved.get("remembered"):
                noted = f"*(Saved — {saved['remembered']}.)*\n\n"

        # Phrase matching, not bare substrings. "help" alone fired this on
        # "help me understand log", which is a question about log, not a
        # request for the capabilities list.
        if re.search(
            r"\b(what can you do|what do you do|how does this work|how do i use (this|you)"
            r"|what (is|are) (this|you)|what are you for|can you help me\??$"
            r"|help me get started|^help$)\b",
            lowered,
        ):
            return noted + greeting + self._capabilities()

        kind = classify_message(text)

        if kind == "social":
            return noted + self._say_social(lowered, greeting, profile)

        if kind == "meta":
            return noted + self._say_meta()

        if kind == "library":
            labels = look_labels()
            listed = "\n".join(f"- {name}" for name in labels)
            return (
                f"I have **{len(labels)} reference looks** measured and stored:\n\n"
                f"{listed}\n\nAsk *\"how do I get the {labels[0].lower()} look?\"* for the "
                f"slider steps, or upload a photo and I will tell you which one it is "
                f"closest to."
            )

        if kind == "glossary":
            entries = lookup(text)
            body = "\n\n".join(render_entry(e) for e in entries)
            return (
                f"{body}\n\nIf you want this applied to an actual photo rather than "
                f"explained, upload one and ask what look it is, or ask how to get a "
                f"look you have in mind."
            )

        # Anything left is a question this app cannot answer from its own
        # measurements. A model may be able to; the deterministic build says so
        # honestly instead of pretending.
        if kind in ("domain_question", "domain_statement"):
            answerer = self._answerer(facts) or self.answer_question
            answer = answerer(text, profile)
            if answer:
                return answer
            return self._cannot_answer(text, in_domain=True, has_model=bool(self._answerer(facts)))

        if kind == "out_of_scope":
            # Deliberately never offered to the model: a colour tool answering
            # "tell me a joke" is scope creep, and a wrong answer there costs
            # more credibility than the reply is worth.
            return self._cannot_answer(text, in_domain=False, has_model=bool(self._answerer(facts)))

        return greeting + (
            "I am not sure what you are after. I can identify a look in a photo, "
            "give you the sliders to recreate one, or check an edit for problems. "
            "Try *how do I get the warm golden look?* or upload a photo and ask "
            "*what look is this?*"
        )

    # -- the pieces ---------------------------------------------------------

    def _general_fallback(self, facts: dict, offer: str) -> str | None:
        """Handle a message that reached an analysis branch with nothing to analyse.

        The routing cues are substrings, so an ordinary question can be caught
        by one: "is anything **wrong** with log footage" hits a critique cue and
        used to be answered with "upload the edit you want me to look at". If
        there is nothing to analyse and the message reads as a general
        question, answer the question instead. Returns None to let the branch
        print its normal "give me something to look at" message.
        """
        from .knowledge import classify_message, lookup, render_entry

        text = (facts.get("user_text") or "").strip()
        if not text:
            return None

        kind = classify_message(text)
        if kind not in ("glossary", "domain_question", "meta", "library"):
            return None
        # "what is wrong with MY edit" really is a request to look at an image.
        if re.search(r"\b(my|this|these|the)\s+(edit|photo|image|picture|shot|file)\b",
                     text, flags=re.IGNORECASE):
            return None

        if kind == "meta":
            return self._say_meta()
        if kind == "library":
            return None  # the library reply is only wired into the chat branch
        if kind == "glossary":
            entries = lookup(text)
            body = "\n\n".join(render_entry(e) for e in entries)
            return f"{body}\n\n{offer}"

        answerer = self._answerer(facts)
        if answerer:
            answer = answerer(text, facts.get("profile") or {})
            if answer:
                return answer
        # A real question that reached this branch only because a routing cue
        # is a substring. Saying "upload the edit you want me to look at" to
        # "is anything wrong with log footage" answers a question nobody asked;
        # saying plainly that it cannot answer is both truer and more useful.
        return self._cannot_answer(text, in_domain=True, has_model=bool(answerer))

    def answer_question(self, text: str, profile: dict) -> str | None:
        """A general answer, when a model is available. None on the offline path.

        ``DemoChatModel`` never answers general questions -- it has nothing to
        answer them with, and inventing one would break the only claim this
        project actually makes.

        A wrapping narrator supplies its own answerer through ``facts`` rather
        than by subclassing, because ``GeminiNarrator`` DELEGATES to a
        ``DemoChatModel`` instance instead of inheriting from it: overriding
        this method on the wrapper had no effect at all, since the code that
        calls it runs on the delegate.
        """
        return None

    @staticmethod
    def _answerer(facts: dict):
        """The active narrator's general-question answerer, if it has one."""
        return facts.get("_answerer")

    def _cannot_answer(self, text: str, in_domain: bool, has_model: bool = False) -> str:
        """Say so plainly, and point at what the app can actually do.

        The wording depends on WHY it cannot answer. Claiming "I have no
        language model configured" while one is running and simply declined the
        question is a lie about the app's own state.
        """
        from .knowledge import lookup, render_entry

        if in_domain and has_model:
            head = (
                "I could not get an answer to that just now. It is a photography "
                "question rather than something I can measure, and I would rather "
                "say so than guess."
            )
        elif in_domain:
            head = (
                "That is a photography question rather than something I can measure, "
                "and I answer from measurements. I do not have a language model "
                "configured, so I would only be guessing."
            )
        else:
            head = (
                "That is outside what I do — I am a colour-grading tool, so I would "
                "only be making something up."
            )

        related = lookup(text, limit=1)
        extra = ""
        if related:
            extra = f"\n\nRelated to what you asked, though:\n\n{render_entry(related[0])}"

        return (
            f"{head}{extra}\n\nWhat I can do: identify the look in a photo, give you "
            f"the Lightroom sliders to recreate one, or check an edit for clipping, "
            f"casts and contrast problems."
        )

    def _say_social(self, lowered: str, greeting: str, profile: dict) -> str:
        if lowered.startswith(("thanks", "thank you", "ta", "cheers", "nice",
                               "great", "cool", "awesome", "perfect", "lovely")):
            return "Any time. Ask me whenever you want a look measured or recreated."
        if lowered.startswith(("bye", "goodbye", "see you", "later", "cya")):
            return (
                "See you. Anything you told me to remember is saved against your "
                "profile, so it will still be here next time."
            )
        if lowered.startswith(("ok", "okay", "sure", "right", "got it", "alright", "fine")):
            return "Right. What would you like to do next?"
        opener = greeting or "Hello. "
        if profile.get("preferred_look"):
            return (
                f"{opener}Want to pick up where you left off? You told me you like "
                f"{profile['preferred_look']} — ask *what should I try?* and I will "
                f"suggest something, or upload a photo and I will measure it."
            )
        return (
            f"{opener}Ask me what a look is, how to get one, or what is wrong with "
            f"an edit. You do not need to upload anything to start — try "
            f"*how do I get the warm golden look?*"
        )

    def _say_meta(self) -> str:
        """Answer questions about the assistant itself, honestly and specifically."""
        return (
            "Fair question, so here is exactly how I work.\n\n"
            "**The numbers are measured, not generated.** When you upload a photo I "
            "convert it to CIELAB and compute real statistics — hue and chroma per "
            "tonal zone, contrast, clipping, per-colour HSL. The same photo always "
            "gives the same numbers. Nothing about that involves an AI image model.\n\n"
            "**The slider advice is arithmetic** on those numbers, through a rule "
            "table that was frozen before it was ever evaluated. It was tested "
            "against 135 graded frames the reference library had never seen: 87% of "
            "the time it moves a slider in the correct direction.\n\n"
            "**What I can get wrong:** the mapping from a measurement back to slider "
            "values is approximate, so the advice is a first guess that you refine by "
            "re-uploading. A colourful subject can also look like a colour grade to "
            "me — a red car makes the reds read strong whatever you did in editing.\n\n"
            "**Where a language model comes in:** only to reword my answers, or to "
            "answer general photography questions, and those are labelled when it "
            "happens. It never invents a number."
        )

    def _capabilities(self) -> str:
        return (
            "I measure the actual colour in a photo and turn it into Lightroom "
            "slider moves. Three things I can do:\n\n"
            "**1. Identify** — \"what look is this?\" Upload a photo and I will tell "
            "you which style it is closest to and what makes it that.\n\n"
            "**2. Recreate** — \"how do I get the teal and orange look?\" I give you "
            "ordered slider steps: Basic panel, Colour Grading wheels, and the "
            "Colour Mixer (HSL) per colour. Upload your result and I will refine it.\n\n"
            "**3. Critique** — \"what is wrong with my edit?\" I check clipping, "
            "contrast, colour casts and saturation.\n\n"
            "I also keep what you tell me — your camera, the looks you like — and use "
            "it in future conversations. Say *remember that ...* and ask *what do you "
            "know about me?* to see it.\n\n"
            "No AI image model is involved. Every number comes from measuring pixels."
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
        "5. Never add, drop or reword a stated fact or preference about the "
        "user. If the draft says they dislike something, do not turn it into "
        "something they like. Asked to rewrite 'you dislike crushed blacks', "
        "this model produced 'you prefer lifted blacks' -- do not do that.\n"
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
        self.verbatim = 0
        self.answers = 0
        self.grounded = 0
        self.ungrounded = 0

    # Replies whose payload is a FACT ABOUT THE USER rather than a computed
    # number are returned verbatim. The model is good at warming up advice and
    # bad at leaving a stated preference alone: asked to rewrite "you dislike
    # crushed blacks" it produced "you prefer warm golden tones and lifted
    # blacks", inventing a preference the user never expressed. Numbers are
    # protected by the prompt because they are checkable; invented preferences
    # are not, so those replies simply do not go to the model.
    def _is_factual_about_user(self, intent: str, facts: dict) -> bool:
        if intent == "chat" and (
            facts.get("is_recall") or facts.get("saved_facts") or facts.get("profile")
        ):
            return True
        # A recommendation built from the stored profile restates the user's
        # preferences back at them, so it carries the same risk as a chat reply
        # about memory even though its intent is `identify`.
        matches = facts.get("matches") or []
        return bool(matches and matches[0].get("from_profile"))

    ANSWER_SYSTEM = (
        "You are the assistant inside LookLab, a colour-grading tool for "
        "photographers. Answer the user's question directly and briefly.\n"
        "RULES:\n"
        "1. Two short paragraphs at most. No preamble, no sign-off.\n"
        "2. If it is a photography, colour or camera question, answer it "
        "properly and concretely.\n"
        "3. If you are not certain of a specific fact -- a camera's exact "
        "capabilities, a model number, a spec -- say you are not certain "
        "rather than guessing. A wrong camera spec stated confidently is much "
        "worse than an admission.\n"
        "4. Never invent anything about LookLab itself. It does exactly three "
        "things: it measures the colour of an uploaded photo, it tells you "
        "which Lightroom sliders to move to match a look, and it critiques an "
        "edit. It does NOT apply LUTs, transform footage, edit or export "
        "images, or handle video. Do not tell the user LookLab can do "
        "something it cannot -- if in doubt, do not mention LookLab at all. "
        "You also cannot see what it measured or what it has stored.\n"
        "5. If the question has nothing to do with photography or colour, say "
        "briefly that it is outside what this tool is for.\n"
        "Plain markdown. No headings."
    )

    # Appended to every model-written answer. The project's claim is that its
    # numbers are measured; a general answer is not a measurement, so it says
    # so. Marking it also means a wrong answer is attributable rather than
    # looking like something the app computed.
    ANSWER_FOOTER = (
        "\n\n---\n*Answered by a language model, not measured. LookLab's slider "
        "advice and every number it quotes come from measuring your image; this "
        "reply does not.*"
    )

    def _answer_from_sources(self, text: str, sources: list) -> str | None:
        """Answer strictly from retrieved passages, with citations. None on failure."""
        from . import research

        prompt = (
            f"Question: {text}\n\n"
            f"Sources:\n\n{research.format_sources(sources)}\n\n"
            f"Answer the question using only these sources, citing them inline."
        )
        try:
            answer = self._pool.generate(
                prompt,
                system=self.GROUNDED_SYSTEM,
                max_output_tokens=900,
                temperature=0.2,
                deadline=self._deadline,
            )
        except Exception:
            self.fallbacks += 1
            return None
        answer = (answer or "").strip()
        if not answer:
            self.fallbacks += 1
            return None

        # The model said the sources do not cover this. Fall through to the
        # unaided path rather than returning nothing: "I could not verify
        # this, but here is what I know" is more useful than silence, as long
        # as the two are told apart. The a6700 question is the case in point --
        # Wikipedia does not list its log profiles, but the answer is known.
        if answer.upper().startswith("INSUFFICIENT_SOURCES"):
            return None

        # Only claim the answer is grounded if it actually cites something.
        # An uncited answer went to the model with sources and came back
        # ignoring them, which is exactly the case the marker must not cover.
        if not re.search(r"\[\d+\]", answer):
            self.ungrounded += 1
            return answer + self.ANSWER_FOOTER

        self.grounded += 1
        return f"{answer}\n\n{research.format_citations(sources)}{self.GROUNDED_FOOTER}"

    GROUNDED_SYSTEM = (
        "You are the assistant inside LookLab, a colour-grading tool for "
        "photographers. Answer the question USING ONLY the numbered sources "
        "provided.\n"
        "RULES:\n"
        "1. Every factual claim must come from a source. Cite it inline as "
        "[1], [2] and so on.\n"
        "2. If the sources do not actually answer the question, reply with "
        "exactly INSUFFICIENT_SOURCES on the first line and nothing else. Do "
        "not fill the gap from memory here -- another step handles that.\n"
        "3. Partial coverage is fine: answer the part the sources support and "
        "say which part they do not.\n"
        "3. Two short paragraphs at most. No preamble, no sign-off, no "
        "headings.\n"
        "4. Never claim anything about LookLab itself. It measures an "
        "uploaded photo's colour, recommends Lightroom sliders, and critiques "
        "an edit. It does not apply LUTs, transform footage or handle video.\n"
        "Plain markdown."
    )

    GROUNDED_FOOTER = (
        "\n\n---\n*Written by a language model from the sources above, not "
        "measured. Follow the links to check it.*"
    )

    def answer_question(self, text: str, profile: dict) -> str | None:
        """Answer a general question, grounded in retrieved sources when possible.

        Two paths, and the difference is visible to the user:

        * **Grounded** -- sources were found, the model was given only those,
          and the answer carries inline citations plus the links.
        * **Ungrounded** -- nothing relevant was retrievable, so the model
          answers from its own weights and the reply says so plainly.

        Returns None on any failure so the caller falls back to saying it
        cannot answer, which is better than a guess.
        """
        try:
            from . import research

            sources = research.retrieve(text)
        except Exception:
            sources = []

        if sources:
            grounded = self._answer_from_sources(text, sources)
            if grounded:
                return grounded

        context = ""
        camera = (profile or {}).get("camera")
        if camera:
            context = (
                f"\n\n(For context, the user has told LookLab they shoot on a "
                f"{camera}. Use this only if it is relevant to the question, and "
                f"do not restate their preferences back to them.)"
            )
        try:
            answer = self._pool.generate(
                text + context,
                system=self.ANSWER_SYSTEM,
                max_output_tokens=900,
                temperature=0.4,
                deadline=self._deadline,
            )
        except Exception:
            self.fallbacks += 1
            return None
        answer = (answer or "").strip()
        if not answer:
            self.fallbacks += 1
            return None
        self.answers += 1
        self.ungrounded += 1
        return answer + self.ANSWER_FOOTER

    def narrate(self, intent: str, facts: dict) -> str:
        # Hand the delegate a way to answer general questions. GeminiNarrator
        # wraps a DemoChatModel rather than subclassing it, so overriding
        # answer_question here would never be reached by the code that calls
        # it -- that code runs on the delegate.
        facts = {**facts, "_answerer": self.answer_question}

        draft = self._demo.narrate(intent, facts)
        if self._is_factual_about_user(intent, facts):
            self.verbatim += 1
            return draft
        # A general answer is already final -- it came from the model with its
        # own instructions and carries the "answered by a model" marker. Sending
        # it through the rewrite prompt would strip that marker.
        if draft.endswith(self.ANSWER_FOOTER):
            return draft
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
        self.rewrites = 0
        self.fallbacks = 0
        self.verbatim = 0
        self.answers = 0
        self.grounded = 0
        self.ungrounded = 0

    # Replies whose payload is a FACT ABOUT THE USER rather than a computed
    # number are returned verbatim. The model is good at warming up advice and
    # bad at leaving a stated preference alone: asked to rewrite "you dislike
    # crushed blacks" it produced "you prefer warm golden tones and lifted
    # blacks", inventing a preference the user never expressed. Numbers are
    # protected by the prompt because they are checkable; invented preferences
    # are not, so those replies simply do not go to the model.
    def _is_factual_about_user(self, intent: str, facts: dict) -> bool:
        if intent == "chat" and (
            facts.get("is_recall") or facts.get("saved_facts") or facts.get("profile")
        ):
            return True
        # A recommendation built from the stored profile restates the user's
        # preferences back at them, so it carries the same risk as a chat reply
        # about memory even though its intent is `identify`.
        matches = facts.get("matches") or []
        return bool(matches and matches[0].get("from_profile"))

    def narrate(self, intent: str, facts: dict) -> str:
        draft = self._demo.narrate(intent, facts)
        if self._is_factual_about_user(intent, facts):
            self.verbatim += 1
            return draft
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
