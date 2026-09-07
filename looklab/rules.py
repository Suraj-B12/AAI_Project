"""Delta -> Lightroom slider instructions, plus technical fault checks.

Flat rules with an explicit ``why`` on every step. The ``why`` string is what
the narrator reads out, so the explanation is grounded in the computation
rather than invented by a language model.

The coefficients are conversion factors from LAB units to slider units.

**They were frozen before ``tools/calibrate.py`` was ever run, and they have
not been tuned against its output.** That is deliberate. The calibration
frames are produced by this project's own ``grading.py``, so fitting the
inverse against them would turn a partial circularity into a total one and
the reported sign agreement would measure nothing. The numbers in the README
are whatever these authored values happened to score on plates the knowledge
base never saw.

The mapping is deliberately many-to-one and is not claimed to be invertible.
It produces a first guess inside a closed loop -- apply, re-measure, correct.
``distance_to_target`` shrinking across turns is the evidence that the loop
works, and it is reported in telemetry every turn.
"""

from __future__ import annotations

from typing import Any

from .color import MATCH_WEIGHTS, circ_dist, weighted_distance, zone_is_measurable
from .grading import wheel_hue_for_lab_hue

# LAB-unit -> slider-unit conversion factors, as authored in the design
# document. NOT fitted to calibration output -- see the module docstring.
# tools/calibrate.py measures them; it does not choose them.
COEFFICIENTS: dict[str, float] = {
    "temp": 1.6,  # per unit of b* (blue <-> yellow)
    "tint": 1.4,  # per unit of a* (green <-> magenta)
    "contrast": 0.9,  # per L* unit of p95-p05 range
    "blacks": 1.2,  # per L* unit of p05
    "whites": 1.1,  # per L* unit of p95
    "shadows": 1.1,  # per L* unit of shadow-zone lightness
    "highlights": 1.0,  # per L* unit of highlight-zone lightness
    "saturation": 2.2,  # per unit of mean chroma
    "cg_sat": 2.5,  # chroma -> colour-grading saturation
    "hsl_sat": 2.0,  # per unit of family chroma -> HSL Saturation
    "hsl_lum": 1.4,  # per L* unit of family lightness -> HSL Luminance
    "hsl_hue": 0.8,  # per degree of family hue offset -> HSL Hue
}

# Below these deltas, a step is noise rather than advice and is not emitted.
THRESHOLDS: dict[str, float] = {
    "temp": 1.5,  # b* units
    "tint": 1.5,  # a* units
    "contrast": 3.0,  # L* units
    "blacks": 2.0,
    "whites": 2.0,
    "shadow_hue": 15.0,  # degrees
    "mid_hue": 18.0,
    "high_hue": 15.0,
    "saturation": 1.5,  # chroma units
    "hsl_sat": 4.0,  # family chroma units
    "hsl_lum": 6.0,  # family L* units
    "hsl_hue": 8.0,  # degrees of family hue offset
}

# A colour family must cover at least this much of BOTH frames before HSL
# advice is offered. Tuning the Orange slider because six orange pixels differ
# is noise dressed as advice.
HSL_MIN_COVERAGE = 0.04

# Sliders are clamped to Lightroom's actual range so the advice is applicable.
SLIDER_RANGE = (-100, 100)

# Distance below which the refine loop declares success.
CONVERGENCE_TOLERANCE = 0.055

# CIELAB hue angles of the named colours, measured rather than assumed.
# Note these are LAB angles, NOT Lightroom colour-grading wheel values -- the
# two differ by 9 to 59 degrees (see grading.wheel_hue_for_lab_hue).
LAB_HUE: dict[str, float] = {
    "red": 31.0,
    "orange": 66.0,
    "yellow": 99.0,
    "green": 142.0,
    "teal": 197.0,
    "cyan": 222.0,
    "blue": 293.0,
    "violet": 307.0,
    "magenta": 331.0,
}

# A typical untouched photograph does not sit at b* = 0. Across the five
# ungraded base plates the median is around b* +12, because most photographic
# content skews warm. Judging "this reads cool" against 0 would therefore call
# a perfectly normal frame cool. This is the threshold below which an image
# genuinely reads cooler than an untouched photo.
NEUTRAL_B_STAR = 8.0


def _clamp(x: float) -> int:
    lo, hi = SLIDER_RANGE
    return int(max(lo, min(hi, round(x))))


def _step(
    slider: str,
    amount: Any,
    why: str,
    magnitude: float,
    panel: str = "Basic",
    detail: str = "",
) -> dict[str, Any]:
    """One instruction.

    ``why`` is plain language a photographer can act on without knowing what
    CIELAB is. ``detail`` carries the measured numbers for anyone who wants
    them, so the interface can show the reason first and the arithmetic second
    instead of forcing "b* +12.0 to +25.5" on everybody.

    ``panel`` says where the control lives in Lightroom, because "Saturation"
    alone is ambiguous -- there are three of them.
    """
    return {
        "slider": slider,
        "amount": amount,
        "why": why,
        "detail": detail,
        "panel": panel,
        "magnitude": round(float(magnitude), 3),
    }


def _warmth_word(delta: float) -> str:
    size = abs(delta)
    scale = "a touch" if size < 4 else ("noticeably" if size < 12 else "a lot")
    return f"{scale} {'warmer' if delta > 0 else 'cooler'}"


def deltas_to_steps(cur: dict, tgt: dict) -> list[dict[str, Any]]:
    """Ordered slider instructions that move ``cur`` toward ``tgt``.

    Ordered by how much of the visible difference each step accounts for, so
    the biggest correction comes first. That matters in the refine loop: a
    photographer who applies only the first step still gets most of the way.

    Every step says what to change in plain language (``why``) and keeps the
    measured numbers separately (``detail``), so the advice is usable without
    knowing what CIELAB is.
    """
    steps: list[dict[str, Any]] = []
    steps += _white_balance_steps(cur, tgt)
    steps += _tone_steps(cur, tgt)
    steps += _presence_steps(cur, tgt)
    steps += _colour_grading_steps(cur, tgt)
    steps += _hsl_steps(cur, tgt)
    steps.sort(key=lambda step: -step["magnitude"])
    return steps


# --- Basic panel: white balance --------------------------------------------

def _white_balance_steps(cur: dict, tgt: dict) -> list[dict[str, Any]]:
    steps = []
    cur_b, tgt_b = float(cur.get("b_global", 0.0)), float(tgt.get("b_global", 0.0))
    d_b = tgt_b - cur_b
    if abs(d_b) > THRESHOLDS["temp"]:
        steps.append(
            _step(
                "Temp",
                _clamp(d_b * COEFFICIENTS["temp"]),
                f"The look you want is {_warmth_word(d_b)} than your photo. "
                f"{'Push toward yellow' if d_b > 0 else 'Push toward blue'}.",
                abs(d_b) * 1.5,
                panel="Basic",
                detail=f"blue-yellow axis {cur_b:+.1f} -> {tgt_b:+.1f}",
            )
        )

    cur_a, tgt_a = float(cur.get("a_global", 0.0)), float(tgt.get("a_global", 0.0))
    d_a = tgt_a - cur_a
    if abs(d_a) > THRESHOLDS["tint"]:
        steps.append(
            _step(
                "Tint",
                _clamp(d_a * COEFFICIENTS["tint"]),
                f"Your photo needs {'more magenta' if d_a > 0 else 'more green'} "
                f"to match. This is the smaller white-balance dial, under Temp.",
                abs(d_a) * 1.2,
                panel="Basic",
                detail=f"green-magenta axis {cur_a:+.1f} -> {tgt_a:+.1f}",
            )
        )
    return steps


# --- Basic panel: tone ------------------------------------------------------

def _tone_steps(cur: dict, tgt: dict) -> list[dict[str, Any]]:
    steps = []
    cur_c, tgt_c = float(cur.get("contrast", 0.0)), float(tgt.get("contrast", 0.0))
    d_c = tgt_c - cur_c
    if abs(d_c) > THRESHOLDS["contrast"]:
        steps.append(
            _step(
                "Contrast",
                _clamp(d_c * COEFFICIENTS["contrast"]),
                "The look has a bigger gap between its darkest and brightest areas."
                if d_c > 0
                else "The look is flatter -- less separation between dark and bright.",
                abs(d_c) * 0.5,
                panel="Basic",
                detail=f"dark-to-bright spread {cur_c:.0f} -> {tgt_c:.0f} (of 100)",
            )
        )

    cur_p05, tgt_p05 = float(cur.get("L_p05", 0.0)), float(tgt.get("L_p05", 0.0))
    d_p05 = tgt_p05 - cur_p05
    if abs(d_p05) > THRESHOLDS["blacks"]:
        steps.append(
            _step(
                "Blacks",
                _clamp(d_p05 * COEFFICIENTS["blacks"]),
                "The darkest parts should sit deeper, closer to true black."
                if d_p05 < 0
                else "Lift the darkest parts -- this look has softer, milkier shadows.",
                abs(d_p05) * 0.6,
                panel="Basic",
                detail=f"darkest tones {cur_p05:.0f} -> {tgt_p05:.0f} (of 100)",
            )
        )

    cur_p95, tgt_p95 = float(cur.get("L_p95", 0.0)), float(tgt.get("L_p95", 0.0))
    d_p95 = tgt_p95 - cur_p95
    if abs(d_p95) > THRESHOLDS["whites"]:
        steps.append(
            _step(
                "Whites",
                _clamp(d_p95 * COEFFICIENTS["whites"]),
                "The brightest parts should go brighter."
                if d_p95 > 0
                else "Pull the brightest parts back -- they are too hot for this look.",
                abs(d_p95) * 0.5,
                panel="Basic",
                detail=f"brightest tones {cur_p95:.0f} -> {tgt_p95:.0f} (of 100)",
            )
        )

    # Shadows and Highlights, from the zone lightness rather than the endpoints.
    for zone, slider, coeff, where in (
        ("shadow", "Shadows", "shadows", "the dark areas"),
        ("high", "Highlights", "highlights", "the bright areas"),
    ):
        if not (zone_is_measurable(cur, zone) and zone_is_measurable(tgt, zone)):
            continue
        # Zone lightness is not stored directly; approximate it from the
        # endpoint that bounds that zone.
        key = "L_p05" if zone == "shadow" else "L_p95"
        delta = float(tgt.get(key, 0.0)) - float(cur.get(key, 0.0))
        if abs(delta) <= 6.0:
            continue
        steps.append(
            _step(
                slider,
                _clamp(delta * COEFFICIENTS[coeff] * 0.6),
                f"Open up {where}." if delta > 0 else f"Bring {where} down.",
                abs(delta) * 0.25,
                panel="Basic",
                detail=f"{where} differ by {delta:+.0f} points",
            )
        )
    return steps


# --- Basic panel: presence --------------------------------------------------

def _presence_steps(cur: dict, tgt: dict) -> list[dict[str, Any]]:
    cur_ch, tgt_ch = float(cur.get("chroma", 0.0)), float(tgt.get("chroma", 0.0))
    d_chroma = tgt_ch - cur_ch
    if abs(d_chroma) <= THRESHOLDS["saturation"]:
        return []
    # Vibrance for a nudge, Saturation for a real move: Vibrance protects
    # colours that are already strong, which is what you want when adjusting
    # a photo with skin in it.
    small = abs(d_chroma) < 4.0
    slider = "Vibrance" if small else "Saturation"
    if d_chroma > 0:
        why = ("Colours should be a little stronger. Vibrance lifts the muted "
               "colours and mostly leaves skin tones alone."
               if small else "Colours should be noticeably stronger overall.")
    else:
        why = ("Pull the colour back slightly -- this look is more restrained."
               if small else "This look is much more muted. Take the colour out.")
    return [
        _step(
            slider,
            _clamp(d_chroma * COEFFICIENTS["saturation"]),
            why,
            abs(d_chroma) * 0.8,
            panel="Basic",
            detail=f"colour strength {cur_ch:.1f} -> {tgt_ch:.1f}",
        )
    ]


# --- Colour Grading wheels: shadows, midtones, highlights, global -----------

def _colour_grading_steps(cur: dict, tgt: dict) -> list[dict[str, Any]]:
    """One step per Colour Grading wheel whose tint genuinely differs."""
    steps = []
    wheels = (
        ("shadow", "Shadows", "the dark areas", "shadow_hue", "shadow_C", 0.12),
        ("mid", "Midtones", "the mid-tones", "mid_hue", "mid_C", 0.10),
        ("high", "Highlights", "the bright areas", "high_hue", "high_C", 0.10),
    )
    for zone, wheel, where, hue_key, chroma_key, weight in wheels:
        if not (zone_is_measurable(cur, zone) and zone_is_measurable(tgt, zone)):
            continue
        threshold = THRESHOLDS.get(f"{zone}_hue", 15.0)
        gap = circ_dist(float(tgt.get(hue_key, 0.0)), float(cur.get(hue_key, 0.0)))
        target_chroma = float(tgt.get(chroma_key, 0.0))
        if gap <= threshold or target_chroma <= 3.0:
            continue
        wheel_hue = wheel_hue_for_lab_hue(float(tgt[hue_key]))
        steps.append(
            _step(
                f"Color Grading > {wheel}",
                f"hue {wheel_hue}, sat {_clamp(target_chroma * COEFFICIENTS['cg_sat'])}",
                f"Tint {where} toward {_colour_name(float(tgt[hue_key]))}. "
                f"This is the biggest single thing that makes the look "
                f"recognisable." if zone == "shadow" else
                f"Tint {where} toward {_colour_name(float(tgt[hue_key]))}.",
                gap * weight,
                panel="Color Grading",
                detail=f"{where} are about {gap:.0f} degrees of hue apart",
            )
        )

    # Global wheel: a cast that runs through every tone equally.
    both = all(zone_is_measurable(sig, "shadow") and zone_is_measurable(sig, "high")
               for sig in (cur, tgt))
    if both:
        cur_split = float(cur.get("split", 0.0))
        tgt_split = float(tgt.get("split", 0.0))
        # A LOW split in the target means shadows and highlights are tinted the
        # same way -- that is a global cast, not a split tone.
        if tgt_split < 25.0 and cur_split > 45.0 and float(tgt.get("chroma", 0.0)) > 5.0:
            wheel_hue = wheel_hue_for_lab_hue(float(tgt.get("shadow_hue", 0.0)))
            steps.append(
                _step(
                    "Color Grading > Global",
                    f"hue {wheel_hue}, sat {_clamp(float(tgt.get('chroma', 0.0)) * 0.8)}",
                    "This look puts the same tint through the whole picture "
                    "rather than splitting warm and cool.",
                    8.0,
                    panel="Color Grading",
                    detail=f"tint spread across tones {cur_split:.0f} -> {tgt_split:.0f} degrees",
                )
            )
    return steps


def _colour_name(lab_hue: float) -> str:
    """Nearest everyday colour word for a CIELAB hue angle."""
    best, best_gap = "neutral", 1e9
    for name, angle in LAB_HUE.items():
        gap = circ_dist(lab_hue, angle)
        if gap < best_gap:
            best, best_gap = name, gap
    return best


# --- Colour Mixer / HSL panel ----------------------------------------------

def _hsl_steps(cur: dict, tgt: dict) -> list[dict[str, Any]]:
    """Per-colour-family advice, in the vocabulary of Lightroom's HSL panel.

    Only families that occupy a real share of BOTH frames are considered --
    otherwise the engine would confidently tell you to move the Purple slider
    because of a few dozen pixels.
    """
    cur_hsl = cur.get("hsl") or {}
    tgt_hsl = tgt.get("hsl") or {}
    if not cur_hsl or not tgt_hsl:
        return []

    steps = []
    for family in cur_hsl:
        c = cur_hsl.get(family) or {}
        t = tgt_hsl.get(family) or {}
        if min(float(c.get("coverage", 0.0)), float(t.get("coverage", 0.0))) < HSL_MIN_COVERAGE:
            continue

        d_sat = float(t.get("saturation", 0.0)) - float(c.get("saturation", 0.0))
        if abs(d_sat) > THRESHOLDS["hsl_sat"]:
            steps.append(
                _step(
                    f"Color Mixer > {family.title()} > Saturation",
                    _clamp(d_sat * COEFFICIENTS["hsl_sat"]),
                    f"The {family} in your photo is "
                    f"{'weaker' if d_sat > 0 else 'stronger'} than the look wants.",
                    abs(d_sat) * 0.35,
                    panel="Color Mixer",
                    detail=f"{family} colour strength "
                    f"{float(c.get('saturation', 0)):.0f} -> {float(t.get('saturation', 0)):.0f}",
                )
            )

        d_lum = float(t.get("luminance", 0.0)) - float(c.get("luminance", 0.0))
        if abs(d_lum) > THRESHOLDS["hsl_lum"]:
            steps.append(
                _step(
                    f"Color Mixer > {family.title()} > Luminance",
                    _clamp(d_lum * COEFFICIENTS["hsl_lum"]),
                    f"The {family} areas should be "
                    f"{'brighter' if d_lum > 0 else 'darker'} without changing "
                    f"anything else.",
                    abs(d_lum) * 0.3,
                    panel="Color Mixer",
                    detail=f"{family} brightness "
                    f"{float(c.get('luminance', 0)):.0f} -> {float(t.get('luminance', 0)):.0f}",
                )
            )

        d_hue = float(t.get("hue_shift", 0.0)) - float(c.get("hue_shift", 0.0))
        if abs(d_hue) > THRESHOLDS["hsl_hue"]:
            steps.append(
                _step(
                    f"Color Mixer > {family.title()} > Hue",
                    _clamp(d_hue * COEFFICIENTS["hsl_hue"]),
                    f"Shift the {family} itself slightly -- it is the right "
                    f"strength but the wrong shade.",
                    abs(d_hue) * 0.22,
                    panel="Color Mixer",
                    detail=f"{family} hue differs by {d_hue:+.0f} degrees",
                )
            )
    # Keep the HSL section from swamping the answer: it can produce 24 steps.
    steps.sort(key=lambda step: -step["magnitude"])
    return steps[:4]


def distance_to_target(cur: dict, tgt: dict, scales: dict | None = None) -> float:
    """Weighted distance from the current signature to the target."""
    return weighted_distance(cur, tgt, MATCH_WEIGHTS, scales)


def has_converged(distance: float) -> bool:
    return float(distance) <= CONVERGENCE_TOLERANCE


# --------------------------------------------------------------------------
# Critique: technical faults, checked deterministically
# --------------------------------------------------------------------------

def technical_faults(sig: dict) -> list[dict[str, Any]]:
    """Objective problems in an image, independent of taste."""
    faults: list[dict[str, Any]] = []

    if float(sig.get("clip_black", 0.0)) > 0.02:
        faults.append(
            {
                "issue": "Crushed blacks",
                "detail": f"{float(sig['clip_black']) * 100:.1f}% of pixels are at L* < 1 "
                f"with no recoverable detail",
                "fix": "raise Blacks, or reduce Contrast",
                "severity": "high" if float(sig["clip_black"]) > 0.06 else "medium",
            }
        )

    if float(sig.get("clip_white", 0.0)) > 0.02:
        faults.append(
            {
                "issue": "Blown highlights",
                "detail": f"{float(sig['clip_white']) * 100:.1f}% of pixels are at L* > 99",
                "fix": "pull Highlights down, or reduce Whites",
                "severity": "high" if float(sig["clip_white"]) > 0.06 else "medium",
            }
        )

    contrast = float(sig.get("contrast", 0.0))
    if contrast < 25.0:
        faults.append(
            {
                "issue": "Flat tonal range",
                "detail": f"L* spans only {contrast:.0f} points between p05 and p95",
                "fix": "add Contrast, or lower Blacks to re-anchor the shadow end",
                "severity": "medium",
            }
        )
    elif contrast > 88.0:
        faults.append(
            {
                "issue": "Very high contrast",
                "detail": f"L* spans {contrast:.0f} points, close to the full range",
                "fix": "recover Highlights and lift Shadows if detail is being lost",
                "severity": "low",
            }
        )

    chroma = float(sig.get("chroma", 0.0))
    if chroma > 42.0:
        faults.append(
            {
                "issue": "Oversaturated",
                "detail": f"mean chroma {chroma:.0f} is high enough that colours will "
                f"clip on print and on wide-gamut displays",
                "fix": "pull Saturation back and use Vibrance instead",
                "severity": "medium",
            }
        )
    elif chroma < 4.0:
        faults.append(
            {
                "issue": "Nearly monochrome",
                "detail": f"mean chroma {chroma:.1f} -- if this is not deliberate, the "
                f"white balance or Saturation is suppressing all colour",
                "fix": "raise Vibrance, or check the Saturation slider is not at -100",
                "severity": "low",
            }
        )

    # A strong cast in an otherwise coherent frame usually means the white
    # balance is wrong rather than that a grade was applied.
    a_g, b_g = float(sig.get("a_global", 0.0)), float(sig.get("b_global", 0.0))
    if abs(b_g) > 18.0:
        faults.append(
            {
                "issue": f"Strong {'yellow' if b_g > 0 else 'blue'} cast",
                "detail": f"global b* is {b_g:+.1f}",
                "fix": f"move Temp {'down' if b_g > 0 else 'up'} unless the cast is intentional",
                "severity": "medium",
            }
        )
    if abs(a_g) > 14.0:
        faults.append(
            {
                "issue": f"Strong {'magenta' if a_g > 0 else 'green'} cast",
                "detail": f"global a* is {a_g:+.1f}",
                "fix": f"move Tint {'down' if a_g > 0 else 'up'}",
                "severity": "medium",
            }
        )

    return faults


def taste_notes(sig: dict, profile: dict) -> list[dict[str, Any]]:
    """Subjective notes, checked against the stored long-term taste profile.

    Kept separate from ``technical_faults`` on purpose: one is measurable and
    the other depends on what this specific user has said they like. The
    profile comes from the long-term store, so these notes are the visible
    payoff of Topic 4.
    """
    notes: list[dict[str, Any]] = []
    if not profile:
        return notes

    def _val(key: str) -> str:
        entry = profile.get(key)
        if isinstance(entry, dict):
            return str(entry.get("value", "")).lower()
        return str(entry or "").lower()

    preferred = _val("preferred_look")
    dislikes = _val("dislikes")
    b_g = float(sig.get("b_global", 0.0))
    shadow_hue = float(sig.get("shadow_hue", 0.0))
    shadow_c = float(sig.get("shadow_C", 0.0))

    if preferred:
        if "warm" in preferred and b_g < NEUTRAL_B_STAR:
            nudge = max(1, abs(round((NEUTRAL_B_STAR - b_g) * COEFFICIENTS["temp"])))
            notes.append(
                {
                    "note": f"You said you prefer {preferred}, but this edit reads cool "
                    f"(b* {b_g:+.1f}, against about {NEUTRAL_B_STAR:+.0f} for an untouched "
                    f"photo). Temp +{nudge} would bring it back toward warm.",
                    "source": "profile.preferred_look",
                }
            )
        if "cool" in preferred and b_g > NEUTRAL_B_STAR + 6:
            notes.append(
                {
                    "note": f"You said you prefer {preferred}, but this edit is warm "
                    f"(b* {b_g:+.1f}).",
                    "source": "profile.preferred_look",
                }
            )
        if "lifted" in preferred and float(sig.get("L_p05", 0.0)) < 6:
            notes.append(
                {
                    "note": f"You like lifted shadows, but L* p05 is "
                    f"{float(sig.get('L_p05', 0)):.0f} -- the shadows are sitting on the floor.",
                    "source": "profile.preferred_look",
                }
            )

    if dislikes and "teal" in dislikes and shadow_c > 4:
        # LAB_HUE["teal"] is measured, not assumed -- see the table above.
        if circ_dist(shadow_hue, LAB_HUE["teal"]) < 45.0:
            notes.append(
                {
                    "note": f"Heads up -- you told me you dislike {dislikes}, and the shadows "
                    f"here are tinted teal (hue {shadow_hue:.0f} degrees, chroma {shadow_c:.1f}).",
                    "source": "profile.dislikes",
                }
            )

    return notes
