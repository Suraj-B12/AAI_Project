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
}

# Below these deltas, a step is noise rather than advice and is not emitted.
THRESHOLDS: dict[str, float] = {
    "temp": 1.5,  # b* units
    "tint": 1.5,  # a* units
    "contrast": 3.0,  # L* units
    "blacks": 2.0,
    "whites": 2.0,
    "shadow_hue": 15.0,  # degrees
    "high_hue": 15.0,
    "saturation": 1.5,  # chroma units
}

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


def _step(slider: str, amount: Any, why: str, magnitude: float) -> dict[str, Any]:
    return {"slider": slider, "amount": amount, "why": why, "magnitude": round(float(magnitude), 3)}


def deltas_to_steps(cur: dict, tgt: dict) -> list[dict[str, Any]]:
    """Ordered slider instructions that move ``cur`` toward ``tgt``.

    Ordered by how much of the visible difference each step accounts for, so
    the largest correction is applied first. That matters in the refine loop:
    a photographer applying only the first step still gets most of the way.
    """
    steps: list[dict[str, Any]] = []

    # --- White balance: the a*/b* axes are literally the Tint/Temp axes -----
    d_b = float(tgt.get("b_global", 0.0)) - float(cur.get("b_global", 0.0))
    if abs(d_b) > THRESHOLDS["temp"]:
        steps.append(
            _step(
                "Temp",
                _clamp(d_b * COEFFICIENTS["temp"]),
                f"target is {'warmer' if d_b > 0 else 'cooler'} overall "
                f"(b* {float(cur.get('b_global', 0)):+.1f} to {float(tgt.get('b_global', 0)):+.1f})",
                abs(d_b) * 1.5,
            )
        )

    d_a = float(tgt.get("a_global", 0.0)) - float(cur.get("a_global", 0.0))
    if abs(d_a) > THRESHOLDS["tint"]:
        steps.append(
            _step(
                "Tint",
                _clamp(d_a * COEFFICIENTS["tint"]),
                f"target leans {'magenta' if d_a > 0 else 'green'} "
                f"(a* {float(cur.get('a_global', 0)):+.1f} to {float(tgt.get('a_global', 0)):+.1f})",
                abs(d_a) * 1.2,
            )
        )

    # --- Tone ---------------------------------------------------------------
    d_c = float(tgt.get("contrast", 0.0)) - float(cur.get("contrast", 0.0))
    if abs(d_c) > THRESHOLDS["contrast"]:
        steps.append(
            _step(
                "Contrast",
                _clamp(d_c * COEFFICIENTS["contrast"]),
                f"tonal range differs by {d_c:+.1f} L* "
                f"({float(cur.get('contrast', 0)):.0f} to {float(tgt.get('contrast', 0)):.0f})",
                abs(d_c) * 0.5,
            )
        )

    d_p05 = float(tgt.get("L_p05", 0.0)) - float(cur.get("L_p05", 0.0))
    if abs(d_p05) > THRESHOLDS["blacks"]:
        steps.append(
            _step(
                "Blacks",
                _clamp(d_p05 * COEFFICIENTS["blacks"]),
                f"target has {'denser' if d_p05 < 0 else 'lifted'} blacks "
                f"(L* p05 {float(cur.get('L_p05', 0)):.0f} to {float(tgt.get('L_p05', 0)):.0f})",
                abs(d_p05) * 0.6,
            )
        )

    d_p95 = float(tgt.get("L_p95", 0.0)) - float(cur.get("L_p95", 0.0))
    if abs(d_p95) > THRESHOLDS["whites"]:
        steps.append(
            _step(
                "Whites",
                _clamp(d_p95 * COEFFICIENTS["whites"]),
                f"target's brightest tones sit {d_p95:+.0f} L* from yours",
                abs(d_p95) * 0.5,
            )
        )

    # --- Colour grading: the split tone is usually the defining difference --
    if zone_is_measurable(cur, "shadow") and zone_is_measurable(tgt, "shadow"):
        hue_gap = circ_dist(float(tgt.get("shadow_hue", 0.0)), float(cur.get("shadow_hue", 0.0)))
        if hue_gap > THRESHOLDS["shadow_hue"] and float(tgt.get("shadow_C", 0.0)) > 3.0:
            steps.append(
                _step(
                    "Color Grading > Shadows",
                    # The measured LAB hue angle is NOT the wheel value -- the
                    # two differ by 9 to 59 degrees depending on the angle.
                    f"hue {wheel_hue_for_lab_hue(float(tgt['shadow_hue']))}, "
                    f"sat {_clamp(float(tgt['shadow_C']) * COEFFICIENTS['cg_sat'])}",
                    f"shadow tint is the defining difference "
                    f"({hue_gap:.0f} degrees of hue apart)",
                    hue_gap * 0.12,
                )
            )

    if zone_is_measurable(cur, "high") and zone_is_measurable(tgt, "high"):
        hue_gap = circ_dist(float(tgt.get("high_hue", 0.0)), float(cur.get("high_hue", 0.0)))
        if hue_gap > THRESHOLDS["high_hue"] and float(tgt.get("high_C", 0.0)) > 3.0:
            steps.append(
                _step(
                    "Color Grading > Highlights",
                    f"hue {wheel_hue_for_lab_hue(float(tgt['high_hue']))}, "
                    f"sat {_clamp(float(tgt['high_C']) * COEFFICIENTS['cg_sat'])}",
                    f"highlights carry a different tint ({hue_gap:.0f} degrees apart)",
                    hue_gap * 0.10,
                )
            )

    # --- Presence -----------------------------------------------------------
    d_chroma = float(tgt.get("chroma", 0.0)) - float(cur.get("chroma", 0.0))
    if abs(d_chroma) > THRESHOLDS["saturation"]:
        # Vibrance below the midpoint, Saturation above it: vibrance protects
        # already-saturated colour, which is what you want for a small nudge.
        slider = "Vibrance" if abs(d_chroma) < 4.0 else "Saturation"
        steps.append(
            _step(
                slider,
                _clamp(d_chroma * COEFFICIENTS["saturation"]),
                f"target is {'more' if d_chroma > 0 else 'less'} colourful "
                f"(mean chroma {float(cur.get('chroma', 0)):.1f} to {float(tgt.get('chroma', 0)):.1f})",
                abs(d_chroma) * 0.8,
            )
        )

    steps.sort(key=lambda s: -s["magnitude"])
    return steps


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
