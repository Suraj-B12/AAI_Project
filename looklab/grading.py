"""A deterministic Lightroom-style slider simulator, in numpy.

Why this file exists
--------------------
The design document's Part 6 asks for a knowledge base built by hand in
Lightroom: three base plates, twelve to fifteen looks, slider values recorded
as you build. That is the only part of the project that cannot be done at a
keyboard, and it is the part that Part 9.2's calibration test depends on --
without recorded slider values there is no ground truth to validate against.

This module removes that dependency by implementing the sliders themselves.
Given a base image and a dict of slider values, ``apply_grade`` produces the
graded image. That gives us (a) a reproducible knowledge base and (b) genuine
ground truth for calibration, because we know exactly which slider values
produced each graded frame.

On circularity -- read this before defending the calibration numbers
--------------------------------------------------------------------
There is an obvious objection: if the same codebase both applies and measures
the grade, is the calibration test measuring anything?

It is, and the reason is that the two halves live in different colour spaces
and are not inverses of each other:

* This simulator works in **sRGB / linear-light RGB**: channel gains, tone
  curves, luminance-masked offsets. It never touches LAB.
* ``color.signature`` measures in **CIELAB**: zone means of a*/b*, chroma-
  weighted circular hue means, L* percentiles.
* ``rules.deltas_to_steps`` maps a LAB delta back to slider units through
  hand-chosen linear coefficients.

So the path slider -> pixels -> LAB -> slider passes through a nonlinear,
image-dependent transformation in each direction. Recovering "Temp +20" from
the measured LAB shift is a real estimation problem, not an algebraic
identity: the same Temp value produces a different b* shift on a dark frame
than on a bright one, which is precisely the non-invertibility the design
document is honest about.

What the calibration test therefore does *not* prove is that these sliders
behave identically to Adobe's. They are modelled on Lightroom's documented
behaviour and share its sign conventions and rough scaling, but they are an
approximation. The claim the numbers support is "the delta engine recovers
the grading operations it was validated against", not "these are Lightroom's
exact transfer functions".

Slider conventions follow Lightroom: most run -100..+100, Temp and Tint are
signed offsets where positive Temp is warmer (yellow) and positive Tint is
more magenta, Exposure is in stops.
"""

from __future__ import annotations

import numpy as np

# Every slider the simulator understands, with its neutral value.
SLIDER_DEFAULTS: dict[str, float] = {
    "temp": 0.0,  # -100 (cool/blue) .. +100 (warm/yellow)
    "tint": 0.0,  # -100 (green) .. +100 (magenta)
    "exposure": 0.0,  # stops
    "contrast": 0.0,  # -100 .. +100
    "highlights": 0.0,  # -100 (recover) .. +100 (brighten)
    "shadows": 0.0,  # -100 (crush) .. +100 (lift)
    "whites": 0.0,  # -100 .. +100
    "blacks": 0.0,  # -100 .. +100
    "vibrance": 0.0,  # -100 .. +100, chroma-weighted saturation
    "saturation": 0.0,  # -100 .. +100, uniform
    "cg_shadow_hue": 0.0,  # 0..360 degrees
    "cg_shadow_sat": 0.0,  # 0..100
    "cg_high_hue": 0.0,  # 0..360 degrees
    "cg_high_sat": 0.0,  # 0..100
}

# Sliders whose value is an angle, so "0" means "no hue chosen" rather than
# "neutral"; they only take effect when the paired saturation is non-zero.
HUE_SLIDERS = frozenset({"cg_shadow_hue", "cg_high_hue"})

# Rec.709 luma weights, applied to linear-light RGB (used for white-balance
# renormalisation, where linear light is the physically correct space).
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)

# Rec.601 luma weights, applied to gamma-encoded sRGB. Tone masks and the
# saturation pivot both use this: it is what raster editors actually do, and
# it keeps mid-grey at ~0.5 so a "highlights" mask covers the tones a
# photographer calls highlights rather than only near-white pixels.
_LUMA_G = np.array([0.299, 0.587, 0.114], dtype=np.float64)


def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1.0 / 2.4)) - 0.055)


def _luma(lin: np.ndarray) -> np.ndarray:
    """Rec.709 luma of a linear-light RGB array."""
    return np.tensordot(lin, _LUMA, axes=([-1], [0]))


def _luma_gamma(img: np.ndarray) -> np.ndarray:
    """Rec.601 luma of a gamma-encoded sRGB array, in [0, 1]."""
    return np.tensordot(img, _LUMA_G, axes=([-1], [0]))


def _smooth_mask(y: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Smoothstep ramp from 0 at ``lo`` to 1 at ``hi`` (works either direction)."""
    if hi == lo:
        return np.where(y >= hi, 1.0, 0.0)
    t = np.clip((y - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _hue_to_unit_rgb(hue_deg: float) -> np.ndarray:
    """Fully saturated RGB for a hue angle, mean-centred so it adds tint, not light."""
    h = (float(hue_deg) % 360.0) / 60.0
    i = int(np.floor(h)) % 6
    f = h - np.floor(h)
    q, t = 1.0 - f, f
    table = [
        (1.0, t, 0.0),
        (q, 1.0, 0.0),
        (0.0, 1.0, t),
        (0.0, q, 1.0),
        (t, 0.0, 1.0),
        (1.0, 0.0, q),
    ]
    rgb = np.array(table[i], dtype=np.float64)
    return rgb - rgb.mean()


_WHEEL_TABLE: list[tuple[float, float]] | None = None


def _wheel_table() -> list[tuple[float, float]]:
    """Measured mapping from colour-grading wheel hue to resulting CIELAB hue.

    These are two different coordinate systems and the relationship between
    them is neither the identity nor a constant offset -- measured, it ranges
    from +9 to +59 degrees across the wheel. Computed once by tinting a neutral
    grey at each wheel angle and measuring the LAB hue that results, so it
    stays correct if the split-tone operation ever changes.
    """
    global _WHEEL_TABLE
    if _WHEEL_TABLE is None:
        import numpy as _np

        from .cielab import rgb2lab

        table = []
        for wheel in range(0, 360, 5):
            grey = _np.full((4, 4, 3), 0.5)
            tinted = _np.clip(grey + 0.30 * _hue_to_unit_rgb(wheel), 0.0, 1.0)
            lab = rgb2lab(tinted)
            a, b = float(lab[..., 1].mean()), float(lab[..., 2].mean())
            table.append((float(_np.degrees(_np.arctan2(b, a)) % 360.0), float(wheel)))
        _WHEEL_TABLE = table
    return _WHEEL_TABLE


def wheel_hue_for_lab_hue(lab_hue: float) -> int:
    """Convert a measured CIELAB hue angle to the wheel value to dial in.

    Without this the engine recommends a LAB angle as though it were a
    Lightroom colour-grading hue, which is simply the wrong unit: asking for
    "hue 296" when the wheel wants 240 sends the user to a different colour.
    """
    target = float(lab_hue) % 360.0
    table = _wheel_table()
    best_wheel, best_err = 0.0, 1e9
    for measured_lab, wheel in table:
        d = abs(measured_lab - target) % 360.0
        err = min(d, 360.0 - d)
        if err < best_err:
            best_wheel, best_err = wheel, err
    return int(round(best_wheel)) % 360


def normalise_sliders(sliders: dict | None) -> dict[str, float]:
    """Fill in defaults and drop unknown keys, so callers can pass partial dicts."""
    out = dict(SLIDER_DEFAULTS)
    for k, v in (sliders or {}).items():
        if k in out:
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                continue
    return out


def apply_grade(rgb01: np.ndarray, sliders: dict | None) -> np.ndarray:
    """Apply a slider dict to a float RGB image in [0, 1]; return a new array.

    Ordering mirrors Lightroom's pipeline: white balance, then exposure and
    tone, then presence, then colour grading. Order matters -- applying
    saturation before a tone curve gives a different result than after.
    """
    s = normalise_sliders(sliders)
    img = np.clip(np.asarray(rgb01, dtype=np.float64), 0.0, 1.0)

    # --- 1. White balance: channel gains in linear light -------------------
    lin = _srgb_to_linear(img)
    temp, tint = s["temp"], s["tint"]
    if temp or tint:
        # +Temp warms: red gain up, blue gain down. +Tint goes magenta: green down.
        gain = np.array(
            [
                1.0 + 0.0045 * temp + 0.0012 * tint,
                1.0 - 0.0022 * tint,
                1.0 - 0.0045 * temp + 0.0012 * tint,
            ],
            dtype=np.float64,
        )
        gain = np.clip(gain, 0.05, 4.0)
        # Renormalise on luma so white balance does not double as an exposure move.
        gain = gain / float(np.dot(gain, _LUMA))
        lin = lin * gain

    # --- 2. Exposure -------------------------------------------------------
    if s["exposure"]:
        lin = lin * (2.0 ** s["exposure"])

    lin = np.clip(lin, 0.0, 1.0)
    img = _linear_to_srgb(lin)

    # --- 3. Tone: shadows / highlights / blacks / whites -------------------
    # Masks run on gamma-space luma so mid-grey sits near 0.5. On linear luma
    # mid-grey is ~0.21, which would push every "highlight" mask up into the
    # near-white pixels only and make the slider a no-op on normal images.
    y = _luma_gamma(img)

    if s["shadows"]:
        amt = s["shadows"] / 100.0
        m = 1.0 - _smooth_mask(y, 0.0, 0.5)  # strongest in deep shadow
        img = img + (amt * 0.32) * m[..., None] * (1.0 - img if amt > 0 else img)

    if s["highlights"]:
        amt = s["highlights"] / 100.0
        m = _smooth_mask(y, 0.5, 1.0)  # strongest in highlight
        img = img + (amt * 0.32) * m[..., None] * (1.0 - img if amt > 0 else img)

    if s["blacks"]:
        amt = s["blacks"] / 100.0
        m = 1.0 - _smooth_mask(y, 0.0, 0.28)  # tighter than shadows: endpoint move
        img = img + (amt * 0.30) * m[..., None] * (1.0 - img if amt > 0 else img)

    if s["whites"]:
        amt = s["whites"] / 100.0
        m = _smooth_mask(y, 0.72, 1.0)
        img = img + (amt * 0.30) * m[..., None] * (1.0 - img if amt > 0 else img)

    img = np.clip(img, 0.0, 1.0)

    # --- 4. Contrast: S-curve about mid-grey in gamma space ----------------
    if s["contrast"]:
        c = s["contrast"] / 100.0
        pivot = 0.5
        d = img - pivot
        # Cubic S-curve: monotone for |c| <= 1, preserves the pivot and endpoints.
        img = np.clip(pivot + d + c * 0.9 * d * (1.0 - 4.0 * d * d) * 0.5 + c * 0.35 * d, 0.0, 1.0)

    # --- 5. Presence: vibrance then saturation -----------------------------
    # The pivot must be luma in the SAME space as the pixels being scaled,
    # otherwise scaling ``img - lum`` shifts lightness as a side effect
    # instead of only moving colourfulness.
    if s["vibrance"] or s["saturation"]:
        lum = _luma_gamma(img)[..., None]
        chroma = img - lum
        if s["vibrance"]:
            v = s["vibrance"] / 100.0
            # Weight by how unsaturated a pixel already is -- vibrance protects
            # already-saturated colours (and, in Lightroom, skin tones).
            mag = np.abs(chroma).max(axis=-1, keepdims=True)
            w = np.clip(1.0 - mag / 0.45, 0.0, 1.0)
            chroma = chroma * (1.0 + v * 0.85 * w)
        if s["saturation"]:
            chroma = chroma * (1.0 + s["saturation"] / 100.0)
        img = np.clip(lum + chroma, 0.0, 1.0)

    # --- 6. Colour grading (split toning) ----------------------------------
    y = _luma_gamma(img)
    if s["cg_shadow_sat"]:
        strength = np.clip(s["cg_shadow_sat"], 0.0, 100.0) / 100.0
        m = 1.0 - _smooth_mask(y, 0.0, 0.55)
        img = img + (0.42 * strength) * m[..., None] * _hue_to_unit_rgb(s["cg_shadow_hue"])
    if s["cg_high_sat"]:
        strength = np.clip(s["cg_high_sat"], 0.0, 100.0) / 100.0
        m = _smooth_mask(y, 0.45, 1.0)
        img = img + (0.42 * strength) * m[..., None] * _hue_to_unit_rgb(s["cg_high_hue"])

    return np.clip(img, 0.0, 1.0)
