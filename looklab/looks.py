"""The reference look library: descriptive names and their ground-truth sliders.

These slider dicts are the ground truth for the calibration test in
``tools/calibrate.py``. They play the role of "record the slider values as you
build" from Part 6 of the design document -- except that here the values are
authored first and the graded frames are produced from them by
``grading.apply_grade``, rather than the other way round.

Names are descriptive only. None of these claims to emulate a specific film
stock or a commercial preset, because none of them were measured against one.
"""

from __future__ import annotations

from typing import Any

LOOKS: list[dict[str, Any]] = [
    {
        "id": "warm_golden",
        "label": "Warm Golden",
        "notes": "Lifted warm shadows, yellow highlights, gentle contrast.",
        "sliders": {
            "temp": 22, "tint": 5, "contrast": 8, "highlights": -18, "shadows": 20,
            "whites": 6, "blacks": -6, "vibrance": 12, "saturation": -3,
            "cg_shadow_hue": 45, "cg_shadow_sat": 18, "cg_high_hue": 52, "cg_high_sat": 12,
        },
    },
    {
        "id": "teal_orange",
        "label": "Teal and Orange",
        "notes": "Cool teal shadows against warm skin. The blockbuster split.",
        "sliders": {
            "temp": 10, "tint": -4, "contrast": 18, "highlights": -22, "shadows": 10,
            "whites": 4, "blacks": -14, "vibrance": 18, "saturation": 0,
            "cg_shadow_hue": 195, "cg_shadow_sat": 32, "cg_high_hue": 35, "cg_high_sat": 22,
        },
    },
    {
        "id": "cool_blue_hour",
        "label": "Cool Blue Hour",
        "notes": "Deep blue cast, dense shadows, restrained colour.",
        "sliders": {
            "temp": -30, "tint": -6, "contrast": 12, "highlights": -10, "shadows": -8,
            "whites": -6, "blacks": -18, "vibrance": 4, "saturation": -10,
            "cg_shadow_hue": 225, "cg_shadow_sat": 26, "cg_high_hue": 210, "cg_high_sat": 10,
        },
    },
    {
        "id": "muted_matte",
        "label": "Muted Matte",
        "notes": "Faded film look: lifted blacks, low contrast, drained colour.",
        "sliders": {
            "temp": 4, "tint": 2, "contrast": -22, "highlights": -14, "shadows": 26,
            "whites": -12, "blacks": 30, "vibrance": -12, "saturation": -18,
            "cg_shadow_hue": 60, "cg_shadow_sat": 10, "cg_high_hue": 40, "cg_high_sat": 8,
        },
    },
    {
        "id": "high_contrast_punchy",
        "label": "High-Contrast Punchy",
        "notes": "Hard tonal range, saturated, no lift. Editorial.",
        "sliders": {
            "temp": 2, "tint": 0, "contrast": 42, "highlights": -30, "shadows": -16,
            "whites": 18, "blacks": -30, "vibrance": 22, "saturation": 8,
            "cg_shadow_hue": 0, "cg_shadow_sat": 0, "cg_high_hue": 0, "cg_high_sat": 0,
        },
    },
    {
        "id": "pastel_soft",
        "label": "Pastel Soft",
        "notes": "Bright, airy, low contrast, pink-leaning highlights.",
        "sliders": {
            "temp": 8, "tint": 8, "contrast": -16, "highlights": 12, "shadows": 30,
            "whites": 14, "blacks": 22, "vibrance": -8, "saturation": -12,
            "cg_shadow_hue": 330, "cg_shadow_sat": 12, "cg_high_hue": 20, "cg_high_sat": 16,
        },
    },
    {
        "id": "green_shadow",
        "label": "Green Shadow",
        "notes": "Olive-tinted shadows, neutral highlights. Thriller grade.",
        "sliders": {
            "temp": -6, "tint": -14, "contrast": 14, "highlights": -12, "shadows": 6,
            "whites": 0, "blacks": -12, "vibrance": 6, "saturation": -6,
            "cg_shadow_hue": 110, "cg_shadow_sat": 28, "cg_high_hue": 90, "cg_high_sat": 8,
        },
    },
    {
        "id": "sun_bleached",
        "label": "Sun-Bleached",
        "notes": "Overexposed highlights, washed colour, warm cast.",
        "sliders": {
            "temp": 18, "tint": 4, "contrast": -10, "highlights": 26, "shadows": 16,
            "whites": 24, "blacks": 12, "vibrance": -16, "saturation": -14,
            "cg_shadow_hue": 40, "cg_shadow_sat": 8, "cg_high_hue": 48, "cg_high_sat": 18,
        },
    },
    {
        "id": "deep_amber",
        "label": "Deep Amber",
        "notes": "Heavy warm cast with crushed shadows. Candlelit.",
        "sliders": {
            "temp": 38, "tint": 10, "contrast": 20, "highlights": -24, "shadows": -10,
            "whites": -4, "blacks": -24, "vibrance": 10, "saturation": 4,
            "cg_shadow_hue": 30, "cg_shadow_sat": 30, "cg_high_hue": 44, "cg_high_sat": 20,
        },
    },
    {
        "id": "neon_night",
        "label": "Neon Night",
        "notes": "Magenta highlights, cyan shadows, high saturation, dense blacks.",
        "sliders": {
            "temp": -14, "tint": 16, "contrast": 28, "highlights": -8, "shadows": -14,
            "whites": 8, "blacks": -28, "vibrance": 28, "saturation": 14,
            "cg_shadow_hue": 200, "cg_shadow_sat": 34, "cg_high_hue": 310, "cg_high_sat": 28,
        },
    },
    {
        "id": "desaturated_grey",
        "label": "Desaturated Grey",
        "notes": "Nearly monochrome, cool, flat. Documentary austerity.",
        "sliders": {
            "temp": -8, "tint": -2, "contrast": 6, "highlights": -6, "shadows": 8,
            "whites": -2, "blacks": -6, "vibrance": -30, "saturation": -46,
            "cg_shadow_hue": 220, "cg_shadow_sat": 8, "cg_high_hue": 220, "cg_high_sat": 6,
        },
    },
    {
        "id": "warm_skin_low_contrast",
        "label": "Warm Skin, Low Contrast",
        "notes": "Portrait-safe: warm but gentle, shadows open, colour restrained.",
        "sliders": {
            "temp": 14, "tint": 6, "contrast": -12, "highlights": -16, "shadows": 24,
            "whites": 2, "blacks": 14, "vibrance": 8, "saturation": -8,
            "cg_shadow_hue": 35, "cg_shadow_sat": 12, "cg_high_hue": 50, "cg_high_sat": 10,
        },
    },
    {
        "id": "cold_steel",
        "label": "Cold Steel",
        "notes": "Blue-grey, high micro-contrast, industrial and clean.",
        "sliders": {
            "temp": -22, "tint": -8, "contrast": 30, "highlights": -18, "shadows": -6,
            "whites": 10, "blacks": -22, "vibrance": -6, "saturation": -18,
            "cg_shadow_hue": 210, "cg_shadow_sat": 20, "cg_high_hue": 200, "cg_high_sat": 14,
        },
    },
    {
        "id": "vintage_fade",
        "label": "Vintage Fade",
        "notes": "Yellow-green cast, heavily lifted blacks, soft highlights.",
        "sliders": {
            "temp": 12, "tint": -10, "contrast": -18, "highlights": -20, "shadows": 22,
            "whites": -10, "blacks": 34, "vibrance": -10, "saturation": -14,
            "cg_shadow_hue": 75, "cg_shadow_sat": 22, "cg_high_hue": 55, "cg_high_sat": 14,
        },
    },
    {
        "id": "rose_dusk",
        "label": "Rose Dusk",
        "notes": "Pink-magenta highlights over cool violet shadows.",
        "sliders": {
            "temp": -4, "tint": 14, "contrast": 10, "highlights": -14, "shadows": 12,
            "whites": 4, "blacks": -8, "vibrance": 14, "saturation": -2,
            "cg_shadow_hue": 265, "cg_shadow_sat": 24, "cg_high_hue": 340, "cg_high_sat": 22,
        },
    },
]

LOOKS_BY_ID: dict[str, dict[str, Any]] = {look["id"]: look for look in LOOKS}
