"""Build ``looklab/kb.json`` from the base plates and the look library.

This is the programmatic replacement for Part 6's Lightroom evening. For each
look in ``looklab.looks.LOOKS`` it applies the authored slider values to each
of the three base plates -- real photographs bundled with scikit-image -- via
``looklab.grading.apply_grade``, measures the resulting signature, and stores
the per-look centroid plus the per-feature spread across plates.

The spread matters twice over. It is the diagnostic the design document asks
for -- a feature with high spread across three very different plates is being
driven by the subject rather than by the grade -- and it is also consumed at
match time by ``weighted_distance``, which down-weights features that are
unreliable for the specific look being compared against.

Run:  python -m tools.build_kb
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from looklab.color import FEATURE_NAMES, feature_stats, signature_from_array  # noqa: E402
from looklab.grading import apply_grade  # noqa: E402
from looklab.kb import KB_PATH  # noqa: E402
from looklab.looks import LOOKS  # noqa: E402
from looklab.plates import PLATE_NAMES, all_plates, heldout_plates, provenance  # noqa: E402


def build() -> dict:
    plates = all_plates()
    entries = []

    for look in LOOKS:
        per_plate = []
        for name in PLATE_NAMES:
            graded = apply_grade(plates[name], look["sliders"])
            per_plate.append(signature_from_array(graded))

        centroid: dict[str, float] = {}
        spread: dict[str, float] = {}
        for feat in FEATURE_NAMES:
            vals = np.array([float(s.get(feat, 0.0)) for s in per_plate])
            if feat.endswith("_hue"):
                # Circular mean: averaging 359 and 1 must give 0, not 180.
                # Zeros mean "zone not measurable", so they are excluded rather
                # than dragged into the average as a real hue of 0 degrees.
                live = vals[vals != 0.0]
                if live.size == 0:
                    centroid[feat], spread[feat] = 0.0, 0.0
                    continue
                rad = np.radians(live)
                cx, cy = float(np.cos(rad).mean()), float(np.sin(rad).mean())
                centroid[feat] = float(np.degrees(np.arctan2(cy, cx)) % 360.0)
                # Circular spread: 0 = identical hues, 1 = uniformly scattered.
                spread[feat] = round(float(1.0 - np.hypot(cx, cy)) * 180.0, 4)
            else:
                centroid[feat] = round(float(vals.mean()), 4)
                spread[feat] = round(float(vals.std()), 4)
            centroid[feat] = round(centroid[feat], 4)

        entries.append(
            {
                "id": look["id"],
                "label": look["label"],
                "notes": look["notes"],
                "signature": centroid,
                "spread": spread,
                "sliders": look["sliders"],
            }
        )

    # The neutral baseline: what "a photo before you did anything to it"
    # measures like. It lets the achieve flow answer "how do I get the deep
    # amber look?" with no upload at all, which is what keeps the text-only
    # spine -- and therefore the whole T2 recipe demo -- independent of image
    # uploads.
    #
    # Taken over ALL five ungraded plates including the held-out ones, and by
    # MEDIAN rather than mean. The mean of the three KB plates alone gives
    # a* +17.2 / b* +21.4 / chroma 28.8, because `coffee` is an unusually
    # saturated frame and drags it; the median over five gives a* +11.4 /
    # b* +12.0 / chroma 22.9, which is a far more plausible "untouched photo".
    all_ungraded = {**plates, **heldout_plates()}
    neutral_sigs = [signature_from_array(img) for img in all_ungraded.values()]
    neutral: dict[str, float] = {}
    for feat in FEATURE_NAMES:
        vals = np.array([float(s.get(feat, 0.0)) for s in neutral_sigs])
        if feat.endswith("_hue"):
            live = vals[vals != 0.0]
            if live.size == 0:
                neutral[feat] = 0.0
                continue
            rad = np.radians(live)
            neutral[feat] = round(
                float(np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360.0), 4
            )
        else:
            neutral[feat] = round(float(np.median(vals)), 4)

    kb = {
        "meta": {
            **provenance(),
            "plates": list(PLATE_NAMES),
            "n_looks": len(entries),
            "n_frames": len(entries) * len(PLATE_NAMES),
            "heldout_plates": list(heldout_plates().keys()),
            "note": (
                "Base plates are real photographs bundled with scikit-image. The grades "
                "applied to them come from looklab.grading, a documented numpy model of "
                "Lightroom-style controls -- NOT from Adobe Lightroom. Slider values are "
                "authored ground truth, which is what makes tools/calibrate.py a real "
                "evaluation rather than a self-check. Calibration is reported on the "
                "held-out plates only. See the README section 'What this validates and "
                "what it does not'."
            ),
        },
        "scales": {
            k: round(v, 6) for k, v in feature_stats([e["signature"] for e in entries]).items()
        },
        "neutral": neutral,
        "looks": entries,
    }
    return kb


def main() -> int:
    kb = build()
    with open(KB_PATH, "w", encoding="utf-8") as fh:
        json.dump(kb, fh, indent=2)
    print(f"wrote {KB_PATH}")
    print(f"  {kb['meta']['n_looks']} looks from {kb['meta']['n_frames']} graded frames")

    # Report the features whose spread across plates is largest relative to
    # their spread across looks -- those are subject-driven, not grade-driven.
    scales = kb["scales"]
    rows = []
    for feat in FEATURE_NAMES:
        across_looks = float(scales.get(feat, 0.0))
        across_plates = float(np.mean([e["spread"].get(feat, 0.0) for e in kb["looks"]]))
        if across_looks > 1e-6:
            rows.append((across_plates / across_looks, feat, across_plates, across_looks))
    rows.sort(reverse=True)
    print("\n  subject-driven features (plate spread / look spread, high = unreliable):")
    for ratio, feat, ap, al in rows[:6]:
        print(f"    {feat:14s} {ratio:6.2f}   plate sd {ap:7.2f} vs look sd {al:7.2f}")
    print("\n  grade-driven features (low ratio = the grade dominates the subject):")
    for ratio, feat, ap, al in rows[-6:]:
        print(f"    {feat:14s} {ratio:6.2f}   plate sd {ap:7.2f} vs look sd {al:7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
