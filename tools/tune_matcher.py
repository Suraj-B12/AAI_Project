"""Select the matcher's feature weights and SPREAD_ALPHA by cross-validation.

Methodology, which is the whole point of this file:

* Candidate weight sets and alpha values are scored by **leave-one-plate-out
  cross-validation over the knowledge-base plates only**. The held-out plates
  are never consulted during selection.
* The winner is then scored **once** on the held-out plates, and that is the
  number reported in the README.

This matters because the obvious alternative -- picking features by their
measured plate-spread ratio -- overfits badly. On the earlier three-plate
knowledge base it scored 60% in-KB and collapsed to 6.7% (chance) on held-out
plates, because the ratio is itself computed from the KB plates. That result
is reproduced below as the ``data_driven`` candidate so the failure is visible
rather than merely asserted.

Run:  python -m tools.tune_matcher
      python -m tools.tune_matcher --apply    (print the constants to paste)
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from looklab.color import (  # noqa: E402
    CIRCULAR_FEATURES,
    FEATURE_NAMES,
    circ_dist,
    feature_stats,
    signature_from_array,
)
from looklab.grading import apply_grade  # noqa: E402
from looklab.looks import LOOKS  # noqa: E402
from looklab.plates import PLATE_NAMES, all_plates, heldout_plates  # noqa: E402

ALPHAS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)

CANDIDATES: dict[str, dict[str, float]] = {
    # As written in the design document, Part 5.3.
    "design_doc_8": {
        "shadow_hue": 2.0, "shadow_C": 1.5, "high_hue": 2.0, "high_C": 1.5,
        "b_global": 1.5, "a_global": 1.0, "contrast": 1.0, "chroma": 1.0,
    },
    # Design doc plus the zone a*/b* channels and hue coherence.
    "plus_zone_ab": {
        "shadow_hue": 2.0, "shadow_C": 1.5, "high_hue": 2.0, "high_C": 1.5,
        "b_global": 1.5, "a_global": 1.0, "contrast": 1.0, "chroma": 1.0,
        "shadow_coh": 1.0, "shadow_a": 1.0, "shadow_b": 1.0, "high_b": 1.0,
        "split": 0.8, "L_p05": 0.6,
    },
    # Drops every tone feature: tone is where exposure and subject live.
    "chroma_only_13": {
        "shadow_hue": 2.5, "high_hue": 2.5, "shadow_a": 1.5, "shadow_b": 1.5,
        "high_a": 1.5, "high_b": 1.5, "shadow_C": 1.5, "high_C": 1.2,
        "a_global": 1.0, "b_global": 1.2, "shadow_coh": 1.0, "split": 1.0,
        "chroma": 0.8,
    },
    # Hue angles carry most of the grade; everything else is support.
    "hue_dominant": {
        "shadow_hue": 3.0, "high_hue": 3.0, "mid_hue": 1.0,
        "shadow_a": 1.5, "shadow_b": 1.5, "high_a": 1.5, "high_b": 1.5,
        "shadow_C": 1.5, "high_C": 1.0, "shadow_coh": 1.0, "split": 1.0,
        "a_global": 1.0, "b_global": 1.0, "chroma": 0.8, "contrast": 0.5,
    },
    # Zone chromaticity only -- no global aggregates at all.
    "zones_only": {
        "shadow_hue": 2.5, "high_hue": 2.5, "shadow_a": 2.0, "shadow_b": 2.0,
        "high_a": 2.0, "high_b": 2.0, "shadow_C": 1.5, "high_C": 1.5,
        "shadow_coh": 1.0, "high_coh": 0.8, "split": 1.0,
    },
}


def signatures() -> dict[tuple[str, str], dict]:
    """Every (look, plate) signature, computed once."""
    plates = {**all_plates(), **heldout_plates()}
    return {
        (look["id"], name): signature_from_array(apply_grade(img, look["sliders"]))
        for look in LOOKS
        for name, img in plates.items()
    }


def build_centroids(sigs: dict, plate_names: list[str]) -> tuple[list[dict], dict]:
    """Centroid + spread per look over a plate subset (mirrors tools.build_kb)."""
    looks = []
    for look in LOOKS:
        per = [sigs[(look["id"], p)] for p in plate_names]
        centroid, spread = {}, {}
        for feat in FEATURE_NAMES:
            vals = np.array([float(s.get(feat, 0.0)) for s in per])
            if feat.endswith("_hue"):
                live = vals[vals != 0.0]
                if live.size == 0:
                    centroid[feat] = spread[feat] = 0.0
                    continue
                rad = np.radians(live)
                cx, cy = float(np.cos(rad).mean()), float(np.sin(rad).mean())
                centroid[feat] = float(np.degrees(np.arctan2(cy, cx)) % 360.0)
                spread[feat] = float(1.0 - np.hypot(cx, cy)) * 180.0
            else:
                centroid[feat] = float(vals.mean())
                spread[feat] = float(vals.std())
        looks.append({"id": look["id"], "signature": centroid, "spread": spread})
    return looks, feature_stats([lk["signature"] for lk in looks])


def distance(sig: dict, look: dict, scales: dict, weights: dict, alpha: float) -> float:
    acc = total = 0.0
    for feat, w in weights.items():
        if w <= 0:
            continue
        unit = 180.0 if feat in CIRCULAR_FEATURES else 1.0
        a, b = float(sig.get(feat, 0.0)), float(look["signature"].get(feat, 0.0))
        d = (circ_dist(a, b) if feat in CIRCULAR_FEATURES else abs(a - b)) / unit
        scale = math.hypot(
            float(look["spread"].get(feat, 0.0)) / unit,
            alpha * (float(scales.get(feat, 1.0)) or 1.0) / unit,
        )
        acc += w * (d / max(scale, 1e-6)) ** 2
        total += w
    return math.sqrt(acc / total) if total else 0.0


def accuracy(sigs, looks, scales, weights, alpha, test_plates) -> tuple[float, float]:
    hits = top3 = n = 0
    for look in LOOKS:
        for plate in test_plates:
            sig = sigs[(look["id"], plate)]
            ranked = sorted((distance(sig, lk, scales, weights, alpha), lk["id"]) for lk in looks)
            n += 1
            hits += ranked[0][1] == look["id"]
            top3 += look["id"] in [i for _, i in ranked[:3]]
    return (100.0 * hits / n, 100.0 * top3 / n) if n else (0.0, 0.0)


def lopo_cv(sigs, weights, alpha, leave_out: int = 2) -> tuple[float, float, float]:
    """Leave-P-plates-out CV over the KB plates. Never sees held-out plates.

    Leave-2-out gives C(8,2)=28 folds rather than LOPO's 8, which matters:
    with only 8 folds the standard error on a top-1 estimate is around 4-5
    percentage points, and candidates were separating by less than that. The
    third return value is the standard error of the top-1 mean, so the caller
    can apply a one-standard-error rule instead of chasing noise.
    """
    from itertools import combinations

    results = []
    for held in combinations(PLATE_NAMES, leave_out):
        train = [p for p in PLATE_NAMES if p not in held]
        if len(train) < 2:
            continue
        looks, scales = build_centroids(sigs, train)
        results.append(accuracy(sigs, looks, scales, weights, alpha, list(held)))
    top1 = np.array([r[0] for r in results])
    top3 = np.array([r[1] for r in results])
    stderr = float(top1.std(ddof=1) / math.sqrt(len(top1))) if len(top1) > 1 else 0.0
    return float(top1.mean()), float(top3.mean()), stderr


def data_driven_weights(sigs) -> dict[str, float]:
    """Weights picked from the measured plate-spread ratio. Included to fail."""
    looks, scales = build_centroids(sigs, list(PLATE_NAMES))
    weights = {}
    for feat in FEATURE_NAMES:
        across_looks = float(scales.get(feat, 0.0))
        if across_looks <= 1e-6:
            continue
        across_plates = float(np.mean([lk["spread"].get(feat, 0.0) for lk in looks]))
        ratio = across_plates / across_looks
        if ratio < 1.0:
            weights[feat] = min(3.0, 1.0 / max(ratio, 0.05))
    return weights


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="print constants to paste into color.py")
    args = parser.parse_args()

    print("Computing signatures for every (look, plate) pair...")
    sigs = signatures()
    held_names = list(heldout_plates().keys())
    print(f"  {len(LOOKS)} looks x {len(PLATE_NAMES)} KB plates = {len(LOOKS) * len(PLATE_NAMES)} frames")
    print(f"  {len(LOOKS)} looks x {len(held_names)} held-out plates = {len(LOOKS) * len(held_names)} frames")

    candidates = dict(CANDIDATES)
    candidates["data_driven"] = data_driven_weights(sigs)

    print(f"\n{'candidate':<16} {'alpha':>5}  {'CV top-1':>14} {'CV top-3':>9}  {'n feats':>7}")
    print("-" * 62)
    rows = []
    for name, weights in candidates.items():
        for alpha in ALPHAS:
            cv1, cv3, se = lopo_cv(sigs, weights, alpha)
            rows.append((cv1, cv3, se, name, weights, alpha))
            print(
                f"{name:<16} {alpha:>5.2f}  {cv1:>7.1f}% +/-{se:<4.1f} {cv3:>8.1f}%  {len(weights):>7}"
            )
        print()

    # One-standard-error rule: among every configuration whose CV top-1 is
    # within one standard error of the best, prefer the most robust rather
    # than the nominal winner. Picking a 1.6-point lead on 28 folds with a
    # 3-4 point standard error is selecting noise.
    rows.sort(key=lambda r: -r[0])
    top_cv1, _, top_se, top_name, _, top_alpha = rows[0]
    threshold = top_cv1 - top_se
    within = [r for r in rows if r[0] >= threshold]

    # Robustness preference: more features is more robust here, because the
    # narrow candidates lean on features (clipping fractions) that are
    # structurally near-zero on most photographs and therefore contribute
    # nothing when they matter most.
    within.sort(key=lambda r: (-len(r[4]), -r[1]))
    cv1, cv3, se, name, weights, alpha = within[0]

    print("=" * 62)
    print(f"Best nominal CV: {top_name} alpha={top_alpha} at {top_cv1:.1f}% +/-{top_se:.1f}")
    print(f"Within one standard error ({threshold:.1f}%): {len(within)} configurations")
    print(f"\nSELECTED (1-SE rule, most robust within noise): {name}, alpha={alpha}")
    print(f"  CV top-1 {cv1:.1f}% +/-{se:.1f}   CV top-3 {cv3:.1f}%   {len(weights)} features")

    looks, scales = build_centroids(sigs, list(PLATE_NAMES))
    h1, h3 = accuracy(sigs, looks, scales, weights, alpha, held_names)
    i1, i3 = accuracy(sigs, looks, scales, weights, alpha, list(PLATE_NAMES))
    chance1, chance3 = 100.0 / len(LOOKS), 300.0 / len(LOOKS)

    print(f"\nREPORTED ONCE on {len(held_names)} held-out plates:")
    print(f"  top-1 {h1:.1f}%  (chance {chance1:.1f}%,  {h1 / chance1:.1f}x)")
    print(f"  top-3 {h3:.1f}%  (chance {chance3:.1f}%,  {h3 / chance3:.1f}x)")
    print(f"\nFor reference only, in-KB plates: top-1 {i1:.1f}%  top-3 {i3:.1f}%")

    dd1, dd3 = accuracy(sigs, looks, scales, candidates["data_driven"], alpha, held_names)
    print(
        f"\nThe overfitting control -- 'data_driven' weights scored on held-out: "
        f"top-1 {dd1:.1f}% top-3 {dd3:.1f}%"
    )

    if args.apply:
        print("\n# Paste into looklab/color.py")
        print("MATCH_WEIGHTS: dict[str, float] = {")
        for feat, w in sorted(weights.items(), key=lambda kv: -kv[1]):
            print(f'    "{feat}": {w},')
        print("}")
        print(f"SPREAD_ALPHA = {alpha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
