"""Part 9.2: validate the delta engine against ground-truth slider values.

Methodology, stated up front because it is what makes the numbers mean
anything:

* The rules in ``looklab/rules.py`` were **frozen before this script was ever
  run**. Their coefficients are the values authored in the design document.
  They have not been fitted to this script's output, because tuning them here
  would turn a partial circularity into a total one.
* The knowledge base is built on three plates. Calibration is reported on the
  **held-out** plates only -- a synthetic colour chart and a photograph that
  never entered ``kb.json``.
* The headline metric is **sign agreement**: when the engine says "Temp up",
  was the true move up? Magnitude error is reported too, but as a diagnostic,
  not as an accuracy claim -- see the honesty note below.

What this validates and what it does not
----------------------------------------
The graded frames are produced by ``looklab/grading.py``, a documented numpy
model of Lightroom-style controls. They are NOT exports from Adobe Lightroom.
So this measures whether the delta engine recovers the direction of the
grading operations it was validated against. It does not, and cannot, measure
agreement with Adobe's transfer functions.

The circularity is real and bounded, and worth naming precisely: the
simulator's Temp operation applies channel gains in linear-light RGB, while
``deltas_to_steps`` reads ``b_global``, a mean of CIELAB b*. Those are
different spaces and the mapping between them is image-dependent -- the same
Temp value produces a different b* shift on a dark frame than on a bright one.
That is why a perfect score is not expected and would in fact be suspicious.
Directional agreement on the temp/tint axes is nonetheless weaker evidence
than on the axes where the coupling is more indirect.

Run:  python -m tools.calibrate
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from looklab import kb as kb_module  # noqa: E402
from looklab.color import signature_from_array  # noqa: E402
from looklab.grading import apply_grade  # noqa: E402
from looklab.looks import LOOKS  # noqa: E402
from looklab.plates import all_plates, heldout_plates  # noqa: E402
from looklab.rules import deltas_to_steps  # noqa: E402

# Map the human-facing slider names the engine emits onto the ground-truth keys.
SLIDER_TO_GROUND_TRUTH: dict[str, str] = {
    "Temp": "temp",
    "Tint": "tint",
    "Contrast": "contrast",
    "Blacks": "blacks",
    "Whites": "whites",
    "Vibrance": "vibrance",
    "Saturation": "saturation",
}

# Colour-grading steps emit a string amount, so they are scored on whether the
# recommended hue lands near the true hue rather than on a signed magnitude.
CG_STEPS = {
    "Color Grading > Shadows": ("cg_shadow_hue", "cg_shadow_sat"),
    "Color Grading > Highlights": ("cg_high_hue", "cg_high_sat"),
}

# A ground-truth move smaller than this is treated as "no move intended", so a
# prediction of zero is not punished for disagreeing with noise.
DEADBAND = 3.0


def _circ_err(a: float, b: float) -> float:
    d = abs(float(a) - float(b)) % 360.0
    return min(d, 360.0 - d)


def evaluate(plates: dict[str, np.ndarray], label: str) -> dict:
    rows: list[dict] = []
    hue_rows: list[dict] = []

    for look in LOOKS:
        truth = look["sliders"]
        for plate_name, plate in plates.items():
            neutral = signature_from_array(plate)
            graded = signature_from_array(apply_grade(plate, truth))
            predicted = {step["slider"]: step["amount"] for step in deltas_to_steps(neutral, graded)}

            for slider, key in SLIDER_TO_GROUND_TRUTH.items():
                actual = float(truth.get(key, 0.0))
                # Vibrance and Saturation are one decision in the engine (it
                # picks one based on magnitude), so score them against whichever
                # ground-truth control actually moved.
                if key in ("vibrance", "saturation"):
                    actual = float(truth.get("vibrance", 0.0)) + float(truth.get("saturation", 0.0))
                pred_raw = predicted.get(slider)
                if key in ("vibrance", "saturation"):
                    pred_raw = predicted.get("Vibrance", predicted.get("Saturation"))
                pred = float(pred_raw) if isinstance(pred_raw, (int, float)) else 0.0

                if abs(actual) < DEADBAND and pred == 0.0:
                    continue  # nothing to predict and nothing predicted
                rows.append(
                    {
                        "look": look["id"],
                        "plate": plate_name,
                        "slider": slider,
                        "actual": actual,
                        "pred": pred,
                        "sign_ok": (np.sign(pred) == np.sign(actual)) or abs(actual) < DEADBAND,
                        "abs_err": abs(pred - actual),
                        "predicted_something": pred != 0.0,
                    }
                )

            for step_name, (hue_key, sat_key) in CG_STEPS.items():
                if float(truth.get(sat_key, 0.0)) < 8.0:
                    continue  # tint too weak to be a stated intent
                amount = predicted.get(step_name)
                if not isinstance(amount, str):
                    hue_rows.append(
                        {"step": step_name, "hit": False, "err": None, "look": look["id"]}
                    )
                    continue
                try:
                    got = float(amount.split("hue")[1].split(",")[0])
                except (IndexError, ValueError):
                    continue
                err = _circ_err(got, float(truth[hue_key]))
                hue_rows.append(
                    {"step": step_name, "hit": err <= 45.0, "err": err, "look": look["id"]}
                )

    scored = [r for r in rows if abs(r["actual"]) >= DEADBAND]
    signed = [r for r in scored if r["predicted_something"]]
    return {
        "label": label,
        "rows": rows,
        "scored": scored,
        "signed": signed,
        "hue_rows": hue_rows,
        "n_frames": len(LOOKS) * len(plates),
    }


def monotonicity(plates: dict[str, np.ndarray]) -> list[dict]:
    """Does a larger true move produce a larger recommendation, in order?

    Sweeps one slider at a time and checks that the predicted amount is
    non-decreasing. This is the property that actually matters for the refine
    loop: the engine does not need the right number, it needs to not reverse
    direction as the gap grows.
    """
    results = []
    sweeps = {
        "temp": ("Temp", [-60, -40, -20, 0, 20, 40, 60]),
        "tint": ("Tint", [-40, -20, 0, 20, 40]),
        "contrast": ("Contrast", [-40, -20, 0, 20, 40]),
        "saturation": ("Saturation", [-40, -20, 0, 20, 40]),
    }
    for key, (slider, values) in sweeps.items():
        for plate_name, plate in plates.items():
            neutral = signature_from_array(plate)
            preds = []
            for v in values:
                graded = signature_from_array(apply_grade(plate, {key: v}))
                steps = {s["slider"]: s["amount"] for s in deltas_to_steps(neutral, graded)}
                amount = steps.get(slider)
                # Saturation and Vibrance are the same decision in the engine.
                if amount is None and slider == "Saturation":
                    amount = steps.get("Vibrance")
                preds.append(float(amount) if isinstance(amount, (int, float)) else 0.0)
            diffs = np.diff(preds)
            results.append(
                {
                    "slider": slider,
                    "plate": plate_name,
                    "monotone": bool(np.all(diffs >= -1e-9)),
                    "spearman_ok": bool(np.corrcoef(values, preds)[0, 1] > 0.9),
                    "values": values,
                    "preds": [round(p, 1) for p in preds],
                }
            )
    return results


def _report(result: dict) -> None:
    scored, signed, hue_rows = result["scored"], result["signed"], result["hue_rows"]
    print(f"\n{'=' * 74}\n  {result['label']}  ({result['n_frames']} graded frames)\n{'=' * 74}")
    if not scored:
        print("  no scoreable rows")
        return

    overall_sign = 100.0 * np.mean([r["sign_ok"] for r in signed]) if signed else 0.0
    recall = 100.0 * np.mean([r["predicted_something"] for r in scored])
    print(f"\n  Sign agreement (simulator round-trip) : {overall_sign:5.1f}%  (n={len(signed)})")
    print(f"  Recall: real moves the engine acted on: {recall:5.1f}%  (n={len(scored)})")

    print(f"\n  {'slider':<12} {'n':>4} {'sign agr':>9} {'recall':>8} {'mean |err|':>11}")
    print(f"  {'-' * 12} {'-' * 4} {'-' * 9} {'-' * 8} {'-' * 11}")
    for slider in SLIDER_TO_GROUND_TRUTH:
        s_rows = [r for r in scored if r["slider"] == slider]
        s_signed = [r for r in s_rows if r["predicted_something"]]
        if not s_rows:
            continue
        sign = 100.0 * np.mean([r["sign_ok"] for r in s_signed]) if s_signed else float("nan")
        rec = 100.0 * np.mean([r["predicted_something"] for r in s_rows])
        err = np.mean([r["abs_err"] for r in s_signed]) if s_signed else float("nan")
        print(f"  {slider:<12} {len(s_rows):>4} {sign:>8.1f}% {rec:>7.1f}% {err:>11.1f}")

    if hue_rows:
        hits = 100.0 * np.mean([h["hit"] for h in hue_rows])
        errs = [h["err"] for h in hue_rows if h["err"] is not None]
        print(
            f"\n  Colour-grading hue within 45 deg      : {hits:5.1f}%  (n={len(hue_rows)}"
            + (f", mean error {np.mean(errs):.0f} deg)" if errs else ")")
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-inkb",
        action="store_true",
        help="also report on the three plates used to build kb.json (diagnostic only)",
    )
    args = parser.parse_args()

    kb_module.reset_cache()
    meta = kb_module.load_kb().get("meta", {})
    print("LookLab calibration")
    print(f"  knowledge base : {meta.get('n_looks', '?')} looks, {meta.get('n_frames', '?')} frames")
    print(f"  kb plates      : {', '.join(meta.get('plates', []))}")
    print(f"  held-out plates: {', '.join(meta.get('heldout_plates', []))}")
    print(
        "\n  NOTE: graded frames come from looklab/grading.py, a documented numpy model\n"
        "  of Lightroom-style controls -- NOT from Adobe Lightroom. These numbers measure\n"
        "  whether the delta engine recovers the DIRECTION of the operations it was\n"
        "  validated against. They are not a claim about agreement with Adobe."
    )

    held = evaluate(heldout_plates(), "HELD-OUT PLATES (the reported result)")
    _report(held)

    if args.include_inkb:
        _report(evaluate(all_plates(), "IN-KB PLATES (diagnostic only -- not the headline)"))

    print(f"\n{'=' * 74}\n  Monotonicity (held-out plates)\n{'=' * 74}")
    mono = monotonicity(heldout_plates())
    ok = sum(1 for m in mono if m["monotone"])
    print(f"\n  Monotone sweeps: {ok}/{len(mono)}")
    print(f"\n  {'slider':<12} {'plate':<10} {'monotone':>9} {'rank-corr>0.9':>14}   predictions")
    for m in mono:
        print(
            f"  {m['slider']:<12} {m['plate']:<10} {str(m['monotone']):>9} "
            f"{str(m['spearman_ok']):>14}   {m['preds']}"
        )

    print(
        "\n  Reading this: monotone means a larger true move always produced a larger\n"
        "  recommendation. That is the property the refine loop needs -- the engine does\n"
        "  not have to be numerically right, it has to not reverse as the gap grows.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
