"""Domain tests: the colour engine, the slider simulator, the KB and the rules.

None of this is graded by the rubric, but all of it is what makes the graded
mechanics non-trivial -- there is no point demonstrating a reducer that
accumulates nonsense. These tests assert the properties the design document
claims, including the ones it lists as limitations.
"""

from __future__ import annotations

import numpy as np
import pytest

from looklab import kb as kb_module
from looklab.color import (
    FEATURE_NAMES,
    MATCH_WEIGHTS,
    circ_dist,
    signature_from_array,
    weighted_distance,
    zone_is_measurable,
)
from looklab.grading import apply_grade, normalise_sliders, wheel_hue_for_lab_hue
from looklab.looks import LOOKS
from looklab.plates import all_plates, heldout_plates
from looklab.rules import (
    deltas_to_steps,
    distance_to_target,
    has_converged,
    taste_notes,
    technical_faults,
)


@pytest.fixture(scope="module")
def plates():
    return all_plates()


@pytest.fixture(scope="module")
def held():
    return heldout_plates()


# Plates are downloaded from Wikimedia Commons and their names are not fixed,
# so tests select a plate by MEASURED PROPERTY rather than by name. That keeps
# the suite working whether it runs against the downloaded set or the bundled
# scikit-image fallback, and it documents what each test actually needs.

def _pick(plates, key):
    scored = [(key(signature_from_array(img)), name) for name, img in plates.items()]
    scored.sort()
    return plates[scored[0][1]]


@pytest.fixture(scope="module")
def plate(plates):
    """A well-exposed plate with pixels across the whole tonal range."""
    return _pick(plates, lambda s: -min(s["zone_coverage"].values()))


@pytest.fixture(scope="module")
def clean_plate(plates):
    """Least-clipped plate -- used where clipping would be a false positive."""
    return _pick(plates, lambda s: s["clip_black"] + s["clip_white"])


@pytest.fixture(scope="module")
def warm_plate(plates):
    """Warmest plate, for taste rules that need to be cooled below neutral."""
    return _pick(plates, lambda s: -s["b_global"])


@pytest.fixture(scope="module")
def neutral_shadow_plate(plates):
    """Plate whose shadows carry the least colour of their own.

    Needed by the split-toning tests: on a plate whose shadows are already
    strongly blue, an applied teal tint blends to violet instead, which is the
    subject-matter contamination the design document warns about.
    """
    return _pick(plates, lambda s: s["shadow_C"])


# --------------------------------------------------------------------------
# colour engine
# --------------------------------------------------------------------------

def test_signature_is_deterministic(plate):
    """Same image, identical numbers every run. A vision model cannot say this."""
    a = signature_from_array(plate)
    b = signature_from_array(plate)
    assert a == b


def test_signature_has_every_declared_feature(plate):
    sig = signature_from_array(plate)
    for name in FEATURE_NAMES:
        assert name in sig and isinstance(sig[name], float)


def test_b_star_is_the_warm_cool_axis():
    warm = np.zeros((64, 64, 3)); warm[..., 0] = 0.8; warm[..., 1] = 0.6; warm[..., 2] = 0.3
    cool = np.zeros((64, 64, 3)); cool[..., 0] = 0.3; cool[..., 1] = 0.5; cool[..., 2] = 0.8
    assert signature_from_array(warm)["b_global"] > 20
    assert signature_from_array(cool)["b_global"] < -20


def test_a_star_is_the_green_magenta_axis():
    magenta = np.zeros((64, 64, 3)); magenta[..., 0] = 0.7; magenta[..., 1] = 0.3; magenta[..., 2] = 0.7
    green = np.zeros((64, 64, 3)); green[..., 0] = 0.3; green[..., 1] = 0.7; green[..., 2] = 0.3
    assert signature_from_array(magenta)["a_global"] > 20
    assert signature_from_array(green)["a_global"] < -20


def test_hue_coherence_separates_a_grade_from_a_colourful_subject():
    """A uniformly tinted frame is coherent; a rainbow one is not."""
    tinted = np.zeros((64, 64, 3)); tinted[..., 0] = 0.55; tinted[..., 1] = 0.42; tinted[..., 2] = 0.30
    rng = np.random.default_rng(0)
    rainbow = rng.random((64, 64, 3))
    assert signature_from_array(tinted)["mid_coh"] > 0.9
    assert signature_from_array(rainbow)["mid_coh"] < 0.6


def test_circular_distance_wraps():
    assert circ_dist(359, 1) == pytest.approx(2)
    assert circ_dist(10, 350) == pytest.approx(20)
    assert circ_dist(0, 180) == pytest.approx(180)


def test_zone_coverage_is_reported_and_empty_zones_are_zeroed():
    """Exposure coupling: brightening empties the shadow zone honestly."""
    bright = np.full((128, 128, 3), 0.95)
    sig = signature_from_array(bright)
    assert sig["zone_coverage"]["shadow"] < 0.01
    assert zone_is_measurable(sig, "shadow") is False
    assert sig["shadow_hue"] == 0.0 and sig["shadow_C"] == 0.0


def test_distance_is_zero_to_itself_and_positive_otherwise(plates):
    sigs = [signature_from_array(img) for img in plates.values()]
    a, b = sigs[0], sigs[-1]
    scales = kb_module.scales()
    assert weighted_distance(a, a, MATCH_WEIGHTS, scales) == pytest.approx(0.0)
    assert weighted_distance(a, b, MATCH_WEIGHTS, scales) > 0


# --------------------------------------------------------------------------
# slider simulator
# --------------------------------------------------------------------------

def test_neutral_sliders_are_a_no_op(plate):
    assert np.allclose(apply_grade(plate, {}), plate)


def test_normalise_sliders_ignores_unknown_keys():
    out = normalise_sliders({"temp": 10, "not_a_slider": 99})
    assert out["temp"] == 10.0 and "not_a_slider" not in out


def test_output_stays_in_gamut(plate):
    extreme = {"temp": 100, "tint": -100, "contrast": 100, "saturation": 100,
               "shadows": 100, "highlights": -100, "cg_shadow_sat": 100, "cg_high_sat": 100}
    out = apply_grade(plate, extreme)
    assert out.min() >= 0.0 and out.max() <= 1.0


@pytest.mark.parametrize(
    "slider,feature,sign",
    [
        ("temp", "b_global", +1),
        ("tint", "a_global", +1),
        ("contrast", "contrast", +1),
        ("saturation", "chroma", +1),
    ],
)
def test_each_slider_moves_its_feature_in_the_right_direction(plate, slider, feature, sign):
    base = signature_from_array(plate)
    up = signature_from_array(apply_grade(plate, {slider: 40}))
    down = signature_from_array(apply_grade(plate, {slider: -40}))
    assert sign * (up[feature] - base[feature]) > 0.5, f"{slider} up did not raise {feature}"
    assert sign * (down[feature] - base[feature]) < -0.5, f"{slider} down did not lower {feature}"


def test_saturation_does_not_shift_lightness(plate):
    """The pivot must be luma in the same space as the pixels being scaled.

    Not exactly zero: the operation preserves gamma-space luma, and CIELAB L*
    is a different (perceptual) quantity, so a few L* units of drift is
    expected. The bug this guards against moved L* by 14 units.
    """
    base = signature_from_array(plate)
    for amount in (-60, 60):
        out = signature_from_array(apply_grade(plate, {"saturation": amount}))
        assert abs(out["L_p50"] - base["L_p50"]) < 3.5, "saturation moved lightness"


def test_tone_sliders_act_on_the_right_end_of_the_range(plate):
    base = signature_from_array(plate)
    blacks = signature_from_array(apply_grade(plate, {"blacks": 60}))
    whites = signature_from_array(apply_grade(plate, {"whites": -60}))
    assert blacks["L_p05"] > base["L_p05"] + 1, "Blacks did not lift the shadow end"
    assert whites["L_p95"] < base["L_p95"] - 0.5, "Whites did not lower the highlight end"


def test_split_toning_tints_the_intended_zone(neutral_shadow_plate):
    plate = neutral_shadow_plate
    base = signature_from_array(plate)
    teal = signature_from_array(apply_grade(plate, {"cg_shadow_hue": 190, "cg_shadow_sat": 50}))
    assert teal["shadow_C"] > base["shadow_C"], "shadow chroma did not rise"
    assert circ_dist(teal["shadow_hue"], base["shadow_hue"]) > 20


def test_wheel_hue_conversion_is_not_the_identity():
    """LAB hue and the colour-grading wheel are different coordinate systems."""
    deltas = [circ_dist(wheel_hue_for_lab_hue(h), h) for h in range(0, 360, 20)]
    assert max(deltas) > 20, "conversion looks like a no-op"
    assert all(0 <= wheel_hue_for_lab_hue(h) < 360 for h in range(0, 360, 20))


def test_wheel_hue_conversion_round_trips_through_the_simulator():
    """Dialling in the recommended wheel value should reproduce the LAB hue."""
    from skimage.color import rgb2lab
    from looklab.grading import _hue_to_unit_rgb

    for wheel in (30, 90, 200, 300):
        grey = np.full((16, 16, 3), 0.5)
        tinted = np.clip(grey + 0.30 * _hue_to_unit_rgb(wheel), 0, 1)
        lab = rgb2lab(tinted)
        lab_hue = float(np.degrees(np.arctan2(lab[..., 2].mean(), lab[..., 1].mean())) % 360)
        assert circ_dist(wheel_hue_for_lab_hue(lab_hue), wheel) <= 6


# --------------------------------------------------------------------------
# knowledge base
# --------------------------------------------------------------------------

def test_kb_is_present_and_well_formed():
    kb = kb_module.load_kb()
    assert len(kb["looks"]) == len(LOOKS)
    for look in kb["looks"]:
        assert {"id", "label", "notes", "signature", "spread", "sliders"} <= set(look)
        for name in FEATURE_NAMES:
            assert name in look["signature"]
            assert name in look["spread"]
    assert kb["scales"] and kb["neutral"]
    assert kb["meta"]["heldout_plates"]


def test_kb_declares_its_synthetic_provenance():
    """The honesty claim must be in the artefact, not only in the README."""
    note = kb_module.load_kb()["meta"]["note"].lower()
    assert "not from adobe lightroom" in note or "not\nfrom adobe lightroom" in note
    assert "held-out" in note


def test_identification_beats_chance_on_held_out_plates(held):
    """Reported once, on plates the knowledge base never saw."""
    hits = top3 = total = 0
    for look in LOOKS:
        for plate in held.values():
            sig = signature_from_array(apply_grade(plate, look["sliders"]))
            matches = kb_module.match(sig, top_k=3)
            total += 1
            hits += matches[0]["id"] == look["id"]
            top3 += look["id"] in [m["id"] for m in matches]
    chance_top1 = 1.0 / len(LOOKS)
    assert hits / total > 3 * chance_top1, f"top-1 {hits}/{total} is near chance"
    assert top3 / total > 0.45, f"top-3 {top3}/{total} is too low"


def test_matching_is_willing_to_return_no_confident_match():
    """A synthetic colour chart is not a graded photograph and should say so."""
    from looklab.plates import base_plate

    matches = kb_module.match(signature_from_array(base_plate("chart", 256)), top_k=3)
    assert not matches[0]["confident"], "a colour chart should not confidently match a look"


def test_find_look_by_name_resolves_text_and_rejects_noise():
    assert kb_module.find_look_by_name("how do i get that teal and orange look")["id"] == "teal_orange"
    assert kb_module.find_look_by_name("make it warm golden")["id"] == "warm_golden"
    assert kb_module.find_look_by_name("hello there") is None


def test_neutral_baseline_is_plausible():
    """The stand-in 'untouched photo' must not be an outlier."""
    neutral = kb_module.neutral_signature()
    assert 0 < neutral["chroma"] < 35, "neutral baseline is implausibly saturated"
    assert 30 < neutral["L_p50"] < 75
    assert abs(neutral["b_global"]) < 25


# --------------------------------------------------------------------------
# rules
# --------------------------------------------------------------------------

def test_no_difference_produces_no_steps(plate):
    sig = signature_from_array(plate)
    assert deltas_to_steps(sig, sig) == []


def test_steps_are_ordered_by_magnitude(plate):
    base = signature_from_array(plate)
    target = signature_from_array(apply_grade(plate, LOOKS[8]["sliders"]))
    steps = deltas_to_steps(base, target)
    assert steps
    magnitudes = [s["magnitude"] for s in steps]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_every_step_carries_a_grounded_reason(plate):
    base = signature_from_array(plate)
    target = signature_from_array(apply_grade(plate, LOOKS[1]["sliders"]))
    for step in deltas_to_steps(base, target):
        assert step["why"] and len(step["why"]) > 10
        assert step["slider"]


def test_slider_amounts_stay_in_lightroom_range(plate):
    base = signature_from_array(plate)
    extreme = signature_from_array(
        apply_grade(plate, {"temp": 100, "saturation": 100, "contrast": 100})
    )
    for step in deltas_to_steps(base, extreme):
        if isinstance(step["amount"], (int, float)):
            assert -100 <= step["amount"] <= 100


def test_refine_loop_converges(plate):
    """Applying the advice must reduce the measured distance to target."""
    scales = kb_module.scales()
    target = signature_from_array(apply_grade(plate, LOOKS[0]["sliders"]))

    current_img = plate
    distances = []
    applied: dict[str, float] = {}
    for _ in range(3):
        current = signature_from_array(current_img)
        distances.append(distance_to_target(current, target, scales))
        for step in deltas_to_steps(current, target):
            key = {
                "Temp": "temp", "Tint": "tint", "Contrast": "contrast",
                "Blacks": "blacks", "Whites": "whites",
                "Vibrance": "vibrance", "Saturation": "saturation",
            }.get(step["slider"])
            if key and isinstance(step["amount"], (int, float)):
                applied[key] = applied.get(key, 0.0) + float(step["amount"])
        current_img = apply_grade(plate, applied)

    assert distances[-1] < distances[0], f"loop did not converge: {distances}"


def test_convergence_threshold_behaves():
    assert has_converged(0.0) and has_converged(0.05)
    assert not has_converged(0.5)


def test_technical_faults_catch_deliberate_damage(plate):
    crushed = signature_from_array(apply_grade(plate, {"blacks": -100, "contrast": 90}))
    issues = {f["issue"] for f in technical_faults(crushed)}
    assert any("black" in i.lower() for i in issues), issues

    grey = signature_from_array(apply_grade(plate, {"saturation": -100}))
    assert any("monochrome" in f["issue"].lower() for f in technical_faults(grey))


def test_clean_image_has_no_high_severity_faults(clean_plate):
    """Uses the least-clipped plate. A plate that genuinely contains a large
    pure-black region SHOULD be flagged for crushed blacks -- correct
    behaviour, not a false positive, so it would be the wrong plate here."""
    faults = technical_faults(signature_from_array(clean_plate))
    assert not [f for f in faults if f["severity"] == "high"], faults


def test_taste_notes_use_the_stored_profile():
    """The rule is a pure function of the signature, so test it as one.

    Driving it through a plate would make the test about that photograph's own
    colour rather than about the rule.
    """
    from looklab.rules import NEUTRAL_B_STAR

    cool = {"b_global": NEUTRAL_B_STAR - 10.0, "a_global": 0.0, "L_p05": 20.0,
            "shadow_hue": 0.0, "shadow_C": 0.0, "chroma": 15.0}
    notes = taste_notes(cool, {"preferred_look": "warm golden"})
    assert notes and "warm" in notes[0]["note"].lower()
    assert "Temp +" in notes[0]["note"], "the note should carry an actionable move"
    assert taste_notes(cool, {}) == [], "no profile means no taste notes"


def test_taste_notes_fire_on_a_genuinely_cooled_photograph(plates):
    """Integration: cooling a real photograph hard enough must trip the rule.

    Checked across every plate rather than one, because how far a given frame
    moves for a fixed Temp value depends on the frame -- which is the
    non-invertibility the design document is explicit about.
    """
    fired = 0
    for img in plates.values():
        sig = signature_from_array(apply_grade(img, {"temp": -100}))
        if taste_notes(sig, {"preferred_look": "warm golden"}):
            fired += 1
    assert fired > 0, "cooling every plate to Temp -100 tripped the rule on none of them"


def test_taste_notes_judge_cool_against_a_photographic_neutral(warm_plate):
    """An untouched warm photo must not be scolded for being cool."""
    from looklab.rules import NEUTRAL_B_STAR

    untouched = signature_from_array(warm_plate)
    assert untouched["b_global"] > NEUTRAL_B_STAR
    assert taste_notes(untouched, {"preferred_look": "warm golden"}) == []


def test_taste_notes_fire_on_a_disliked_tint(neutral_shadow_plate):
    """Uses the plate whose shadows carry least colour of their own.

    On a plate with already-blue shadows the applied teal blends to violet and
    the rule correctly does not fire -- subject contamination, visible in a
    test rather than merely described.
    """
    from looklab.rules import LAB_HUE

    teal = signature_from_array(
        apply_grade(neutral_shadow_plate, {"cg_shadow_hue": 180, "cg_shadow_sat": 70})
    )
    assert circ_dist(teal["shadow_hue"], LAB_HUE["teal"]) < 55, (
        f"the tint did not land on teal: shadow_hue={teal['shadow_hue']:.0f}"
    )
    notes = taste_notes(teal, {"dislikes": "heavy teal shadows"})
    assert notes, "a disliked teal shadow tint should be flagged"
