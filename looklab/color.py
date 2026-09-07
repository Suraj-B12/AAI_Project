"""CIELAB colour signature extraction, distance and matching.

Why CIELAB and not HSV (Part 5.1 of the design doc):

1. HSV hue is undefined for near-grey pixels but still votes in a mean. A
   desaturated shadow has an essentially random hue; averaging thousands of
   them injects noise with the same weight as genuinely tinted pixels.
2. HSV ``V = max(R,G,B)`` is not perceptual brightness, so zone-splitting on
   V puts the wrong pixels in the wrong zones.
3. HSV is computed on gamma-encoded sRGB, so the means are wrong in a way
   that varies with exposure.

CIELAB fixes all three and has a fourth property that matters here: ``a*``
and ``b*`` are the green-magenta and blue-yellow axes, which are exactly the
Tint and Temp axes a photographer already thinks in. ``L*`` is perceptual
lightness, so zone splits are meaningful.

Everything in this module is deterministic: the same image always produces
the same numbers. No model is involved anywhere.
"""

from __future__ import annotations

import io
import math
from typing import Any, Iterable

import numpy as np
from PIL import Image

from .cielab import rgb2lab

# L* zone boundaries. L* runs 0..100, so these are perceptual thirds.
ZONES: dict[str, tuple[float, float]] = {
    "shadow": (0.0, 25.0),
    "mid": (25.0, 75.0),
    "high": (75.0, 100.0),
}

# Minimum pixels in a zone before its statistics are trusted.
MIN_ZONE_PIXELS = 100

# Features every signature carries, in a stable order.
ZONE_FEATURES = ("a", "b", "C", "hue", "coh")
GLOBAL_FEATURES = (
    "L_p05",
    "L_p50",
    "L_p95",
    "contrast",
    "chroma",
    "a_global",
    "b_global",
    "clip_black",
    "clip_white",
    "split",
)

FEATURE_NAMES: tuple[str, ...] = tuple(
    [f"{z}_{f}" for z in ZONES for f in ZONE_FEATURES] + list(GLOBAL_FEATURES)
)

# Features that are angles in degrees and therefore need circular distance.
CIRCULAR_FEATURES = frozenset(
    [f"{z}_hue" for z in ZONES]
)

# Matching uses a reduced, weighted subset (Part 5.3). Twenty-five features
# against fifteen reference looks is a bad nearest-neighbour problem -- in
# high dimensions with tiny N, distances concentrate and everything looks
# equally far away. These are grade-driven rather than subject-driven.
# Mid-tone hue, chroma and coherence are deliberately excluded: that is where
# the subject lives. ``tools/build_kb.py`` reports the plate-spread /
# look-spread ratio per feature, which confirms the split empirically --
# shadow_hue and high_hue score lowest (most grade-driven), the mid_* features
# highest (most subject-driven).
#
# Selected by ``tools/tune_matcher.py``: leave-2-plates-out cross-validation
# over the eight knowledge-base plates (28 folds), with a one-standard-error
# rule. Held-out plates were never consulted during selection and are scored
# exactly once.
#
# Every entry is a CHROMATICITY feature of the shadow or highlight zone. Two
# whole families are deliberately absent:
#
#   * tone (contrast, L_p05/50/95) -- measured as subject-driven, and exposure
#     moves pixels between zones, so tone conflates the grade with the scene.
#   * every mid_* feature -- the midtones are where the subject lives.
#
# Global a*/b*/chroma are absent for the same reason: on a set of eight very
# different photographs, a frame-wide mean says more about what was
# photographed than about how it was graded.
MATCH_WEIGHTS: dict[str, float] = {
    "shadow_hue": 2.5,  # tints in shadows are almost always the grade
    "high_hue": 2.5,  # highlight tint likewise
    "shadow_a": 2.0,
    "shadow_b": 2.0,
    "high_a": 2.0,
    "high_b": 2.0,
    "shadow_C": 1.5,
    "high_C": 1.5,
    "shadow_coh": 1.0,  # is the shadow tint a grade or a colourful subject?
    "split": 1.0,  # angular gap between highlight and shadow tint
    "high_coh": 0.8,
}

# How much between-look spread to mix into the per-look within-spread when
# scaling a feature (see ``weighted_distance``). 0.0 uses within-look spread
# alone, which is unstable for features that happen to be near-identical
# across the base plates; large values wash the per-look reliability out.
#
# On the selection method: the nominal CV winner was a five-feature set picked
# directly from each feature's measured plate-spread ratio. It led by 1.6
# points with a standard error of 2.7, i.e. by less than noise, and it leant on
# clipping fractions that are structurally zero for most photographs. The
# one-standard-error rule preferred this broader set instead -- which then
# scored 54.1% top-1 on the held-out plates against that candidate's 37.0%.
# The rule was applied before the held-out plates were scored.
SPREAD_ALPHA = 1.5


# --------------------------------------------------------------------------
# HSL colour families -- Lightroom's HSL / Colour Mixer panel
# --------------------------------------------------------------------------
#
# Lightroom's HSL panel has eight named colour bands, and a photographer edits
# each one's Hue, Saturation and Luminance separately. To give advice in those
# terms we have to measure in those terms, so pixels are bucketed by hue into
# the same eight families and each family's chroma, lightness and hue offset
# are measured independently.
#
# The angles are CIELAB hue angles, MEASURED rather than assumed -- a colour
# picker's "red" is not at LAB 0 degrees. These come from converting each
# named colour and reading its hue back (see tests/test_domain.py).
HSL_FAMILIES: dict[str, float] = {
    "red": 31.0,
    "orange": 66.0,
    "yellow": 99.0,
    "green": 142.0,
    "aqua": 197.0,
    "blue": 271.0,
    "purple": 307.0,
    "magenta": 331.0,
}

# A pixel must carry at least this much chroma to be assigned to a family.
# Near-grey pixels have an essentially arbitrary hue, so letting them vote
# would smear every family toward the image's average cast -- the same reason
# the zone hue means are chroma-weighted.
HSL_MIN_CHROMA = 6.0

# A family needs at least this share of the frame before its statistics are
# reported. Below it, advice would be based on a handful of pixels.
HSL_MIN_COVERAGE = 0.01


def circ_dist(h1: float, h2: float) -> float:
    """Shortest angular distance between two hue angles, in degrees (0..180)."""
    d = abs(float(h1) - float(h2)) % 360.0
    return float(min(d, 360.0 - d))


# Backwards-compatible private alias used in the design document.
_circ_dist = circ_dist


def _empty_signature() -> dict[str, float]:
    return {name: 0.0 for name in FEATURE_NAMES}


def signature_from_array(rgb01: np.ndarray) -> dict[str, float]:
    """Compute a signature from a float RGB array in [0, 1], shape (H, W, 3)."""
    rgb01 = np.asarray(rgb01, dtype=np.float64)
    if rgb01.ndim != 3 or rgb01.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) RGB array, got {rgb01.shape}")
    rgb01 = np.clip(rgb01, 0.0, 1.0)

    lab = rgb2lab(rgb01)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.sqrt(a**2 + b**2)  # chroma
    h = np.degrees(np.arctan2(b, a)) % 360.0  # hue angle

    out = _empty_signature()
    coverage: dict[str, float] = {}
    total = float(L.size)

    for name, (lo, hi) in ZONES.items():
        # Top zone is closed at the top so L* == 100 pixels are not discarded.
        m = (L >= lo) & (L < hi) if hi < 100.0 else (L >= lo) & (L <= hi)
        coverage[name] = float(int(m.sum()) / total)
        if int(m.sum()) < MIN_ZONE_PIXELS:
            # Zone is empty or too small to trust -- leave the zeros in place.
            # This is the "exposure coupling" limitation of Part 5.4 made
            # explicit: brightening an image moves pixels out of the shadow
            # zone, so its statistics stop being measurable. Reporting zero
            # with zero coverage is honest; inventing a hue from 20 pixels
            # would not be.
            continue
        out[f"{name}_a"] = float(a[m].mean())
        out[f"{name}_b"] = float(b[m].mean())
        out[f"{name}_C"] = float(C[m].mean())

        # Chroma-weighted circular mean: grey pixels do not vote.
        w = C[m]
        rad = np.radians(h[m])
        vx = float((w * np.cos(rad)).sum())
        vy = float((w * np.sin(rad)).sum())
        out[f"{name}_hue"] = float(np.degrees(math.atan2(vy, vx)) % 360.0)
        # Coherence: magnitude of the mean hue vector, 0..1. Near 1 means the
        # zone is consistently tinted one way -- a grade. Near 0 means the hues
        # cancel -- a colourful subject, not a grade. This is the built-in
        # defence against subject-matter contamination.
        out[f"{name}_coh"] = float(math.hypot(vx, vy) / (float(w.sum()) + 1e-9))

    out["L_p05"] = float(np.percentile(L, 5))
    out["L_p50"] = float(np.percentile(L, 50))
    out["L_p95"] = float(np.percentile(L, 95))
    out["contrast"] = out["L_p95"] - out["L_p05"]
    out["chroma"] = float(C.mean())
    out["a_global"] = float(a.mean())  # tint proxy (green <-> magenta)
    out["b_global"] = float(b.mean())  # temp proxy (blue <-> yellow)
    out["clip_black"] = float((L < 1.0).mean())
    out["clip_white"] = float((L > 99.0).mean())
    out["split"] = circ_dist(out["high_hue"], out["shadow_hue"])

    # Not features -- diagnostic metadata and HSL-panel measurements. Every
    # function that iterates a signature iterates FEATURE_NAMES explicitly, so
    # extra keys are safe and the tuned matcher is unaffected.
    out["zone_coverage"] = coverage  # type: ignore[assignment]
    out["hsl"] = hsl_families(L, a, b, C, h)  # type: ignore[assignment]
    return out


def hsl_families(
    L: np.ndarray, a: np.ndarray, b: np.ndarray, C: np.ndarray, h: np.ndarray
) -> dict[str, dict[str, float]]:
    """Per-colour-family statistics, in Lightroom's HSL vocabulary.

    Each pixel with enough chroma is assigned to its nearest family centre,
    and the family reports:

    ``saturation``  mean chroma of its pixels -- the Saturation slider's axis
    ``luminance``   mean L* -- the Luminance slider's axis
    ``hue_shift``   signed offset from the family centre, in degrees, which is
                    what the Hue slider moves
    ``coverage``    share of the frame, so advice can be skipped for a family
                    that barely appears

    Grey pixels are excluded rather than assigned to whichever family their
    noise happens to point at.
    """
    names = list(HSL_FAMILIES)
    centres = np.array([HSL_FAMILIES[n] for n in names], dtype=np.float64)

    strong = C >= HSL_MIN_CHROMA
    out: dict[str, dict[str, float]] = {}
    if not bool(strong.any()):
        return {n: {"saturation": 0.0, "luminance": 0.0, "hue_shift": 0.0, "coverage": 0.0}
                for n in names}

    hs, Cs, Ls = h[strong], C[strong], L[strong]
    # Circular distance from every strong pixel to every family centre.
    diff = np.abs(hs[:, None] - centres[None, :]) % 360.0
    diff = np.minimum(diff, 360.0 - diff)
    nearest = np.argmin(diff, axis=1)
    total = float(L.size)

    for index, name in enumerate(names):
        member = nearest == index
        count = int(member.sum())
        coverage = count / total
        if count < MIN_ZONE_PIXELS or coverage < HSL_MIN_COVERAGE:
            out[name] = {"saturation": 0.0, "luminance": 0.0, "hue_shift": 0.0,
                         "coverage": round(coverage, 4)}
            continue
        # Signed circular offset from the centre, chroma-weighted like the
        # zone hue means so the more saturated pixels dominate.
        offset = (hs[member] - HSL_FAMILIES[name] + 180.0) % 360.0 - 180.0
        weights = Cs[member]
        out[name] = {
            "saturation": round(float(Cs[member].mean()), 3),
            "luminance": round(float(Ls[member].mean()), 3),
            "hue_shift": round(float(np.average(offset, weights=weights)), 3),
            "coverage": round(coverage, 4),
        }
    return out


def zone_is_measurable(sig: dict, zone: str, min_fraction: float = 0.01) -> bool:
    """Whether a zone held enough pixels for its statistics to mean anything."""
    cov = sig.get("zone_coverage") or {}
    return float(cov.get(zone, 1.0)) >= min_fraction


def load_rgb(source: Any, max_side: int = 512) -> np.ndarray:
    """Open an image from a path, bytes, or file object; return float RGB [0,1].

    Uses ``thumbnail`` with LANCZOS rather than a naive resize: point sampling
    aliases and shifts colour statistics.
    """
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(bytes(source))
    img = Image.open(source)
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    return np.asarray(img, dtype=np.float64) / 255.0


def signature(path_or_bytes: Any, max_side: int = 512) -> dict[str, float]:
    """Compute the LAB signature of an image given a path, bytes or file object."""
    return signature_from_array(load_rgb(path_or_bytes, max_side=max_side))


# --------------------------------------------------------------------------
# Distance and matching
# --------------------------------------------------------------------------


def feature_stats(signatures: Iterable[dict[str, float]]) -> dict[str, float]:
    """Per-feature standard deviations over a collection, used for z-scoring.

    Computed over the knowledge base so distances are scale-free: ``contrast``
    spans tens of L* units while ``a_global`` spans single digits, and an
    un-normalised Euclidean distance would be dominated by whichever feature
    happens to have the largest units.
    """
    sigs = list(signatures)
    stats: dict[str, float] = {}
    for name in FEATURE_NAMES:
        vals = [float(s.get(name, 0.0)) for s in sigs]
        if len(vals) < 2:
            stats[name] = 1.0
            continue
        sd = float(np.std(vals))
        stats[name] = sd if sd > 1e-6 else 1.0
    return stats


def weighted_distance(
    a: dict[str, float],
    b: dict[str, float],
    weights: dict[str, float] | None = None,
    scales: dict[str, float] | None = None,
    spread: dict[str, float] | None = None,
    alpha: float = SPREAD_ALPHA,
) -> float:
    """Weighted, scale-normalised distance between two signatures.

    Hue features use circular distance (normalised by 180 degrees); everything
    else uses an absolute difference. Both are divided by a per-feature scale
    so the result is unit-free -- without that, ``contrast`` (tens of L* units)
    would swamp ``a_global`` (single digits).

    ``scales`` is the between-look standard deviation over the whole library.
    ``spread`` is the optional within-look standard deviation across base
    plates for the specific look being compared against. When it is supplied,
    the effective scale becomes ``hypot(within, alpha * between)``, which
    down-weights features that are unreliable *for that look* -- a feature
    that swings wildly across plates is being driven by the subject, so a
    mismatch on it is weak evidence. This is a regularised Mahalanobis-style
    scaling, and it is what lifts top-1 accuracy from 60% to 84%.

    Returns the weighted RMS, so values are comparable across weight sets.
    """
    weights = weights or MATCH_WEIGHTS
    scales = scales or {}
    total_w = 0.0
    acc = 0.0
    for name, w in weights.items():
        if w <= 0:
            continue
        av, bv = float(a.get(name, 0.0)), float(b.get(name, 0.0))
        circular = name in CIRCULAR_FEATURES
        unit = 180.0 if circular else 1.0
        d = (circ_dist(av, bv) if circular else abs(av - bv)) / unit

        between = (float(scales.get(name, 1.0)) or 1.0) / unit
        if spread is not None:
            within = float(spread.get(name, 0.0)) / unit
            scale = math.hypot(within, alpha * between)
        else:
            scale = between
        acc += w * (d / max(scale, 1e-6)) ** 2
        total_w += w
    if total_w <= 0:
        return 0.0
    return float(math.sqrt(acc / total_w))


def signature_delta(current: dict[str, float], target: dict[str, float]) -> dict[str, float]:
    """target - current, per feature. Hue features use signed circular difference."""
    out: dict[str, float] = {}
    for name in FEATURE_NAMES:
        cv, tv = float(current.get(name, 0.0)), float(target.get(name, 0.0))
        if name in CIRCULAR_FEATURES:
            d = (tv - cv + 180.0) % 360.0 - 180.0
            out[name] = float(d)
        else:
            out[name] = float(tv - cv)
    return out
