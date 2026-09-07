"""Base plates: the neutral images every reference look is measured on.

Part 6 of the design document asks for base plates from a personal photo
library, exported as neutral sRGB so the RAW profile stops being a free
variable. Those exports do not exist, so this module uses freely-licensed real
photographs instead, in two tiers:

**Tier 1 -- downloaded plates** (``looklab/data/plates/``). Sixteen photographs
pulled from Wikimedia Commons by ``tools/fetch_plates.py``, filtered so every
L* zone is populated, each carrying its licence and author in ``MANIFEST.json``
and ``ATTRIBUTION.md``. Only public-domain, CC0, CC BY and CC BY-SA images are
accepted; anything else is skipped rather than guessed at.

**Tier 2 -- bundled fallback.** If the download has not been run, the sample
photographs that ship inside scikit-image are used instead. They are real
photographs, they need no network, and they keep a fresh clone working. This
is why the repo is functional immediately after ``git clone``.

Train/test split
----------------
Plates are split so calibration and identification accuracy are reported on
photographs the knowledge base never saw. The split is by index within each
content category -- ``portrait_daylight_0`` builds the KB, ``portrait_daylight_1``
is held out -- so the held-out set spans the same subject categories but
entirely different photographs. That measures the thing that actually matters:
does this generalise to a new photo?

A synthetic 24-patch colour chart is also held out. It has no subject matter at
all, so it isolates the grade from subject contamination completely.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

import numpy as np

PLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "plates")
MANIFEST_PATH = os.path.join(PLATE_DIR, "MANIFEST.json")

# Longest edge the plates are resized to. Keeps the KB build fast while
# leaving every L* zone well above MIN_ZONE_PIXELS.
PLATE_SIZE = 384

# Used only when no downloaded plates are present.
FALLBACK_KB = ("astronaut", "chelsea", "coffee")
FALLBACK_HELDOUT = ("rocket",)

# Always held out, downloaded plates or not.
SYNTHETIC_HELDOUT = ("chart",)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def _resize(rgb01: np.ndarray, max_side: int) -> np.ndarray:
    """Downscale with PIL/LANCZOS -- point sampling would shift the statistics."""
    from PIL import Image

    img = Image.fromarray((np.clip(rgb01, 0, 1) * 255).astype(np.uint8))
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    return np.asarray(img, dtype=np.float64) / 255.0


def _load_sample(name: str) -> np.ndarray:
    """Load a scikit-image sample photograph as float RGB in [0, 1]."""
    import skimage.data as data

    arr = np.asarray(getattr(data, name)())
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[2] == 4:
        arr = arr[..., :3]
    return np.clip(arr.astype(np.float64) / 255.0, 0.0, 1.0)


def _load_file(path: str) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float64) / 255.0


@lru_cache(maxsize=1)
def manifest() -> dict:
    """Downloaded-plate manifest, or an empty one when none were fetched."""
    if not os.path.exists(MANIFEST_PATH):
        return {"plates": []}
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {"plates": []}
    # Only trust entries whose file actually exists on disk.
    data["plates"] = [
        p for p in data.get("plates", []) if os.path.exists(os.path.join(PLATE_DIR, p.get("file", "")))
    ]
    return data


def using_downloaded_plates() -> bool:
    return len(manifest()["plates"]) >= 4


def _split_names() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(kb plate names, held-out plate names)."""
    entries = manifest()["plates"]
    if len(entries) < 4:
        return FALLBACK_KB, FALLBACK_HELDOUT + SYNTHETIC_HELDOUT

    kb, held = [], []
    for entry in entries:
        name = entry["name"]
        # Category suffix _0 builds the KB, everything else is held out, so the
        # two sets share subject categories but never share a photograph.
        (kb if name.endswith("_0") else held).append(name)

    # Degenerate manifests (all one index) still need a usable split.
    if not kb or not held:
        names = [e["name"] for e in entries]
        half = max(1, len(names) // 2)
        kb, held = names[:half], names[half:]
    return tuple(kb), tuple(held) + SYNTHETIC_HELDOUT


PLATE_NAMES: tuple[str, ...] = _split_names()[0]
HELDOUT_NAMES: tuple[str, ...] = _split_names()[1]


def _colour_chart(size: int) -> np.ndarray:
    """A 24-patch colour chart over a neutral grey ramp.

    Deliberately synthetic: there is no subject to contaminate the statistics,
    the patch colours are exact, and the grey ramp guarantees every L* zone is
    populated. The cleanest possible calibration target.
    """
    patches: list[tuple[float, float, float]] = []
    for sat in (0.85, 0.55, 0.30):
        for k in range(6):
            hue = k / 6.0
            i = int(hue * 6) % 6
            f = hue * 6 - int(hue * 6)
            v = 0.80
            p, q, t = v * (1 - sat), v * (1 - sat * f), v * (1 - sat * (1 - f))
            patches.append([(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i])
    for k in range(6):
        g = 0.04 + k * (0.94 - 0.04) / 5.0
        patches.append((g, g, g))

    rows, cols = 4, 6
    img = np.zeros((size, size, 3), dtype=np.float64)
    ch, cw = size // rows, size // cols
    for idx, colour in enumerate(patches[: rows * cols]):
        r, c = divmod(idx, cols)
        img[r * ch : (r + 1) * ch, c * cw : (c + 1) * cw] = colour
    if ch * rows < size:
        img[ch * rows :, :] = img[ch * rows - 1, :]
    if cw * cols < size:
        img[:, cw * cols :] = img[:, cw * cols - 1][:, None, :]

    # A little grain: perfectly uniform patches give hue coherence of exactly
    # 1.0, which is not a realistic input to the measurement code.
    rng = np.random.default_rng(24)
    return np.clip(img + rng.normal(0.0, 0.012, img.shape), 0.0, 1.0)


@lru_cache(maxsize=64)
def _cached_plate(name: str, size: int) -> tuple:
    """Cached loader. Returns a tuple so lru_cache can hold it; unpacked below."""
    if name == "chart":
        arr = _colour_chart(size)
    else:
        entry = next((p for p in manifest()["plates"] if p["name"] == name), None)
        if entry is not None:
            arr = _resize(_load_file(os.path.join(PLATE_DIR, entry["file"])), size)
        else:
            arr = _resize(_load_sample(name), size)
    return (arr.tobytes(), arr.shape)


def base_plate(name: str, size: int = PLATE_SIZE) -> np.ndarray:
    """Return a named plate as float RGB in [0, 1]."""
    known = set(PLATE_NAMES) | set(HELDOUT_NAMES) | set(FALLBACK_KB) | set(FALLBACK_HELDOUT)
    if name not in known and name != "chart":
        raise KeyError(f"unknown plate {name!r}; expected one of {sorted(known)}")
    raw, shape = _cached_plate(name, size)
    return np.frombuffer(raw, dtype=np.float64).reshape(shape).copy()


def all_plates(size: int = PLATE_SIZE) -> dict[str, np.ndarray]:
    """The plates the knowledge base is built from."""
    return {name: base_plate(name, size) for name in PLATE_NAMES}


def heldout_plates(size: int = PLATE_SIZE) -> dict[str, np.ndarray]:
    """Plates reserved for calibration; never used to build ``kb.json``."""
    return {name: base_plate(name, size) for name in HELDOUT_NAMES}


def provenance() -> dict:
    """Where the plates came from, for the KB metadata and the README."""
    entries = manifest()["plates"]
    return {
        "downloaded": using_downloaded_plates(),
        "source": "Wikimedia Commons" if entries else "scikit-image bundled samples",
        "n_available": len(entries),
        "kb_plates": list(PLATE_NAMES),
        "heldout_plates": list(HELDOUT_NAMES),
        "licences": sorted({p["licence"] for p in entries}) if entries else ["BSD (scikit-image)"],
    }
