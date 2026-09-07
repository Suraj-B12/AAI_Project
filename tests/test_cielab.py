"""Proof that the in-house sRGB->CIELAB conversion is correct.

``looklab/cielab.py`` replaced ``skimage.color.rgb2lab`` to drop scikit-image
and scipy -- 143 MB of the 206 MB install, for one function. Replacing a
reference implementation with your own is only acceptable if you can show they
agree, so this file does that two ways:

* **Against scikit-image**, when it is installed: bit-for-bit equality across
  the sRGB cube, the greyscale ramp, both piecewise knees, random images and
  every base plate. scikit-image is an optional dev dependency, so these tests
  skip cleanly when it is absent.
* **Against published reference values**, always: known LAB coordinates for
  white, black, mid-grey and the sRGB primaries, so correctness is still
  pinned down on a machine that has never had scikit-image installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from looklab.cielab import linear_to_srgb, rgb2lab, srgb_to_linear

def _sk():
    """Return skimage's rgb2lab, or skip the test if it is not installed."""
    color = pytest.importorskip(
        "skimage.color", reason="scikit-image is an optional dev dependency"
    )
    return color.rgb2lab


# --------------------------------------------------------------------------
# Equivalence with the reference implementation
# --------------------------------------------------------------------------

def test_matches_skimage_across_the_srgb_cube():
    sk = _sk()
    g = np.linspace(0.0, 1.0, 20)
    r, gr, b = np.meshgrid(g, g, g, indexing="ij")
    cube = np.stack([r, gr, b], axis=-1).reshape(-1, 3)
    assert np.abs(sk(cube) - rgb2lab(cube)).max() < 1e-12


def test_matches_skimage_on_the_greyscale_ramp():
    sk = _sk()
    ramp = np.repeat(np.linspace(0.0, 1.0, 2001)[:, None], 3, axis=1)
    assert np.abs(sk(ramp) - rgb2lab(ramp)).max() < 1e-12


def test_matches_skimage_at_both_piecewise_knees():
    """The two curves each have a linear segment near zero; check the joins."""
    sk = _sk()
    probes = np.array(
        [
            [0.04045] * 3,  # sRGB companding knee, exactly
            [0.04045 + 1e-9] * 3,
            [0.04045 - 1e-9] * 3,
            [0.0031308] * 3,
            [0.008856] * 3,  # the CIE f() knee, in XYZ terms
            [0.0] * 3,
            [1.0] * 3,
        ],
        dtype=np.float64,
    )
    assert np.abs(sk(probes) - rgb2lab(probes)).max() < 1e-12


def test_matches_skimage_on_random_images():
    sk = _sk()
    rng = np.random.default_rng(0)
    for shape in ((64, 64, 3), (129, 7, 3), (1, 1, 3)):
        img = rng.random(shape)
        assert np.abs(sk(img) - rgb2lab(img)).max() < 1e-12


def test_matches_skimage_on_every_base_plate():
    """The images this project actually measures, not just synthetic ones."""
    sk = _sk()
    from looklab.plates import all_plates, heldout_plates

    for name, plate in {**all_plates(), **heldout_plates()}.items():
        assert np.abs(sk(plate) - rgb2lab(plate)).max() < 1e-12, name


# --------------------------------------------------------------------------
# Correctness without scikit-image present
# --------------------------------------------------------------------------

def test_known_reference_colours():
    """Published CIELAB values for colours whose coordinates are documented.

    These hold with no reference implementation available, so correctness is
    still pinned on a deployment that installs only the runtime requirements.
    """
    lab = rgb2lab(
        np.array(
            [
                [1.0, 1.0, 1.0],  # white -> L*=100, a*=b*=0
                [0.0, 0.0, 0.0],  # black -> L*=0,   a*=b*=0
                [1.0, 0.0, 0.0],  # sRGB red
                [0.0, 1.0, 0.0],  # sRGB green
                [0.0, 0.0, 1.0],  # sRGB blue
            ],
            dtype=np.float64,
        )
    )
    white, black, red, green, blue = lab

    assert white[0] == pytest.approx(100.0, abs=1e-6)
    # Not exactly zero -- see test_greys_are_neutral_to_within_the_matrix_rounding.
    assert white[1] == pytest.approx(0.0, abs=5e-3)
    assert white[2] == pytest.approx(0.0, abs=5e-3)

    assert black[0] == pytest.approx(0.0, abs=1e-6)
    assert black[1] == pytest.approx(0.0, abs=1e-6)

    # Standard sRGB primary coordinates under D65/2.
    assert red[0] == pytest.approx(53.24, abs=0.05)
    assert red[1] == pytest.approx(80.09, abs=0.05)
    assert red[2] == pytest.approx(67.20, abs=0.05)

    assert green[0] == pytest.approx(87.73, abs=0.05)
    assert green[1] == pytest.approx(-86.18, abs=0.05)

    assert blue[0] == pytest.approx(32.30, abs=0.05)
    assert blue[2] == pytest.approx(-107.86, abs=0.05)


def test_mid_grey_sits_at_l_53_not_l_50():
    """sRGB 0.5 grey is L* 53.4, not 50 -- the transfer curve is not linear.

    A useful sanity check that companding is actually being applied: skip the
    gamma step and mid-grey lands near L* 76 instead.
    """
    lab = rgb2lab(np.array([[0.5, 0.5, 0.5]]))[0]
    assert lab[0] == pytest.approx(53.39, abs=0.05)


def test_lightness_is_monotone_in_grey_level():
    ramp = np.repeat(np.linspace(0.0, 1.0, 500)[:, None], 3, axis=1)
    L = rgb2lab(ramp)[:, 0]
    assert np.all(np.diff(L) > 0)
    assert L[0] == pytest.approx(0.0, abs=1e-9)
    assert L[-1] == pytest.approx(100.0, abs=1e-9)


def test_greys_are_neutral_to_within_the_matrix_rounding():
    """Greys land near a*=b*=0, but not exactly, and that is correct.

    The published sRGB->XYZ matrix has row sums (0.950456, 1, 1.088754) while
    the D65 reference white is (0.95047, 1, 1.08883). Those differ in the fifth
    decimal, so a perfectly neutral grey picks up a residual a* of about
    -0.0025 at white. scikit-image produces exactly the same residual -- it is
    a property of the standard constants, not of this implementation.

    Asserting exact zero here would be asserting something false about CIELAB.
    """
    ramp = np.repeat(np.linspace(0.0, 1.0, 200)[:, None], 3, axis=1)
    lab = rgb2lab(ramp)
    assert np.abs(lab[:, 1]).max() < 5e-3
    assert np.abs(lab[:, 2]).max() < 1e-2
    # Still far below anything a photograph could show: chroma < 0.01 against
    # the ~5-40 range the project actually measures.
    assert np.hypot(lab[:, 1], lab[:, 2]).max() < 1e-2


# --------------------------------------------------------------------------
# Companding
# --------------------------------------------------------------------------

def test_companding_round_trips():
    x = np.linspace(0.0, 1.0, 1000)
    assert np.abs(linear_to_srgb(srgb_to_linear(x)) - x).max() < 1e-12


def test_companding_is_piecewise_not_a_plain_power():
    """A plain 2.2 power is visibly wrong near black, which is where we measure."""
    x = np.linspace(0.0, 0.04, 50)
    assert np.abs(srgb_to_linear(x) - x**2.2).max() > 1e-4
    # Below the knee the curve is exactly linear.
    assert np.abs(srgb_to_linear(x) - x / 12.92).max() < 1e-12


# --------------------------------------------------------------------------
# Shape and input handling
# --------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(3,), (5, 3), (4, 6, 3), (2, 3, 4, 3)])
def test_accepts_any_trailing_three_axis(shape):
    rng = np.random.default_rng(1)
    out = rgb2lab(rng.random(shape))
    assert out.shape == shape


def test_rejects_a_wrong_final_axis():
    with pytest.raises(ValueError, match="length 3"):
        rgb2lab(np.zeros((4, 4)))
    with pytest.raises(ValueError, match="length 3"):
        rgb2lab(np.zeros((4, 4, 4)))


def test_out_of_range_input_is_clamped_not_wrapped():
    """A slightly out-of-gamut float must not produce a wild LAB value."""
    lab = rgb2lab(np.array([[1.4, -0.3, 0.5]]))[0]
    clamped = rgb2lab(np.array([[1.0, 0.0, 0.5]]))[0]
    assert np.allclose(lab, clamped)


def test_signature_pipeline_still_works_without_skimage(monkeypatch):
    """Import-guard check: measuring must not reach scikit-image at all."""
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("skimage") or name.startswith("scipy"):
            raise ImportError(f"{name} is blocked by this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)

    from looklab.color import signature_from_array

    rng = np.random.default_rng(2)
    sig = signature_from_array(rng.random((64, 64, 3)))
    assert 0.0 <= sig["L_p50"] <= 100.0
    assert "shadow_hue" in sig
