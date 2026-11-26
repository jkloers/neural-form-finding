"""``regularise_rectangle``: the rectangle closest to a quadrilateral that ought to be one."""

import numpy as np
import pytest

from nff.calibration.photo_geometry import regularise_rectangle


def _angles(q):
    """Interior angles of a quadrilateral, in degrees."""
    out = []
    for i in range(4):
        a, b = q[i] - q[i - 1], q[(i + 1) % 4] - q[i]
        c = np.dot(-a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        out.append(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
    return np.array(out)


def _sides(q):
    return np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1)


def _rect(cx, cy, w, h, deg):
    a = np.radians(deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    local = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    return np.array([cx, cy]) + local @ R.T


def test_output_is_a_rectangle():
    q = np.array([[0.0, 0.0], [1000.0, 40.0], [960.0, 620.0], [-30.0, 570.0]])
    r = regularise_rectangle(q)
    assert np.allclose(_angles(r), 90.0, atol=1e-6)
    s = _sides(r)
    assert s[0] == pytest.approx(s[2], abs=1e-6)
    assert s[1] == pytest.approx(s[3], abs=1e-6)


def test_a_rectangle_is_a_fixed_point():
    r0 = _rect(120.0, -80.0, 1740.0, 2714.0, -15.6)
    assert np.allclose(regularise_rectangle(r0), r0, atol=1e-6)


def test_winding_is_preserved():
    """The caller draws the result as a polygon and warps masks alongside it, so corner i of the
    output must be corner i of the input, not its mirror."""
    r0 = _rect(0.0, 0.0, 300.0, 500.0, 20.0)
    for q in (r0, r0[::-1].copy()):
        r = regularise_rectangle(q)
        assert np.hypot(*(r - q).T).max() < 1e-6


def test_stays_near_every_corner():
    """A least-squares rectangle, not one pinned to a subset of the corners."""
    rng = np.random.default_rng(0)
    r0 = _rect(500.0, 900.0, 1689.0, 2680.0, -15.6)
    q = r0 + rng.normal(0.0, 20.0, r0.shape)
    d = np.hypot(*(regularise_rectangle(q) - q).T)
    assert d.max() < 60.0
    assert d.min() > 0.0


def test_the_paper_target():
    """The measured floor tape of the 3x3 deployment, which Fig. 10 draws as a rectangle.

    Pinned because the plate is authored against these numbers: the tape is 2.74 deg out of square,
    and the rectangle drawn in its place is 1689 x 2680 mm and never more than 34.7 mm away, which
    is 2.8 bp on the printed page.
    """
    q = np.array([[1234.42, -247.39], [-371.88, 179.55], [320.07, 2809.31], [1958.56, 2293.30]])
    r = regularise_rectangle(q)
    assert np.abs(_angles(q) - 90.0).max() == pytest.approx(2.74, abs=0.01)
    assert np.allclose(_angles(r), 90.0, atol=1e-6)
    assert sorted(np.round(_sides(r), 0))[::2] == [1689.0, 2680.0]
    assert np.hypot(*(r - q).T).max() == pytest.approx(34.7, abs=0.1)
