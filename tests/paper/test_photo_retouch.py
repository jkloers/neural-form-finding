"""Synthetic tests for the publication retouching of a rectified photograph.

The subject is a set of operations that ERASE things from a photograph, so the properties worth
pinning are the ones that keep such an operation honest: it must not touch a pixel outside its
mask, it must report the area it covered, and its masks must be geometry rather than luck. Each
test builds a ground truth, damages it, and asks for the damage back.
"""

import os

import numpy as np
import pytest

from nff_paper.lab import photo_retouch as prt

SHAPE = (180, 240)


def _ground(shape=SHAPE):
    """A smooth floor with a gentle gradient, which is what a flattened lab floor looks like."""
    y, x = np.mgrid[0:shape[0], 0:shape[1]]
    base = 0.78 + 0.06 * (x / shape[1]) - 0.03 * (y / shape[0])
    return np.repeat(base[..., None], 3, axis=-1).astype(float)


def _grid(shape=SHAPE, mm_per_px=2.0):
    return (np.arange(shape[1]) * mm_per_px, np.arange(shape[0]) * mm_per_px)


# ── the property that makes a retouch auditable ───────────────────────────────────────────────

def test_inpaint_leaves_every_unmasked_pixel_bit_exact():
    """A retouch that quietly nudges the rest of the frame is not a retouch, it is a re-render."""
    img = _ground()
    img[40:60, 70:110] = np.nan
    mask = np.zeros(SHAPE, bool)
    mask[40:60, 70:110] = True
    out = prt.inpaint(img, mask, iters=40)
    assert np.array_equal(out[~mask], np.asarray(img, float)[~mask])


def test_inpaint_reconstructs_a_smooth_ground_it_never_saw():
    truth = _ground()
    img = truth.copy()
    mask = np.zeros(SHAPE, bool)
    mask[60:90, 100:150] = True
    img[mask] = np.nan
    out = prt.inpaint(img, mask, iters=400)
    assert np.abs(out[mask] - truth[mask]).max() < 0.01


def test_every_operation_reports_its_area():
    """The ledger is the disclosure. An operation that runs without a number is the failure mode."""
    img = _ground()
    img[:12, :12] = np.nan
    gx, gy = _grid()
    out, ledger = prt.retouch(img, gx=gx, gy=gy, glare=False,
                              objects=[dict(kind="disc", xy=(200.0, 200.0), r=20.0, why="weight")])
    assert set(ledger) == {"empty", "objects"}
    for st in ledger.values():
        assert st["px"] > 0 and 0.0 < st["frac"] < 1.0 and st["mm2"] > 0.0
    assert ledger["objects"]["named"] == ["weight"]
    assert any("weight" in line for line in prt.ledger_lines(ledger))
    assert np.isfinite(out).all()


# ── the empty corners ─────────────────────────────────────────────────────────────────────────

def test_fill_empty_leaves_no_hole_and_no_white():
    """White is what makes a warp's empty corner read as a tear; the floor is what belongs there."""
    img = _ground()
    ys, xs = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    img[xs + ys < 60] = np.nan                       # a keystoned corner, as an inverse warp leaves
    out, st = prt.fill_empty(img, iters=300)
    assert np.isfinite(out).all()
    assert out.max() < 0.99, "the empty region came back white"
    assert st["px"] == int((xs + ys < 60).sum())


def test_empty_mask_finds_a_partially_nan_pixel():
    """A single channel lost is still a pixel with no data."""
    img = _ground()
    img[5, 5, 1] = np.nan
    assert prt.empty_mask(img)[5, 5]


# ── the reflection ────────────────────────────────────────────────────────────────────────────

def test_specular_mask_finds_the_highlight_and_not_the_floor():
    """⚠ The floor is bright EVERYWHERE, so an unrestricted search returns the floor, not the glare.

    This is the whole reason ``inside`` exists, and it is the mistake the parameter prevents.
    """
    img = _ground()
    y, x = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    inside = (x > 40) & (x < 200) & (y > 30) & (y < 150)
    blob = np.exp(-(((x - 120) / 18.0) ** 2 + ((y - 80) / 12.0) ** 2))
    img = np.clip(img + 0.9 * blob[..., None], 0, 1)

    m = prt.specular_mask(img, inside, min_area_px=20)
    assert m.any()
    assert m[80, 120], "the crest of the highlight is not in the mask"
    assert not m[inside].all(), "the mask swallowed the whole specimen"
    assert (m & ~inside).sum() < 0.30 * m.sum(), "the mask escaped onto the floor"


def test_glare_is_removed_by_the_retouch():
    truth = _ground()
    y, x = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    inside = (x > 40) & (x < 200) & (y > 30) & (y < 150)
    blob = np.exp(-(((x - 120) / 18.0) ** 2 + ((y - 80) / 12.0) ** 2))
    img = np.clip(truth + 0.9 * blob[..., None], 0, 1)
    gx, gy = _grid()

    # The blob CLIPS, exactly as a photographed highlight does, so its excess over the floor is
    # bounded by the headroom above the floor rather than by the amplitude added.
    before = float(np.abs(img - truth)[inside].max())
    out, ledger = prt.retouch(img, gx=gx, gy=gy, inside=inside, glare_kw=dict(min_area_px=20))
    after = float(np.abs(out - truth)[inside].max())
    assert before > 0.15 and after < 0.05, f"glare {before:.2f} -> {after:.2f}"
    assert ledger["glare"]["px"] > 0


# ── named regions ─────────────────────────────────────────────────────────────────────────────

def test_region_mask_is_millimetres_not_pixels():
    """A region pinned in millimetres must cover the same object at any grid resolution."""
    shapes = [dict(kind="disc", xy=(240.0, 180.0), r=40.0, why="weight")]
    areas = []
    for mm_per_px in (2.0, 1.0, 0.5):
        gx = np.arange(0.0, 480.0, mm_per_px)
        gy = np.arange(0.0, 360.0, mm_per_px)
        m = prt.region_mask(gx, gy, shapes)
        areas.append(m.sum() * mm_per_px ** 2)
    assert np.allclose(areas, np.pi * 40.0 ** 2, rtol=0.02)


def test_region_mask_segment_covers_a_cable_end_to_end():
    gx, gy = _grid()
    m = prt.region_mask(gx, gy, [dict(kind="segment", a=(20.0, 20.0), b=(400.0, 300.0), w=12.0)])
    assert m[int(20 / 2.0), int(20 / 2.0)] and m[int(300 / 2.0), int(400 / 2.0)]
    assert m[int(160 / 2.0), int(210 / 2.0)], "the middle of the cable is not covered"


def test_unknown_shape_is_an_error_not_a_silent_pass():
    gx, gy = _grid()
    with pytest.raises(ValueError):
        prt.region_mask(gx, gy, [dict(kind="blob", xy=(1.0, 1.0), r=1.0)])


def test_neutralise_keeps_the_shading_and_drops_the_hue():
    img = _ground()
    mask = np.zeros(SHAPE, bool)
    mask[20:30, :] = True
    img[mask] = np.array([0.84, 0.16, 0.16])         # target tape, the red the plate spends on error
    out = prt.neutralise(img, mask)
    r, g, b = out[mask].mean(axis=0)
    assert abs(r - g) < 0.03 and abs(g - b) < 0.05, "the hue survived"
    assert 0.5 < r < 1.0, "the tape went black instead of neutral"
    assert np.array_equal(out[~mask], img[~mask])


# ── the framing that removed the white corners ────────────────────────────────────────────────

def test_sheet_framing_keeps_the_panel_inside_the_photograph():
    """⚠ The white corners were a FRAMING choice, not a fill colour.

    An inverse warp fills whatever its source did not cover, and a photograph's footprint in the
    rectified frame is a keystone, so an axis-aligned crop that reaches out to the taped target
    leaves a tenth of the panel empty while one that stops at the sheet leaves almost none. This
    pins both numbers, because raising the margin far enough silently brings the corners back.
    """
    pytest.importorskip("shapely")
    from shapely.geometry import Polygon, box
    from nff_paper.lab.photo_rectify import apply_h
    hb = pytest.importorskip("nff_paper.figures.plot_hinge_buckle_figure")

    measured = f"{hb.DEPLOY}/derived/measured.npz"
    if not os.path.exists(measured):
        pytest.skip("lab measurement not present (data/ is gitignored)")
    z = np.load(measured, allow_pickle=True)
    R = hb.rot_h(hb.ROT_DEG)
    foot = Polygon(apply_h(R @ z["Deployed_H"],
                           np.array([[0, 0], [4032, 0], [4032, 3024], [0, 3024]], float)))
    empty = {}
    for frame_on, margin in (("sheet", 0.022), ("target", 0.018)):
        x0, x1, y0, y1, tiles, _ = hb.context_window(frame_on, margin)
        rect = box(x0, y0, x1, y1)
        empty[frame_on] = rect.difference(foot).area / rect.area
        if frame_on == "sheet":
            pts = tiles.reshape(-1, 2)
            assert (x0 <= pts[:, 0]).all() and (pts[:, 0] <= x1).all(), "a panel is cropped away"
            assert (y0 <= pts[:, 1]).all() and (pts[:, 1] <= y1).all(), "a panel is cropped away"
    assert empty["sheet"] < 0.01, "the sheet framing has grown corners again: %.3f" % empty["sheet"]
    assert empty["target"] > 0.04, "the target framing was the reason to move; it no longer is"
