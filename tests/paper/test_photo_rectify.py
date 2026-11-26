"""Synthetic recovery tests for the deployment-photograph measurement chain.

Every test renders geometry through a KNOWN homography and then asks the code to recover it, so a
failure localises to a stage rather than to "the numbers moved". The failure modes pinned here are
the ones the real photographs actually exhibit: a fiducial broken into pieces by the specimen lying
across it, a decoy blob inside the frame, and a handedness that cannot be assumed because image y
runs downward.
"""

import numpy as np
import pytest

from nff_paper.lab import photo_geometry as pg
from nff_paper.lab import photo_rectify as pr

SHAPE = (900, 1200)


def _render_segment(mask, a, b, half_width):
    """Paint a thick line segment into a boolean mask."""
    ys, xs = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
    P = np.stack([xs.ravel(), ys.ravel()], axis=-1).astype(float)
    ab = np.asarray(b, float) - np.asarray(a, float)
    t = np.clip(((P - a) @ ab) / float(ab @ ab), 0.0, 1.0)
    d = np.linalg.norm(P - (np.asarray(a, float) + t[:, None] * ab), axis=1)
    mask |= (d <= half_width).reshape(mask.shape)


def _render_polygon(mask, poly, half_width=4.0):
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        _render_segment(mask, a, b, half_width)


def _known_homography():
    """A world->image map with real perspective, not just a similarity."""
    world = pr.rect_corners(1739.8, 2714.3, cyclic_from="ll")
    image = np.array([[210.0, 760.0], [1010.0, 830.0], [935.0, 130.0], [265.0, 95.0]])
    return pr.homography_dlt(world, image), world, image


def test_dlt_round_trip_and_normalisation_invariance():
    H, world, image = _known_homography()
    assert np.allclose(pr.apply_h(H, world), image, atol=1e-6)
    # a homography is scale-free: the recovered map must not depend on the units of its inputs
    H2 = pr.homography_dlt(world * 1000.0, image)
    assert np.allclose(pr.apply_h(H2, world * 1000.0), image, atol=1e-6)


def test_quad_recovered_from_a_broken_fiducial_with_a_decoy():
    """The real fiducial arrives in 2-4 pieces with a speck inside it; that is what is tested."""
    H, world, image = _known_homography()
    mask = np.zeros(SHAPE, bool)
    _render_polygon(mask, image, half_width=5.0)
    # cut the tape into pieces, as the specimen does where it crosses
    for (x0, x1, y0, y1) in [(430, 560, 60, 190), (900, 1010, 380, 520), (300, 420, 700, 830)]:
        mask[y0:y1, x0:x1] = False
    # a speck of the same colour lying inside the frame
    _render_segment(mask, (600, 420), (625, 445), 4.0)

    quad = pr.fit_quad_klines(mask, min_area_frac=0.02, end_trim=0.10)
    assert quad.n_components >= 3, "the test must actually present a broken fiducial"
    # corners are cyclic but may start anywhere and run either way
    best = min(min(np.linalg.norm(np.roll(c, k, axis=0) - image, axis=1).max()
                   for k in range(4))
               for c in (quad.corners, quad.corners[::-1]))
    assert best < 0.5, f"corner error {best:.2f} px"


def test_tile_poses_recovered_through_a_homography():
    """End to end: pose known polygons against a rendered mask and recover the identity."""
    H, world, image = _known_homography()
    # inside the world rectangle the homography is defined on, or the tiles project off-image
    polys = np.array([
        [[-780.0, -1150.0], [-60.0, -1190.0], [-40.0, -380.0], [-800.0, -350.0]],
        [[40.0, -1180.0], [790.0, -1140.0], [770.0, -360.0], [20.0, -400.0]],
        [[-760.0, -240.0], [-70.0, -260.0], [-90.0, 620.0], [-780.0, 590.0]],
    ])
    mask = np.zeros(SHAPE, bool)
    for poly in polys:
        _render_polygon(mask, pr.apply_h(H, poly), half_width=3.0)

    ctxs = [pg.FitContext(pg.matched_field(mask, s), np.ones(SHAPE, np.float32), H, SHAPE)
            for s in (14.0, 7.0, 3.5)]
    # start displaced, so the fit has something to recover rather than a pose already correct
    x0 = np.tile(np.array([18.0, -22.0, 1.1]), (len(polys), 1))
    poses, scores, cover = pg.fit_all_tiles(ctxs, polys, spacing=6.0, coarse=None,
                                            x0=x0, bounds=(90.0, 10.0))
    assert np.hypot(poses[:, 0], poses[:, 1]).max() < 4.0, poses
    assert np.abs(poses[:, 2]).max() < 0.4, poses
    assert cover.min() > 0.85


def test_score_penalises_hiding_rather_than_rewarding_it():
    """A pose that pushes the object out of frame must score WORSE, not better.

    Dividing by the surviving sample count instead of the full one makes leaving the frame a way to
    raise the score, and the optimiser finds that immediately.
    """
    H, world, image = _known_homography()
    mask = np.zeros(SHAPE, bool)
    poly = np.array([[200.0, 200.0], [900.0, 200.0], [900.0, 1000.0], [200.0, 1000.0]])
    _render_polygon(mask, pr.apply_h(H, poly), half_width=3.0)
    ctx = pg.FitContext(pg.matched_field(mask, 5.0), np.ones(SHAPE, np.float32), H, SHAPE)
    samples = pg.polygon_edge_samples(poly, spacing=6.0)
    on = ctx.score(samples)
    off = ctx.score(pg.apply_pose(samples, [9000.0, 9000.0, 0.0], poly.mean(0)))
    assert on > 0.5 and off < 0.05, (on, off)


def test_hue_windows_stay_separated():
    """Pin the colour separation the thresholds rely on.

    Red and orange tape sit close in hue, and the whole mask stage rests on a trough between them.
    A re-shoot under different lighting must fail here, loudly, rather than silently mis-mask.
    """
    red_hi = pr.TAPE_HUES["red"]["h"][1]
    orange_lo, orange_hi = pr.TAPE_HUES["orange"]["h"]
    green_lo = pr.TAPE_HUES["green"]["h"][0]
    assert red_hi < orange_lo, "red and orange windows overlap"
    assert orange_hi < green_lo, "orange and green windows overlap"
    assert orange_lo - red_hi >= 3.0, "no measured trough between red and orange"


def test_kabsch_does_not_absorb_scale():
    """Scale is metric here; a fit that rescales would hide the error being measured."""
    P = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]])
    Q = P * 1.10                                  # a pure 10% dilation
    R, t, rms = pg.kabsch_2d(P, Q)
    assert np.allclose(R @ R.T, np.eye(2), atol=1e-9)
    assert abs(np.linalg.det(R) - 1.0) < 1e-9
    assert rms > 2.0, "a dilation was silently absorbed"


def test_warp_round_trip_places_a_known_point():
    H, world, image = _known_homography()
    img = np.zeros(SHAPE, np.float32)
    px = pr.apply_h(H, np.array([[400.0, 700.0]]))[0]
    img[int(round(px[1])), int(round(px[0]))] = 1.0
    x = np.arange(300.0, 500.0, 2.0)
    y = np.arange(600.0, 800.0, 2.0)
    out = pr.warp_to_grid(img, np.linalg.inv(H), x, y, fill=0.0)
    iy, ix = np.unravel_index(int(np.argmax(out)), out.shape)
    assert abs(x[ix] - 400.0) <= 4.0 and abs(y[iy] - 700.0) <= 4.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
