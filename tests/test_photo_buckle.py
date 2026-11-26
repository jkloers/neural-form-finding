"""The strip reader and the out-of-plane observables it feeds.

The synthetic cases pin the two things a real photograph cannot check: that the width is measured
PERPENDICULAR to the strip rather than down an image column, and that a strip lying flat reads as
flat however it is oriented in the frame.
"""

import numpy as np
import pytest

from nff.calibration import photo_buckle as pb


def synth_strip(width_px=40, length_px=600, angle_deg=0.0, bow_px=0.0, shape=(700, 900)):
    """A straight-or-bowed strip of CONSTANT true width, drawn at an arbitrary image angle."""
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W]
    t = np.radians(angle_deg)
    cx, cy = W / 2.0, H / 2.0
    u = (xx - cx) * np.cos(t) + (yy - cy) * np.sin(t)          # along the strip
    v = -(xx - cx) * np.sin(t) + (yy - cy) * np.cos(t)         # across it
    centre = bow_px * np.cos(np.pi * np.clip(u / length_px, -0.5, 0.5))
    return (np.abs(u) <= length_px / 2) & (np.abs(v - centre) <= width_px / 2)


@pytest.mark.parametrize("angle", [0.0, 20.0, 45.0, -35.0])
def test_width_is_perpendicular_not_columnwise(angle):
    """A rotated strip of constant width must read the SAME width at every angle.

    A column scan would return ``width / cos(angle)`` and so inflate by 41% at 45 deg, which is the
    bug this measurement existed to avoid.

    ⚠ The tolerance absorbs a discretisation bias of 0 to 2 px: the distance transform counts the
    boundary pixel, and how much of it lands inside depends on how the band sits on the grid, so a
    40 px band reads 42 axis-aligned and 40 at -35 deg. It is an OFFSET, not a scale error, so it
    cancels in the baseline-to-width RATIO that :func:`face_turn` uses, and enters
    :func:`scale_from_width` as at most 2 px on the measured width. On the real strip, about 160 px
    wide at full resolution, that is at most 1.3% on every millimetre quoted, far inside the
    elevation band.
    """
    rib = pb.ribbon_from_mask(synth_strip(width_px=40, angle_deg=angle))
    assert abs(np.median(rib.width_px) - 41.0) <= 1.5


def test_flat_strip_reads_no_turn_and_no_bow():
    rib = pb.ribbon_from_mask(synth_strip(angle_deg=15.0))
    base = pb.flat_width_baseline(rib)
    assert np.degrees(pb.face_turn(rib, base)).max() < 6.0
    _, peak = pb.chord_bow(rib)
    assert peak < 3.0


def test_chord_bow_tracks_a_known_bow():
    rib = pb.ribbon_from_mask(synth_strip(angle_deg=10.0, bow_px=50.0))
    _, peak = pb.chord_bow(rib)
    assert 40.0 < peak < 60.0


def test_centreline_survives_a_one_pixel_dilation():
    from scipy import ndimage as ndi
    m = synth_strip(angle_deg=25.0, bow_px=30.0)
    a = pb.ribbon_from_mask(m)
    b = pb.ribbon_from_mask(ndi.binary_dilation(m, np.ones((3, 3), bool)))
    assert abs(a.length_px - b.length_px) / a.length_px < 0.05
    assert abs(pb.chord_bow(a)[1] - pb.chord_bow(b)[1]) < 4.0


def test_holes_are_filled_before_the_ridge_is_taken():
    """Hardware threaded through the ligament punches a hole exactly where the crest is."""
    m = synth_strip(width_px=60, angle_deg=0.0)
    H, W = m.shape
    yy, xx = np.mgrid[0:H, 0:W]
    punched = m & ~((xx - W / 2) ** 2 + (yy - H / 2) ** 2 < 12 ** 2)
    rib = pb.ribbon_from_mask(pb.largest_component(punched))
    assert abs(np.median(rib.width_px) - 60.0) < 3.0


def test_scale_from_width_inverts():
    rib = pb.ribbon_from_mask(synth_strip(width_px=50, angle_deg=8.0))
    mm_per_px = pb.scale_from_width(rib, 22.334)
    assert abs(mm_per_px * 50.0 - 22.334) < 1.0


def test_amplitude_band_brackets_and_orders_with_elevation():
    rib = pb.ribbon_from_mask(synth_strip(angle_deg=12.0, bow_px=40.0))
    base = pb.flat_width_baseline(rib)
    est = pb.amplitude_estimate(rib, base, 0.25, camera_elev_deg=45.0, elev_band_deg=15.0)
    assert est["lo_mm"] <= est["bow_mm"] <= est["hi_mm"]
    # a steeper view foreshortens the vertical harder, so the same bow implies a taller buckle
    assert est["hi_mm"] > est["lo_mm"]


def test_face_turn_is_not_integrated_into_a_height():
    """Regression: the turn is a twist about the strip's own tangent, never a slope.

    Integrating ``sin(turn)`` along the strip returned 173 mm on a strip whose bow said 25 mm.
    ``amplitude_estimate`` must therefore report the bow alone.
    """
    rib = pb.ribbon_from_mask(synth_strip(angle_deg=0.0, bow_px=40.0))
    base = pb.flat_width_baseline(rib)
    est = pb.amplitude_estimate(rib, base, 0.25)
    assert "turn_mm" not in est
    assert est["bow_mm"] < 5.0 * 40.0 * 0.25


def test_buckle_law_shape():
    assert pb.buckle_law(0.0) == pytest.approx(0.0)
    assert pb.buckle_law(90.0) == pytest.approx(25.7)
    assert pb.buckle_law(16.58) == pytest.approx(13.73, abs=0.05)


def test_band_indicator_selects_material_below_the_cut():
    xyz = np.array([[0, 0.0, 0], [0, -5.0, 0], [0, -40.0, 0], [0, 1.0, 0]])
    ind = pb.band_indicator(xyz, half_width_mm=11.17)
    assert list(ind) == [1.0, 1.0, 0.0, 0.0]


def test_height_profile_reads_the_cut_edge_not_the_band_mean():
    """The tape's centreline sits on the cut, and that edge carries the largest amplitude."""
    n = 4000
    rng = np.random.default_rng(0)
    x = rng.uniform(-100, 100, n)
    y = rng.uniform(-22.33, 0.0, n)
    z = np.full(n, 1.0)
    xyz = np.column_stack([x, y, z])
    # a dome that decays away from the cut, so edge and band-mean genuinely differ
    uz = 20.0 * np.exp(-(x / 40.0) ** 2) * (1.0 + y / 22.33)
    disp = np.column_stack([np.zeros(n), np.zeros(n), uz])
    _, edge = pb.height_profile_3d(xyz, disp, half_width_mm=11.17, thickness_mm=0.5,
                                   edge_frac=0.25)
    _, whole = pb.height_profile_3d(xyz, disp, half_width_mm=11.17, thickness_mm=0.5,
                                    edge_frac=1.0)
    assert edge.max() > whole.max()


# ── the relief hole as a shared fiducial ──────────────────────────────────────────────────────

def _circle3d(n=400, radius=3.575):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(t), radius * np.sin(t), np.zeros(n)])


@pytest.mark.parametrize("tilt", [0.0, 20.0, 40.0, 60.0, 75.0])
def test_projected_circle_ratio_is_the_cosine_of_its_tilt(tilt):
    """A circle viewed at angle psi from its normal images with minor/major = cos(psi).

    This is the whole basis of the camera fit, so it is pinned end to end: build a circle, view it
    orthographically from a known direction, and check the fitted ellipse reports that direction.
    """
    C = _circle3d()
    psi = np.radians(tilt)
    d = np.array([np.sin(psi), 0.0, np.cos(psi)])          # view direction, circle normal is +z
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, d)
    right /= np.linalg.norm(right)
    up2 = np.cross(d, right)
    e = pb.fit_ellipse(np.column_stack([C @ right, C @ up2]))
    assert abs(e.tilt_deg - tilt) < 2.0
    assert e.major == pytest.approx(2 * 3.575, rel=1e-3)


def test_major_axis_is_not_foreshortened_by_tilt():
    """The scale must come from the MAJOR axis: it is the one diameter tilt leaves alone."""
    C = _circle3d()
    majors = []
    for tilt in (0.0, 30.0, 60.0):
        psi = np.radians(tilt)
        d = np.array([np.sin(psi), 0.0, np.cos(psi)])
        right = np.cross([0.0, 1.0, 0.0], d)
        right /= np.linalg.norm(right)
        majors.append(pb.fit_ellipse(np.column_stack([C @ right, C @ np.cross(d, right)])).major)
    assert max(majors) - min(majors) < 1e-2 * majors[0]
    assert pb.Ellipse(0, 0, majors[0], majors[0], 0.0).mm_per_px(7.15) == \
        pytest.approx(7.15 / majors[0])


def test_rotation_invariant_residuals_cannot_see_a_mirror():
    """Why the camera fit uses ABSOLUTE bearings.

    Reflecting a view preserves both the ellipse's axis ratio and its angle to any other feature,
    so a cost built only from those is blind to a mirrored camera -- which is exactly the render
    that came out of the first attempt.
    """
    C = _circle3d()
    psi = np.radians(45.0)
    d = np.array([np.sin(psi), 0.0, np.cos(psi)])
    right = np.cross([0.0, 1.0, 0.0], d)
    right /= np.linalg.norm(right)
    uv = np.column_stack([C @ right, C @ np.cross(d, right)])
    # Rotate into an OBLIQUE frame first. With the major axis lying on the mirror axis the
    # reflection is the identity, and the test would pass while proving nothing.
    t = np.radians(30.0)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    uv = uv @ R.T
    e, e_mirror = pb.fit_ellipse(uv), pb.fit_ellipse(uv * [1, -1])
    assert e.ratio == pytest.approx(e_mirror.ratio, abs=1e-6)          # invariant: blind
    d_ang = abs(np.degrees(np.mod(e.angle - e_mirror.angle + np.pi / 2, np.pi) - np.pi / 2))
    assert d_ang > 20.0                                                # bearing: not blind


def test_fit_hole_recovers_a_synthetic_disc():
    """The ray walker plus the conic fit, on a drawn ellipse with a bite taken out of it."""
    H = W = 400
    yy, xx = np.mgrid[0:H, 0:W]
    a, b, ang = 70.0, 42.0, np.radians(25.0)
    u = (xx - 200) * np.cos(ang) + (yy - 190) * np.sin(ang)
    v = -(xx - 200) * np.sin(ang) + (yy - 190) * np.cos(ang)
    disc = (u / a) ** 2 + (v / b) ** 2 <= 1.0
    disc &= ~((xx > 235) & (yy < 150))                    # an occluder over part of the rim
    img = np.where(disc, 0.15, 0.75)
    e = pb.fit_hole(img, (203, 193), 60.0)
    assert abs(e.major - 2 * a) < 7.0
    assert abs(e.ratio - b / a) < 0.06
