"""The one damage measure: Delta = <PEEQ>_lig / eps_f.

Synthetic C3D15 meshes only -- no CalculiX needed. These pin the properties the definition was
chosen for: exactness on a uniform field, volume weighting, restriction to the ligament,
mesh-convergence (where the old whole-mesh p99 drifts), and intensivity.
"""

import numpy as np
import pytest

from nff.rve.damage import (LIG_RADIUS_FRAC, _region_average, element_volumes, ligament_elements,
                            mean_triaxiality, peak_peeq, plastic_damage, stress_triaxiality)


def unit_wedge_mesh(w_lig=4.0, n=6, thickness=0.5, half_span=None, grade=1.0):
    """A structured C3D15 wedge slab centred on the ligament disc.

    ``n`` quads per side over ``[-half_span, half_span]`` in x and the same range about the disc
    centre in y, each split into two wedges and extruded one layer through ``thickness``.
    Mid-side nodes are real edge midpoints, so the connectivity is a valid C3D15 block.

    ``grade > 1`` clusters the y-stations toward the disc centre (a graded mesh, like the real
    refinement ball) while covering the SAME domain -- so a volume integral is unchanged but any
    per-node statistic shifts.

    Returns:
        (xyz, conn) with ``conn`` 1-based, shape (2*n*n, 15).
    """
    half_span = half_span if half_span is not None else LIG_RADIUS_FRAC * w_lig
    yc = -0.5 * w_lig
    xs = np.linspace(-half_span, half_span, n + 1)
    t = np.linspace(-1.0, 1.0, n + 1)
    ys = yc + half_span * np.sign(t) * np.abs(t) ** grade

    nodes, index = [], {}

    def nid(x, y, z):                       # dedupe by rounded coordinate -> 1-based id
        key = (round(x, 9), round(y, 9), round(z, 9))
        if key not in index:
            nodes.append([x, y, z])
            index[key] = len(nodes)
        return index[key]

    def mid(a, b):
        return [(a[i] + b[i]) / 2.0 for i in range(3)]

    conn = []
    for i in range(n):
        for j in range(n):
            q = [(xs[i], ys[j]), (xs[i + 1], ys[j]), (xs[i + 1], ys[j + 1]), (xs[i], ys[j + 1])]
            for tri in ((0, 1, 2), (0, 2, 3)):
                bot = [(q[k][0], q[k][1], 0.0) for k in tri]
                top = [(q[k][0], q[k][1], thickness) for k in tri]
                corners = bot + top
                edges = [(0, 1), (1, 2), (2, 0),          # bottom triangle
                         (3, 4), (4, 5), (5, 3),          # top triangle
                         (0, 3), (1, 4), (2, 5)]          # vertical
                pts = corners + [mid(corners[a], corners[b]) for a, b in edges]
                conn.append([nid(*p) for p in pts])
    return np.array(nodes, float), np.array(conn, int)


def _nodal(xyz, fn):
    return np.array([fn(x, y, z) for x, y, z in xyz], float)


def test_uniform_field_over_the_ligament_is_exactly_peeq_over_eps_f():
    xyz, conn = unit_wedge_mesh()
    frame = {"PEEQ": np.full(len(xyz), 0.42)}
    assert plastic_damage(frame, xyz, conn, 4.0, eps_f=2.0,
                          region="ligament") == pytest.approx(0.21, rel=1e-12)

    # Whole-window numerator over the SAME (ligament) denominator scales by the volume ratio, so a
    # uniformly-yielded window reads higher than a uniformly-yielded ligament -- by construction.
    vol = element_volumes(xyz, conn)
    ratio = vol.sum() / vol[ligament_elements(xyz, conn, 4.0)].sum()
    assert plastic_damage(frame, xyz, conn, 4.0, eps_f=2.0) == pytest.approx(0.21 * ratio, rel=1e-12)


def test_absent_plastic_field_is_nan_not_zero():
    xyz, conn = unit_wedge_mesh()
    assert np.isnan(plastic_damage({"STRESS": np.zeros((len(xyz), 6))}, xyz, conn, 4.0, 1.784))
    assert np.isnan(peak_peeq({}, xyz, conn, 4.0))
    # a frame whose field length disagrees with the mesh must not be silently broadcast
    assert np.isnan(plastic_damage({"PEEQ": np.ones(3)}, xyz, conn, 4.0, 1.784))


def test_weighting_is_by_volume_not_by_element_count():
    """Big elements must count for more than small ones -- an unweighted mean would not."""
    w_lig = 4.0
    xyz, conn = unit_wedge_mesh(w_lig=w_lig, n=6, grade=2.5)   # graded => unequal element volumes
    vol = element_volumes(xyz, conn)
    mask = ligament_elements(xyz, conn, w_lig)
    assert np.all(vol > 0) and vol[mask].max() > 2.0 * vol[mask].min()

    peeq = _nodal(xyz, lambda x, y, z: 1.0 if y > -0.5 * w_lig else 0.0)
    vals = peeq[conn - 1].mean(axis=1)
    got = plastic_damage({"PEEQ": peeq}, xyz, conn, w_lig, 1.0, region="ligament")

    assert got == pytest.approx(float(np.dot(vol[mask], vals[mask]) / vol[mask].sum()), rel=1e-12)
    assert got != pytest.approx(float(vals[mask].mean()), rel=1e-6)     # NOT the unweighted mean


def test_elements_outside_the_ligament_are_ignored():
    """Extra material far from the hinge must not dilute the damage reading."""
    w_lig = 4.0
    xyz, conn = unit_wedge_mesh(w_lig=w_lig)
    peeq = np.full(len(xyz), 0.3)
    inside = plastic_damage({"PEEQ": peeq}, xyz, conn, w_lig, 1.0, region="ligament")

    far, far_conn = unit_wedge_mesh(w_lig=w_lig)          # a copy translated well clear of the disc
    far[:, 0] += 50.0
    xyz2 = np.vstack([xyz, far])
    conn2 = np.vstack([conn, far_conn + len(xyz)])
    peeq2 = np.concatenate([peeq, np.zeros(len(far))])    # pristine material out there

    assert ligament_elements(xyz2, conn2, w_lig).sum() == ligament_elements(xyz, conn, w_lig).sum()
    assert plastic_damage({"PEEQ": peeq2}, xyz2, conn2, w_lig, 1.0,
                          region="ligament") == pytest.approx(inside, rel=1e-12)


def test_smooth_plastic_band_converges_under_refinement():
    """A volume integral must stop moving as the mesh refines."""
    w_lig, band = 4.0, 0.35
    hot = lambda x, y, z: float(np.exp(-((y + 0.5 * w_lig) / band) ** 2))   # plastic band at the root

    deltas = []
    for n in (8, 16, 32, 64):
        xyz, conn = unit_wedge_mesh(w_lig=w_lig, n=n)
        deltas.append(plastic_damage({"PEEQ": _nodal(xyz, hot)}, xyz, conn, w_lig, 1.0))

    assert abs(deltas[-1] - deltas[-2]) / deltas[-1] < 0.01                 # settled to <1%
    assert deltas[-1] == pytest.approx(deltas[-2], rel=0.01)


def test_grading_the_mesh_moves_a_nodal_percentile_but_not_the_integral():
    """THE reason for a volume integral over an order statistic.

    The real RVE mesh is GRADED: fine in the ligament, coarse in the panels. A per-node percentile
    is then weighted by where the nodes happen to be, not by material -- regrade the mesh and the
    number moves, with no change in physics. A volume-weighted integral carries the element volumes
    explicitly, so it does not care. That is the difference between the old ``damage_p99`` and
    ``Delta``, on the SAME field over the SAME domain.
    """
    # a LOCALIZED hot spot at the cut tip, ~1% of the domain -- so the top percentile is decided by
    # how many nodes land inside it, which is precisely what grading changes
    w_lig, sigma = 4.0, 0.25
    yc = -0.5 * w_lig
    hot = lambda x, y, z: float(np.exp(-((x ** 2 + (y - yc) ** 2) / sigma ** 2)))

    def read(grade):
        xyz, conn = unit_wedge_mesh(w_lig=w_lig, n=80, grade=grade)
        peeq = _nodal(xyz, hot)
        return (plastic_damage({"PEEQ": peeq}, xyz, conn, w_lig, 1.0),
                float(np.percentile(peeq, 99.0)))

    d_uniform, p99_uniform = read(1.0)      # uniform stations
    d_graded, p99_graded = read(2.5)        # clustered into the plastic band, same domain

    assert d_graded == pytest.approx(d_uniform, rel=0.05)       # integral: physics-invariant
    assert abs(p99_graded - p99_uniform) / p99_uniform > 0.10    # percentile: mesh-dependent


def test_damage_is_intensive():
    """A volume AVERAGE: growing the ligament at a fixed field must not change the reading.

    The extensive alternative (total plastic work) does change -- which is why it is useless as a
    design objective.
    """
    w_lig = 4.0
    peeq_of = lambda x, y, z: 0.5
    small_xyz, small_conn = unit_wedge_mesh(w_lig=w_lig, n=8, thickness=0.5)
    big_xyz, big_conn = unit_wedge_mesh(w_lig=w_lig, n=8, thickness=1.5)     # 3x the volume

    d_small = plastic_damage({"PEEQ": _nodal(small_xyz, peeq_of)}, small_xyz, small_conn, w_lig, 1.0,
                             region="ligament")
    d_big = plastic_damage({"PEEQ": _nodal(big_xyz, peeq_of)}, big_xyz, big_conn, w_lig, 1.0,
                           region="ligament")
    assert d_big == pytest.approx(d_small, rel=1e-12)      # intensive

    total = lambda xyz, conn: float(np.dot(element_volumes(xyz, conn),
                                           _nodal(xyz, peeq_of)[conn - 1].mean(axis=1)))
    assert total(big_xyz, big_conn) == pytest.approx(3.0 * total(small_xyz, small_conn), rel=1e-9)


def test_peak_is_at_least_the_average():
    xyz, conn = unit_wedge_mesh(w_lig=4.0, n=8)
    peeq = _nodal(xyz, lambda x, y, z: float(np.exp(-((y + 2.0) / 0.4) ** 2)))
    eps_f = 1.784
    delta = plastic_damage({"PEEQ": peeq}, xyz, conn, 4.0, eps_f)
    assert peak_peeq({"PEEQ": peeq}, xyz, conn, 4.0) >= delta * eps_f


def test_mean_triaxiality_reads_uniaxial_tension_as_one_third():
    xyz, conn = unit_wedge_mesh(w_lig=4.0, n=4)
    S = np.tile([100.0, 0.0, 0.0, 0.0, 0.0, 0.0], (len(xyz), 1))    # uniaxial tension
    assert mean_triaxiality({"STRESS": S}, xyz, conn, 4.0) == pytest.approx(1.0 / 3.0, rel=1e-9)
    assert np.isnan(mean_triaxiality({}, xyz, conn, 4.0))


def test_mean_triaxiality_is_weighted_where_plasticity_actually_is():
    """A bending field cancels under a plain volume average -- so weight by plastic strain.

    Tension on one side of the ligament and compression on the other average to eta ~ 0 whatever
    the real stress state, which would audit nothing. Weighting by PEEQ reports the triaxiality of
    the material that is actually yielding.
    """
    w_lig, yc = 4.0, -2.0
    xyz, conn = unit_wedge_mesh(w_lig=w_lig, n=8)

    tensile = xyz[:, 1] > yc
    S = np.zeros((len(xyz), 6))
    S[tensile, 0] = +100.0                       # eta = +1/3
    S[~tensile, 0] = -100.0                      # eta = -1/3
    peeq = np.where(tensile, 0.5, 0.0)           # only the tensile side has yielded

    vol_weighted = _region_average(stress_triaxiality(S), xyz, conn,
                                   ligament_elements(xyz, conn, w_lig))
    plastic_weighted = mean_triaxiality({"STRESS": S, "PEEQ": peeq}, xyz, conn, w_lig,
                                        region="ligament")

    # elements straddling y = yc carry nodes of both signs, so a step field smears a little short
    # of the exact +1/3 -- the claim is that it lands on the tensile state, not that it is exact
    assert plastic_weighted == pytest.approx(1.0 / 3.0, abs=0.06)
    # the volume average all but cancels -- several times smaller, and nowhere near +1/3
    assert abs(vol_weighted) < 0.05
    assert abs(vol_weighted) < 0.2 * plastic_weighted


def test_default_region_is_the_whole_window_not_the_ligament_disc():
    """Delta sums plastic work over the whole RVE but normalises by the LIGAMENT volume.

    User-directed 2026-07-28: the window is the hinge at the local scale, so every stress in it
    counts; but dividing by the window volume would dilute Delta to ~1e-5 and kill the design-loss
    term, so the denominator stays the hinge's own size.
    """
    w_lig = 4.0
    xyz, conn = unit_wedge_mesh(w_lig=w_lig, n=6, grade=1.0)
    vol = element_volumes(xyz, conn)
    mask = ligament_elements(xyz, conn, w_lig)
    assert not mask.all(), "mesh must extend beyond the ligament for this test to mean anything"

    peeq = _nodal(xyz, lambda x, y, z: 1.0 if y > -0.5 * w_lig else 0.0)
    vals = peeq[conn - 1].mean(axis=1)

    whole = plastic_damage({"PEEQ": peeq}, xyz, conn, w_lig, 1.0)
    disc = plastic_damage({"PEEQ": peeq}, xyz, conn, w_lig, 1.0, region="ligament")
    # numerator over the whole window, denominator always the ligament volume
    assert whole == pytest.approx(float(np.dot(vol, vals) / vol[mask].sum()), rel=1e-12)
    assert whole > disc


def test_whole_window_damage_counts_plasticity_the_disc_would_miss():
    """A hot spot OUTSIDE the ligament disc must move Delta -- that is the point of the change."""
    w_lig = 4.0
    xyz, conn = unit_wedge_mesh(w_lig=w_lig, n=6, grade=1.0)
    mask = ligament_elements(xyz, conn, w_lig)
    cen = xyz[conn[:, :6] - 1].mean(axis=1)
    dist = np.hypot(cen[:, 0], cen[:, 1] + 0.5 * w_lig)
    far = cen[np.argmax(dist)]                                 # the element furthest from the disc

    cold = _nodal(xyz, lambda x, y, z: 0.0)
    hot = _nodal(xyz, lambda x, y, z: 1.0 if np.hypot(x - far[0], y - far[1]) < 0.2 * w_lig else 0.0)
    assert hot.sum() > 0 and not mask.all()

    whole = plastic_damage({"PEEQ": hot}, xyz, conn, w_lig, 1.0)
    disc = plastic_damage({"PEEQ": hot}, xyz, conn, w_lig, 1.0, region="ligament")
    assert whole > plastic_damage({"PEEQ": cold}, xyz, conn, w_lig, 1.0)
    assert whole > 10.0 * disc                                 # the disc barely sees it
