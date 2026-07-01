"""Tests for the parametric single-hinge RVE geometry (nff.rve.geometry).

Run: JAX_PLATFORMS=cpu conda run -n kgnn_mac python -m pytest tests/test_rve_geometry.py -q
"""
import numpy as np
import pytest
from shapely.geometry import Point

from nff.rve.geometry import (
    RVEParams, build_rve_domain, classify_boundary, ligament_present,
)


def test_domain_is_single_positive_polygon():
    dom = build_rve_domain(RVEParams())
    assert dom.geom_type == "Polygon"
    assert dom.is_valid
    assert dom.area > 0.0
    # area is close to the Saint-Venant half-disk minus a thin slit
    p = RVEParams()
    half_disk = 0.5 * np.pi * p.r_win ** 2
    assert dom.area < half_disk                                    # less than the full half-disk
    assert dom.area > 0.9 * half_disk                              # slit removes only a little


def test_ligament_present_when_retracted():
    assert ligament_present(build_rve_domain(RVEParams(w_lig=8.0)), RVEParams(w_lig=8.0))
    # a cut that reaches the secondary edge (no retraction) leaves no ligament there
    p0 = RVEParams(w_lig=0.0, rho=0.0)
    assert not ligament_present(build_rve_domain(p0), p0)


def test_boundary_tags_nonempty_and_cover_exterior():
    p = RVEParams()
    dom = build_rve_domain(p)
    tags = classify_boundary(dom, p)
    for k in ("rigid_A", "rigid_B", "free"):
        assert len(tags[k]) > 0, k
    tagged = sum(np.hypot(b[0]-a[0], b[1]-a[1]) for segs in tags.values() for a, b in segs)
    assert tagged == pytest.approx(dom.exterior.length, rel=1e-9)   # every edge tagged once


def test_rigid_handles_on_opposite_sides():
    p = RVEParams()
    tags = classify_boundary(build_rve_domain(p), p)
    xa = np.mean([0.5*(a[0]+b[0]) for a, b in tags["rigid_A"]])
    xb = np.mean([0.5*(a[0]+b[0]) for a, b in tags["rigid_B"]])
    assert xa < 0 < xb                                              # A left, B right


def test_free_boundary_includes_top_and_slit():
    p = RVEParams()
    tags = classify_boundary(build_rve_domain(p), p)
    ys = [0.5*(a[1]+b[1]) for a, b in tags["free"]]
    assert max(ys) == pytest.approx(0.0, abs=1e-6)                 # secondary cut at the top
    assert min(ys) < -p.w_lig                                      # slit runs below the tip


@pytest.mark.parametrize("alpha", [78.0, 90.0, 102.0])
def test_cut_angle_tilts_slit(alpha):
    p = RVEParams(alpha_deg=alpha)
    dom = build_rve_domain(p)
    assert ligament_present(dom, p)
    um = p.main_cut_dir()
    # a point along the main-cut axis, below the tip, is a void (outside the domain)
    q = np.array([0.0, -p.w_lig]) + 3.0 * p.w_lig * um
    assert not dom.contains(Point(q[0], q[1]))
