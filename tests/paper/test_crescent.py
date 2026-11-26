"""The katagami element: exact arcs, and the SVG spec's own reading of them."""

import numpy as np
import pytest

from nff_paper.topology_art.crescent import (
    Crescent,
    crescent_area_fn,
    crescent_path_fn,
    crescent_polygon_fn,
    transform_crescent_fn,
)
from nff_paper.topology_art.svg_export import path_data_fn


def svg_arc_points(start, end, radius, large_arc, sweep, n=200):
    """Sample an SVG elliptical arc using the spec's endpoint-to-centre parameterisation.

    Written out rather than assumed: the sweep flag is the one place a plausible-looking element can
    silently render as its own mirror image, and only the spec's own conversion settles it.
    """
    start, end = np.asarray(start, float), np.asarray(end, float)
    half = 0.5 * (start - end)
    scale = (half[0] ** 2 + half[1] ** 2) / radius**2
    if scale > 1.0:
        radius *= np.sqrt(scale)
    numerator = radius**2 * radius**2 - radius**2 * half[1] ** 2 - radius**2 * half[0] ** 2
    denominator = radius**2 * half[1] ** 2 + radius**2 * half[0] ** 2
    factor = np.sqrt(max(numerator / denominator, 0.0))
    if large_arc == sweep:
        factor = -factor
    centre_prime = factor * np.array([radius * half[1] / radius, -radius * half[0] / radius])
    centre = centre_prime + 0.5 * (start + end)

    def angle_of(point):
        return np.arctan2(point[1] - centre[1], point[0] - centre[0])

    theta0, theta1 = angle_of(start), angle_of(end)
    delta = (theta1 - theta0) % (2.0 * np.pi)
    if not sweep:
        delta -= 2.0 * np.pi
    sweep_angles = theta0 + np.linspace(0.0, delta, n)
    return centre + radius * np.stack([np.cos(sweep_angles), np.sin(sweep_angles)], axis=1)


@pytest.mark.parametrize("bulges", [(0.62, 0.30), (0.40, 0.12), (0.75, 0.55), (0.30, 0.05)])
def test_svg_arcs_reproduce_the_intended_crescent(bulges):
    """What a renderer draws must be what the geometry meant, sweep flag included."""
    crescent = Crescent(centre=np.array([3.0, -2.0]), angle=0.7, length=2.5,
                        bulge_outer=bulges[0], bulge_inner=bulges[1])
    path = crescent_path_fn(crescent)
    outline = crescent_polygon_fn(crescent, n=200)

    current = np.asarray(path.start, float)
    for (_, points) in path.segments:
        (radius, _), (large_arc, sweep), end = points
        drawn = svg_arc_points(current, end, float(radius), int(large_arc), int(sweep))
        # Every point a renderer would draw must lie on the shape we sampled independently.
        gaps = np.linalg.norm(drawn[:, None, :] - outline[None, :, :], axis=2).min(axis=1)
        assert gaps.max() < 5e-3 * crescent.length, f"SVG arc departs by {gaps.max():.3e}"
        current = np.asarray(end, float)


def test_outline_is_simple_and_matches_the_closed_form_area():
    shapely = pytest.importorskip("shapely.geometry")
    crescent = Crescent(centre=np.zeros(2), angle=0.0, length=1.0)
    ring = shapely.LinearRing(crescent_polygon_fn(crescent, n=800))
    assert ring.is_simple
    assert shapely.Polygon(ring).area == pytest.approx(crescent_area_fn(crescent), rel=2e-4)


def test_waist_thickness_is_the_bulge_difference():
    """The two arcs are measured to the same side, so their difference is the waist."""
    crescent = Crescent(centre=np.zeros(2), angle=0.0, length=1.0,
                        bulge_outer=0.62, bulge_inner=0.30)
    outline = crescent_polygon_fn(crescent, n=2001)
    on_axis = outline[np.abs(outline[:, 0]) < 1e-3]
    assert on_axis[:, 1].max() - on_axis[:, 1].min() == pytest.approx(0.32, abs=2e-3)


def test_similarity_keeps_the_shape_exactly():
    """Only translate, rotate and uniform scale are ever applied, so arcs stay arcs."""
    base = Crescent(centre=np.array([1.0, 2.0]), angle=0.3, length=1.4)
    moved = transform_crescent_fn(base, scale=2.5, offset=np.array([-4.0, 7.0]))
    assert [k for k, _ in crescent_path_fn(moved).segments] == ["A", "A"]
    assert crescent_area_fn(moved) == pytest.approx(2.5**2 * crescent_area_fn(base), rel=1e-12)

    got = crescent_polygon_fn(moved, n=64)
    want = 2.5 * crescent_polygon_fn(base, n=64) + np.array([-4.0, 7.0])
    assert np.abs(got - want).max() < 1e-12


def test_path_data_emits_arc_commands():
    data = path_data_fn(crescent_path_fn(Crescent(centre=np.zeros(2), angle=0.0, length=1.0)))
    assert data.startswith("M") and data.count("A") == 2 and data.endswith("Z")
