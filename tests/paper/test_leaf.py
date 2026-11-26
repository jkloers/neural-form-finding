"""The leaf element: directional, smooth at the base, pointed at the tip, exact as SVG arcs."""

import numpy as np
import pytest

from nff_paper.topology_art.leaf import (
    ElementSpec,
    Leaf,
    element_layers_fn,
    inner_leaf_fn,
    leaf_path_fn,
    leaf_polygon_fn,
    rim_fraction_fn,
    transform_leaf_fn,
)
from nff_paper.topology_art.leaf import _points_inside_fn
from tests.paper.test_crescent import svg_arc_points


@pytest.mark.parametrize("width", [0.30, 0.46, 0.62, 0.80])
def test_svg_arcs_reproduce_the_intended_leaf(width):
    """What a renderer draws must be the shape the geometry meant, sweep flags included."""
    leaf = Leaf(centre=np.array([2.0, -1.0]), angle=0.9, length=3.0, width=width)
    outline = leaf_polygon_fn(leaf, n=300)
    current = np.asarray(leaf_path_fn(leaf).start, float)
    for kind, points in leaf_path_fn(leaf).segments:
        assert kind == "A"
        (radius, _), (large_arc, sweep), end = points
        drawn = svg_arc_points(current, end, float(radius), int(large_arc), int(sweep))
        gaps = np.linalg.norm(drawn[:, None, :] - outline[None, :, :], axis=2).min(axis=1)
        assert gaps.max() < 5e-3 * leaf.length, f"SVG arc departs by {gaps.max():.3e}"
        current = np.asarray(end, float)


def test_outline_is_simple_and_spans_the_requested_length_and_width():
    shapely = pytest.importorskip("shapely.geometry")
    leaf = Leaf(centre=np.zeros(2), angle=0.0, length=2.0, width=0.46)
    outline = leaf_polygon_fn(leaf, n=400)
    assert shapely.LinearRing(outline).is_simple
    assert outline[:, 0].max() - outline[:, 0].min() == pytest.approx(2.0, rel=1e-4)
    width = outline[:, 1].max() - outline[:, 1].min()
    assert 0.46 * 2.0 <= width < 2.0  # bowed flanks widen it beyond the bare base diameter


def test_the_shape_is_directional():
    """A base and a tip, not two identical ends: that asymmetry is the point of the element."""
    leaf = Leaf(centre=np.zeros(2), angle=0.0, length=2.0, width=0.46)
    outline = leaf_polygon_fn(leaf, n=600)

    def width_at(x, tol=0.02):
        band = outline[np.abs(outline[:, 0] - x) < tol]
        return float(np.ptp(band[:, 1])) if len(band) else 0.0

    # Just inside each end: a point tapers to nothing, a round base does not.
    assert width_at(0.94) < 0.5 * width_at(-0.94), "tip is not sharper than the base"
    assert outline[np.argmax(outline[:, 0])][1] == pytest.approx(0.0, abs=1e-9)

    # The decisive property: the shape is not symmetric end to end.
    mirrored = outline * np.array([-1.0, 1.0])
    gaps = np.linalg.norm(outline[:, None, :] - mirrored[None, :, :], axis=2).min(axis=1)
    assert gaps.max() > 0.05 * leaf.length  # a symmetric element would give ~0


def _same_leaf(one, other):
    """Leaf holds its centre as an array, so dataclass equality cannot be used directly."""
    return (np.allclose(one.centre, other.centre) and one.angle == other.angle
            and one.length == other.length and one.width == other.width and one.bow == other.bow)


def _legacy_layers(leaf, *, rings, inner, duty):
    """The nesting exactly as it was written inline in the exporter, before it was shared.

    Kept verbatim so the refactor has something to be measured against; if the two ever disagree,
    a render changed silently.
    """
    layers = [(leaf, "pale")]
    for ring in range(1, max(rings, 1)):
        period, phase = divmod(ring, 2)
        rim = inner * (period + (duty if phase else 0.0))
        nested = inner_leaf_fn(leaf, rim) if inner > 0.0 else None
        if nested is None:
            break
        layers.append((nested, "dark" if phase else "pale"))
    return layers


@pytest.mark.parametrize("rings,inner,duty", [(2, 0.075, 0.5), (2, 0.14, 0.5), (1, 0.09, 0.5),
                                              (11, 0.062, 0.34), (2, 0.0, 0.5)])
def test_shared_decomposition_reproduces_the_inline_one(rings, inner, duty):
    leaf = Leaf(centre=np.array([3.0, 1.0]), angle=-0.7, length=2.4, width=0.62, bow=0.22)
    want = _legacy_layers(leaf, rings=rings, inner=inner, duty=duty)
    got = element_layers_fn(leaf, ElementSpec(rings=rings, rim=inner, duty=duty))
    assert [tag for _, tag in got] == [tag for _, tag in want]
    for (drawn, _), (expected, _) in zip(got, want):
        assert _same_leaf(drawn, expected)


@pytest.mark.parametrize("shift", [0.0, 0.3, 0.6, 0.9])
@pytest.mark.parametrize("turn", [0.0, 0.5, 1.0])
def test_a_moved_core_never_leaves_its_leaf(shift, turn):
    """The rim is the only room the core has, so the parameters are fractions of it."""
    leaf = Leaf(centre=np.array([-2.0, 0.5]), angle=1.2, length=3.0, width=0.62, bow=0.22)
    layers = element_layers_fn(leaf, ElementSpec(rim=0.11, core_shift=shift, core_turn=turn))
    assert len(layers) == 2, "a core inside the declared range must survive"
    outline = leaf_polygon_fn(layers[0][0], n=400)
    assert _points_inside_fn(outline, leaf_polygon_fn(layers[1][0], n=200)).all()


def test_a_core_pushed_past_the_rim_is_refused_rather_than_drawn_outside():
    leaf = Leaf(centre=np.zeros(2), angle=0.0, length=2.0, width=0.62, bow=0.22)
    assert inner_leaf_fn(leaf, 0.05, shift=0.4) is None  # a shift eight times the rim
    assert len(element_layers_fn(leaf, ElementSpec(rim=0.05, core_shift=4.0))) == 1


def test_an_absolute_slot_holds_one_width_across_the_size_range():
    """A rim given as a fraction turns the size field into a change of line weight."""
    slot = 0.08
    lengths = np.array([0.4, 1.0, 2.5, 4.0])  # a 10x range, as a real sheet carries
    widths = np.array([rim_fraction_fn(length, rim_abs=slot) * length for length in lengths])
    assert np.allclose(widths, slot, rtol=1e-12)

    proportional = np.array([rim_fraction_fn(length, rim_frac=0.09) * length
                             for length in lengths])
    assert proportional.max() / proportional.min() == pytest.approx(lengths.max() / lengths.min())


def test_the_slot_only_stops_scaling_when_it_would_swallow_the_leaf():
    tiny = rim_fraction_fn(0.05, rim_abs=0.12, max_frac=0.24)
    assert tiny == pytest.approx(0.24), "the clamp is what keeps a degenerate core out"


def test_inverting_swaps_every_band_and_nothing_else():
    leaf = Leaf(centre=np.zeros(2), angle=0.3, length=2.0, width=0.62, bow=0.22)
    straight = element_layers_fn(leaf, ElementSpec(rim=0.11))
    flipped = element_layers_fn(leaf, ElementSpec(rim=0.11, invert=True))
    assert [tag for _, tag in flipped] == ["dark", "pale"]
    assert all(_same_leaf(one, other)
               for (one, _), (other, _) in zip(flipped, straight))


def test_similarity_keeps_the_shape_exactly():
    base = Leaf(centre=np.array([1.0, 2.0]), angle=0.3, length=1.4)
    moved = transform_leaf_fn(base, scale=2.5, offset=np.array([-4.0, 7.0]))
    assert [k for k, _ in leaf_path_fn(moved).segments] == ["A", "A", "A"]
    assert moved.bow == base.bow
    got = leaf_polygon_fn(moved, n=48)
    want = 2.5 * leaf_polygon_fn(base, n=48) + np.array([-4.0, 7.0])
    assert np.abs(got - want).max() < 1e-12
