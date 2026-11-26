"""The clipped-ink guard. Nothing in this house crops to content, so a gutter is right or it is not."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nff.utils import figstyle as FST                                  # noqa: E402


def _figure(left, dpi=100.0, label="departure from\nideal 1-DOF  (%)"):
    FST.use_agg()
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=FST.figsize_in(*FST.page_bp(1.0, aspect=1.9)), dpi=dpi)
    ax = fig.add_axes([left, 0.2, 0.6, 0.65])          # never flush with an edge, so 0 is unambiguous
    ax.set_yscale("log")
    ax.plot([1, 10], [1e-3, 1e2])
    ax.set_ylabel(label)
    return fig, ax


def test_a_generous_gutter_reports_everything_inside():
    fig, _ = _figure(0.30)
    worst, edge, offenders = FST.audit_margins(fig)
    assert worst < 0.0, f"nothing should overflow, yet the {edge} edge reports {worst:+.2f} bp"
    assert offenders == []


def test_a_starved_gutter_is_an_error_naming_the_edge():
    fig, _ = _figure(0.005)
    with pytest.raises(ValueError, match="outside the page"):
        FST.audit_margins(fig)


def test_the_overflow_is_reported_in_points_not_pixels():
    """The trap: ``Figure.get_tightbbox`` returns inches but ``Artist.get_tightbbox`` returns display
    pixels. Mixing them scales every number by dpi/72, so the same figure must measure the same at two
    dpi settings."""
    a = FST.audit_margins(_figure(0.30, dpi=100.0)[0])[0]
    b = FST.audit_margins(_figure(0.30, dpi=200.0)[0])[0]
    assert abs(a - b) < 1.0, f"the measurement moved with dpi: {a:.2f} bp at 100, {b:.2f} bp at 200"


def test_an_artist_taken_out_of_layout_is_not_an_offender():
    """``set_in_layout(False)`` is matplotlib's own escape hatch and must be honoured."""
    fig, ax = _figure(0.30)
    bad = fig.text(-0.4, 0.5, "deliberately outside the page")
    with pytest.raises(ValueError, match="outside the page"):
        FST.audit_margins(fig)
    bad.set_in_layout(False)
    assert FST.audit_margins(fig)[0] < 0.0


def test_the_tolerance_forgives_a_sub_point_clip_and_no_more():
    fig, _ = _figure(0.005)
    with pytest.raises(ValueError):
        FST.audit_margins(fig, pad_bp=0.25)
    worst = FST.audit_margins(fig, pad_bp=1e6)[0]
    assert worst > 0.25, "this figure is meant to overflow; the tolerance is what excuses it"
