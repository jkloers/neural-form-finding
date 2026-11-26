"""The training film's trajectory maths and panel geometry.

Everything here is pure numpy / matplotlib: no run folder, no physics, no ffmpeg. The properties
under test are the ones a bad frame would not announce -- an interpolated design that drifts off the
recorded one, and two panels whose boxes stop sharing an edge.
"""

import numpy as np
import pytest

from nff.closed.training_film import (
    _pchip, _panel_rects, _camera, _loss_series, _loss_limits,
    resample_trajectory, FIG_W_IN, FIG_H_IN,
)


def _traj(n_snap=12, seed=0):
    rng = np.random.default_rng(seed)
    epochs = np.arange(n_snap)
    return epochs, {"z": rng.normal(size=(n_snap, 4, 4)),
                    "bnd_logits": rng.normal(size=(n_snap, 7)),
                    "w_lig_logit": rng.normal(size=(n_snap, 9))}


def test_pchip_is_exact_at_the_recorded_samples():
    """A frame that lands on a recorded epoch must BE that epoch, not an approximation of it.

    This is what lets the film claim every frame is a design the optimizer could have produced: the
    interpolant passes through the samples, so the recorded designs survive resampling untouched.
    """
    epochs, leaves = _traj()
    for name, stack in leaves.items():
        got = _pchip(epochs.astype(float), stack, epochs.astype(float))
        assert np.allclose(got, stack, atol=0, rtol=0), name


def test_pchip_does_not_overshoot_a_monotone_leaf():
    """A converging logit must not swing past where the optimizer put it.

    A natural cubic spline would; PCHIP is used precisely because it will not, and an overshoot
    would show on screen as the cut pattern wobbling backwards near convergence.
    """
    epochs = np.arange(8).astype(float)
    stack = np.exp(-epochs).reshape(-1, 1)          # monotone decreasing
    dense = np.linspace(0, 7, 400)
    got = _pchip(epochs, stack, dense).ravel()
    assert got.max() <= stack.max() + 1e-12
    assert got.min() >= stack.min() - 1e-12
    assert np.all(np.diff(got) <= 1e-12)            # still monotone


def test_pchip_clamps_outside_the_recorded_range():
    """Hold frames sit exactly on the endpoints rather than extrapolating into invented designs."""
    epochs = np.arange(5).astype(float)
    stack = np.linspace(0, 1, 5).reshape(-1, 1)
    got = _pchip(epochs, stack, np.array([-3.0, 0.0, 4.0, 9.0])).ravel()
    assert got[0] == pytest.approx(stack[0, 0])
    assert got[-1] == pytest.approx(stack[-1, 0])


def test_resample_holds_then_sweeps_the_whole_range():
    epochs, leaves = _traj()
    n_frames, hold_start, hold_end = 200, 10, 20
    t, frames = resample_trajectory(epochs, leaves, n_frames,
                                    hold_start=hold_start, hold_end=hold_end)
    assert len(t) == n_frames
    assert np.all(t[:hold_start] == epochs[0])
    assert np.all(t[-hold_end:] == epochs[-1])
    assert t[hold_start] == pytest.approx(float(epochs[0]))
    assert t[-hold_end - 1] == pytest.approx(float(epochs[-1]))
    assert np.all(np.diff(t) >= -1e-12)             # never runs backwards
    for name, stack in leaves.items():
        assert frames[name].shape == (n_frames,) + stack.shape[1:], name


def test_resample_survives_a_single_snapshot_run():
    """A run killed before its second checkpoint still has to render rather than raise."""
    epochs = np.array([0])
    leaves = {"z": np.zeros((1, 3, 3))}
    t, frames = resample_trajectory(epochs, leaves, 30)
    assert len(t) == 30
    assert frames["z"].shape == (30, 3, 3)


def test_panels_share_their_top_and_bottom_edges():
    """The alignment requirement, asserted on the rectangles rather than on a rendered frame."""
    for sheet_aspect in (0.5, 1.0, 2.0):
        rect_geom, rect_loss = _panel_rects(sheet_aspect)
        assert rect_geom[1] == pytest.approx(rect_loss[1], abs=1e-12)
        assert rect_geom[3] == pytest.approx(rect_loss[3], abs=1e-12)
        assert rect_loss[0] > rect_geom[0] + rect_geom[2]      # loss sits clear to the right
        assert rect_loss[2] > 0


def test_camera_matches_the_axes_box_aspect():
    """Equal data units without a set_aspect call -- the call that would resize the box.

    If the window's aspect equals the box's, a millimetre is the same length on both axes, which is
    what set_aspect('equal') would otherwise have to enforce by shrinking the box.
    """
    rect_geom, _ = _panel_rects(0.5)
    box_aspect = rect_geom[2] * FIG_W_IN / (rect_geom[3] * FIG_H_IN)
    x0, x1, y0, y1 = _camera((0.0, 0.0, 2844.8, 5689.6), box_aspect)
    assert (x1 - x0) / (y1 - y0) == pytest.approx(box_aspect, rel=1e-12)
    assert x0 < 0.0 and x1 > 2844.8 and y0 < 0.0 and y1 > 5689.6   # content enclosed


def test_camera_left_anchor_keeps_content_against_the_left_edge():
    x0, x1, y0, y1 = _camera((0.0, 0.0, 100.0, 400.0), 2.0, anchor="left")
    assert (x1 - x0) / (y1 - y0) == pytest.approx(2.0, rel=1e-12)
    assert x0 < 0.0 and x0 > -50.0        # padding only, not half the slack


def test_loss_series_keeps_only_the_terms_a_run_carried():
    """A zero-weighted term emits no key at all, and must not become a flat row under the total."""
    history = [{"comp_chamfer": 3.0, "comp_damage": 1.0, "total": 4.0},
               {"comp_chamfer": 2.0, "comp_damage": 0.5, "total": 2.5}]
    epochs, active, total = _loss_series(history)
    labels = [lbl for lbl, _, _ in active]
    assert labels == ["Shape (Chamfer)", "Hinge damage"]
    assert "Compression" not in labels
    assert np.allclose(total, [4.0, 2.5])
    assert np.allclose(epochs, [0.0, 1.0])


def test_loss_series_drops_a_present_but_identically_zero_term():
    history = [{"comp_chamfer": 3.0, "comp_ood": 0.0, "total": 3.0}] * 3
    _, active, _ = _loss_series(history)
    assert [lbl for lbl, _, _ in active] == ["Shape (Chamfer)"]


def test_loss_limits_are_not_set_by_the_untrained_spike():
    """An opening epoch orders of magnitude high must not flatten the rest of the film."""
    n = 120
    total = np.concatenate([[5000.0], np.full(n - 1, 10.0)])
    history = [{"comp_chamfer": t, "total": t} for t in total]
    epochs, active, tot = _loss_series(history)
    (_, _), (y_min, y_max) = _loss_limits(epochs, active, tot)
    assert y_max < 100.0            # the spike is clipped away, not accommodated
    assert y_min <= 0.0


def test_loss_limits_leave_room_past_the_last_epoch():
    """The playhead dot rides the total; at the final frame it must sit inside the panel."""
    history = [{"comp_chamfer": 1.0, "total": 1.0}] * 50
    epochs, active, total = _loss_series(history)
    (x0, x1), _ = _loss_limits(epochs, active, total)
    assert x0 == 0.0
    assert x1 > float(epochs[-1])
