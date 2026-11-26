"""Presentation film: the flat cut pattern morphing across training, beside the full loss.

The artefact this replaces is ``training_evolution.gif``, a byproduct of the run driver: six frames
a second, no loss, and drawn through ``plot_tessellation``, which calls ``plt.style.use('default')``
and so ignores the house charter entirely. This module authors the film deliberately instead.

Three decisions carry the result.

**The design is interpolated, not the vertices.** A run records a few hundred epochs; a fluid clip
needs a couple of thousand frames. Blending vertex positions between two recorded designs would put
frames on the screen that no design produces, and a blend of two valid Tutte embeddings is not
itself guaranteed valid. So the interpolation happens in LOGIT space -- on ``z``, ``bnd_logits`` and
``w_lig_logit`` -- and every interpolated design is pushed back through the same collinearity solve
the optimizer used. Every frame is therefore an exact, realisable cut pattern.

**The panels are aligned by construction, not by adjustment.** ``set_aspect('equal')`` shrinks an
axes' BOX to fit its data, which silently breaks a shared top and bottom edge with the panel beside
it. Here the geometry panel's rectangle is sized from the sheet's own aspect and the data window is
padded to match it exactly, so equal axis scaling is already true and both panels span the same
figure-coordinate band.

**The ligaments are real.** ``learn_w_lig`` makes the per-hinge ligament width a design variable, so
the film draws the true manufacturing geometry -- kerf slots, fillets, ligament bridges -- from
:func:`nff.closed.cut_export.build_run_cut_geometry` at each frame's own ``w_lig``. The ligaments
visibly thicken and thin as the optimizer trades shape against damage, which a schematic hides.
"""

import os
import json
import shutil

import numpy as np

from nff.utils.figstyle import ORANGE, TEAL, RED, GREY, INK, PURPLE, GREEN


# ── The loss stack, in charter colours. ────────────────────────────────────────────────────────
# The colour of a term is the colour that term's SUBJECT already has everywhere else in the paper:
# orange is the sheet (so, shape), red is failure, teal is the hinge, grey is secondary annotation.
# Order is bottom-of-stack first; a term absent from a run's history is dropped silently, since the
# loss registry emits keys only for terms with a non-zero weight.
LOSS_TERMS = (
    ("comp_chamfer",        "Shape (Chamfer)",  ORANGE),
    ("comp_damage",         "Hinge damage",     RED),
    ("comp_compression",    "Compression",      TEAL),
    ("comp_shear_path",     "Shear path",       GREEN),
    ("comp_ood",            "Out of domain",    PURPLE),
    ("comp_hinge_gap",      "Hinge gap",        GREY),
    ("comp_regularization", "Regularisation",   GREY),
)

FIG_W_IN, FIG_H_IN = 12.8, 7.2          # x dpi 150 = exactly 1920 x 1080

# A film is watched at the full frame, not shrunk onto a slide, so `slide_bp`'s contract is the
# wrong one here: it sizes type for a 144 mm-wide projection and lands at ~38 px a line, which
# swamps the frame. Size against the pixel grid instead -- at 150 dpi a point is 2.08 px, so 13 pt
# is a ~27 px line, the comfortable reading weight for 1080p.
FILM_BODY_PT = 13.0
FILM_SCALE = 8.0 / FILM_BODY_PT         # 8.0 is figstyle's FS_BASE

# A hinge zoom was tried in a column beside the sheet and cut: at any window wide enough to hold
# the ligament, the fillet that terminates each cut reads as a black blob rather than as a hinge,
# and the panel confused more than it showed. The ligament width is reported as a number instead,
# and the width it would have taken goes to the loss curve, which has 200 epochs to develop.
GEOM_PANEL_WIDEN = 1.0


def load_trajectory(run_dir):
    """Read a run's persisted design trajectory and its per-epoch metrics.

    Args:
        run_dir: a run folder holding ``design_traj.npz`` and ``loss_history.json``.

    Returns:
        ``(epochs, leaves, history)`` -- ``epochs`` is (n_snap,) int, ``leaves`` maps each design
        leaf name to an (n_snap, ...) array, and ``history`` is one metrics dict per epoch.

    Raises:
        FileNotFoundError: if the run predates trajectory persistence, in which case only
        ``best_params.pkl`` survives and there is nothing to animate.
    """
    traj_path = os.path.join(run_dir, "design_traj.npz")
    if not os.path.exists(traj_path):
        raise FileNotFoundError(
            f"{traj_path} not found. The film replays the design trajectory, which only runs made "
            f"with --traj-every persist; a run that wrote just best_params.pkl cannot be animated.")
    with np.load(traj_path) as npz:
        epochs = np.asarray(npz["epochs"], dtype=int)
        leaves = {k: np.asarray(npz[k]) for k in npz.files if k != "epochs"}

    hist_path = os.path.join(run_dir, "loss_history.json")
    history = []
    if os.path.exists(hist_path):
        with open(hist_path) as fh:
            history = json.load(fh)
    return epochs, leaves, history


def _pchip(x, y_stack, x_query):
    """Monotone cubic interpolation of a stack of samples along axis 0.

    PCHIP rather than a natural spline because a natural spline overshoots between samples: on a
    logit that is converging it would swing the design past where the optimizer ever put it, and the
    overshoot shows up as a visible wobble in the cut pattern.
    """
    from scipy.interpolate import PchipInterpolator
    flat = y_stack.reshape(len(x), -1)
    if len(x) < 2:
        out = np.repeat(flat, len(x_query), axis=0)
    else:
        out = PchipInterpolator(x, flat, axis=0, extrapolate=False)(x_query)
        out = np.nan_to_num(out, nan=0.0)
        out[x_query <= x[0]] = flat[0]
        out[x_query >= x[-1]] = flat[-1]
    return out.reshape((len(x_query),) + y_stack.shape[1:])


def resample_trajectory(epochs, leaves, n_frames, *, ease="smoothstep",
                        hold_start=0, hold_end=0):
    """Resample a recorded trajectory onto a dense, eased frame schedule.

    Args:
        epochs: (n_snap,) recorded epoch numbers.
        leaves: design leaf name -> (n_snap, ...) array.
        n_frames: total frames including holds.
        ease: ``smoothstep`` eases the sweep at both ends so the film settles instead of stopping
            dead; ``linear`` advances one epoch's worth per frame throughout.
        hold_start: frames held on the untrained design before the sweep.
        hold_end: frames held on the final design after it.

    Returns:
        ``(t_epoch, frames)`` -- ``t_epoch`` is the (n_frames,) fractional epoch each frame sits at,
        and ``frames`` maps each leaf name to an (n_frames, ...) interpolated array.
    """
    from nff_paper.closed_support.shield_render import ease_fractions

    n_sweep = max(1, int(n_frames) - int(hold_start) - int(hold_end))
    frac = ease_fractions(n_sweep, ease=ease)
    t_sweep = epochs[0] + frac * (epochs[-1] - epochs[0])
    t_epoch = np.concatenate([np.full(hold_start, float(epochs[0])),
                              t_sweep,
                              np.full(hold_end, float(epochs[-1]))])
    return t_epoch, {k: _pchip(epochs.astype(float), v, t_epoch) for k, v in leaves.items()}


def _panel_rects(sheet_aspect, *, left=0.028, right=0.988, bottom=0.175, top=0.880, gap=0.075):
    """Two axes rectangles that share their top and bottom edges exactly.

    The geometry panel's width comes from the sheet's own aspect at the band's height, so equal
    axis scaling is already true of the box and no ``set_aspect`` call is needed -- which is the
    whole point, since ``set_aspect`` resizes the box and would break the shared edge with the loss
    panel beside it. The loss panel takes everything left over.

    Args:
        sheet_aspect: sheet width / height, in physical units.

    Returns:
        ``(rect_geom, rect_loss)``, each ``[x0, y0, w, h]`` in figure coordinates.
    """
    h = top - bottom
    w_geom = GEOM_PANEL_WIDEN * sheet_aspect * h * (FIG_H_IN / FIG_W_IN)
    x_loss = left + w_geom + gap
    return [left, bottom, w_geom, h], [x_loss, bottom, right - x_loss, h]


def _camera(bounds, box_aspect, margin=0.035, anchor="center"):
    """Data window matching an axes box's aspect, so units are equal without rescaling the box.

    Args:
        bounds: ``(x0, y0, x1, y1)`` of the content.
        box_aspect: the axes box's width / height in physical units.
        margin: padding as a fraction of the larger content span.
        anchor: ``left`` keeps the content against the window's left edge, leaving the slack the
            widened panel carries as one clear column for the hinge zoom rather than as two
            useless half-columns.

    Returns:
        ``(x0, x1, y0, y1)``.
    """
    x0, y0, x1, y1 = bounds
    pad = margin * max(x1 - x0, y1 - y0)
    x0, x1, y0, y1 = x0 - pad, x1 + pad, y0 - pad, y1 + pad
    cx, cy, w, h = 0.5 * (x0 + x1), 0.5 * (y0 + y1), x1 - x0, y1 - y0
    if w / h < box_aspect:
        w = h * box_aspect
    else:
        h = w / box_aspect
    y_lo, y_hi = cy - h / 2, cy + h / 2
    if anchor == "left":
        return x0, x0 + w, y_lo, y_hi
    return cx - w / 2, cx + w / 2, y_lo, y_hi


def _draw_sheet(ax, geom, *, lw):
    """The sheet filled, with the removed material drawn on top.

    The cut polygons are the true ones, but a kerf is 0.5 mm against a 2.8 m sheet -- under a
    tenth of a pixel once the whole sheet is on screen -- so they are STROKED as well as filled.
    The stroke gives each slot a floor width without altering the geometry underneath it, which is
    the only way the pattern reads as a pattern rather than as a solid orange rectangle.
    """
    from nff.utils.visualization import _plot_shapely
    _plot_shapely(ax, geom["sheet"], facecolor=ORANGE, edgecolor="none", zorder=1)
    _plot_shapely(ax, geom["cuts"], facecolor=INK, edgecolor=INK, lw=lw, zorder=2)


def _loss_series(history):
    """Weighted loss components and the total, as dense arrays over the recorded epochs.

    Only terms the run actually carried are returned: the loss registry emits ``comp_<term>`` keys
    for non-zero weights alone, so asking for the rest would stack a row of zeros under the total.
    """
    n = len(history)
    epochs = np.arange(n, dtype=float)

    def series(key):
        return np.array([float(h.get(key, 0.0)) for h in history])

    active = [(lbl, col, series(key)) for key, lbl, col in LOSS_TERMS
              if any(key in h for h in history) and np.any(np.abs(series(key)) > 1e-12)]
    total = series("total")
    return epochs, active, total


def _loss_limits(epochs, active, total):
    """Axis limits fixed once over the whole run, so nothing rescales mid-clip.

    The first epochs are excluded from the upper limit: an untrained design routinely scores an
    order of magnitude above everything that follows, and letting it set the scale flattens the
    entire rest of the film into the bottom of the panel.
    """
    skip = min(10, max(1, len(epochs) // 20))
    stack = np.sum([a for _, _, a in active], axis=0) if active else np.zeros(len(epochs))
    finite = lambda v: v[np.isfinite(v)]
    above = np.concatenate([finite(stack[skip:]), finite(total[skip:])])
    y_max = float(np.percentile(above, 99)) * 1.30 if above.size else 1.0
    below = finite(np.minimum(total, 0.0)[skip:])
    y_min = float(np.min(below)) * 1.15 if below.size and np.min(below) < 0 else -0.03 * y_max
    # A hair of slack past the last epoch, so the playhead dot sits inside the panel on the final
    # frame instead of being sliced in half by the axis.
    x_max = max(1.0, float(epochs[-1]) if len(epochs) else 1.0)
    return (0.0, x_max * 1.015), (y_min, y_max)


def render_training_film(run_dir, out_mp4, *, duration=30.0, fps=60, dpi=150,
                         ease="smoothstep", hold_start=None, hold_end=None,
                         keep_frames=False, crf=18, title=None, progress_every=120):
    """Render the film and encode it to H.264.

    Args:
        run_dir: a run folder carrying ``config.yaml``, ``design_traj.npz`` and
            ``loss_history.json``.
        out_mp4: output path.
        duration: clip length in seconds, holds included.
        fps: frame rate; ``duration * fps`` frames are rendered.
        dpi: 150 with the module's figure size gives exactly 1920x1080.
        ease: passed to :func:`resample_trajectory`.
        hold_start, hold_end: hold frames; default to a third of a second and one second.
        keep_frames: keep the PNG sequence beside the movie (an ``animategraphics`` deck wants it).
        crf: x264 quality, lower is better.
        title: figure title; ``None`` derives one from the run folder name.
        progress_every: frames between progress lines.

    Returns:
        The path written.
    """
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import matplotlib.pyplot as plt

    from nff.config.experiment import load_and_parse_config
    from nff.closed.setup import (build_closed_initial_state, init_closed_les_params,
                                  build_surrogate_energy)
    from nff.closed.cut_export import build_run_cut_geometry
    from nff_paper.closed_support.shield_render import write_mp4
    from nff.topology.closed_builder_jax import solve_cut_vertices_jax, boundary_flat_from_logits
    from nff.utils.figstyle import audit_type

    epochs, leaves, history = load_trajectory(run_dir)
    if len(history) == 0:
        raise RuntimeError(f"{run_dir}/loss_history.json is empty; the loss panel has nothing to draw.")
    print(f"  trajectory: {len(epochs)} snapshots over epochs {epochs[0]}-{epochs[-1]}, "
          f"{len(history)} metric rows, leaves {sorted(leaves)}")

    config = load_and_parse_config(os.path.join(run_dir, "config.yaml"))
    initial_state, _ = build_closed_initial_state(config)
    params0, static_features = init_closed_les_params(config)
    _, _, geometry_fn, _, w_lig_logit0 = build_surrogate_energy(
        config, static_features, initial_state, params0)
    struct, sliders = static_features["struct"], static_features["sliders"]

    n_frames = max(2, int(round(duration * fps)))
    hold_start = int(round(fps / 3)) if hold_start is None else int(hold_start)
    hold_end = int(fps) if hold_end is None else int(hold_end)
    t_epoch, frames = resample_trajectory(epochs, leaves, n_frames, ease=ease,
                                          hold_start=hold_start, hold_end=hold_end)

    def geometry_at(i):
        """Frame ``i``'s exact cut geometry, rebuilt from its interpolated design."""
        p = {k: jnp.asarray(v[i]) for k, v in frames.items()}
        hinge_w_lig = geometry_fn(p).w_lig if geometry_fn is not None else None
        coords = np.asarray(solve_cut_vertices_jax(
            struct, boundary_flat_from_logits(sliders, p["bnd_logits"]),
            jax.nn.sigmoid(p["z"])))
        geom, w_lig, _ = build_run_cut_geometry(
            initial_state, coords, struct, config, config.hinge_model, hinge_w_lig)
        return geom, np.asarray(w_lig, dtype=float)

    # The flat outline is the stock rectangle at every epoch -- boundary sliders move only
    # tangentially and the corners are pinned -- so one probe fixes the camera for the whole film.
    geom0, _ = geometry_at(0)
    sx0, sy0, sx1, sy1 = geom0["sheet"].bounds
    sheet_aspect = (sx1 - sx0) / (sy1 - sy0)
    rect_geom, rect_loss = _panel_rects(sheet_aspect)
    cam = _camera((sx0, sy0, sx1, sy1),
                  rect_geom[2] * FIG_W_IN / (rect_geom[3] * FIG_H_IN), anchor="center")

    ep_axis, active, total = _loss_series(history)
    (lx0, lx1), (ly0, ly1) = _loss_limits(ep_axis, active, total)
    print(f"  loss panel: {len(active)} live terms -> {[lbl for lbl, _, _ in active]}")

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=dpi, facecolor="white")
    ax_geom = fig.add_axes(rect_geom)
    ax_loss = fig.add_axes(rect_loss)
    # The alignment the film is judged on, asserted rather than eyeballed.
    assert abs(ax_geom.get_position().y0 - ax_loss.get_position().y0) < 1e-9
    assert abs(ax_geom.get_position().y1 - ax_loss.get_position().y1) < 1e-9

    fig.suptitle(title if title is not None else _default_title(run_dir),
                 x=rect_geom[0], ha="left", y=0.965, va="top", color=INK, weight="bold")

    out_dir = os.path.join(run_dir, "film_frames")
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    for i in range(n_frames):
        geom, w_lig = geometry_at(i)
        ax_geom.clear()
        _draw_sheet(ax_geom, geom, lw=0.45)
        ax_geom.set_xlim(cam[0], cam[1])
        ax_geom.set_ylim(cam[2], cam[3])
        ax_geom.axis("off")
        ax_geom.set_title("Cut pattern", color=INK, loc="left", pad=6)
        ax_geom.text(sx0, sy0 - 0.030 * (sy1 - sy0),
                     rf"epoch {t_epoch[i]:.0f}    ·    $w_\mathrm{{lig}}$ {w_lig.mean():.1f} mm",
                     color=GREY, ha="left", va="top")

        _draw_loss(ax_loss, ep_axis, active, total, t_epoch[i],
                   xlim=(lx0, lx1), ylim=(ly0, ly1))

        if i == 0:
            audit_type(fig)
        fig.savefig(os.path.join(out_dir, f"frame_{i:04d}.png"), dpi=dpi)
        if progress_every and (i % progress_every == 0 or i == n_frames - 1):
            print(f"    frame {i + 1}/{n_frames}", flush=True)

    plt.close(fig)
    write_mp4(out_dir, out_mp4, fps=fps, crf=crf)
    if not keep_frames:
        shutil.rmtree(out_dir)
    print(f"  Saved training film to {out_mp4}  ({n_frames} frames, {fps} fps, "
          f"{n_frames / fps:.1f} s)")
    return out_mp4


def _default_title(run_dir):
    """Grid and hinge model read off the run folder, so the film says what it is showing."""
    return os.path.basename(os.path.normpath(run_dir)).replace("run_", "").replace("_", " ")


def _draw_loss(ax, epochs, active, total, t_now, *, xlim, ylim):
    """The weighted loss components stacked under the total, revealed up to ``t_now``.

    Limits are passed in rather than derived per frame: a panel that rescales as the curve grows
    reads as the loss moving when it is only the axis moving.
    """
    ax.clear()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    k = int(np.searchsorted(epochs, t_now, side="right"))
    k = max(1, min(k, len(epochs)))
    x = epochs[:k]

    if active:
        ax.stackplot(x, *[a[:k] for _, _, a in active],
                     labels=[lbl for lbl, _, _ in active],
                     colors=[col for _, col, _ in active], alpha=0.78, zorder=2)
    ax.plot(x, total[:k], color=INK, lw=1.8, zorder=6)
    ax.axhline(0.0, color=GREY, lw=0.6, alpha=0.5, zorder=1)

    # The playhead: a rule at the current epoch and a dot riding the total.
    y_now = float(total[k - 1])
    ax.axvline(x[-1], color=GREY, lw=0.8, alpha=0.55, zorder=7)
    if np.isfinite(y_now):
        ax.plot([x[-1]], [y_now], marker="o", ms=4.5, color=INK, zorder=8)

    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Weighted loss contribution")
    ax.set_title("Loss", color=INK, loc="left", pad=6)
    if active:
        # Below the axes, not inside it: the stack fills the panel, so an in-plot legend would
        # cover the very terms it names. Anchoring outside leaves the axes box untouched, so the
        # alignment with the geometry panel survives.
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.115),
                  ncol=min(len(active), 3), frameon=False, handlelength=1.4,
                  columnspacing=1.8, borderpad=0.0,
                  fontsize=FILM_BODY_PT * 0.85)
