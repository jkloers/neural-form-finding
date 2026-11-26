"""The compression-shift figure: where a penalty moved the hinge paths, on one hex field.

A single panel of ``(eta_a, eta_s)`` -- ligament axial strain against shear, both in units of
ligament width. Each hexagon holds the SIGNED change in how many distinct hinge paths cross it,
after minus before:

    orange  fewer hinges route here      green  more hinges route here
    grey    same in both                 blank  crossed by neither

Lives in ``nff/closed/`` rather than beside the other figure scripts because ``run_closed`` emits it
as part of every run, and ``nff/closed/`` may not import ``nff/scripts/``. The CLI wrapper is
``legacy/experiments/figures/plot_compression_shift.py``.

The one-panel form is deliberate. A before/after pair asks the reader to subtract two dense clouds
by eye, which is what makes such a comparison unreadable; binning both on a single grid and drawing
only the difference does that subtraction for them, and the grey backbone keeps the unchanged
structure visible so the change is read against the shape of the mechanism rather than against
empty space.
"""


import numpy as np

from nff.utils.figstyle import use_agg, apply_charter, ORANGE, INK, GREY
use_agg()
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import RegularPolygon                      # noqa: E402
from matplotlib.collections import LineCollection                  # noqa: E402
from matplotlib.lines import Line2D                                # noqa: E402

from nff.hinge.path_binning import (PROJECTIONS, densify, trajectories,  # noqa: E402
                                           endpoints, n_unique_delta, ID_OFFSET, N_RESAMPLE)


GRIDSIZE = 84             # hexagons across x; the row count is DERIVED -- see _regular_gridsize
# Endpoints put ONE point per hinge on the plane instead of 400, so the same grid would leave
# almost every occupied cell holding a single hinge and the field would read as noise.
GRIDSIZE_ENDPOINTS = 30
UNCHANGED = "#F2F4F6"     # the grey backbone: crossed by both, net change zero
ELECTRIC_GREEN = "#00C853"
# Three stops: Princeton orange for loss, light grey for no change, electric green for gain.
# A wider multi-stop ramp with dark burnt-orange and deep-green anchors was tried and rejected --
# the dark ends read as muddy and the extra hues obscured the two-sided story rather than grading it.
_RAMP = (ORANGE, UNCHANGED, ELECTRIC_GREEN)
_SATURATION_PCT = 88      # percentile of |change| that reaches full colour; see build_figure
_HEX_EDGE = "#FFFFFF"
_COMPRESSION_TINT = "#D62828"
_ORIGIN = "#000000"


def _regular_gridsize(nx: int, extent) -> tuple:
    """``(nx, ny)`` giving GEOMETRICALLY REGULAR hexagons over ``extent``.

    Passing ``gridsize`` as a bare int does not give regular cells. matplotlib then picks
    ``ny = nx / sqrt(3)``, which is only correct when the data extent is square -- for any other
    extent the cells come out visibly squashed along one axis.

    From the hexbin source a cell spans ``sx = W / nx`` wide by ``2 * sy / 3`` tall, with
    ``sy = H / ny``. A regular hexagon of width ``sx`` has height ``2 * sx / sqrt(3)``, so

        2 * sy / 3 = 2 * sx / sqrt(3)   =>   sy = sqrt(3) * sx
        =>  ny = nx * H / (sqrt(3) * W)

    That is exact in DATA coordinates, so it only reads as regular on screen if the axes also map
    both axes at the same scale -- hence ``set_aspect('equal')`` alongside it. Both are required;
    either alone still leaves the cells distorted.
    """
    lo_x, hi_x, lo_y, hi_y = extent
    ny = max(1, int(round(nx * (hi_y - lo_y) / (np.sqrt(3.0) * (hi_x - lo_x)))))
    return (int(nx), ny)


def _colormap():
    """Orange (paths lost) -> light grey (unchanged) -> green (paths gained).

    Diverging about zero because the two signs are different events, not two ends of one scale, and
    seated on grey rather than white at the midpoint so an unchanged cell still reads as OCCUPIED --
    white is reserved for "no path came here at all".
    """
    return LinearSegmentedColormap.from_list("lost_unchanged_gained", _RAMP)


def _overlay_paths(ax, pairs, proj, n_paths=None, seed=0):
    """Draw the trajectories themselves over the hex field, old design against new.

    The hexagons answer WHERE hinges ended up but throw away the shape of the journey -- whether a
    hinge ran straight out, curved, or doubled back. These lines put that back, and they are the
    only part of the figure that shows the deployment as a process rather than as a distribution.

    ★ HUE is reserved for the hex field (the difference); the paths are separated by VALUE instead,
    old in grey and new in ink. Colouring the lines orange/green as well would put two different
    meanings on the same channel -- "this cell lost hinges" and "this line belongs to the old
    design" -- and the reader has no way to tell which one a given patch of colour is speaking for.

    Args:
        n_paths: draw a random subset of this many hinges per design; ``None`` draws all 275. Every
            path is drawn faintly, so the full set reads as a density and a subset reads as
            individual trajectories.
        seed: fixes which hinges the subset picks, so the figure is reproducible.
    """
    rng = np.random.default_rng(seed)
    for (before, after) in pairs:
        n_total = trajectories(before, proj).shape[0]
        subset = None
        if n_paths is not None and n_paths < n_total:
            # The SAME hinges in both designs, or the comparison is between two different samples
            # and any apparent shift could be the draw rather than the design.
            subset = rng.choice(n_total, int(n_paths), replace=False)
        # A thinned set is meant to be read line by line, so it earns more weight and opacity than
        # the full 275, which is only ever legible as a density.
        lw, a_old, a_new = ((0.9, 0.55, 0.75) if subset is not None else (0.45, 0.18, 0.26))
        for paths, colour, alpha in ((before, GREY, a_old), (after, INK, a_new)):
            traj = trajectories(paths, proj)
            if subset is not None:
                traj = traj[subset]
            ax.add_collection(LineCollection(
                list(traj), colors=colour, linewidths=lw, alpha=alpha, zorder=3.8,
                capstyle="round"))


def _cell_key(xy):
    """Hashable identity for a hex cell centre.

    Every sub-problem is binned on the same ``(extent, gridsize)``, so cell centres come out
    bit-identical; rounding only guards against a last-bit difference turning one cell into two.
    """
    return (round(float(xy[0]), 10), round(float(xy[1]), 10))


def _bin_one(pts, ids, extent, grid):
    """``{cell key: signed change in distinct-path count}`` for ONE before/after pair.

    Done on a throwaway figure purely to reuse matplotlib's hex tiling, which interleaves two
    offset grids and is not worth reimplementing.
    """
    fig = plt.figure()
    ax = fig.add_subplot()
    hb = ax.hexbin(pts[:, 0], pts[:, 1], C=ids, reduce_C_function=n_unique_delta,
                   gridsize=grid, mincnt=1, extent=extent)
    out = {_cell_key(o): float(v) for o, v in zip(np.asarray(hb.get_offsets()),
                                                  np.asarray(hb.get_array()))}
    plt.close(fig)
    return out


def build_figure(pairs, out_png, gridsize=GRIDSIZE, n_resample=N_RESAMPLE,
                 use_endpoints=False, show_paths=False, n_paths=None):
    """One hex panel of the signed change in distinct-path count, averaged over sub-problems.

    Args:
        pairs: sequence of ``(before, after)`` ``HingePaths``. One pair reduces to the plain
            before/after figure; several are AVERAGED CELL BY CELL, each contributing equally.
        out_png: destination.

    Averaging per cell rather than pooling all paths into one cloud is deliberate. Pooling weights a
    sub-problem by how much of the plane its paths happen to cover, so a grip whose strain cloud
    spreads wider would speak louder than one whose paths stay compact -- an artefact of that grip's
    geometry, not of the penalty. Averaging the per-cell change gives every sub-problem the same
    vote. A cell absent from one sub-problem contributes 0 there, which is the correct reading: no
    paths crossed it before OR after, so nothing changed.

    Returns:
        ``(out_png, stats)``. Statistics come from the RAW trajectories, never the arc-length
        resampled cloud -- resampling reweights points to make the density fair to LOOK at, which is
        exactly what a reported number must not have had done to it.
    """
    apply_charter()
    proj = PROJECTIONS['eta_a_eta_s']

    # Reduce each design to the point cloud the chosen view bins: the whole arc-length-resampled
    # trajectory, or just its deployed endpoint.
    reduce_fn = ((lambda p, off: endpoints(p, proj, off)) if use_endpoints
                 else (lambda p, off: densify(p, proj, n_resample, id_offset=off)))
    dense = [(reduce_fn(b, 0), reduce_fn(a, ID_OFFSET)) for b, a in pairs]
    every = np.concatenate([p for (pb, _), (pa, _) in dense for p in (pb, pa)], axis=0)
    lo_x, hi_x = float(every[:, 0].min()), float(every[:, 0].max())
    lo_y, hi_y = float(every[:, 1].min()), float(every[:, 1].max())
    pad = 0.045 * max(hi_x - lo_x, hi_y - lo_y)
    extent = (lo_x - pad, hi_x + pad, lo_y - pad, hi_y + pad)
    grid = _regular_gridsize(gridsize, extent)

    # Per-sub-problem change on the SHARED grid, then the mean over sub-problems.
    totals = {}
    for (pb, ib), (pa, ia) in dense:
        for key, value in _bin_one(np.concatenate([pb, pa]), np.concatenate([ib, ia]),
                                   extent, grid).items():
            totals[key] = totals.get(key, 0.0) + value
    n_pairs = len(dense)
    mean_delta = {k: v / n_pairs for k, v in totals.items()}

    pts = every
    ids = np.concatenate([i for (_, ib), (_, ia) in dense for i in (ib, ia)], axis=0)

    fig, ax = plt.subplots(figsize=(8.6, 7.4), dpi=200)
    # EQUAL ASPECT is what makes the hexagons regular: hexbin lays them out to be regular in the
    # axes box, so any other aspect stretches every cell. With an int gridsize matplotlib then
    # derives the y count itself to keep them so.
    ax.set_aspect('equal')

    # The BAD region, drawn first so every hexagon sits on top of it: eta_a < 0 is a ligament being
    # squeezed rather than pivoted, which is the regime the penalty exists to empty out.
    ax.axvspan(extent[0], 0.0, color=_COMPRESSION_TINT, alpha=0.055, lw=0, zorder=0)

    # Two hexbins over the same grid, so the GAIN cells are drawn as their own layer ON TOP of the
    # loss/unchanged field. Hexagons tile without overlapping, so this is not about occlusion: it is
    # about the white cell borders. Drawn in one pass, a lone green cell surrounded by orange has
    # half its outline painted by whichever neighbour happens to come later in the collection, and
    # in a field that is overwhelmingly orange the few gains recede. Drawn last, every green cell
    # owns its whole border and reads forward -- which matters because the gains are the point and
    # they are always the smaller population.
    #
    # The values hexbin computes are discarded either way; each layer is filled from the per-cell
    # mean assembled above and masked to its own sign.
    cmap = _colormap()
    cmap.set_bad(alpha=0.0)                       # a masked cell must vanish, not paint grey

    def _layer(keep, zorder):
        h = ax.hexbin(pts[:, 0], pts[:, 1], C=ids, reduce_C_function=n_unique_delta,
                      gridsize=grid, cmap=cmap, mincnt=1,
                      extent=extent, edgecolors=_HEX_EDGE, linewidths=0.25, zorder=zorder)
        vals = np.array([mean_delta.get(_cell_key(o), 0.0)
                         for o in np.asarray(h.get_offsets())])
        h.set_array(np.ma.masked_where(~keep(vals), vals))
        return h, vals

    losses, values = _layer(lambda v: v <= 0.0, 2)
    gains, _ = _layer(lambda v: v > 0.0, 3)

    # Colour limit from a high PERCENTILE of the changed cells, not from the maximum. The
    # distribution is heavy-tailed -- a handful of cells near the origin, where every path is
    # crowded together, carry several times the change of anywhere else -- so scaling to the
    # maximum drives the entire body of the figure to within a few percent of grey and hides the
    # very redistribution it is meant to show. The few extremes then saturate at the ramp's deep
    # ends, which costs nothing: their sign is the message, not their exact magnitude.
    #
    # ONE norm shared by both layers, or the two signs would be scaled independently and a small
    # gain would be painted as strongly as a large loss.
    changed = np.abs(values[values != 0])
    lim = float(np.percentile(changed, _SATURATION_PCT)) if changed.size else 1.0
    norm = TwoSlopeNorm(vmin=-max(lim, 1e-9), vcenter=0.0, vmax=max(lim, 1e-9))
    losses.set_norm(norm)
    gains.set_norm(norm)

    # No origin marker in the endpoint view: no hinge ENDS unstrained, so the cell would either be
    # empty or -- worse -- get painted at whatever occupied cell happens to be nearest the origin.
    if not use_endpoints:
        _mark_origin(ax, losses, extent, grid)

    if show_paths:
        _overlay_paths(ax, pairs, proj, n_paths)

    # A quiet rule at eta_a = 0. Not decoration: it is the sign change between a ligament being
    # pivoted and one being squeezed, so without it the axis carries no physical landmark.
    ax.axvline(0.0, color=_COMPRESSION_TINT, lw=0.9, ls=(0, (4, 4)), alpha=0.55, zorder=4)
    ax.text(extent[0] + 0.02 * (extent[1] - extent[0]), extent[3] - 0.04 * (extent[3] - extent[2]),
            "COMPRESSION   $\\eta_a < 0$", fontsize=9.5, color=_COMPRESSION_TINT,
            va="top", ha="left", alpha=0.95, zorder=5)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel(r"$\eta_a = a\,/\,w_{\mathrm{lig}}$", fontsize=12, labelpad=9, color=INK)
    ax.set_ylabel(r"$\eta_s = s\,/\,w_{\mathrm{lig}}$", fontsize=12, labelpad=9, color=INK)
    ax.grid(False)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(0.7)
    ax.tick_params(length=3.2, width=0.7, labelsize=10, colors=INK)

    _legend(ax, show_origin=not use_endpoints, show_paths=show_paths)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    stats = {'n_sub_problems': n_pairs, 'compressive_pct_before': [], 'compressive_pct_after': [],
             'max_theta_deg_before': [], 'max_theta_deg_after': []}
    for b, a in pairs:
        for key, paths in (('before', b), ('after', a)):
            eta = np.asarray(paths.eta)
            stats[f'compressive_pct_{key}'].append(float((eta[..., 0] < 0).mean()) * 100.0)
            # The DEPLOYED-state fraction, which is what the endpoint view actually draws. It is
            # always the smaller number: compression here is transient, so a hinge can be squeezed
            # mid-deployment and finish in tension.
            stats.setdefault(f'endpoint_compressive_pct_{key}', []).append(
                float((eta[-1, :, 0] < 0).mean()) * 100.0)
            stats[f'max_theta_deg_{key}'].append(float(np.degrees(eta[..., 2]).max()))
    stats['delta_sum'] = float(sum(mean_delta.values()))
    return out_png, stats


def _mark_origin(ax, hb, extent, grid):
    """Fill the cell at ``(0, 0)`` black -- the state every hinge starts from.

    Every path begins at zero strain, so this one cell is crossed by all 275 hinges in BOTH designs
    and its net change is therefore trivially zero. Left in the diverging colour it reads as a
    meaningful "unchanged" result; black says instead that it is the common origin and carries no
    comparison at all.

    Drawn on the grid cell nearest the origin rather than at the origin itself, so it lines up with
    the tiling instead of straddling two cells.
    """
    offsets = np.asarray(hb.get_offsets())
    if offsets.size == 0:
        return
    centre = offsets[np.argmin(np.hypot(offsets[:, 0], offsets[:, 1]))]
    nx, ny = grid
    radius = (extent[3] - extent[2]) / (3.0 * ny)      # centre-to-vertex, matching hexbin's cells
    # Above BOTH hex layers (2 = losses, 3 = gains), or a gain cell at the origin would bury it.
    ax.add_patch(RegularPolygon(tuple(centre), numVertices=6, radius=radius, orientation=0.0,
                                facecolor=_ORIGIN, edgecolor=_HEX_EDGE, linewidth=0.25, zorder=3.5))


def _legend(ax, show_origin=True, show_paths=False):
    """Hexagon swatches, so the key is drawn in the same mark as the data.

    ``show_origin`` is False in the endpoint view, which draws no origin cell -- a key entry for a
    mark that is not on the canvas is worse than no key at all.

    A continuous colour bar was tried instead, on the argument that discrete swatches understate a
    ramp that carries magnitude. It read worse: the bar competes with the hex field for attention
    and stops the key looking like the thing it labels.
    """
    handles = [RegularPolygon((0, 0), numVertices=6, radius=0.5, orientation=0.0,
                              facecolor=c, edgecolor=_HEX_EDGE, linewidth=0.35)
               for c in (ORANGE, UNCHANGED, ELECTRIC_GREEN, _ORIGIN)]
    labels = ["fewer hinges route here", "same", "more hinges route here",
              "unstrained: every hinge starts here"]
    if not show_origin:
        handles, labels = handles[:3], labels[:3]
    if show_paths:
        # Line keys for line marks: a hexagon swatch would misdescribe the trajectory overlay.
        handles = handles + [Line2D([], [], color=GREY, lw=1.4),
                             Line2D([], [], color=INK, lw=1.4)]
        labels = labels + ["old paths: shape only", "new paths: + compression"]
    # "removed"/"created" read as though hinges appear and vanish. They do not: there are 275 in
    # both designs and the figure shows where those SAME hinges route instead.
    leg = ax.legend(handles, labels,
                    handler_map={RegularPolygon: _HexHandler()},
                    loc="lower right", frameon=False, fontsize=10.5,
                    handlelength=1.5, handletextpad=0.7, borderpad=0.0, labelspacing=0.7)
    for text in leg.get_texts():
        text.set_color(INK)


class _HexHandler:
    """Draws a legend key as a hexagon rather than matplotlib's default rectangle."""

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        r = handlebox.height * 0.72
        patch = RegularPolygon((handlebox.xdescent + handlebox.width / 2.0,
                                handlebox.ydescent + handlebox.height / 2.0),
                               numVertices=6, radius=r, orientation=np.pi / 2.0,
                               facecolor=orig_handle.get_facecolor(),
                               edgecolor=orig_handle.get_edgecolor(),
                               linewidth=orig_handle.get_linewidth(),
                               transform=handlebox.get_transform())
        handlebox.add_artist(patch)
        return patch


