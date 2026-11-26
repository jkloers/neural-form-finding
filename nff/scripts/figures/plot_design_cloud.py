"""The design cloud: where independent optimisations travel, and where they stop.

Panel a is a genuine linear slice of design space -- the plane spanned by the first two
principal components of the converged parameter vectors -- so the objective can be evaluated
on it exactly and drawn as contours. The paths are the optimisations themselves, projected
onto that plane, and the sheets are the designs at either end of them.

    JAX_PLATFORMS=cpu conda run -n kgnn_mac python \
        nff/scripts/figures/plot_design_cloud.py \
        --run data/outputs/design_cloud/circle_7x7_n200_s1 --projection all --paper-dir paper

WARNING: the manuscript's plate is the `all` projection, NOT the `pca` default. Its caption says
"the plane of the two leading principal components of the starting and converged patterns
together", which is what `all` spans; `pca` spans the converged spread alone and puts the two
families in different places under a different objective slice. Running the default and copying
the result into paper/Images REPLACES the published figure with a different one, and nothing in
the output says so -- both write a file called solution_set.pdf. Always pass --projection all
when regenerating for the paper.
"""

import argparse
import json
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from nff.utils.figstyle import (FS_ANNOT, FS_BASE, GREY, INK, ORANGE, TEAL, apply_charter,
                                figsize_in, page_bp, panel_letter, paper_rc,
                                save_paper_figure, use_agg)

use_agg()
import matplotlib.pyplot as plt                                       # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection     # noqa: E402
from matplotlib.colors import LinearSegmentedColormap                # noqa: E402

from nff.scripts.figures.plot_design_degeneracy import panel_spectrum  # noqa: E402

FAMILY_COLOURS = (ORANGE, TEAL, "#6A4C93")
FAMILY_FILLS = ("#F9C79A", "#A9DBD4", "#C6B8DC")


# ══════════════════════════════════════════════════════════════════════════════
# The plane
# ══════════════════════════════════════════════════════════════════════════════

def pca_plane(vectors):
    """Mean and the two leading directions of a set of design vectors.

    PCA on the RAW parameter vectors, deliberately: the plane has to be a linear subspace of
    the space the forward map takes, or a point on it could not be turned back into a design
    and the contours would be decoration. Designs are not rotation-canonicalised first --
    the chart's C4 action has no closed form -- which costs little, since the quotient moves
    the median pairwise distance by about 1.5 mm out of ~89.
    """
    X = np.asarray(vectors, dtype=float)
    mean = X.mean(axis=0)
    _, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    basis = Vt[:2]
    var = (S ** 2) / max(len(X) - 1, 1)
    coords = (X - mean) @ basis.T
    residual = np.linalg.norm((X - mean) - coords @ basis, axis=1)
    return {"mean": mean, "basis": basis, "coords": coords,
            "kept": float(var[:2].sum() / var.sum()), "residual": residual}


def lda_plane(vectors, groups, shrink="auto"):
    """A plane chosen to separate the groups, rather than to span their variance.

    Three groups (starts, family A, family B) give two discriminant directions, so the plane
    is exactly what linear discriminant analysis offers. It is fitted ON the groups it then
    separates, so the visible gaps are optimistic by construction -- the honest number is the
    cross-validated accuracy quoted alongside, not the picture.

    The two directions are orthonormalised (they need not be orthogonal), which leaves the
    subspace untouched and keeps a point on the plane mappable back to a real design.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(vectors, dtype=float)
    scaler = StandardScaler().fit(X)
    lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage=shrink, n_components=2)
    lda.fit(scaler.transform(X), np.asarray(groups))
    raw = (lda.scalings_[:, :2].T / scaler.scale_[None, :])
    basis, _ = np.linalg.qr(raw.T)
    basis = basis.T
    mean = X.mean(axis=0)
    coords = (X - mean) @ basis.T
    residual = np.linalg.norm((X - mean) - coords @ basis, axis=1)
    return {"mean": mean, "basis": basis, "coords": coords, "kept": float("nan"),
            "residual": residual}


FIG_W_DEFAULT = page_bp(1.0, 1.0)[0]   # the page column, the default canvas width


def axis_scales_mm(basis, jacobian):
    """Millimetres of cut-vertex movement per unit along each plane axis.

    The parameters are logits of different kinds, so a raw distance in the plane mixes
    incommensurable directions and means nothing physical. Rescaling each axis by how far it
    actually moves the sheet makes the axes readable without changing the plane.
    """
    n_nodes = jacobian.shape[0] // 2
    return np.linalg.norm(jacobian @ basis.T, axis=0) / np.sqrt(n_nodes)


def loss_grid(loss_fn, unravel, plane, extent, n=90, rows_per_chunk=12):
    """The objective evaluated on the plane, exactly.

    Every grid point is mapped back to a full design and scored with the same loss the runs
    minimised, so the contours are the real landscape restricted to this slice -- not an
    interpolation of the scattered endpoint values.
    """
    import jax
    import jax.numpy as jnp

    nu, nv = (n, n) if np.isscalar(n) else n
    u = np.linspace(extent[0], extent[1], nu)
    v = np.linspace(extent[2], extent[3], nv)
    UU, VV = np.meshgrid(u, v)
    pts = (plane["mean"][None, :]
           + UU.reshape(-1, 1) * plane["basis"][0][None, :]
           + VV.reshape(-1, 1) * plane["basis"][1][None, :])

    batched = jax.jit(jax.vmap(lambda vec: loss_fn(unravel(vec), 1.0)[0]))
    out = []
    for i in range(0, len(pts), nu * rows_per_chunk):
        out.append(np.asarray(batched(jnp.asarray(pts[i:i + nu * rows_per_chunk]))))
    return u, v, np.concatenate(out).reshape(nv, nu)


# ══════════════════════════════════════════════════════════════════════════════
# Drawing
# ══════════════════════════════════════════════════════════════════════════════

def _clear_spot(anchor, half_w, half_h, occupied, frame, taken, n_dirs=16):
    """A home for a thumbnail: near its anchor, inside the frame, over as little as possible.

    Hand-picked offsets do not survive a change of projection -- one sheet ended up hanging
    over the axis and another sat on the paths. Scoring candidates against what is actually
    drawn keeps the placement sensible on any plane.
    """
    ax_pt = np.asarray(anchor, dtype=float)
    # Distances are measured in units of the glyph, so a clearance means the same thing on
    # both axes however the plane is scaled.
    scaled = np.column_stack([occupied[:, 0] / half_w, occupied[:, 1] / half_h])
    best, best_score = None, None
    for radius in (2.5, 3.4, 4.4, 5.6, 7.0, 8.5):
        for ang in np.linspace(0.0, 2.0 * np.pi, n_dirs, endpoint=False):
            pos = ax_pt + radius * np.array([half_w * np.cos(ang), half_h * np.sin(ang)])
            if not (frame[0] + half_w <= pos[0] <= frame[1] - half_w
                    and frame[2] + half_h <= pos[1] <= frame[3] - half_h):
                continue
            q = np.array([pos[0] / half_w, pos[1] / half_h])
            # Maximise room, do not merely accept the first empty spot: the nearest
            # acceptable place put a sheet in the middle of the path bundle, where the
            # occupancy count happened to be low but the lines still read through it.
            clearance = float(np.min(np.max(np.abs(scaled - q), axis=1)))
            if any(abs(t[0] - pos[0]) < 2.3 * half_w and abs(t[1] - pos[1]) < 2.3 * half_h
                   for t in taken):
                continue
            # Nearest spot that is CLEAR ENOUGH, not the emptiest one anywhere: maximising
            # clearance alone banished every sheet to a far corner on a long leader.
            score = (0 if clearance >= 1.75 else 1, radius, -clearance)
            if best_score is None or score < best_score:
                best, best_score = pos, score
    return ax_pt + np.array([3.0 * half_w, 0.0]) if best is None else best


def _sheet_backing(ax, centre_xy, half_w, half_h, zorder):
    """An opaque card under a thumbnail, so contour lines do not run through the design."""
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((centre_xy[0] - 1.12 * half_w, centre_xy[1] - 1.12 * half_h),
                           2.24 * half_w, 2.24 * half_h, facecolor="white",
                           edgecolor="none", zorder=zorder))


def _sheet_polys(tiles, centre_xy, half_w, half_h, sheet_half):
    """Tile polygons of one design, placed as a thumbnail at ``centre_xy``.

    The plane's two axes carry different units, so a thumbnail has to be scaled per axis or
    it would be sheared by the aspect ratio of the panel it sits in.
    """
    s = np.array([half_w, half_h]) / sheet_half
    return [(t * s) + np.asarray(centre_xy) for t in tiles]


def spread_selection(points, k):
    """Greedy farthest-point pick of ``k`` designs covering the plane."""
    pts = np.asarray(points, dtype=float)
    rng = np.ptp(pts, axis=0)
    rng[rng == 0] = 1.0
    scaled = pts / rng
    chosen = [int(np.argmax(np.linalg.norm(scaled - scaled.mean(axis=0), axis=1)))]
    while len(chosen) < min(k, len(pts)):
        d = np.min(np.linalg.norm(scaled[:, None, :] - scaled[None, chosen, :], axis=-1),
                   axis=1)
        chosen.append(int(np.argmax(d)))
    return np.array(chosen)


def _landscape_cmap():
    """Light yellow through to orange: light where the objective is low.

    Kept pale at the top end so the family dots, which are saturated orange and teal with ink
    edges, still read against it.
    """
    return LinearSegmentedColormap.from_list(
        "landscape", ["#FFFFFF", "#FEF6DA", "#FCE7A4", "#F8C877", "#EFA24B"])


def place_near(anchor_xy, half_w, half_h, order, vertical=True, gap=2.5):
    """Thumbnail positions a short step from their own points.

    Kept close to the anchor rather than banished to the frame edge, so the reader does not
    have to trace a long leader to learn which design is which. The step is perpendicular to
    the bundle of paths, which is horizontal here, so the sheets clear the paths instead of
    covering them.
    """
    pts = np.asarray(anchor_xy, dtype=float)
    out = np.array(pts, dtype=float)
    side = np.where(np.arange(len(order)) % 2 == 0, 1.0, -1.0)
    for slot, idx in enumerate(order):
        if vertical:
            out[idx, 1] = pts[idx, 1] + side[slot] * gap * half_h
        else:
            out[idx, 0] = pts[idx, 0] + side[slot] * gap * half_w
    # Nudge apart only within a band, preserving order so leaders cannot cross.
    for s in (1.0, -1.0):
        band = [i for slot, i in enumerate(order) if side[slot] == s]
        band.sort(key=lambda i: pts[i, 0])
        for a, b in zip(band, band[1:]):
            out[b, 0] = max(out[b, 0], out[a, 0] + 2.25 * half_w)
    return out


def exemplar_starts(start_vectors, labels):
    """One typical starting design per family: the start nearest that family's mean start.

    An earlier version picked the two closest starts that diverge, which made a striking
    picture in the plane -- until the distance was measured in the full design space instead
    of the projection. There the closest cross-family pair is 8.27 apart against a typical
    nearest neighbour of 9.45: barely closer than average, because in ~90 dimensions random
    points are all about equally far apart. The pair looked special only in the projection,
    so the figure no longer claims it. Typical members carry the same point honestly.
    """
    S = np.asarray(start_vectors, dtype=float)
    out = []
    for c in np.unique(labels):
        members = np.flatnonzero(labels == c)
        centre = S[members].mean(axis=0)
        out.append(int(members[np.argmin(np.linalg.norm(S[members] - centre, axis=1))]))
    D = np.linalg.norm(S[:, None, :] - S[None, :, :], axis=-1)
    typical = float(np.median(np.min(np.where(np.eye(len(S), dtype=bool), np.inf, D), axis=1)))
    cross = np.where(labels[:, None] != labels[None, :], D, np.inf)
    return out, float(cross.min()), typical


def _faded_segments(path, fidelity, colour, lo=0.03, hi=0.40):
    """A path split into segments whose opacity follows how faithfully it is placed.

    A point far off the plane lands at a projected position whose objective has nothing to
    do with the design's own. Drawing such a stretch at full strength invites the reader to
    trace it against the contours, which is exactly the reading that is not available.
    """
    from matplotlib.colors import to_rgb

    seg = np.stack([path[:-1], path[1:]], axis=1)
    f = 0.5 * (np.asarray(fidelity)[:-1] + np.asarray(fidelity)[1:])
    rgb = to_rgb(colour)
    return seg, np.array([(*rgb, lo + (hi - lo) * x) for x in f])


def panel_cloud(ax, plane_xy, traj_xy, labels, tiles_end, tiles_start, start_xy,
                contours, sheet_half, scale=1.0, n_end=2, n_starts=2, fidelity=None,
                keep=None):
    """Paths, endpoints, starting patterns, and the objective they descended.

    Path opacity tracks how faithfully each point can be placed on this plane -- see
    ``_faded_segments``. Only the converged end of a run is drawn solidly, because only
    there does the plane's objective match the design's own (2%, against 91% at the start).
    """
    cs = None
    if contours is None:
        # No landscape on a plane that was fitted to separate groups: the objective there is
        # nothing like the objective at the designs (measured 790% off), so contours would
        # be a decoration that reads as a measurement.
        u = v = None
    else:
        u, v, Z = contours
        lo, hi = float(np.nanmin(Z)), float(np.nanpercentile(Z, 92))
        levels = np.unique(lo + (hi - lo) * np.geomspace(0.02, 1.0, 9))
        cs = ax.contour(u, v, Z, levels=levels, colors=GREY, linewidths=0.35, alpha=0.75,
                        zorder=1)
        ax.clabel(cs, cs.levels[::2], fmt=lambda x: f"{x:.2f}", fontsize=FS_ANNOT / scale,
                  inline=True, colors=GREY)

    # Every optimisation, as the path it actually took, faded where the plane cannot place
    # it faithfully.
    if fidelity is None:
        ax.add_collection(LineCollection(traj_xy, colors=INK, linewidths=0.28, alpha=0.30,
                                         zorder=2))
    else:
        segs, cols = [], []
        for path, fid in zip(traj_xy, fidelity):
            sg, cl = _faded_segments(path, fid, INK)
            segs.append(sg)
            cols.append(cl)
        ax.add_collection(LineCollection(np.concatenate(segs), colors=np.concatenate(cols),
                                         linewidths=0.28, zorder=2))

    # A sheet must come out SQUARE on the page. The two axes carry different ranges over a
    # non-square box, so the half-height follows from the half-width through the box aspect;
    # taking a fixed fraction of each span stretches every thumbnail.
    all_xy = np.vstack([plane_xy, start_xy])
    span_x = np.ptp(u) if u is not None else 1.35 * np.ptp(all_xy[:, 0])
    span_y = np.ptp(v) if v is not None else 1.35 * np.ptp(all_xy[:, 1])
    box = ax.get_position()
    fw, fh = ax.figure.get_size_inches()
    aspect_box = (box.width * fw) / (box.height * fh)
    half_w = 0.052 * span_x
    half_h = half_w * aspect_box * (span_y / span_x)

    for c in np.unique(labels):
        m = labels == c
        colour = FAMILY_COLOURS[c % len(FAMILY_COLOURS)]
        # Above the highlighted paths: a path arrives AT its family, so it should run
        # beneath the endpoints rather than across them.
        ax.scatter(plane_xy[m, 0], plane_xy[m, 1], s=9, c=colour, edgecolors=INK,
                   linewidths=0.28, zorder=10)

    # The starting patterns are the interesting ones -- they differ wildly, where the
    # converged ones barely differ at all -- so they get most of the illustrations.
    # Every start, so the reader can see where the optimisations all begin.
    ax.scatter(start_xy[:, 0], start_xy[:, 1], s=1.8, c=INK, alpha=0.75, linewidths=0,
               zorder=3)

    if keep is None:
        keep = list(range(min(2, len(start_xy))))

    # The frame is fixed BEFORE anything is placed, so a thumbnail can be required to sit
    # inside it. Deciding the limits afterwards is what let a sheet hang over the axis.
    if u is not None:
        frame = (u.min(), u.max(), v.min(), v.max())
    else:
        mx, my = 0.14 * np.ptp(all_xy[:, 0]), 0.14 * np.ptp(all_xy[:, 1])
        frame = (all_xy[:, 0].min() - mx, all_xy[:, 0].max() + mx,
                 all_xy[:, 1].min() - my, all_xy[:, 1].max() + my)

    # Everything a thumbnail should avoid landing on.
    # Sample ALONG each path, not just at its snapshots. Early in a run consecutive
    # snapshots are far apart, so the drawn segment sweeps regions holding no vertex -- which
    # is how a thumbnail once landed in the middle of the bundle on an "empty" spot.
    dense = []
    for t in traj_xy:
        for f in np.linspace(0.0, 1.0, 20, endpoint=False):
            dense.append(t[:-1] * (1.0 - f) + t[1:] * f)
    occupied = np.vstack([plane_xy, start_xy] + dense)

    # Family names, close to their own clusters, and part of what the sheets avoid.
    label_pts = []
    for c in np.unique(labels):
        m = labels == c
        anchor = np.array([plane_xy[m, 0].mean(), plane_xy[m, 1].max() + 0.35 * half_h])
        anchor[1] = min(anchor[1], frame[3] - 1.2 * half_h)
        ax.text(anchor[0], anchor[1], chr(65 + c), ha="center", va="bottom",
                color=FAMILY_COLOURS[c % len(FAMILY_COLOURS)],
                fontsize=FS_BASE / scale, fontweight="bold", zorder=12)
        label_pts.extend([anchor + np.array([dx * half_w, dy * half_h])
                          for dx in (-0.4, 0.0, 0.4) for dy in (0.1, 0.5)])
    occupied = np.vstack([occupied, np.array(label_pts)])

    # One typical start per family, with its path on top in the family colour.
    # Family A solid, family B dashed, the same two styles panel c uses, so the reader reads
    # the two panels as the same two runs.
    for slot, k in enumerate(sorted(keep, key=lambda i: labels[i])):
        colour = FAMILY_COLOURS[labels[k] % len(FAMILY_COLOURS)]
        style = "solid" if slot == 0 else (0, (3.5, 2))
        ax.plot(traj_xy[k][:, 0], traj_xy[k][:, 1], color="white", lw=2.3, alpha=0.9,
                zorder=8, solid_capstyle="round")
        ax.plot(traj_xy[k][:, 0], traj_xy[k][:, 1], color=colour, lw=1.4, alpha=1.0,
                ls=style, zorder=9, solid_capstyle="round")

    placed = []
    start_pos = {}
    for k in sorted(keep, key=lambda i: labels[i]):
        start_pos[k] = _clear_spot(start_xy[k], half_w, half_h, occupied, frame, placed)
        placed.append(start_pos[k])
    for k in keep:
        colour = FAMILY_COLOURS[labels[k] % len(FAMILY_COLOURS)]
        ax.plot([start_xy[k, 0], start_pos[k][0]], [start_xy[k, 1], start_pos[k][1]],
                color=colour, lw=0.4, alpha=0.9, zorder=4)
        _sheet_backing(ax, start_pos[k], half_w, half_h, zorder=5)
        ax.add_collection(PolyCollection(
            _sheet_polys(tiles_start[k], start_pos[k], half_w, half_h, sheet_half),
            facecolors="#F2F3F5", edgecolors=colour, linewidths=0.4, zorder=6))
    ax.scatter(start_xy[keep, 0], start_xy[keep, 1], s=13, facecolors="white",
               edgecolors=INK, linewidths=0.6, zorder=11)

    # One converged sheet per family is enough: within a family they are near-identical.
    for c in np.unique(labels):
        members = np.flatnonzero(labels == c)
        sub = plane_xy[members]
        k = int(members[np.argmin(np.linalg.norm(sub - sub.mean(0), axis=1))])
        pos = _clear_spot(plane_xy[k], half_w, half_h, occupied, frame, placed)
        placed.append(pos)
        ax.plot([plane_xy[k, 0], pos[0]], [plane_xy[k, 1], pos[1]], color=GREY, lw=0.3,
                alpha=0.85, zorder=4)
        _sheet_backing(ax, pos, half_w, half_h, zorder=6)
        ax.add_collection(PolyCollection(
            _sheet_polys(tiles_end[k], pos, half_w, half_h, sheet_half),
            facecolors=FAMILY_FILLS[c % len(FAMILY_FILLS)], edgecolors=INK,
            linewidths=0.18, zorder=7))

    ax.set_xlim(frame[0], frame[1])
    ax.set_ylim(frame[2], frame[3])
    ax.grid(False)
    ax.tick_params(length=2.4, pad=1.5)
    return cs


def panel_convergence(ax, steps, errors, labels, seeds, keep, radius, scale=1.0, n_marks=7):
    """Outline error against step, for the two runs highlighted in panel a.

    The same pair, seen the other way round: panel a shows where those two designs went,
    this shows what it cost them. Their family medians sit behind as context, so a reader
    can tell whether the two exemplars are typical of their families.

    The error is drawn as a fraction of the target radius, not in millimetres. This ensemble is
    purely geometric -- a closed-form deployment with no material and no load -- so a millimetre
    would put a physical size on a run that has none. The denominator is the one the validity
    ladder already prints in its own 'Chamfer / radius' row.
    """
    errors = [100.0 * np.asarray(e, dtype=float) / radius for e in errors]

    for c in np.unique(labels):
        colour = FAMILY_COLOURS[c % len(FAMILY_COLOURS)]
        members = np.flatnonzero(labels == c)
        common = steps[members[0]]
        stack = np.stack([np.interp(common, steps[k], errors[k]) for k in members])
        ax.plot(common, np.median(stack, axis=0), color=colour, lw=2.4, alpha=0.20,
                zorder=2, solid_capstyle="round")

    # The two highlighted runs, in the colours and dashes panel a gave their paths.
    for slot, k in enumerate(sorted(keep, key=lambda i: labels[i])):
        colour = FAMILY_COLOURS[labels[k] % len(FAMILY_COLOURS)]
        idx = np.unique(np.linspace(0, len(steps[k]) - 1, n_marks).astype(int))
        ax.plot(steps[k], errors[k], color=colour, lw=1.3,
                ls="-" if slot == 0 else (0, (3.5, 2)),
                marker="o" if slot == 0 else "s", markevery=list(idx), ms=2.6,
                mfc="white", mew=0.8, zorder=4 + slot,
                label=f"family {chr(65 + labels[k])}")
    ax.set_xscale("symlog", linthresh=200)
    ax.set_yscale("log")
    from matplotlib.ticker import FixedLocator, FuncFormatter
    ax.yaxis.set_major_locator(FixedLocator([3, 5, 7, 10, 15, 20]))
    ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.set_xlabel("optimisation step")
    ax.set_ylabel("outline error / target radius")
    ax.legend(loc="upper right", fontsize=FS_ANNOT / scale, handletextpad=0.4,
              borderpad=0.35, labelspacing=0.25)
    ax.tick_params(length=2.4, pad=1.5)


# ══════════════════════════════════════════════════════════════════════════════
# Plate
# ══════════════════════════════════════════════════════════════════════════════

def build_figure(run_dir, out_path, projection="pca", scale=1.0, width_bp=None, aspect=1.48):
    import jax
    import jax.flatten_util
    import jax.numpy as jnp

    from nff.closed.analytic_targets import circle_outline_unit
    from nff.closed.design_ensemble import (build_symmetry, circle_problem, distance_matrix,
                                            load_ensemble, problem_statics, split_families)
    from nff.closed.design_landscape import coords_jacobian
    from nff.closed.shield_design import _interior_cut_bases, build_loss

    records = load_ensemble(run_dir)
    with open(os.path.join(run_dir, "summary.json")) as fh:
        summary = json.load(fh)
    land = dict(np.load(os.path.join(run_dir, "landscape.npz")))
    meta = summary["config"]

    cfg = circle_problem(M=meta["M"], N=meta["N"], sheet_mm=meta["sheet_mm"],
                         radius_mm=meta["radius_mm"])
    statics = problem_statics(cfg)
    group = build_symmetry(cfg, statics)
    loss_fn = build_loss(statics["forward_fn"], circle_outline_unit(480), statics["plan"],
                         cfg, _interior_cut_bases(statics["struct"]))

    labels, fam = split_families(distance_matrix([r["coords"] for r in records], group))

    finals = np.stack([r["traj_params"][-1] for r in records])
    starts_p = np.stack([r["traj_params"][0] for r in records])
    if projection == "all":
        # PCA over the starts AND the converged designs. Including the starts puts the
        # start-to-end direction into the leading component, which is exactly what a basis
        # fitted to the endpoints alone cannot see.
        plane = pca_plane(np.vstack([starts_p, finals]))
    elif projection == "lda":
        # starts = 0, family A = 1, family B = 2
        plane = lda_plane(np.vstack([starts_p, finals]),
                          np.concatenate([np.zeros(len(starts_p)), labels + 1]))
    else:
        plane = pca_plane(finals)
    traj_xy = [((r["traj_params"] - plane["mean"]) @ plane["basis"].T) for r in records]
    start_xy = np.stack([t[0] for t in traj_xy])
    plane["coords"] = (finals - plane["mean"]) @ plane["basis"].T

    sample = {k[len("params__"):]: jnp.asarray(v)
              for k, v in records[0].items() if k.startswith("params__")}
    _, unravel = jax.flatten_util.ravel_pytree(sample)

    # PC1 gets a modest margin; PC2 is opened out far enough to see the low-objective set
    # close. Probed along PC2 first: the objective is flat to about 10, then climbs (0.19 at
    # 20, 0.67 at 30), so +-32 shows the boundary with room to spare.
    pad = 0.30
    xs = np.concatenate([plane["coords"][:, 0], start_xy[:, 0]])
    ys = np.concatenate([plane["coords"][:, 1], start_xy[:, 1]])
    extent = (xs.min() - pad * np.ptp(xs), xs.max() + pad * np.ptp(xs),
              ys.min() - pad * np.ptp(ys), ys.max() + pad * np.ptp(ys))
    contours = loss_grid(loss_fn, unravel, plane, extent, n=(120, 120))

    # Does a design actually sit on the plane it is drawn on?
    jac = coords_jacobian(statics["forward_fn"], sample, statics["centre"])
    mm_per_unit = axis_scales_mm(plane["basis"], jac)
    n_nodes = jac.shape[0] // 2
    resid_mm = float(np.median([np.linalg.norm(jac @ (f - plane["mean"] - c @ plane["basis"]))
                                / np.sqrt(n_nodes)
                                for f, c in zip(finals, plane["coords"])]))

    # The contours are exact ON the plane; whether they describe a design depends on how far
    # off the plane that design sits. Compare the on-plane value at each projected position
    # against the loss the design actually has.
    on_plane = np.asarray(jax.jit(jax.vmap(lambda vv: loss_fn(unravel(vv), 1.0)[0]))(
        jnp.asarray(plane["mean"][None, :] + plane["coords"] @ plane["basis"])))
    true_loss = np.array([float(r["loss"]) for r in records])
    loss_gap = float(np.median(np.abs(on_plane - true_loss) / true_loss))

    # How faithfully can each point of each path be placed on this plane? Compare the design's
    # own objective against the objective at its projected position. This is the quantity that
    # decides whether a stretch of path may be read against the contours at all.
    flat_all = np.concatenate([r["traj_params"] for r in records])
    proj_all = plane["mean"][None, :] + ((flat_all - plane["mean"]) @ plane["basis"].T
                                         ) @ plane["basis"]
    ev = jax.jit(jax.vmap(lambda vv: loss_fn(unravel(vv), 1.0)[0]))
    true_l, proj_l = [], []
    for i in range(0, len(flat_all), 1024):
        true_l.append(np.asarray(ev(jnp.asarray(flat_all[i:i + 1024]))))
        proj_l.append(np.asarray(ev(jnp.asarray(proj_all[i:i + 1024]))))
    gap = np.abs(np.concatenate(proj_l) - np.concatenate(true_l)) / np.concatenate(true_l)
    fid = np.clip(1.0 - gap / 0.25, 0.0, 1.0)          # faithful below a 25% disagreement
    n_snap = len(records[0]["traj_params"])
    fidelity = [fid[i * n_snap:(i + 1) * n_snap] for i in range(len(records))]

    # A contour set is exact on the plane it is drawn on, whichever plane that is. Whether it
    # may be READ against the data is a separate question, and it is decided by measurement:
    # if the designs themselves sit so far off the plane that its objective is nothing like
    # theirs, the contours describe a slice nobody in the panel lives on. Endpoint gap is
    # 1.8% on the PCA plane and ~790% on the discriminant plane, so one gets contours and the
    # other does not.
    if loss_gap > 0.10:
        contours = None

    pair, pair_gap, pair_typ = exemplar_starts(starts_p, labels)

    sheet_half = 0.5 * cfg.sheet_w
    tiles_end = [np.asarray(r["flat_tiles"]) - statics["centre"] for r in records]
    tiles_start = [np.asarray(r["flat_tiles_start"]) - statics["centre"] for r in records]

    # A page column at aspect 1.48 is height-bound on a 16:9 body and reaches little more than
    # half the measure. None of the three axes fixes a data aspect, so a flatter canvas simply
    # gives them more width.
    w_bp = float(width_bp) if width_bp else FIG_W_DEFAULT
    h_bp = w_bp / float(aspect)
    apply_charter(**paper_rc(scale))
    fig = plt.figure(figsize=figsize_in(w_bp, h_bp))

    # The left gutter holds the PC2 label, and paper_rc(scale) enlarges type without moving any
    # layout constant, so at scale 0.72 the label ran 13 bp off the canvas. Reserve the gutter in
    # proportion to the type instead of pinning it: identity at scale 1.0, so the page figure is
    # unchanged. The margin audit is what catches this if the coefficient is ever wrong.
    _gut = 0.090 * (1.0 / float(scale) - 1.0)
    ax_a = fig.add_axes([0.098 + _gut, 0.205, 0.470 - _gut, 0.745])
    ax_b = fig.add_axes([0.735, 0.645, 0.250, 0.320])
    ax_c = fig.add_axes([0.735, 0.205, 0.250, 0.320])

    panel_cloud(ax_a, plane["coords"], traj_xy, labels, tiles_end, tiles_start,
                start_xy, contours, sheet_half, scale,
                fidelity=fidelity if contours is not None else None, keep=pair)
    names = ("LD1", "LD2") if projection == "lda" else ("PC1", "PC2")

    ax_a.set_xlabel(names[0])
    ax_a.set_ylabel(names[1])

    panel_spectrum(ax_b, land["evals_shape"], land["evals_total"], land["pca_evals"],
                   int(land["n_flat"]), scale)

    steps = [np.asarray(r["traj_steps"], dtype=float) for r in records]
    errors = [np.asarray(r["traj_error"], dtype=float) for r in records]
    seeds = np.array([int(r["seed"]) for r in records])
    # The target radius, taken from the config actually deployed against rather than from the
    # summary, so the denominator cannot drift from the circle the panel is scored on.
    # `fixed_scale` is the target DIAMETER: target_unit has unit height.
    panel_convergence(ax_c, steps, errors, labels, seeds, pair, 0.5 * cfg.fixed_scale, scale)

    for ax, letter, x in ((ax_a, "a", -0.135), (ax_b, "b", -0.250), (ax_c, "c", -0.250)):
        panel_letter(ax, letter, x=x, y=1.03, scale=scale)

    pdf, lo, hi = save_paper_figure(fig, out_path, w_bp, h_bp, dpi=600, scale=scale)
    plt.close(fig)
    return pdf, {"kept_2d": plane["kept"], "off_plane_mm": resid_mm,
                 "path_faithful_frac": float(np.mean(fid > 0.5)),
                 "contours_drawn": contours is not None,
                 "highlighted_seeds": [int(records[k]["seed"]) for k in pair],
                 "pair_gap_vs_typical": [pair_gap, pair_typ],
                 "on_plane_loss_gap": loss_gap,
                 "mm_per_unit": mm_per_unit.tolist(), "family_sizes": fam["sizes"],
                 "separation_ratio": fam["separation_ratio"], "type_pt": [lo, hi]}


def caption_text(info):
    """The caption, written beside the figure so the two cannot drift apart."""
    a, b = info["family_sizes"]
    return (
        "Structure of the geometric solution set. Two hundred optimisations of one target, "
        "each from its own random starting pattern, with no physics anywhere. "
        "\\textbf{(a)}~The designs in the plane of the two leading principal components of "
        "the starting and converged patterns together. Black dots are the starting patterns, "
        "orange and teal the two families of converged designs, "
        f"{a} and {b} of them. Grey contours are the objective evaluated on this plane. "
        "Thin lines are the optimisation paths, drawn faint where the plane places a design "
        "too far from its own objective for the contours to describe it. Two runs are "
        "highlighted, one per family, with the cut pattern each began from and reached. "
        "\\textbf{(b)}~Curvature of the objective at a converged design, by design "
        "direction, for the shape term alone and with the regularity floors added, against "
        "the variance the ensemble occupies. The shaded band marks the directions the target "
        "leaves flat. \\textbf{(c)}~Outline error against optimisation step for the two "
        "highlighted runs, over their family medians.\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="ensemble run folder")
    ap.add_argument("--width-bp", type=float, default=None,
                    help="canvas width. Default is the page column")
    ap.add_argument("--aspect", type=float, default=1.48,
                    help="canvas width / height")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="printed width / authored width. Type is authored at target/scale so it PRINTS at the ladder size. A plate shown on a slide at 0.75 of its authored width takes its 8pt labels to 6pt unless this is set")
    ap.add_argument("--out", default=None)
    ap.add_argument("--paper-dir", default=None,
                    help="also write <dir>/Images/solution_set.{pdf,png,txt} for the "
                         "manuscript")
    ap.add_argument("--projection", default="pca", choices=("pca", "all", "lda"),
                    help="pca spans the converged spread; all spans starts and converged "
                         "designs together and separates every group while still carrying "
                         "the objective; lda is fitted to separate the groups and does not "
                         "carry it")
    args = ap.parse_args()
    default = {"pca": "design_cloud.pdf", "all": "design_cloud_all.pdf",
               "lda": "design_cloud_lda.pdf"}[args.projection]
    out = args.out or os.path.join(args.run, default)
    pdf, info = build_figure(args.run, out, projection=args.projection, scale=args.scale,
                             width_bp=args.width_bp, aspect=args.aspect)
    print(f"  wrote {pdf}")
    for k, v in info.items():
        print(f"    {k}: {v}")

    if args.paper_dir:
        import shutil
        images = os.path.join(args.paper_dir, "Images")
        os.makedirs(images, exist_ok=True)
        stem = os.path.join(images, "solution_set")
        for ext in (".pdf", ".png"):
            shutil.copyfile(os.path.splitext(pdf)[0] + ext, stem + ext)
        with open(stem + ".txt", "w") as fh:
            fh.write(caption_text(info))
        print(f"  wrote {stem}.pdf, .png and .txt")


if __name__ == "__main__":
    main()
