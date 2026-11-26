"""The range over which the sheet follows its ideal deployment, and what leaving it looks like.

    (a) two independent errors from the ideal deployment against relative hinge rotation, with one
        unit cell of the sheet drawn at four openings. They agree while the sheet is nearly shut and
        then part company: near full opening the non-rotational energy share reaches 17% while the
        panel-rotation residual stays near a thousandth of a per cent, so the sheet leaves the ideal
        deployment ENERGETICALLY while still LOOKING like it.
    (b) that positional error at the same opening, exaggerated, carrying two channels at once.
        Pale grey is the nearest ideal deployment, one scalar fitted to all 600 panel corners; the
        panels over it are the equilibrium with each panel's rigid offset and rotation from that ideal
        multiplied by ``AMPLIFY``, so their OFFSET from the grey is the error. Their COLOUR is the
        mean normal stress each panel carries. The two channels agree: force runs on diagonals from
        the grip to the two supports, and the error is a boundary layer on exactly those held
        panels, while the middle of the sheet is both unstressed and coincident. The supports and the
        pull are drawn, so the panel states the boundary-value problem it solves.

The U-curve panel is imported rather than copied: it is the same graphic as in the two-story plate,
and this project has already been burned by plotting code living in more than one place.

⚠⚠ ONE ANGLE. ``theta`` is the RELATIVE HINGE rotation; ``psi`` -- the per-panel rotation, half of it
-- appears nowhere.

⚠⚠ (b) is a THIRD measure of error -- position, as a fraction of panel pitch -- and it is some
2000x larger than (a)'s panel-rotation residual. It is not (a)'s teal number made visible, and it is
not a mode shape: it is the difference between two nonlinear solutions, so the factor is neither a
linearisation nor a multiple of the load. Every panel drawn is still a square; the one kinematic law
the exaggeration breaks is pin closure.
"""

import argparse
import json
import os

import numpy as np

from nff_paper.closed_support.one_dof import amplify_departure, fit_configuration, graph_distance
from nff_paper.closed_support.shield_kinematics import build_deploy_plan, pin_residual
from nff_paper.figures.plot_discussion_modes import (PATCH_COLS, PATCH_ROWS, ROOT, STATE_FILL,
                                                       WINDOW_FILL, one_percent_window,
                                                       panel_departures, panel_states, _read)
from nff.topology.closed_builder_jax import build_deploy_structure
from nff.utils import figstyle as FST

ASPECT = 2.376                  # one row of three inside \textwidth. The plate no longer overruns
                               # the text block, which cost (b) most of its width: the sheet is the
                               # widest thing here, so it is what had to give.
SLOPE_TOL = 0.05               # how far from 1/k the fitted slopes may sit before (c)'s claim fails
GHOST_ALPHA = 0.82             # the reference is GREY -- secondary annotation -- not a new colour.
                               # Safe as alpha: an ideal configuration has no overlapping panels, so
                               # nothing double-darkens.
AMPLIFY = 8                    # the factor DRAWN, and the number the caption cites
BOX_FILL = "#F7F5F2"           # the card fill of Figure 1, so a grouped row reads the same way
BOX_PAD = 5.0                  # points of clear space between a row's card and its content
ROW2_H = 144.0                 # points of canvas for the second row. Row 1 keeps its own budget
                               # untouched, so extending downward cannot move anything already
                               # approved: every row-1 box is still the same size in points.
POCKET_SWEEP = "data/outputs/pocket/sweep"
POCKET_RUN = "data/outputs/pocket/pocket_15"
DROP_YLIM = (1.5, 40.0)        # shared by (d) and (f). (f) MUST be pinned: its own data spans
                               # 8.70 to 8.72, and an autoscaled axis draws that 0.3% as a collapse.
TARGET_BP = 1.8                # printed separation the worst panel must reach to be seen at all.
                               # Lowered from 2.5 for the three-panel row, where the sheet gets 150 bp:
                               # cropping it to buy width back would cost the coincident middle and the
                               # force diagonals, which are the panel's content. 1.8 bp between a
                               # filled panel and its grey reference is legible in print.
FRAC_CAP = 0.45                # amplified error may not exceed this fraction of a panel pitch. Raised
                               # twice, both times because the sheet lost width: once to the
                               # boundary-condition glyphs, once to fit \textwidth. Whether 0.44 of a
                               # pitch still READS as a sheet is a question for the eye, not the number.
PAD_FRAC = 0.02
# Room for the glyphs, per side, as a fraction of the sheet's larger span: the supports and their
# ground bars sit off the LEFT edge, the load arrow (0.125 * span, plus its gap) off the RIGHT.
GLYPH_PAD = (0.21, 0.13, 0.03, 0.03)
BC_WEIGHT = 0.55               # glyph stroke scale; 1.0 is sized for a full-page loading diagram
# The problem, from figmat_1_deploy.yaml and sweep_localisation.loads_for('open', .): a PIN at panel 9
# (x and y fixed, rotation free), an x-ROLLER at panel 0, and one +x pull on panel 145. Three
# constraints against three rigid-body freedoms, with the mechanism's own freedom left free.
PIN_FACE, ROLLER_FACE, LOAD_FACE = 9, 0, 145


def panel_compliance(ax, mult, void, angle):
    """(c) Both errors against the slip stiffnesses, log-log, with a 1/k reference.

    The reference is anchored on the first void point, so "inverse proportion" is checkable by eye
    rather than taken from the caption. ``k_rot`` is deliberately NOT scaled: it is the mechanism's own
    stiffness, and scaling it would change which deployment the load produces rather than how far that
    deployment sits from the ideal one.
    """
    ref = void[0] * (mult / mult[0]) ** -1.0
    ax.plot(mult, ref, color=FST.GREY, lw=0.7, ls=(0, (2.4, 1.8)), zorder=1)
    for vals, col, lab in ((void, FST.INK, "void defect"),
                           (angle, FST.TEAL, "rotation residual")):
        ax.plot(mult, vals, "-", color=col, lw=1.2, solid_capstyle="round", zorder=3, label=lab)
        ax.plot(mult, vals, "o", color=col, ms=2.0, zorder=4,
                markeredgecolor="white", markeredgewidth=0.25)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$k/k^{\mathrm{ref}}$")
    ax.set_ylabel("relative error")
    ax.legend(frameon=False, fontsize=FST.FS_ANNOT, loc="upper right", handlelength=1.1,
              handletextpad=0.4, borderpad=0.12, labelspacing=0.22)


def read_drops(sweep_dir, arm):
    """The two drops per panel from one arm of the probe sweep, gated levels dropped."""
    with open(os.path.join(sweep_dir, f"{arm}.json")) as fh:
        doc = json.load(fh)
    rows = [r for r in doc["rows"] if "rejected" not in r]
    first = np.array([float(r["own"]["ratios"][0]) for r in rows])
    after = np.array([float(r["own"]["ratios"][1]) for r in rows])
    return doc, rows, first, after


def panel_drops(ax, x, first, after, log_x=False, uniformity=None):
    """(d) and (f). How far a load is felt, as the energy drop from one panel to the next.

    Two series on one axis on purpose. The first drop is what the loaded panel keeps; every drop
    after it is the rate the disturbance dies at. Reading them together is the result, because one
    of them moves with the control variable and the other does not.
    """
    ax.plot(x, first, "-o", color=FST.PURPLE, lw=1.2, ms=2.4, zorder=4,
            label=r"$\delta\mathcal{W}_0/\delta\mathcal{W}_1$")
    ax.plot(x, after, "--s", color=FST.GREY, lw=1.1, ms=2.2, zorder=3,
            label=r"$\delta\mathcal{W}_n/\delta\mathcal{W}_{n+1},\ n\geq 1$")
    if uniformity is not None:
        # The deployment's own uniformity, on the same axis and pointing the other way. Both are
        # dimensionless ratios of hinge rotations, so they share an axis honestly. Opening the sheet
        # spreads the deployment out towards one while pulling a new disturbance in.
        ax.plot(uniformity[0], uniformity[1], "-", color=FST.INK, lw=1.3, zorder=2,
                label=r"$\vartheta_{\max}/2|\phi|$")
        ax.axhline(1.0, color=FST.GREY, lw=0.7, ls=(0, (1.4, 1.8)), zorder=1)
    if log_x:
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(*DROP_YLIM)
    ax.set_ylabel(r"$\delta\mathcal{W}_n/\delta\mathcal{W}_{n+1}$")
    ax.legend(frameon=False, fontsize=FST.FS_ANNOT, loc="upper left", handlelength=1.1,
              handletextpad=0.4, borderpad=0.12, labelspacing=0.22)


def level_sheet(nodes, centroids, M, N):
    """Rotate a drawn sheet so its panel grid is square to the page.

    The sheet in (a) is held by a pin and a roller and cannot turn. This one is held by a single
    restrained panel, which leaves the whole sheet free to rotate about it, so the deployed geometry
    arrives at whatever angle the mechanism settled at. That angle carries no information and reads
    as a drawing error beside (a), so it is taken out. Only a rigid rotation is applied; no distance
    in the figure changes.
    """
    xy = np.asarray(centroids, dtype=float)
    first = xy[0 * int(N):1 * int(N)].mean(axis=0)
    last = xy[(int(M) - 1) * int(N):int(M) * int(N)].mean(axis=0)
    ang = np.arctan2(*(last - first)[::-1])
    c, s_ = np.cos(-ang), np.sin(-ang)
    rot = np.array([[c, -s_], [s_, c]])
    return np.asarray(nodes, dtype=float) @ rot.T


def panel_pocket(ax, nodes, energy, floor=3e-3, bcs=None):
    """(d) The sheet, each panel shaded by the ligament energy a central load adds to it.

    The same graphic as (a) asking the other question. There the sheet is drawn around its supports
    and the error is positional; here it is drawn around a load whose supports carry nothing, and the
    shading is energy.

    The colour scale is logarithmic and floored above the interior level. The field falls by orders
    of magnitude within a few panels, so a linear ramp would leave one bright panel on a flat ground
    and a floor at the interior level would paint the whole sheet mid-scale.
    """
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    tiles = np.asarray(nodes, dtype=float)
    v = np.asarray(energy, dtype=float)
    v = np.clip(v / v.max(), floor, 1.0)
    # Purple, because (a) spends teal and orange on compression and tension. A sequential ramp
    # in either of those hues reads as a stress field, which this is not.
    cmap = LinearSegmentedColormap.from_list("charter_seq", ["#FFFFFF", FST.PURPLE, "#2E1B45"])
    faces = PolyCollection(list(tiles), cmap=cmap, norm=LogNorm(vmin=floor, vmax=1.0),
                           edgecolors=STATE_FILL, linewidths=0.16, zorder=3)
    faces.set_array(v)
    ax.add_collection(faces)
    if bcs is not None:
        # The panel states the problem it solves, as (a) does. The support keeps its glyph; the load
        # does not. An arrow lands on the darkest panels of the field, where neither black nor red
        # reads, so the loaded panels are outlined instead. An outline cannot be lost on any ground.
        from matplotlib.collections import PolyCollection as _PC
        from nff.utils.visualization import draw_supports_and_loads, faces_as_tessellation
        loaded = [int(f) for f in bcs.pop("loaded_faces", [])]
        bcs.pop("load_specs", None)
        draw_supports_and_loads(ax, faces_as_tessellation(tiles), load_specs=[], **bcs)
        if loaded:
            ax.add_collection(_PC([tiles[i] for i in loaded], facecolors="none",
                                  edgecolors=FST.RED, linewidths=0.9, zorder=6))

    ax.set_aspect("equal")
    ax.axis("off")
    (x0, x1), (y0, y1), _ = view_of(tiles)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    return faces


def panel_reach_vs_stiffness(ax, mult, lam):
    """(f) The reach of a load against the ligament stiffness, beside (c)'s error against the same.

    Deliberately the same axis and the same range as (c), because the pair is the statement. There
    the two error measures fall as the inverse of the stiffness over three decades. Here the length
    the disturbance decays over does not move at all. The contrast between the mechanism and the
    ligaments sets how large a disturbance is and not how far it reaches, and only reading the two
    panels together says that.
    """
    ax.plot(mult, lam, "-o", color=FST.INK, lw=1.2, ms=2.6, zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(0.22, 420.0)
    ax.set_ylim(0.0, 2.0)
    ax.set_xlabel(r"$k/k^{\mathrm{ref}}$")
    ax.set_ylabel(r"$\lambda$  (panels)")


def grid_hinges(M, N):
    """Panel adjacency of the closed sheet: the four-neighbour grid, face id ``col * N + row``.

    The count matches the builder's own, ``2MN - M - N``, which is exactly the number of edges in
    that grid, so the two agree without needing the solved bond list.
    """
    pairs = []
    for i in range(int(M)):
        for j in range(int(N)):
            k = i * int(N) + j
            if i + 1 < int(M):
                pairs.append((k, (i + 1) * int(N) + j))
            if j + 1 < int(N):
                pairs.append((k, i * int(N) + j + 1))
    return np.asarray(pairs, dtype=np.int64)


def pin_thetas(ax, tiles, pairs, phi, picks=3, colour=None, scale=1.0):
    """Mark a few hinges with their own relative rotation, spread over the sheet.

    One number for the whole sheet hides whether the opening is shared. Marking the widest, the
    narrowest and a median hinge says in three numbers whether the tessellation opens together.
    Hinges are chosen by value and then by distance from each other, so the three do not land in one
    corner and describe only that corner.
    """
    centres = np.asarray(tiles, dtype=float).mean(axis=1)
    th = np.degrees(np.abs(phi[pairs[:, 1]] - phi[pairs[:, 0]]))
    order = [int(np.argmin(th)), int(np.argsort(th)[len(th) // 2]), int(np.argmax(th))]

    chosen, used = [], []
    for idx in order:
        mid = 0.5 * (centres[pairs[idx, 0]] + centres[pairs[idx, 1]])
        if all(np.hypot(*(mid - u)) > 0.18 * np.ptp(centres[:, 0]) for u in used):
            used.append(mid)
            chosen.append((idx, mid))
    for idx, mid in chosen[:picks]:
        ax.plot([mid[0]], [mid[1]], "o", ms=2.2, color=colour or FST.RED, zorder=10)
        ax.annotate(rf"${th[idx]:.1f}^\circ$", mid, textcoords="offset points", xytext=(4, 4),
                    ha="left", va="bottom", zorder=10,
                    fontsize=FST.printed_pt(FST.FS_MICRO, scale), color=colour or FST.RED,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                              boxstyle="round,pad=0.12"))
    return [th[i] for i, _ in chosen[:picks]]


def view_of(*tile_sets, pad=None):
    """The padded view holding every panel set, and its width / height.

    ``pad`` is ``(left, right, bottom, top)`` as fractions of the larger span, defaulting to
    ``PAD_FRAC`` on every side. It is asymmetric because the boundary-condition glyphs are: the
    support code draws them OUTSIDE the material, and in this problem the supports sit on the left
    edge and the load arrow on the right. Padding all four sides to the widest requirement would
    shrink the sheet, and the sheet's printed size is what makes the exaggeration visible at all.
    """
    pts = np.concatenate([np.asarray(t).reshape(-1, 2) for t in tile_sets])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = float(np.max(hi - lo))
    left, right, bottom, top = pad if pad is not None else (PAD_FRAC,) * 4
    xlim = (lo[0] - left * span, hi[0] + right * span)
    ylim = (lo[1] - bottom * span, hi[1] + top * span)
    return xlim, ylim, (xlim[1] - xlim[0]) / (ylim[1] - ylim[0])


def panel_amplified(ax, ideal, amplified, stress, cell=(), lw=0.25, bcs=None):
    """(b) The reference filled pale, the exaggerated equilibrium laid over it and shaded by stress.

    Two channels, one drawing. The OFFSET between the pale reference and the tiles over it is the
    kinematic error; the COLOUR of those tiles is the force each carries. The colour sits on the
    exaggerated tiles rather than on the reference because the force belongs to the equilibrium, not
    to the ideal mechanism it is compared against.

    The stress is signed, so the map has to be diverging about zero: teal in compression and orange in
    tension, the same reading as everywhere else in this project. Symmetric limits, so the two signs
    share one scale and compression is allowed to look as small as it is.

    Unlike ``panel_states`` this panel DRAWS panel edges. There, an edge colour separated panels at
    the corners they physically touch and an opened sheet read as loose diamonds; here the exaggerated
    sheet genuinely is loose at its pins, and that is the measurement.
    """
    from matplotlib.collections import PolyCollection

    ax.add_collection(PolyCollection(list(ideal), facecolors=FST.GREY, edgecolors="none",
                                     alpha=GHOST_ALPHA, zorder=2))
    faces = PolyCollection(list(amplified), edgecolors=STATE_FILL, linewidths=lw, zorder=4,
                           cmap=FST.charter_diverging(), clim=(-1.0, 1.0))
    faces.set_array(np.asarray(stress))
    ax.add_collection(faces)
    if len(cell):
        # THE SAME four panels drawn on the curve in (a), filled the same black. Four of 150 give up
        # their stress colour so that the reader can find, in the whole sheet, the unit cell they have
        # been reading along the curve. The supports need no extra mark: their glyphs point at them.
        ax.add_collection(PolyCollection([amplified[k] for k in cell], facecolors=STATE_FILL,
                                         edgecolors=STATE_FILL, linewidths=lw, zorder=5))
    if bcs:
        # The sanctioned glyph code, placement rules and all. Its own docstring forbids re-deriving
        # them per figure, so the wall's normal, the arrow's side and the rollers come from there.
        from nff.utils.visualization import draw_supports_and_loads, faces_as_tessellation
        draw_supports_and_loads(ax, faces_as_tessellation(amplified), **bcs)

    xlim, ylim, _ = view_of(ideal, amplified, pad=GLYPH_PAD)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    return faces


def tile_rotations(flat, nodes):
    """(n_panels,) rotation of each rigid panel, from one of its edges. Radians, wrapped."""
    def edge_angle(t):
        d = np.asarray(t)[:, 1, :] - np.asarray(t)[:, 0, :]
        return np.arctan2(d[:, 1], d[:, 0])
    return (edge_angle(nodes) - edge_angle(flat) + np.pi) % (2 * np.pi) - np.pi


def load_tile_stress(run, level, flat, actual, tol=1e-9):
    """Per-panel mean normal stress at the drawn state, normalised by its own peak.

    The stored states carry no forces and the sweep carries no node positions, so the force field has
    to be paired to the drawing by load level. That pairing is only sound if the two files record the
    SAME solve, which is checked rather than assumed: every panel's rotation must agree.

    Normalised because the stress is a force per unit length in the reduced model's own units, on a
    sheet whose pitch is 1 -- there is no honest physical label to put on the bar. The spatial
    distribution is what the panel is for.
    """
    z = np.load(os.path.join(ROOT, run, "sweep.npz"))
    levels = np.asarray(z["levels"])
    i = int(np.argmin(np.abs(levels - level)))
    assert abs(levels[i] - level) <= 1e-6 * max(abs(level), 1.0), (
        f"the sweep has no level matching the drawn state's {level:g}; nearest is {levels[i]:g}")
    off = float(np.abs(tile_rotations(flat, actual) - np.asarray(z[f"theta__{i}"])).max())
    assert off < tol, (f"the sweep's level-{levels[i]:g} solve differs from the drawn state by "
                       f"{off:.2e} rad per panel, so its forces belong to a different equilibrium")
    stress = np.asarray(z[f"pressure__{i}"])
    return stress / np.abs(stress).max(), stress


def load_departure(states_path, index, rows=10):
    """The stored equilibrium, the ideal 1-DOF configuration nearest it, and how far apart they are.

    ``deploy_tiles`` reads only the tile count and the pin graph, never the pitch, so the ideal is
    rebuilt from the stored flat corners alone -- no config, no re-solve. Its pins close to ~1e-14,
    which is the check that the reconstruction really is a mechanism configuration.
    """
    st = np.load(states_path)
    k = range(int(st["levels"].size))[index]
    flat, actual = np.asarray(st[f"flat__{k}"]), np.asarray(st[f"nodes__{k}"])
    n_tiles = actual.shape[0]
    cols = n_tiles // rows
    plan = build_deploy_plan(build_deploy_structure(cols, rows), cols, rows)

    fit = fit_configuration(plan, flat, actual)
    ideal = np.asarray(fit["nodes"])
    pitch = float(np.linalg.norm(flat[rows].mean(axis=0) - flat[0].mean(axis=0)))
    assert np.abs(np.asarray(pin_residual(plan, ideal))).max() < 1e-9 * pitch, \
        "the reconstructed reference does not close its pins, so it is not a 1-DOF configuration"

    dev = np.linalg.norm(actual - ideal, axis=-1) / pitch
    return {"plan": plan, "ideal": ideal, "actual": actual, "flat": flat, "pitch": pitch, "dev": dev,
            "level": float(st["levels"][k]), "theta_deg": 2.0 * abs(np.degrees(fit["psi"])),
            "rmse": fit["rmse"] / pitch, "worst_tile": int(np.argmax(dev.max(axis=1))),
            "dev_max": float(dev.max())}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--energy", default="energy_open_squares")
    ap.add_argument("--sweep", default="open_squares", help="run holding the paired force field")
    ap.add_argument("--compliance", default="compliance")
    ap.add_argument("--states", default=os.path.join(ROOT, "insets", "two_states.npz"))
    ap.add_argument("--out", default=os.path.join(ROOT, "onedof_validity"))
    ap.add_argument("--lo-edge", type=float, default=None)
    ap.add_argument("--hi-edge", type=float, default=None)
    ap.add_argument("--n-states", type=int, default=4)
    ap.add_argument("--state-index", type=int, default=-1,
                    help="which stored level panel (b) draws; -1 is the most open")
    ap.add_argument("--amplify", type=float, default=float(AMPLIFY))
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--pocket-sweep", default=POCKET_SWEEP,
                    help="directory holding deployment.json and stiffness.json for row 2")
    ap.add_argument("--pocket-run", default=POCKET_RUN,
                    help="probe_pocket run holding pocket.npz for the sheet drawing of (e)")
    args = ap.parse_args()

    FST.use_agg()
    FST.apply_charter()
    import matplotlib.pyplot as plt

    en = _read(args.energy, "energy_partition.json")["rows"]
    theta = np.degrees([r['theta_mean'] for r in en])
    energy_pct = 100 * np.array([r['non_rotational'] for r in en])
    shape_pct = 100 * np.maximum(1.0 - np.array([r['r2'] for r in en]), 1e-8)

    dp = load_departure(args.states, args.state_index)
    amplified = amplify_departure(dp["ideal"], dp["actual"], args.amplify)
    amp_dev = float(np.linalg.norm(amplified - dp["ideal"], axis=-1).max() / dp["pitch"])
    gap = float(np.abs(np.asarray(pin_residual(dp["plan"], amplified))).max() / dp["pitch"])
    stress, stress_raw = load_tile_stress(args.sweep, dp["level"], dp["flat"], dp["actual"])

    st = np.load(args.states)
    keep = np.array([0, 1, 2, int(st["levels"].size) - 1])[:args.n_states]
    nodes_list = [st[f"nodes__{k}"] for k in keep]
    state_theta = np.interp(np.asarray(st["levels"])[keep], [r['level'] for r in en], theta)

    cp = _read(args.compliance, "compliance.json")["rows"]
    order = np.argsort([r['mult'] for r in cp])
    mult = np.array([r['mult'] for r in cp])[order]
    void = np.array([r['para_mean'] for r in cp])[order]
    # Stored against the per-panel rotation; halve it to read against theta = 2 phi. Slope unchanged.
    resid = np.array([r['angle_resid'] for r in cp])[order] / 2.0

    # ── row 2 inputs: the probe sweeps and one drawn state ──────────────────
    _, dep_rows, dep_first, dep_after = read_drops(args.pocket_sweep, "deployment")
    dep_theta = np.array([r["theta"]["mean"] for r in dep_rows])
    _, stf_rows, stf_first, stf_after = read_drops(args.pocket_sweep, "stiffness")
    stf_mult = np.array([r["stiffness_factor"] for r in stf_rows])
    stf_peak = np.array([float(r["own"]["shells"][0]) for r in stf_rows])
    stf_lam = np.array([float(r["own"]["decay_length"]) for r in stf_rows])
    pk = np.load(os.path.join(args.pocket_run, "pocket.npz"))
    with open(os.path.join(args.pocket_run, "summary.json")) as fh:
        pk_meta = json.load(fh)["sheet"]
    pk_dist = graph_distance(pk["face_pairs"], int(pk["centroids"].shape[0]),
                             [int(pk_meta["centre_face"])])
    # The deployment's own uniformity, from the open sweep that (a) and (b) already come from.
    loc_rows = _read(args.sweep, "summary.json")["rows"]
    loc_theta = np.array([2.0 * abs(np.degrees(r["psi"])) for r in loc_rows])
    loc_ratio = np.array([float(r["localisation"]) for r in loc_rows])
    loc_order = np.argsort(loc_theta)
    loc_theta, loc_ratio = loc_theta[loc_order], loc_ratio[loc_order]

    lo_w, hi_w = one_percent_window(theta, energy_pct)
    if args.lo_edge is not None and args.hi_edge is not None:
        lo_w, hi_w = args.lo_edge, args.hi_edge

    w_bp, h_top = FST.page_bp(1.0, aspect=ASPECT)
    assert w_bp > h_top, "row 1 must be landscape; page_bp's aspect is WIDTH/HEIGHT"
    # The canvas grows DOWNWARD. Every row-1 box keeps its size and its distance from the top, so
    # the approved row renders identically in points; only its normalised origin changes.
    h_bp = h_top + ROW2_H
    # Boxes in points, then normalised. Each plotted panel needs its own left gutter for a rotated
    # label over log tick labels; (b) is axis-off and needs none. Nothing here crops to content.
    # The sheet leads each row: the object first, then what was measured on it. The two plot
    # columns each keep a left gutter for a rotated label over log ticks; the sheets are axis-off.
    # `left` is clear space between the card wall and the drawing, not just a margin: the
    # sheets are axis-off, so their ink starts exactly at the axes edge.
    left, w_b, gut, w_a = 15.0, 118.0, 42.0, 185.0
    right = 9.0
    bot_a, top_a = 34.0, 34.0
    h_a = h_top - bot_a - top_a
    y_a = bot_a + ROW2_H                       # row 1's origin on the taller canvas
    bot_2, top_2 = 34.0, 12.0
    h_2 = ROW2_H - bot_2 - top_2

    with plt.rc_context(FST.paper_rc(scale=args.scale)):
        fig = plt.figure(figsize=FST.figsize_in(w_bp, h_bp))
        x_a = left + w_b + gut
        ax_a = fig.add_axes([x_a / w_bp, y_a / h_bp, w_a / w_bp, h_a / h_bp])
        panel_departures(ax_a, theta, energy_pct, shape_pct, args.lo_edge, args.hi_edge,
                         headroom=400.0)
        ax_a.set_xticks([0, 30, 60, 90])
        ax_a.set_xlabel(r"$\vartheta$  (deg)")
        # Teal is spent on compression in (a); the kinematic error takes ink instead.
        for line in ax_a.get_lines():
            if line.get_color() == FST.TEAL:
                line.set_color(FST.INK)
        ax_a.legend(frameon=False, fontsize=FST.FS_ANNOT, loc="lower left", handlelength=1.2,
                    handletextpad=0.4, borderpad=0.15, labelspacing=0.25)

        # Size (b) to the drawing's own aspect so an equal-aspect axes does not letterbox, and hang
        # it from (a)'s top edge so the panel letters sit at one height.
        x_b, cb_block = left, 20.0
        h_b = min(w_b / view_of(dp["ideal"], amplified, pad=GLYPH_PAD)[2], h_a - cb_block)
        # The drawing and its colorbar are one block, centred in the row. Hung from the top it left
        # all its slack underneath and read as sitting high in its own panel.
        y_block = y_a + max(h_a - h_b - cb_block, 0.0) / 2.0
        y_sheet = y_block + cb_block
        ax_b = fig.add_axes([x_b / w_bp, y_sheet / h_bp, w_b / w_bp, h_b / h_bp])
        cell = [c * 10 + r for c in PATCH_COLS for r in PATCH_ROWS]
        bcs = dict(clamped_faces=[PIN_FACE], clamped_dofs=[0, 1], x_roller_faces=[ROLLER_FACE],
                   load_specs=[{'face': LOAD_FACE, 'dof': 0, 'value': dp["level"]}],
                   weight_scale=BC_WEIGHT)
        faces = panel_amplified(ax_b, dp["ideal"], amplified, stress, cell=cell, bcs=bcs)
        pinned_a = pin_thetas(ax_b, amplified, grid_hinges(15, 10),
                              tile_rotations(dp["flat"], dp["actual"]), scale=args.scale)

        # The compliance panel and its counterpart in row 2 were removed at the author's request.
        # The slope check stays: the body text still quotes -0.97 and -0.99, so the guard that
        # those numbers are what the data says must outlive the panel that drew them.
        slopes = {}
        for name, vals in (("void defect", void), ("rotation residual", resid)):
            slopes[name] = float(np.polyfit(np.log10(mult), np.log10(vals), 1)[0])
            assert abs(slopes[name] + 1.0) < SLOPE_TOL, (
                f"{name} scales as k^{slopes[name]:+.3f}, not 1/k -- the panel's claim does not hold "
                "and the caption must change with the data")

        cax = fig.add_axes([(x_b + 0.26 * w_b) / w_bp, (y_sheet - 12.0) / h_bp,
                            0.48 * w_b / w_bp, 3.8 / h_bp])
        cb = fig.colorbar(faces, cax=cax, orientation="horizontal", ticks=[-1, 0, 1])
        cb.set_label(r"$\bar{\sigma}/\bar{\sigma}_{\max}$", fontsize=FST.printed_pt(FST.FS_ANNOT,
                                                                                    args.scale),
                     labelpad=1.5)
        cax.tick_params(labelsize=FST.printed_pt(FST.FS_ANNOT, args.scale), length=1.6, pad=1.2)
        cb.outline.set_linewidth(0.4)

        # The worst-departing panel must be IN VIEW, or the drawing quietly understates the sheet.
        (x0, x1), (y0, y1) = ax_b.get_xlim(), ax_b.get_ylim()
        worst = dp["ideal"][dp["worst_tile"]]
        assert (x0 <= worst[:, 0]).all() and (worst[:, 0] <= x1).all() and \
               (y0 <= worst[:, 1]).all() and (worst[:, 1] <= y1).all(), \
            f"tile {dp['worst_tile']} carries the largest error and is outside the drawn view"

        bp_per_pitch = w_b / ((x1 - x0) / dp["pitch"])
        a_vis = TARGET_BP / (dp["dev_max"] * bp_per_pitch)
        a_cap = FRAC_CAP / dp["dev_max"]
        assert a_vis <= a_cap, (
            f"no factor is both visible and plausible at {w_b:.0f} bp: {a_vis:.1f}x is needed to "
            f"print {TARGET_BP} bp but {a_cap:.1f}x already puts a panel {FRAC_CAP:.0%} of a pitch "
            "out of place")
        assert abs(round(min(a_vis, a_cap)) - args.amplify) < 1e-9, (
            f"the data now wants {min(a_vis, a_cap):.2f}x, not {args.amplify:g}x -- the caption "
            "cites the factor, so it must change with the data")

        # Each card says what its row asks. The two questions are the figure's structure, and the
        # caption should not be the only place they are stated.
        for y_top, title in ((y_a + h_a, "Error from the ideal deployment"),
                             (bot_2 + h_2, "Range of a local load")):
            fig.text(0.5, (y_top + 10.0) / h_bp, title, ha="center", va="bottom",
                     fontsize=FST.printed_pt(FST.FS_BASE, args.scale), color=FST.INK)

        for letter, x0 in (("b", x_a), ("a", x_b + 6.0)):
            FST.panel_letter(fig, letter, x=max(x0 / w_bp - 0.028, (BOX_PAD + 3.0) / w_bp),
                             y=(y_a + h_a) / h_bp + 0.012, scale=args.scale)

        # ── row 2: how far a local load is felt, on the same three controls ──
        ax_d = fig.add_axes([x_a / w_bp, bot_2 / h_bp, w_a / w_bp, h_2 / h_bp])
        panel_drops(ax_d, dep_theta, dep_first, dep_after)
        ax_d.set_xlim(0, 90)
        ax_d.set_xticks([0, 30, 60, 90])
        ax_d.set_xlabel(r"$\vartheta$  (deg)")
        ax_d.axvspan(lo_w, hi_w, color=WINDOW_FILL, lw=0, zorder=0)

        ax_e = fig.add_axes([x_b / w_bp, bot_2 / h_bp, w_b / w_bp, h_2 / h_bp])
        probe_bcs = dict(clamped_faces=[0], clamped_dofs=[0, 1, 2], weight_scale=BC_WEIGHT,
                         loaded_faces=[int(pk_meta["centre_face"]), int(pk_meta["partner_face"])])
        pocket_faces = panel_pocket(ax_e, level_sheet(pk["nodes"], pk["centroids"],
                                                      pk_meta["M"], pk_meta["N"]), pk["force__0"],
                                    bcs=probe_bcs)
        pinned_c = pin_thetas(ax_e, level_sheet(pk["nodes"], pk["centroids"], pk_meta["M"],
                                                pk_meta["N"]), pk["face_pairs"],
                              np.asarray(pk["background_displacement"])[:, 2], scale=args.scale)

        cax2 = fig.add_axes([(x_b + 0.26 * w_b) / w_bp, (bot_2 - 9.0) / h_bp,
                             0.48 * w_b / w_bp, 3.8 / h_bp])
        cb2 = fig.colorbar(pocket_faces, cax=cax2, orientation="horizontal")
        cb2.set_label(r"$\delta\mathcal{W}/\delta\mathcal{W}_{\max}$",
                      fontsize=FST.printed_pt(FST.FS_ANNOT, args.scale), labelpad=1.5)
        cax2.tick_params(labelsize=FST.printed_pt(FST.FS_ANNOT, args.scale), length=1.6, pad=1.2)
        cb2.outline.set_linewidth(0.4)

        for letter, x0 in (("d", x_a), ("c", x_b + 6.0)):
            FST.panel_letter(fig, letter, x=max(x0 / w_bp - 0.028, (BOX_PAD + 3.0) / w_bp),
                             y=(bot_2 + h_2) / h_bp + 0.012, scale=args.scale)

        # ── one card per row ────────────────────────────────────────────────
        # The two rows ask different questions of the same sheet, and nothing but vertical
        # whitespace said so. The card is Figure 1's, at its lightest: a fill this pale reads as
        # grouping and never as data, which matters here because (a) and (d) are drawings on white.
        from matplotlib.patches import FancyBboxPatch
        # A contains its own x label, B its row title, and 67 - A - B is the gap between the two
        # cards. Keep that positive and visible or the rows read as one block.
        for y0_bp, y1_bp in ((bot_2 - 32.0, bot_2 + h_2 + 19.0),
                             (y_a - 26.0, y_a + h_a + 19.0)):
            fig.add_artist(FancyBboxPatch(
                (BOX_PAD / w_bp, y0_bp / h_bp),
                (w_bp - 2 * BOX_PAD) / w_bp, (y1_bp - y0_bp) / h_bp,
                boxstyle="round,pad=0,rounding_size=0.012",
                transform=fig.transFigure, facecolor=BOX_FILL, edgecolor=FST.GREY,
                lw=0.4, alpha=0.55, zorder=-3))

        # Lifted clear of the curve. The unit cell is four panels around a void, so a curve
        # passing behind it shows through the middle and reads as cutting the glyph.
        # Numbers on the graphic rather than in the caption: the state drawn, the exaggeration, the
        # band's extent, the sheet, and the two ends of the first-drop curve. No prose on a panel.
        # The sheet fills its axes, so these sit on a white ground rather than on the lattice.
        _ground = dict(facecolor="white", edgecolor="none", alpha=0.88,
                       boxstyle="round,pad=0.18")
        ax_b.text(0.995, 0.99, r"$\vartheta=87.2^\circ$", transform=ax_b.transAxes, ha="right",
                  va="top", fontsize=FST.printed_pt(FST.FS_ANNOT, args.scale), color=FST.INK,
                  bbox=_ground, zorder=9)
        # Named once, in the panel where it is defined. The same band is drawn in (d) and carries
        # its meaning across without a second label.
        _top = ax_a.get_ylim()[1]
        ax_a.text(0.5 * (lo_w + hi_w), _top * 0.42, "validity band", ha="center", va="top",
                  fontsize=FST.printed_pt(FST.FS_ANNOT, args.scale), color=FST.INK)
        ax_a.text(0.5 * (lo_w + hi_w), _top * 0.10, rf"${lo_w:.1f}$–${hi_w:.1f}^\circ$",
                  ha="center", va="top",
                  fontsize=FST.printed_pt(FST.FS_ANNOT, args.scale), color=FST.INK)
        ax_a.text(89.0, 1.0, r"$1\%$", ha="right", va="bottom",
                  fontsize=FST.printed_pt(FST.FS_ANNOT, args.scale), color=FST.GREY)
        # Same corner, same style as (a): the two sheets are drawn at different openings, and that
        # only reads as a comparison if the two labels sit in the same place.
        ax_e.text(0.995, 0.99, r"$\vartheta=43.8^\circ$", transform=ax_e.transAxes, ha="right",
                  va="top", fontsize=FST.printed_pt(FST.FS_ANNOT, args.scale), color=FST.INK,
                  bbox=_ground, zorder=9)
        for xx, yy, ha in ((dep_theta[0], dep_first[0], "left"),
                           (dep_theta[-1], dep_first[-1], "right")):
            ax_d.annotate(f"${yy:.1f}$", (xx, yy), textcoords="offset points",
                          xytext=(3 if ha == "left" else -3, -4), ha=ha, va="top",
                          fontsize=FST.printed_pt(FST.FS_ANNOT, args.scale), color=FST.PURPLE)

        state_axes = panel_states(fig, ax_a, nodes_list, state_theta, theta, energy_pct,
                                  width_frac=0.11, gap=0.022)
        assert len({(a.get_xlim(), a.get_ylim()) for a in state_axes}) == 1, \
            "the states must share one scale or the progression is an artifact"

        # Which opening (b) draws: a ring on that opening's own point on the curve. The four black
        # panels inside (b) are the same unit cell drawn along the curve, so the correspondence is
        # carried by the drawing itself and needs no leader line.
        assert abs(state_theta[-1] - dp["theta_deg"]) < 1.5, (
            f"the last inset is at theta {state_theta[-1]:.1f} deg but (b) draws {dp['theta_deg']:.1f} "
            "deg, so marking that point as (b)'s state would assert something false")
        ax_a.plot([state_theta[-1]], [np.interp(state_theta[-1], theta, energy_pct)], "o",
                  ms=5.0, markerfacecolor="white", markeredgecolor=STATE_FILL,
                  markeredgewidth=0.9, zorder=8)

        pdf, lo_pt, hi_pt = FST.save_paper_figure(fig, args.out, w_bp, h_bp, scale=args.scale,
                                                  strict_margins=True)

    print(f"[plot] {pdf}   {w_bp:.1f} x {h_bp:.1f} bp   aspect {w_bp/h_bp:.2f} landscape")
    print(f"[plot] type {lo_pt:.1f}-{hi_pt:.1f} pt printed")
    i = int(np.argmin(energy_pct))
    lo_w, hi_w = one_percent_window(theta, energy_pct)
    print(f"[plot] (a) below 1% from theta {lo_w:.1f} to {hi_w:.1f} deg")
    print(f"[plot] (a) {len(theta)} levels; window min {energy_pct[i]:.2f}% at theta {theta[i]:.1f} "
          f"deg; at theta {theta[-1]:.0f} deg work {energy_pct[-1]:.1f}% vs shape {shape_pct[-1]:.2e}%")
    print(f"[plot] (a) states at theta = " + ", ".join(f"{t:.0f}" for t in state_theta) + " deg, "
          f"{len(nodes_list)} of them")
    print(f"[plot] (b) theta {dp['theta_deg']:.1f} deg, {dp['ideal'].shape[0]} panels, "
          f"{w_b:.0f} bp wide at {bp_per_pitch:.1f} bp per pitch")
    print(f"[plot] (b) error {dp['dev_max']*100:.2f}% of pitch, worst on tile "
          f"{dp['worst_tile']}; rms {dp['rmse']*100:.2f}%")
    print(f"[plot] (b) drawn {args.amplify:g}x -> {amp_dev*100:.1f}% of pitch, "
          f"{amp_dev*bp_per_pitch:.2f} bp printed; amplified pin gap {gap*100:.1f}% of pitch")
    print(f"[plot] (b) factor window: visible from {a_vis:.2f}x, plausible to {a_cap:.2f}x")
    print(f"[plot] (b) panel stress {stress_raw.min():+.1f} to {stress_raw.max():+.1f} in the model's "
          f"own units; compression reaches {100*abs(stress.min()):.0f}% of the tensile peak")
    print(f"[plot] (c) {len(mult)} stiffness levels over "
          f"{np.log10(mult.max()/mult.min()):.1f} decades")
    for name, sl in slopes.items():
        print(f"[plot] (c) {name}: log-log slope {sl:+.4f}")
    print(f"[plot] (d) {len(dep_theta)} openings, theta {dep_theta.min():.1f}-{dep_theta.max():.1f} "
          f"deg; first drop {dep_first.min():.2f} to {dep_first.max():.2f}, "
          f"later drops {dep_after.min():.2f} to {dep_after.max():.2f}")
    print(f"[plot] (d) uniformity {loc_ratio.max():.3f} at theta "
          f"{loc_theta[np.argmax(loc_ratio)]:.1f} deg falling to {loc_ratio[-1]:.3f}")
    print(f"[plot] (a) hinges marked at " + ", ".join(f"{v:.1f}" for v in pinned_a) + " deg")
    print(f"[plot] (c) hinges marked at " + ", ".join(f"{v:.1f}" for v in pinned_c) + " deg")
    print(f"[plot] (c) {pk['nodes'].shape[0]} panels, {pk_meta['M']}x{pk_meta['N']} sheet, "
          f"probe on {pk_meta['centre_face']} and {pk_meta['partner_face']}")
    print(f"[plot] (f) stiffness x{stf_mult.min():g} to x{stf_mult.max():g}, {len(stf_mult)} levels: "
          f"lambda {stf_lam.min():.3f}-{stf_lam.max():.3f} panels "
          f"({100*(stf_lam.max()/stf_lam.min()-1):.1f}% spread); peak energy falls "
          f"{stf_peak.max()/stf_peak.min():.0f}x over the same range")


if __name__ == "__main__":
    main()
