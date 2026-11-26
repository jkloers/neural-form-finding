"""Measure a photographed kirigami deployment against the simulation that designed it.

Three photographs of one sheet -- closed, part deployed, deployed -- are rectified into the
simulation's own millimetre frame, and the nine known rigid tiles are posed against the tape until
their edges lie on it. What comes out is a measured ``(dx, dy, dtheta)`` per tile: the same degrees
of freedom Stage 2 solves for, so measurement and prediction are directly subtractable.

THE METRIC ANCHOR IS THE SHEET, NOT THE TAPED TARGET. The closed sheet is a known 1219.2 x 2438.4 mm
rectangle, cut from squared stock; the target rectangle is tape laid on a floor by hand. So the
closed photograph is rectified onto the sheet, the target rectangle is MEASURED in that frame, and
the other two photographs -- where the sheet is no longer a rectangle -- are rectified onto the
measured target, which is static across the series. The target's measured size is an output.

The closed photograph doubles as the error floor: its answer is known to be zero, so whatever the
fit returns there is the combined cost of hand-cutting, tape placement and rectification, and no
deployed number can be quoted below it.

    PYTHONPATH=$(pwd) python3 nff/scripts/calibration/analyze_deployment_photos.py \
        --photo-dir data/experiments/pet45_3x3_deployment_20260814/raw \
        --sim data/outputs/runs/run_20260813_151917_pet45_3x3_mid_phys/load_path_steps.npz \
        --out-dir data/experiments/pet45_3x3_deployment_20260814
"""

import argparse
import json
import os

import numpy as np
from scipy.spatial import ConvexHull

from nff.calibration import photo_geometry as pg
from nff.calibration import photo_rectify as pr

STATES = ("Closed", "Deploying", "Deployed")
SIGMAS = (25.0, 12.0, 6.0)           # matched-filter blur, image pixels, coarse to fine


def _sheet_boundary_mask(tape: np.ndarray, band_px: float = 45.0) -> np.ndarray:
    """Tape pixels lying within ``band_px`` of the convex hull of all tape.

    The sheet's outer boundary is the only closed quadrilateral in the mask; the internal cuts have
    to be excluded before four lines can be fitted to it.
    """
    ys, xs = np.nonzero(tape)
    P = np.column_stack([xs, ys]).astype(float)
    hull = P[ConvexHull(P).vertices]
    d = np.full(len(P), np.inf)
    for a, b in zip(hull, np.roll(hull, -1, axis=0)):
        ab = b - a
        t = np.clip(((P - a) @ ab) / float(ab @ ab), 0.0, 1.0)
        d = np.minimum(d, np.linalg.norm(P - (a + t[:, None] * ab), axis=1))
    out = np.zeros_like(tape)
    sel = P[d < band_px].astype(int)
    out[sel[:, 1], sel[:, 0]] = True
    return out


def _edge_samples(polys: np.ndarray, spacing: float = 4.0) -> np.ndarray:
    return np.concatenate([pg.polygon_edge_samples(t, spacing=spacing) for t in polys])


def anchor_on_closed(rgb, masks, tape, weight, flat_polys, sheet_wh):
    """Homography for the closed photograph, resolved against the cut pattern itself.

    The sheet's four corners fix the homography completely, leaving only which image corner is which
    design corner. The pattern is chiral and asymmetric, so scoring the full cut pattern under all
    eight labellings decides it with no free parameters.

    Returns:
        ``(H, score, (flip, roll), sheet_quad, dst_mm)`` where ``dst_mm`` are the design-frame
        sheet corners in the same order as ``sheet_quad.corners``.
    """
    quad = pr.fit_quad_klines(_sheet_boundary_mask(tape), min_area_frac=0.02, end_trim=0.10)
    design = np.array([[0.0, 0.0], [sheet_wh[0], 0.0], list(sheet_wh), [0.0, sheet_wh[1]]])
    samples = _edge_samples(flat_polys)
    field = pg.matched_field(tape, 10.0)
    best = None
    for flip in (0, 1):
        base = design[::-1] if flip else design
        for roll in range(4):
            dst = np.roll(base, roll, axis=0)
            H = pr.homography_dlt(quad.corners, dst)
            sc = pg.FitContext(field, weight, np.linalg.inv(H), tape.shape).score(samples)
            if best is None or sc > best[1]:
                best = (H, sc, (flip, roll), dst)
    return best[0], best[1], best[2], quad, best[3]


def match_corner_order(ref_px: np.ndarray, corners_px: np.ndarray) -> np.ndarray:
    """Roll a corner list so corner 0 is the one nearest the reference photo's corner 0.

    The fiducial is static and the camera moved only between shots, so nearest-in-image is
    unambiguous: the corners are thousands of pixels apart and the viewpoint shifted by hundreds.
    """
    k = int(np.argmin(np.linalg.norm(corners_px - ref_px[0], axis=1)))
    return np.roll(corners_px, -k, axis=0)


def invert_curve(curve: np.ndarray, t: np.ndarray, value: float) -> tuple[float, int]:
    """Where on a monotone simulated curve a measured value sits.

    Returns:
        ``(load_fraction, nearest_step)``. The fraction is NaN when the measurement lies beyond the
        curve's range -- which is a result, not a failure: it means the simulation never reaches
        what was measured, and silently clamping to the endpoint would hide exactly that.
    """
    c = np.asarray(curve, float)
    k = int(np.argmin(np.abs(c - value)))
    if value < c.min() or value > c.max():
        return float("nan"), k
    j = int(np.searchsorted(c, value)) - 1
    j = max(0, min(j, len(c) - 2))
    span = c[j + 1] - c[j]
    frac = (value - c[j]) / span if abs(span) > 1e-12 else 0.0
    return float(t[j] + frac * (t[j + 1] - t[j])), k


def sim_descriptor_curves(sim) -> dict:
    """The simulated value of every seed-free shape descriptor, at every load step."""
    nodes = sim["face_nodes"]
    flat_xy = sim["flat_face_xy"]
    pull = int(sim["loaded_faces"][0]) if len(sim["loaded_faces"]) else 5
    clamp = int(sim["clamped_faces"][0]) if len(sim["clamped_faces"]) else 3
    rect = np.array([pg.min_area_rect(n.reshape(-1, 2))[:2] for n in nodes])
    gauged = ((sim["face_xy"][:, pull] - flat_xy[pull])
              - (sim["face_xy"][:, clamp] - flat_xy[clamp]))
    return {
        "rect_long_mm": rect[:, 1],
        "rect_short_mm": rect[:, 0],
        "mean_opening_deg": sim["opening_deg"].mean(axis=1),
        "pull_travel_mm": np.linalg.norm(sim["face_xy"][:, pull] - flat_xy[pull], axis=1),
        "pull_travel_gauged_mm": np.linalg.norm(gauged, axis=1),
    }


def measured_descriptors(meas_nodes, flat_nodes, opening, face_pairs, clamp, pull) -> dict:
    """The same descriptors, read off the measured tile polygons."""
    short, long_, _ = pg.min_area_rect(meas_nodes.reshape(-1, 2))
    d_pull = meas_nodes[pull].mean(0) - flat_nodes[pull].mean(0)
    d_clamp = meas_nodes[clamp].mean(0) - flat_nodes[clamp].mean(0)
    return {
        "rect_long_mm": float(long_),
        "rect_short_mm": float(short),
        "mean_opening_deg": float(np.mean(opening)),
        "pull_travel_mm": float(np.linalg.norm(d_pull)),
        "pull_travel_gauged_mm": float(np.linalg.norm(d_pull - d_clamp)),
    }


# Fallback when no descriptor is usable. The pull sets the EXTENSION, so extension is the one the
# operator controlled directly, and it stays informative where hinge opening has gone flat.
MATCH_ON = "rect_long_mm"


# Only these two may SET the comparison point. Width is excluded because it is the quantity a sheet
# that leaves the plane gets wrong, so matching on it would absorb the very discrepancy being
# measured; pull travel is excluded because it is contaminated by the hand grip's slip. Both are
# still reported -- as predictions to be tested, not as knobs.
MATCH_CANDIDATES = ("rect_long_mm", "mean_opening_deg")


def choose_match_descriptor(curves: dict, meas: dict, sd: dict, t: np.ndarray) -> str:
    """The descriptor that best resolves the load fraction NEAR THE MEASURED VALUE.

    Conditioning reverses along the path: hinge opening moves ten times faster per load fraction at
    the start than at the end, while extension keeps going once opening has saturated. Fixing one
    descriptor for all three photographs therefore resolves the early frames well and the late one
    badly, or the reverse.

    The criterion is SIGNAL TO NOISE -- local slope divided by that descriptor's own measurement
    scatter -- not slope alone. Slope alone prefers whichever quantity happens to have the larger
    dynamic range, which is a property of the units, and it picked the descriptor that measured the
    closed sheet 7 mm worse.
    """
    best, best_snr = None, -np.inf
    for key in MATCH_CANDIDATES:
        c = curves.get(key)
        if c is None or not np.isfinite(meas.get(key, np.nan)):
            continue
        c = np.asarray(c, float)
        frac, k = invert_curve(c, t, meas[key])
        if np.isnan(frac):                       # measurement outside the curve: cannot resolve
            continue
        j = max(1, min(k, len(c) - 2))
        slope = abs(c[j + 1] - c[j - 1]) / max(t[j + 1] - t[j - 1], 1e-9)
        noise = max(float(sd.get(key, 0.0)), 1e-6)
        snr = slope / noise
        if snr > best_snr:
            best, best_snr = key, snr
    return best or MATCH_ON


def buckling_budget(width_sim: float, width_meas: float,
                    wavelengths=(400.0, 600.0, 900.0)) -> dict:
    """Out-of-plane amplitude that would account for a shortfall in projected width.

    A homography measures the sheet's projection onto the floor, so material that has left the plane
    is simply missing from the width. Treating the shortfall as excess arc length in a sinusoidal
    ripple, ``length = lambda (1 + pi^2 A^2 / lambda^2)``, gives ``A = lambda sqrt(deficit) / pi``.
    The wavelength is not measured, so an amplitude is quoted for a plausible range of it -- this is
    an order-of-magnitude budget that says whether buckling can explain the shortfall at all, not a
    measurement of the ripple.
    """
    if width_sim <= 0:
        return {"deficit_mm": 0.0, "deficit_frac": 0.0, "amplitude_mm": {}}
    deficit = float(width_sim - width_meas)
    frac = deficit / float(width_sim)
    amps = ({f"{int(w)}": float(w * np.sqrt(frac) / np.pi) for w in wavelengths}
            if frac > 0 else {})
    return {"deficit_mm": deficit, "deficit_frac": frac, "amplitude_mm": amps}


def format_buckling(b) -> str:
    if not b or b["deficit_frac"] <= 0:
        return "  [buckling] no width shortfall to explain"
    amps = "  ".join(f"lam {k} mm -> A {v:.0f} mm" for k, v in b["amplitude_mm"].items())
    return (f"  [buckling] the sheet is {b['deficit_mm']:.1f} mm ({b['deficit_frac'] * 100:.1f}%) "
            f"narrower than predicted at matched extension\n"
            f"             as out-of-plane arc length that is   {amps}")


def _fit_at_step(ctxs, sim, flat_polys, step, args, x0=None, spacing=None, links=None):
    """Pose every tile against one simulated configuration, and read the kinematics back off it."""
    model = sim["face_nodes"][step]
    poses, scores, cover = pg.fit_all_tiles(
        ctxs, model, spacing=spacing or args.spacing, coarse=None,
        x0=np.zeros((len(model), 3)) if x0 is None else x0,
        bounds=(args.bound_mm, args.bound_deg),
        links=links, link_weight=args.link_weight)
    meas = np.stack([pg.apply_pose(model[i], poses[i], model[i].mean(0)) for i in range(len(model))])
    rot = pg.tile_rotations(flat_polys, meas)
    return dict(poses=poses, scores=scores, cover=cover, measured=meas, rot=rot,
                opening=pg.hinge_opening(rot, sim["face_pairs"]), model=model)


def analyse_state(name, path, sim, fiducial, flat_polys, args):
    """Fit the nine tiles in one photograph, on the load step whose opening the photograph shows.

    The step and the fit are mutually dependent -- the fit is initialised from a configuration, and
    the configuration is chosen by what the fit measures -- so this iterates to a fixed point and
    always returns a fit performed AT the step it reports.

    Args:
        fiducial: ``(corners_px, dst_mm)`` for this photograph, which defines its homography and is
            what the jitter ensemble perturbs.
    """
    corners_px, dst_mm = fiducial
    H = pr.homography_dlt(corners_px, dst_mm)
    rgb = pr.load_rgb(path)
    masks = pr.colour_masks(rgb)
    weight = (~pr.no_evidence_mask(rgb)).astype(np.float32)
    tape = pr.clean_mask(masks["orange"] | masks["green"], min_area_frac=0.005)
    fields = {s: pg.matched_field(tape, s) for s in SIGMAS}
    ctxs = [pg.FitContext(fields[s], weight, np.linalg.inv(H), tape.shape) for s in SIGMAS]
    links = pg.hinge_links(flat_polys, sim["face_pairs"]) if args.link_weight > 0 else None
    curves = sim_descriptor_curves(sim)
    t_sim = sim["t"]
    clamp = int(sim["clamped_faces"][0]) if len(sim["clamped_faces"]) else 3
    pull = int(sim["loaded_faces"][0]) if len(sim["loaded_faces"]) else 5

    # Scan rather than iterate to a fixed point. A pose fit is bounded about the configuration that
    # seeds it, so what it measures inherits a little of that configuration; iterating on that
    # dependence can cycle. Fitting on a spread of seeds and taking the MEDIAN gives a
    # seed-independent measurement, and the spread across seeds is an honest uncertainty on it.
    n_steps = len(sim["face_nodes"])
    if name == "Closed":
        scan = [0]
    else:
        scan = list(range(0, n_steps, args.scan_stride))
        if n_steps - 1 not in scan:
            scan.append(n_steps - 1)
    seen = []
    for k in scan:
        f = _fit_at_step(ctxs, sim, flat_polys, k, args, spacing=args.spacing * 1.5, links=links)
        seen.append((k, f, float(np.mean(f["scores"]))))
    # A seed far from the truth cannot reach it: the pose bounds hold the tiles near the
    # configuration they started from, and the fit lands off the tape. Those seeds report a value
    # that is mostly their seed's, so they are rejected on fit quality before the median is taken --
    # otherwise a failed fit would widen the uncertainty it should not be in.
    quality = np.array([q for _, _, q in seen])
    good = quality >= args.seed_accept * quality.max()

    def desc_of(f):
        return measured_descriptors(f["measured"], flat_polys, f["opening"],
                                    sim["face_pairs"], clamp, pull)

    kept = [f for (_, f, _), ok in zip(seen, good) if ok]
    per_seed = [desc_of(f) for f in kept]
    median_desc = {k: float(np.median([d[k] for d in per_seed])) for k in curves}
    sd_desc = {k: (float(np.std([d[k] for d in per_seed])) if len(per_seed) > 1 else 0.0)
               for k in curves}

    if name == "Closed":
        # Nothing to match: the operator reports the sheet closed, and its geometry is known
        # exactly. Comparing it against the flat design is what makes this frame a measurement of
        # the error FLOOR rather than one more fitted state.
        match_on, step = "flat (assumed closed)", 0
        t_star = float(t_sim[0])
    else:
        match_on = choose_match_descriptor(curves, median_desc, sd_desc, t_sim)
        t_star, step = invert_curve(curves[match_on], t_sim, median_desc[match_on])
    fit = _fit_at_step(ctxs, sim, flat_polys, step, args, links=links)

    # Every descriptor, and where each one puts the photograph on the load path. They disagree, and
    # that disagreement is the measurement: a state the simulation reaches in extension but not in
    # width is not a point on the path at all, so no single matched step can be honest about it.
    meas_desc = desc_of(fit)
    table = {}
    for key, curve in curves.items():
        frac, k = invert_curve(curve, t_sim, meas_desc[key])
        table[key] = {"measured": meas_desc[key], "sim_at_match": float(curve[step]),
                      "load_fraction": frac, "nearest_step": int(k),
                      "seed_sd": sd_desc[key],
                      "sim_range": [float(curve.min()), float(curve.max())],
                      "beyond_path": bool(np.isnan(frac))}
    if match_on in curves:
        lo, _ = invert_curve(curves[match_on], t_sim, median_desc[match_on] - sd_desc[match_on])
        hi, _ = invert_curve(curves[match_on], t_sim, median_desc[match_on] + sd_desc[match_on])
    else:
        lo = hi = t_star

    fit["scan"] = [(int(k), float(desc_of(f).get(match_on, np.nan)), float(q))
                   for k, f, q in seen]
    fit["match_on"] = match_on
    fit["match_seed_median"] = median_desc.get(match_on)
    fit["match_seed_sd"] = sd_desc.get(match_on)
    fit["descriptors"] = table
    fit["t_interval"] = (float(lo), float(hi))

    jitter = None
    if args.jitter_n > 0:
        jitter = _jitter_ensemble(fields, weight, tape.shape, corners_px, dst_mm,
                                  sim, flat_polys, step, fit, args)
    return dict(rgb=rgb, masks=masks, tape=tape, step=int(step), t_star=float(t_star), H=H,
                jitter=jitter, **fit)


def _jitter_ensemble(fields, weight, shape, corners_px, dst_mm, sim, flat_polys, step, fit, args):
    """Re-run the fit under fiducial corners perturbed by the tape half-width.

    The datum of a hand-laid tape line is uncertain by roughly half its width, and that uncertainty
    is systematic rather than statistical, so it is injected at the corners and propagated through
    the whole measurement rather than read off the fit's own residuals. Each draw restarts from the
    converged poses and refines at the finest scale only, which is what makes the ensemble
    affordable.

    Returns:
        dict of per-draw arrays: ``centroid_rms``, ``rot_rms``, ``opening`` (n, n_hinges),
        ``shape_rms``.
    """
    rng = np.random.default_rng(args.seed)
    model = sim["face_nodes"][step]
    out = {"centroid_rms": [], "rot_rms": [], "opening": [], "shape_rms": []}
    for _ in range(args.jitter_n):
        Hj = pr.homography_dlt(corners_px + rng.normal(0.0, args.jitter_px, corners_px.shape),
                               dst_mm)
        ctx = [pg.FitContext(fields[SIGMAS[-1]], weight, np.linalg.inv(Hj), shape)]
        f = _fit_at_step(ctx, sim, flat_polys, step, args, x0=fit["poses"],
                         spacing=args.spacing * 1.5, links=None)
        d = np.hypot(f["poses"][:, 0], f["poses"][:, 1])
        _, _, srms = pg.kabsch_2d(f["measured"].reshape(-1, 2), model.reshape(-1, 2))
        out["centroid_rms"].append(float(np.sqrt(np.mean(d ** 2))))
        out["rot_rms"].append(float(np.sqrt(np.mean(f["poses"][:, 2] ** 2))))
        out["opening"].append(f["opening"])
        out["shape_rms"].append(srms)
    return {k: np.asarray(v) for k, v in out.items()}


def write_images(name, r, red_design, out_dir, mm_per_px=1.5, win=(-430, 2020, -300, 2870)):
    """Write the rectified frame three ways: as shot, filtered, and with the prediction on it.

    The filtered version is the one to look at. The sheet is transparent, so a photograph of it is
    mostly floor; flattening the floor to grey and keeping only the tape turns the frame into the
    line drawing the measurement actually reads.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(win[0], win[1], mm_per_px)
    y = np.arange(win[2], win[3], mm_per_px)
    raw = np.clip(np.nan_to_num(pr.warp_to_grid(r["rgb"], r["H"], x, y, fill=np.nan), nan=1.0), 0, 1)
    clean = pr.clean_render(r["rgb"], r["masks"])
    cln = np.clip(np.nan_to_num(pr.warp_to_grid(clean, r["H"], x, y, fill=np.nan), nan=1.0), 0, 1)
    for tag, img in (("rect", raw), ("clean", cln)):
        plt.imsave(os.path.join(out_dir, f"{name}_{tag}.png"), np.flipud(img))

    fig, ax = plt.subplots(figsize=(7.0, 7.0 * (win[3] - win[2]) / (win[1] - win[0])))
    ax.imshow(cln, extent=[x[0], x[-1], y[0], y[-1]], origin="lower")
    q = np.vstack([red_design, red_design[:1]])
    ax.plot(q[:, 0], q[:, 1], color="#D62828", lw=1.6)
    for i, poly in enumerate(r["model"]):
        a = np.vstack([poly, poly[:1]])
        ax.plot(a[:, 0], a[:, 1], color="#08519c", lw=1.4)
        b = np.vstack([r["measured"][i], r["measured"][i][:1]])
        ax.plot(b[:, 0], b[:, 1], color="#1A1A1A", lw=0.9, ls="--")
        ax.text(*r["measured"][i].mean(0), str(i), color="#08519c", fontsize=9,
                ha="center", va="center", weight="bold")
    ax.set_aspect("equal")
    ax.set_title(f"{name}: sim (blue) vs measured (black dashed), step {r['step']}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}_overlay.png"), dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--photo-dir", required=True)
    p.add_argument("--sim", required=True, help="load_path_steps.npz from dump_load_path_steps.py")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--sheet-mm", type=float, nargs=2, default=[1219.2, 2438.4],
                   help="true flat sheet size; the metric standard")
    p.add_argument("--spacing", type=float, default=4.0, help="edge sample spacing, mm")
    p.add_argument("--bound-mm", type=float, default=60.0, help="pose search bound about the sim")
    p.add_argument("--bound-deg", type=float, default=8.0)
    p.add_argument("--seed-accept", type=float, default=0.90,
                   help="keep seeds whose mean fit score is at least this fraction of the best")
    p.add_argument("--scan-stride", type=int, default=4,
                   help="load-step stride for the seed scan that measures the opening")
    p.add_argument("--min-coverage", type=float, default=0.55,
                   help="below this fraction of edge samples on tape, a tile is flagged")
    p.add_argument("--jitter-n", type=int, default=0, help="corner-perturbation ensemble draws")
    p.add_argument("--jitter-px", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--link-weight", type=float, default=1.0,
                   help="strength of the hinge-connectivity tie; 0 fits the tiles independently")
    p.add_argument("--no-images", action="store_true", help="skip the rectified image exports")
    args = p.parse_args()

    sim = np.load(args.sim)
    flat = sim["flat_face_nodes"]
    face_pairs = sim["face_pairs"]
    os.makedirs(args.out_dir, exist_ok=True)
    derived = os.path.join(args.out_dir, "derived")
    os.makedirs(derived, exist_ok=True)

    # ── anchor on the closed sheet ──
    cpath = os.path.join(args.photo_dir, "Closed.jpg")
    rgb0 = pr.load_rgb(cpath)
    m0 = pr.colour_masks(rgb0)
    w0 = (~pr.no_evidence_mask(rgb0)).astype(np.float32)
    tape0 = pr.clean_mask(m0["orange"] | m0["green"], min_area_frac=0.005)
    H0, sc0, lab, sheet_quad, sheet_dst = anchor_on_closed(
        rgb0, m0, tape0, w0, flat, tuple(args.sheet_mm))
    red0 = pr.fit_quad_klines(m0["red"], end_trim=0.10)
    red_design = pr.apply_h(H0, red0.corners)
    rs, rl, ra = pg.min_area_rect(red_design)
    print(f"anchor: pattern score {sc0:.4f}  labelling flip={lab[0]} roll={lab[1]}")
    print(f"target rectangle MEASURED {rs:.1f} x {rl:.1f} mm at {ra:.2f} deg, "
          f"centre ({red_design.mean(0)[0]:.1f}, {red_design.mean(0)[1]:.1f})")

    clamp = int(sim["clamped_faces"][0]) if len(sim["clamped_faces"]) else 3
    pull = int(sim["loaded_faces"][0]) if len(sim["loaded_faces"]) else 5
    results, out = {}, {}
    for name in STATES:
        path = os.path.join(args.photo_dir, f"{name}.jpg")
        if name == "Closed":
            fid = (sheet_quad.corners, sheet_dst)
        else:
            red = pr.fit_quad_klines(pr.colour_masks(pr.load_rgb(path))["red"], end_trim=0.10)
            fid = (match_corner_order(red0.corners, red.corners), red_design)
        r = analyse_state(name, path, sim, fid, flat, args)
        results[name] = r

        model = sim["face_nodes"][r["step"]]
        P = r["poses"]
        d = np.hypot(P[:, 0], P[:, 1])
        # clamp-gauged: the simulation holds the clamp rigidly, the lab held it by hand, so the
        # clamp's own pose is the gauge and is removed before the rest is compared.
        Pg = P - P[clamp]
        dg = np.hypot(Pg[:, 0], Pg[:, 1])
        sim_open = sim["opening_deg"][r["step"]]
        R, t, shape_rms = pg.kabsch_2d(r["measured"].reshape(-1, 2), model.reshape(-1, 2))
        swing = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
        # Per tile, what is left once the whole sheet's rigid motion is removed. The raw error is
        # dominated by the hand grip wandering; this is the part the model is actually answerable
        # for, so it is what the figure shows.
        aligned = r["measured"].reshape(-1, 2) @ R.T + t
        aligned = aligned.reshape(r["measured"].shape)
        shape_resid = np.linalg.norm(aligned.mean(1) - model.mean(1), axis=-1)
        flags = [i for i in range(len(P)) if r["cover"][i] < args.min_coverage]
        j = r["jitter"]
        out[name] = dict(
            step=r["step"], t_star=r["t_star"], load_fraction=float(sim["t"][r["step"]]),
            pose=P.tolist(), score=r["scores"].tolist(), coverage=r["cover"].tolist(),
            low_coverage_tiles=flags,
            centroid_err_mm=d.tolist(),
            centroid_rms_mm=float(np.sqrt(np.mean(d ** 2))), centroid_max_mm=float(d.max()),
            rot_err_deg=P[:, 2].tolist(), rot_rms_deg=float(np.sqrt(np.mean(P[:, 2] ** 2))),
            clamp_face=clamp, pull_face=pull, clamp_pose=P[clamp].tolist(),
            clamp_slip_mm=float(d[clamp]),
            gauged_centroid_rms_mm=float(np.sqrt(np.mean(dg ** 2))),
            gauged_centroid_max_mm=float(dg.max()),
            pull_travel_measured_mm=float(np.linalg.norm(
                r["measured"][pull].mean(0) - flat[pull].mean(0))),
            pull_travel_sim_mm=float(np.linalg.norm(model[pull].mean(0) - flat[pull].mean(0))),
            shape_only_rms_mm=shape_rms, shape_only_swing_deg=swing,
            shape_resid_mm=shape_resid.tolist(),
            shape_only_translation_mm=t.tolist(),
            opening_measured_deg=r["opening"].tolist(), opening_sim_deg=sim_open.tolist(),
            opening_rmse_deg=float(np.sqrt(np.mean((r["opening"] - sim_open) ** 2))),
            opening_sd_deg=(j["opening"].std(0).tolist() if j else None),
            match_on=r.get("match_on"), match_seed_median=r.get("match_seed_median"),
            match_seed_sd=r.get("match_seed_sd"),
            descriptors=r.get("descriptors"),
            buckling=buckling_budget(
                (r.get("descriptors") or {}).get("rect_short_mm", {}).get("sim_at_match", 0.0),
                (r.get("descriptors") or {}).get("rect_short_mm", {}).get("measured", 0.0)),
            t_interval=list(r.get("t_interval", (float("nan"), float("nan")))),
            equilibrium=(name != "Deploying"),
            seed_scan=[[int(k), float(o), float(q)] for k, o, q in r.get("scan", [])],
            centroid_rms_sd_mm=(float(j["centroid_rms"].std()) if j else None),
            shape_rms_sd_mm=(float(j["shape_rms"].std()) if j else None),
            measured_nodes_mm=r["measured"].tolist(), sim_nodes_mm=model.tolist(),
            homography=r["H"].tolist())
        print(f"\n{name}: matched on {r['match_on']} -> step {r['step']} of {len(sim['t'])}, "
              f"load {sim['t'][r['step']]:.3f}"
              + ("" if name != "Deploying" else "   [NOT AN EQUILIBRIUM: shot mid-pull]"))
        print("  descriptor            measured        sim@match   -> load fraction")
        for key, dd in (r.get("descriptors") or {}).items():
            frac = "BEYOND the path" if dd["beyond_path"] else f"{dd['load_fraction']:.3f}"
            star = " *" if key == r["match_on"] else "  "
            print(f"   {star}{key:22s} {dd['measured']:9.1f} {dd['sim_at_match']:12.1f}   -> {frac}")
        print(f"  centroid err RMS {out[name]['centroid_rms_mm']:.1f} mm  "
              f"max {out[name]['centroid_max_mm']:.1f}  rot RMS {out[name]['rot_rms_deg']:.2f} deg")
        print(f"  clamp (tile {clamp}) slipped {d[clamp]:.1f} mm; gauged on it the centroid RMS is "
              f"{out[name]['gauged_centroid_rms_mm']:.1f} mm")
        print(f"  pull (tile {pull}) travelled {out[name]['pull_travel_measured_mm']:.1f} mm, "
              f"sim {out[name]['pull_travel_sim_mm']:.1f} mm")
        print(f"  shape-only (Kabsch, no scale) RMS {shape_rms:.1f} mm, swing {swing:+.2f} deg")
        print(f"  hinge opening RMSE {out[name]['opening_rmse_deg']:.2f} deg   "
              f"coverage {np.min(r['cover']):.2f}-{np.max(r['cover']):.2f}"
              + (f"   LOW-COVERAGE tiles {flags}" if flags else ""))
        if name != "Closed":
            print(format_buckling(out[name]["buckling"]))

    summary = dict(
        sim=os.path.abspath(args.sim), photo_dir=os.path.abspath(args.photo_dir),
        sheet_mm=list(args.sheet_mm), anchor_score=float(sc0),
        anchor_labelling=dict(flip=int(lab[0]), roll=int(lab[1])),
        target_measured_mm=[float(rs), float(rl)], target_measured_angle_deg=float(ra),
        target_measured_centre_mm=red_design.mean(0).tolist(),
        target_design_mm=[1739.798, 2714.264], target_design_angle_deg=-15.3868,
        target_design_centre_mm=[789.032, 1312.266],
        red_design_corners_mm=red_design.tolist(),
        noise_floor_mm=out["Closed"]["centroid_rms_mm"],
        noise_floor_deg=out["Closed"]["rot_rms_deg"],
        states=out)
    if not args.no_images:
        rect_dir = os.path.join(args.out_dir, "rectified")
        os.makedirs(rect_dir, exist_ok=True)
        for name in STATES:
            write_images(name, results[name], red_design, rect_dir)
        print(f"wrote rectified/filtered/overlay images to {rect_dir}")

    with open(os.path.join(derived, "metrics.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    np.savez_compressed(
        os.path.join(derived, "measured.npz"),
        **{f"{n}_nodes": np.asarray(out[n]["measured_nodes_mm"]) for n in STATES},
        **{f"{n}_pose": np.asarray(out[n]["pose"]) for n in STATES},
        **{f"{n}_H": np.asarray(out[n]["homography"]) for n in STATES},
        **{f"{n}_step": np.int32(out[n]["step"]) for n in STATES},
        red_design_corners=red_design, flat_face_nodes=flat, face_pairs=face_pairs)
    print(f"\nwrote {derived}/metrics.json and measured.npz")
    print(f"NOISE FLOOR from the closed state: {summary['noise_floor_mm']:.1f} mm, "
          f"{summary['noise_floor_deg']:.2f} deg")


if __name__ == "__main__":
    main()
