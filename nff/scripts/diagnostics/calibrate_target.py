"""Calibrate the TARGET SHAPE and the LOAD to a requested deployment angle.

The closed pipeline needs a target that a deployed sheet can actually reach, otherwise "match the
target precisely" is unachievable by construction and the chamfer stalls at whatever the geometry
forbids. This derives the target from the tessellation's own mechanism kinematics at a chosen
deployment angle, and solves for the load that reaches that angle.

THE LAW. Two rigid tiles hinged at a shared CORNER. Each tile centre sits a distance ``d`` from
the hinge, and the hinge is offset from the centre-to-centre line by an angle ``psi``. Rotating
the pair apart by a relative angle ``theta`` swings the hinge toward that line, so the centre
separation -- the panel pitch -- goes as

    lam(theta) = cos(psi - theta/2) / cos(psi)

which is 1 at theta = 0, and maximal at FULL deployment theta = 2*psi where the centres and the
hinge become collinear:  lam_max = 1 / cos(psi).

For a SQUARE tile the corner sits at 45 deg, so lam_max = sqrt(2) at theta = 90 deg -- the
familiar rotating-squares result. ⚠ That is the square case ONLY. A tile of pitch
``spacing`` x ``spacing_y`` has a DIFFERENT psi along each axis,

    psi_x = atan2(spacing_y/2, spacing/2)      psi_y = atan2(spacing/2, spacing_y/2)

so an elongated sheet deploys ANISOTROPICALLY (and its two axes reach full deployment at
different angles -- the shorter axis locks first, which is what makes the mechanism saturate).
Assuming sqrt(2)-at-90 on a non-square sheet is the mistake this script exists to prevent.

Run:
    PYTHONPATH=$(pwd) JAX_PLATFORMS=cpu conda run -n kgnn_mac \
        python nff/scripts/diagnostics/calibrate_target.py \
            --config data/configs/closed/sheet_4x8ft_rom.yaml --angle 45
"""

import argparse
import copy
import re

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np

from nff.config.experiment import load_and_parse_config
from nff.closed.setup import (build_closed_initial_state, init_closed_les_params,
                              build_surrogate_energy)
from nff.closed.deploy import _boundary_cloud
from nff.stages.pipeline import forward_pipeline


def hinge_offset_angles(spacing: float, spacing_y: float):
    """(psi_x, psi_y) in degrees -- the corner-hinge offset from each centre-to-centre line."""
    return (float(np.degrees(np.arctan2(spacing_y / 2.0, spacing / 2.0))),
            float(np.degrees(np.arctan2(spacing / 2.0, spacing_y / 2.0))))


def opening_ratio(psi_deg: float, theta_deg: float) -> float:
    """lam(theta) = cos(psi - theta/2)/cos(psi), clamped at full deployment theta = 2*psi."""
    th = min(float(theta_deg), 2.0 * float(psi_deg))
    return float(np.cos(np.radians(psi_deg - th / 2.0)) / np.cos(np.radians(psi_deg)))


def deploy(config, load_scale: float, n_steps: int = 12, *,
           init_noise: float | None = None, init_seed: int | None = None):
    """Deploy the untrained design at ``load_scale`` x the config loads.

    ⚠ The deployment MUST run under the configured hinge model. This function used to call
    ``forward_pipeline`` without ``bond_energy_fn``, which silently fell back to the linear springs
    and therefore calibrated the ROM no matter what ``hinge_model:`` said. The two models are not
    interchangeable: on the standard sheet the surrogate needs roughly a NINTH of the ROM's load to
    reach the same angle, because it reproduces the saturating (buckled) hinge moment that a single
    quadratic spring cannot. A ROM load applied to a surrogate run drives the sheet through its own
    kinematic lock, where the hinges stop rotating and start translating out of the trained domain.

    ⚠ The starting design must be the SAME one the run will train from. This used to force
    ``init_noise = 0`` unconditionally, which is right for a deterministic uniform start and wrong
    the moment a run uses a seeded random start: the target would then be sized for a sheet that is
    never trained. ``init_noise``/``init_seed`` default to whatever the config carries; pass them
    explicitly to override.

    Returns (theta_rel_deg per hinge, flat boundary cloud, deployed boundary cloud, swing_deg),
    where ``swing`` is the MEAN face rotation -- the rigid-body mode. The deployment mode
    counter-rotates neighbouring tiles and averages to ~0, so a large mean is the signature of a
    sheet swinging bodily about its supports rather than opening, which no design can cancel.
    """
    c = copy.deepcopy(config)
    c.physics.num_load_steps = int(n_steps)
    c.topology['loads'] = [{**dict(l), 'value': float(l['value']) * load_scale}
                           for l in c.topology.get('loads', [])]
    # loads are baked into the state at build time -- scale BEFORE building, not after
    state, _ = build_closed_initial_state(c)
    if init_noise is not None:
        c.topology['init_noise'] = float(init_noise)
    if init_seed is not None:
        c.topology['init_seed'] = int(init_seed)
    params, sf = init_closed_les_params(c)

    # Same construction as run_closed.py and extract_hinge_paths.py. Returns (None,)*5 on the ROM
    # branch, which is exactly the spring default forward_pipeline already applies.
    bond_energy, _, geometry_fn, _, w_lig_logit0 = build_surrogate_energy(c, sf, state, params)
    if w_lig_logit0 is not None:
        params = {**params, 'w_lig_logit': w_lig_logit0}
    hinge_model_type = getattr(getattr(c, 'hinge_model', None), 'type', 'rom')
    if hinge_model_type == 'surrogate' and bond_energy is None:
        raise RuntimeError(
            "hinge_model.type is 'surrogate' but no bond energy was built, so this deployment "
            "would silently run on the linear springs and calibrate the WRONG model. Fix the "
            "checkpoint path in hinge_model.checkpoint rather than letting it fall back.")

    res = forward_pipeline(state, c.target, c.validity, c.physics, map_type=c.mapping.type,
                           map_params=params, static_features=sf, load_specs=c.topology['loads'],
                           bond_energy_fn=bond_energy,
                           # Solver-side geometry only when a surrogate bond energy reads it — the
                           # ROM's ligament energy takes it as an ignored kwarg. Same rule as
                           # nff/training/loss.py.
                           hinge_geometry=(geometry_fn(params)
                                           if (geometry_fn is not None and bond_energy is not None)
                                           else None))
    vs = res['valid_state']
    disp = np.asarray(res['solution'].fields[-1])
    # relative rotation across each hinge = theta_2 - theta_1 of the two faces it bonds
    bc = np.asarray(vs.bond_connectivity)
    n_nodes = np.asarray(vs.centroid_node_vectors).shape[1]
    th = np.asarray(disp[:, 2])
    theta_rel = np.degrees(np.abs(th[bc[:, 1] // n_nodes] - th[bc[:, 0] // n_nodes]))
    swing = float(np.degrees(np.mean(th)))
    return (theta_rel, _boundary_cloud(vs, np.zeros_like(disp)), _boundary_cloud(vs, disp), swing)


CRITERIA = {
    "max":  (lambda th, depl: float(th.max()), "theta_max", "deg"),
    "mean": (lambda th, depl: float(th.mean()), "theta_mean", "deg"),
    "bbox": (lambda th, depl: float(depl[:, 0].max() - depl[:, 0].min()), "bbox_w", "u"),
}


def solve_load_for_goal(config, goal: float, *, n_steps: int = 12, tol: float = 0.05,
                        max_iter: int = 40, criterion: str = "max", verbose: bool = True,
                        init_noise: float | None = None, init_seed: int | None = None):
    """Bisect the load scale until the chosen deployment measure reaches ``goal``.

    ``criterion='max'`` (default) drives the WORST hinge to the angle. That is the right reading of
    a deployment-angle spec: it is a CEILING the material imposes, not a value every hinge must
    hit. The distinction is invisible when the sheet opens uniformly (the distributed 3-tile grip
    holds every hinge inside 2.3 deg, so mean and max agree to half a degree) and decisive when it
    does not -- a single-tile grip funnels everything through one column, and bisecting on the MEAN
    there asks the loaded hinges to go to 80+ deg, tearing the sheet apart, to drag the mean up.

    ``criterion='mean'`` is kept for the uniform case, where it is the sheet-level number.

    ``criterion='bbox'`` bisects on the DEPLOYED WIDTH instead, for a family of problems that must
    share one target rectangle. Different load cases reach a given hinge angle at different overall
    extensions, so matching on angle would leave each row aiming at a shape it happens to overshoot
    or undershoot; matching on width puts every row at the same place and makes the shared target a
    fair comparison. All five rows still report their angles, which is what says the sheet actually
    deployed.
    """
    kw = dict(init_noise=init_noise, init_seed=init_seed)
    measure, label, unit = CRITERIA[criterion]
    lo, hi = 1e-3, 1.0
    th, _, dp, _ = deploy(config, hi, n_steps, **kw)
    while measure(th, dp) < goal and hi < 1e5:            # bracket from above first
        hi *= 4.0
        th, _, dp, _ = deploy(config, hi, n_steps, **kw)
    if measure(th, dp) < goal:
        raise SystemExit(f"the mechanism saturates below {label} = {goal} {unit} "
                         f"(reached {measure(th, dp):.2f} at {hi:g}x load) -- it cannot get there")
    best = None
    for _ in range(max_iter):
        mid = np.sqrt(lo * hi)                           # geometric bisection: load spans decades
        theta, flat, depl, swing = deploy(config, mid, n_steps, **kw)
        got = measure(theta, depl)
        best = (mid, theta, flat, depl, swing)
        if verbose:
            print(f"    load x{mid:9.4f}  {label} {got:7.3f} {unit}  "
                  f"(theta max {theta.max():.2f} mean {theta.mean():.2f} "
                  f"spread {theta.max()-theta.min():.2f}, swing {swing:+.2f} deg)")
        if abs(got - goal) < tol:
            break
        lo, hi = (mid, hi) if got < goal else (lo, mid)
    return best


def read_target(config_path: str):
    """The four ``target_*`` keys of an already-calibrated config, as (cx, cy, half_w, half_h)."""
    topo = load_and_parse_config(config_path).topology
    missing = [k for k in ("target_cx", "target_cy", "target_half_w", "target_half_h")
               if k not in topo]
    if missing:
        raise SystemExit(f"{config_path} carries no calibrated target ({', '.join(missing)} "
                         f"missing) -- calibrate that config first, then point --target-from at it")
    return tuple(float(topo[k]) for k in
                 ("target_cx", "target_cy", "target_half_w", "target_half_h"))


_SCALAR = r"[-+0-9.eE]+"


def _block_span(lines, block: str):
    """(start, end) line indices of a top-level ``block:`` body, end exclusive."""
    start = next((i for i, ln in enumerate(lines) if re.match(rf"^{re.escape(block)}\s*:", ln)), None)
    if start is None:
        raise KeyError(f"block '{block}:' not found")
    end = start + 1
    for i in range(start + 1, len(lines)):
        end = i
        if lines[i].strip() and not lines[i].startswith((" ", "\t", "#")):
            return start, i
    return start, len(lines)


def _set_in_block(lines, block: str, key: str, value):
    """Replace ``key:`` inside ``block:``, or insert it if absent. Comments are preserved."""
    start, end = _block_span(lines, block)
    pat = re.compile(rf"^(\s*){re.escape(key)}\s*:\s*{_SCALAR}(.*)$")
    for i in range(start + 1, end):
        m = pat.match(lines[i])
        if m:
            lines[i] = f"{m.group(1)}{key}: {value}{m.group(2)}"
            return lines
    lines.insert(start + 1, f"  {key}: {value}")
    return lines


def write_config(src_path: str, dst_path: str, *, target, load_scale: float, header: str,
                 overrides=None):
    """Write a runnable config by substituting into the SOURCE TEXT.

    Text substitution, not a YAML round-trip: these configs carry long comment banners recording how
    each number was calibrated, and ``yaml.dump`` would silently delete every one of them.

    ``target`` is (cx, cy, half_w, half_h); ``load_scale`` multiplies every ``loads[].value``;
    ``overrides`` maps ``(block, key) -> value``. Every substitution must land — a key that is not
    found raises, because a writer that silently emits an unchanged config is how a sweep ends up
    with N identical arms.
    """
    with open(src_path) as fh:
        lines = fh.read().split("\n")

    for key, val in zip(("target_cx", "target_cy", "target_half_w", "target_half_h"), target):
        _set_in_block(lines, "tessellation", key, f"{val:.4f}")

    load_pat = re.compile(rf"^(\s*-\s*\{{face:\s*\d+,\s*dof:\s*\d+,\s*value:\s*)({_SCALAR})(\}}.*)$")
    n_loads = 0
    for i, ln in enumerate(lines):
        m = load_pat.match(ln)
        if m:
            lines[i] = f"{m.group(1)}{float(m.group(2)) * load_scale:.2f}{m.group(3)}"
            n_loads += 1
    if n_loads == 0:
        raise KeyError(f"no `- {{face: .., dof: .., value: ..}}` load lines found in {src_path}")

    for (block, key), val in (overrides or {}).items():
        _set_in_block(lines, block, key, val)

    body = "\n".join(lines)
    banner = "\n".join(f"# {ln}" if ln else "#" for ln in header.split("\n"))
    with open(dst_path, "w") as fh:
        fh.write(banner + "\n" + body)
    return n_loads


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--angle", type=float, default=45.0,
                   help="deployment angle [deg] -- the load is bisected until the hinges reach it")
    p.add_argument("--shape-angle", type=float, default=None,
                   help="angle at which the TARGET RECTANGLE is sized; defaults to --angle. Set it "
                        "BELOW --angle to leave the design headroom: the sheet can open to --angle "
                        "but is asked to match a smaller shape, so matching requires pulling IN, "
                        "which is the direction hinge stiffness (learnable w_lig) controls. A "
                        "target sized at the edge of reachability saturates the optimizer against "
                        "the kinematic limit instead, and w_lig gets no leverage.")
    p.add_argument("--criterion", choices=tuple(CRITERIA), default="max",
                   help="drive the WORST hinge to --angle (max, default: the angle is a material "
                        "CEILING), the sheet average (mean), or the DEPLOYED WIDTH to --bbox-target "
                        "(bbox -- for a family of load cases that must share one target rectangle)")
    p.add_argument("--bbox-target", type=float, default=None,
                   help="deployed bbox width [pipeline units] to bisect the load to; required by "
                        "--criterion bbox. Take it from the run that DEFINES the shared target.")
    p.add_argument("--target-from", default=None,
                   help="copy the four target_* keys from this already-calibrated config instead of "
                        "sizing a new rectangle, so every row of a family aims at the SAME shape")
    p.add_argument("--steps", type=int, default=12, help="load steps in each trial deploy")
    p.add_argument("--tol", type=float, default=0.05, help="angle tolerance [deg]")
    p.add_argument("--load-scale", type=float, default=None,
                   help="SKIP the bisection and deploy at this multiple of the config's loads. Use "
                        "when you are choosing the FORCE directly (a force ladder) rather than "
                        "solving for an angle.")
    p.add_argument("--from-deployed", action="store_true",
                   help="size the target from the MEASURED deployed bbox x --pull-in-*, not from "
                        "the analytic lam(theta). ⚠ REQUIRED for a concentrated (single-tile) grip: "
                        "lam(theta) assumes the sheet opens uniformly, which such a grip "
                        "deliberately breaks, so the analytic rectangle is simply wrong there.")
    p.add_argument("--center-on-clamp", action="store_true",
                   help="centre the target in x on the CLAMPED faces instead of on the deployed "
                        "bounding box. The clamp is fixed in all DOFs, so its centreline is the "
                        "problem's main axis and does not drift as the sheet swings; the deployed "
                        "bbox midpoint does.")
    p.add_argument("--derotate", action="store_true",
                   help="remove the rigid SWING before measuring the deployed bbox. A rotated "
                        "sheet's axis-aligned bbox is wider than the sheet, so a swinging grip "
                        "sizes the target to an inflated extent the mechanism can never reach. "
                        "⚠ It does not square the sheet up: the target ends the right SIZE, but "
                        "the deployed sheet still sits crooked inside it.")
    p.add_argument("--pull-in-x", type=float, default=0.953,
                   help="target half-width as a fraction of the deployed half-width (--from-deployed)")
    p.add_argument("--pull-in-y", type=float, default=0.990,
                   help="target half-height as a fraction of the deployed half-height")
    p.add_argument("--init-noise", type=float, default=None,
                   help="Gaussian sigma (logit units) on the STARTING design; defaults to the "
                        "config's own value. The target must be sized at the same start the run "
                        "trains from.")
    p.add_argument("--init-seed", type=int, default=None, help="seed for --init-noise")
    p.add_argument("--write-config", default=None,
                   help="write a runnable YAML here (source comments preserved) instead of only "
                        "printing the paste block")
    args = p.parse_args()

    config = load_and_parse_config(args.config)
    topo = config.topology
    sx = float(topo.get('spacing', 1.0))
    sy = float(topo.get('spacing_y', sx))
    psi_x, psi_y = hinge_offset_angles(sx, sy)

    print(f"\n  TESSELLATION  pitch {sx} x {sy}   ({topo.get('M')} x {topo.get('N')} panels)")
    print(f"    psi_x = {psi_x:5.2f} deg   lam_max = {1/np.cos(np.radians(psi_x)):.4f}  "
          f"(full deployment at {2*psi_x:.1f} deg)")
    print(f"    psi_y = {psi_y:5.2f} deg   lam_max = {1/np.cos(np.radians(psi_y)):.4f}  "
          f"(full deployment at {2*psi_y:.1f} deg)")
    if abs(sx - sy) < 1e-9:
        print("    square tiles -> the classic sqrt(2) at 90 deg")
    else:
        print(f"    NOT square -> the sqrt(2)-at-90 rule does NOT apply; the y axis locks at "
              f"{2*psi_y:.1f} deg")

    shape_angle = args.angle if args.shape_angle is None else float(args.shape_angle)
    lam_x_dep = opening_ratio(psi_x, args.angle)          # what the LOAD is bisected to reach
    lam_y_dep = opening_ratio(psi_y, args.angle)
    lam_x = opening_ratio(psi_x, shape_angle)             # what the TARGET is sized at
    lam_y = opening_ratio(psi_y, shape_angle)
    print(f"\n  OPENING RATIO at theta = {args.angle:g} deg:  "
          f"lam_x = {lam_x_dep:.4f}   lam_y = {lam_y_dep:.4f}")
    if shape_angle != args.angle:
        print(f"  TARGET sized at theta = {shape_angle:g} deg instead:  "
              f"lam_x = {lam_x:.4f}   lam_y = {lam_y:.4f}   <- headroom for the design to pull in")

    init_kw = dict(init_noise=args.init_noise, init_seed=args.init_seed)
    noise_eff = args.init_noise if args.init_noise is not None else float(topo.get('init_noise', 0.0))
    seed_eff = args.init_seed if args.init_seed is not None else int(topo.get('init_seed', 0))
    print(f"\n  STARTING DESIGN  init_noise = {noise_eff:g}  init_seed = {seed_eff}"
          + ("   (uniform r_init)" if noise_eff == 0.0 else "   (seeded random start)"))

    if args.criterion == "bbox" and args.bbox_target is None and args.load_scale is None:
        raise SystemExit("--criterion bbox needs --bbox-target (the deployed width to reach)")

    if args.load_scale is not None:
        print(f"\n  deploying at the GIVEN load scale x{args.load_scale:g} (no bisection) ...")
        scale = float(args.load_scale)
        theta, flat, depl, swing = deploy(config, scale, args.steps, **init_kw)
    elif args.criterion == "bbox":
        print(f"\n  solving for the load whose DEPLOYED WIDTH is {args.bbox_target:g} u ...")
        scale, theta, flat, depl, swing = solve_load_for_goal(
            config, float(args.bbox_target), n_steps=args.steps, tol=max(args.tol, 1e-3),
            criterion="bbox", **init_kw)
    else:
        print(f"\n  solving for the load whose {args.criterion.upper()} hinge angle is "
              f"{args.angle:g} deg ...")
        scale, theta, flat, depl, swing = solve_load_for_goal(
            config, args.angle, n_steps=args.steps, tol=args.tol, criterion=args.criterion,
            **init_kw)
    base = [float(l['value']) for l in topo.get('loads', [])]
    print(f"\n  LOAD  x{scale:.4f}  ->  " + ", ".join(f"{b*scale:.2f} N" for b in base))
    print(f"    hinge angle: max {theta.max():.2f}  mean {theta.mean():.2f}  min {theta.min():.2f}"
          f"  spread {theta.max()-theta.min():.2f} deg")
    print(f"    swing (mean face rotation): {swing:+.2f} deg"
          + ("   <- RIGID-BODY MODE, no design can cancel it" if abs(swing) > 5.0 else ""))

    w0 = flat[:, 0].max() - flat[:, 0].min()
    h0 = flat[:, 1].max() - flat[:, 1].min()

    # ── de-rotate before measuring, so the target is sized to the SHAPE and not to its bbox ──
    # A swinging sheet is measured through an axis-aligned bounding box, and the bbox of a rotated
    # shape is larger than the shape: w*cos+h*sin by w*sin+h*cos. Sizing the target from it asks the
    # design to expand further than the mechanism can, which buys nothing and drives extra
    # compression. Removing the rigid swing first sizes the rectangle to what the tessellation
    # actually spans. ⚠ It does NOT make the sheet sit square in that rectangle -- the swing is a
    # mechanism mode and `type: rectangle` carries no rotation -- so expect a right-sized target
    # that the deployed sheet still sits crooked inside.
    if args.derotate and abs(swing) > 1e-9:
        raw_w = depl[:, 0].max() - depl[:, 0].min()
        raw_h = depl[:, 1].max() - depl[:, 1].min()
        ctr = depl.mean(0)
        a = np.radians(-swing)
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        depl = (depl - ctr) @ rot.T + ctr
        new_w = depl[:, 0].max() - depl[:, 0].min()
        new_h = depl[:, 1].max() - depl[:, 1].min()
        print(f"\n  DE-ROTATED by {-swing:+.2f} deg before sizing the target")
        print(f"    raw bbox        {raw_w:.4f} x {raw_h:.4f} u   (inflated by the swing)")
        print(f"    de-rotated bbox {new_w:.4f} x {new_h:.4f} u   "
              f"({100*(new_w/raw_w-1):+.1f}% in x, {100*(new_h/raw_h-1):+.1f}% in y)")

    w1 = depl[:, 0].max() - depl[:, 0].min()
    h1 = depl[:, 1].max() - depl[:, 1].min()
    print(f"\n  flat sheet      {w0:.4f} x {h0:.4f} u")
    print(f"  deployed bbox   {w1:.4f} x {h1:.4f} u   -> measured lam {w1/w0:.4f} / {h1/h0:.4f}")
    print(f"  analytic  lam   {lam_x_dep:.4f} / {lam_y_dep:.4f}   "
          f"(measured is {100*(w1/w0/lam_x_dep-1):+.1f}% / {100*(h1/h0/lam_y_dep-1):+.1f}% off -- the excess is "
          f"the tile CORNERS swinging outside the pitch)")

    if args.target_from:
        # A family of load cases sharing ONE target. Sizing a fresh rectangle per row would make
        # each row aim at its own shape, and "the same target under five different problems" would
        # quietly become five different problems AND five different targets.
        cx, cy, half_w, half_h = read_target(args.target_from)
        sized_as = f"COPIED verbatim from {args.target_from}"
    elif args.from_deployed:
        # A concentrated grip does NOT open the sheet uniformly -- lam(theta) is derived for a
        # uniform mechanism and mis-sizes the rectangle badly here (hinge-angle spread ~12 deg
        # against the distributed grip's 2.7). Size from what the sheet actually did instead, and
        # let --pull-in-* set the design headroom explicitly.
        half_w, half_h = args.pull_in_x * w1 / 2.0, args.pull_in_y * h1 / 2.0
        sized_as = (f"from MEASURED deployed bbox x pull-in "
                    f"{args.pull_in_x:.3f} / {args.pull_in_y:.3f}")
    else:
        half_w, half_h = lam_x * w0 / 2.0, lam_y * h0 / 2.0
        sized_as = f"analytic lam at theta = {shape_angle:g} deg"
    if not args.target_from:
        # Anchor: bottom edge ON the deployed bottom edge -- the clamped tiles cannot move away from
        # it, so a target that floats above/below it puts the clamp permanently outside and makes
        # its chamfer error unfixable.
        # In x, the default is the deployed bbox midpoint. On a swinging grip that midpoint drifts
        # with the swing, dragging the target off the axis the sheet is actually pulled along;
        # --center-on-clamp anchors on the clamped faces instead, which are fixed by construction.
        if args.center_on_clamp:
            st, _ = build_closed_initial_state(config)
            clamped = config.topology.get('bc_clamped')
            if not isinstance(clamped, list) or not clamped:
                raise SystemExit("--center-on-clamp needs boundary_conditions.clamped_faces to be "
                                 "an explicit face list")
            cx = float(np.mean(np.asarray(st.face_centroids)[[int(f) for f in clamped], 0]))
            anchored = f"x on the CLAMPED faces {clamped} (fixed axis)"
        else:
            cx = 0.5 * (depl[:, 0].min() + depl[:, 0].max())
            anchored = "x on the deployed bbox midpoint"
        cy = depl[:, 1].min() + half_h
        print(f"    anchored: {anchored}, bottom edge on the deployed bottom")
    print(f"\n  TARGET RECTANGLE ({sized_as})  "
          f"half_w {half_w:.4f}  half_h {half_h:.4f}  center ({cx:.4f}, {cy:.4f})")
    print(f"    pull-in the design must supply: {100*(1 - 2*half_w/w1):+.1f}% in x, "
          f"{100*(1 - 2*half_h/h1):+.1f}% in y  (negative = the target is LARGER than the deploy, "
          f"i.e. no headroom)")
    print(f"    spans x [{cx-half_w:+.4f}, {cx+half_w:+.4f}]  y [{cy-half_h:+.4f}, {cy+half_h:+.4f}]")
    print(f"    vs deployed x [{depl[:,0].min():+.4f}, {depl[:,0].max():+.4f}]  "
          f"y [{depl[:,1].min():+.4f}, {depl[:,1].max():+.4f}]")

    print("\n  ── paste into the config ──────────────────────────────────────────")
    print(f"  target_cx: {cx:.4f}\n  target_cy: {cy:.4f}\n"
          f"  target_half_w: {half_w:.4f}\n  target_half_h: {half_h:.4f}")
    for l in topo.get('loads', []):
        print(f"  - {{face: {l['face']}, dof: {l['dof']}, value: {float(l['value'])*scale:.2f}}}")
    print()

    if args.write_config:
        drive = (f"--load-scale {args.load_scale:g}" if args.load_scale is not None
                 else f"--criterion bbox --bbox-target {args.bbox_target:g}"
                 if args.criterion == "bbox"
                 else f"--angle {args.angle:g} --shape-angle {shape_angle:g} "
                      f"--criterion {args.criterion}")
        sizing = (f"      --target-from {args.target_from}" if args.target_from
                  else f"      --from-deployed{' --derotate' if args.derotate else ''} "
                       f"--pull-in-x {args.pull_in_x:g} "
                       f"--pull-in-y {args.pull_in_y:g}" if args.from_deployed
                  else "      (target sized from the analytic lam)") \
            + (" --center-on-clamp" if args.center_on_clamp else "")
        bbox_line = (f"  deployed bbox {w1:.4f} x {h1:.4f} u"
                     + (f"  (DE-ROTATED by {-swing:+.2f} deg; the raw bbox is inflated "
                        f"by the swing)\n" if args.derotate else "\n"))
        header = (
            "═══ GENERATED by calibrate_target.py — do not hand-edit the four target_* keys "
            "or the load values ═══\n"
            "\n"
            f"  calibrate_target.py --config {args.config} {drive} \\\n"
            + sizing
            + f" \\\n      --init-noise {noise_eff:g} --init-seed {seed_eff}\n"
            "\n"
            f"  load          x{scale:.4f}  ->  " + ", ".join(f"{b*scale:.2f} N" for b in base) + "\n"
            f"  hinge angle   max {theta.max():.2f}  mean {theta.mean():.2f}  min {theta.min():.2f}"
            f"  spread {theta.max()-theta.min():.2f} deg\n"
            f"  swing         {swing:+.2f} deg (mean face rotation; large = rigid-body mode)\n"
            + bbox_line
            + f"  target        {sized_as}\n"
            f"                half_w {half_w:.4f}  half_h {half_h:.4f}  "
            f"center ({cx:.4f}, {cy:.4f})\n"
            f"  pull-in the design must supply: {100*(1 - 2*half_w/w1):+.1f}% in x, "
            f"{100*(1 - 2*half_h/h1):+.1f}% in y\n")
        overrides = {("tessellation", "init_noise"): f"{noise_eff:g}",
                     ("tessellation", "init_seed"): f"{seed_eff}"}
        n = write_config(args.config, args.write_config, target=(cx, cy, half_w, half_h),
                         load_scale=scale, header=header, overrides=overrides)
        print(f"  wrote {args.write_config}  ({n} load line(s) rescaled)\n")


if __name__ == "__main__":
    main()
