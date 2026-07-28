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

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np

from nff.config.experiment import load_and_parse_config
from nff.closed.setup import build_closed_initial_state, init_closed_les_params
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


def deploy(config, load_scale: float, n_steps: int = 12):
    """Deploy the untrained (uniform) design at ``load_scale`` x the config loads.

    Returns (theta_rel_deg per hinge, flat boundary cloud, deployed boundary cloud).
    """
    c = copy.deepcopy(config)
    c.physics.num_load_steps = int(n_steps)
    c.topology['loads'] = [{**dict(l), 'value': float(l['value']) * load_scale}
                           for l in c.topology.get('loads', [])]
    # loads are baked into the state at build time -- scale BEFORE building, not after
    state, _ = build_closed_initial_state(c)
    c.topology['init_noise'] = 0.0
    c.topology['init_seed'] = 0
    params, sf = init_closed_les_params(c)
    res = forward_pipeline(state, c.target, c.validity, c.physics, map_type=c.mapping.type,
                           map_params=params, static_features=sf, load_specs=c.topology['loads'])
    vs = res['valid_state']
    disp = np.asarray(res['solution'].fields[-1])
    # relative rotation across each hinge = theta_2 - theta_1 of the two faces it bonds
    bc = np.asarray(vs.bond_connectivity)
    n_nodes = np.asarray(vs.centroid_node_vectors).shape[1]
    th = np.asarray(disp[:, 2])
    theta_rel = np.degrees(np.abs(th[bc[:, 1] // n_nodes] - th[bc[:, 0] // n_nodes]))
    return theta_rel, _boundary_cloud(vs, np.zeros_like(disp)), _boundary_cloud(vs, disp)


def solve_load_for_angle(config, angle_deg: float, *, n_steps: int = 12, tol: float = 0.05,
                         max_iter: int = 40, criterion: str = "max", verbose: bool = True):
    """Bisect the load scale until the relative hinge rotation reaches ``angle_deg``.

    ``criterion='max'`` (default) drives the WORST hinge to the angle. That is the right reading of
    a deployment-angle spec: it is a CEILING the material imposes, not a value every hinge must
    hit. The distinction is invisible when the sheet opens uniformly (the distributed 3-tile grip
    holds every hinge inside 2.3 deg, so mean and max agree to half a degree) and decisive when it
    does not -- a single-tile grip funnels everything through one column, and bisecting on the MEAN
    there asks the loaded hinges to go to 80+ deg, tearing the sheet apart, to drag the mean up.

    ``criterion='mean'`` is kept for the uniform case, where it is the sheet-level number.
    """
    reduce = (lambda t: t.max()) if criterion == "max" else (lambda t: t.mean())
    lo, hi = 1e-3, 1.0
    theta_hi, _, _ = deploy(config, hi, n_steps)
    while reduce(theta_hi) < angle_deg and hi < 1e5:      # bracket from above first
        hi *= 4.0
        theta_hi, _, _ = deploy(config, hi, n_steps)
    if reduce(theta_hi) < angle_deg:
        raise SystemExit(f"the mechanism saturates below {angle_deg} deg "
                         f"(reached {reduce(theta_hi):.2f} deg at {hi:g}x load) -- it cannot get there")
    best = None
    for _ in range(max_iter):
        mid = np.sqrt(lo * hi)                           # geometric bisection: load spans decades
        theta, flat, depl = deploy(config, mid, n_steps)
        best = (mid, theta, flat, depl)
        if verbose:
            print(f"    load x{mid:9.4f}  theta_{criterion} {reduce(theta):6.2f} deg  "
                  f"(mean {theta.mean():.2f}, spread {theta.max()-theta.min():.2f})")
        if abs(reduce(theta) - angle_deg) < tol:
            break
        lo, hi = (mid, hi) if reduce(theta) < angle_deg else (lo, mid)
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--angle", type=float, default=45.0, help="deployment angle [deg]")
    p.add_argument("--criterion", choices=("max", "mean"), default="max",
                   help="drive the WORST hinge to --angle (max, default: the angle is a material "
                        "CEILING) or the sheet average (mean)")
    p.add_argument("--steps", type=int, default=12, help="load steps in each trial deploy")
    p.add_argument("--tol", type=float, default=0.05, help="angle tolerance [deg]")
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

    lam_x = opening_ratio(psi_x, args.angle)
    lam_y = opening_ratio(psi_y, args.angle)
    print(f"\n  OPENING RATIO at theta = {args.angle:g} deg:  lam_x = {lam_x:.4f}   lam_y = {lam_y:.4f}")

    print(f"\n  solving for the load whose {args.criterion.upper()} hinge angle is "
          f"{args.angle:g} deg ...")
    scale, theta, flat, depl = solve_load_for_angle(config, args.angle, n_steps=args.steps,
                                                    tol=args.tol, criterion=args.criterion)
    base = [float(l['value']) for l in topo.get('loads', [])]
    print(f"\n  LOAD  x{scale:.4f}  ->  " + ", ".join(f"{b*scale:.2f} N" for b in base))
    print(f"    hinge angle: max {theta.max():.2f}  mean {theta.mean():.2f}  min {theta.min():.2f}"
          f"  spread {theta.max()-theta.min():.2f} deg")

    w0 = flat[:, 0].max() - flat[:, 0].min()
    h0 = flat[:, 1].max() - flat[:, 1].min()
    w1 = depl[:, 0].max() - depl[:, 0].min()
    h1 = depl[:, 1].max() - depl[:, 1].min()
    print(f"\n  flat sheet      {w0:.4f} x {h0:.4f} u")
    print(f"  deployed bbox   {w1:.4f} x {h1:.4f} u   -> measured lam {w1/w0:.4f} / {h1/h0:.4f}")
    print(f"  analytic  lam   {lam_x:.4f} / {lam_y:.4f}   "
          f"(measured is {100*(w1/w0/lam_x-1):+.1f}% / {100*(h1/h0/lam_y-1):+.1f}% off -- the excess is "
          f"the tile CORNERS swinging outside the pitch)")

    half_w, half_h = lam_x * w0 / 2.0, lam_y * h0 / 2.0
    # Anchor: centred in x on the deployed sheet, and bottom edge ON the deployed bottom edge --
    # the clamped tiles cannot move away from it, so a target that floats above/below it puts the
    # clamp permanently outside and makes its chamfer error unfixable.
    cx = 0.5 * (depl[:, 0].min() + depl[:, 0].max())
    cy = depl[:, 1].min() + half_h
    print(f"\n  TARGET RECTANGLE  half_w {half_w:.4f}  half_h {half_h:.4f}  "
          f"center ({cx:.4f}, {cy:.4f})")
    print(f"    spans x [{cx-half_w:+.4f}, {cx+half_w:+.4f}]  y [{cy-half_h:+.4f}, {cy+half_h:+.4f}]")
    print(f"    vs deployed x [{depl[:,0].min():+.4f}, {depl[:,0].max():+.4f}]  "
          f"y [{depl[:,1].min():+.4f}, {depl[:,1].max():+.4f}]")

    print("\n  ── paste into the config ──────────────────────────────────────────")
    print(f"  target_cx: {cx:.4f}\n  target_cy: {cy:.4f}\n"
          f"  target_half_w: {half_w:.4f}\n  target_half_h: {half_h:.4f}")
    for l in topo.get('loads', []):
        print(f"  - {{face: {l['face']}, dof: {l['dof']}, value: {float(l['value'])*scale:.2f}}}")
    print()


if __name__ == "__main__":
    main()
