"""Deploy ONE hinge in CalculiX and cache every frame, for the 3D hinge renders.

The frames feed :mod:`nff_paper.figures.render_hinge_3d` and the case-study insets in
:mod:`nff_paper.figures.plot_compressed_hinge`. The oracle is driven through the SAME
``solver_kwargs`` the dataset campaign uses, so a rendered hinge is the geometry the surrogate was
actually trained on -- not a look-alike re-parameterisation.

Two drive modes, and the choice decides what the render is allowed to claim:

**Pure rotation** (default; ``eta_a = eta_s = 0``) is the spine the deployed pipeline rides and the
slice the energy figure draws. It answers "what does a hinge at this rotation look like".

    conda run -n ccx python nff_paper/figures/deploy_hinge_frames.py \
        --w-lig 25 --theta 45 --material PET --thickness 0.5 --r-win 100

**Path replay** (``--paths``/``--hinge``) drives the oracle along the MEASURED ``(a, s, theta)``
polyline one ligament actually rode in a deployed sheet, at that ligament's OWN ``w_lig`` and
``alpha``, via :class:`DeploymentPath`. It answers "what did THIS ligament do".

    conda run -n ccx python nff_paper/figures/deploy_hinge_frames.py \
        --paths data/outputs/hinge_paths/pet45_5x5_base/hinge_paths --hinge 19 \
        --material PET --thickness 0.5 --r-win 100

⚠ WHY THE DISTINCTION IS NOT COSMETIC. A pure-rotation frame carries zero axial and zero shear, so
it is BLIND to the strain components a hinge-path objective acts on. Two designs whose ligaments
differ 16x in ``eta_s`` pick adjacent frames of the same rotation sweep and render identically --
measured on the 5x5 pair, 13.63% vs 13.73% of nodes past yield. A render used to COMPARE two designs
must be a replay; only a render illustrating the rotation spine may be a ray.

The oracle is elastoplastic, so ``W`` and the accumulated damage are path-dependent: replaying the
real route is the only way the plastic strain in the picture is the plastic strain the ligament has.

★ NODAL PEEQ is saved as ``peeq_nodal`` (n_frames, n_nodes) whenever CalculiX returns it -- the
field a damage render actually needs. The pre-existing ``peeq`` key stays the per-frame ligament
SCALAR it always was, so older readers are unaffected. Von Mises saturates past yield on a flat flow
curve like PET's and cannot separate two states of different accumulated plastic strain; paint
``peeq_nodal`` instead wherever it is present.
"""

import argparse
import os

import numpy as np

from nff.rve.ccx_solver import deploy
from nff.rve.damage import _nodal_peeq
from nff.rve.hinge_function import (HingeConstants, HingeGeometry, DeploymentRay, DeploymentPath,
                                    to_rve_params, solver_kwargs)


def von_mises(S):
    """Nodal von Mises stress [MPa] from the CalculiX Cauchy tensor ``[sxx, syy, szz, sxy, syz, szx]``."""
    S = np.asarray(S, float)
    sxx, syy, szz, sxy, syz, szx = (S[:, k] for k in range(6))
    return np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                   + 3.0 * (sxy ** 2 + syz ** 2 + szx ** 2))


def _resample(polyline, n):
    """Resample a ``(k, 3)`` polyline to ``n`` states by ARC LENGTH in ``(a, s, theta)``.

    Load-step spacing is not path spacing: a ligament that stalls mid-deployment leaves many states
    piled in one place and few across the stretch that matters, so for a FIGURE the arc-length
    spacing is the honest one.

    ⚠⚠ IT IS THE WRONG THING TO DRIVE A SOLVE WITH, and the default is 0 for that reason. The
    pipeline's own stepping is gentle at the start precisely where the ligament's buckling mode is
    being established; arc length skips that dead zone and hands CalculiX a first increment from
    REST to a fully developed state. Measured on ligament 19 of the 5x5 reference: ``--resample 30``
    put state 1 at ``theta = 8.48 deg`` against the measured first step's ``0.077 deg``, and step 1
    alone never converged (cutbacks 7.9e-2 -> 1.2e-3, no field output at all). Replaying the
    polyline verbatim converges.

    Note also that the metric mixes units -- ``a``, ``s`` in mm against ``theta`` in rad -- so
    "arc length" here is dominated by whichever component happens to be numerically larger.
    """
    p = np.asarray(polyline, float)
    if n <= 0 or n >= len(p):
        return p
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    t = np.concatenate([[0.0], np.cumsum(seg)])
    if t[-1] <= 0:
        return p
    grid = np.linspace(0.0, t[-1], n + 1)[1:]          # drop 0: the origin is implicit
    return np.stack([np.interp(grid, t, p[:, c]) for c in range(3)], axis=1)


def _load_polyline(stem, hinge):
    """``(polyline, w_lig, alpha_deg)`` for one ligament of a saved hinge-path archive.

    Read as a plain npz rather than through ``nff.closed.hinge_paths.load_paths``: that module
    imports JAX at module level and this script runs in the ``ccx`` environment, which has none.
    The archive layout is the one ``save_paths`` writes.

    Step 0 is dropped -- it is the undeformed origin, which :class:`DeploymentPath` takes as
    implicit at ``t = 0``.
    """
    z = np.load(stem + ".npz")
    u = np.asarray(z["u"], float)
    if not 0 <= hinge < u.shape[1]:
        raise SystemExit(f"--hinge {hinge} out of range: the archive holds {u.shape[1]} ligaments")
    return u[1:, hinge, :], float(z["w_lig"][hinge]), float(np.degrees(z["alpha"][hinge]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--w-lig", type=float, default=None,
                    help="ligament width [mm]; required for a ray, taken from the archive on replay")
    ap.add_argument("--theta", type=float, default=45.0, help="full-deployment rotation [deg]")
    ap.add_argument("--alpha", type=float, default=None,
                    help="angle between the two cuts [deg]; taken from the archive on replay")
    ap.add_argument("--material", default="PET")
    ap.add_argument("--thickness", type=float, default=0.5, help="sheet gauge [mm]")
    ap.add_argument("--r-win", type=float, default=100.0, help="Saint-Venant window radius [mm]")
    ap.add_argument("--fillet-ratio", type=float, default=0.16)
    ap.add_argument("--n-steps", type=int, default=30, help="prescribed states along the ray")
    ap.add_argument("--paths", default=None,
                    help="hinge-path archive STEM (no .npz): replay the MEASURED (a, s, theta) "
                         "polyline of --hinge instead of a pure-rotation ray")
    ap.add_argument("--hinge", type=int, default=None, help="ligament index within --paths")
    ap.add_argument("--resample", type=int, default=0,
                    help="on replay, resample the polyline to this many states by arc length; "
                         "0 keeps every solved step")
    ap.add_argument("--ncpus", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=5400)
    ap.add_argument("--out", default=None, help="npz path; default data/outputs/hinge_frames_<tag>.npz")
    args = ap.parse_args()

    if (args.paths is None) != (args.hinge is None):
        ap.error("--paths and --hinge go together")

    if args.paths:
        polyline, w_lig, alpha_deg = _load_polyline(args.paths, args.hinge)
        polyline = _resample(polyline, args.resample)
        # The archive is the authority on the ligament's own geometry; an explicit flag still wins,
        # so a what-if at a different width stays available.
        w_lig = args.w_lig if args.w_lig is not None else w_lig
        alpha_deg = args.alpha if args.alpha is not None else alpha_deg
        ray = DeploymentPath(polyline=polyline)
        # ⚠ THE SOURCE ARCHIVE MUST BE IN THE TAG. The tag is the scratch workdir name, so two
        # replays of the SAME hinge index from two designs -- exactly the comparison this mode
        # exists for -- would otherwise share `/tmp/hinge_render/<tag>` and clobber each other's
        # deck and .frd when run concurrently.
        src = os.path.basename(os.path.dirname(os.path.abspath(args.paths))) or "paths"
        tag = f"{str(args.material).lower()}_{src}_h{args.hinge}"
    else:
        if args.w_lig is None:
            ap.error("--w-lig is required unless replaying a path")
        w_lig, alpha_deg = args.w_lig, (args.alpha if args.alpha is not None else 90.0)
        ray = DeploymentRay(theta1_deg=args.theta, eta_a=0.0, eta_s=0.0, n_steps=args.n_steps)
        tag = f"{str(args.material).lower()}_w{w_lig:g}"
    tag = tag.replace(".", "p")

    const = HingeConstants(thickness=args.thickness, w_c=0.2, r_win=args.r_win,
                           fillet_ratio=args.fillet_ratio, material=args.material,
                           n_through=2, lc_fillet_frac=0.4, lc_min_floor=0.06,
                           min_inc=1e-3, field_freq=1000, el_fields="PEEQ, S",
                           stop_at_fracture=False)     # PET folds never reach eps_f -- the poll is pure cost
    geo = HingeGeometry(w_lig=w_lig, alpha_deg=alpha_deg, fillet_ratio=args.fillet_ratio)
    out = args.out or f"data/outputs/hinge_frames_{tag}.npz"

    r = deploy(to_rve_params(geo, const), ncpus=args.ncpus, timeout=args.timeout,
               eps_f=None, workdir=f"/tmp/hinge_render/{tag}", **solver_kwargs(geo, ray, const))

    frames, xyz, conn = r["frames"], r["xyz"], r["conn"]
    disp = np.array([f["DISP"] for f in frames])
    vm = np.array([von_mises(f["STRESS"]) if np.size(f.get("STRESS", [])) else
                   np.zeros(len(xyz)) for f in frames])
    # NODAL plastic strain, the field the damage renders want. A frame CalculiX truncated can be
    # missing it; those render blank rather than aborting a solve that is otherwise complete.
    peeq_nodal = np.array([(lambda p: np.zeros(len(xyz)) if p is None or len(p) != len(xyz) else p)
                           (_nodal_peeq(f)) for f in frames])

    extra = {}
    if args.paths:
        # The DRIVEN route, kept with the frames: without it a replay archive is indistinguishable
        # from a ray archive, and the render would silently be re-labelled as a rotation slice.
        extra = dict(driven_states=np.asarray(ray.polyline, float), source_paths=str(args.paths),
                     source_hinge=int(args.hinge))
    np.savez_compressed(out, xyz=xyz, conn=conn, disp=disp, vm=vm, peeq_nodal=peeq_nodal,
                        theta_deg=r["theta_deg"], damage=r["damage"], peeq=r["peeq_lig"],
                        uz_max=r["uz_max"], w_lig=w_lig, alpha_deg=alpha_deg,
                        thickness=args.thickness, r_win=args.r_win, material=str(args.material),
                        drive=("path" if args.paths else "ray"), **extra)
    have_peeq = float(np.nanmax(peeq_nodal)) if peeq_nodal.size else float("nan")
    print(f"w_lig={w_lig:g} alpha={alpha_deg:.1f}deg t={args.thickness:g} r_win={args.r_win:g}  "
          f"drive={'path' if args.paths else 'ray'}  "
          f"{len(frames)} frames  stop={r['stop_reason']}  theta<={r['theta_deg'][-1]:.1f}deg  "
          f"uz_max={r['uz_max'][-1]:.2f}mm (uz/t={r['uz_max'][-1] / args.thickness:.1f})  "
          f"vm_max={vm.max():.1f}MPa  peeq_nodal_max={have_peeq:.4f}  "
          f"D_max={np.nanmax(r['damage']):.4f}  -> {out}")


if __name__ == "__main__":
    main()
