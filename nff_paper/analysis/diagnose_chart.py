"""Stage-0 chart ablation diagnostic.

Phase 0 (geometry only, no physics, seconds): the chart's tangent rank, its hinge-gap
content, the analytic dimension scaling, and the flat-sheet validity rate as a function of
geometric step size.

    JAX_PLATFORMS=cpu conda run -n kgnn_mac python nff_paper/analysis/diagnose_chart.py \
        --config closed/sheet_4x8ft_rom --phase0

Writes ``<out>/phase0.npz`` and prints the falsifiable predictions with a PASS/FAIL verdict.
"""

import argparse
import json
import os
import subprocess

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from nff.config.experiment import load_and_parse_config
from nff.closed.chart_ablation import (
    chart_basis, chart_dims, chart_jacobian, closed_start_geometry, flat_validity,
    geom_coords, geom_shape, hinge_gap_operator, rms_displacement,
)
from nff.closed.hinge_path_ensemble import resolve_length_scale
from nff.stages.mapping import apply_closed_les_mapping


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def phase0(config, n_dir: int = 200, n_steps: int = 24, seed: int = 0,
           h_max_frac: float = 0.5) -> dict:
    """The cheap decisive checks. Predictions are stated before they are measured."""
    length_scale = resolve_length_scale(config)
    M, N = int(config.topology['M']), int(config.topology['N'])

    y0, params0, static_features, state, tess = closed_start_geometry(config)
    n_faces, max_nodes = geom_shape(state)
    n_nodes = n_faces * max_nodes
    n_geom = y0.size

    J, theta0, unravel, _ = chart_jacobian(config)
    Q_A, k_A, svals = chart_basis(J)
    B = hinge_gap_operator(state)

    dims = chart_dims(M, N)
    tear_ratio = float(np.linalg.norm(B @ J) / max(np.linalg.norm(J), 1e-300))

    rng = np.random.default_rng(seed)
    D = rng.normal(size=(4000, n_geom))
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    # The tearing content of a direction is its ORTHOGONAL PROJECTION onto row(B), whose
    # expectation is rank(B)/n_geom. ||Bd||^2 overstates it by 2x -- each row e_a - e_b has
    # norm sqrt(2), so B is not a projector.
    _, sb, Vtb = np.linalg.svd(B, full_matrices=False)
    rank_B = int((sb > 1e-10 * sb[0]).sum())
    P_B = Vtb[:rank_B].T                       # row(B) lives in R^n_geom, so use V not U
    tear_frac_mc = float(np.mean(np.sum((D @ P_B) ** 2, axis=1)))
    btb_ratio_mc = float(np.mean(np.sum((D @ B.T) ** 2, axis=1)))

    print(f"\n  sheet {M}x{N}: {n_faces} panels, {B.shape[0] // 2} hinges, "
          f"max_nodes {max_nodes}, length_scale {length_scale:.4g} mm/unit")
    print(f"  geometric DOF n_geom = {n_geom}   raw chart params P = {theta0.size}")

    print("\n  ── Phase-0 predictions ─────────────────────────────────────────────")
    p1 = k_A == dims['k_A']
    print(f"  1  rank(J_A) = {k_A}, predicted {dims['k_A']}                         "
          f"{'PASS' if p1 else 'FAIL'}")
    p2 = tear_ratio < 1e-8
    print(f"  2  ||B J_A|| / ||J_A|| = {tear_ratio:.3e}, predicted < 1e-8          "
          f"{'PASS' if p2 else 'FAIL'}")
    p3 = abs(tear_frac_mc - dims['hinge_gap_fraction']) < 0.02 and rank_B == 2 * dims['n_hinge']
    print(f"  3  rank(B) = {rank_B} of {2 * dims['n_hinge']}; E_d[||P_B d||^2] = "
          f"{tear_frac_mc:.4f}, predicted {dims['hinge_gap_fraction']:.4f}   "
          f"{'PASS' if p3 else 'FAIL'}")
    print(f"     (E_d[||Bd||^2] = {btb_ratio_mc:.4f} vs {dims['expected_BtB_ratio']:.4f} "
          f"-- 2x larger, B is not a projector)")
    p4 = (dims['n_faces'] == n_faces) and (dims['n_hinge'] == B.shape[0] // 2) \
        and (dims['n_geom'] == n_geom)
    print(f"  4  chart_dims vs builder (faces/hinges/n_geom)                     "
          f"{'PASS' if p4 else 'FAIL'}")

    # ── validity vs geometric step ────────────────────────────────────────────────
    # Arm A perturbs the chart parameters; arms B and C perturb the panel nodes directly.
    # B and C share this curve exactly: a penalty reweights the loss, it does not shrink
    # the reachable set -- which is precisely why it cannot give a guarantee.
    #
    # The sweep is bounded to steps below ``h_max_frac`` of the panel pitch. Beyond that the
    # perturbation is comparable to the sheet itself, which is a region no optimizer visits;
    # sampling it says nothing about the design problem and mixes in a separate open question
    # about the construction's behaviour at extreme non-uniform aspect ratios.
    panel_pitch = length_scale * float(config.topology.get('spacing', 1.0))
    h_max = h_max_frac * panel_pitch
    scales = np.logspace(-2.5, 1.5, n_steps)
    records = {'A': [], 'BC': []}

    y_of_theta = jax.jit(lambda flat: geom_coords(
        apply_closed_les_mapping(state, unravel(flat), static_features)))

    y0_np = np.asarray(y0)
    dirs_A = rng.normal(size=(n_dir, theta0.size))
    dirs_A /= np.linalg.norm(dirs_A, axis=1, keepdims=True)
    for d in dirs_A:
        for s in scales:
            y = np.asarray(y_of_theta(theta0 + s * d))
            h = rms_displacement(y, y0_np, n_nodes, length_scale)
            if h > h_max:
                continue
            v = flat_validity(y, state, B=B)
            records['A'].append((h, float(v['connected']), float(v['non_overlapping']),
                                 float(v['non_inverted']), v['max_hinge_gap'] * length_scale,
                                 float(v['finite'])))

    # Arm B/C directions live in y itself; ||d|| = 1 so h = s * length_scale exactly.
    dirs_B = rng.normal(size=(n_dir, n_geom))
    dirs_B /= np.linalg.norm(dirs_B, axis=1, keepdims=True)
    for d in dirs_B:
        for s in scales:
            y = y0_np + s * d * np.sqrt(n_nodes)
            h = rms_displacement(y, y0_np, n_nodes, length_scale)
            if h > h_max:
                continue
            v = flat_validity(y, state, B=B)
            records['BC'].append((h, float(v['connected']), float(v['non_overlapping']),
                                  float(v['non_inverted']), v['max_hinge_gap'] * length_scale,
                                  float(v['finite'])))

    out = {'git_sha': _git_sha(), 'M': M, 'N': N, 'n_faces': n_faces, 'max_nodes': max_nodes,
           'n_geom': n_geom, 'P_raw': int(theta0.size), 'k_A': k_A, 'length_scale': length_scale,
           'n_hinges': B.shape[0] // 2, 'rank_B': rank_B, 'tear_ratio': tear_ratio,
           'tear_frac_mc': tear_frac_mc, 'btb_ratio_mc': btb_ratio_mc,
           'dims': dims, 'svals': svals, 'J': J, 'Q_A': Q_A, 'B': B, 'y0': y0_np,
           'cols': 'h_mm, connected, non_overlapping, non_inverted, max_gap_mm, finite',
           'h_max_mm': h_max, 'panel_pitch_mm': panel_pitch,
           'valid_A': np.array(records['A']), 'valid_BC': np.array(records['BC']),
           'predictions': {'rank': bool(p1), 'tear_ratio': bool(p2),
                           'tear_fraction': bool(p3), 'dims': bool(p4)}}

    # Three failure modes, reported separately: they occur at different scales, and only
    # connectivity is removed by the chart as an exact identity.
    edges = np.logspace(np.log10(0.5), np.log10(h_max), 12)
    print("\n  ── flat-sheet validity vs geometric step (rate per bin) ───────────")
    print("  h [mm]                     " + " ".join(f"{e:>5.0f}" for e in edges[:-1]))
    for arm, key in (('A  (closed_les)', 'valid_A'), ('B/C (free vertices)', 'valid_BC')):
        rows = out[key]
        for label, col in (('connected', 1), ('non-overlapping', 2), ('non-inverted', 3)):
            line = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
                line.append(f"{rows[m, col].mean():5.2f}" if m.sum() >= 5 else "    .")
            print(f"  {arm:20s} {label:16s}" + " ".join(line))
        print()

    print("  ── max hinge gap [mm] vs step -- threshold-free ───────────────────")
    for arm, key in (('A  (closed_les)', 'valid_A'), ('B/C (free vertices)', 'valid_BC')):
        rows = out[key]
        line = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
            line.append(f"{rows[m, 4].mean():5.0e}" if m.sum() >= 5 else "    .")
        print(f"  {arm:22s} " + " ".join(line))

    gap_A = float(np.nanmax(out['valid_A'][:, 4]))
    p5 = gap_A < 1e-6 * panel_pitch
    print(f"\n  5  arm A max hinge gap over the WHOLE sweep = {gap_A:.3e} mm "
          f"(pitch {panel_pitch:.0f} mm)  {'PASS' if p5 else 'FAIL'}")
    out['predictions']['connectivity_exact'] = bool(p5)
    out['max_gap_A_mm'] = gap_A
    return out


def phase1(config, out_dir: str, eps_list=(3e-4, 1e-3, 3e-3)) -> dict:
    """Conditioning, all of it on the common geometric metric.

    One geometric Hessian ``H_y`` is formed at the shared reference design; each chart enters
    through the orthonormal basis of its own tangent image, so the operators compared act on
    subspaces of the same Euclidean space in the same units.
    """
    from nff.scripts.diagnostics.diagnose_landscape import fd_hessian, spectrum_stats
    from nff.closed.chart_ablation import build_arm_loss_fn, build_geom_loss_fn

    length_scale = resolve_length_scale(config)
    loss_geom, y0, state, _ = build_geom_loss_fn(config)
    J, theta0, _, _ = chart_jacobian(config)
    Q_A, k_A, _ = chart_basis(J)
    B = hinge_gap_operator(state)

    print("\n  ── geometric Hessian H_y (72x72) at the shared reference design ───")
    H_by_eps = {}
    for eps in eps_list:
        H, _, _ = fd_hessian(loss_geom, y0, eps=eps)
        H_by_eps[eps] = H
    H_y = H_by_eps[eps_list[len(eps_list) // 2]]

    def _stats(H, label):
        s = spectrum_stats(H)
        print(f"  {label:34s} cond_eff {s['cond']:.3e}  participation {s['participation']:6.2f}  "
              f"lam_max {s['lam_max']:.3e}  n_neg_stiff {s['n_neg_stiff']:3d}  n_flat {s['n_flat']:3d}")
        return s

    print()
    s_full = _stats(H_y, "B/C  free vertices (72 dof)")
    H_A = Q_A.T @ H_y @ Q_A
    s_A = _stats(H_A, f"A    closed chart ({k_A} dof)")

    # eps-stability: quote no condition number that moves with the finite-difference step.
    print("\n  ── FD-step stability ──────────────────────────────────────────────")
    for eps, H in H_by_eps.items():
        sf = spectrum_stats(H)
        sa = spectrum_stats(Q_A.T @ H @ Q_A)
        print(f"    eps {eps:.0e}   cond_eff  full {sf['cond']:.3e}   chart {sa['cond']:.3e}"
              f"   lam_max full {sf['lam_max']:.4e}")

    # Does the chart keep the curvature that matters, or throw it away?
    w, V = np.linalg.eigh(H_y)
    order = np.argsort(-np.abs(w))
    captured = {}
    for m in (1, 3, 5, 10, 20):
        idx = order[:m]
        num = float(np.sum(np.abs(w[idx]) * np.sum((Q_A.T @ V[:, idx]) ** 2, axis=0)))
        captured[m] = num / float(np.sum(np.abs(w[idx])))
    print("\n  ── does the chart keep the useful curvature? ──────────────────────")
    print("    top-m modes of H_y captured by the chart image: " +
          "  ".join(f"m={m}: {c:.3f}" for m, c in captured.items()))
    angles = np.rad2deg(np.arccos(np.clip(
        np.linalg.svd(Q_A.T @ V[:, order[:k_A]], compute_uv=False), -1, 1)))
    print(f"    principal angles to the top-{k_A} eigenspace [deg]: "
          f"min {angles.min():.1f}  median {np.median(angles):.1f}  max {angles.max():.1f}")

    # The penalty steelman, analytically: hinge_gap is exactly ||By||^2, so H_C(w) is closed form.
    print("\n  ── penalty Pareto (analytic; no retraining) ───────────────────────")
    BtB = B.T @ B
    pareto = []
    for wgt in np.logspace(-3, 6, 10):
        s = spectrum_stats(H_y + 2.0 * wgt * BtB)
        pareto.append((wgt, s['cond'], s['lam_max']))
        print(f"    w {wgt:9.3g}   cond_eff {s['cond']:.3e}   lam_max {s['lam_max']:.3e}")
    print(f"    arm A (no penalty needed)   cond_eff {s_A['cond']:.3e}   "
          f"lam_max {s_A['lam_max']:.3e}")

    # Raw per-arm Hessians -- chart-dependent, reported because it is what Adam sees.
    print("\n  ── raw per-arm Hessians (chart-dependent; NOT the headline) ───────")
    raw = {}
    for arm in ('A', 'B'):
        lf, p0, _, _ = build_arm_loss_fn(config, arm)
        Hr, _, _ = fd_hessian(lf, p0, eps=1e-3)
        raw[arm] = Hr
        sr = spectrum_stats(Hr)
        print(f"    arm {arm}  P={Hr.shape[0]:3d}  cond_eff {sr['cond']:.3e}  "
              f"participation {sr['participation']:.2f}  n_flat {sr['n_flat']}")

    np.savez_compressed(os.path.join(out_dir, "phase1.npz"),
                        H_y=H_y, Q_A=Q_A, B=B, J=J, y0=np.asarray(y0),
                        eigvals_full=w, pareto=np.array(pareto),
                        H_raw_A=raw['A'], H_raw_B=raw['B'],
                        length_scale=length_scale, k_A=k_A, git_sha=_git_sha(),
                        **{f"H_y_eps_{e:.0e}": h for e, h in H_by_eps.items()})
    print(f"\n  wrote {out_dir}/phase1.npz")
    return {'cond_full': s_full['cond'], 'cond_chart': s_A['cond'], 'k_A': k_A,
            'captured': captured, 'pareto': pareto}


def phase2(config, out_dir: str, n_dir: int = 24, n_steps: int = 22,
           h_max_frac: float = 0.5) -> dict:
    """Landscape probes, all on the shared millimetre axis.

    Transects, the deployed-validity sweep and the gradient-reliability probe. Directions are
    scaled by their INDUCED geometric displacement rather than by parameter norm, so the arms
    overlay on one axis (the physical analogue of filter normalisation).
    """
    from nff.closed.chart_ablation import (build_arm_loss_fn, build_geom_loss_fn, deploy_validity,
                                           init_direct_from_closed, lift_geom)

    length_scale = resolve_length_scale(config)
    panel_pitch = length_scale * float(config.topology.get('spacing', 1.0))
    h_max = h_max_frac * panel_pitch

    loss_geom, y0, state, _ = build_geom_loss_fn(config)
    n_faces, max_nodes = geom_shape(state)
    n_nodes = n_faces * max_nodes
    y0_np = np.asarray(y0)

    lossA, pA, stateA, _ = build_arm_loss_fn(config, 'A')
    thetaA, unravelA = __import__('jax.flatten_util', fromlist=['x']).ravel_pytree(pA)
    _, _, sfA, _, _ = closed_start_geometry(config)
    y_of_theta = jax.jit(lambda f: geom_coords(
        apply_closed_les_mapping(state, unravelA(f), sfA)))

    L_geom_j = jax.jit(loss_geom)
    L_A_j = jax.jit(lambda f: lossA(unravelA(f)))
    g_geom = jax.jit(jax.grad(loss_geom))
    g_A = jax.jit(jax.grad(lambda f: lossA(unravelA(f))))

    rng = np.random.default_rng(0)
    steps = np.linspace(-h_max, h_max, n_steps)

    print("\n  ── loss transects on the shared mm axis ───────────────────────────")
    transects = {'A': [], 'B': []}
    for k in range(n_dir):
        dA = rng.normal(size=thetaA.size); dA /= np.linalg.norm(dA)
        # calibrate the parameter step so the induced geometric step is 1 mm, then scale
        probe = np.asarray(y_of_theta(thetaA + 1e-3 * dA))
        rate = rms_displacement(probe, y0_np, n_nodes, length_scale) / 1e-3
        for h in steps:
            f = thetaA + (h / max(rate, 1e-12)) * dA
            y = np.asarray(y_of_theta(f))
            transects['A'].append((rms_displacement(y, y0_np, n_nodes, length_scale) * np.sign(h),
                                   float(L_A_j(f))))
        dB = rng.normal(size=y0_np.size); dB /= np.linalg.norm(dB)
        for h in steps:
            y = y0_np + (h / length_scale) * np.sqrt(n_nodes) * dB
            transects['B'].append((h, float(L_geom_j(jnp.asarray(y)))))
    for arm in ('A', 'B'):
        a = np.array(transects[arm])
        fin = np.isfinite(a[:, 1])
        print(f"    arm {arm}: finite {fin.mean():6.1%}   L range "
              f"[{np.nanmin(a[fin, 1]):.4g}, {np.nanmax(a[fin, 1]):.4g}]")

    print("\n  ── gradient reliability: analytic vs central FD ───────────────────")
    grad_rel = {}
    for arm in ('A', 'B'):
        rows = []
        for k in range(32):
            if arm == 'A':
                d = rng.normal(size=thetaA.size); d /= np.linalg.norm(d)
                gd = float(np.dot(np.asarray(g_A(thetaA)), d))
                f = lambda t: float(L_A_j(thetaA + t * d))
            else:
                d = rng.normal(size=y0_np.size); d /= np.linalg.norm(d)
                gd = float(np.dot(np.asarray(g_geom(y0)), d))
                f = lambda t: float(L_geom_j(jnp.asarray(y0_np + t * d)))
            for eps in (1e-4, 1e-3, 1e-2):
                fd = (f(eps) - f(-eps)) / (2 * eps)
                rel = abs(fd - gd) / max(abs(gd), 1e-30)
                rows.append((eps, gd, fd, rel))
        a = np.array(rows)
        grad_rel[arm] = a
        for eps in (1e-4, 1e-3, 1e-2):
            m = a[:, 0] == eps
            print(f"    arm {arm}  eps {eps:.0e}   median |rel err| {np.median(a[m, 3]):.3e}   "
                  f"p90 {np.percentile(a[m, 3], 90):.3e}")

    print("\n  ── deployed validity vs geometric step ────────────────────────────")
    dep = {'A': [], 'B': []}
    hs = np.logspace(np.log10(1.0), np.log10(h_max), 8)
    for k in range(8):
        dA = rng.normal(size=thetaA.size); dA /= np.linalg.norm(dA)
        probe = np.asarray(y_of_theta(thetaA + 1e-3 * dA))
        rate = rms_displacement(probe, y0_np, n_nodes, length_scale) / 1e-3
        dB = rng.normal(size=y0_np.size); dB /= np.linalg.norm(dB)
        for h in hs:
            pa = unravelA(thetaA + (h / max(rate, 1e-12)) * dA)
            va = deploy_validity(config, pa, state, 'closed_les', sfA)
            dep['A'].append((h, float(va['finite']), float(va['valid'])))
            yb = y0_np + (h / length_scale) * np.sqrt(n_nodes) * dB
            pb = lift_geom(jnp.asarray(yb), n_faces, max_nodes)
            vb = deploy_validity(config, pb, state, 'direct_vertices', None)
            dep['B'].append((h, float(vb['finite']), float(vb['valid'])))
    for arm in ('A', 'B'):
        a = np.array(dep[arm])
        print(f"    arm {arm}: deployed finite {a[:, 1].mean():6.1%}   "
              f"finite AND non-overlapping {a[:, 2].mean():6.1%}")

    np.savez_compressed(os.path.join(out_dir, "phase2.npz"),
                        transect_A=np.array(transects['A']), transect_B=np.array(transects['B']),
                        grad_A=grad_rel['A'], grad_B=grad_rel['B'],
                        deploy_A=np.array(dep['A']), deploy_B=np.array(dep['B']),
                        length_scale=length_scale, h_max=h_max, git_sha=_git_sha())
    print(f"\n  wrote {out_dir}/phase2.npz")
    return {'transects': transects, 'grad_rel': grad_rel, 'deploy': dep}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="closed/sheet_4x8ft_rom")
    ap.add_argument("--out", default="data/outputs/chart_ablation")
    ap.add_argument("--phase0", action="store_true")
    ap.add_argument("--phase1", action="store_true")
    ap.add_argument("--phase2", action="store_true")
    ap.add_argument("--n-dir", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--h-max-frac", type=float, default=0.5,
                    help="sweep bound as a fraction of the panel pitch")
    args = ap.parse_args()

    config = load_and_parse_config(f"data/configs/{args.config}.yaml")
    os.makedirs(args.out, exist_ok=True)

    if args.phase0:
        res = phase0(config, n_dir=args.n_dir, seed=args.seed, h_max_frac=args.h_max_frac)
        np.savez_compressed(os.path.join(args.out, "phase0.npz"),
                            **{k: v for k, v in res.items()
                               if isinstance(v, (np.ndarray, int, float, str))})
        with open(os.path.join(args.out, "phase0_summary.json"), "w") as fh:
            json.dump({k: v for k, v in res.items()
                       if not isinstance(v, np.ndarray)}, fh, indent=2, default=float)
        verdict = all(res['predictions'].values())
        print(f"\n  wrote {args.out}/phase0.npz")
        print(f"  PHASE 0 {'PASSES -- proceed' if verdict else 'FAILS -- STOP and rethink'}")

    if args.phase1:
        phase1(config, args.out)

    if args.phase2:
        phase2(config, args.out)


if __name__ == "__main__":
    main()
