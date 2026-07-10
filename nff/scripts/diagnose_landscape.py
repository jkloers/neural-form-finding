"""Outer design-loss landscape + Hessian conditioning diagnostic.

Sibling of ``diagnose_conditioning.py``. That script probes the INNER physics
solve (the total-potential-energy Hessian at static equilibrium). This one
probes the OUTER inverse-design problem: the loss ``L(theta)`` the optimizer
actually descends, as a function of the design parameters ``theta`` (for
``closed_les``: ``{z, bnd_logits, w_lig_logit}``; for the GNN pipeline: the
network weights).

It answers two questions a differentiable-form-finding paper needs:

  1. WHAT DOES THE LANDSCAPE LOOK LIKE?  A 2D slice of ``L`` around the trained
     design (default axes = the two stiffest Hessian eigenvectors, so the plot
     shows the most-curved subspace) plus a 1D "transect spaghetti" over many
     random directions — smooth/monotone transects are stronger evidence of a
     benign landscape than any single hand-picked 2D slice.

  2. HOW IS IT CONDITIONED?  The full eigenspectrum of the design-loss Hessian:
       cond = lam_max / lam_min+        overall anisotropy (hard-to-optimize)
       # lam < 0                        indefiniteness (plastic-softening saddle;
                                        this is what physics.backward_reg fixes)
       participation ratio             effective # of directions that matter
     and a render of the stiffest eigenvector as a design perturbation — for the
     open pipeline this direction typically points along a constraint violation;
     in the closed chart such a direction does not exist.

PIPELINE-AGNOSTIC BY CONSTRUCTION.  Everything downstream of ``loss_fn`` is
written against a flat vector via ``ravel_pytree`` and knows nothing about the
map type. The only per-pipeline piece is a small adapter that rebuilds the exact
``loss_fn(params)`` a run optimized (deterministic from the run's config, since
the init is seeded). Add a new pipeline by adding one branch to
``build_loss_fn_from_run``.

The Hessian is formed by finite-differencing the ANALYTIC gradient (the
first-order autodiff that already flows through the jaxopt custom_vjp and trains
the model) — NOT second-order autodiff, which is fragile/expensive through the
implicit-diff rule. 2*P gradient evals for a P-parameter design.

Usage:
    JAX_PLATFORMS=cpu conda run -n kgnn_mac python nff/scripts/diagnose_landscape.py \
        --run-dir data/outputs/runs/run_20260707_203518_rect_5x5_shear \
        --landscape --grid 21 --render-eigvec
"""

import os
import time
import pickle
import argparse

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nff.config.experiment import load_and_parse_config, TargetConfig
from nff.config.targets import get_target_points
from nff.stages.pipeline import forward_pipeline
from nff.stages.geometry import deformed_vertices
from nff.stages.physics.kinematics import face_to_node_kinematics_fn
from nff.training.loss import compute_end_to_end_loss
from nff.scripts.closed_setup import (build_closed_initial_state, init_closed_les_params,
                                      build_surrogate_energy)
from nff.scripts.run_closed import _boundary_cloud, _fit_circle, _global_verts, _deployed_hinge_xy


# ── per-pipeline adapters ───────────────────────────────────────────────────────
# Each returns (loss_fn, params0, meta). loss_fn : params_pytree -> scalar loss,
# reproducing EXACTLY the objective the run optimized. params0 is the evaluation
# point (the trained best_params). meta carries geometry for the eigvec render.

def _closed_target_and_energy(config, initial_state, static_features, init_params):
    """Reconstruct the closed_les surrogate energy + fixed target (mirrors run_closed).

    Deterministic from config: the init (init_closed_les_params) is seeded, so the
    surrogate calibration and the deploy-fitted target are bit-reproducible.
    """
    load_specs = config.topology.get('loads', [])
    bond_energy, stability_fn, geometry_fn, damage_fn, w_lig_logit0 = build_surrogate_energy(
        config, static_features, initial_state, init_params)

    circle_fit = config.training.geometric_loss_type == "circle_fit"
    rect_mode = config.target.type == 'rectangle'
    target_cloud = None

    def _geom(mp):
        return geometry_fn(mp) if geometry_fn is not None else None

    if circle_fit:
        target_eff = config.target
    else:
        res0 = forward_pipeline(initial_state, config.target, config.validity, config.physics,
                                map_type=config.mapping.type, map_params=init_params,
                                static_features=static_features, load_specs=load_specs,
                                bond_energy_fn=bond_energy, hinge_geometry=_geom(init_params))
        cl0 = _boundary_cloud(res0['valid_state'], res0['solution'].fields[-1])
        sc = float(config.topology.get('target_radius_scale', 1.0))
        if rect_mode:
            if 'target_half_w' in config.topology:
                cen0 = np.array([float(config.topology.get('target_cx', config.target.center[0])),
                                 float(config.topology.get('target_cy', config.target.center[1]))])
                hw = float(config.topology['target_half_w']); hh = float(config.topology['target_half_h'])
            else:
                mn, mx = cl0.min(axis=0), cl0.max(axis=0)
                C = (mn + mx) / 2.0
                scx = float(config.topology.get('target_scale_x', sc))
                scy = float(config.topology.get('target_scale_y', sc))
                clamp_faces = config.topology.get('bc_clamped') or []
                clamp_faces = clamp_faces if isinstance(clamp_faces, list) else []
                if clamp_faces:
                    P = np.asarray(initial_state.face_centroids)[np.asarray(clamp_faces, int)].mean(0)
                    cen0 = P + np.array([scx, scy]) * (C - P)
                else:
                    cen0 = C
                hw, hh = (mx[0] - mn[0]) / 2.0 * scx, (mx[1] - mn[1]) / 2.0 * scy
            target_cloud = get_target_points(
                {'type': 'rectangle', 'center': cen0, 'radius': max(hw, hh),
                 'half_w': hw, 'half_h': hh}, n_points=400)
            target_eff = TargetConfig(type='circle', center=(float(cen0[0]), float(cen0[1])),
                                      radius=float(max(hw, hh)))
        else:
            cen0, rad0 = _fit_circle(cl0)
            rad0 *= sc
            target_eff = TargetConfig(type='circle', center=(float(cen0[0]), float(cen0[1])),
                                      radius=float(rad0))
    return (bond_energy, stability_fn, geometry_fn, damage_fn, w_lig_logit0,
            target_eff, target_cloud, load_specs)


def build_loss_fn_from_run(run_dir, solver_tol=None):
    """Rebuild (loss_fn, params0, meta) for whatever pipeline a run used.

    Reads the run's own config.yaml + best_params.pkl so it is fully self-contained.
    ``solver_tol`` overrides the inner physics tolerance (tighten to ~1e-7 for
    glass-smooth landscape/transect curves, since the default 1e-5 leaves ~1%
    numerical raggedness in L). The trajectory capture below always uses the run's
    ORIGINAL tolerance so the path it reproduces is the real one.
    """
    import dataclasses as dc
    config = load_and_parse_config(os.path.join(run_dir, "config.yaml"))
    # Tighten ONLY the analysis physics; leave config.physics at the run's real tolerance so
    # capture_trajectory reproduces the actual optimizer path (trained at that tolerance).
    phys_analysis = (dc.replace(config.physics, solver_tol=float(solver_tol))
                     if solver_tol is not None else config.physics)
    with open(os.path.join(run_dir, "best_params.pkl"), "rb") as f:
        params0 = {k: jnp.asarray(v, dtype=float) for k, v in pickle.load(f).items()}

    mt = config.mapping.type
    if mt == 'closed_les':
        initial_state, tess = build_closed_initial_state(config)
        init_params, static_features = init_closed_les_params(config)  # seeded init (calibration + target)
        (bond_energy, stability_fn, geometry_fn, damage_fn, w_lig_logit0,
         target_eff, target_cloud, load_specs) = \
            _closed_target_and_energy(config, initial_state, static_features, init_params)
        tcloud = jnp.asarray(target_cloud, dtype=jnp.float64) if target_cloud is not None else None

        def loss_fn(params):
            return compute_end_to_end_loss(
                params, initial_state, target_eff, config.validity, phys_analysis, config.training,
                map_type=mt, static_features=static_features, load_specs=load_specs,
                target_cloud=tcloud, bond_energy_fn=bond_energy, stability_fn=stability_fn,
                hinge_geometry_fn=geometry_fn)[0]

        meta = dict(config=config, tess=tess, initial_state=initial_state,
                    static_features=static_features, geometry_fn=geometry_fn, damage_fn=damage_fn,
                    bond_energy=bond_energy, stability_fn=stability_fn, target_eff=target_eff,
                    target_cloud=tcloud, load_specs=load_specs, w_lig_logit0=w_lig_logit0)
        return loss_fn, params0, meta

    raise NotImplementedError(
        f"No adapter for map_type={mt!r} yet. Add a branch here that rebuilds loss_fn "
        f"the same way its driver (e.g. train.py) does — the generic core below is unchanged.")


# ── generic core (pipeline-agnostic: operates only on loss_fn + a flat theta) ────

def fd_hessian(loss_fn, params0, eps=1e-3):
    """Dense symmetric Hessian of L wrt the flattened design vector, via central
    finite differences of the ANALYTIC gradient. 2*P gradient evaluations."""
    theta0, unravel = ravel_pytree(params0)
    P = theta0.shape[0]
    grad_flat = jax.jit(jax.grad(lambda th: loss_fn(unravel(th))))

    cols = np.zeros((P, P))
    t0 = time.time()
    for j in range(P):
        e = jnp.zeros(P).at[j].set(eps)
        gp = np.asarray(grad_flat(theta0 + e))
        gm = np.asarray(grad_flat(theta0 - e))
        cols[:, j] = (gp - gm) / (2.0 * eps)
        if j == 0 or (j + 1) % 16 == 0 or j == P - 1:
            print(f"    Hessian col {j + 1:3d}/{P}  ({(time.time() - t0):.0f}s)")
    H = 0.5 * (cols + cols.T)
    return H, theta0, unravel


def spectrum_stats(H, floor_rel=1e-6):
    """Eigen-decomposition + the scalars a paper reports.

    Two condition numbers are reported. ``cond_raw`` = λ_max/λ_min⁺ is dominated by
    the numerical/degeneracy floor and is NOT meaningful. ``cond_eff`` is taken over
    the "meaningful" spectrum — eigenvalues with |λ| above ``floor_rel``·λ_max —
    which is the anisotropy an optimizer actually feels.
    """
    w, V = np.linalg.eigh(H)                       # ascending
    order = np.argsort(np.abs(w))[::-1]            # by |lambda|, descending
    w_abs = np.abs(w)
    lam_max = float(w_abs.max())
    pos = w[w > 0]
    cond_raw = lam_max / float(pos.min()) if pos.size else float('inf')

    floor = floor_rel * lam_max
    meaningful = w_abs[w_abs >= floor]
    cond_eff = lam_max / float(meaningful.min()) if meaningful.size else float('nan')
    # participation ratio of the curvature: (sum|lam|)^2 / sum(lam^2) -> effective # of stiff modes
    pr = float((w_abs.sum() ** 2) / (np.sum(w_abs ** 2) + 1e-300))
    return dict(eigvals=w, eigvecs=V, cond=cond_eff, cond_raw=cond_raw, lam_max=lam_max,
                lam_min_eff=float(meaningful.min()) if meaningful.size else float('nan'),
                floor=floor, n_meaningful=int(meaningful.size),
                n_neg=int((w < 0).sum()), n_neg_stiff=int(((w < 0) & (w_abs >= floor)).sum()),
                n_flat=int((w_abs < floor).sum()),
                participation=pr, stiff_dir=V[:, int(np.abs(w).argmax())], order=order)


def plot_spectrum(stats, out_path, title):
    w = stats['eigvals']
    w_sorted = w[np.argsort(np.abs(w))[::-1]]
    idx = np.arange(1, w_sorted.size + 1)
    colors = ['#D62828' if v < 0 else '#F58025' for v in w_sorted]
    floor = stats['floor']

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].scatter(idx, np.abs(w_sorted), c=colors, s=26, zorder=3)
    ax[0].axhline(floor, color='#6C757D', ls=':', lw=1)
    ax[0].text(len(idx) * 0.55, floor * 1.3, 'noise/flat floor', color='#6C757D', fontsize=8)
    ax[0].set_yscale('log'); ax[0].set_xlabel('mode (by |λ|)'); ax[0].set_ylabel('|λ|')
    ax[0].set_title(f"design-loss Hessian spectrum   cond_eff={stats['cond']:.1e}   "
                    f"#λ<0(stiff)={stats['n_neg_stiff']}", fontsize=10)
    ax[0].grid(True, which='both', alpha=0.2)
    ax[0].scatter([], [], c='#F58025', label='λ > 0 (convex)')
    ax[0].scatter([], [], c='#D62828', label='λ < 0 (saddle)')
    ax[0].legend(frameon=False, fontsize=9)

    # signed log-magnitude histogram, split by sign so tiny-positive ≠ negative visually
    lg = np.log10(np.abs(w) + 1e-30)
    ax[1].hist(lg[w > 0], bins=25, color='#F58025', alpha=0.85, label='λ > 0')
    ax[1].hist(lg[w < 0], bins=25, color='#D62828', alpha=0.85, label='λ < 0')
    ax[1].axvline(np.log10(floor), color='#6C757D', ls=':', lw=1, label='floor')
    ax[1].set_xlabel('log₁₀|λ|'); ax[1].set_ylabel('count')
    ax[1].set_title(f"curvature magnitude   participation≈{stats['participation']:.1f} of {w.size}",
                    fontsize=10)
    ax[1].legend(frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out_path}")


def landscape_2d(loss_fn, theta0, unravel, d1, d2, R, n):
    """Evaluate L on an n×n grid of theta0 + a·d1 + b·d2, a,b ∈ [-R, R]."""
    Lj = jax.jit(lambda th: loss_fn(unravel(th)))
    a = np.linspace(-R, R, n)
    Z = np.full((n, n), np.nan)
    t0 = time.time()
    for i in range(n):
        for k in range(n):
            th = theta0 + a[i] * d1 + a[k] * d2
            v = float(Lj(th))
            Z[i, k] = v if np.isfinite(v) else np.nan
        print(f"    landscape row {i + 1:2d}/{n}  ({(time.time() - t0):.0f}s)")
    return a, Z


def plot_landscape(a, Z, out_path, title, axis_labels=('v₁ (stiffest)', 'v₂ (2nd stiffest)'),
                   traj=None):
    A, B = np.meshgrid(a, a, indexing='ij')
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    finite = np.isfinite(Z)
    lo, hi = np.nanpercentile(Z[finite], [2, 98]) if finite.any() else (0, 1)
    levels = np.linspace(lo, hi, 25)
    cf = ax[0].contourf(A, B, np.clip(Z, lo, hi), levels=levels, cmap='viridis')
    ax[0].contour(A, B, np.clip(Z, lo, hi), levels=levels, colors='k', linewidths=0.3, alpha=0.4)
    if (~finite).any():
        ax[0].contourf(A, B, (~finite).astype(float), levels=[0.5, 1.5], colors=['#111111'])
    # optimizer trajectory projected onto the eigvec plane (start ○ → end ★)
    if traj is not None and len(traj):
        traj = np.asarray(traj)
        ax[0].plot(traj[:, 0], traj[:, 1], '-', color='#D62828', lw=1.6, alpha=0.9, zorder=6)
        ax[0].scatter(traj[0, 0], traj[0, 1], facecolors='none', edgecolors='#D62828',
                      s=70, lw=1.8, zorder=7)
    ax[0].scatter([0], [0], c='#D62828', s=130, marker='*', zorder=8)
    ax[0].annotate('trained design', (0, 0), textcoords='offset points', xytext=(8, 6),
                   color='#D62828', fontsize=9)
    ax[0].set_xlabel(axis_labels[0]); ax[0].set_ylabel(axis_labels[1])
    ax[0].set_xlim(a.min(), a.max()); ax[0].set_ylim(a.min(), a.max())
    ax[0].set_title('loss landscape (2D slice)')
    fig.colorbar(cf, ax=ax[0], shrink=0.85, label='L(θ)')

    # 1D transects through the optimum along the two axes
    mid = len(a) // 2
    ax[1].plot(a, Z[:, mid], color='#F58025', lw=2, label=axis_labels[0])
    ax[1].plot(a, Z[mid, :], color='#264653', lw=2, label=axis_labels[1])
    ax[1].axvline(0, color='#D62828', ls='--', lw=1)
    ax[1].set_xlabel('step along direction'); ax[1].set_ylabel('L(θ)')
    ax[1].set_title('transects through optimum'); ax[1].legend(frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_transect_spaghetti(loss_fn, theta0, unravel, out_path, title, n_dirs=20, R=1.5, n=25, seed=0):
    """Many random unit directions through the optimum — a cheap, hard-to-cherry-pick
    smoothness check to back up the single 2D slice."""
    Lj = jax.jit(lambda th: loss_fn(unravel(th)))
    rng = np.random.default_rng(seed)
    P = theta0.shape[0]
    a = np.linspace(-R, R, n)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for _ in range(n_dirs):
        d = rng.normal(size=P); d = d / (np.linalg.norm(d) + 1e-12)
        d = jnp.asarray(d) * float(np.linalg.norm(np.asarray(theta0)) / np.sqrt(P) + 1e-9)
        ys = [float(Lj(theta0 + s * d)) for s in a]
        ax.plot(a, ys, color='#F58025', alpha=0.35, lw=1.2)
    ax.axvline(0, color='#D62828', ls='--', lw=1)
    ax.set_xlabel('step along random direction'); ax.set_ylabel('L(θ)')
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out_path}")


def render_eigvec(meta, params0, unravel, direction, out_path, title, amp=1.0):
    """Render the trained design vs a ±amp perturbation along a Hessian eigenvector,
    as the DEPLOYED sheet, hinges colored by ductile damage D.

    The geometric change along the stiffest direction is subtle (the sheet stays
    valid — the closed-chart guarantee); the *damage* recoloring reveals WHY the
    direction is stiff: it drives specific hinges toward the material failure
    barrier (D→1). That is the physical, not geometric, source of the curvature.
    """
    cfg = meta['config']; tess = meta['tess']; ist = meta['initial_state']
    gfn = meta['geometry_fn']; be = meta['bond_energy']; dfn = meta['damage_fn']

    def deploy(ptree):
        geom = gfn(ptree) if gfn else None
        res = forward_pipeline(ist, cfg.target, cfg.validity, cfg.physics,
                               map_type=cfg.mapping.type, map_params=ptree,
                               static_features=meta['static_features'], load_specs=meta['load_specs'],
                               bond_energy_fn=be, hinge_geometry=geom)
        v = res['valid_state']; d = res['solution'].fields[-1]
        V = _global_verts(tess, np.asarray(deformed_vertices(v, d)))
        hinge_xy = D = None
        if dfn is not None and geom is not None:
            cnv = v.centroid_node_vectors; nf, nn, _ = cnv.shape
            nd = face_to_node_kinematics_fn(d, cnv).reshape(nf * nn, 3)
            D = np.asarray(dfn(nd, geom, res['reference_bond_vectors']))
            hinge_xy = _deployed_hinge_xy(v, d)
        return V, hinge_xy, D

    theta0, _ = ravel_pytree(params0)
    faces = [np.asarray(f.vertex_indices) for f in tess.faces]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    sm = None
    for ax, sgn, lab in zip(axes, [-amp, 0.0, +amp],
                            ['− stiffest eigvec', 'trained design', '+ stiffest eigvec']):
        V, hinge_xy, D = deploy(unravel(theta0 + sgn * jnp.asarray(direction)))
        for f in faces:
            ax.fill(V[f, 0], V[f, 1], facecolor='#EEE7DF', edgecolor='#6C757D', lw=0.6, alpha=0.9)
        if D is not None:
            sm = ax.scatter(hinge_xy[:, 0], hinge_xy[:, 1], c=D, cmap='inferno', vmin=0.0, vmax=1.0,
                            s=45, zorder=5, edgecolors='k', linewidths=0.3)
            ax.set_title(f"{lab}\nmax D={D.max():.2f}  mean D={D.mean():.2f}", fontsize=10)
        else:
            ax.set_title(lab, fontsize=10)
        ax.set_aspect('equal'); ax.axis('off')
    if sm is not None:
        fig.colorbar(sm, ax=axes, shrink=0.7, label='ductile damage D  (1 = failure)')
    fig.suptitle(title, fontsize=11)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out_path}")


def capture_trajectory(meta):
    """Re-run the seeded training to recover the optimizer path in design space.

    The closed_les init is deterministic (seeded), so this reproduces the ACTUAL
    trajectory that produced best_params (the recovered best should match). Returns
    a list of flattened per-epoch design vectors.
    """
    from nff.training.trainer import create_train_step, TrainState
    cfg = meta['config']
    optimizer, step = create_train_step(
        meta['initial_state'], meta['target_eff'], cfg.validity, cfg.physics, cfg.training,
        map_type=cfg.mapping.type, use_jit=True, load_specs=meta['load_specs'],
        static_features=meta['static_features'], target_cloud=meta['target_cloud'],
        bond_energy_fn=meta['bond_energy'], stability_fn=meta['stability_fn'],
        hinge_geometry_fn=meta['geometry_fn'])

    init_params, _ = init_closed_les_params(cfg)                       # seeded init
    if meta['w_lig_logit0'] is not None:
        init_params = {**init_params, 'w_lig_logit': meta['w_lig_logit0']}
    state = TrainState(params=init_params, opt_state=optimizer.init(init_params),
                       rng=jax.random.PRNGKey(0))
    path = [ravel_pytree(init_params)[0]]
    best = (float('inf'), init_params)
    for _ in range(cfg.training.num_epochs):
        state, loss, aux = step(state)
        path.append(ravel_pytree(state.params)[0])
        ch = float(aux.get('chamfer_total', float('inf')))
        if np.isfinite(ch) and ch < best[0]:
            best = (ch, state.params)
    return [np.asarray(p) for p in path], ravel_pytree(best[1])[0], best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None, help="output dir (default: <run-dir>/landscape)")
    ap.add_argument("--eps", type=float, default=1e-3, help="finite-difference step for the Hessian")
    ap.add_argument("--landscape", action="store_true", help="also compute the 2D loss surface")
    ap.add_argument("--grid", type=int, default=21, help="landscape grid resolution (n×n)")
    ap.add_argument("--radius", type=float, default=2.0, help="landscape half-window along each eigvec")
    ap.add_argument("--transects", action="store_true", help="also draw the random-direction spaghetti")
    ap.add_argument("--render-eigvec", action="store_true", help="render the stiffest eigenvector as a deploy")
    ap.add_argument("--solver-tol", type=float, default=None,
                    help="override inner physics tolerance (e.g. 1e-7 for glass-smooth curves)")
    ap.add_argument("--with-trajectory", action="store_true",
                    help="reproduce the seeded training path and overlay it on the landscape")
    args = ap.parse_args()

    out_dir = args.out or os.path.join(args.run_dir, "landscape")
    os.makedirs(out_dir, exist_ok=True)
    tag = os.path.basename(os.path.normpath(args.run_dir))

    print(f"[diagnose_landscape] run={tag}")
    loss_fn, params0, meta = build_loss_fn_from_run(args.run_dir, solver_tol=args.solver_tol)
    theta0_flat, _ = ravel_pytree(params0)
    P = theta0_flat.shape[0]
    L0 = float(loss_fn(params0))
    print(f"  design dim P={P}   L(θ*)={L0:.4e}")
    shapes = {k: tuple(np.asarray(v).shape) for k, v in params0.items()}
    print(f"  params: {shapes}")

    # ── Hessian + spectrum ──
    print("  finite-difference Hessian (2P gradient evals)...")
    H, theta0, unravel = fd_hessian(loss_fn, params0, eps=args.eps)
    stats = spectrum_stats(H)
    print(f"\n  ── design-loss Hessian spectrum ──")
    print(f"     λ_max          = {stats['lam_max']:.4e}")
    print(f"     cond_eff       = {stats['cond']:.4e}  (over |λ|≥floor, the anisotropy felt by SGD)")
    print(f"     cond_raw       = {stats['cond_raw']:.4e}  (floor-dominated; not meaningful)")
    print(f"     # λ < 0 (all)  = {stats['n_neg']}   |  stiff = {stats['n_neg_stiff']}  (plastic-softening saddle)")
    print(f"     # flat (|λ|<floor) = {stats['n_flat']}  (near-null directions)")
    print(f"     participation  = {stats['participation']:.2f} of {P} directions")
    np.save(os.path.join(out_dir, "hessian.npy"), H)
    plot_spectrum(stats, os.path.join(out_dir, "spectrum.png"), f"{tag}")

    # ── landscape along the two stiffest eigenvectors ──
    if args.landscape:
        V = stats['eigvecs']; w = stats['eigvals']
        top2 = np.argsort(np.abs(w))[::-1][:2]
        v1 = np.asarray(V[:, top2[0]]); v2 = np.asarray(V[:, top2[1]])
        d1 = jnp.asarray(v1); d2 = jnp.asarray(v2)

        traj_proj = None
        if args.with_trajectory:
            print("\n  reproducing seeded training trajectory...")
            path, best_flat, best_ch = capture_trajectory(meta)
            th0 = np.asarray(theta0)
            drift = float(np.linalg.norm(best_flat - th0))
            print(f"    recovered best chamfer={best_ch:.4e}   ‖best_recovered − best_saved‖={drift:.2e}"
                  f"  ({'MATCH' if drift < 1e-2 else 'differs — check seeding'})")
            traj_proj = np.array([[float((p - th0) @ v1), float((p - th0) @ v2)] for p in path])

        print(f"\n  2D landscape along eigvecs (|λ|={abs(w[top2[0]]):.2e}, {abs(w[top2[1]]):.2e})...")
        a, Z = landscape_2d(loss_fn, theta0, unravel, d1, d2, R=args.radius, n=args.grid)
        np.savez(os.path.join(out_dir, "landscape.npz"), a=a, Z=Z,
                 traj=traj_proj if traj_proj is not None else np.empty((0, 2)))
        plot_landscape(a, Z, os.path.join(out_dir, "landscape.png"), f"{tag}", traj=traj_proj)

    if args.transects:
        print("\n  random-direction transect spaghetti...")
        plot_transect_spaghetti(loss_fn, theta0, unravel,
                                os.path.join(out_dir, "transects.png"),
                                f"{tag} — loss along 20 random design directions")

    if args.render_eigvec:
        print("\n  rendering stiffest eigenvector as a deploy...")
        render_eigvec(meta, params0, unravel, stats['stiff_dir'],
                      os.path.join(out_dir, "eigvec_top.png"),
                      f"{tag} — stiffest curvature direction", amp=args.radius)

    print(f"\nDone. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
