"""nff/sofa/hinge_optimizer.py — corner-hinge Bézier optimizer via the Tesseract oracle.

Client-side (runs in kgnn_mac; never imports ``Sofa``). Optimizes the 9 hinge
parameters against the Dockerised SOFA oracle. The oracle input dict is assembled by
:mod:`nff.sofa.oracle_payload`; the HTTP transport is the ``tesseract_core`` SDK.
All parameters live directly in physical space (metres) and are updated by a NumPy
Adam step; gap + the four edge reaches are kept positive by projecting to a small
floor after each step (no softplus — that froze gap/reaches ~500× slower than the
control points).

Objective — a lean hinge that survives MANY closing cycles (low-cycle fatigue),
keeping the face gap small::

    loss = w_fatigue · ε_plastic / ε_yield    (minimise plastic strain → maximise N_f)
         + w_mat     · hinge_area / area₀      (lean hinge — minimise material)
         + w_gap     · (gap / gap₀)²           (keep the face gap small/controlled)

``ε_plastic = max(0, ε − ε_yield)`` is the per-fold plastic strain; cycles-to-failure
come from Coffin-Manson (:mod:`nff.sofa.fatigue`). The optimizer minimises the oracle's
*smooth* (KS) principal strain (less noisy than the hard max); material + gap are
analytic client-side FD (no SOFA call).

Usage
-----
    docker run -p 8000:8000 nff-sofa-oracle          # start the oracle
    conda run -n kgnn_mac python nff/sofa/hinge_optimizer.py \\
        --config data/configs/sofa/hinge_opt_2face.yaml [--n-epochs 30] [--lr 0.05]

Outputs  ``data/outputs/hinge_opt/<timestamp>_<config>/``
    config.yaml · convergence.npz · final_state.npz
"""
from __future__ import annotations

import argparse
import datetime
import pathlib
import shutil
import sys

import numpy as np
import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tesseract_core import Tesseract
from nff.sofa import oracle_payload as op
from nff.sofa.hinge_geometry import compute_hinge_geometry
from nff.sofa.hinge_objective import build_objective, hinge_area, phys_from_params as _phys

OUTPUTS_DIR           = REPO / 'data' / 'outputs' / 'hinge_opt'
TESSERACT_DEFAULT_URL = op.DEFAULT_URL


# ── Optimizer (loss + gradient assembly live in nff.sofa.hinge_objective) ──────

class _NumpyAdam:
    """Minimal Adam optimizer in NumPy — no JAX or optax dependency.

    The gradient here is external (finite-differenced server-side by the SOFA
    oracle plus cheap analytic terms), so there is no JAX autodiff graph for
    optax to ride on — Adam is then just its short update formula on a flat
    9-vector. Adam rather than plain SGD because the FD gradient is noisy and the
    9 knobs span ~50x in scale: the m/v running averages low-pass the noise and
    adapt the step size per parameter.
    """

    def __init__(self, lr: float, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m: np.ndarray | None = None   # 1st moment: smoothed gradient (direction)
        self.v: np.ndarray | None = None   # 2nd moment: smoothed squared gradient (scale)
        self.t: int = 0

    def update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grads
        self.v = self.b2 * self.v + (1 - self.b2) * grads ** 2
        # Bias-correction: m and v start at 0 so they under-estimate early;
        # dividing by (1 - b**t) cancels that exactly and fades to a no-op as t grows.
        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ── Initial design ─────────────────────────────────────────────────────────────

def _initial_params(cfg: dict, cs) -> np.ndarray:
    """Assemble the initial 9-vector: gap + 4 reaches, then the 4 free control points.

    The control points default to the symmetric mesh geometry at the initial gap;
    ``initial_concave_bow_m`` bows them INWARD for a stretched concave start (steers
    the optimizer out of the thick-bulb basin).
    """
    sofa_cfg = cfg.get('sofa', {})
    gap_init   = float(sofa_cfg.get('gap_initial', 0.003))
    reach_init = float(sofa_cfg.get('reach_initial', gap_init))
    pos_init = np.array([
        gap_init,
        float(sofa_cfg.get('s0_top_initial', reach_init)),
        float(sofa_cfg.get('s0_bot_initial', reach_init)),
        float(sofa_cfg.get('s1_top_initial', reach_init)),
        float(sofa_cfg.get('s1_bot_initial', reach_init)),
    ], dtype=np.float64)

    hd0 = compute_hinge_geometry(cs, gap=gap_init)['hinge_data'][0]
    cp = {'bcu_x': hd0['bc_up'][0], 'bcu_y': hd0['bc_up'][1],
          'bcl_x': hd0['bc_lo'][0], 'bcl_y': hd0['bc_lo'][1]}
    bow = float(sofa_cfg.get('initial_concave_bow_m', 0.0))
    if bow > 0:
        for (kx, ky), pa, pb in [(('bcu_x', 'bcu_y'), 'p0_top', 'p1_top'),
                                 (('bcl_x', 'bcl_y'), 'p0_bot', 'p1_bot')]:
            mid = 0.5 * (np.asarray(hd0[pa]) + np.asarray(hd0[pb]))
            out = np.array([cp[kx], cp[ky]]) - mid               # default = outward
            inward = mid - bow * out / (np.linalg.norm(out) + 1e-12)
            cp[kx], cp[ky] = float(inward[0]), float(inward[1])

    free_init = np.array([float(sofa_cfg.get(f'{n}_initial', cp[n]))
                          for n in op.FREE_NAMES], dtype=np.float64)
    return np.concatenate([pos_init, free_init]).astype(np.float64)


def _lr_schedule_fn(lr: float, n_epochs: int, schedule: str, hold_frac: float):
    """Return ``epoch → learning_rate``.

    'cosine' anneals lr → 5% over the run (damps momentum overshoot). 'hold_cosine'
    holds full lr to reach the deep basin, then cosine-anneals to lock it in.
    'constant' = fixed lr.
    """
    def lr_at(epoch: int) -> float:
        if schedule == 'cosine' and n_epochs > 1:
            return lr * (0.05 + 0.95 * 0.5 * (1.0 + np.cos(np.pi * epoch / (n_epochs - 1))))
        if schedule == 'hold_cosine' and n_epochs > 1:
            h = max(1, int(round(hold_frac * (n_epochs - 1))))
            if epoch <= h:
                return lr
            frac = (epoch - h) / max(1, (n_epochs - 1 - h))
            return lr * (0.05 + 0.95 * 0.5 * (1.0 + np.cos(np.pi * frac)))
        return lr
    return lr_at


# ── Optimization loop ─────────────────────────────────────────────────────────

def run_optimization(cfg: dict, n_epochs: int, lr: float, tesseract_url: str,
                     out_dir: pathlib.Path, capture_fields: bool = True) -> dict:
    sofa_cfg = cfg.get('sofa', {})
    opt_cfg  = cfg.get('optimization', {})

    cs_static = op.build_physical_cs(cfg)
    clamped_faces = sorted({int(f) for f in cs_static.constrained_face_DOF_pairs[:, 0]})
    loaded_faces  = sorted({int(f) for f in cs_static.loaded_face_DOF_pairs[:, 0]})

    params   = _initial_params(cfg, cs_static)
    pos_mask = np.array([n in op.POS_NAMES for n in op.PARAM_NAMES])   # gap + 4 reaches
    floor    = float(sofa_cfg.get('param_floor', 0.0005))             # 0.5 mm min

    # The loss and its gradient assembly live in nff.sofa.hinge_objective; the
    # optimizer only drives it. References (area₀, gap₀) are fixed from the initial design.
    obj = build_objective(cfg, cs_static, params)

    oracle = Tesseract.from_url(tesseract_url)   # SDK handle: HTTP + response decoding via tesseract_core
    optimizer = _NumpyAdam(lr)
    lr_at = _lr_schedule_fn(lr, n_epochs, str(opt_cfg.get('lr_schedule', 'cosine')),
                            float(opt_cfg.get('lr_hold_frac', 0.4)))

    history: dict = {k: [] for k in op.PARAM_NAMES + [
        'total_loss', 'loss_fatigue', 'loss_mat', 'loss_gap',
        'max_strain', 'plastic_strain', 'cycles_Nf', 'max_vm_rot', 'hinge_area']}

    p0 = _phys(params)
    print(f"\nHinge optimization (9-param quadratic Bézier, physical space): {n_epochs} epochs, lr={lr}")
    print(f"  Tesseract URL: {tesseract_url}")
    print(f"  loss = {obj.w_fatigue}·ε_plastic/ε_y + {obj.w_mat}·area/area₀ + {obj.w_gap}·(gap/gap₀)²")
    print(f"  ε_yield={obj.eps_yield*100:.2f}%  ε_fracture={obj.eps_frac*100:.1f}%  "
          f"Coffin-Manson(ε_f'={obj.fat_ef}, c={obj.fat_c}); target ≥ {obj.n_target:.0f} cycles")
    print(f"  Initial: gap={p0['gap']*1e3:.2f} mm  "
          f"reach s0=({p0['s0_top']*1e3:.2f},{p0['s0_bot']*1e3:.2f}) "
          f"s1=({p0['s1_top']*1e3:.2f},{p0['s1_bot']*1e3:.2f}) mm  area={obj.area_ref*1e6:.1f} mm²\n")

    def _save_convergence():
        np.savez(
            str(out_dir / 'convergence.npz'),
            **{name: np.array(history[name]) for name in op.PARAM_NAMES},
            total_loss      = np.array(history['total_loss']),
            loss_fatigue    = np.array(history['loss_fatigue']),
            loss_mat        = np.array(history['loss_mat']),
            loss_gap        = np.array(history['loss_gap']),
            max_strain      = np.array(history['max_strain']),
            plastic_strain  = np.array(history['plastic_strain']),
            cycles_Nf       = np.array(history['cycles_Nf']),
            max_vm_rot      = np.array(history['max_vm_rot']),
            hinge_area      = np.array(history['hinge_area']),
            fracture_strain = np.array(obj.eps_frac),
            yield_strain    = np.array(obj.eps_yield),
            target_cycles   = np.array(obj.n_target),
            stress          = np.array(history['max_vm_rot']),   # backward-compat alias
        )

    for epoch in range(n_epochs):
        phys    = _phys(params)
        payload = op.build_payload(cs_static, phys, cfg, clamped_faces, loaded_faces)

        # ── Oracle calls (apply + strain Jacobian) — robust to a hung sim ──────
        try:
            fwd = oracle.apply(payload)
            jac = oracle.jacobian(payload, jac_inputs=op.PARAM_NAMES,
                                  jac_outputs=['smooth_principal_strain'])
        except Exception as ex:
            print(f"  epoch {epoch+1}: oracle call failed ({type(ex).__name__}: {ex}); "
                  "stopping early — keeping the epochs completed so far.")
            break

        # ── Loss + gradient assembly (nff.sofa.hinge_objective) ───────────────
        # One call owns all three terms and how each is differentiated: the strain
        # gradient is the oracle's FD Jacobian (the 54-sim cost), material is a cheap
        # analytic FD, gap is closed-form. aux carries the reporting scalars.
        loss, grad, aux = obj.loss_and_grad(params, fwd, jac)

        optimizer.lr = lr_at(epoch)                              # lr schedule (damps overshoot)
        params = optimizer.update(params, grad)
        # Keep gap + the four reaches positive by a hard projection to a floor — not
        # softplus, which froze these knobs ~500x slower than the free control points.
        params[pos_mask] = np.maximum(params[pos_mask], floor)

        # ── Record + save every epoch (a crash never loses everything) ────────
        for name in op.PARAM_NAMES:
            history[name].append(phys[name])
        history['total_loss'].append(loss)
        history['loss_fatigue'].append(aux['l_fat'])
        history['loss_mat'].append(aux['l_mat'])
        history['loss_gap'].append(aux['l_gap'])
        history['max_strain'].append(aux['strain'])
        history['plastic_strain'].append(aux['eps_p'])
        history['cycles_Nf'].append(min(aux['n_f'], 1e9))   # cap inf for storage
        history['max_vm_rot'].append(aux['max_vm'])
        history['hinge_area'].append(aux['area'])
        _save_convergence()

        _nf_s = '∞' if not np.isfinite(aux['n_f']) else f"{aux['n_f']:.1f}"
        print(f"  epoch {epoch+1:3d}/{n_epochs}  "
              f"loss={loss:.3f} (fat={aux['l_fat']:.2f} mat={aux['l_mat']:.2f} gap={aux['l_gap']:.2f})  "
              f"ε={aux['strain']*100:.2f}%(peak {aux['true_eps']*100:.1f}%) "
              f"ε_p={aux['eps_p']*100:.2f}% N_f={_nf_s}cyc  "
              f"σ_max={aux['max_vm']/1e6:.0f}MPa area={aux['area']*1e6:.1f}mm²  gap={phys['gap']*1e3:.3f} mm  "
              f"s0=({phys['s0_top']*1e3:.2f},{phys['s0_bot']*1e3:.2f}) "
              f"s1=({phys['s1_top']*1e3:.2f},{phys['s1_bot']*1e3:.2f}) mm  "
              f"bc_up=({phys['bcu_x']*1e3:.2f},{phys['bcu_y']*1e3:.2f})  "
              f"bc_lo=({phys['bcl_x']*1e3:.2f},{phys['bcl_y']*1e3:.2f}) mm")

    if not history['total_loss']:
        print("No epochs completed — aborting before final-state capture.")
        return history
    if not capture_fields:
        return history   # search mode: metrics come from convergence.npz; skip the field sim

    _capture_final_state(cfg, cs_static, history, clamped_faces, loaded_faces,
                         oracle, out_dir)
    return history


def _capture_final_state(cfg, cs_static, history, clamped_faces, loaded_faces,
                         oracle, out_dir) -> None:
    """Re-run the oracle at the best design and save the deformed field for viz."""
    best_idx  = int(np.argmin(history['total_loss']))
    best_phys = {name: float(history[name][best_idx]) for name in op.PARAM_NAMES}
    print(f"\nCapturing final-state field at best design (epoch {best_idx + 1}) ...")
    payload = op.build_payload(cs_static, best_phys, cfg, clamped_faces, loaded_faces)
    payload['return_fields']        = True
    payload['skip_secondary_modes'] = True   # rotation field only
    try:
        fwd = oracle.apply(payload)
        np.savez(
            str(out_dir / 'final_state.npz'),
            von_mises_field = np.asarray(fwd['von_mises_field']),
            deformed_nodes  = np.asarray(fwd['deformed_nodes']),
            mesh_tets       = np.asarray(fwd['mesh_tets']),
            best_idx        = np.array(best_idx),
            face_centroids             = cs_static.face_centroids,
            centroid_node_vectors      = cs_static.centroid_node_vectors,
            hinge_node_pairs           = cs_static.hinge_node_pairs,
            hinge_adj_info             = cs_static.hinge_adj_info,
            constrained_face_DOF_pairs = cs_static.constrained_face_DOF_pairs,
            loaded_face_DOF_pairs      = cs_static.loaded_face_DOF_pairs,
        )
        nf = history['cycles_Nf'][best_idx]
        print(f"  final_state.npz saved (best epoch {best_idx + 1}: "
              f"ε_max={history['max_strain'][best_idx]*100:.2f}%, "
              f"ε_p={history['plastic_strain'][best_idx]*100:.2f}%, "
              f"N_f={'∞' if nf >= 1e9 else f'{nf:.0f}'} cyc, "
              f"area={history['hinge_area'][best_idx]*1e6:.1f} mm²).")
    except Exception as ex:
        print(f"  WARNING: final-state field capture failed ({ex}); "
              "visualizer will skip the von Mises panel.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description='Optimize SOFA hinge dimensions via Tesseract HTTP.')
    p.add_argument('--config', default='data/configs/sofa/hinge_opt_2face.yaml',
                   help='Path to YAML config.')
    p.add_argument('--n-epochs', type=int, default=None, help='Override optimization.n_epochs.')
    p.add_argument('--lr', type=float, default=None, help='Override optimization.learning_rate.')
    p.add_argument('--tesseract-url', default=TESSERACT_DEFAULT_URL,
                   help='URL of the running Tesseract server (Docker).')
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    opt_cfg  = cfg.get('optimization', {})
    n_epochs = args.n_epochs if args.n_epochs is not None else int(opt_cfg.get('n_epochs', 30))
    lr       = args.lr       if args.lr       is not None else float(opt_cfg.get('learning_rate', 0.05))
    url      = args.tesseract_url

    # Verify the server is reachable before committing to a long run.
    try:
        Tesseract.from_url(url).health()
    except Exception:
        print(f"ERROR: cannot reach Tesseract server at {url}")
        print("Start it:  docker run -p 8000:8000 -e TESSERACT_RUNTIME_SERVE_HOST=0.0.0.0 nff-sofa-oracle:latest serve")
        raise SystemExit(1)

    config_name = pathlib.Path(args.config).stem
    timestamp   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir     = OUTPUTS_DIR / f'{timestamp}_{config_name}'
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out_dir / 'config.yaml')

    history  = run_optimization(cfg, n_epochs, lr, url, out_dir)
    best_idx = int(np.argmin(history['total_loss']))
    b = lambda k: history[k][best_idx] * 1e3
    nf = history['cycles_Nf'][best_idx]
    print(f'\nBest (epoch {best_idx + 1}): '
          f'ε_max={history["max_strain"][best_idx]*100:.2f}%  '
          f'ε_p={history["plastic_strain"][best_idx]*100:.2f}%  '
          f'N_f={"∞" if nf >= 1e9 else f"{nf:.0f}"} cyc  '
          f'area={history["hinge_area"][best_idx]*1e6:.1f} mm²  gap={b("gap"):.3f} mm  '
          f's0=({b("s0_top"):.2f},{b("s0_bot"):.2f}) s1=({b("s1_top"):.2f},{b("s1_bot"):.2f}) mm  '
          f'bc_up=({b("bcu_x"):.2f},{b("bcu_y"):.2f}) bc_lo=({b("bcl_x"):.2f},{b("bcl_y"):.2f}) mm')
    print(f'Results saved → {out_dir}')


if __name__ == '__main__':
    main()
