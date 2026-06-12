"""sofa/hinge_optimizer.py — corner-hinge Bézier optimizer via the Tesseract oracle.

Client-side (runs in kgnn_mac; never imports ``Sofa``). Optimizes the 9 hinge
parameters against the Dockerised SOFA oracle through :mod:`nff.sofa.tesseract_client`.
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
    conda run -n kgnn_mac python sofa/hinge_optimizer.py \\
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
import requests
import yaml

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from nff.sofa import tesseract_client as tc
from nff.sofa.fatigue import cycles_to_failure
from nff.sofa.mesh_builder_gmsh import compute_hinge_geometry

OUTPUTS_DIR           = REPO / 'data' / 'outputs' / 'hinge_opt'
TESSERACT_DEFAULT_URL = tc.DEFAULT_URL


# ── Hinge objective helpers (physical-space params, no SOFA) ───────────────────

def _phys(params: np.ndarray) -> dict:
    """Parameter vector → name→value dict (keys == Tesseract schema names)."""
    return {n: float(v) for n, v in zip(tc.PARAM_NAMES, params)}


def _bezier_from_phys(phys: dict) -> dict:
    """Physical param dict → bezier_params for compute_hinge_geometry / the oracle."""
    return {
        's0_top': phys['s0_top'], 's0_bot': phys['s0_bot'],
        's1_top': phys['s1_top'], 's1_bot': phys['s1_bot'],
        'bc_up_xy': [phys['bcu_x'], phys['bcu_y']],
        'bc_lo_xy': [phys['bcl_x'], phys['bcl_y']],
    }


def _hinge_area(phys: dict, cs) -> float:
    """Hinge-strip area [m²] from the Bézier boundary — analytic, no SOFA.

    Lens area between the upper and lower arcs (shoelace on a sampled boundary);
    a cheap regulariser that rewards lean hinges.
    """
    geo = compute_hinge_geometry(cs, gap=phys['gap'], bezier_params=_bezier_from_phys(phys))

    def _bez(p0, c, p2, n=40):
        t = np.linspace(0.0, 1.0, n)[:, None]
        return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * c + t ** 2 * p2

    total = 0.0
    for hd in geo['hinge_data']:
        up = _bez(hd['p0_top'], hd['bc_up'], hd['p1_top'])
        lo = _bez(hd['p0_bot'], hd['bc_lo'], hd['p1_bot'])
        poly = np.vstack([up, lo[::-1]])
        x, y = poly[:, 0], poly[:, 1]
        total += 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return float(total)


def _area_grad(params: np.ndarray, cs, eps: float = 1e-5) -> np.ndarray:
    """d(hinge_area)/d(param) via central FD on the cheap analytic area."""
    g = np.zeros(len(params))
    for i in range(len(params)):
        pp = params.copy(); pp[i] += eps
        pm = params.copy(); pm[i] -= eps
        g[i] = (_hinge_area(_phys(pp), cs) - _hinge_area(_phys(pm), cs)) / (2 * eps)
    return g


class _NumpyAdam:
    """Minimal Adam optimizer in NumPy — no JAX or optax dependency."""

    def __init__(self, lr: float, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m: np.ndarray | None = None
        self.v: np.ndarray | None = None
        self.t: int = 0

    def update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grads
        self.v = self.b2 * self.v + (1 - self.b2) * grads ** 2
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
                          for n in tc.FREE_NAMES], dtype=np.float64)
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
    mat_cfg  = cfg.get('material', {})
    loss_cfg = cfg.get('loss', {})
    opt_cfg  = cfg.get('optimization', {})

    cs_static = tc.build_physical_cs(cfg)
    clamped_faces = sorted({int(f) for f in cs_static.constrained_face_DOF_pairs[:, 0]})
    loaded_faces  = sorted({int(f) for f in cs_static.loaded_face_DOF_pairs[:, 0]})

    params   = _initial_params(cfg, cs_static)
    pos_mask = np.array([n in tc.POS_NAMES for n in tc.PARAM_NAMES])   # gap + 4 reaches
    floor    = float(sofa_cfg.get('param_floor', 0.0005))             # 0.5 mm min

    # Loss weights + the plasticity / low-cycle-fatigue criterion.
    w_fatigue = float(loss_cfg.get('w_fatigue', loss_cfg.get('w_strain', 5.0)))
    w_mat     = float(loss_cfg.get('w_mat', 2.0))
    w_gap     = float(loss_cfg.get('w_gap', 0.5))
    eps_frac  = float(mat_cfg.get('fracture_strain', 0.045))
    eps_yield = float(mat_cfg.get('yield_strain',
                      float(mat_cfg.get('yield_strength', 50e6)) /
                      float(mat_cfg.get('young_modulus', 3.5e9))))
    fat_ef    = float(mat_cfg.get('fatigue_ductility_coeff', 0.05))   # ε_f'
    fat_c     = float(mat_cfg.get('fatigue_ductility_exp', -0.6))     # c (< 0)
    n_target  = float(loss_cfg.get('target_cycles', 100.0))

    gap_init = float(sofa_cfg.get('gap_initial', 0.003))
    gap_ref  = max(gap_init, 1e-6)
    area_ref = max(_hinge_area(_phys(params), cs_static), 1e-12)
    area_min = float(loss_cfg.get('min_hinge_area_m2', 20e-6))        # 20 mm² floor

    optimizer = _NumpyAdam(lr)
    lr_at = _lr_schedule_fn(lr, n_epochs, str(opt_cfg.get('lr_schedule', 'cosine')),
                            float(opt_cfg.get('lr_hold_frac', 0.4)))

    history: dict = {k: [] for k in tc.PARAM_NAMES + [
        'total_loss', 'loss_fatigue', 'loss_mat', 'loss_gap',
        'max_strain', 'plastic_strain', 'cycles_Nf', 'max_vm_rot', 'hinge_area']}

    p0 = _phys(params)
    print(f"\nHinge optimization (9-param quadratic Bézier, physical space): {n_epochs} epochs, lr={lr}")
    print(f"  Tesseract URL: {tesseract_url}")
    print(f"  loss = {w_fatigue}·ε_plastic/ε_y + {w_mat}·area/area₀ + {w_gap}·(gap/gap₀)²")
    print(f"  ε_yield={eps_yield*100:.2f}%  ε_fracture={eps_frac*100:.1f}%  "
          f"Coffin-Manson(ε_f'={fat_ef}, c={fat_c}); target ≥ {n_target:.0f} cycles")
    print(f"  Initial: gap={p0['gap']*1e3:.2f} mm  "
          f"reach s0=({p0['s0_top']*1e3:.2f},{p0['s0_bot']*1e3:.2f}) "
          f"s1=({p0['s1_top']*1e3:.2f},{p0['s1_bot']*1e3:.2f}) mm  area={area_ref*1e6:.1f} mm²\n")

    def _save_convergence():
        np.savez(
            str(out_dir / 'convergence.npz'),
            **{name: np.array(history[name]) for name in tc.PARAM_NAMES},
            total_loss      = np.array(history['total_loss']),
            loss_fatigue    = np.array(history['loss_fatigue']),
            loss_mat        = np.array(history['loss_mat']),
            loss_gap        = np.array(history['loss_gap']),
            max_strain      = np.array(history['max_strain']),
            plastic_strain  = np.array(history['plastic_strain']),
            cycles_Nf       = np.array(history['cycles_Nf']),
            max_vm_rot      = np.array(history['max_vm_rot']),
            hinge_area      = np.array(history['hinge_area']),
            fracture_strain = np.array(eps_frac),
            yield_strain    = np.array(eps_yield),
            target_cycles   = np.array(n_target),
            stress          = np.array(history['max_vm_rot']),   # backward-compat alias
        )

    for epoch in range(n_epochs):
        phys    = _phys(params)
        payload = tc.build_payload(cs_static, phys, cfg, clamped_faces, loaded_faces)

        # ── Oracle calls (apply + strain Jacobian) — robust to a hung sim ──────
        try:
            fwd = tc.apply(tesseract_url, payload)
            jac = tc.jacobian(tesseract_url, payload, tc.PARAM_NAMES, ['smooth_principal_strain'])
        except Exception as ex:
            print(f"  epoch {epoch+1}: oracle call failed ({type(ex).__name__}: {ex}); "
                  "stopping early — keeping the epochs completed so far.")
            break

        max_vm   = tc.decode_scalar(fwd['max_von_mises_stress'])
        strain   = tc.decode_scalar(fwd['smooth_principal_strain'])   # smooth KS surrogate (design)
        true_eps = tc.decode_scalar(fwd['max_principal_strain'])      # true peak (reporting only)
        area     = _hinge_area(phys, cs_static)
        gp       = phys['gap']

        # loss = w_fatigue·ε_plastic/ε_yield + w_mat·area/area₀ + w_gap·(gap/gap₀)²
        eps_p = max(0.0, strain - eps_yield)
        n_f   = cycles_to_failure(eps_p, fat_ef, fat_c)
        l_fat = w_fatigue * eps_p / eps_yield
        l_mat = w_mat * area / area_ref
        l_gap = w_gap * (gp / gap_ref) ** 2
        loss  = l_fat + l_mat + l_gap

        # ── Gradients ─────────────────────────────────────────────────────────
        def _jac_val(output_key, inp_key):
            return tc.decode_scalar(jac[output_key][inp_key]) if output_key in jac else 0.0
        dstrain = np.array([_jac_val('smooth_principal_strain', ki) for ki in tc.PARAM_NAMES])
        # d(ε_p)/d(param) = dstrain when plastic (strain > yield), else 0.
        d_fat = (w_fatigue / eps_yield) * (1.0 if strain > eps_yield else 0.0) * dstrain
        # Material + gap are analytic / cheap-FD (no SOFA). Degeneracy guard: stop the
        # material pull once the hinge is at the min area (else it collapses).
        n_p   = len(tc.PARAM_NAMES)
        d_mat = (np.zeros(n_p) if area <= area_min
                 else (w_mat / area_ref) * _area_grad(params, cs_static))
        d_gap = np.zeros(n_p); d_gap[0] = w_gap * 2.0 * gp / gap_ref ** 2
        grad  = d_fat + d_mat + d_gap

        optimizer.lr = lr_at(epoch)                              # lr schedule (damps overshoot)
        params = optimizer.update(params, grad)
        params[pos_mask] = np.maximum(params[pos_mask], floor)   # project to positivity

        # ── Record + save every epoch (a crash never loses everything) ────────
        for name in tc.PARAM_NAMES:
            history[name].append(phys[name])
        history['total_loss'].append(loss)
        history['loss_fatigue'].append(l_fat)
        history['loss_mat'].append(l_mat)
        history['loss_gap'].append(l_gap)
        history['max_strain'].append(strain)
        history['plastic_strain'].append(eps_p)
        history['cycles_Nf'].append(min(n_f, 1e9))   # cap inf for storage
        history['max_vm_rot'].append(max_vm)
        history['hinge_area'].append(area)
        _save_convergence()

        _nf_s = '∞' if not np.isfinite(n_f) else f'{n_f:.1f}'
        print(f"  epoch {epoch+1:3d}/{n_epochs}  "
              f"loss={loss:.3f} (fat={l_fat:.2f} mat={l_mat:.2f} gap={l_gap:.2f})  "
              f"ε={strain*100:.2f}%(peak {true_eps*100:.1f}%) ε_p={eps_p*100:.2f}% N_f={_nf_s}cyc  "
              f"σ_max={max_vm/1e6:.0f}MPa area={area*1e6:.1f}mm²  gap={phys['gap']*1e3:.3f} mm  "
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
                         tesseract_url, out_dir)
    return history


def _capture_final_state(cfg, cs_static, history, clamped_faces, loaded_faces,
                         tesseract_url, out_dir) -> None:
    """Re-run the oracle at the best design and save the deformed field for viz."""
    best_idx  = int(np.argmin(history['total_loss']))
    best_phys = {name: float(history[name][best_idx]) for name in tc.PARAM_NAMES}
    print(f"\nCapturing final-state field at best design (epoch {best_idx + 1}) ...")
    payload = tc.build_payload(cs_static, best_phys, cfg, clamped_faces, loaded_faces)
    payload['return_fields']        = True
    payload['skip_secondary_modes'] = True   # rotation field only
    try:
        fwd = tc.apply(tesseract_url, payload)
        np.savez(
            str(out_dir / 'final_state.npz'),
            von_mises_field = tc.decode_array(fwd['von_mises_field']),
            deformed_nodes  = tc.decode_array(fwd['deformed_nodes']),
            mesh_tets       = tc.decode_array(fwd['mesh_tets']),
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
        requests.get(f"{url}/health", timeout=30)
    except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
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
