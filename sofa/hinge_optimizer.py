"""
sofa/hinge_optimizer.py — Bézier hinge shape optimizer via Tesseract HTTP.

Optimizes 5 hinge parameters using the Tesseract physics oracle (SOFA FEM
running in Docker). The /apply endpoint returns all three load-case outputs;
the /jacobian endpoint returns server-side central-FD gradients. Client-side
chain rule propagates gradients through the softplus latent reparameterization
to a numpy Adam update — no JAX tracing or subprocess calls required.

Parameters (physical)
  arm_width  — gap between panel edges [m]
  fold_top   — far anchor depth along face edge [m]
  fold_bot   — near anchor depth [m] (0 = corner; must be < fold_top)
  waist_top  — Bézier control-point fold-dir offset for far curve [m]
  waist_bot  — Bézier control-point fold-dir offset for near curve [m]

Gradient cost per epoch:
  POST /apply    →  1 forward pass (3 SOFA sims: rotation + shear + tension)
  POST /jacobian →  10 central-FD perturbations × 3 modes = 30 SOFA sims
  Total: ~33 SOFA simulations per optimizer step.

Usage
-----
    # Start the Tesseract server:
    docker run -p 8000:8000 nff-sofa-oracle

    conda run -n kgnn_mac python sofa/hinge_optimizer.py \\
        --config data/configs/sofa/hinge_opt_2face.yaml \\
        [--n-epochs 30] [--lr 0.05] [--tesseract-url http://localhost:8000]

Outputs  data/outputs/hinge_opt/<timestamp>_<config>/
    config.yaml         — copy of driving config
    convergence.npz     — per-epoch arrays for all 5 params + losses
    convergence.png     — 6-panel convergence figure
"""

from __future__ import annotations

import argparse
import datetime
import os
import pathlib
import shutil
import sys
import types

import numpy as np
import requests
import yaml


# JAX CPU + x64 — needed only for CentroidalState.from_tessellation().
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')
import jax
jax.config.update('jax_enable_x64', True)

REPO = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from nff.topology.core    import UnitPattern
from nff.topology.builder import build_tessellation
from nff.stages.state     import CentroidalState

PATTERNS_FILE = REPO / 'data' / 'library' / 'patterns.yaml'
OUTPUTS_DIR   = REPO / 'data' / 'outputs' / 'hinge_opt'

TESSERACT_DEFAULT_URL = 'http://localhost:8000'

# Differentiable outputs requested from Tesseract /jacobian.
_JAC_INPUTS  = ['arm_width_physical', 'fold_top', 'fold_bot', 'waist_top', 'waist_bot']
_JAC_OUTPUTS = ['strain_energy', 'energy_shear', 'energy_tension', 'max_von_mises_stress']
# Physical param ordering must match _JAC_INPUTS.
_PHYS_ORDER  = ['arm_width', 'fold_top', 'fold_bot', 'waist_top', 'waist_bot']


# ── Numerics ──────────────────────────────────────────────────────────────────

def _softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable log(1 + exp(x))."""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


class _NumpyAdam:
    """Minimal Adam optimizer in numpy — no JAX or optax dependency."""

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


# ── Tesseract HTTP client ─────────────────────────────────────────────────────

def _call_tesseract_apply(url: str, payload: dict) -> dict:
    """POST /apply and return the OutputSchema as a plain dict."""
    try:
        resp = requests.post(f"{url}/apply", json={"inputs": payload}, timeout=300)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot connect to Tesseract server at {url}.\n"
            "Start the server: docker run -p 8000:8000 nff-sofa-oracle"
        )
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(
            f"Tesseract /apply failed ({resp.status_code}):\n{resp.text[:800]}"
        ) from e


def _call_tesseract_jacobian(
    url: str,
    payload: dict,
    jac_inputs: list[str],
    jac_outputs: list[str],
) -> dict:
    """POST /jacobian and return {output: {input: float}} Jacobian dict.

    Tesseract HTTP body convention:
      {"inputs": <InputSchema dict>, "jac_inputs": [...], "jac_outputs": [...]}
    """
    body = {"inputs": payload, "jac_inputs": jac_inputs, "jac_outputs": jac_outputs}
    try:
        resp = requests.post(f"{url}/jacobian", json=body, timeout=1200)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot connect to Tesseract server at {url}.\n"
            "Start the server: docker run -p 8000:8000 nff-sofa-oracle"
        )
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(
            f"Tesseract /jacobian failed ({resp.status_code}):\n{resp.text[:800]}"
        ) from e


# ── Pattern loading ───────────────────────────────────────────────────────────

def _load_pattern(pattern_name: str) -> UnitPattern:
    with open(PATTERNS_FILE) as f:
        patterns = yaml.safe_load(f)
    if pattern_name not in patterns:
        raise ValueError(f"Pattern '{pattern_name}' not found in {PATTERNS_FILE}")
    raw = patterns[pattern_name]
    internal_hinges = []
    for h in raw.get('internal_hinges', []):
        hc = dict(h)
        if 'angle_factor' in hc:
            hc['angle'] = float(hc.pop('angle_factor')) * float(np.pi)
        internal_hinges.append(hc)
    return UnitPattern(
        vertices=np.array(raw['vertices'], dtype=float),
        faces=raw['faces'],
        internal_hinges=internal_hinges,
        external_hinges=raw.get('external_hinges', []),
    )


# ── CentroidalState setup ─────────────────────────────────────────────────────

def build_physical_cs(cfg: dict) -> types.SimpleNamespace:
    """Build a SimpleNamespace with the CS fields needed for Tesseract payloads.

    The tessellation is built from the pattern at JAX-normalized scale, then
    scaled by face_size_m to produce SI-unit coordinates for SOFA.
    """
    tess_cfg = cfg.get('tessellation', {})
    sofa_cfg = cfg.get('sofa', {})
    bc_cfg   = cfg.get('boundary_conditions', {})

    pattern_name = tess_cfg.get('pattern', 'unit_2face')
    nx = int(tess_cfg.get('width', 1))
    ny = int(tess_cfg.get('height', 1))

    pattern      = _load_pattern(pattern_name)
    tessellation = build_tessellation(pattern, nx=nx, ny=ny)

    n_faces      = len(tessellation.faces)
    clamped_face = int(bc_cfg.get('clamped_face', 0))
    loaded_face  = int(bc_cfg.get('loaded_face', n_faces - 1))

    tessellation.set_face_dofs(clamped_face, [0, 1, 2])
    tessellation.set_face_load(loaded_face, dof_id=2, value=1.0)

    cs = CentroidalState.from_tessellation(tessellation)

    face_size_m = float(sofa_cfg.get('face_size_m', 0.100))

    return types.SimpleNamespace(
        face_centroids             = np.array(cs.face_centroids)        * face_size_m,
        centroid_node_vectors      = np.array(cs.centroid_node_vectors) * face_size_m,
        hinge_node_pairs           = np.array(cs.hinge_node_pairs,           dtype=np.int32),
        hinge_adj_info             = np.array(cs.hinge_adj_info,             dtype=np.int32),
        constrained_face_DOF_pairs = np.array(cs.constrained_face_DOF_pairs, dtype=np.int32),
        loaded_face_DOF_pairs      = np.array(cs.loaded_face_DOF_pairs,      dtype=np.int32),
    )


# ── Payload builder ───────────────────────────────────────────────────────────

def _build_tesseract_payload(
    cs: types.SimpleNamespace,
    phys: dict,
    cfg: dict,
    clamped_faces: list,
    loaded_faces: list,
) -> dict:
    """Assemble the full Tesseract InputSchema as a JSON-serializable dict."""
    sofa_cfg = cfg.get('sofa', {})
    mat_cfg  = cfg.get('material', {})

    return {
        # CS topology
        "face_centroids":        cs.face_centroids.tolist(),
        "centroid_node_vectors": cs.centroid_node_vectors.tolist(),
        "hinge_node_pairs":      cs.hinge_node_pairs.tolist(),
        "hinge_adj_info":        cs.hinge_adj_info.tolist(),
        "clamped_faces":         clamped_faces,
        "loaded_faces":          loaded_faces,
        # Differentiable hinge params
        "arm_width_physical":   float(phys['arm_width']),
        "fold_top":             float(phys['fold_top']),
        "fold_bot":             float(phys['fold_bot']),
        "waist_top":            float(phys['waist_top']),
        "waist_bot":            float(phys['waist_bot']),
        # Mesh resolution
        "n_ctrl":               int(sofa_cfg.get('n_ctrl', 3)),
        "sheet_thickness":      float(sofa_cfg.get('sheet_thickness', 0.001)),
        "n_face":               int(sofa_cfg.get('n_face', 4)),
        "n_hinge":              int(sofa_cfg.get('n_hinge', 2)),
        "n_z":                  int(sofa_cfg.get('n_z', 2)),
        # Loading
        "rotation_angle_deg":     float(sofa_cfg.get('rotation_angle_deg', -5.0)),
        "applied_moment":         0.0,
        "loading_mode":           'rotation',
        "shear_displacement_m":   float(sofa_cfg.get('shear_displacement_m', 0.005)),
        "tension_displacement_m": float(sofa_cfg.get('tension_displacement_m', 0.005)),
        # Material
        "young_modulus":  float(mat_cfg.get('young_modulus', 3.5e9)),
        "poisson_ratio":  float(mat_cfg.get('poisson_ratio', 0.36)),
        "yield_strength": float(mat_cfg.get('yield_strength', 50e6)),
        # FD step
        "fd_eps": float(sofa_cfg.get('fd_eps', 1e-5)),
    }


# ── Latent-to-physical Jacobian ───────────────────────────────────────────────

def _lat_to_phys_jacobian(
    latent: np.ndarray,
    arm_scale: float, delta_scale: float, fold_bot_scale: float,
    waist_top_scale: float, waist_bot_scale: float,
) -> np.ndarray:
    """5×5 Jacobian: d(physical) / d(latent).

    physical = [arm_width, fold_top, fold_bot, waist_top, waist_bot]
    latent   = [lat0,      lat1,     lat2,     lat3,      lat4     ]

    lat[1] → delta = fold_top - fold_bot  (via delta_scale)
    lat[2] → fold_bot                     (via fold_bot_scale)
    fold_top = fold_bot + delta  → both lat[1] and lat[2] affect fold_top.
    """
    sig = _sigmoid(latent)
    J = np.zeros((5, 5), dtype=np.float64)
    J[0, 0] = sig[0] * arm_scale           # d(arm_width)/d(lat0)
    J[1, 1] = sig[1] * delta_scale         # d(fold_top)/d(lat1)  via delta
    J[1, 2] = sig[2] * fold_bot_scale      # d(fold_top)/d(lat2)  via fold_bot
    J[2, 2] = sig[2] * fold_bot_scale      # d(fold_bot)/d(lat2)
    J[3, 3] = sig[3] * waist_top_scale     # d(waist_top)/d(lat3)
    J[4, 4] = sig[4] * waist_bot_scale     # d(waist_bot)/d(lat4)
    return J


# ── Optimization loop ─────────────────────────────────────────────────────────

def run_optimization(
    cfg: dict,
    n_epochs: int,
    lr: float,
    tesseract_url: str,
    out_dir: pathlib.Path,
) -> dict:
    sofa_cfg = cfg.get('sofa', {})
    loss_cfg = cfg.get('loss', {})
    mat_cfg  = cfg.get('material', {})

    arm_init       = float(sofa_cfg.get('arm_width_initial',   0.010))
    fold_top_init  = float(sofa_cfg.get('fold_top_initial',
                           sofa_cfg.get('fold_length_initial', 0.003)))
    fold_bot_init  = float(sofa_cfg.get('fold_bot_initial',   0.000))
    waist_top_init = float(sofa_cfg.get('waist_top_initial',  0.000))
    waist_bot_init = float(sofa_cfg.get('waist_bot_initial',  0.000))

    alpha    = float(loss_cfg.get('alpha',    1.0))
    beta     = float(loss_cfg.get('beta',     1.0))
    gamma    = float(loss_cfg.get('gamma',    1.0))
    lambda_p = float(loss_cfg.get('lambda_p', 1e-10))
    yield_str = float(mat_cfg.get('yield_strength', 50e6))

    cs_static = build_physical_cs(cfg)

    # Extract clamped/loaded face indices from CS DOF arrays.
    clamped_faces = sorted({int(f) for f in cs_static.constrained_face_DOF_pairs[:, 0]})
    loaded_faces  = sorted({int(f) for f in cs_static.loaded_face_DOF_pairs[:, 0]})

    # ── Latent-space scales ───────────────────────────────────────────────────
    # softplus(0) = log(2) ≈ 0.6931  →  scale = init / softplus(0)
    sp0 = np.log(2.0)
    delta_fold_init  = max(fold_top_init - fold_bot_init, fold_top_init * 0.1)
    waist_scale_dflt = fold_top_init * 0.5   # fallback scale for zero inits

    arm_scale       = arm_init       / sp0
    delta_scale     = delta_fold_init / sp0
    fold_bot_scale  = fold_bot_init  / sp0 if fold_bot_init  > 0 else waist_scale_dflt
    waist_top_scale = waist_top_init / sp0 if waist_top_init > 0 else waist_scale_dflt
    waist_bot_scale = waist_bot_init / sp0 if waist_bot_init > 0 else waist_scale_dflt

    # Initial latent values: lat=0 → physical=init for positive inits;
    # lat=-5 → softplus(-5)*scale ≈ 0 for zero inits.
    def _lat0(init: float) -> float:
        return 0.0 if init > 0 else -5.0

    latent = np.array([
        _lat0(arm_init),
        _lat0(delta_fold_init),
        _lat0(fold_bot_init),
        _lat0(waist_top_init),
        _lat0(waist_bot_init),
    ], dtype=np.float64)

    def to_physical(lat: np.ndarray) -> dict:
        """5-element latent → physical param dict (all positive by construction)."""
        arm_w    = _softplus(lat[0]) * arm_scale
        delta    = _softplus(lat[1]) * delta_scale
        fold_bot = _softplus(lat[2]) * fold_bot_scale
        fold_top = fold_bot + delta
        return {
            'arm_width': arm_w,
            'fold_top':  fold_top,
            'fold_bot':  fold_bot,
            'waist_top': _softplus(lat[3]) * waist_top_scale,
            'waist_bot': _softplus(lat[4]) * waist_bot_scale,
        }

    optimizer = _NumpyAdam(lr)
    history: dict = {k: [] for k in [
        'arm_width', 'fold_top', 'fold_bot', 'waist_top', 'waist_bot',
        'total_loss', 'energy_rot', 'max_vm_rot', 'energy_shear', 'energy_tension',
    ]}

    print(f"\nHinge optimization (Bézier 5-param): {n_epochs} epochs, lr={lr}")
    print(f"  Tesseract URL: {tesseract_url}")
    print(f"  loss = {alpha}*E_rot - {beta}*E_shear - {gamma}*E_tens + {lambda_p}*yield_penalty")
    print(f"  Initial: arm={arm_init*1e3:.2f} mm  fold_top={fold_top_init*1e3:.2f} mm  "
          f"fold_bot={fold_bot_init*1e3:.2f} mm  "
          f"waist=({waist_top_init*1e3:.2f},{waist_bot_init*1e3:.2f}) mm\n")

    for epoch in range(n_epochs):
        phys    = to_physical(latent)
        payload = _build_tesseract_payload(cs_static, phys, cfg, clamped_faces, loaded_faces)

        # ── Forward pass: all metrics ─────────────────────────────────────────
        fwd    = _call_tesseract_apply(tesseract_url, payload)
        def _get_val(v):
            if isinstance(v, dict):
                if 'data' in v and 'buffer' in v['data']: return float(v['data']['buffer'])
                if 'value' in v: return float(v['value'])
            return float(v)
        
        e_rot    = _get_val(fwd['strain_energy'])
        max_vm   = _get_val(fwd['max_von_mises_stress'])
        e_shear  = _get_val(fwd['energy_shear'])
        e_tens   = _get_val(fwd['energy_tension'])
        yield_ex = max(0.0, max_vm - yield_str)
        loss     = alpha * e_rot - beta * e_shear - gamma * e_tens + lambda_p * yield_ex

        # ── Backward pass: Jacobian from Tesseract ────────────────────────────
        jac = _call_tesseract_jacobian(tesseract_url, payload, _JAC_INPUTS, _JAC_OUTPUTS)

        # d(loss)/d(physical_param) for each of the 5 differentiable params.
        # _JAC_INPUTS order: arm_width_physical, fold_top, fold_bot, waist_top, waist_bot
        # _PHYS_ORDER order: arm_width,          fold_top, fold_bot, waist_top, waist_bot
        yield_active = 1.0 if max_vm > yield_str else 0.0
        dloss_dphys = np.array([
            (alpha    * _get_val(jac['strain_energy'][ki])
             - beta   * _get_val(jac['energy_shear'][ki])
             - gamma  * _get_val(jac['energy_tension'][ki])
             + lambda_p * yield_active * _get_val(jac['max_von_mises_stress'][ki]))
            for ki in _JAC_INPUTS
        ], dtype=np.float64)

        # Chain through latent space: d(loss)/d(lat) = J_phys_lat.T @ d(loss)/d(phys)
        J_phys_lat = _lat_to_phys_jacobian(
            latent, arm_scale, delta_scale,
            fold_bot_scale, waist_top_scale, waist_bot_scale,
        )
        dloss_dlat = J_phys_lat.T @ dloss_dphys

        latent = optimizer.update(latent, dloss_dlat)

        # ── Record ────────────────────────────────────────────────────────────
        history['arm_width'].append(phys['arm_width'])
        history['fold_top'].append(phys['fold_top'])
        history['fold_bot'].append(phys['fold_bot'])
        history['waist_top'].append(phys['waist_top'])
        history['waist_bot'].append(phys['waist_bot'])
        history['total_loss'].append(loss)
        history['energy_rot'].append(e_rot)
        history['max_vm_rot'].append(max_vm)
        history['energy_shear'].append(e_shear)
        history['energy_tension'].append(e_tens)

        print(f"  epoch {epoch+1:3d}/{n_epochs}  "
              f"loss={loss:.4e}  "
              f"E_rot={e_rot:.3e} J  "
              f"σ_max={max_vm/1e6:.1f} MPa  "
              f"arm={phys['arm_width']*1e3:.3f} mm  "
              f"fold_top={phys['fold_top']*1e3:.3f} mm  "
              f"fold_bot={phys['fold_bot']*1e3:.3f} mm  "
              f"waist=({phys['waist_top']*1e3:.2f},{phys['waist_bot']*1e3:.2f}) mm")

    np.savez(
        str(out_dir / 'convergence.npz'),
        arm_width      = np.array(history['arm_width']),
        fold_top       = np.array(history['fold_top']),
        fold_bot       = np.array(history['fold_bot']),
        waist_top      = np.array(history['waist_top']),
        waist_bot      = np.array(history['waist_bot']),
        total_loss     = np.array(history['total_loss']),
        energy_rot     = np.array(history['energy_rot']),
        max_vm_rot     = np.array(history['max_vm_rot']),
        energy_shear   = np.array(history['energy_shear']),
        energy_tension = np.array(history['energy_tension']),
        # backward-compat aliases
        fold_length    = np.array(history['fold_top']),
        stress         = np.array(history['max_vm_rot']),
    )
    return history



# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description='Optimize SOFA hinge dimensions via Tesseract HTTP.')
    p.add_argument('--config',
                   default='data/configs/sofa/hinge_opt_2face.yaml',
                   help='Path to YAML config.')
    p.add_argument('--n-epochs',    type=int,   default=None,
                   help='Override optimization.n_epochs from config.')
    p.add_argument('--lr',          type=float, default=None,
                   help='Override optimization.learning_rate from config.')
    p.add_argument('--tesseract-url', default=TESSERACT_DEFAULT_URL,
                   help='URL of the running Tesseract server (Docker).')
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    opt_cfg  = cfg.get('optimization', {})
    n_epochs = args.n_epochs if args.n_epochs is not None else int(opt_cfg.get('n_epochs', 30))
    lr       = args.lr       if args.lr       is not None else float(opt_cfg.get('learning_rate', 0.05))
    url      = args.tesseract_url

    # Verify server is reachable before committing to a long run.
    try:
        requests.get(url, timeout=5)
    except requests.exceptions.ConnectionError:
        print(f"ERROR: cannot reach Tesseract server at {url}")
        print("Start the server:  docker run -p 8000:8000 nff-sofa-oracle")
        raise SystemExit(1)

    config_name = pathlib.Path(args.config).stem
    timestamp   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir     = OUTPUTS_DIR / f'{timestamp}_{config_name}'
    out_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(args.config, out_dir / 'config.yaml')

    history = run_optimization(cfg, n_epochs, lr, url, out_dir)

    best_idx = int(np.argmin(history['max_vm_rot']))
    print(f'\nBest (epoch {best_idx + 1}): '
          f'σ_max={history["max_vm_rot"][best_idx]:.4e} Pa  '
          f'arm={history["arm_width"][best_idx]*1e3:.3f} mm  '
          f'fold_top={history["fold_top"][best_idx]*1e3:.3f} mm  '
          f'fold_bot={history["fold_bot"][best_idx]*1e3:.3f} mm  '
          f'waist=({history["waist_top"][best_idx]*1e3:.3f},'
          f'{history["waist_bot"][best_idx]*1e3:.3f}) mm')
    print(f'Results saved → {out_dir}')


if __name__ == '__main__':
    main()
