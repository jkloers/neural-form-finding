"""
sofa/hinge_optimizer.py — Corner-hinge cubic Bézier optimizer via Tesseract HTTP.

Optimizes 7 hinge parameters using the Tesseract physics oracle (SOFA FEM
running in Docker). The /apply endpoint returns all three load-case outputs;
the /jacobian endpoint returns server-side central-FD gradients. Client-side
chain rule propagates gradients through the softplus latent reparameterization
to a numpy Adam update — no JAX tracing or subprocess calls required.

Parameters (physical)
  arm_width  — gap between panel edges [m]                       (softplus, positive)
  fold_top   — far anchor depth along face edge [m]              (softplus, positive)
  fold_bot   — near anchor depth [m] (≈0 for corner)            (softplus, positive)
  bc1_x, bc1_y   — interior Bézier CP 1, upper wing [m]         (free in ℝ)
  bc2_x, bc2_y   — interior Bézier CP 2, upper wing [m]         (free in ℝ)
  bc1l_x, bc1l_y — interior Bézier CP 1, lower wing [m]         (free in ℝ)
  bc2l_x, bc2l_y — interior Bézier CP 2, lower wing [m]         (free in ℝ)
  Lower wing CPs initialised as the earm-mirror of upper wing (symmetric start).

Gradient cost per epoch:
  POST /apply    →  1 forward pass (1–3 SOFA sims depending on skip_secondary_modes)
  POST /jacobian →  22 central-FD perturbations × 1 mode = 22 SOFA sims (rotation-only)
  Total: ~23 SOFA simulations per optimizer step (rotation-only mode).

Usage
-----
    # Start the Tesseract server:
    docker run -p 8000:8000 nff-sofa-oracle

    conda run -n kgnn_mac python sofa/hinge_optimizer.py \\
        --config data/configs/sofa/hinge_opt_2face.yaml \\
        [--n-epochs 30] [--lr 0.05] [--tesseract-url http://localhost:8000]

Outputs  data/outputs/hinge_opt/<timestamp>_<config>/
    config.yaml         — copy of driving config
    convergence.npz     — per-epoch arrays for all 7 params + losses
    convergence.png     — convergence figure
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
_JAC_INPUTS  = [
    'arm_width_physical', 'fold_top', 'fold_bot',
    'bc1_x', 'bc1_y', 'bc2_x', 'bc2_y',
    'bc1l_x', 'bc1l_y', 'bc2l_x', 'bc2l_y',
]
_JAC_OUTPUTS = ['strain_energy', 'energy_shear', 'energy_tension', 'max_von_mises_stress']
# Physical param ordering must match _JAC_INPUTS.
_PHYS_ORDER  = [
    'arm_width', 'fold_top', 'fold_bot',
    'bc1_x', 'bc1_y', 'bc2_x', 'bc2_y',
    'bc1l_x', 'bc1l_y', 'bc2l_x', 'bc2l_y',
]


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
        # Differentiable hinge params (11-param corner-hinge cubic Bézier)
        "arm_width_physical":   float(phys['arm_width']),
        "fold_top":             float(phys['fold_top']),
        "fold_bot":             float(phys['fold_bot']),
        "bc1_x":                float(phys['bc1_x']),
        "bc1_y":                float(phys['bc1_y']),
        "bc2_x":                float(phys['bc2_x']),
        "bc2_y":                float(phys['bc2_y']),
        "bc1l_x":               float(phys['bc1l_x']),
        "bc1l_y":               float(phys['bc1l_y']),
        "bc2l_x":               float(phys['bc2l_x']),
        "bc2l_y":               float(phys['bc2l_y']),
        # Mesh resolution
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
        "skip_secondary_modes":   bool(sofa_cfg.get('skip_secondary_modes', False)),
        "n_steps":                int(sofa_cfg.get('n_steps', 500)),
        "fem_method":             str(sofa_cfg.get('fem_method', 'polar')),
        "rotation_pivot_auto":    bool(sofa_cfg.get('rotation_pivot_auto', True)),
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
) -> np.ndarray:
    """11×11 Jacobian: d(physical) / d(latent).

    physical = [arm_width, fold_top, fold_bot,
                bc1_x, bc1_y, bc2_x, bc2_y,
                bc1l_x, bc1l_y, bc2l_x, bc2l_y]
    latent   = [lat0,    lat1,    lat2,
                lat3,  lat4,  lat5,  lat6,
                lat7,   lat8,   lat9,   lat10 ]

    lat[0] → arm_width         (softplus, positive)
    lat[1] → delta = fold_top - fold_bot  (softplus, positive)
    lat[2] → fold_bot          (softplus, positive)
    fold_top = fold_bot + delta → both lat[1] and lat[2] affect fold_top.
    lat[3..10] → bc coords (upper + lower)  (identity, free in ℝ)
    """
    sig = _sigmoid(latent[:3])
    J = np.zeros((11, 11), dtype=np.float64)
    J[0, 0] = sig[0] * arm_scale       # d(arm_width)/d(lat0)
    J[1, 1] = sig[1] * delta_scale     # d(fold_top)/d(lat1)  via delta
    J[1, 2] = sig[2] * fold_bot_scale  # d(fold_top)/d(lat2)  via fold_bot
    J[2, 2] = sig[2] * fold_bot_scale  # d(fold_bot)/d(lat2)
    for i in range(3, 11):             # bc coords: identity
        J[i, i] = 1.0
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

    arm_init      = float(sofa_cfg.get('arm_width_initial',   0.003))
    fold_top_init = float(sofa_cfg.get('fold_top_initial',
                          sofa_cfg.get('fold_length_initial', 0.008)))
    fold_bot_init = float(sofa_cfg.get('fold_bot_initial',   0.00005))
    bc1_x_init    = float(sofa_cfg.get('bc1_x_initial',      0.14425))
    bc1_y_init    = float(sofa_cfg.get('bc1_y_initial',      0.07937))
    bc2_x_init    = float(sofa_cfg.get('bc2_x_initial',      0.13859))
    bc2_y_init    = float(sofa_cfg.get('bc2_y_initial',      0.07937))
    # Lower wing: default to earm-mirror of upper wing (symmetric start).
    bc1l_x_init   = float(sofa_cfg.get('bc1l_x_initial',     0.14159))
    bc1l_y_init   = float(sofa_cfg.get('bc1l_y_initial',     0.07937))
    bc2l_x_init   = float(sofa_cfg.get('bc2l_x_initial',     0.14725))
    bc2l_y_init   = float(sofa_cfg.get('bc2l_y_initial',     0.07937))

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
    delta_fold_init = max(fold_top_init - fold_bot_init, fold_top_init * 0.1)
    fold_bot_scale_dflt = fold_top_init * 0.1

    arm_scale      = arm_init       / sp0
    delta_scale    = delta_fold_init / sp0
    fold_bot_scale = fold_bot_init   / sp0 if fold_bot_init > 0 else fold_bot_scale_dflt

    # Initial latent values: lat=0 → physical=init via softplus for positive dims;
    # bc coords: latent IS physical (identity).
    def _lat0_positive(init: float) -> float:
        return 0.0 if init > 0 else -5.0

    latent = np.array([
        _lat0_positive(arm_init),
        _lat0_positive(delta_fold_init),
        _lat0_positive(fold_bot_init),
        bc1_x_init,    # identity mapping — latent = physical
        bc1_y_init,
        bc2_x_init,
        bc2_y_init,
        bc1l_x_init,
        bc1l_y_init,
        bc2l_x_init,
        bc2l_y_init,
    ], dtype=np.float64)

    def to_physical(lat: np.ndarray) -> dict:
        """11-element latent → physical param dict."""
        arm_w    = _softplus(lat[0]) * arm_scale
        delta    = _softplus(lat[1]) * delta_scale
        fold_bot = _softplus(lat[2]) * fold_bot_scale
        fold_top = fold_bot + delta
        return {
            'arm_width': arm_w,
            'fold_top':  fold_top,
            'fold_bot':  fold_bot,
            'bc1_x':     lat[3],
            'bc1_y':     lat[4],
            'bc2_x':     lat[5],
            'bc2_y':     lat[6],
            'bc1l_x':    lat[7],
            'bc1l_y':    lat[8],
            'bc2l_x':    lat[9],
            'bc2l_y':    lat[10],
        }

    optimizer = _NumpyAdam(lr)
    history: dict = {k: [] for k in [
        'arm_width', 'fold_top', 'fold_bot',
        'bc1_x', 'bc1_y', 'bc2_x', 'bc2_y',
        'bc1l_x', 'bc1l_y', 'bc2l_x', 'bc2l_y',
        'total_loss', 'energy_rot', 'max_vm_rot', 'energy_shear', 'energy_tension',
    ]}

    print(f"\nHinge optimization (corner-hinge cubic Bézier, 11-param): {n_epochs} epochs, lr={lr}")
    print(f"  Tesseract URL: {tesseract_url}")
    print(f"  loss = {alpha}*E_rot - {beta}*E_shear - {gamma}*E_tens + {lambda_p}*yield_penalty")
    print(f"  Initial: arm={arm_init*1e3:.2f} mm  fold_top={fold_top_init*1e3:.2f} mm  "
          f"fold_bot={fold_bot_init*1e3:.3f} mm")
    print(f"           bc1_up=({bc1_x_init*1e3:.1f}, {bc1_y_init*1e3:.1f}) mm  "
          f"bc2_up=({bc2_x_init*1e3:.1f}, {bc2_y_init*1e3:.1f}) mm")
    print(f"           bc1_lo=({bc1l_x_init*1e3:.1f}, {bc1l_y_init*1e3:.1f}) mm  "
          f"bc2_lo=({bc2l_x_init*1e3:.1f}, {bc2l_y_init*1e3:.1f}) mm\n")

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
        # When rotation-only, only request strain_energy gradient (no shear/tension sims).
        skip_secondary = payload.get('skip_secondary_modes', False)
        jac_outputs_req = ['strain_energy', 'max_von_mises_stress']
        if not skip_secondary:
            jac_outputs_req += ['energy_shear', 'energy_tension']
        jac = _call_tesseract_jacobian(tesseract_url, payload, _JAC_INPUTS, jac_outputs_req)

        # d(loss)/d(physical_param) for each of the 7 differentiable params.
        # _JAC_INPUTS order: arm_width_physical, fold_top, fold_bot, bc1_x, bc1_y, bc2_x, bc2_y
        yield_active = 1.0 if max_vm > yield_str else 0.0
        def _jac_val(output_key, inp_key):
            if output_key not in jac:
                return 0.0
            return _get_val(jac[output_key][inp_key])

        dloss_dphys = np.array([
            (alpha    * _jac_val('strain_energy', ki)
             - beta   * _jac_val('energy_shear', ki)
             - gamma  * _jac_val('energy_tension', ki)
             + lambda_p * yield_active * _jac_val('max_von_mises_stress', ki))
            for ki in _JAC_INPUTS
        ], dtype=np.float64)

        # Chain through latent space: d(loss)/d(lat) = J_phys_lat.T @ d(loss)/d(phys)
        J_phys_lat = _lat_to_phys_jacobian(latent, arm_scale, delta_scale, fold_bot_scale)
        dloss_dlat = J_phys_lat.T @ dloss_dphys

        latent = optimizer.update(latent, dloss_dlat)

        # ── Record ────────────────────────────────────────────────────────────
        history['arm_width'].append(phys['arm_width'])
        history['fold_top'].append(phys['fold_top'])
        history['fold_bot'].append(phys['fold_bot'])
        history['bc1_x'].append(phys['bc1_x'])
        history['bc1_y'].append(phys['bc1_y'])
        history['bc2_x'].append(phys['bc2_x'])
        history['bc2_y'].append(phys['bc2_y'])
        history['bc1l_x'].append(phys['bc1l_x'])
        history['bc1l_y'].append(phys['bc1l_y'])
        history['bc2l_x'].append(phys['bc2l_x'])
        history['bc2l_y'].append(phys['bc2l_y'])
        history['total_loss'].append(loss)
        history['energy_rot'].append(e_rot)
        history['max_vm_rot'].append(max_vm)
        history['energy_shear'].append(e_shear)
        history['energy_tension'].append(e_tens)

        print(f"  epoch {epoch+1:3d}/{n_epochs}  "
              f"loss={loss:.4e}  E_rot={e_rot:.3e} J  σ_max={max_vm/1e6:.1f} MPa  "
              f"arm={phys['arm_width']*1e3:.3f} mm  fold_top={phys['fold_top']*1e3:.3f} mm  "
              f"fold_bot={phys['fold_bot']*1e3:.4f} mm  "
              f"bc1_up=({phys['bc1_x']*1e3:.2f},{phys['bc1_y']*1e3:.2f})  "
              f"bc2_up=({phys['bc2_x']*1e3:.2f},{phys['bc2_y']*1e3:.2f})  "
              f"bc1_lo=({phys['bc1l_x']*1e3:.2f},{phys['bc1l_y']*1e3:.2f})  "
              f"bc2_lo=({phys['bc2l_x']*1e3:.2f},{phys['bc2l_y']*1e3:.2f}) mm")

    np.savez(
        str(out_dir / 'convergence.npz'),
        arm_width      = np.array(history['arm_width']),
        fold_top       = np.array(history['fold_top']),
        fold_bot       = np.array(history['fold_bot']),
        bc1_x          = np.array(history['bc1_x']),
        bc1_y          = np.array(history['bc1_y']),
        bc2_x          = np.array(history['bc2_x']),
        bc2_y          = np.array(history['bc2_y']),
        bc1l_x         = np.array(history['bc1l_x']),
        bc1l_y         = np.array(history['bc1l_y']),
        bc2l_x         = np.array(history['bc2l_x']),
        bc2l_y         = np.array(history['bc2l_y']),
        total_loss     = np.array(history['total_loss']),
        energy_rot     = np.array(history['energy_rot']),
        max_vm_rot     = np.array(history['max_vm_rot']),
        energy_shear   = np.array(history['energy_shear']),
        energy_tension = np.array(history['energy_tension']),
        fold_length    = np.array(history['fold_top']),   # backward-compat alias
        stress         = np.array(history['max_vm_rot']), # backward-compat alias
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
        requests.get(f"{url}/health", timeout=30)
    except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
        print(f"ERROR: cannot reach Tesseract server at {url}")
        print("Start the server:  docker run -p 8000:8000 -e TESSERACT_RUNTIME_SERVE_HOST=0.0.0.0 nff-sofa-oracle:latest serve")
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
          f'fold_bot={history["fold_bot"][best_idx]*1e3:.4f} mm  '
          f'bc1_up=({history["bc1_x"][best_idx]*1e3:.2f},{history["bc1_y"][best_idx]*1e3:.2f}) mm  '
          f'bc2_up=({history["bc2_x"][best_idx]*1e3:.2f},{history["bc2_y"][best_idx]*1e3:.2f}) mm  '
          f'bc1_lo=({history["bc1l_x"][best_idx]*1e3:.2f},{history["bc1l_y"][best_idx]*1e3:.2f}) mm  '
          f'bc2_lo=({history["bc2l_x"][best_idx]*1e3:.2f},{history["bc2l_y"][best_idx]*1e3:.2f}) mm')
    print(f'Results saved → {out_dir}')


if __name__ == '__main__':
    main()
