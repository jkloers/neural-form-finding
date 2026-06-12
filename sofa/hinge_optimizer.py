"""
sofa/hinge_optimizer.py — Corner-hinge cubic Bézier optimizer via Tesseract HTTP.

Optimizes 13 hinge parameters using the Tesseract physics oracle (SOFA FEM
running in Docker). The /apply endpoint returns all three load-case outputs;
the /jacobian endpoint returns server-side central-FD gradients. Client-side
chain rule propagates gradients through the softplus latent reparameterization
to a numpy Adam update — no JAX tracing or subprocess calls required.

Parameters (physical) — 5 positive (softplus) + 8 free (ℝ)
  gap                     — rigid face separation along ê_arm [m]   (softplus, positive)
  s0_top, s0_bot          — upper/lower endpoint reach on face 0 [m] (softplus, positive)
  s1_top, s1_bot          — upper/lower endpoint reach on face 1 [m] (softplus, positive)
  bc1u_x, bc1u_y          — interior Bézier CP 1, UPPER arc [m]       (free in ℝ)
  bc2u_x, bc2u_y          — interior Bézier CP 2, UPPER arc [m]       (free in ℝ)
  bc1l_x, bc1l_y          — interior Bézier CP 1, LOWER arc [m]       (free in ℝ)
  bc2l_x, bc2l_y          — interior Bézier CP 2, LOWER arc [m]       (free in ℝ)
  CP defaults come from the symmetric mesh geometry at the initial gap.

Gradient cost per epoch:
  POST /apply    →  1 forward pass (1–3 SOFA sims depending on skip_secondary_modes)
  POST /jacobian →  26 central-FD perturbations × 1 mode = 26 SOFA sims (rotation-only)
  Total: ~27 SOFA simulations per optimizer step (rotation-only mode).

Usage
-----
    # Start the Tesseract server:
    docker run -p 8000:8000 nff-sofa-oracle

    conda run -n kgnn_mac python sofa/hinge_optimizer.py \\
        --config data/configs/sofa/hinge_opt_2face.yaml \\
        [--n-epochs 30] [--lr 0.05] [--tesseract-url http://localhost:8000]

Outputs  data/outputs/hinge_opt/<timestamp>_<config>/
    config.yaml         — copy of driving config
    convergence.npz     — per-epoch arrays for all 13 params + losses
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

# Parameter layout: 5 positive (softplus) followed by 8 free CP coords.
# Names match the Tesseract InputSchema field names exactly.
_POS_NAMES   = ['gap', 's0_top', 's0_bot', 's1_top', 's1_bot']
_FREE_NAMES  = ['bc1u_x', 'bc1u_y', 'bc2u_x', 'bc2u_y',
                'bc1l_x', 'bc1l_y', 'bc2l_x', 'bc2l_y']
_PARAM_NAMES = _POS_NAMES + _FREE_NAMES            # 13, == _JAC_INPUTS order
_N_POS       = len(_POS_NAMES)

_JAC_INPUTS  = list(_PARAM_NAMES)


# ── Numerics ──────────────────────────────────────────────────────────────────

def _softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable log(1 + exp(x))."""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


def _decode_array(v) -> np.ndarray:
    """Decode a Tesseract array-output JSON value into a numpy array."""
    import base64
    if isinstance(v, dict) and v.get('object_type') == 'array':
        shape, dtype, data = tuple(v['shape']), v['dtype'], v['data']
        if data.get('encoding') == 'base64':
            buf = base64.b64decode(data['buffer'])
            return np.frombuffer(buf, dtype=dtype).reshape(shape).copy()
        return np.asarray(data['buffer'], dtype=dtype).reshape(shape)
    return np.asarray(v)


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
        # Differentiable hinge params (13-param corner-hinge cubic Bézier)
        **{name: float(phys[name]) for name in _PARAM_NAMES},
        # Mesh resolution
        "sheet_thickness":      float(sofa_cfg.get('sheet_thickness', 0.001)),
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

def _lat_to_phys_jacobian(latent: np.ndarray, pos_scale: np.ndarray) -> np.ndarray:
    """13×13 diagonal Jacobian: d(physical) / d(latent).

    The first 5 params (gap, s0_top, s0_bot, s1_top, s1_bot) are positive via
    softplus: phys = softplus(lat) * scale → d(phys)/d(lat) = sigmoid(lat)*scale.
    The remaining 8 CP coordinates map identically (latent == physical).
    """
    J = np.zeros((13, 13), dtype=np.float64)
    diag_pos = _sigmoid(latent[:_N_POS]) * pos_scale
    J[np.arange(_N_POS), np.arange(_N_POS)] = diag_pos
    for i in range(_N_POS, 13):
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
    mat_cfg  = cfg.get('material', {})

    yield_str = float(mat_cfg.get('yield_strength', 50e6))

    cs_static = build_physical_cs(cfg)

    # ── Positive-param initial values (gap + 4 reaches) ───────────────────────
    gap_init   = float(sofa_cfg.get('gap_initial',   0.003))
    reach_init = float(sofa_cfg.get('reach_initial', gap_init))
    pos_init = np.array([
        gap_init,
        float(sofa_cfg.get('s0_top_initial', reach_init)),
        float(sofa_cfg.get('s0_bot_initial', reach_init)),
        float(sofa_cfg.get('s1_top_initial', reach_init)),
        float(sofa_cfg.get('s1_bot_initial', reach_init)),
    ], dtype=np.float64)

    # ── Free CP initial values — symmetric mesh geometry at the initial gap ───
    from nff.sofa.mesh_builder_gmsh import compute_hinge_geometry
    _hd0 = compute_hinge_geometry(cs_static, gap=gap_init)['hinge_data'][0]
    _cp_default = {
        'bc1u_x': _hd0['bc1_up'][0], 'bc1u_y': _hd0['bc1_up'][1],
        'bc2u_x': _hd0['bc2_up'][0], 'bc2u_y': _hd0['bc2_up'][1],
        'bc1l_x': _hd0['bc1_lo'][0], 'bc1l_y': _hd0['bc1_lo'][1],
        'bc2l_x': _hd0['bc2_lo'][0], 'bc2l_y': _hd0['bc2_lo'][1],
    }
    free_init = np.array([
        float(sofa_cfg.get(f'{name}_initial', _cp_default[name])) for name in _FREE_NAMES
    ], dtype=np.float64)

    # Extract clamped/loaded face indices from CS DOF arrays.
    clamped_faces = sorted({int(f) for f in cs_static.constrained_face_DOF_pairs[:, 0]})
    loaded_faces  = sorted({int(f) for f in cs_static.loaded_face_DOF_pairs[:, 0]})

    # ── Latent space ──────────────────────────────────────────────────────────
    # softplus(0) = log(2) → scale = init / softplus(0) so that lat=0 → init.
    sp0 = np.log(2.0)
    pos_scale = np.where(pos_init > 0, pos_init / sp0, 1.0)
    # lat0 for positive dims: 0 if init>0 (→ softplus(0)*scale=init), else -5.
    latent = np.concatenate([
        np.where(pos_init > 0, 0.0, -5.0),
        free_init,                       # identity mapping — latent = physical
    ]).astype(np.float64)

    def to_physical(lat: np.ndarray) -> dict:
        """13-element latent → physical param dict (keys == schema names)."""
        pos  = _softplus(lat[:_N_POS]) * pos_scale
        free = lat[_N_POS:]
        return {name: float(v) for name, v in
                zip(_PARAM_NAMES, np.concatenate([pos, free]))}

    optimizer = _NumpyAdam(lr)
    history: dict = {k: [] for k in _PARAM_NAMES + [
        'total_loss', 'energy_rot', 'max_vm_rot']}

    _p0 = to_physical(latent)
    print(f"\nHinge optimization (corner-hinge cubic Bézier, 13-param): {n_epochs} epochs, lr={lr}")
    print(f"  Tesseract URL: {tesseract_url}")
    print(f"  loss = σ_max / σ_yield  (minimize peak hinge stress; < 1 → survives)")
    print(f"  Initial: gap={_p0['gap']*1e3:.2f} mm  "
          f"reach s0=({_p0['s0_top']*1e3:.2f},{_p0['s0_bot']*1e3:.2f}) "
          f"s1=({_p0['s1_top']*1e3:.2f},{_p0['s1_bot']*1e3:.2f}) mm")
    print(f"           bc1_up=({_p0['bc1u_x']*1e3:.1f}, {_p0['bc1u_y']*1e3:.1f}) mm  "
          f"bc2_up=({_p0['bc2u_x']*1e3:.1f}, {_p0['bc2u_y']*1e3:.1f}) mm")
    print(f"           bc1_lo=({_p0['bc1l_x']*1e3:.1f}, {_p0['bc1l_y']*1e3:.1f}) mm  "
          f"bc2_lo=({_p0['bc2l_x']*1e3:.1f}, {_p0['bc2l_y']*1e3:.1f}) mm\n")

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
        # Objective: keep the hinge below yield while the panels close (displacement-
        # controlled rotation already forces the closing motion). loss = σ_max/σ_yield;
        # < 1 → survives, > 1 → breaks.
        loss     = max_vm / yield_str

        # ── Backward pass: Jacobian from Tesseract ────────────────────────────
        jac = _call_tesseract_jacobian(
            tesseract_url, payload, _JAC_INPUTS, ['max_von_mises_stress'])

        # d(loss)/d(param) = (1/σ_yield) · d(σ_max)/d(param), ordered as _JAC_INPUTS.
        def _jac_val(output_key, inp_key):
            if output_key not in jac:
                return 0.0
            return _get_val(jac[output_key][inp_key])

        dloss_dphys = np.array(
            [_jac_val('max_von_mises_stress', ki) / yield_str for ki in _JAC_INPUTS],
            dtype=np.float64)

        # Chain through latent space: d(loss)/d(lat) = J_phys_lat.T @ d(loss)/d(phys)
        J_phys_lat = _lat_to_phys_jacobian(latent, pos_scale)
        dloss_dlat = J_phys_lat.T @ dloss_dphys

        latent = optimizer.update(latent, dloss_dlat)

        # ── Record ────────────────────────────────────────────────────────────
        for name in _PARAM_NAMES:
            history[name].append(phys[name])
        history['total_loss'].append(loss)
        history['energy_rot'].append(e_rot)
        history['max_vm_rot'].append(max_vm)

        print(f"  epoch {epoch+1:3d}/{n_epochs}  "
              f"loss=σ/σy={loss:.3f}  σ_max={max_vm/1e6:.1f} MPa  "
              f"gap={phys['gap']*1e3:.3f} mm  "
              f"s0=({phys['s0_top']*1e3:.2f},{phys['s0_bot']*1e3:.2f}) "
              f"s1=({phys['s1_top']*1e3:.2f},{phys['s1_bot']*1e3:.2f}) mm  "
              f"bc1_up=({phys['bc1u_x']*1e3:.2f},{phys['bc1u_y']*1e3:.2f})  "
              f"bc2_up=({phys['bc2u_x']*1e3:.2f},{phys['bc2u_y']*1e3:.2f})  "
              f"bc1_lo=({phys['bc1l_x']*1e3:.2f},{phys['bc1l_y']*1e3:.2f})  "
              f"bc2_lo=({phys['bc2l_x']*1e3:.2f},{phys['bc2l_y']*1e3:.2f}) mm")

    np.savez(
        str(out_dir / 'convergence.npz'),
        **{name: np.array(history[name]) for name in _PARAM_NAMES},
        total_loss     = np.array(history['total_loss']),     # σ_max / σ_yield
        energy_rot     = np.array(history['energy_rot']),
        max_vm_rot     = np.array(history['max_vm_rot']),
        yield_strength = np.array(yield_str),
        stress         = np.array(history['max_vm_rot']),     # backward-compat alias
    )

    # ── Final state at the best design — capture the von Mises field for viz ──
    best_idx  = int(np.argmin(history['max_vm_rot']))
    best_phys = {name: float(history[name][best_idx]) for name in _PARAM_NAMES}
    print(f"\nCapturing final-state field at best design (epoch {best_idx + 1}) ...")
    payload = _build_tesseract_payload(cs_static, best_phys, cfg, clamped_faces, loaded_faces)
    payload['return_fields']        = True
    payload['skip_secondary_modes'] = True   # rotation field only
    try:
        fwd = _call_tesseract_apply(tesseract_url, payload)
        np.savez(
            str(out_dir / 'final_state.npz'),
            von_mises_field = _decode_array(fwd['von_mises_field']),
            deformed_nodes  = _decode_array(fwd['deformed_nodes']),
            mesh_tets       = _decode_array(fwd['mesh_tets']),
            best_idx        = np.array(best_idx),
            # CS topology so the visualizer can rebuild the initial/optimal meshes.
            face_centroids             = cs_static.face_centroids,
            centroid_node_vectors      = cs_static.centroid_node_vectors,
            hinge_node_pairs           = cs_static.hinge_node_pairs,
            hinge_adj_info             = cs_static.hinge_adj_info,
            constrained_face_DOF_pairs = cs_static.constrained_face_DOF_pairs,
            loaded_face_DOF_pairs      = cs_static.loaded_face_DOF_pairs,
        )
        print(f"  final_state.npz saved ({len(history['max_vm_rot'])} epochs, "
              f"σ_max={history['max_vm_rot'][best_idx]/1e6:.1f} MPa).")
    except Exception as ex:
        print(f"  WARNING: final-state field capture failed ({ex}); "
              "visualizer will skip the von Mises panel.")

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
    b = lambda k: history[k][best_idx] * 1e3
    print(f'\nBest (epoch {best_idx + 1}): '
          f'σ_max={history["max_vm_rot"][best_idx]:.4e} Pa  '
          f'gap={b("gap"):.3f} mm  '
          f's0=({b("s0_top"):.2f},{b("s0_bot"):.2f}) s1=({b("s1_top"):.2f},{b("s1_bot"):.2f}) mm  '
          f'bc1_up=({b("bc1u_x"):.2f},{b("bc1u_y"):.2f}) bc2_up=({b("bc2u_x"):.2f},{b("bc2u_y"):.2f}) mm  '
          f'bc1_lo=({b("bc1l_x"):.2f},{b("bc1l_y"):.2f}) bc2_lo=({b("bc2l_x"):.2f},{b("bc2l_y"):.2f}) mm')
    print(f'Results saved → {out_dir}')


if __name__ == '__main__':
    main()
