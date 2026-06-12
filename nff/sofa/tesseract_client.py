"""nff/sofa/tesseract_client.py — typed HTTP client for the SOFA Tesseract oracle.

Client-side (kgnn_mac); never imports ``Sofa``. This is the single, public bridge
between the JAX world and the Dockerised SOFA oracle:

  * ``apply`` / ``jacobian``       — the two oracle endpoints,
  * ``decode_scalar`` / ``decode_array`` — un-wrap Tesseract's JSON output values,
  * ``build_physical_cs``          — Tessellation → SI-unit CentroidalState fields,
  * ``build_payload``              — assemble the InputSchema dict.

Every consumer (the hinge optimizer and the viz scripts) goes through this module,
so the payload format and the output decoders are defined exactly once.
"""
from __future__ import annotations

import base64
import pathlib
import types

import numpy as np
import requests

# JAX (CPU, x64) is needed only by build_physical_cs → CentroidalState.from_tessellation.
import os
os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')
import jax
jax.config.update('jax_enable_x64', True)

from nff.topology.core    import UnitPattern
from nff.topology.builder import build_tessellation
from nff.stages.state     import CentroidalState

_REPO          = pathlib.Path(__file__).resolve().parents[2]
PATTERNS_FILE  = _REPO / 'data' / 'library' / 'patterns.yaml'
DEFAULT_URL    = 'http://localhost:8000'

# Parameter layout — names match the Tesseract InputSchema fields exactly.
POS_NAMES   = ['gap', 's0_top', 's0_bot', 's1_top', 's1_bot']   # positive, floored
FREE_NAMES  = ['bcu_x', 'bcu_y', 'bcl_x', 'bcl_y']              # one CP per arc
PARAM_NAMES = POS_NAMES + FREE_NAMES                            # 9, == Jacobian inputs


# ── Output decoders ────────────────────────────────────────────────────────────

def decode_scalar(value) -> float:
    """Un-wrap a Tesseract scalar output (``{'data':{'buffer':…}}`` / ``{'value':…}``)."""
    if isinstance(value, dict):
        if 'data' in value and 'buffer' in value['data']:
            return float(value['data']['buffer'])
        if 'value' in value:
            return float(value['value'])
    return float(value)


def decode_array(value) -> np.ndarray:
    """Un-wrap a Tesseract array output (base64 or plain list) into a NumPy array."""
    if isinstance(value, dict) and value.get('object_type') == 'array':
        shape, dtype, data = tuple(value['shape']), value['dtype'], value['data']
        if data.get('encoding') == 'base64':
            buf = base64.b64decode(data['buffer'])
            return np.frombuffer(buf, dtype=dtype).reshape(shape).copy()
        return np.asarray(data['buffer'], dtype=dtype).reshape(shape)
    return np.asarray(value)


# ── Endpoints ──────────────────────────────────────────────────────────────────

def apply(url: str, payload: dict, timeout: float = 600.0) -> dict:
    """POST /apply and return the OutputSchema as a plain dict."""
    return _post(url, 'apply', {'inputs': payload}, timeout)


def jacobian(url: str, payload: dict, jac_inputs: list[str],
             jac_outputs: list[str], timeout: float = 1800.0) -> dict:
    """POST /jacobian and return ``{output: {input: value}}``."""
    body = {'inputs': payload, 'jac_inputs': jac_inputs, 'jac_outputs': jac_outputs}
    return _post(url, 'jacobian', body, timeout)


def _post(url: str, endpoint: str, body: dict, timeout: float) -> dict:
    try:
        resp = requests.post(f"{url}/{endpoint}", json=body, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot connect to Tesseract server at {url}.\n"
            "Start the server: docker run -p 8000:8000 nff-sofa-oracle"
        )
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(
            f"Tesseract /{endpoint} failed ({resp.status_code}):\n{resp.text[:800]}"
        ) from exc


# ── CentroidalState → payload ──────────────────────────────────────────────────

def _load_pattern(pattern_name: str) -> UnitPattern:
    import yaml
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


def build_physical_cs(cfg: dict) -> types.SimpleNamespace:
    """Build the SI-unit CentroidalState fields needed for Tesseract payloads.

    The tessellation is built from the pattern at JAX-normalized scale, then scaled
    by ``face_size_m`` to physical metres for SOFA.
    """
    tess_cfg = cfg.get('tessellation', {})
    sofa_cfg = cfg.get('sofa', {})
    bc_cfg   = cfg.get('boundary_conditions', {})

    pattern      = _load_pattern(tess_cfg.get('pattern', 'unit_2face'))
    tessellation = build_tessellation(pattern, nx=int(tess_cfg.get('width', 1)),
                                      ny=int(tess_cfg.get('height', 1)))

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


def build_payload(cs: types.SimpleNamespace, phys: dict, cfg: dict,
                  clamped_faces: list, loaded_faces: list) -> dict:
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
        # Differentiable hinge params (9-param corner-hinge quadratic Bézier)
        **{name: float(phys[name]) for name in PARAM_NAMES},
        # Mesh resolution
        "sheet_thickness":      float(sofa_cfg.get('sheet_thickness', 0.001)),
        "n_z":                  int(sofa_cfg.get('n_z', 2)),
        "mesh_refine":          float(sofa_cfg.get('mesh_refine', 1.0)),
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
