"""
nff/sofa/hinge_optimizer.py — Black-box SOFA hinge optimization via JAX/Optax.

Optimizes arm_width_physical + fold_length to minimize max von Mises stress
under a prescribed in-plane rotation.  Gradients are computed via central
finite differences (4 SOFA calls per gradient step, ~20–120 s per epoch).

Bridges the reality gap: SOFA models the physical hinge geometry (3D FEM,
PLA material) while JAX/Optax handles the optimization loop.

Architecture
------------
  CentroidalState (static, from tessellation)
         │
         ▼
  build_mesh_from_centroidal_state(arm_width, fold_length)  [Python/kgnn_mac]
         │  save NPZ
         ▼
  run_sofa.sh run_hinge_eval.py                             [Homebrew Python/SOFA]
         │  load NPZ
         ▼
  max_von_mises_stress  ──►  custom_vjp + pure_callback ──►  jax.grad
         │
         ▼
  optax.adam update  →  (arm_width, fold_length)  [latent softplus space]

Usage
-----
    conda run -n kgnn_mac python nff/sofa/hinge_optimizer.py \\
        --config data/configs/sofa/hinge_opt_2face.yaml \\
        [--n-epochs 30] [--lr 0.05]

Outputs  data/outputs/hinge_opt/<timestamp>_<config>/
    config.yaml         — copy of driving config
    convergence.npz     — per-epoch: latent_params, arm_width, fold_length, stress
    convergence.png     — 3-panel: stress, arm_width, fold_length traces
"""

from __future__ import annotations

import argparse
import datetime
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import types

# JAX CPU + x64 before any jax import
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
import jax
jax.config.update('jax_enable_x64', True)

import numpy as np
import jax.numpy as jnp
import optax
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO))

from nff.topology.core import UnitPattern
from nff.topology.builder import build_tessellation
from nff.stages.state import CentroidalState
from nff.sofa.mesh_builder import build_mesh_from_centroidal_state

PATTERNS_FILE = REPO / 'data' / 'library' / 'patterns.yaml'
SOFA_RUNNER   = REPO / 'sofa' / 'run_sofa.sh'
EVAL_SCRIPT   = REPO / 'sofa' / 'run_hinge_eval.py'
OUTPUTS_DIR   = REPO / 'data' / 'outputs' / 'hinge_opt'


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

def build_physical_cs(cfg: dict):
    """Build a SimpleNamespace with the CS fields needed by the mesh builder.

    The tessellation is built from the pattern at JAX-normalized scale, then
    multiplied by face_size_m to produce SI-unit coordinates for SOFA.
    Only face_centroids, centroid_node_vectors, hinge_node_pairs, hinge_adj_info,
    constrained_face_DOF_pairs, and loaded_face_DOF_pairs are populated — the
    mesh builder needs only these six fields.
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
    # Placeholder load value — SOFA uses rotation_angle_deg, not this scalar.
    tessellation.set_face_load(loaded_face, dof_id=2, value=1.0)

    cs = CentroidalState.from_tessellation(tessellation)

    # Scale from normalized JAX units (face_size=1) to physical metres.
    face_size_m = float(sofa_cfg.get('face_size_m', 0.100))

    return types.SimpleNamespace(
        face_centroids             = np.array(cs.face_centroids)        * face_size_m,
        centroid_node_vectors      = np.array(cs.centroid_node_vectors) * face_size_m,
        hinge_node_pairs           = cs.hinge_node_pairs,
        hinge_adj_info             = cs.hinge_adj_info,
        constrained_face_DOF_pairs = cs.constrained_face_DOF_pairs,
        loaded_face_DOF_pairs      = cs.loaded_face_DOF_pairs,
    )


# ── Mesh + SOFA subprocess helpers ───────────────────────────────────────────

def _save_mesh_npz(cs, arm_width: float, fold_length: float,
                   sheet_thickness: float, n_face: int, n_hinge: int,
                   n_z: int, path: pathlib.Path) -> None:
    """Build hex mesh and persist as NPZ."""
    nodes, hexes, bc_masks = build_mesh_from_centroidal_state(
        cs,
        fold_length        = fold_length,
        sheet_thickness    = sheet_thickness,
        n_face             = n_face,
        n_hinge            = n_hinge,
        n_z                = n_z,
        arm_width_physical = arm_width,
    )
    n_faces = len(cs.face_centroids)
    save_dict: dict = {
        'nodes':   nodes,
        'hexes':   hexes,
        'n_faces': np.array(n_faces, dtype=np.int32),
    }
    for k, v in bc_masks.items():
        save_dict[f'{k}_mask'] = v
    np.savez(str(path), **save_dict)


def _run_sofa(mesh_npz: pathlib.Path, out_npz: pathlib.Path,
              rotation_deg: float, young: float, nu: float,
              yield_str: float, thickness: float) -> float:
    """Invoke SOFA via subprocess; return max_von_mises_stress [Pa]."""
    cmd = [
        str(SOFA_RUNNER), str(EVAL_SCRIPT),
        '--mesh-npz',           str(mesh_npz),
        '--out-file',           str(out_npz),
        '--rotation-angle-deg', str(rotation_deg),
        '--young-modulus',      str(young),
        '--poisson-ratio',      str(nu),
        '--yield-strength',     str(yield_str),
        '--sheet-thickness',    str(thickness),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"SOFA eval failed (returncode={proc.returncode}):\n"
            f"{proc.stderr[-3000:]}"
        )
    res = np.load(str(out_npz))
    return float(res['max_von_mises_stress'])


def _run_sofa_full(mesh_npz: pathlib.Path, out_npz: pathlib.Path,
                   arm_width: float, fold_length: float,
                   rotation_deg: float, young: float, nu: float,
                   yield_str: float, thickness: float) -> pathlib.Path:
    """Invoke SOFA with full-result NPZ (metadata + vm_per_hex); return out_npz."""
    cmd = [
        str(SOFA_RUNNER), str(EVAL_SCRIPT),
        '--mesh-npz',           str(mesh_npz),
        '--out-file',           str(out_npz),
        '--arm-width',          str(arm_width),
        '--fold-length',        str(fold_length),
        '--rotation-angle-deg', str(rotation_deg),
        '--young-modulus',      str(young),
        '--poisson-ratio',      str(nu),
        '--yield-strength',     str(yield_str),
        '--sheet-thickness',    str(thickness),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"SOFA eval failed (returncode={proc.returncode}):\n"
            f"{proc.stderr[-3000:]}"
        )
    return out_npz


# ── JAX-compatible SOFA stress function via custom_vjp ────────────────────────

def make_sofa_stress_fn(cs_static, cfg: dict):
    """Return a JAX-differentiable function: (arm_width, fold_length) → stress.

    The function wraps SOFA via jax.pure_callback + jax.custom_vjp so that
    jax.grad / jax.value_and_grad work correctly.  No JIT is applied to the
    outer optimization loop — each step makes ~5 synchronous SOFA calls.

    Gradient cost: 4 SOFA calls per step (central FD, 2 per differentiable param).
    """
    sofa_cfg = cfg.get('sofa', {})
    mat_cfg  = cfg.get('material', {})

    rotation_deg = float(sofa_cfg.get('rotation_angle_deg', -5.0))
    thickness    = float(sofa_cfg.get('sheet_thickness', 0.001))
    n_face       = int(sofa_cfg.get('n_face', 4))
    n_hinge      = int(sofa_cfg.get('n_hinge', 2))
    n_z          = int(sofa_cfg.get('n_z', 2))
    fd_eps       = float(sofa_cfg.get('fd_eps', 1e-5))
    young        = float(mat_cfg.get('young_modulus', 3.5e9))
    nu           = float(mat_cfg.get('poisson_ratio', 0.36))
    yield_str    = float(mat_cfg.get('yield_strength', 50e6))

    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix='hinge_opt_'))
    _counter = {'n': 0}

    def _call_sofa(arm_w: np.float64, fold_l: np.float64) -> np.float64:
        """One synchronous SOFA evaluation; returns max_von_mises_stress."""
        _counter['n'] += 1
        idx = _counter['n']
        mesh_path   = tmpdir / f'mesh_{idx}.npz'
        result_path = tmpdir / f'result_{idx}.npz'
        _save_mesh_npz(cs_static, float(arm_w), float(fold_l),
                       thickness, n_face, n_hinge, n_z, mesh_path)
        stress = _run_sofa(mesh_path, result_path,
                           rotation_deg, young, nu, yield_str, thickness)
        return np.float64(stress)

    scalar_spec = jax.ShapeDtypeStruct((), jnp.float64)

    @jax.custom_vjp
    def sofa_stress(arm_w: jax.Array, fold_l: jax.Array) -> jax.Array:
        """Max von Mises stress [Pa] from SOFA FEM at the given hinge dimensions."""
        return jax.pure_callback(_call_sofa, scalar_spec, arm_w, fold_l)

    def _fwd(arm_w, fold_l):
        stress = sofa_stress(arm_w, fold_l)
        return stress, (arm_w, fold_l)

    def _bwd(res, g):
        arm_w, fold_l = res

        def _fd(g_val: np.float64,
                aw: np.float64,
                fl: np.float64):
            """Central FD for both params; 4 SOFA calls total."""
            d_arm  = (_call_sofa(aw + fd_eps, fl) -
                      _call_sofa(aw - fd_eps, fl)) / (2.0 * fd_eps)
            d_fold = (_call_sofa(aw, fl + fd_eps) -
                      _call_sofa(aw, fl - fd_eps)) / (2.0 * fd_eps)
            return np.float64(g_val * d_arm), np.float64(g_val * d_fold)

        return jax.pure_callback(_fd, (scalar_spec, scalar_spec), g, arm_w, fold_l)

    sofa_stress.defvjp(_fwd, _bwd)
    return sofa_stress


# ── Softplus param transform ──────────────────────────────────────────────────

def _softplus_scale(init_physical: float) -> float:
    """Return s such that softplus(0) * s == init_physical."""
    sp0 = float(jax.nn.softplus(jnp.zeros(())))   # log(2) ≈ 0.6931
    return init_physical / sp0


# ── Optimization loop ─────────────────────────────────────────────────────────

def run_optimization(cfg: dict, n_epochs: int, lr: float,
                     out_dir: pathlib.Path) -> dict:
    sofa_cfg = cfg.get('sofa', {})
    arm_init  = float(sofa_cfg.get('arm_width_initial',   0.010))
    fold_init = float(sofa_cfg.get('fold_length_initial', 0.003))

    cs_static   = build_physical_cs(cfg)
    sofa_stress = make_sofa_stress_fn(cs_static, cfg)

    arm_scale  = _softplus_scale(arm_init)
    fold_scale = _softplus_scale(fold_init)

    def to_physical(lat: jax.Array):
        """(latent_arm, latent_fold) → (arm_width_m, fold_length_m)."""
        return (jax.nn.softplus(lat[0]) * arm_scale,
                jax.nn.softplus(lat[1]) * fold_scale)

    def loss_fn(lat: jax.Array) -> jax.Array:
        arm_w, fold_l = to_physical(lat)
        return sofa_stress(arm_w, fold_l)

    # latent=0 → physical = arm_init / fold_init (by construction of softplus_scale)
    params    = jnp.zeros(2, dtype=jnp.float64)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    history: dict = {
        'latent':      [],
        'arm_width':   [],
        'fold_length': [],
        'stress':      [],
    }

    print(f"\nHinge optimization: {n_epochs} epochs, lr={lr}")
    print(f"Initial: arm_width={arm_init*1e3:.2f} mm, "
          f"fold_length={fold_init*1e3:.2f} mm\n")

    for epoch in range(n_epochs):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        arm_phys, fold_phys = to_physical(params)

        history['latent'].append(np.array(params))
        history['arm_width'].append(float(arm_phys))
        history['fold_length'].append(float(fold_phys))
        history['stress'].append(float(loss))

        print(f"  epoch {epoch+1:3d}/{n_epochs}  "
              f"stress={float(loss):.4e} Pa  "
              f"arm={float(arm_phys)*1e3:.3f} mm  "
              f"fold={float(fold_phys)*1e3:.3f} mm")

    np.savez(
        str(out_dir / 'convergence.npz'),
        latent      = np.array(history['latent']),
        arm_width   = np.array(history['arm_width']),
        fold_length = np.array(history['fold_length']),
        stress      = np.array(history['stress']),
    )
    _plot_convergence(history, out_dir / 'convergence.png', arm_init, fold_init)
    return history


# ── Visualization ─────────────────────────────────────────────────────────────

def _plot_convergence(history: dict, out_path: pathlib.Path,
                      arm_init: float, fold_init: float) -> None:
    epochs = range(1, len(history['stress']) + 1)
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    axes[0].semilogy(epochs, history['stress'], 'b-o', markersize=4, linewidth=1.5)
    axes[0].set_ylabel('Max von Mises Stress [Pa]')
    axes[0].set_title('SOFA Hinge Optimization — Convergence')
    axes[0].grid(True, alpha=0.3)

    arm_mm = [v * 1e3 for v in history['arm_width']]
    axes[1].plot(epochs, arm_mm, 'r-o', markersize=4, linewidth=1.5, label='arm_width')
    axes[1].axhline(arm_init * 1e3, color='r', linestyle='--', alpha=0.4,
                    label=f'init = {arm_init*1e3:.1f} mm')
    axes[1].set_ylabel('arm_width [mm]')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fold_mm = [v * 1e3 for v in history['fold_length']]
    axes[2].plot(epochs, fold_mm, 'g-o', markersize=4, linewidth=1.5, label='fold_length')
    axes[2].axhline(fold_init * 1e3, color='g', linestyle='--', alpha=0.4,
                    label=f'init = {fold_init*1e3:.1f} mm')
    axes[2].set_ylabel('fold_length [mm]')
    axes[2].set_xlabel('Epoch')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_comparison(
    init_png: pathlib.Path, final_png: pathlib.Path,
    arm_init: float, fold_init: float,
    arm_final: float, fold_final: float,
    stress_init: float, stress_final: float,
    best_epoch: int, out_path: pathlib.Path,
) -> None:
    """Stack initial and final 3-panel SOFA figures into one comparison PNG."""
    img_i = plt.imread(str(init_png))
    img_f = plt.imread(str(final_png))

    fig, axes = plt.subplots(2, 1, figsize=(22, 17), facecolor='#FAFAFA')
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01, hspace=0.12)

    axes[0].imshow(img_i)
    axes[0].axis('off')
    axes[0].set_title(
        f"Initial  —  arm = {arm_init*1e3:.1f} mm,  "
        f"fold = {fold_init*1e3:.1f} mm,  "
        f"σ_max = {stress_init/1e6:.1f} MPa",
        fontsize=12, fontweight='bold', pad=6,
    )

    axes[1].imshow(img_f)
    axes[1].axis('off')
    axes[1].set_title(
        f"Optimized (best epoch {best_epoch})  —  arm = {arm_final*1e3:.1f} mm,  "
        f"fold = {fold_final*1e3:.1f} mm,  "
        f"σ_max = {stress_final/1e6:.1f} MPa",
        fontsize=12, fontweight='bold', pad=6,
    )

    reduction = 100.0 * (1.0 - stress_final / stress_init)
    fig.suptitle(
        f"Hinge Optimization Comparison  —  {reduction:.1f}% stress reduction",
        fontsize=14, fontweight='bold', y=0.97,
    )

    fig.savefig(str(out_path), dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
    plt.close(fig)
    print(f"Comparison → {out_path}")


def run_phase3(history: dict, cfg: dict, out_dir: pathlib.Path) -> None:
    """Generate initial vs final SOFA visualizations.

    Reads the best epoch from `history`, builds hex meshes at the initial and
    best-found hinge dimensions, runs two SOFA evaluations, and renders:
        phase3/cs_mesh_initial.npz   phase3/cs_mesh_final.npz
        phase3/sofa_initial.npz      phase3/sofa_final.npz
        phase3/sofa_initial.png      phase3/sofa_final.png
        phase3/comparison.png
    """
    sofa_cfg = cfg.get('sofa', {})
    mat_cfg  = cfg.get('material', {})

    arm_init  = float(sofa_cfg.get('arm_width_initial',   0.010))
    fold_init = float(sofa_cfg.get('fold_length_initial', 0.003))

    best_idx   = int(np.argmin(history['stress']))
    arm_final  = history['arm_width'][best_idx]
    fold_final = history['fold_length'][best_idx]

    rotation_deg = float(sofa_cfg.get('rotation_angle_deg', -5.0))
    thickness    = float(sofa_cfg.get('sheet_thickness', 0.001))
    n_face       = int(sofa_cfg.get('n_face', 4))
    n_hinge      = int(sofa_cfg.get('n_hinge', 2))
    n_z          = int(sofa_cfg.get('n_z', 2))
    young        = float(mat_cfg.get('young_modulus', 3.5e9))
    nu           = float(mat_cfg.get('poisson_ratio', 0.36))
    yield_str    = float(mat_cfg.get('yield_strength', 50e6))

    p3_dir = out_dir / 'phase3'
    p3_dir.mkdir(exist_ok=True)

    cs_static = build_physical_cs(cfg)

    print('\nPhase 3: running SOFA on initial and final designs ...')

    # Initial design
    mesh_init = p3_dir / 'cs_mesh_initial.npz'
    out_init  = p3_dir / 'sofa_initial.npz'
    _save_mesh_npz(cs_static, arm_init, fold_init,
                   thickness, n_face, n_hinge, n_z, mesh_init)
    _run_sofa_full(mesh_init, out_init, arm_init, fold_init,
                   rotation_deg, young, nu, yield_str, thickness)
    print(f'  initial SOFA → {out_init}')

    # Best/final design
    mesh_final = p3_dir / 'cs_mesh_final.npz'
    out_final  = p3_dir / 'sofa_final.npz'
    _save_mesh_npz(cs_static, arm_final, fold_final,
                   thickness, n_face, n_hinge, n_z, mesh_final)
    _run_sofa_full(mesh_final, out_final, arm_final, fold_final,
                   rotation_deg, young, nu, yield_str, thickness)
    print(f'  final   SOFA → {out_final}')

    # Load visualize.py from sofa/ (no SOFA runtime — matplotlib only)
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location('sofa_visualize', REPO / 'sofa' / 'visualize.py')
    viz   = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(viz)

    data_init  = viz.load_npz(str(out_init))
    data_final = viz.load_npz(str(out_final))

    png_init  = p3_dir / 'sofa_initial.png'
    png_final = p3_dir / 'sofa_final.png'
    viz.make_figure(data_init,  str(png_init))
    viz.make_figure(data_final, str(png_final))

    stress_init  = data_init['qois']['max_von_mises_stress']
    stress_final = data_final['qois']['max_von_mises_stress']

    _plot_comparison(
        png_init, png_final,
        arm_init,  fold_init,
        arm_final, fold_final,
        stress_init, stress_final,
        best_idx + 1,
        p3_dir / 'comparison.png',
    )

    print(f'\nPhase 3 complete → {p3_dir}/')


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Optimize SOFA hinge dimensions to minimize von Mises stress.')
    p.add_argument('--config',      default='data/configs/sofa/hinge_opt_2face.yaml',
                   help='Path to YAML config.')
    p.add_argument('--n-epochs',    type=int,   default=None,
                   help='Override optimization.n_epochs from config.')
    p.add_argument('--lr',          type=float, default=None,
                   help='Override optimization.learning_rate from config.')
    p.add_argument('--phase3-only', default=None, metavar='RUN_DIR',
                   help='Skip optimization; run Phase 3 on an existing run directory '
                        '(must contain config.yaml + convergence.npz).')
    args = p.parse_args()

    if args.phase3_only:
        run_dir = pathlib.Path(args.phase3_only)
        with open(run_dir / 'config.yaml') as f:
            cfg = yaml.safe_load(f)
        conv = np.load(str(run_dir / 'convergence.npz'))
        history = {
            'arm_width':   list(conv['arm_width']),
            'fold_length': list(conv['fold_length']),
            'stress':      list(conv['stress']),
        }
        run_phase3(history, cfg, run_dir)
        return

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    opt_cfg  = cfg.get('optimization', {})
    n_epochs = args.n_epochs if args.n_epochs is not None else int(opt_cfg.get('n_epochs', 30))
    lr       = args.lr       if args.lr       is not None else float(opt_cfg.get('learning_rate', 0.05))

    config_name = pathlib.Path(args.config).stem
    timestamp   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir     = OUTPUTS_DIR / f'{timestamp}_{config_name}'
    out_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(args.config, out_dir / 'config.yaml')

    history = run_optimization(cfg, n_epochs, lr, out_dir)

    best_idx    = int(np.argmin(history['stress']))
    best_stress = history['stress'][best_idx]
    best_arm    = history['arm_width'][best_idx]
    best_fold   = history['fold_length'][best_idx]

    print(f'\nBest (epoch {best_idx + 1}): '
          f'stress={best_stress:.4e} Pa  '
          f'arm={best_arm*1e3:.3f} mm  '
          f'fold={best_fold*1e3:.3f} mm')

    run_phase3(history, cfg, out_dir)
    print(f'Results saved → {out_dir}')


if __name__ == '__main__':
    main()
