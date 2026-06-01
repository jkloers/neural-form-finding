"""
train.py — Neural Form-Finding Training Pipeline.

Two modes:

  Single-file (legacy):
    python train.py --config-dir gnn/benchmark --config-name exp1_rot_clamp0_viz

  Architecture + problem suite (new):
    python train.py --arch architectures/egnn_base --suite problems/suite_2x2_rdqk
    python train.py --arch architectures/egnn_base --suite problems/suite_2x2_rdqk \
                    --problem-ids p001,p002,p010
"""

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import sys
sys.path.append(os.path.abspath('src'))

import jax
jax.config.update("jax_enable_x64", True)

import csv
import yaml
import jax.numpy as jnp
import numpy as np
import argparse
from types import SimpleNamespace
import datetime
import shutil

from src.topology.builder import build_tessellation
from src.problem.conditions import configure_tessellation
from src.problem.config import (
    load_and_parse_config,
    load_arch_config,
    load_problem_suite,
    merge_arch_problem,
    _parse_full_raw,
)
from src.jax_backend.state import CentroidalState
from src.jax_backend.pipeline import forward_pipeline
from src.jax_backend.training.trainer import train_pipeline
from src.utils.pipeline_viz import visualize_pipeline_results, plot_loss_history


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_initial_state(config):
    """Build tessellation and CentroidalState from a parsed config."""
    topo = config.topology
    topo_obj = SimpleNamespace(**topo)

    tessellation = build_tessellation(
        topo.get('pattern'),
        topo.get('width', 5),
        topo.get('height', 5),
    )

    requested_area = topo.get('total_area')
    if requested_area:
        current_area = tessellation.compute_total_area()
        scale = np.sqrt(requested_area / current_area)
        tessellation.update_vertices(tessellation.vertices * scale)

    configure_tessellation(tessellation, topo_obj)

    if config.mapping.type.startswith('gnn_'):
        target_center = np.array(
            getattr(config.target, 'center', [0.0, 0.0]), dtype=float)
        tess_centroid = np.mean(tessellation.get_face_centroids(), axis=0)
        tessellation.update_vertices(
            tessellation.vertices - tess_centroid + target_center)

    return CentroidalState.from_tessellation(tessellation, target_cfg=config.target), tessellation


def _init_gnn_params(config, initial_state):
    """Initialise GNN parameters from architecture config."""
    from jax_backend.gnn.graph_builder import build_static_features

    map_type = config.mapping.type
    gnn_cfg  = config.mapping.params if isinstance(config.mapping.params, dict) else {}
    hidden_dim  = int(gnn_cfg.get('hidden_dim', 16))
    num_layers  = int(gnn_cfg.get('num_layers', 2))
    seed        = int(gnn_cfg.get('seed', 0))
    key         = jax.random.PRNGKey(seed)

    static_features = build_static_features(initial_state, map_type)
    node_feat_dim   = static_features['node_feat_dim']

    if map_type == 'gnn_egnn':
        from jax_backend.gnn.egnn import init_egnn
        params = init_egnn(key, node_feat_dim, hidden_dim, num_layers)
    elif map_type == 'gnn_mpnn':
        from jax_backend.gnn.mpnn import init_mpnn
        params = init_mpnn(key, node_feat_dim, hidden_dim, num_layers)
    else:
        from jax_backend.gnn.dummy_gnn import init_dummy_gnn
        params = init_dummy_gnn(key, node_feat_dim, hidden_dim)

    # Expose num_layers so apply_egnn/apply_mpnn receive it as an explicit static arg.
    return params, {**static_features, 'num_layers': num_layers}


def _init_map_params(config, initial_state):
    """Initialise mapping parameters (GNN or analytical)."""
    if config.mapping.type.startswith('gnn_'):
        params, static_features = _init_gnn_params(config, initial_state)
        return params, static_features
    else:
        raw = config.mapping.params
        params = raw if isinstance(raw, dict) else {}
        if config.mapping.learn_global_scale and 'log_scale' not in params:
            params = {**params, 'log_scale': jnp.array(0.0)}
        return params, None


def _run_one_problem(config, problem_label, run_dir):
    """Train and evaluate one problem. Returns summary metrics dict."""
    print(f"\n{'═'*60}")
    print(f"  PROBLEM: {problem_label}")
    print(f"{'═'*60}")

    initial_state, tessellation = _build_initial_state(config)
    initial_area = tessellation.compute_total_area()

    map_params, static_features = _init_map_params(config, initial_state)

    # load_specs: raw list of load dicts from the config.
    # Typed loads ('type' key present) are handled by force_types.py in the pipeline.
    # Legacy loads (no 'type') are already applied to the tessellation by conditions.py.
    load_specs = config.topology.get('loads', []) or []

    _on_cpu = jax.default_backend() == 'cpu'
    _use_jit = _on_cpu or not config.mapping.type.startswith('gnn_')

    optimized_params, history_loss = train_pipeline(
        map_params,
        initial_state,
        config.target,
        config.validity,
        config.physics,
        config.training,
        map_type=config.mapping.type,
        use_shirley_chiu=config.mapping.use_shirley_chiu,
        strict_boundary_fit=config.mapping.strict_boundary_fit,
        learn_global_scale=config.mapping.learn_global_scale,
        use_jit=_use_jit,
        load_specs=load_specs,
    )

    # Sub-directory for this problem
    prob_dir = None
    if run_dir and config.visualization.save_outputs:
        prob_dir = os.path.join(run_dir, problem_label)
        os.makedirs(prob_dir, exist_ok=True)

    _any_viz = (config.visualization.save_outputs or config.visualization.show_plots
                or config.visualization.stage0 or config.visualization.stage1
                or config.visualization.stage2 or config.visualization.energy_plot)

    if _any_viz:
        plot_loss_history(history_loss, config, run_dir=prob_dir)

        result = forward_pipeline(
            initial_state,
            config.target,
            config.validity,
            config.physics,
            map_type=config.mapping.type,
            map_params=optimized_params,
            use_shirley_chiu=config.mapping.use_shirley_chiu,
            strict_boundary_fit=config.mapping.strict_boundary_fit,
            static_features=static_features,
            load_specs=load_specs,
        )

        target_params = {
            'type': config.target.type,
            'center': config.target.center,
            'radius': config.target.radius,
        }
        visualize_pipeline_results(
            result, tessellation, config, target_params,
            problem_label + "_trained", run_dir=prob_dir,
            load_specs=load_specs)

    final_metrics = history_loss[-1] if history_loss else {}
    return {
        'final_loss':    float(final_metrics.get('total', float('nan'))),
        'final_chamfer': float(final_metrics.get('chamfer_total', float('nan'))),
        'final_energy':  float(final_metrics.get('energy', float('nan'))),
        'initial_area':  float(initial_area),
    }


# ── Single-problem mode (legacy) ──────────────────────────────────────────────

def run_single(args):
    config_path = f"data/configs/{args.config_dir}/{args.config_name}.yaml"
    config = load_and_parse_config(config_path)
    print(f"Loaded config: {config_path}")

    mode_str = ("LEARN SCALE" if config.mapping.learn_global_scale
                else "FIXED MATERIAL")
    print(f"\n{'═'*60}\n PIPELINE MODE: {mode_str}\n{'═'*60}")

    run_dir = None
    if config.visualization.save_outputs:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = f"data/outputs/runs/run_{timestamp}_{args.config_name}"
        os.makedirs(run_dir, exist_ok=True)
        shutil.copy(config_path, os.path.join(run_dir, "config.yaml"))
        print(f"Output directory: {run_dir}")

    metrics = _run_one_problem(config, args.config_name, run_dir)
    print(f"\nFinal — loss: {metrics['final_loss']:.4e} | "
          f"chamfer: {metrics['final_chamfer']:.4e}")
    print("\nTraining complete.")


# ── Multi-problem mode (arch + suite) ────────────────────────────────────────

def run_suite(args):
    arch_path  = f"data/configs/{args.arch}.yaml"
    suite_path = f"data/configs/{args.suite}.yaml"

    arch_raw = load_arch_config(arch_path)
    all_problems = load_problem_suite(suite_path)

    # Filter by --problem-ids if provided
    if args.problem_ids:
        ids = set(args.problem_ids.split(','))
        problems = [p for p in all_problems if p['id'] in ids]
        if not problems:
            raise ValueError(f"No problems matched ids: {ids}")
    else:
        problems = all_problems

    arch_name = os.path.basename(args.arch)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"data/outputs/runs/run_{timestamp}_{arch_name}"
    os.makedirs(run_dir, exist_ok=True)

    # Save arch config and suite for reproducibility
    shutil.copy(arch_path,  os.path.join(run_dir, "arch_config.yaml"))
    shutil.copy(suite_path, os.path.join(run_dir, "problem_suite.yaml"))

    print(f"\n{'═'*60}")
    print(f"  ARCH  : {arch_path}")
    print(f"  SUITE : {suite_path}")
    print(f"  PROBLEMS: {len(problems)}")
    print(f"  OUTPUT: {run_dir}")
    print(f"{'═'*60}")

    summary_rows = []

    for problem in problems:
        pid   = problem['id']
        pname = problem['name']
        diff  = problem.get('difficulty', '?')
        label = f"{pid}_{pname}"

        merged = merge_arch_problem(arch_raw, problem)
        # config_dir must point to arch location so pattern lookup works
        config = _parse_full_raw(merged, os.path.dirname(arch_path))

        # Save the merged config for this problem
        prob_dir = os.path.join(run_dir, label)
        os.makedirs(prob_dir, exist_ok=True)
        with open(os.path.join(prob_dir, "merged_config.yaml"), 'w') as f:
            yaml.dump(merged, f, default_flow_style=False, allow_unicode=True)

        try:
            metrics = _run_one_problem(config, label, run_dir)
            metrics.update({'id': pid, 'name': pname, 'difficulty': diff})
            summary_rows.append(metrics)
            print(f"  ✓ {label:40s} "
                  f"loss={metrics['final_loss']:.3e}  "
                  f"chamfer={metrics['final_chamfer']:.3e}  "
                  f"diff={diff}")
        except Exception as exc:
            print(f"  ✗ {label:40s} FAILED: {exc}")
            summary_rows.append({
                'id': pid, 'name': pname, 'difficulty': diff,
                'final_loss': float('nan'), 'final_chamfer': float('nan'),
                'final_energy': float('nan'), 'initial_area': float('nan'),
            })

    # Write summary CSV
    summary_path = os.path.join(run_dir, "summary.csv")
    fieldnames = ['id', 'name', 'difficulty',
                  'final_loss', 'final_chamfer', 'final_energy', 'initial_area']
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(summary_rows)

    # Print summary table
    print(f"\n{'─'*70}")
    print(f"  {'ID':<8} {'Name':<30} {'Diff':>5}  {'Chamfer':>10}  {'Loss':>10}")
    print(f"{'─'*70}")
    for r in sorted(summary_rows, key=lambda x: x['difficulty']):
        print(f"  {r['id']:<8} {r['name']:<30} {r['difficulty']:>5}  "
              f"{r['final_chamfer']:>10.3e}  {r['final_loss']:>10.3e}")
    print(f"{'─'*70}")
    print(f"\nSummary saved to: {summary_path}")
    print(f"Run directory   : {run_dir}")
    print("\nTraining complete.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Form-Finding Training.")

    # Legacy single-file mode
    parser.add_argument("--config-dir",  type=str, default="poster/complex",
                        help="Subdirectory under data/configs/ (legacy mode)")
    parser.add_argument("--config-name", type=str, default=None,
                        help="Config filename without .yaml (legacy mode)")

    # Arch + suite mode
    parser.add_argument("--arch", type=str, default=None,
                        help="Architecture config path relative to data/configs/ "
                             "(e.g. architectures/egnn_base)")
    parser.add_argument("--suite", type=str, default=None,
                        help="Problem suite path relative to data/configs/ "
                             "(e.g. problems/suite_2x2_rdqk)")
    parser.add_argument("--problem-ids", type=str, default=None,
                        help="Comma-separated problem IDs to run "
                             "(e.g. p001,p005,p010). Omit to run all.")

    args = parser.parse_args()

    if args.arch and args.suite:
        run_suite(args)
    elif args.config_name:
        run_single(args)
    else:
        parser.error("Provide either --arch + --suite, or --config-name.")
