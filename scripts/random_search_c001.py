"""
random_search_c001.py — Random hyperparameter search on c001 (hub_clamp_moment_1hop).

Architecture: MPNN + alternating projection + physics solver, 2×2 RDQK_D.
Goal: maximise void closure while keeping chamfer reasonable.

Usage:
    python random_search_c001.py [--budget-hours 3] [--max-trials 300] [--seed 7]

Outputs to data/outputs/search_c001_run3_<timestamp>/
    results.csv     — all trials with metrics
    trial_NNN/      — config.yaml + stage0/1/2 + loss plot (when chamfer < 0.04)

═══ Fixed from Runs 1 & 2 ═════════════════════════════════════════════════════
  inner_depth = 1           — clear winner in both runs
  coverage_w  = 0.5         — key unlock: allows inward contraction on closing
  strategy    = 'both'      — void_closure + closure_delta owns the Pareto front
  num_layers  ∈ {2, 3}      — L=2 and L=3 both appear in good configs
  hidden_dim  ∈ {16, 32}    — hidden=8 noisier, dropped
  deformation = 0.0, openness = 0.0, contact_loss = 0.0
  learning_rate = 0.005, optimizer = adam, lr_schedule = cosine, num_epochs = 500

═══ Key fixes in Run 3 ════════════════════════════════════════════════════════
  chamfer_w  ≥ 2000         — HARD FLOOR. Without it, MPNN distorts tile shapes
                              to collapse void area → degenerate face intersections
                              (root cause of t201 failure in Run 2: chamfer_w=461)
  closure_delta ≤ 100       — HARD CAP. Large cd_w drives total loss negative
                              (91/234 Run 2 trials had negative total loss), which
                              causes the optimizer to sacrifice shape for reward
  k_contact ∈ {5000,10000,  — NEW. Test whether stronger face repulsion prevents
               20000}          physics-level intersections at F=15
  force ∈ {10, 12, 15}      — Remove 8 (too little closure), remove 20 (instability)
  n_proj_iters ∈ {50, 100}  — Drop 20; needs adequate projection for stability
  All 3 stages visualised   — Stage 0 shows MPNN map locality; Stage 1 shows
  for good trials             projection correction; Stage 2 shows equilibrium
"""

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
jax.config.update("jax_enable_x64", True)

import csv
import math
import time
import yaml
import argparse
import datetime
import numpy as np

from nff.config.experiment import _parse_full_raw
from nff.scripts.train import _build_initial_state, _init_map_params
from nff.training.trainer import train_pipeline
from nff.stages.pipeline import forward_pipeline


# ── Problem definition ────────────────────────────────────────────────────────

C001_MATERIAL = {
    'k_stretch': 1000.0,
    'k_shear':   1000.0,
    'k_rot':     0.5,
    'density':   1.0,
}

def _c001_physics(k_contact, num_load_steps=5):
    return {
        'use_contact':        True,
        'linearized_strains': True,
        'updated_lagrangian': False,
        'k_contact':          float(k_contact),
        'min_angle':          1.0,
        'cutoff_angle':       5.0,
        'incremental':        True,
        'num_load_steps':     int(num_load_steps),
        'solver_maxiter':     1000,
        'solver_tol':         1.0e-5,
    }


# ── Parameter sampling ────────────────────────────────────────────────────────

def sample_trial(rng, trial_idx):
    """Sample one random hyperparameter configuration (Run 3 search space)."""
    # ── Fixed from Runs 1 & 2 ────────────────────────────────────────────────
    inner_depth = 1
    coverage_w  = 0.5
    # -------------------------------------------------------------------------

    hidden_dim   = int(rng.choice([16, 32]))
    num_layers   = int(rng.choice([2, 3]))
    n_proj_iters = int(rng.choice([50, 100]))
    k_contact    = float(rng.choice([5000.0, 10000.0, 20000.0]))

    # Log-uniform continuous (chamfer_w has a hard floor at 2000)
    chamfer       = float(np.exp(rng.uniform(np.log(2000),  np.log(8000))))
    hinge_gap     = float(np.exp(rng.uniform(np.log(80),    np.log(400))))
    void_closure  = float(np.exp(rng.uniform(np.log(200),   np.log(1000))))
    # closure_delta hard cap at 100 — prevents loss from going negative
    closure_delta = float(np.exp(rng.uniform(np.log(10),    np.log(100))))

    stretching  = float(rng.choice([0.5, 1.0]))
    shearing    = float(rng.choice([5.0, 10.0]))
    force_value = float(rng.choice([10.0, 12.0, 15.0]))

    return {
        'trial_idx':      trial_idx,
        'hidden_dim':     hidden_dim,
        'num_layers':     num_layers,
        'inner_depth':    inner_depth,
        'n_proj_iters':   n_proj_iters,
        'coverage_w':     coverage_w,
        'k_contact':      k_contact,
        'chamfer':        round(chamfer,       1),
        'hinge_gap':      round(hinge_gap,     1),
        'void_closure':   round(void_closure,  1),
        'closure_delta':  round(closure_delta, 1),
        'stretching':     stretching,
        'shearing':       shearing,
        'force_value':    force_value,
    }


# ── Config construction ───────────────────────────────────────────────────────

def build_merged_dict(p):
    """Build a fully-merged config dict from sampled params + c001 problem."""
    return {
        'tessellation': {
            'width':      2,
            'height':     2,
            'pattern':    'unit_RDQK_D',
            'total_area': 3.0,
        },
        'target': {
            'type':   'circle',
            'center': [0.0, 0.0],
            'radius': 1.0,
        },
        'mapping': {
            'map_type':            'gnn_mpnn',
            'domain_restriction':  1.0,
            'use_shirley_chiu':    False,
            'strict_boundary_fit': False,
            'learn_global_scale':  False,
            'map_params': {
                'hidden_dim':  p['hidden_dim'],
                'num_layers':  p['num_layers'],
                'inner_depth': p['inner_depth'],
                'seed':        p['trial_idx'],
            },
        },
        'optimization_weights': {
            'validity_method': 'alternating_projection',
            'n_proj_iters':    p['n_proj_iters'],
        },
        'training': {
            'num_epochs':          500,
            'learning_rate':       0.005,
            'optimizer':           'adam',
            'lr_schedule':         'cosine',
            'geometric_loss_type': 'boundary_vertices',
            'grad_clip':           1.0,
        },
        'loss_weights': {
            'chamfer':        p['chamfer'],
            'coverage':       p['coverage_w'],   # 0.5 — allows inward contraction
            'material_area':  0.0,
            'stretching':     p['stretching'],
            'shearing':       p['shearing'],
            'bending':        0.0,
            'contact':        0.0,
            'regularization': 0.0001,
            'hinge_gap':      p['hinge_gap'],
            'openness':       0.0,
            'deformation':    0.0,
            'void_closure':   p['void_closure'],
            'closure_delta':  p['closure_delta'],  # capped at 100
        },
        'boundary_conditions': {
            'clamped_faces': [9],
        },
        'loads': [
            {
                'type':  'global_frame',
                'face':  12,
                'dof':   2,
                'value': p['force_value'],
            }
        ],
        'physics':  _c001_physics(p['k_contact']),
        'material': C001_MATERIAL,
        # All three stages saved for qualifying trials (chamfer < 0.04)
        'visualization': {
            'stage0':                True,
            'stage1':                True,
            'stage2':                True,
            'energy_plot':           True,
            'animation':             False,
            'show_plots':            False,
            'save_outputs':          True,
            'show_hinges':           True,
            'show_hinge_indices':    False,
            'show_external_forces':  True,
            'show_kinematic_blocks': True,
        },
    }


# ── Metrics extraction ────────────────────────────────────────────────────────

def extract_metrics(history_loss, p):
    """Extract final metrics and derive raw void area from weighted loss terms."""
    final = history_loss[-1] if history_loss else {}

    chamfer     = float(final.get('chamfer_total', float('nan')))
    total_loss  = float(final.get('total',         float('nan')))
    hinge_gap_l = float(final.get('hinge_gap',     0.0))
    energy      = float(final.get('energy',        float('nan')))
    vc_loss     = float(final.get('void_closure',  0.0))
    cd_loss     = float(final.get('closure_delta', 0.0))

    # void_stage2 = exp(void_closure_loss / w_vc) - 1
    void_stage2_est = float('nan')
    if p['void_closure'] > 0.0 and math.isfinite(vc_loss):
        try:
            void_stage2_est = math.exp(vc_loss / p['void_closure']) - 1.0
        except OverflowError:
            void_stage2_est = float('inf')

    # delta = exp(-closure_delta_loss / w_cd) - 1
    delta_est = float('nan')
    if p['closure_delta'] > 0.0 and math.isfinite(cd_loss):
        try:
            delta_est = math.exp(-cd_loss / p['closure_delta']) - 1.0
        except OverflowError:
            delta_est = float('inf')

    return {
        'final_chamfer':    chamfer,
        'final_total_loss': total_loss,
        'final_hinge_gap':  hinge_gap_l,
        'final_energy':     energy,
        'void_stage2_est':  void_stage2_est,
        'void_delta_est':   delta_est,
        'vc_loss_raw':      vc_loss,
        'cd_loss_raw':      cd_loss,
    }


# ── Single trial ──────────────────────────────────────────────────────────────

def run_trial(p, trial_dir):
    """Run one training trial. Returns metrics dict."""
    merged = build_merged_dict(p)

    os.makedirs(trial_dir, exist_ok=True)
    with open(os.path.join(trial_dir, 'config.yaml'), 'w') as f:
        yaml.dump(merged, f, default_flow_style=False, allow_unicode=True)

    config = _parse_full_raw(merged, 'data/configs/architectures')
    initial_state, tessellation = _build_initial_state(config)
    map_params, static_features = _init_map_params(config, initial_state)
    load_specs = config.topology.get('loads', []) or []

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
        use_jit=True,
        load_specs=load_specs,
        static_features=static_features,
    )

    metrics = extract_metrics(history_loss, p)

    # Save all stages + loss plot for good trials (chamfer < 0.04)
    if math.isfinite(metrics['final_chamfer']) and metrics['final_chamfer'] < 0.04:
        try:
            from nff.utils.pipeline_viz import visualize_pipeline_results, plot_loss_history
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
                'type':   config.target.type,
                'center': config.target.center,
                'radius': config.target.radius,
            }
            plot_loss_history(history_loss, config, run_dir=trial_dir)
            visualize_pipeline_results(
                result, tessellation, config, target_params,
                f"trial_{p['trial_idx']:03d}_c001", run_dir=trial_dir,
                load_specs=load_specs)
        except Exception as viz_exc:
            print(f"  [viz skipped: {viz_exc}]")

    return metrics


# ── Leaderboard ───────────────────────────────────────────────────────────────

def print_leaderboard(rows, top_n=8):
    """Print top trials sorted by void_stage2 (primary) then chamfer."""
    def sort_key(r):
        vs2 = r.get('void_stage2_est', float('nan'))
        chf = r.get('final_chamfer',   float('nan'))
        return (vs2 if math.isfinite(vs2) else 999.0,
                chf if math.isfinite(chf) else 999.0)

    valid = [r for r in rows if math.isfinite(r.get('final_chamfer', float('nan')))]
    valid.sort(key=sort_key)

    print(f"\n{'─'*108}")
    print(f"  {'#':>4}  {'vs2':>8}  {'chamfer':>9}  {'energy':>8}  "
          f"{'h':>3}  {'L':>2}  {'proj':>5}  {'kc':>7}  "
          f"{'chf_w':>6}  {'hgap':>5}  {'vc_w':>6}  {'cd_w':>5}  {'F':>5}")
    print(f"{'─'*108}")
    for r in valid[:top_n]:
        vs2   = r.get('void_stage2_est', float('nan'))
        vs2_s = f"{vs2:.3f}" if math.isfinite(vs2) else '  n/a'
        eng   = r.get('final_energy', float('nan'))
        eng_s = f"{eng:.2e}" if math.isfinite(eng) else '  n/a'
        print(f"  {r['trial_idx']:>4}  {vs2_s:>8}  {r['final_chamfer']:>9.4f}  {eng_s:>8}  "
              f"{r['hidden_dim']:>3}  {r['num_layers']:>2}  {r['n_proj_iters']:>5}  "
              f"{r['k_contact']:>7.0f}  "
              f"{r['chamfer']:>6.0f}  {r['hinge_gap']:>5.0f}  "
              f"{r['void_closure']:>6.0f}  {r['closure_delta']:>5.0f}  {r['force_value']:>5.0f}")
    print(f"{'─'*108}\n")


# ── CSV helpers ───────────────────────────────────────────────────────────────

CSV_FIELDS = [
    'trial_idx',
    'hidden_dim', 'num_layers', 'inner_depth', 'n_proj_iters', 'coverage_w', 'k_contact',
    'chamfer', 'hinge_gap', 'void_closure', 'closure_delta',
    'stretching', 'shearing', 'force_value',
    'final_chamfer', 'final_total_loss', 'final_hinge_gap',
    'final_energy', 'void_stage2_est', 'void_delta_est',
    'vc_loss_raw', 'cd_loss_raw', 'trial_duration_s', 'status',
]


def append_csv(csv_path, row):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ── Main search loop ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Random search on c001 (Run 3).")
    parser.add_argument('--budget-hours', type=float, default=3.0)
    parser.add_argument('--max-trials',   type=int,   default=300)
    parser.add_argument('--seed',         type=int,   default=7)
    args = parser.parse_args()

    rng      = np.random.default_rng(args.seed)
    budget_s = args.budget_hours * 3600

    timestamp  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    search_dir = f"data/outputs/search_c001_run3_{timestamp}"
    os.makedirs(search_dir, exist_ok=True)
    csv_path = os.path.join(search_dir, 'results.csv')

    print(f"\n{'═'*72}")
    print(f"  Random Search Run 3 — c001, MPNN + alt-proj")
    print(f"  Fixed: d=1, cov=0.5, strategy=both, L∈{{2,3}}, h∈{{16,32}}")
    print(f"  Key fix: chamfer_w≥2000 (floor), cd_w≤100 (cap), k_contact varied")
    print(f"  Budget: {args.budget_hours}h  |  Max trials: {args.max_trials}")
    print(f"  Output: {search_dir}")
    print(f"{'═'*72}\n")

    all_rows = []
    t_start  = time.time()

    for trial_idx in range(args.max_trials):
        elapsed = time.time() - t_start
        if elapsed >= budget_s:
            print(f"\nTime budget exhausted after {elapsed/3600:.2f}h.")
            break

        p         = sample_trial(rng, trial_idx)
        trial_dir = os.path.join(search_dir, f"trial_{trial_idx:03d}")

        print(f"\n{'─'*72}")
        print(f"  Trial {trial_idx:03d}  elapsed={elapsed/60:.1f}min  "
              f"remaining={(budget_s-elapsed)/60:.0f}min")
        print(f"  arch: h={p['hidden_dim']} L={p['num_layers']} d=1  "
              f"proj={p['n_proj_iters']}  kc={p['k_contact']:.0f}")
        print(f"  loss: chamfer={p['chamfer']:.0f}  hgap={p['hinge_gap']:.0f}  "
              f"vc={p['void_closure']:.0f}  cd={p['closure_delta']:.0f}  "
              f"stretch={p['stretching']}  shear={p['shearing']}")
        print(f"  F={p['force_value']}  seed={trial_idx}")

        t0     = time.time()
        status = 'ok'
        metrics = {}
        try:
            metrics = run_trial(p, trial_dir)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            status = f'error: {str(exc)[:80]}'
            metrics = {k: float('nan') for k in
                       ['final_chamfer', 'final_total_loss', 'final_hinge_gap',
                        'final_energy', 'void_stage2_est', 'void_delta_est',
                        'vc_loss_raw', 'cd_loss_raw']}

        duration = time.time() - t0
        row = {**p, **metrics, 'trial_duration_s': round(duration, 1), 'status': status}
        all_rows.append(row)
        append_csv(csv_path, row)

        if status == 'ok':
            vs2   = metrics.get('void_stage2_est', float('nan'))
            chf   = metrics['final_chamfer']
            eng   = metrics.get('final_energy', float('nan'))
            tot   = metrics.get('final_total_loss', float('nan'))
            vs2_s = f"{vs2:.4f}" if math.isfinite(vs2) else 'n/a'
            eng_s = f"{eng:.2e}" if math.isfinite(eng) else 'n/a'
            print(f"  → vs2={vs2_s}  chamfer={chf:.4f}  "
                  f"energy={eng_s}  total={tot:.2e}  t={duration/60:.1f}min")

        if (trial_idx + 1) % 5 == 0 or trial_idx == 0:
            print_leaderboard(all_rows)

    print("\n" + "═"*72)
    print("  FINAL LEADERBOARD — sorted by void_s2 (primary), chamfer (secondary)")
    print_leaderboard(all_rows, top_n=15)
    print(f"  Results: {csv_path}")
    print(f"  Total trials: {len(all_rows)}")
    print("═"*72)


if __name__ == '__main__':
    main()
