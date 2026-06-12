#!/usr/bin/env python3
"""
scripts/param_search.py — overnight random search over hinge init, loss weights, mesh.

Samples random configs (initialization state, loss weights, mesh refinement, learning
rate), runs the fatigue optimizer for each against the Tesseract oracle, and records
the best cycles-to-failure N_f / strain / area. The search is:
  - TIME-BUDGETED  (stops after --budget-hours; default 4.5 h)
  - ROBUST         (per-trial try/except; saves results.json after EVERY trial)
  - SELF-HEALING   (restarts the oracle container if a trial wedges/hangs it)

Run at whatever the base config specifies (currently TPU @ 45°). Needs the oracle
that supports mesh_refine (≥ 0.7.0).

Usage:
    python scripts/param_search.py [--budget-hours 4.5] [--epochs 15]
        [--container nff_sofa_v070] [--tesseract-url http://localhost:8000] [--seed 0]
"""
import sys
import time
import json
import copy
import argparse
import pathlib
import datetime
import subprocess

import numpy as np
import yaml
import requests

REPO = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
from sofa.hinge_optimizer import run_optimization

BASE_CFG = REPO / 'data' / 'configs' / 'sofa' / 'hinge_opt_2face.yaml'
OUT = REPO / 'data' / 'outputs' / 'hinge_search'


def _sample(rng):
    """One random trial config (the search space)."""
    return {
        'gap_initial':           float(rng.uniform(0.003, 0.009)),    # 3–9 mm
        'reach_initial':         float(rng.uniform(0.003, 0.008)),    # 3–8 mm
        'initial_concave_bow_m': float(rng.choice([0.0, 0.001, 0.0015, 0.0025, 0.004])),
        'mesh_refine':           float(rng.choice([1.0, 1.5, 2.0])),
        'w_fatigue':             float(rng.uniform(3.0, 12.0)),
        'w_mat':                 float(rng.uniform(0.5, 4.0)),
        'w_gap':                 float(rng.choice([0.1, 0.25, 0.5, 1.0])),
        'learning_rate':         float(rng.choice([0.0003, 0.0005, 0.001])),
    }


def _apply(base, s):
    cfg = copy.deepcopy(base)
    cfg.setdefault('sofa', {}); cfg.setdefault('loss', {})
    cfg['sofa']['gap_initial']           = s['gap_initial']
    cfg['sofa']['reach_initial']         = s['reach_initial']
    cfg['sofa']['initial_concave_bow_m'] = s['initial_concave_bow_m']
    cfg['sofa']['mesh_refine']           = s['mesh_refine']
    cfg['loss']['w_fatigue']             = s['w_fatigue']
    cfg['loss']['w_mat']                 = s['w_mat']
    cfg['loss']['w_gap']                 = s['w_gap']
    return cfg


def _healthy(url):
    try:
        return requests.get(f'{url}/health', timeout=10).ok
    except Exception:
        return False


def _restart(container, url):
    print(f'  [restarting oracle container {container}]', flush=True)
    subprocess.run(['docker', 'restart', container], capture_output=True)
    for _ in range(60):
        if _healthy(url):
            return True
        time.sleep(2)
    print('  [WARNING: oracle still unhealthy after restart]', flush=True)
    return False


def main():
    ap = argparse.ArgumentParser(description='Overnight hinge hyperparameter search.')
    ap.add_argument('--budget-hours', type=float, default=4.5)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--container', default='nff_sofa_v070')
    ap.add_argument('--tesseract-url', default='http://localhost:8000')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    base = yaml.safe_load(open(BASE_CFG))
    rng = np.random.default_rng(args.seed)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = OUT / f'search_{ts}'; out_dir.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(base, open(out_dir / 'base_config.yaml', 'w'))

    angle = base.get('sofa', {}).get('rotation_angle_deg', '?')
    E = base.get('material', {}).get('young_modulus', '?')
    print(f'Param search → {out_dir.name}  (budget {args.budget_hours}h, '
          f'{args.epochs} ep/trial)\n  base: E={E} @ {angle}°', flush=True)

    results, t0, budget = [], time.time(), args.budget_hours * 3600.0
    i = 0
    while time.time() - t0 < budget:
        i += 1
        s = _sample(rng)
        trial_dir = out_dir / f'trial_{i:03d}'; trial_dir.mkdir(exist_ok=True)
        el = (time.time() - t0) / 3600.0
        print(f'\n=== trial {i}  (t={el:.2f}h / {args.budget_hours}h) === '
              + '  '.join(f'{k}={v:.4g}' for k, v in s.items()), flush=True)
        rec = {'trial': i, **s}
        try:
            if not _healthy(args.tesseract_url):
                _restart(args.container, args.tesseract_url)
            hist = run_optimization(_apply(base, s), args.epochs,
                                    s['learning_rate'], args.tesseract_url, trial_dir,
                                    capture_fields=False)
            if hist['total_loss']:
                b = int(np.argmin(hist['total_loss']))
                rec.update({
                    'best_epoch':  b + 1,
                    'Nf':          float(hist['cycles_Nf'][b]),
                    'eps_max':     float(hist['max_strain'][b]),
                    'eps_plastic': float(hist['plastic_strain'][b]),
                    'area_mm2':    float(hist['hinge_area'][b]) * 1e6,
                    'sigma_MPa':   float(hist['max_vm_rot'][b]) / 1e6,
                    'status':      'ok',
                })
                print(f'  → N_f={rec["Nf"]:.0f} cyc  ε={rec["eps_max"]*100:.2f}%  '
                      f'area={rec["area_mm2"]:.1f} mm²', flush=True)
            else:
                rec['status'] = 'no_epochs'
        except Exception as ex:
            rec['status'] = f'fail: {type(ex).__name__}'
            print(f'  trial {i} FAILED: {ex}', flush=True)
            _restart(args.container, args.tesseract_url)
        results.append(rec)
        json.dump(results, open(out_dir / 'results.json', 'w'), indent=2)

    # ── Rank + summary ─────────────────────────────────────────────────────────
    ok = sorted((r for r in results if r.get('status') == 'ok'),
                key=lambda r: -r['Nf'])
    print(f'\n=== SEARCH DONE: {len(results)} trials, {len(ok)} ok ===', flush=True)
    print('Top 8 by cycles-to-failure N_f:', flush=True)
    for r in ok[:8]:
        print(f'  trial {r["trial"]:3d}: N_f={r["Nf"]:8.0f}  ε={r["eps_max"]*100:5.2f}%  '
              f'area={r["area_mm2"]:5.1f}mm²  |  gap0={r["gap_initial"]*1e3:.1f} '
              f'reach0={r["reach_initial"]*1e3:.1f} bow={r["initial_concave_bow_m"]*1e3:.2f} '
              f'refine={r["mesh_refine"]} wf={r["w_fatigue"]:.1f} wm={r["w_mat"]:.1f} '
              f'wg={r["w_gap"]} lr={r["learning_rate"]}', flush=True)
    json.dump(results, open(out_dir / 'results.json', 'w'), indent=2)
    print(f'Results → {out_dir / "results.json"}', flush=True)
    # Leave the oracle healthy (the last trial's sim can wedge the single-threaded server).
    if not _healthy(args.tesseract_url):
        _restart(args.container, args.tesseract_url)


if __name__ == '__main__':
    main()
