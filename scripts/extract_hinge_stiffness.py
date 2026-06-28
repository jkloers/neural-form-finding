#!/usr/bin/env python3
"""
scripts/extract_hinge_stiffness.py — calibrate JAX hinge springs from a SOFA hinge.

Takes the optimal Bézier hinge from a hinge_opt run and measures its three
small-deformation stiffnesses by sweeping the rotation / shear / tension load
modes at small magnitudes and reading the SOFA elastic energy. For a linear
spring  U = ½·k·x²,  so  k = 2U/x²  (fitted as the slope of U vs ½x²).

These are the physical analogues of the JAX Stage-2 hinge springs
(k_rot, k_shear, k_stretch) — the first rung of SOFA→JAX calibration: the
macro pipeline's phenomenological stiffnesses grounded in the real optimized
hinge mechanics.

Outputs (in the run dir):
    hinge_stiffness.json   the three k's, ratios, linearity R², effective arm
    hinge_stiffness.png    U-vs-x² fits + per-magnitude tangent k (linearity)

Usage:
    python scripts/extract_hinge_stiffness.py [run_dir] [--tesseract-url URL]
        [--rot-deg 1,2,3,4,5] [--disp-mm 0.1,0.2,0.3,0.4,0.5] [--n-steps 20]
    (run_dir defaults to the latest hinge_opt run)
"""
import sys
import json
import argparse
import pathlib

import matplotlib
matplotlib.use('Agg')
import numpy as np
import yaml
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from tesseract_core import Tesseract
from nff.sofa import oracle_payload as op
from nff.sofa.hinge_viz import P_BG, P_DARK

OUTPUTS = REPO / 'data' / 'outputs' / 'hinge_opt'

# Per-mode colours (rotation / shear / tension).
C_ROT, C_SHEAR, C_STRETCH = '#F58025', '#1B6B3A', '#16324A'


def _latest_run():
    runs = sorted(d for d in OUTPUTS.iterdir() if (d / 'convergence.npz').exists())
    if not runs:
        sys.exit(f'No hinge_opt runs under {OUTPUTS}')
    return runs[-1]


def _fit_stiffness(x, U):
    """Fit U = ½·k·x² through the origin; return (k, R²) and per-point tangent k."""
    x, U = np.asarray(x, float), np.asarray(U, float)
    a = 0.5 * x ** 2                      # U = k·a
    k = float(np.sum(U * a) / np.sum(a * a))
    ss_res = float(np.sum((U - k * a) ** 2))
    ss_tot = float(np.sum((U - U.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    k_point = 2.0 * U / x ** 2            # secant stiffness at each magnitude
    return k, r2, k_point


def sweep(run_dir, url, rot_deg, disp_mm, n_steps):
    """Sweep the three modes at small magnitudes; return energies per magnitude."""
    cfg = yaml.safe_load(open(run_dir / 'config.yaml'))
    cs = op.build_physical_cs(cfg)
    conv = np.load(run_dir / 'convergence.npz')
    best = int(np.argmin(conv['total_loss']))
    best_phys = {n: float(conv[n][best]) for n in op.PARAM_NAMES}
    clamped = sorted({int(f) for f in cs.constrained_face_DOF_pairs[:, 0]})
    loaded  = sorted({int(f) for f in cs.loaded_face_DOF_pairs[:, 0]})
    print(f'Optimal design = epoch {best + 1} of {run_dir.name}')
    print(f'  params: ' + '  '.join(f'{n}={best_phys[n]*1000:.2f}mm' for n in op.PARAM_NAMES[:5]))

    base = op.build_payload(cs, best_phys, cfg, clamped, loaded)
    base['skip_secondary_modes'] = False     # run rotation + shear + tension
    base['return_fields'] = False
    base['n_steps'] = int(n_steps)

    rot_rad = np.radians(rot_deg)
    disp_m  = np.asarray(disp_mm) * 1e-3
    rec = {'theta': [], 'E_rot': [], 'u': [], 'E_shear': [], 'E_tension': []}
    n = max(len(rot_rad), len(disp_m))
    oracle = Tesseract.from_url(url)
    print(f'Sweeping {n} magnitudes (rotation + shear + tension per call) ...')
    for i in range(n):
        th = float(rot_rad[min(i, len(rot_rad) - 1)])
        u  = float(disp_m[min(i, len(disp_m) - 1)])
        p = dict(base)
        p['rotation_angle_deg']     = float(np.degrees(th))
        p['shear_displacement_m']   = u
        p['tension_displacement_m'] = u
        fwd = oracle.apply(p)
        er = float(fwd['strain_energy'])
        es = float(fwd['energy_shear'])
        et = float(fwd['energy_tension'])
        rec['theta'].append(th); rec['E_rot'].append(er)
        rec['u'].append(u); rec['E_shear'].append(es); rec['E_tension'].append(et)
        print(f'  θ={np.degrees(th):4.1f}°  E_rot={er:.3e} J   '
              f'u={u*1e3:.2f}mm  E_shear={es:.3e}  E_tension={et:.3e}')
    return {k: np.asarray(v) for k, v in rec.items()}


def main():
    ap = argparse.ArgumentParser(description='Extract hinge spring stiffnesses from SOFA.')
    ap.add_argument('run_dir', nargs='?', default=None)
    ap.add_argument('--tesseract-url', default='http://localhost:8000')
    ap.add_argument('--rot-deg', default='1,2,3,4,5')
    ap.add_argument('--disp-mm', default='0.1,0.2,0.3,0.4,0.5')
    ap.add_argument('--n-steps', type=int, default=20)
    ap.add_argument('--replot', action='store_true',
                    help='Redraw from the saved hinge_stiffness.json raw data (no SOFA).')
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir) if args.run_dir else _latest_run()

    if args.replot:
        raw = json.loads((run_dir / 'hinge_stiffness.json').read_text())['raw']
        rec = {'theta': np.radians(raw['rotation_deg']), 'E_rot': np.asarray(raw['E_rot_J']),
               'u': np.asarray(raw['disp_mm']) * 1e-3,
               'E_shear': np.asarray(raw['E_shear_J']), 'E_tension': np.asarray(raw['E_tension_J'])}
        print(f'Replotting {run_dir.name} from saved raw data (no SOFA).')
    else:
        rot_deg = [float(x) for x in args.rot_deg.split(',')]
        disp_mm = [float(x) for x in args.disp_mm.split(',')]
        rec = sweep(run_dir, args.tesseract_url, rot_deg, disp_mm, args.n_steps)

    k_rot, r2_rot, kp_rot = _fit_stiffness(rec['theta'], rec['E_rot'])
    k_shear, r2_sh, kp_sh = _fit_stiffness(rec['u'], rec['E_shear'])
    k_stretch, r2_st, kp_st = _fit_stiffness(rec['u'], rec['E_tension'])
    L_eff = float(np.sqrt(k_rot / k_stretch)) if k_stretch > 0 else float('nan')

    out = {
        'run': run_dir.name,
        'k_rot_Nm_per_rad':   k_rot,
        'k_shear_N_per_m':    k_shear,
        'k_stretch_N_per_m':  k_stretch,
        'ratio_shear_stretch': k_shear / k_stretch if k_stretch else None,
        'effective_arm_m':    L_eff,
        'linearity_R2': {'rot': r2_rot, 'shear': r2_sh, 'stretch': r2_st},
        'raw': {
            'rotation_deg': list(np.degrees(rec['theta'])), 'E_rot_J': list(rec['E_rot']),
            'disp_mm': list(rec['u'] * 1e3),
            'E_shear_J': list(rec['E_shear']), 'E_tension_J': list(rec['E_tension']),
        },
    }
    (run_dir / 'hinge_stiffness.json').write_text(json.dumps(out, indent=2))

    print('\n── Optimal-hinge stiffnesses (small-deformation) ──')
    print(f'  k_rot     = {k_rot:.4g} N·m/rad   (R²={r2_rot:.4f})')
    print(f'  k_shear   = {k_shear:.4g} N/m      (R²={r2_sh:.4f})')
    print(f'  k_stretch = {k_stretch:.4g} N/m      (R²={r2_st:.4f})')
    print(f'  k_shear/k_stretch = {k_shear/k_stretch:.3f}   effective arm = {L_eff*1e3:.2f} mm')

    # ── Plot: one panel per mode, energy vs squared load. A linear spring U=½kx²
    #    is a STRAIGHT LINE through the origin (slope = ½k); the fitted k is printed. ──
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), facecolor=P_BG)
    panels = [
        ('rotation', rec['theta'], rec['E_rot'], k_rot, r2_rot, C_ROT,
         r'rotation$^2$   θ²  [rad²]', f'k$_{{rot}}$ = {k_rot:.3g} N·m/rad'),
        ('shear', rec['u'], rec['E_shear'], k_shear, r2_sh, C_SHEAR,
         r'shear$^2$   u²  [m²]', f'k$_{{shear}}$ = {k_shear:,.0f} N/m'),
        ('tension', rec['u'], rec['E_tension'], k_stretch, r2_st, C_STRETCH,
         r'tension$^2$   u²  [m²]', f'k$_{{stretch}}$ = {k_stretch:,.0f} N/m'),
    ]
    for ax, (name, x, U, k, r2, c, xl, klabel) in zip(axes, panels):
        x2 = np.asarray(x) ** 2; U = np.asarray(U)
        ax.scatter(x2, U, color=c, s=60, zorder=3)
        lo = np.linspace(0, x2.max() * 1.03, 50)
        ax.plot(lo, 0.5 * k * lo, color=c, lw=2.2, zorder=2)        # U = ½·k·x²
        ax.set_title(name, fontsize=13, color=P_DARK, fontweight='bold', pad=6)
        ax.set_xlabel(xl, fontsize=10, color=P_DARK)
        ax.set_ylabel('elastic energy  U  [J]', fontsize=10, color=P_DARK)
        ax.text(0.05, 0.93, f'{klabel}\nR² = {r2:.3f}', transform=ax.transAxes,
                fontsize=12.5, fontweight='bold', va='top', color=c)
        ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        ax.set_facecolor('white'); ax.grid(alpha=0.18); ax.tick_params(labelsize=8)
        ax.ticklabel_format(style='sci', scilimits=(-2, 3), axis='both')
    fig.suptitle('Optimal-hinge stiffness — energy vs squared load   '
                 '(straight line through origin = linear spring; slope ∝ stiffness)',
                 fontsize=12.5, fontweight='bold', color=P_DARK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(run_dir / 'hinge_stiffness.png', dpi=150, facecolor=P_BG)
    plt.close(fig)
    print(f'\n  → {run_dir.name}/hinge_stiffness.json')
    print(f'  → {run_dir.name}/hinge_stiffness.png')


if __name__ == '__main__':
    main()
