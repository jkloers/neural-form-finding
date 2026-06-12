#!/usr/bin/env python3
"""
scripts/animate_closing.py — animate the hinge closing with live von Mises stress.

Takes the BEST design from a hinge_opt run, sweeps the rotation angle 0 → max on
that fixed geometry by calling the Tesseract oracle per angle (return_fields=True),
and animates the deforming mesh coloured by von Mises stress — you watch the panels
swing shut and the stress bloom in the hinge waist.

Stay in the MODEL-VALID regime: ≤ ~45° for TPU. At 90° the SvK law over-reads stress
badly (needs a hyperelastic oracle), so the field would be unphysical.

Usage:
    python scripts/animate_closing.py [run_dir] [--max-angle 45] [--frames 14]
        [--tesseract-url http://localhost:8000]
    (run_dir defaults to the latest hinge_opt run)
"""
import sys
import argparse
import pathlib

import matplotlib
matplotlib.use('Agg')
import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib.animation as manim
from matplotlib.collections import PolyCollection, LineCollection

REPO = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))

from sofa.hinge_optimizer import (build_physical_cs, _build_tesseract_payload,
                                  _call_tesseract_apply, _decode_array, _PARAM_NAMES)
from visualize_hinge_run import _bottom_tris, _edges, P_BG, P_DARK

OUTPUTS = REPO / 'data' / 'outputs' / 'hinge_opt'

# Regime palette (Princeton-adjacent).
C_ELASTIC = '#5CB87F'   # light green — elastic, unlimited cycles
C_PLASTIC = '#E8A33D'   # amber — plastic, finite cycles
C_CURVE   = '#F58025'   # Princeton orange — the strain path
C_MARK    = '#16324A'   # deep blue — current operating point
C_SVK     = '#6C757D'   # gray — beyond St-Venant–Kirchhoff validity

# St-Venant–Kirchhoff stays physically trustworthy only up to ~20% strain;
# beyond that the materially-linear law extrapolates (the hyperelastic-upgrade zone).
SVK_VALID_STRAIN = 0.20


def _scalar(v):
    """Decode a Tesseract scalar diff output (mirrors the optimizer's _get_val)."""
    if isinstance(v, dict):
        if 'data' in v and 'buffer' in v['data']:
            return float(v['data']['buffer'])
        if 'value' in v:
            return float(v['value'])
    return float(v)


def _cycles_to_failure(eps_p, fat_ef, fat_c):
    """Coffin-Manson low-cycle fatigue: N_f = ½·(ε_plastic/ε_f')^(1/c)."""
    if eps_p <= 0.0:
        return np.inf
    return 0.5 * (eps_p / fat_ef) ** (1.0 / fat_c)


def _latest_run():
    runs = sorted(d for d in OUTPUTS.iterdir() if (d / 'convergence.npz').exists())
    if not runs:
        sys.exit(f'No hinge_opt runs under {OUTPUTS}')
    return runs[-1]


def capture_closing(run_dir, url, max_angle, frames):
    """Sweep the rotation angle on the run's best design; return per-angle states."""
    cfg = yaml.safe_load(open(run_dir / 'config.yaml'))
    cs = build_physical_cs(cfg)
    conv = np.load(run_dir / 'convergence.npz')
    best = int(np.argmin(conv['total_loss']))
    best_phys = {n: float(conv[n][best]) for n in _PARAM_NAMES}
    clamped = sorted({int(f) for f in cs.constrained_face_DOF_pairs[:, 0]})
    loaded  = sorted({int(f) for f in cs.loaded_face_DOF_pairs[:, 0]})

    angles = np.linspace(max_angle / frames, max_angle, frames)
    print(f'Capturing closing 0→{max_angle:.0f}° ({frames} frames) on best design '
          f'(epoch {best + 1}) ...')
    states = []
    for a in angles:
        payload = _build_tesseract_payload(cs, best_phys, cfg, clamped, loaded)
        payload['rotation_angle_deg']   = float(a)
        payload['return_fields']        = True
        payload['skip_secondary_modes'] = True
        fwd = _call_tesseract_apply(url, payload)
        nodes = _decode_array(fwd['deformed_nodes'])
        vm    = _decode_array(fwd['von_mises_field'])
        tets  = _decode_array(fwd['mesh_tets'])
        strain = _scalar(fwd['max_principal_strain'])
        states.append((nodes, vm, tets, float(a), strain))
        print(f'  {a:5.1f}°  →  max σ = {vm.max()/1e6:6.1f} MPa   ε_max = {strain*100:5.1f} %')
    return states


def animate_closing(states, out, fps=6):
    """Animate the closing: wide view (panels swinging shut) + zoom (stress in hinge)."""
    # Connectivity is constant across angles — compute the bottom-tri map once.
    tri, owner = _bottom_tris(states[0][0], states[0][2])
    edg = _edges(tri)
    vmax = max(float(s[1].max()) for s in states) / 1e6   # fixed colour scale [MPa]

    allxy = np.vstack([s[0][:, :2] * 1000 for s in states])
    pad = 0.05 * float(np.ptp(allxy, axis=0).max())
    wlo = allxy.min(0) - pad; whi = allxy.max(0) + pad        # wide bounds

    # Zoom window: the high-stress (hinge) region at full close, fixed across frames.
    xyL, vmL = states[-1][0][:, :2] * 1000, states[-1][1]
    hi = vmL[owner] > 0.05 * vmL.max()
    hpts = xyL[tri[hi]].reshape(-1, 2) if hi.any() else xyL[tri].reshape(-1, 2)
    zc = 0.5 * (hpts.min(0) + hpts.max(0))
    zh = 0.65 * float(np.ptp(hpts, axis=0).max()) + 3.0

    fig, (axW, axZ) = plt.subplots(1, 2, figsize=(14, 7), facecolor=P_BG,
                                   gridspec_kw={'wspace': 0.05})
    fig.suptitle('Closing — live von Mises stress', fontsize=14,
                 fontweight='bold', color=P_DARK, y=0.96)
    xy0 = states[0][0][:, :2] * 1000
    arr0 = states[0][1][owner] / 1e6
    pcW = PolyCollection(xy0[tri], array=arr0, cmap='magma', edgecolors='none',
                         clim=(0, vmax), zorder=1)
    pcZ = PolyCollection(xy0[tri], array=arr0, cmap='magma', edgecolors='none',
                         clim=(0, vmax), zorder=1)
    axW.add_collection(pcW)
    axZ.add_collection(pcZ)
    lcZ = LineCollection(xy0[edg], colors='white', lw=0.25, alpha=0.35, zorder=2)
    axZ.add_collection(lcZ)
    for ax, ttl in [(axW, 'panels closing'), (axZ, 'stress in the hinge')]:
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(ttl, fontsize=11, color=P_DARK, pad=4)
    axW.set_xlim(wlo[0], whi[0]); axW.set_ylim(wlo[1], whi[1])
    axZ.set_xlim(zc[0] - zh, zc[0] + zh); axZ.set_ylim(zc[1] - zh, zc[1] + zh)
    cb = fig.colorbar(pcZ, ax=axZ, fraction=0.046, pad=0.02)
    cb.set_label('von Mises stress [MPa]', fontsize=9); cb.ax.tick_params(labelsize=8)
    ann = axW.text(0.03, 0.04, '', transform=axW.transAxes, fontsize=12,
                   fontweight='bold', color=P_DARK)

    def update(i):
        nodes, vm = states[i][0], states[i][1]; a = states[i][3]
        xy = nodes[:, :2] * 1000; arr = vm[owner] / 1e6
        pcW.set_verts(xy[tri]); pcW.set_array(arr)
        pcZ.set_verts(xy[tri]); pcZ.set_array(arr)
        lcZ.set_segments(xy[edg])
        ann.set_text(f'{a:.0f}°   (σ_max {vm.max()/1e6:.0f} MPa)')
        return ()

    anim = manim.FuncAnimation(fig, update, frames=len(states), blit=False)
    anim.save(str(out), writer=manim.PillowWriter(fps=fps), dpi=115)
    plt.close(fig); print(f'  → {out.name}')


def animate_behavior(states, eps_yield, fat_ef, fat_c, out, fps=6):
    """Stress field (left) beside the strain-vs-rotation failure path (right).

    Left panel: the deforming hinge coloured by von Mises stress (spatial view).
    Right panel: peak strain traced against the imposed rotation, over shaded
    elastic / plastic regimes, with the SvK validity ceiling and a live cycles-
    to-failure readout — the rigorous "how it behaves physically" story.
    """
    tri, owner = _bottom_tris(states[0][0], states[0][2])
    edg = _edges(tri)
    vmax = max(float(s[1].max()) for s in states) / 1e6
    angles = np.array([s[3] for s in states])
    strain = np.array([s[4] for s in states])

    # Zoom window on the hinge waist at full close (fixed across frames).
    xyL, vmL = states[-1][0][:, :2] * 1000, states[-1][1]
    hi = vmL[owner] > 0.05 * vmL.max()
    hpts = xyL[tri[hi]].reshape(-1, 2) if hi.any() else xyL[tri].reshape(-1, 2)
    zc = 0.5 * (hpts.min(0) + hpts.max(0))
    zh = 0.65 * float(np.ptp(hpts, axis=0).max()) + 3.0

    fig, (axH, axC) = plt.subplots(1, 2, figsize=(14, 7), facecolor=P_BG,
                                   gridspec_kw={'wspace': 0.22,
                                                'width_ratios': [1, 1.15]})
    fig.suptitle('Mechanical behavior while closing', fontsize=14,
                 fontweight='bold', color=P_DARK, y=0.96)

    # --- left: deforming hinge, von Mises field ---
    xy0 = states[0][0][:, :2] * 1000; arr0 = states[0][1][owner] / 1e6
    pcH = PolyCollection(xy0[tri], array=arr0, cmap='magma', edgecolors='none',
                         clim=(0, vmax), zorder=1)
    lcH = LineCollection(xy0[edg], colors='white', lw=0.25, alpha=0.35, zorder=2)
    axH.add_collection(pcH); axH.add_collection(lcH)
    axH.set_aspect('equal'); axH.axis('off')
    axH.set_title('stress in the hinge', fontsize=11, color=P_DARK, pad=4)
    axH.set_xlim(zc[0] - zh, zc[0] + zh); axH.set_ylim(zc[1] - zh, zc[1] + zh)
    cb = fig.colorbar(pcH, ax=axH, fraction=0.046, pad=0.02)
    cb.set_label('von Mises stress [MPa]', fontsize=9); cb.ax.tick_params(labelsize=8)

    # --- right: strain path over failure regimes ---
    ymax = max(1.25 * float(strain.max()), eps_yield * 1.4, SVK_VALID_STRAIN * 1.15) * 100
    axC.set_facecolor('white')
    axC.set_xlim(0, angles.max() * 1.02); axC.set_ylim(0, ymax)
    axC.set_xlabel('imposed rotation  [deg]', fontsize=10, color=P_DARK)
    axC.set_ylabel('peak strain  ε$_{max}$  [%]', fontsize=10, color=P_DARK)
    axC.set_title('strain path & failure regimes', fontsize=11, color=P_DARK, pad=4)
    axC.tick_params(labelsize=8)

    yld = eps_yield * 100
    axC.axhspan(0, yld, color=C_ELASTIC, alpha=0.18, lw=0)
    axC.axhspan(yld, ymax, color=C_PLASTIC, alpha=0.16, lw=0)
    axC.axhline(yld, color=C_ELASTIC, lw=1.5)
    axC.text(angles.max() * 0.015, yld * 0.5, 'elastic · unlimited cycles',
             fontsize=8, color='#2C6B47', va='center')
    if yld < ymax * 0.92:
        axC.text(angles.max() * 0.015, (yld + ymax) * 0.5, 'plastic · finite cycles',
                 fontsize=8, color='#9A6B12', va='center')
    svk = SVK_VALID_STRAIN * 100
    if svk < ymax:
        axC.axhspan(svk, ymax, facecolor='none', edgecolor=C_SVK,
                    hatch='///', lw=0, alpha=0.55, zorder=0)
        axC.axhline(svk, color=C_SVK, lw=1.2, ls=':')
        axC.text(angles.max() * 0.99, svk, ' SvK validity limit', fontsize=7.5,
                 color=C_SVK, va='bottom', ha='right')

    line, = axC.plot([], [], color=C_CURVE, lw=2.6, solid_capstyle='round', zorder=4)
    mark, = axC.plot([], [], 'o', color=C_MARK, ms=9, zorder=5)
    ann = axC.text(0.035, 0.96, '', transform=axC.transAxes, fontsize=11,
                   fontweight='bold', color=P_DARK, va='top',
                   bbox=dict(boxstyle='round,pad=0.4', fc='white', ec=C_MARK, alpha=0.85))

    def update(i):
        nodes, vm, _, a, e = states[i]
        xy = nodes[:, :2] * 1000; arr = vm[owner] / 1e6
        pcH.set_verts(xy[tri]); pcH.set_array(arr); lcH.set_segments(xy[edg])
        line.set_data(angles[:i + 1], strain[:i + 1] * 100)
        mark.set_data([a], [e * 100])
        eps_p = max(0.0, e - eps_yield)
        nf = _cycles_to_failure(eps_p, fat_ef, fat_c)
        if e < eps_yield:
            txt = f'{a:.0f}°    ε = {e*100:.1f}%\nelastic · unlimited cycles'
        else:
            life = '∞' if not np.isfinite(nf) else f'{nf:,.0f}'
            txt = f'{a:.0f}°    ε = {e*100:.1f}%\nplastic · N$_f$ ≈ {life} cycles'
            if e > SVK_VALID_STRAIN:
                txt += '\n⚠ beyond SvK validity (extrapolation)'
        ann.set_text(txt)
        return ()

    anim = manim.FuncAnimation(fig, update, frames=len(states), blit=False)
    anim.save(str(out), writer=manim.PillowWriter(fps=fps), dpi=115)
    plt.close(fig); print(f'  → {out.name}')


def main():
    ap = argparse.ArgumentParser(description='Animate the hinge closing with live stress.')
    ap.add_argument('run_dir', nargs='?', default=None)
    ap.add_argument('--max-angle', type=float, default=45.0)
    ap.add_argument('--frames', type=int, default=14)
    ap.add_argument('--tesseract-url', default='http://localhost:8000')
    ap.add_argument('--reuse', action='store_true',
                    help='Re-animate from saved closing_states.npz (skip the sims).')
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir) if args.run_dir else _latest_run()
    print(f'Run: {run_dir.name}')
    npz = run_dir / 'closing_states.npz'

    # Failure thresholds: yield strain from the run's convergence log; fatigue
    # ductility constants from the material config.
    conv = np.load(run_dir / 'convergence.npz')
    eps_yield = float(conv['yield_strain'])
    cfg = yaml.safe_load(open(run_dir / 'config.yaml'))
    mat = cfg.get('material', {})
    fat_ef = float(mat.get('fatigue_ductility_coeff', 1.0))
    fat_c  = float(mat.get('fatigue_ductility_exp', -0.5))

    if args.reuse and npz.exists() and 'strain_seq' in np.load(npz):
        d = np.load(npz)
        tets = d['tets']
        states = [(d['nodes_seq'][i], d['vm_seq'][i], tets,
                   float(d['angles'][i]), float(d['strain_seq'][i]))
                  for i in range(len(d['angles']))]
        print(f'  reusing {len(states)} saved states (no sims).')
    else:
        states = capture_closing(run_dir, args.tesseract_url, args.max_angle, args.frames)
        np.savez(str(npz),
                 angles   = np.array([s[3] for s in states]),
                 strain_seq = np.array([s[4] for s in states]),
                 nodes_seq = np.stack([s[0] for s in states]),
                 vm_seq    = np.stack([s[1] for s in states]),
                 tets      = states[0][2])

    animate_closing(states, run_dir / 'closing.gif')
    animate_behavior(states, eps_yield, fat_ef, fat_c, run_dir / 'behavior.gif')
    print('Done.')


if __name__ == '__main__':
    main()
