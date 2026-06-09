"""
sofa/sweep_moments.py — Moment sweep: run SOFA for increasing moment values.

Runs evaluate_unit_cell in moment mode for each value in --moments, saves
one .npz per run plus a summary .npz with all per-face kinematics.

Usage (via run_sofa.sh):
    ./sofa/run_sofa.sh sofa/sweep_moments.py \\
        --config  data/configs/sofa/c001_mpnn_2x2.yaml \\
        --mesh-npz data/outputs/runs/<run>/cs_mesh_fixed.npz \\
        --moments  0.001,0.002,0.005,0.010,0.020,0.050 \\
        --out-dir  data/outputs/runs/<run>/moment_sweep/

Outputs (all in --out-dir):
    sweep_M<value>.npz         — full SOFA result for each moment
    sweep_summary.npz          — stacked kinematics array + metadata
    sweep_summary.txt          — human-readable table
"""

import sys
import os
import argparse
import math
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, REPO_ROOT)

try:
    import Sofa
    import Sofa.Core
    import Sofa.Simulation
except ImportError as e:
    sys.exit(f"Cannot import SOFA: {e}\nRun via ./sofa/run_sofa.sh")

from simulate_cell import evaluate_unit_cell
from materials import vm_stress_per_hex
from nff.sofa.config_to_physical import physical_scale_from_config


# ── geometry helpers ──────────────────────────────────────────────────────────

def _face_rotation_rad(nodes_nat, nodes_cur, mask):
    """Least-squares in-plane rotation of a face from its deformed node cloud."""
    pts_nat = nodes_nat[mask, :2]
    pts_cur = nodes_cur[mask, :2]
    c_nat = pts_nat.mean(0)
    c_cur = pts_cur.mean(0)
    H = (pts_nat - c_nat).T @ (pts_cur - c_cur)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    return math.atan2(R[1, 0], R[0, 0])


def _face_centroid_displacement_mm(nodes_nat, nodes_cur, mask):
    """Centroid shift of top-z nodes, in mm."""
    z_nat = nodes_nat[mask, 2]
    top   = z_nat > (z_nat.max() - 1e-9)
    nat_xy = nodes_nat[mask][top, :2].mean(0)
    cur_xy = nodes_cur[mask][top, :2].mean(0)
    return (cur_xy - nat_xy) * 1e3   # m → mm


# ── per-run analysis ──────────────────────────────────────────────────────────

def _analyze(r, n_faces, young_modulus, poisson_ratio):
    """Return a flat dict of scalar outputs for one SOFA run."""
    bc = r['bc_masks']
    out = {
        'strain_energy':        r['strain_energy'],
        'max_xy_displacement_mm': r['max_xy_displacement'] * 1e3,
        'max_z_displacement_mm':  r['max_z_displacement']  * 1e3,
        'max_von_mises_mpa':    r['max_von_mises_stress']  / 1e6,
        'sigma_over_sigmay':    r['first_yield_fraction'],
    }
    for fi in range(n_faces):
        fkey = f'f{fi}'
        if fkey not in bc:
            continue
        mask = bc[fkey].astype(bool)
        dth = _face_rotation_rad(r['nodes_nat'], r['nodes_cur'], mask)
        dxy = _face_centroid_displacement_mm(r['nodes_nat'], r['nodes_cur'], mask)
        out[f'f{fi}_dtheta_deg'] = math.degrees(dth)
        out[f'f{fi}_dx_mm']      = float(dxy[0])
        out[f'f{fi}_dy_mm']      = float(dxy[1])
    return out


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Moment sweep on the 2×2 RDQK CS mesh.')
    ap.add_argument('--config',    required=True,
                    help='Path to sofa YAML config.')
    ap.add_argument('--mesh-npz',  required=True,
                    help='Pre-built CS mesh .npz (from build_mesh_from_centroidal_state).')
    ap.add_argument('--moments',   default='0.001,0.002,0.005,0.010,0.020,0.050',
                    help='Comma-separated moment values in N·m.')
    ap.add_argument('--out-dir',   required=True,
                    help='Output directory for per-run .npz and summary files.')
    args = ap.parse_args()

    import yaml
    with open(args.config) as f:
        raw = yaml.safe_load(f)
    phys = physical_scale_from_config(raw)

    moment_values = [float(m.strip()) for m in args.moments.split(',')]
    os.makedirs(args.out_dir, exist_ok=True)

    # ── load fixed mesh ───────────────────────────────────────────────────────
    print(f"Loading mesh from {args.mesh_npz} ...")
    d = np.load(args.mesh_npz)
    nodes = d['nodes']
    hexes = d['hexes']
    n_faces = int(d['n_faces']) if 'n_faces' in d else sum(
        1 for k in d.files
        if k.startswith('f') and k.endswith('_mask') and k[1:-5].isdigit())
    bc_masks = {}
    for i in range(n_faces):
        m = d[f'f{i}_mask'].astype(bool)
        bc_masks[f'f{i}']     = m
        bc_masks[f'face_{i}'] = m
    bc_masks['clamped'] = d.get('clamped_mask', d['f0_mask']).astype(bool)
    bc_masks['loaded']  = d.get('loaded_mask',  d['f1_mask']).astype(bool)
    print(f"  {len(nodes)} nodes, {len(hexes)} hexes, {n_faces} faces")

    # ── summary table header ──────────────────────────────────────────────────
    clamped_faces_cfg = raw.get('boundary_conditions', {}).get('clamped_faces', [0])
    loaded_faces_cfg  = [int(l['face']) for l in raw.get('loads', []) if 'face' in l]
    print(f"\nClamped faces: {clamped_faces_cfg}")
    print(f"Loaded  faces: {loaded_faces_cfg}")
    print(f"\nSweeping {len(moment_values)} moments: "
          f"{[f'{m:.4f}' for m in moment_values]} N·m\n")

    sep = "─" * 110
    header = (f"{'M [N·m]':>10}  {'E [J]':>10}  {'σ/σ_y':>6}  "
              f"{'XY_max[mm]':>10}  {'Z_max[mm]':>8}  "
              + "  ".join(f"F{fi:2d}_dθ[°]" for fi in range(min(n_faces, 8)))
              + ("  ..." if n_faces > 8 else ""))
    print(sep)
    print(header)
    print(sep)

    all_rows  = []
    all_moms  = []
    all_etots = []

    for M in moment_values:
        tag      = f"M{M:.4f}".replace('.', 'p')
        out_path = os.path.join(args.out_dir, f"sweep_{tag}.npz")

        print(f"  Running M={M:.4f} N·m ...", flush=True)
        r = evaluate_unit_cell(
            nodes, hexes, bc_masks,
            rotation_angle_deg = 0.0,
            applied_moment     = M,
            loading_mode       = 'moment',
            sheet_thickness    = phys.sheet_thickness,
            young_modulus      = phys.young_modulus,
            poisson_ratio      = phys.poisson_ratio,
            yield_strength     = phys.yield_strength,
        )

        # Save full result
        bc       = r['bc_masks']
        vm_hex   = vm_stress_per_hex(r['nodes_nat'], r['nodes_cur'], r['hexes'],
                                      phys.young_modulus, phys.poisson_ratio)
        n_f_out  = sum(1 for k in bc if k.startswith('f') and k[1:].isdigit())
        fmasks   = {f'f{i}_mask': bc[f'f{i}'] for i in range(n_f_out)}
        np.savez(out_path,
                 nodes_nat=r['nodes_nat'], nodes_cur=r['nodes_cur'],
                 hexes=r['hexes'], vm_per_hex=vm_hex,
                 **fmasks,
                 applied_moment       = np.float64(M),
                 is_moment_mode       = np.bool_(True),
                 strain_energy        = np.float64(r['strain_energy']),
                 max_von_mises_stress = np.float64(r['max_von_mises_stress']),
                 max_xy_displacement  = np.float64(r['max_xy_displacement']),
                 max_z_displacement   = np.float64(r['max_z_displacement']),
                 first_yield_fraction = np.float64(r['first_yield_fraction']),
                 n_faces              = np.int32(n_f_out))

        ana  = _analyze(r, n_faces, phys.young_modulus, phys.poisson_ratio)
        dths = [ana.get(f'f{fi}_dtheta_deg', float('nan')) for fi in range(n_faces)]

        row_str = (f"  {M:10.4f}  {ana['strain_energy']:10.3e}  "
                   f"{ana['sigma_over_sigmay']:6.2f}  "
                   f"{ana['max_xy_displacement_mm']:10.3f}  "
                   f"{ana['max_z_displacement_mm']:8.4f}  "
                   + "  ".join(f"{dths[fi]:10.2f}" for fi in range(min(n_faces, 8)))
                   + ("  ..." if n_faces > 8 else ""))
        print(row_str)

        all_rows.append(dths)
        all_moms.append(M)
        all_etots.append(ana['strain_energy'])

    print(sep)

    # ── summary .npz ─────────────────────────────────────────────────────────
    summary_path = os.path.join(args.out_dir, 'sweep_summary.npz')
    dth_arr = np.array(all_rows, dtype=np.float64)   # (n_moments, n_faces)
    np.savez(summary_path,
             moments   = np.array(all_moms),
             dtheta    = dth_arr,
             energies  = np.array(all_etots),
             n_faces   = np.int32(n_faces),
             clamped_faces = np.array(clamped_faces_cfg, dtype=np.int32),
             loaded_faces  = np.array(loaded_faces_cfg,  dtype=np.int32))
    print(f"\nSummary saved → {summary_path}")

    # ── summary .txt ─────────────────────────────────────────────────────────
    txt_path = os.path.join(args.out_dir, 'sweep_summary.txt')
    with open(txt_path, 'w') as fh:
        fh.write(f"Moment sweep — {args.config}\n")
        fh.write(f"Clamped faces: {clamped_faces_cfg}  Loaded faces: {loaded_faces_cfg}\n\n")
        fh.write(header + "\n" + sep + "\n")
        for i, M in enumerate(all_moms):
            dths = all_rows[i]
            row = (f"{M:10.4f}  {all_etots[i]:10.3e}  "
                   + "  ".join(f"{dths[fi]:10.2f}" for fi in range(n_faces)))
            fh.write(row + "\n")
    print(f"Table    saved → {txt_path}")
    print(f"\nDone.  To visualize:")
    print(f"  conda run -n kgnn_mac python nff/sofa/plot_moment_sweep.py "
          f"--sweep-dir {args.out_dir}")


if __name__ == '__main__':
    main()
