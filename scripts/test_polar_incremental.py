"""
Quick test: method='polar' incremental loading at multiple rotation angles.

Builds a minimal 2-panel hex mesh (no CentroidalState required) and runs
evaluate_unit_cell at 5°, 15°, 30°, 45° with n_steps=50 for speed.

Run via:
    ./sofa/run_sofa.sh scripts/test_polar_incremental.py
    ./sofa/run_sofa.sh scripts/test_polar_incremental.py --method small  # compare

Expected output: strain_energy and max_xy_displacement increasing with angle,
no NaN / explosion.  method='polar' should reach 45° cleanly.
"""

import argparse
import os
import sys
import time

import numpy as np

# Allow running from repo root or sofa/ directory.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_repo_root, "sofa"))

from simulate_cell import evaluate_unit_cell


# ── Minimal 2-panel hex mesh ───────────────────────────────────────────────────

def _make_two_panel_mesh(
    L: float = 0.010,       # panel side length [m]
    g: float = 0.001,       # gap / hinge width [m]
    t: float = 0.001,       # sheet thickness [m]
    nx: int  = 4,           # hexes per panel in x
    ny: int  = 4,           # hexes in y
    nz: int  = 2,           # hexes through thickness
    ny_hinge: int = 4,      # hexes in y within the hinge strip
    fold_top: float = 0.003 # hinge strip height [m] (fold zone extent)
):
    """
    Two rectangular panels (left clamped, right loaded) connected by a thin hinge strip.

    Layout (top view, XY):
      Panel 0 (clamped):  x ∈ [-L, 0],    y ∈ [0, L]
      Hinge strip:        x ∈ [0, g],      y ∈ [0, fold_top]
      Panel 1 (loaded):   x ∈ [g, g+L],   y ∈ [0, L]

    The fold zone on each panel (y ∈ [0, fold_top]) is meshed at ny_hinge
    resolution; the bulk (y ∈ [fold_top, L]) at ny resolution.

    Returns nodes (N,3), hexes (H,8), bc_masks dict.
    """
    # y breakpoints: bulk | fold zone
    y_bulk  = np.linspace(fold_top, L, ny + 1)
    y_hinge = np.linspace(0.0, fold_top, ny_hinge + 1)
    y_all   = np.concatenate([y_hinge, y_bulk[1:]])

    z_all = np.linspace(-t / 2, t / 2, nz + 1)

    # x for panel 0, hinge, panel 1
    x_p0    = np.linspace(-L, 0.0, nx + 1)
    x_hinge = np.linspace(0.0, g, 3)          # 2 hinge elements
    x_p1    = np.linspace(g, g + L, nx + 1)
    x_all   = np.concatenate([x_p0, x_hinge[1:-1], x_p1])

    Nx = len(x_all)
    Ny = len(y_all)
    Nz = len(z_all)

    # Build node array (Nx × Ny × Nz)
    Xg, Yg, Zg = np.meshgrid(x_all, y_all, z_all, indexing='ij')
    nodes = np.column_stack([Xg.ravel(), Yg.ravel(), Zg.ravel()])

    def nid(ix, iy, iz):
        return ix * Ny * Nz + iy * Nz + iz

    hexes = []
    for ix in range(Nx - 1):
        for iy in range(Ny - 1):
            for iz in range(Nz - 1):
                a = nid(ix,   iy,   iz  )
                b = nid(ix+1, iy,   iz  )
                c = nid(ix+1, iy+1, iz  )
                d = nid(ix,   iy+1, iz  )
                e = nid(ix,   iy,   iz+1)
                f = nid(ix+1, iy,   iz+1)
                gg = nid(ix+1, iy+1, iz+1)
                h = nid(ix,   iy+1, iz+1)
                hexes.append([a, b, c, d, e, f, gg, h])

    hexes = np.array(hexes, dtype=np.int32)
    n = len(nodes)

    tol = min(L, g, t) * 1e-6
    clamped = nodes[:, 0] <= -L + tol          # left face of panel 0
    loaded  = nodes[:, 0] >= g + L - tol       # right face of panel 1

    bc_masks = {
        'clamped': clamped,
        'loaded':  loaded,
        'f0':      clamped,
        'f1':      loaded,
    }

    return nodes, hexes, bc_masks


# ── Test runner ────────────────────────────────────────────────────────────────

def run_test(method: str, n_steps: int, angles_deg: list):
    print(f"\n{'─'*62}")
    print(f"  method='{method}'   n_steps={n_steps}")
    print(f"{'─'*62}")
    print(f"  {'angle':>8}  {'U [mJ]':>10}  {'max_XY [mm]':>13}  {'max_Z [mm]':>12}  {'σ/σy':>6}  {'time':>7}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*13}  {'-'*12}  {'-'*6}  {'-'*7}")

    nodes, hexes, bc_masks = _make_two_panel_mesh()

    for angle in angles_deg:
        t0 = time.time()
        try:
            res = evaluate_unit_cell(
                nodes, hexes, bc_masks,
                rotation_angle_deg = angle,
                loading_mode       = 'rotation',
                n_steps            = n_steps,
                fem_method         = method,
                young_modulus      = 3.5e9,
                poisson_ratio      = 0.36,
                yield_strength     = 50e6,
            )
            U     = res['strain_energy']       * 1e3    # J → mJ
            dxy   = res['max_xy_displacement'] * 1e3    # m → mm
            dz    = res['max_z_displacement']  * 1e3
            sigma = res['first_yield_fraction']
            ok    = "OK" if not np.isnan(U) and U < 1e6 else "DIVERGED"
            print(f"  {angle:>7.1f}°  {U:>10.4f}  {dxy:>13.4f}  {dz:>12.6f}  {sigma:>6.3f}  {time.time()-t0:>6.1f}s  {ok}")
        except Exception as exc:
            print(f"  {angle:>7.1f}°  ERROR: {exc}")

    print()


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="polar",
                        choices=["polar", "small", "large"],
                        help="HexahedronFEM method")
    parser.add_argument("--n-steps", type=int, default=50,
                        help="Incremental load steps (default 50 = fast test)")
    parser.add_argument("--angles", nargs="+", type=float,
                        default=[5.0, 15.0, 30.0, 45.0, 60.0],
                        help="Rotation angles to test [deg]")
    args = parser.parse_args()

    run_test(args.method, args.n_steps, args.angles)
