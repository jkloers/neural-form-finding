"""
simulate_cell.py — Public entry point for the 1×1 unit_RDQK_0 kirigami simulation.

Pipeline:
  geometry.py      : build_unified_mesh  — faces + hinges, one mesh
  scene_builder.py : build_scene         — SOFA scene + BCs
  materials.py     : svk_energy, vm_stress_per_tet  — post-processing

Loading regime: F0 clamped, F1 driven IN-PLANE (XY) → kirigami opening in 2D plane.
Same material (PLA) throughout; hinge strips are geometrically thin (fold_length << face_size)
so deformation concentrates there.

Extension path:
  Phase 3c : face-face contact (FreeMotionAnimationLoop)
  Phase 3d : plasticity (d_plasticYieldThreshold)
  Phase 4  : Tesseract HTTP API → JAX training reward
"""

import sys
import threading

import numpy as np

try:
    import Sofa
    import Sofa.Core
    import Sofa.Simulation
except ImportError as e:
    sys.exit(f"Cannot import SOFA: {e}\nRun via ./sofa/run_sofa.sh")

from geometry     import build_unified_mesh
from materials    import hex_to_5tets, svk_energy, vm_stress_per_tet
from scene_builder import build_scene, N_STEPS, DT

# ── Material defaults: PLA ─────────────────────────────────────────────────────
YOUNG_MODULUS  = 3.5e9
POISSON_RATIO  = 0.36
YIELD_STRENGTH = 55e6
DENSITY        = 1250.0

# ── Geometry defaults ──────────────────────────────────────────────────────────
FACE_SIZE       = 0.100   # m — 100 mm square face
SHEET_THICKNESS = 0.001   # m — 1 mm sheet

_SOFA_LOCK = threading.Lock()


def evaluate_unit_cell(
    hinge_arm_width:      float = 0.010,
    hinge_fold_length:    float = 0.003,
    rotation_angle_deg:   float = 45.0,
    applied_moment:       float = 0.0,
    loading_mode:         str   = 'rotation',
    face_size:            float = FACE_SIZE,
    sheet_thickness:      float = SHEET_THICKNESS,
    young_modulus:        float = YOUNG_MODULUS,
    poisson_ratio:        float = POISSON_RATIO,
    yield_strength:       float = YIELD_STRENGTH,
    n_face:               int   = 4,
    n_hinge:              int   = 4,
    n_z:                  int   = 2,
    mesh_data:            tuple = None,
) -> dict:
    """
    Simulate the 1×1 kirigami unit cell and return mechanical quantities.

    Parameters
    ----------
    hinge_arm_width : float
        Gap between adjacent face panels (hinge spans this) [m]. Default: 10 mm.
    hinge_fold_length : float
        Thin dimension of the hinge strip along the face edge [m]. Default: 3 mm.
        The strip spans arm_width in one direction and fold_length (tiny) in the
        other — making it a slender beam that stores bending moment about z.
    rotation_angle_deg : float
        In-plane rotation angle for F1 about hinge corner (x_fold, 0) [degrees].
        Negative = CW = correct RDQK opening direction. 0 = flat, -45 = canonical.
    applied_moment : float
        In-plane torque (about z-axis) [N·m] — used when loading_mode='moment'.
    loading_mode : str
        'rotation' — displacement-controlled, prescribes exact in-plane angle.
        'moment'   — force-controlled (in-plane tangential forces, F1 finds equilibrium).
    face_size : float
        Square face panel side [m]. Default: 100 mm.
    sheet_thickness : float
        Plate thickness in z [m]. Default: 1 mm.
    young_modulus, poisson_ratio, yield_strength : material params.
    n_face, n_hinge, n_z : mesh resolution (elements per region).
    mesh_data : tuple or None
        Optional pre-built (nodes, hexes, bc_masks) from
        nff.sofa.mesh_builder.build_mesh_from_centroidal_state.
        When provided, skips the build_unified_mesh call entirely.
        bc_masks must contain 'f0'..'f3' and 'clamped'/'loaded' keys.

    Returns
    -------
    dict with keys:
      strain_energy         [J]    — total SvK elastic energy
      max_von_mises_stress  [Pa]   — peak von Mises stress
      max_xy_displacement   [m]    — peak in-plane |XY| displacement on free nodes
      max_z_displacement    [m]    — peak |z| buckling on all nodes (undesired)
      first_yield_fraction  []     — max_vm / yield_strength
      nodes_nat             (N,3)  — natural node positions
      nodes_cur             (N,3)  — equilibrium node positions
      hexes                 (H,8)  — hex connectivity
      bc_masks              dict   — 'f0'..'f3' bool masks
    """
    if mesh_data is not None:
        nodes, hexes, bc_masks = mesh_data
    else:
        nodes, hexes, bc_masks = build_unified_mesh(
            face_size       = face_size,
            arm_width       = hinge_arm_width,
            fold_length     = hinge_fold_length,
            sheet_thickness = sheet_thickness,
            n_face          = n_face,
            n_hinge         = n_hinge,
            n_z             = n_z,
        )

    with _SOFA_LOCK:
        root = Sofa.Core.Node("root")
        mstate = build_scene(
            root, nodes, hexes, bc_masks,
            rotation_angle_deg = rotation_angle_deg,
            applied_moment     = applied_moment,
            loading_mode       = loading_mode,
            young              = young_modulus,
            nu                 = poisson_ratio,
            sheet_thickness    = sheet_thickness,
        )
        Sofa.Simulation.init(root)
        for _ in range(N_STEPS):
            Sofa.Simulation.animate(root, DT)
        nodes_cur = np.array(mstate.position.value, dtype=np.float64)
        Sofa.Simulation.unload(root)

    # Post-processing on equilibrium positions
    tets = hex_to_5tets(hexes)
    strain_e = svk_energy(nodes, nodes_cur, tets, young_modulus, poisson_ratio)
    vm       = vm_stress_per_tet(nodes, nodes_cur, tets, young_modulus, poisson_ratio)
    max_vm   = float(np.max(vm))

    # In-plane displacement: XY motion of non-clamped nodes (desired mechanism signal)
    free_mask = ~bc_masks['f0'] & ~bc_masks['f1']
    if free_mask.any():
        disp_xy = np.sqrt(
            (nodes_cur[free_mask, 0] - nodes[free_mask, 0])**2 +
            (nodes_cur[free_mask, 1] - nodes[free_mask, 1])**2
        )
        max_xy = float(np.max(disp_xy))
    else:
        max_xy = 0.0

    # Z buckling: out-of-plane deformation (undesired, should stay near 0)
    max_z = float(np.max(np.abs(nodes_cur[:, 2] - nodes[:, 2])))

    return {
        "strain_energy":        strain_e,
        "max_von_mises_stress": max_vm,
        "max_xy_displacement":  max_xy,
        "max_z_displacement":   max_z,
        "first_yield_fraction": max_vm / yield_strength,
        # raw data for visualisation / debugging
        "nodes_nat":  nodes,
        "nodes_cur":  nodes_cur,
        "hexes":      hexes,
        "bc_masks":   bc_masks,
    }


# ── Validation ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 72)
    print("SOFA Unit Cell — Unified Mesh, Kirigami Loading  (PLA, 20 cm)")
    print("=" * 72)
    hdr = f"{'Config':<40} {'E [J]':>9} {'σ_vm [MPa]':>11} {'z_max [mm]':>11} {'σ/σ_y':>7}"
    print(hdr)
    print("-" * len(hdr))

    cases = [
        dict(label="canonical  (w=10mm, L=3mm, θ=−45°)",
             hinge_arm_width=0.010, hinge_fold_length=0.003,
             rotation_angle_deg=-45.0),
        dict(label="short hinge (L=1.5mm, θ=−45°)",
             hinge_arm_width=0.010, hinge_fold_length=0.0015,
             rotation_angle_deg=-45.0),
        dict(label="thin sheet  (t=0.5mm, θ=−45°)",
             hinge_arm_width=0.010, hinge_fold_length=0.003,
             rotation_angle_deg=-45.0, sheet_thickness=0.0005),
        dict(label="large angle (θ=−90°)",
             hinge_arm_width=0.010, hinge_fold_length=0.003,
             rotation_angle_deg=-90.0),
    ]

    for case in cases:
        label = case.pop("label")
        r = evaluate_unit_cell(**case)
        print(f"  {label:<40} {r['strain_energy']:>9.4e}"
              f" {r['max_von_mises_stress']/1e6:>11.2f}"
              f" {r['max_z_displacement']*1e3:>11.4f}"
              f" {r['first_yield_fraction']:>7.3f}")

    print("-" * len(hdr))
    print("\nExpected:")
    print("  short hinge → higher stress, easier fold (EI ∝ L³)")
    print("  thin sheet  → energy ↓ ×8 (EI ∝ t³), more z-buckling")
    print("  large angle → energy ↑, stress ↑, z-buckling ↑")
    print("=" * 72)
