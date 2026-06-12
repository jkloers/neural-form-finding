"""
simulate_cell.py — SOFA unit-cell simulation from a pre-built CS mesh.

The mesh (nodes, tets, bc_masks) must be built upstream by
nff.sofa.mesh_builder_gmsh.build_mesh_gmsh with the desired gap and
Bézier hinge-shape parameters.

Pipeline:
  scene_builder.py : build_scene  — SOFA scene + BCs
  materials.py     : svk_energy, vm_stress_per_tet — post-processing
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

from materials     import svk_energy, vm_stress_per_tet, max_principal_strain_per_tet
from scene_builder import build_scene, N_STEPS_DEFAULT, DT

_SOFA_LOCK = threading.Lock()


def evaluate_unit_cell(
    nodes:                  np.ndarray,
    elements:               np.ndarray,
    bc_masks:               dict,
    rotation_angle_deg:     float = 45.0,
    applied_moment:         float = 0.0,
    loading_mode:           str   = 'rotation',
    shear_displacement_m:   float = 0.005,
    tension_displacement_m: float = 0.005,
    sheet_thickness:        float = 0.001,
    young_modulus:          float = 3.5e9,
    poisson_ratio:          float = 0.36,
    yield_strength:         float = 55e6,
    n_steps:                int   = N_STEPS_DEFAULT,
    fem_method:             str   = 'polar',
    rotation_pivot:         tuple | None = None,
    clamp_mode:             str   = 'full',
) -> dict:
    """
    Simulate a kirigami unit cell and return mechanical quantities.

    Parameters
    ----------
    nodes           : (N, 3) float64 — stress-free node positions [m]
    elements        : (M, 4) int32   — tetrahedron connectivity
    bc_masks        : dict — 'clamped'/'loaded' bool masks
    rotation_angle_deg : in-plane rotation of loaded face [deg].
                         Negative = CW = correct RDQK opening direction.
                         Used when loading_mode='rotation'.
    applied_moment  : in-plane torque [N·m]. Used when loading_mode='moment'.
    loading_mode    : 'rotation' | 'moment'
    sheet_thickness : plate thickness in z [m] — used for mass estimate.
    young_modulus, poisson_ratio, yield_strength : material params.

    Returns
    -------
    dict with keys:
      strain_energy         [J]    — total SvK elastic energy
      max_von_mises_stress  [Pa]   — peak von Mises stress
      max_xy_displacement   [m]    — peak in-plane |XY| displacement on free nodes
      max_z_displacement    [m]    — peak |z| buckling on all nodes
      first_yield_fraction  []     — max_vm / yield_strength
      nodes_nat             (N,3)  — natural node positions (= input nodes)
      nodes_cur             (N,3)  — equilibrium node positions
      tets                  (M,4)  — tetrahedron connectivity
      vm                    (M,)   — per-tet von Mises stress [Pa]
      bc_masks              dict   — as provided
    """
    with _SOFA_LOCK:
        root = Sofa.Core.Node("root")
        try:
            mstate = build_scene(
                root, nodes, elements, bc_masks,
                rotation_angle_deg     = rotation_angle_deg,
                applied_moment         = applied_moment,
                loading_mode           = loading_mode,
                shear_displacement_m   = shear_displacement_m,
                tension_displacement_m = tension_displacement_m,
                young                  = young_modulus,
                nu                     = poisson_ratio,
                sheet_thickness        = sheet_thickness,
                fem_method             = fem_method,
                n_steps                = n_steps,
                rotation_pivot         = rotation_pivot,
                clamp_mode             = clamp_mode,
            )
            Sofa.Simulation.init(root)
            for _ in range(n_steps):
                Sofa.Simulation.animate(root, DT)
            nodes_cur = np.array(mstate.position.value, dtype=np.float64)
        finally:
            Sofa.Simulation.unload(root)

    tets = np.asarray(elements)
    strain_e = svk_energy(nodes, nodes_cur, tets, young_modulus, poisson_ratio)
    vm       = vm_stress_per_tet(nodes, nodes_cur, tets, young_modulus, poisson_ratio)
    max_vm   = float(np.max(vm))
    eps      = max_principal_strain_per_tet(nodes, nodes_cur, tets)
    max_eps  = float(np.max(eps))

    _clamped  = bc_masks.get('clamped', bc_masks.get('f0', np.zeros(len(nodes), dtype=bool)))
    free_mask = ~_clamped
    if free_mask.any():
        disp_xy = np.sqrt(
            (nodes_cur[free_mask, 0] - nodes[free_mask, 0])**2 +
            (nodes_cur[free_mask, 1] - nodes[free_mask, 1])**2
        )
        max_xy = float(np.max(disp_xy))
    else:
        max_xy = 0.0

    max_z = float(np.max(np.abs(nodes_cur[:, 2] - nodes[:, 2])))

    return {
        "strain_energy":        strain_e,
        "max_von_mises_stress": max_vm,
        "max_principal_strain": max_eps,
        "max_xy_displacement":  max_xy,
        "max_z_displacement":   max_z,
        "first_yield_fraction": max_vm / yield_strength,
        "nodes_nat":  nodes,
        "nodes_cur":  nodes_cur,
        "tets":       tets,
        "vm":         vm,
        "bc_masks":   bc_masks,
    }
