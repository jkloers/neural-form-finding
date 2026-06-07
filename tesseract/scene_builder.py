"""
scene_builder.py — SOFA scene for the unified kirigami unit-cell mesh.

Loading regime (kirigami closing):
  F0 face nodes : FixedConstraint (fully clamped, z = 0)
  F1 face nodes : LinearMovementConstraint → [0, 0, applied_displacement]
  F2, F3 nodes  : free — driven kinematically through hinges H1, H2, H3
"""

import numpy as np

# ── Defaults: PLA ──────────────────────────────────────────────────────────────
YOUNG_MODULUS  = 3.5e9
POISSON_RATIO  = 0.36
DENSITY        = 1250.0

N_STEPS = 100
DT      = 0.01


def build_scene(root, nodes: np.ndarray, hexes: np.ndarray,
                bc_masks: dict, applied_displacement: float,
                young: float = YOUNG_MODULUS,
                nu:    float = POISSON_RATIO,
                sheet_thickness: float = 0.001):
    """
    Populate a SOFA root node with the unified kirigami mesh.

    Parameters
    ----------
    root               : Sofa.Core.Node
    nodes              : (N, 3) float64 — natural node positions
    hexes              : (H, 8) int32   — hex connectivity
    bc_masks           : dict 'f0'..'f3' → (N,) bool
    applied_displacement : float  — peak z-displacement of F1 [m]
    young, nu          : material (same for faces and hinges)
    sheet_thickness    : used only to estimate density volume

    Returns
    -------
    mstate : SOFA MechanicalObject (query .position.value for equilibrium)
    """
    root.gravity = [0.0, 0.0, 0.0]
    root.dt = DT
    T_final = N_STEPS * DT

    root.addObject("RequiredPlugin", pluginName=" ".join([
        "Sofa.Component.AnimationLoop",
        "Sofa.Component.StateContainer",
        "Sofa.Component.SolidMechanics.FEM.Elastic",
        "Sofa.Component.Constraint.Projective",
        "Sofa.Component.Mass",
        "Sofa.Component.ODESolver.Backward",
        "Sofa.Component.LinearSolver.Direct",
        "Sofa.Component.Topology.Container.Dynamic",
        "Sofa.Component.Visual",
    ]))
    root.addObject("DefaultAnimationLoop")
    root.addObject("VisualStyle", displayFlags="showBehavior showForceFields")
    root.addObject("EulerImplicitSolver", name="ode",
                   rayleighStiffness=0.1, rayleighMass=0.1)
    root.addObject("SparseLDLSolver", name="solver",
                   template="CompressedRowSparseMatrixMat3x3d")

    cell = root.addChild("kirigami")

    mstate = cell.addObject(
        "MechanicalObject", template="Vec3d", name="DoFs",
        position=nodes.tolist(),
    )

    cell.addObject("HexahedronSetTopologyContainer", name="topo",
                   hexahedra=hexes.tolist())
    cell.addObject("HexahedronSetTopologyModifier")
    cell.addObject("HexahedronSetGeometryAlgorithms", template="Vec3d")

    # Rough total volume for mass — use full bounding box × sheet thickness fraction
    cell.addObject("UniformMass", totalMass=float(DENSITY * len(hexes) * 1e-8))

    cell.addObject(
        "HexahedronFEMForceField", template="Vec3d", name="FEM",
        youngModulus=young, poissonRatio=nu,
        method="large",
    )

    # ── Boundary conditions ────────────────────────────────────────────────────
    f0_idx   = np.where(bc_masks['f0'])[0]
    f1_nodes = np.where(bc_masks['f1'])[0]

    cell.addObject("FixedConstraint", name="clamp_F0", indices=f0_idx.tolist())

    # Rotation loading: F1 rotates about H0's fold axis at x = a (F0 right edge).
    # z_disp(x) = (x − a) / (a + w) * applied_displacement
    #   x = a       (F0 right edge, clamped)  → z = 0
    #   x = a+w     (H0–F1 boundary)          → z = w/(a+w)*δ  (small, ~0.5 mm)
    #   x = 2a+w    (F1 far edge)              → z = δ
    # This bends H0 gently (no shear overload) and drives the kirigami mechanism.
    x_f1      = nodes[f1_nodes, 0]
    x_left    = nodes[f0_idx, 0].max()  # = a, fold axis = F0 right edge
    x_span    = x_f1.max() - x_left    # = a + w

    for xi in np.unique(x_f1):
        xi_idx  = f1_nodes[x_f1 == xi].tolist()
        z_disp  = float((xi - x_left) / x_span * applied_displacement)
        cell.addObject(
            "LinearMovementConstraint", name=f"drive_F1_{len(xi_idx)}_{z_disp:.5f}",
            template="Vec3d",
            indices=xi_idx,
            keyTimes=[0.0, T_final],
            movements=[[0.0, 0.0, 0.0], [0.0, 0.0, z_disp]],
        )

    return mstate
