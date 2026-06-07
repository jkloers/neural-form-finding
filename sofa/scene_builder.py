"""
scene_builder.py — SOFA scene for the unified kirigami unit-cell mesh.

Loading regime options (controlled by loading_mode):

  'rotation'  (default) — displacement-controlled:
    F0 : FixedConstraint (clamped)
    F1 : LinearMovementConstraint — z_disp(x) = (x-a)/(a+w) * applied_displacement
         Prescribes a rigid-body rotation of F1 about the fold axis at x = a.
         Stable even on near-mechanism structures.

  'moment'    — force-controlled:
    F0 : FixedConstraint (clamped)
    F1 : ConstantForceField — f_z(x_i) proportional to (x_i - x_fold)
         Applies a pure bending moment M [N·m] about the fold axis; zero net force.
         F1 finds its own equilibrium angle.
         Use small applied_moment values to avoid divergence.

  F2, F3 are always free — driven kinematically through hinges H1, H2, H3.
"""

import numpy as np

# ── Defaults: PLA ──────────────────────────────────────────────────────────────
YOUNG_MODULUS  = 3.5e9
POISSON_RATIO  = 0.36
DENSITY        = 1250.0

N_STEPS = 100
DT      = 0.01


def build_scene(root, nodes: np.ndarray, hexes: np.ndarray,
                bc_masks: dict,
                applied_displacement: float = 0.010,
                applied_moment: float = 0.0,
                loading_mode: str = 'rotation',
                young: float = YOUNG_MODULUS,
                nu:    float = POISSON_RATIO,
                sheet_thickness: float = 0.001):
    """
    Populate a SOFA root node with the unified kirigami mesh.

    Parameters
    ----------
    root                 : Sofa.Core.Node
    nodes                : (N, 3) float64 — natural node positions
    hexes                : (H, 8) int32   — hex connectivity
    bc_masks             : dict 'f0'..'f3' → (N,) bool
    applied_displacement : float — peak z-displacement of F1 [m] (rotation mode)
    applied_moment       : float — bending moment about fold axis [N·m] (moment mode)
    loading_mode         : 'rotation' | 'moment'
    young, nu            : material (same for faces and hinges)
    sheet_thickness      : used only to estimate density volume

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
        "Sofa.Component.MechanicalLoad",
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

    # F0: fully clamped
    cell.addObject("FixedConstraint", name="clamp_F0", indices=f0_idx.tolist())

    # Fold axis = right edge of F0 = left edge of H0 strip
    x_fold = float(nodes[f0_idx, 0].max())   # = a
    x_f1   = nodes[f1_nodes, 0]
    x_span = float(x_f1.max() - x_fold)       # = a + w

    if loading_mode == 'rotation':
        # Displacement-controlled rotation of F1 about the fold axis.
        # z_disp(x) = (x - x_fold) / x_span * applied_displacement
        # Prescribes the angle; stable on near-mechanism structures.
        for xi in np.unique(x_f1):
            xi_idx = f1_nodes[x_f1 == xi].tolist()
            z_disp = float((xi - x_fold) / x_span * applied_displacement)
            cell.addObject(
                "LinearMovementConstraint",
                name=f"drive_F1_{len(xi_idx)}_{z_disp:.5f}",
                template="Vec3d",
                indices=xi_idx,
                keyTimes=[0.0, T_final],
                movements=[[0.0, 0.0, 0.0], [0.0, 0.0, z_disp]],
            )

    elif loading_mode == 'moment':
        # Force-controlled: ConstantForceField with f_z(x) ∝ (x - x_fold).
        # Net moment about x = x_fold = applied_moment [N·m]; net vertical force = 0.
        # The structure finds its own equilibrium rotation angle.
        lever  = x_f1 - x_fold                     # (n_f1,)
        denom  = float(np.sum(lever ** 2))
        if denom < 1e-30:
            raise ValueError("All F1 nodes are at the fold axis — cannot apply moment.")
        scale  = applied_moment / denom             # [N / m²]
        forces = np.zeros((len(f1_nodes), 3))
        forces[:, 2] = scale * lever                # f_z = scale * (x - x_fold)
        cell.addObject(
            "ConstantForceField", name="moment_F1",
            template="Vec3d",
            indices=f1_nodes.tolist(),
            forces=forces.tolist(),
        )

    else:
        raise ValueError(f"Unknown loading_mode: {loading_mode!r}")

    return mstate
