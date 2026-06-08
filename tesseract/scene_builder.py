"""
scene_builder.py — SOFA scene for the unified kirigami unit-cell mesh.

The kirigami mechanism is IN-PLANE (XY plane). Faces rotate about the z-axis;
hinge strips bend in the XY plane. No out-of-plane motion is prescribed.

Loading regime options (controlled by loading_mode):

  'rotation'  (default) — displacement-controlled, in-plane rigid-body rotation:
    F0 : FixedConstraint (clamped)
    F1 : LinearMovementConstraint — in-plane rotation of F1 about pivot (x_fold, y_min).
         Each node at (x₀,y₀) gets:
           dx = (x₀-xp)*(cosθ-1) - (y₀-yp)*sinθ
           dy = (x₀-xp)*sinθ     + (y₀-yp)*(cosθ-1)
           dz = 0   (in-plane, no out-of-plane motion prescribed)
         Works for any angle θ.  Pivot xp=x_fold (right edge of F0), yp=0.

  'moment'    — force-controlled in-plane torque (CCW about z-axis):
    F0 : FixedConstraint (clamped)
    F1 : ConstantForceField — tangential in-plane forces on F1 nodes.
         F_i = scale * (−dy_i, dx_i, 0) where (dx_i,dy_i) = (x_i−xp, y_i−yp).
         Scale chosen so net moment = applied_moment [N·m] about pivot.

  F2, F3 are always free — driven kinematically through hinges H1, H2, H3.
"""

import numpy as np

N_STEPS = 500     # more steps → smoother large-rotation ramp
DT      = 0.01


def build_scene(root, nodes: np.ndarray, hexes: np.ndarray,
                bc_masks: dict,
                rotation_angle_deg: float = 45.0,
                applied_moment: float = 0.0,
                loading_mode: str = 'rotation',
                young: float = 3.5e9,
                nu:    float = 0.36,
                sheet_thickness: float = 0.001,
                density: float = 1250.0):
    """
    Populate a SOFA root node with the unified kirigami mesh.

    Parameters
    ----------
    root                : Sofa.Core.Node
    nodes               : (N, 3) float64 — natural node positions
    hexes               : (H, 8) int32   — hex connectivity
    bc_masks            : dict 'f0'..'f3' → (N,) bool
    rotation_angle_deg  : float — fold angle for F1 about the H0 axis [degrees]
                          0° = flat, 90° = F1 stands vertical.
                          Used when loading_mode='rotation'.
    applied_moment      : float — bending moment [N·m], used when loading_mode='moment'.
    loading_mode        : 'rotation' | 'moment'
    young, nu           : material (same for faces and hinges)
    sheet_thickness     : used only to estimate density volume

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

    cell.addObject("UniformMass", totalMass=float(density * len(hexes) * 1e-8))

    cell.addObject(
        "HexahedronFEMForceField", template="Vec3d", name="FEM",
        youngModulus=young, poissonRatio=nu,
        method="large",
    )

    # ── Boundary conditions ────────────────────────────────────────────────────
    # Use 'clamped' and 'loaded' masks when available (N-face CS mesh path).
    # Fall back to 'f0'/'f1' for backward compatibility with parametric meshes.
    clamped_mask = bc_masks.get('clamped', bc_masks.get('f0'))
    loaded_mask  = bc_masks.get('loaded',  bc_masks.get('f1'))
    clamped_idx  = np.where(clamped_mask)[0]
    loaded_nodes = np.where(loaded_mask)[0]

    cell.addObject("FixedConstraint", name="clamp_face", indices=clamped_idx.tolist())

    # In-plane pivot: right edge of clamped face, bottom of structure.
    x_fold = float(nodes[clamped_idx, 0].max())
    y_min  = float(nodes[:, 1].min())
    xp, yp = x_fold, y_min

    if loading_mode == 'rotation':
        # In-plane rotation of the loaded face about z-axis through pivot (xp, yp).
        # Each node at (x₀,y₀) gets:
        #   dx = (x₀-xp)*(cosθ-1) - (y₀-yp)*sinθ
        #   dy = (x₀-xp)*sinθ     + (y₀-yp)*(cosθ-1)
        #   dz = 0   (no out-of-plane motion)
        # Group by (x,y) so z-layer nodes share the same constraint.
        theta = np.radians(rotation_angle_deg)
        cosT  = float(np.cos(theta))
        sinT  = float(np.sin(theta))

        xy_load   = nodes[loaded_nodes, :2]
        xy_round  = np.round(xy_load, decimals=9)
        unique_xy, inverse = np.unique(xy_round, axis=0, return_inverse=True)

        for k in range(len(unique_xy)):
            x0, y0    = float(unique_xy[k, 0]), float(unique_xy[k, 1])
            group_idx = loaded_nodes[inverse == k].tolist()
            dx = (x0 - xp) * (cosT - 1.0) - (y0 - yp) * sinT
            dy = (x0 - xp) * sinT          + (y0 - yp) * (cosT - 1.0)
            cell.addObject(
                "LinearMovementConstraint",
                name=f"drive_loaded_{k}",
                template="Vec3d",
                indices=group_idx,
                keyTimes=[0.0, T_final],
                movements=[[0.0, 0.0, 0.0], [float(dx), float(dy), 0.0]],
            )

    elif loading_mode == 'moment':
        # In-plane torque (CCW about z-axis) about loaded-face centroid.
        # Pivot at loaded-face centroid ensures net force = 0 — a pure torque,
        # matching JAX's generalized force on DOF 2 (no net Fx or Fy).
        xp = float(nodes[loaded_nodes, 0].mean())
        yp = float(nodes[loaded_nodes, 1].mean())
        # Tangential force: F_i = scale * (−dy_i, dx_i, 0)
        # Net moment = scale * sum(dx_i² + dy_i²) = applied_moment
        dx_nodes = nodes[loaded_nodes, 0] - xp
        dy_nodes = nodes[loaded_nodes, 1] - yp
        denom    = float(np.sum(dx_nodes**2 + dy_nodes**2))
        if denom < 1e-30:
            raise ValueError("All loaded-face nodes are at pivot — cannot apply in-plane moment.")
        scale  = applied_moment / denom
        forces = np.zeros((len(loaded_nodes), 3))
        forces[:, 0] = -scale * dy_nodes   # Fx tangential
        forces[:, 1] =  scale * dx_nodes   # Fy tangential
        cell.addObject(
            "ConstantForceField", name="moment_loaded",
            template="Vec3d",
            indices=loaded_nodes.tolist(),
            forces=forces.tolist(),
        )

    else:
        raise ValueError(f"Unknown loading_mode: {loading_mode!r}")

    return mstate
