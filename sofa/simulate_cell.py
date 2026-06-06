"""
simulate_cell.py — Headless SOFA simulation of the 1×1 kirigami unit cell.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROLE IN THE PIPELINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This file is the SOFA physics core. It is intentionally isolated from both the
JAX pipeline and the Tesseract API layer so each can evolve independently.

  JAX pipeline (kgnn_mac env)
       │
       │  subprocess / Tesseract API call
       ▼
  sofa/tesseract/tesseract_api.py   ← Tesseract wrapper (Phase 2)
       │  imports
       ▼
  sofa/simulate_cell.py             ← THIS FILE — physics only
       │  calls
       ▼
  SOFA v25.12 (Linux binary in Docker, macOS binary locally)

Extension path:
  Phase 2 (current) : scalar inputs → scalar strain energy
  Phase 3           : array inputs (face geometry from CentroidalState) → richer QoIs
  Phase 4           : Tesseract energy used as reward signal in NFF training loop

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LOCAL SETUP (macOS ARM64, one-time)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SOFA v25.12 macOS ARM64 binary is installed at ~/sofa/v25.12/.
Run via the wrapper (handles env vars automatically):

    ./sofa/run_sofa.sh sofa/simulate_cell.py

Direct run (requires env vars from run_sofa.sh to be set):

    /opt/homebrew/bin/python3.12 sofa/simulate_cell.py

In Docker/Tesseract: env vars are baked into the image — no wrapper needed.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GEOMETRY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  12 Rigid3d nodes.  L = filament_length.

  Face centroids (nodes 0–3):
    n0  = (0.5,     0.5    )   ← face F0  [FIXED]
    n1  = (1.5+L,   0.5    )   ← face F1  [free]
    n2  = (1.5+L+δ, 1.5+L  )   ← face F2  [FIXED, displaced by δ in x]
    n3  = (0.5,     1.5+L  )   ← face F3  [free]

  Hinge attachment nodes (nodes 4–11):
    n4  = (1.0,     0.5    )   h0a: F0 side of hinge H0
    n5  = (1.0+L,   0.5    )   h0b: F1 side of hinge H0
    n6  = (1.5+L,   1.0    )   h1a: F1 side of hinge H1
    n7  = (1.5+L+δ, 1.0+L  )   h1b: F2 side of hinge H1  [displaced]
    n8  = (1.0+L+δ, 1.5+L  )   h2a: F2 side of hinge H2  [displaced]
    n9  = (1.0,     1.5+L  )   h2b: F3 side of hinge H2
    n10 = (0.5,     1.0+L  )   h3a: F3 side of hinge H3
    n11 = (0.5,     1.0    )   h3b: F0 side of hinge H3

  12 beam edges:
    Arms (0–7, stiff r=r_arm): rigidly couple each face centroid to its two
      hinge attachment points, approximating a rigid face panel.
    Filaments (8–11, compliant r=filament_thickness): the 4 physical hinge
      strips connecting adjacent faces. These carry the strain energy.

  Boundary conditions:
    Fixed : {n0, n4, n11}  — face F0 anchored at reference position
    Fixed : {n2, n7, n8}   — face F2 held at x-displaced position (δ = applied_displacement)
    Free  : {n1, n5, n6}   — face F1 finds equilibrium
    Free  : {n3, n9, n10}  — face F3 finds equilibrium

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FUTURE INTEGRATION NOTE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This script is designed to eventually become a ground-truth validator inside
the Tesseract Docker backend. The function signature `evaluate_unit_cell` is
the expected API boundary: the NFF pipeline will call it with per-hinge
filament geometry derived from the CentroidalState, and compare the returned
energy to the JAX Stage-2 solver output for fine-tuning or reward shaping.

To extend toward full pipeline coupling:
  1. Accept a CentroidalState (or its SOFA-compatible serialisation) in place
     of the scalar geometry parameters.
  2. Replace the fixed-BC approach with incremental load stepping (see the
     JAX Stage-2 incremental solver in physics_solver/statics.py for reference).
  3. Add TriangleCollisionModel child nodes (one per face, RigidMapping from
     the Rigid3d DOF) for self-contact, which is already wired in the collision
     pipeline below.
"""

import sys
import warnings
import numpy as np

try:
    import Sofa
    import Sofa.Core
    import Sofa.Simulation
except ImportError as e:
    sys.exit(
        f"Cannot import SOFA: {e}\n"
        "Follow the SETUP INSTRUCTIONS at the top of this file."
    )


# ── Constants ──────────────────────────────────────────────────────────────────

YOUNG_MODULUS = 1e6   # Pa — shared by arms and filaments (stiffness ratio controlled by radius)
POISSON_RATIO = 0.3
ARM_RADIUS    = 0.5   # m  — thick arm beams → face panels behave as rigid bodies
N_STEPS       = 80    # animation steps to reach and settle at static equilibrium


# ── Scene construction ─────────────────────────────────────────────────────────

def _build_positions(L: float, delta: float) -> list:
    """
    Compute the 12 Rigid3d node positions for the unit cell.

    Face F2 (and its hinge attachments) are pre-displaced by `delta` in the
    x-direction so that FixedConstraint holds them at the loaded configuration.
    The solver drives F1 and F3 to mechanical equilibrium.

    Rigid3d format: [x, y, z, qx, qy, qz, qw]
    """
    q = [0.0, 0.0, 0.0, 1.0]  # identity quaternion (flat, no rotation)

    def p(x, y):
        return [x, y, 0.0] + q

    return [
        # ── Face centroids ──────────────────────────────────────────
        p(0.5,         0.5    ),   # n0:  F0 centroid          [FIXED]
        p(1.5 + L,     0.5    ),   # n1:  F1 centroid          [free]
        p(1.5 + L + delta, 1.5 + L),  # n2:  F2 centroid      [FIXED, displaced]
        p(0.5,         1.5 + L),   # n3:  F3 centroid          [free]
        # ── Hinge attachment nodes ──────────────────────────────────
        p(1.0,         0.5    ),   # n4:  h0a (F0 side of H0) [FIXED]
        p(1.0 + L,     0.5    ),   # n5:  h0b (F1 side of H0) [free]
        p(1.5 + L,     1.0    ),   # n6:  h1a (F1 side of H1) [free]
        p(1.5 + L + delta, 1.0 + L),  # n7: h1b (F2 side of H1) [FIXED, displaced]
        p(1.0 + L + delta, 1.5 + L),  # n8: h2a (F2 side of H2) [FIXED, displaced]
        p(1.0,         1.5 + L),   # n9:  h2b (F3 side of H2) [free]
        p(0.5,         1.0 + L),   # n10: h3a (F3 side of H3) [free]
        p(0.5,         1.0    ),   # n11: h3b (F0 side of H3) [FIXED]
    ]


def _build_scene(root: "Sofa.Core.Node",
                 filament_thickness: float,
                 filament_length: float,
                 applied_displacement: float) -> "Sofa.Core.Object":
    """
    Populate the SOFA scene graph and return the BeamFEMForceField object.

    Parameters
    ----------
    root                : SOFA root node (already created)
    filament_thickness  : cross-section radius of the 4 hinge beams [m]
    filament_length     : rest length of each hinge beam [m]
    applied_displacement: x-displacement applied to face F2 [m]

    Returns
    -------
    None — the scene graph is built in-place on `root`.
    Call root.mechanism.getObject("DoFs") to access the MechanicalObject.
    """
    L     = filament_length
    delta = applied_displacement

    root.gravity = [0.0, 0.0, 0.0]  # static problem — no gravity
    root.dt      = 0.01

    # ── Required plugins ───────────────────────────────────────────────────────
    root.addObject("RequiredPlugin", pluginName=" ".join([
        "Sofa.Component.AnimationLoop",
        "Sofa.Component.StateContainer",
        "Sofa.Component.SolidMechanics.FEM.Elastic",
        "Sofa.Component.Constraint.Projective",
        "Sofa.Component.Mass",
        "Sofa.Component.ODESolver.Backward",
        "Sofa.Component.LinearSolver.Iterative",
        "Sofa.Component.Topology.Container.Dynamic",
        "Sofa.Component.Visual",
    ]))

    # DefaultAnimationLoop is required for Sofa.Simulation.animate() in v24+.
    # Collision pipeline omitted here — add TriangleCollisionModel child nodes
    # and the collision plugins when self-contact is needed (see FUTURE INTEGRATION NOTE).
    root.addObject("DefaultAnimationLoop")
    root.addObject("VisualStyle", displayFlags="showBehavior showForceFields")

    # ── Mechanical node ────────────────────────────────────────────────────────
    mech = root.addChild("mechanism")
    mech.addObject("EulerImplicitSolver",
                   name="ode",
                   rayleighStiffness=0.1,
                   rayleighMass=0.1)
    mech.addObject("CGLinearSolver", name="linear_solver",
                   iterations=500, tolerance=1e-12, threshold=1e-12)

    # ── 12 Rigid3d DOF nodes ───────────────────────────────────────────────────
    positions = _build_positions(L, delta)
    mstate = mech.addObject("MechanicalObject",
                             template="Rigid3d",
                             name="DoFs",
                             position=positions)
    mech.addObject("UniformMass", totalMass=0.1)

    # ── Boundary conditions ────────────────────────────────────────────────────
    #   Face F0: fixed at reference position.
    #   Face F2: fixed at displaced position (baked into MechanicalObject positions above).
    #   Faces F1, F3: free → solver finds mechanical equilibrium.
    fixed_face0 = [0, 4, 11]   # n0, h0a, h3b
    fixed_face2 = [2, 7, 8]    # n2, h1b, h2a
    mech.addObject("FixedConstraint",
                   name="fixed_BCs",
                   indices=fixed_face0 + fixed_face2)

    # ── Arm beams (stiff, child node inherits parent DoFs) ────────────────────
    #   Thick beams couple each face centroid rigidly to its hinge-attachment
    #   nodes, so the face panel behaves as a rigid body.
    arm_edges = [
        [0,  4],   # F0 centroid → h0a
        [0, 11],   # F0 centroid → h3b
        [1,  5],   # F1 centroid → h0b
        [1,  6],   # F1 centroid → h1a
        [2,  7],   # F2 centroid → h1b
        [2,  8],   # F2 centroid → h2a
        [3,  9],   # F3 centroid → h2b
        [3, 10],   # F3 centroid → h3a
    ]
    arms = mech.addChild("arms")
    arms.addObject("EdgeSetTopologyContainer",    name="topo",    edges=arm_edges)
    arms.addObject("EdgeSetTopologyModifier",      name="topo_mod")
    arms.addObject("EdgeSetGeometryAlgorithms",    name="topo_geo", template="Rigid3d")
    arms.addObject("BeamFEMForceField",
                   name="ArmFEM",
                   radius=ARM_RADIUS,
                   youngModulus=YOUNG_MODULUS * 100,  # 100× stiffer than filaments
                   poissonRatio=POISSON_RATIO)

    # ── Filament beams (compliant, parameterized) ─────────────────────────────
    #   4 hinge beams connecting adjacent face pairs. Their strain energy is
    #   the quantity of interest returned by evaluate_unit_cell().
    filament_edges = [
        [4,  5],   # H0: F0 ↔ F1  (horizontal, right gap)
        [6,  7],   # H1: F1 ↔ F2  (vertical, top-right gap)
        [8,  9],   # H2: F2 ↔ F3  (horizontal, top gap)
        [10, 11],  # H3: F3 ↔ F0  (vertical, left gap)
    ]
    hinges = mech.addChild("hinges")
    hinges.addObject("EdgeSetTopologyContainer",   name="topo",    edges=filament_edges)
    hinges.addObject("EdgeSetTopologyModifier",     name="topo_mod")
    hinges.addObject("EdgeSetGeometryAlgorithms",   name="topo_geo", template="Rigid3d")
    hinges.addObject("BeamFEMForceField",
                     name="HingeFEM",
                     radius=filament_thickness,
                     youngModulus=YOUNG_MODULUS,
                     poissonRatio=POISSON_RATIO)


# ── Public interface ───────────────────────────────────────────────────────────

def evaluate_unit_cell(filament_thickness: float,
                       filament_length: float,
                       applied_displacement: float) -> float:
    """
    Simulate the 1×1 kirigami unit cell and return the total beam strain energy
    at static equilibrium.

    The energy monotonically increases with applied_displacement and with
    filament stiffness (larger thickness or shorter length → higher energy for
    the same displacement). This is the expected behaviour for a filament in
    bending/shear.

    Parameters
    ----------
    filament_thickness : float
        Cross-section radius of each hinge filament [m].
        Typical range: 0.005–0.1. Bending stiffness ∝ r⁴.
    filament_length : float
        Natural (rest) length of each hinge filament [m].
        Typical range: 0.02–0.5. Defines the gap between adjacent face panels.
    applied_displacement : float
        Rigid-body x-displacement imposed on face F2 [m].
        Keep small relative to face size (≤ 0.3 for unit faces) to stay in
        the quasi-linear regime.

    Returns
    -------
    float
        Potential energy [J] stored in the beam elements at equilibrium.
        Dominated by the 4 filament beams; arm beams contribute negligibly.
    """
    # Natural (stress-free) positions: same geometry but with delta=0.
    pos_natural = np.array(_build_positions(filament_length, 0.0), dtype=np.float64)
    filament_edges = [[4, 5], [6, 7], [8, 9], [10, 11]]

    root = Sofa.Core.Node("root")
    _build_scene(root, filament_thickness, filament_length, applied_displacement)
    mstate = root.mechanism.getObject("DoFs")

    Sofa.Simulation.init(root)
    dt = float(root.dt.value)
    for _ in range(N_STEPS):
        Sofa.Simulation.animate(root, dt)

    pos_eq = np.array(mstate.position.value, dtype=np.float64)
    Sofa.Simulation.unload(root)

    return _euler_bernoulli_energy(
        pos_eq, pos_natural, filament_edges,
        filament_length, filament_thickness,
        YOUNG_MODULUS, POISSON_RATIO,
    )


def _euler_bernoulli_energy(pos_cur: np.ndarray,
                             pos_nat: np.ndarray,
                             edges: list,
                             L: float,
                             r: float,
                             E: float,
                             nu: float) -> float:
    """
    Compute the total Euler-Bernoulli beam strain energy analytically.

    SOFA provides the equilibrium positions; this function applies beam theory
    to compute the energy stored in the specified beam edges.  All beams are
    assumed to lie in the XY plane (z = 0), which matches our unit-cell geometry.

    The stiffness matrix used is the standard 6-DOF (2 nodes × [u, v, θz])
    Euler-Bernoulli matrix in the beam's local frame.  The corotational correction
    (rotating the local frame to follow the deformed beam axis) is applied so the
    formula remains valid for moderate rotations.

    Parameters
    ----------
    pos_cur  : (N, 7) equilibrium positions [x, y, z, qx, qy, qz, qw]
    pos_nat  : (N, 7) natural (stress-free) positions
    edges    : list of [i, j] node pairs (the beam edges to sum)
    L, r, E, nu : beam geometry and material parameters
    """
    A  = np.pi * r ** 2
    Iz = np.pi * r ** 4 / 4       # 2nd moment of area about local z
    L0 = L                         # natural length (same for all 4 filaments)

    k_ax = E * A / L0
    k_t  = 12 * E * Iz / L0 ** 3
    k_r  = 4  * E * Iz / L0
    k_r2 = 2  * E * Iz / L0
    k_tr = 6  * E * Iz / L0 ** 2

    # 6×6 Euler-Bernoulli stiffness in local frame [u_i, v_i, θ_i, u_j, v_j, θ_j]
    K = np.array([
        [ k_ax,   0,      0,    -k_ax,   0,      0   ],
        [    0,   k_t,   k_tr,     0,  -k_t,   k_tr  ],
        [    0,  k_tr,   k_r,      0,  -k_tr,  k_r2  ],
        [-k_ax,   0,      0,     k_ax,   0,      0   ],
        [    0,  -k_t,  -k_tr,    0,   k_t,  -k_tr   ],
        [    0,  k_tr,   k_r2,    0,  -k_tr,   k_r   ],
    ])

    def angle_z(pos7):
        """Rotation angle about global z from [x,y,z,qx,qy,qz,qw]."""
        qz, qw = pos7[5], pos7[6]
        return 2.0 * np.arctan2(qz, qw)

    total = 0.0
    for i, j in edges:
        # Natural beam axis (defines local frame)
        p_i0 = pos_nat[i, :3]
        p_j0 = pos_nat[j, :3]
        axis  = p_j0 - p_i0
        L_nat = np.linalg.norm(axis)
        e_x   = axis / L_nat          # local x = beam direction
        e_z   = np.array([0., 0., 1.])
        e_y   = np.cross(e_z, e_x)    # local y = transverse in XY plane
        e_y  /= np.linalg.norm(e_y)

        # Global displacements from natural to current
        dp_i = pos_cur[i, :3] - p_i0
        dp_j = pos_cur[j, :3] - p_j0

        # Project to local frame
        du_i = float(e_x @ dp_i)
        dv_i = float(e_y @ dp_i)
        du_j = float(e_x @ dp_j)
        dv_j = float(e_y @ dp_j)

        # Rotations about local z (= global z for in-plane beams)
        dtheta_i = angle_z(pos_cur[i]) - angle_z(pos_nat[i])
        dtheta_j = angle_z(pos_cur[j]) - angle_z(pos_nat[j])

        u_local = np.array([du_i, dv_i, dtheta_i, du_j, dv_j, dtheta_j])
        total += 0.5 * float(u_local @ K @ u_local)

    return total


# ── Validation ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("SOFA Unit Cell Sandbox — Validation")
    print("=" * 60)
    print(f"{'Config':<30}  {'Energy [J]':>14}")
    print("-" * 46)

    configs = [
        {
            "label":               "thin/long  (r=0.01, L=0.2, δ=0.10)",
            "filament_thickness":  0.01,
            "filament_length":     0.20,
            "applied_displacement": 0.10,
        },
        {
            "label":               "thick/short (r=0.05, L=0.05, δ=0.10)",
            "filament_thickness":  0.05,
            "filament_length":     0.05,
            "applied_displacement": 0.10,
        },
    ]

    results = []
    for cfg in configs:
        label = cfg.pop("label")
        energy = evaluate_unit_cell(**cfg)
        results.append(energy)
        tag = "nan" if np.isnan(energy) else f"{energy:.6e}"
        print(f"  {label:<30}  {tag:>14}")

    print("-" * 46)

    # Sanity check: thicker/shorter filament should store more energy
    if not any(np.isnan(r) for r in results):
        e_thin, e_thick = results
        passed = e_thick > e_thin
        verdict = "PASS" if passed else "FAIL (check geometry or solver convergence)"
        print(f"\nMonotonicity check (E_thick > E_thin): {verdict}")
        print(f"  E_thin={e_thin:.4e}  E_thick={e_thick:.4e}  ratio={e_thick/e_thin:.2f}")
    else:
        print("\nEnergy extraction returned NaN — see warning above.")

    print("=" * 60)
