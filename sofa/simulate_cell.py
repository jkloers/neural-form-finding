"""
simulate_cell.py — Phase 3a: full 3D volumetric hinge FEM.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROLE IN THE PIPELINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  JAX pipeline (kgnn_mac env, Python 3.10)
       │  subprocess / Tesseract HTTP API
       ▼
  tesseract/tesseract_api.py   ← Tesseract wrapper
       │  imports
       ▼
  sofa/simulate_cell.py        ← THIS FILE — physics only
       │  calls
       ▼
  SOFA v25.12 (Linux in Docker, macOS ARM64 locally)

Extension path:
  Phase 2 (superseded) : 12-node Rigid3d + BeamFEM proxy
  Phase 3a (current)   : full 3D hexahedral FEM of hinge strips
  Phase 3b             : face-face contact (FreeMotionAnimationLoop)
  Phase 3c             : plasticity activated (d_plasticYieldThreshold)
  Phase 4              : Tesseract energy as training reward for NFF

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GEOMETRY  (unit_RDQK_0, 1×1, physical metres)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  a = face_size, w = hinge_arm_width, L = hinge_fold_length, t = sheet_thickness.

  Face centroids (natural):
    F0 (fixed) : (a/2,         a/2)
    F1 (free)  : (3a/2+w,      a/2)      equilibrium ≈ F0 + (δ/2, 0)
    F2 (loaded): (3a/2+w,      3a/2+w)   displaced by δ in x
    F3 (free)  : (a/2,         3a/2+w)   equilibrium ≈ F0 + (δ/2, 0)

  Hinge strips (rectangular box a×w×L×t):
    H0 (F0↔F1): centre (a+w/2,       a/2),        arm in x, fold in y
    H1 (F1↔F2): centre (3a/2+w,      a+w/2),      arm in y, fold in x
    H2 (F3↔F2): centre (a+w/2,       3a/2+w),     arm in x, fold in y
    H3 (F0↔F3): centre (a/2,         a+w/2),      arm in y, fold in x

  Mesh: HINGE_NX × HINGE_NY × HINGE_NZ structured hexahedral nodes.
  FEM : HexahedronFEMForceField (large, corotational) — avoids Delaunay topology issues.

  Boundary conditions (incremental loading via LinearMovementConstraint):
    Face-A nodes → move by face_a_fraction × δ in x over N_STEPS.
    Face-B nodes → move by face_b_fraction × δ in x over N_STEPS.
    Interior nodes → free; z_perturbation seeds out-of-plane buckling.

  Ring symmetry: by equal stiffness, F1 and F3 displace by δ/2 in x.
  → face_a_fraction: 0.0 (F0), 0.5 (F1/F3), 1.0 (F2).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LOCAL SETUP (macOS ARM64)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    ./sofa/run_sofa.sh sofa/simulate_cell.py
"""

import sys
import threading
import warnings

import numpy as np

try:
    import Sofa
    import Sofa.Core
    import Sofa.Simulation
except ImportError as e:
    sys.exit(
        f"Cannot import SOFA: {e}\n"
        "Run via ./sofa/run_sofa.sh which sets the required env vars."
    )

# ── Material defaults: PLA 3D-printing polymer ─────────────────────────────────
YOUNG_MODULUS  = 3.5e9   # Pa
POISSON_RATIO  = 0.36
YIELD_STRENGTH = 55e6    # Pa
DENSITY        = 1250.0  # kg/m³

# ── Geometry defaults: 20 cm tessellation, 1 mm sheet ─────────────────────────
FACE_SIZE       = 0.100  # m
SHEET_THICKNESS = 0.001  # m

# ── Hex mesh resolution (nodes per direction) ──────────────────────────────────
#    (nx-1)×(ny-1)×(nz-1) hexahedral elements per hinge
HINGE_NX = 3   # arm direction   (across gap)
HINGE_NY = 5   # fold direction  (along shared edge)
HINGE_NZ = 3   # thickness       (z)

N_STEPS = 80   # simulation steps for quasi-static convergence
DT      = 0.02 # s per step → T_total = 1.6 s

_SOFA_LOCK = threading.Lock()


# ══════════════════════════════════════════════════════════════════════════════
# Structured hexahedral mesh builder
# ══════════════════════════════════════════════════════════════════════════════

def _node_index(ix: int, iy: int, iz: int, nx: int, ny: int) -> int:
    return iz * (nx * ny) + iy * nx + ix


def _build_hex_mesh(center_xy: tuple,
                     arm_axis: int,
                     arm_width: float,
                     fold_length: float,
                     sheet_thickness: float,
                     nx: int = HINGE_NX,
                     ny: int = HINGE_NY,
                     nz: int = HINGE_NZ) -> tuple:
    """
    Build a structured hexahedral mesh for one rectangular hinge strip.

    The hinge strip is centred at `center_xy`. The arm direction (perpendicular
    to the fold axis) is either x (arm_axis=0) or y (arm_axis=1). Thickness
    is always the z-axis.

    Returns
    -------
    nodes       : (N, 3) float64  — natural node positions (stress-free)
    hexes       : (H, 8) int32    — hex connectivity, SOFA/VTK node ordering
    face_a_mask : (N,)  bool      — nodes on face-A side (negative arm)
    face_b_mask : (N,)  bool      — nodes on face-B side (positive arm)
    """
    cx, cy = center_xy

    arm_c   = np.linspace(-arm_width / 2,     arm_width / 2,     nx)
    fold_c  = np.linspace(-fold_length / 2,   fold_length / 2,   ny)
    thick_c = np.linspace(-sheet_thickness / 2, sheet_thickness / 2, nz)

    nodes, fa_mask, fb_mask = [], [], []
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                a, f, t = arm_c[ix], fold_c[iy], thick_c[iz]
                if arm_axis == 0:
                    x, y = cx + a, cy + f
                else:
                    x, y = cx + f, cy + a
                nodes.append([x, y, t])
                fa_mask.append(ix == 0)
                fb_mask.append(ix == nx - 1)

    nodes   = np.array(nodes,   dtype=np.float64)
    fa_mask = np.array(fa_mask, dtype=bool)
    fb_mask = np.array(fb_mask, dtype=bool)

    # Hexahedral connectivity — SOFA/VTK ordering: bottom face CCW → top face CCW
    #
    #   7─────6         iz+1
    #  /|    /|
    # 4─────5 |          z
    # | 3───|─2          │
    # |/    |/           └──x (arm)
    # 0─────1      iy=fold direction
    #
    hexes = []
    for iz in range(nz - 1):
        for iy in range(ny - 1):
            for ix in range(nx - 1):
                n = _node_index
                hexes.append([
                    n(ix,   iy,   iz,   nx, ny),
                    n(ix+1, iy,   iz,   nx, ny),
                    n(ix+1, iy+1, iz,   nx, ny),
                    n(ix,   iy+1, iz,   nx, ny),
                    n(ix,   iy,   iz+1, nx, ny),
                    n(ix+1, iy,   iz+1, nx, ny),
                    n(ix+1, iy+1, iz+1, nx, ny),
                    n(ix,   iy+1, iz+1, nx, ny),
                ])

    hexes = np.array(hexes, dtype=np.int32)
    return nodes, hexes, fa_mask, fb_mask


def _hex_to_5tets(hexes: np.ndarray) -> np.ndarray:
    """
    Decompose hexahedra into 5 tetrahedra each, for energy/stress computation.

    The decomposition is consistent across adjacent hexes (same diagonal choice
    on every shared face), so there are no volumetric gaps or overlaps.

    Verified positive orientation and total volume = 1 for a unit cube:
      4 tets × V=1/6 + 1 tet × V=2/6 = 1.0

    Used ONLY for the analytical SvK energy and von Mises computation;
    the SOFA simulation uses the hexahedral topology directly.
    """
    tets = []
    for h in hexes:
        n0, n1, n2, n3, n4, n5, n6, n7 = h
        tets.extend([
            [n0, n1, n2, n5],   # V = 1/6
            [n0, n2, n3, n7],   # V = 1/6
            [n0, n5, n7, n4],   # V = 1/6
            [n2, n5, n6, n7],   # V = 1/6
            [n0, n5, n2, n7],   # V = 2/6  (centre tet)
        ])
    return np.array(tets, dtype=np.int32)


# ══════════════════════════════════════════════════════════════════════════════
# Hinge topology for unit_RDQK_0 (1×1 tessellation)
# ══════════════════════════════════════════════════════════════════════════════

def _get_hinge_configs(face_size: float,
                        arm_width: float,
                        fold_length: float) -> list:
    """
    Return descriptors for all 4 hinges in the 1×1 unit_RDQK_0 pattern.

    face_a/b_fraction: fraction of applied_displacement (δ, in x) carried by
    the corresponding face at ring-symmetric equilibrium (F0=0, F1=F3=0.5, F2=1).
    """
    a, w = face_size, arm_width
    return [
        dict(name="H0", center_xy=(a + w/2,       a/2),
             arm_axis=0, arm_width=arm_width, fold_length=fold_length,
             face_a_fraction=0.0, face_b_fraction=0.5),   # F0 ↔ F1
        dict(name="H1", center_xy=(a + w + a/2,   a + w/2),
             arm_axis=1, arm_width=arm_width, fold_length=fold_length,
             face_a_fraction=0.5, face_b_fraction=1.0),   # F1 ↔ F2
        dict(name="H2", center_xy=(a + w/2,       a + w + a/2),
             arm_axis=0, arm_width=arm_width, fold_length=fold_length,
             face_a_fraction=0.5, face_b_fraction=1.0),   # F3 ↔ F2
        dict(name="H3", center_xy=(a/2,            a + w/2),
             arm_axis=1, arm_width=arm_width, fold_length=fold_length,
             face_a_fraction=0.0, face_b_fraction=0.5),   # F0 ↔ F3
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Analytical energy and stress (bypasses SOFA v25.12 API gaps)
# ══════════════════════════════════════════════════════════════════════════════

def _svk_energy(pos_nat: np.ndarray,
                 pos_cur: np.ndarray,
                 tets: np.ndarray,
                 young: float,
                 nu: float) -> float:
    """
    Saint Venant-Kirchhoff total strain energy over a tetrahedral mesh.

    W = (λ/2)·(tr E)² + μ·‖E‖²_F   [J/m³],  E_total = Σ W·V_elem  [J]

    E = ½(FᵀF − I)  (Green-Lagrange strain),  F = dx·(dX)⁻¹  (deformation gradient).
    Valid for moderate strains (~20 %); equivalent to linear elasticity at small strains.
    """
    lam = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu  = young / (2.0 * (1.0 + nu))

    total = 0.0
    for tet in tets:
        dX = (pos_nat[tet[1:]] - pos_nat[tet[0]]).T   # 3×3 reference edges
        dx = (pos_cur[tet[1:]] - pos_cur[tet[0]]).T   # 3×3 deformed edges
        det_dX = np.linalg.det(dX)
        if abs(det_dX) < 1e-30:
            continue
        F   = dx @ np.linalg.inv(dX)
        E   = 0.5 * (F.T @ F - np.eye(3))
        trE = np.trace(E)
        # ‖E‖²_F = tr(E²) for symmetric E
        total += ((lam / 2.0) * trE**2 + mu * np.sum(E**2)) * abs(det_dX) / 6.0
    return total


def _vm_stress_per_tet(pos_nat: np.ndarray,
                        pos_cur: np.ndarray,
                        tets: np.ndarray,
                        young: float,
                        nu: float) -> np.ndarray:
    """
    Per-element von Mises (Cauchy) stress [Pa].

    SvK constitutive law: S = λ·tr(E)·I + 2μ·E  (2nd Piola-Kirchhoff).
    Cauchy: σ = (1/J)·F·S·Fᵀ.
    von Mises: σ_vm = √(3/2 · s:s),  s = σ − (1/3)tr(σ)·I  (deviatoric).
    """
    lam = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu  = young / (2.0 * (1.0 + nu))

    vm_list = []
    for tet in tets:
        dX = (pos_nat[tet[1:]] - pos_nat[tet[0]]).T
        dx = (pos_cur[tet[1:]] - pos_cur[tet[0]]).T
        det_dX = np.linalg.det(dX)
        if abs(det_dX) < 1e-30:
            continue
        F = dx @ np.linalg.inv(dX)
        J = np.linalg.det(F)
        if abs(J) < 1e-10:
            continue
        E     = 0.5 * (F.T @ F - np.eye(3))
        S     = lam * np.trace(E) * np.eye(3) + 2.0 * mu * E
        sigma = (F @ S @ F.T) / J
        s     = sigma - (np.trace(sigma) / 3.0) * np.eye(3)
        vm_list.append(float(np.sqrt(1.5 * np.sum(s**2))))

    return np.array(vm_list) if vm_list else np.array([0.0])


# ══════════════════════════════════════════════════════════════════════════════
# SOFA scene construction
# ══════════════════════════════════════════════════════════════════════════════

def _build_scene(root,
                  configs: list,
                  meshes: list,
                  applied_displacement: float,
                  young: float,
                  nu: float,
                  z_perturbation: float,
                  sheet_thickness: float) -> tuple:
    """
    Populate the SOFA scene with 4 hinge child nodes.

    All nodes start at natural (stress-free) positions.
    Boundary nodes are driven incrementally by LinearMovementConstraint.
    Interior nodes are free with a small z-perturbation to seed buckling.

    Uses HexahedronFEMForceField (corotational large) to avoid the duplicate-
    triangle topology errors that Delaunay-tetrahedral meshes produce.

    Returns fems, mstates (one per hinge, indexed 0-3).
    """
    root.gravity = [0.0, 0.0, 0.0]
    root.dt      = DT
    T_final      = N_STEPS * DT

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

    fems, mstates = [], []
    δ = applied_displacement

    for cfg, (nat, hexes, fa_mask, fb_mask) in zip(configs, meshes):
        fa_idx        = np.where(fa_mask)[0].tolist()
        fb_idx        = np.where(fb_mask)[0].tolist()
        interior_mask = ~fa_mask & ~fb_mask

        # Seed out-of-plane buckling on interior nodes
        pos = nat.copy()
        pos[interior_mask, 2] += z_perturbation

        hinge = root.addChild(cfg["name"])

        mstate = hinge.addObject(
            "MechanicalObject", template="Vec3d", name="DoFs",
            position=pos.tolist(),
        )

        hinge.addObject("HexahedronSetTopologyContainer",
                         name="topo", hexahedra=hexes.tolist())
        hinge.addObject("HexahedronSetTopologyModifier")
        hinge.addObject("HexahedronSetGeometryAlgorithms", template="Vec3d")

        V_hinge = cfg["arm_width"] * cfg["fold_length"] * sheet_thickness
        hinge.addObject("UniformMass", totalMass=DENSITY * V_hinge)

        ff = hinge.addObject(
            "HexahedronFEMForceField", template="Vec3d", name="FEM",
            youngModulus=young,
            poissonRatio=nu,
            method="large",    # corotational — handles large rotations correctly
        )

        # ── Boundary conditions (incremental loading) ───────────────────────
        # All nodes start at natural position. LinearMovementConstraint drives
        # face-side nodes to their target displaced positions over T_final seconds.
        # This makes the natural configuration the stress-free FEM reference.
        move_a = [cfg["face_a_fraction"] * δ, 0.0, 0.0]
        move_b = [cfg["face_b_fraction"] * δ, 0.0, 0.0]

        if cfg["face_a_fraction"] == 0.0:
            # Face A has zero displacement → simple FixedConstraint
            hinge.addObject("FixedConstraint", name="fix_A", indices=fa_idx)
        else:
            hinge.addObject("LinearMovementConstraint", name="move_A",
                             template="Vec3d", indices=fa_idx,
                             keyTimes=[0.0, T_final],
                             movements=[[0.0, 0.0, 0.0], move_a])

        hinge.addObject("LinearMovementConstraint", name="move_B",
                         template="Vec3d", indices=fb_idx,
                         keyTimes=[0.0, T_final],
                         movements=[[0.0, 0.0, 0.0], move_b])

        fems.append(ff)
        mstates.append(mstate)

    return fems, mstates


# ══════════════════════════════════════════════════════════════════════════════
# Public interface
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_unit_cell(
    hinge_arm_width:      float = 0.005,
    hinge_fold_length:    float = 0.020,
    applied_displacement: float = 0.010,
    face_size:            float = FACE_SIZE,
    sheet_thickness:      float = SHEET_THICKNESS,
    young_modulus:        float = YOUNG_MODULUS,
    poisson_ratio:        float = POISSON_RATIO,
    yield_strength:       float = YIELD_STRENGTH,
    z_perturbation:       float = 1e-4,
) -> dict:
    """
    Simulate the 1×1 kirigami unit cell and return mechanical quantities.

    Parameters
    ----------
    hinge_arm_width : float
        Physical gap between adjacent face panels — the hinge arm length [m].
        Bending stiffness ∝ 1/arm_width³.  Default: 5 mm.
    hinge_fold_length : float
        Length of the uncut hinge strip along the shared edge [m].  Default: 20 mm.
    applied_displacement : float
        Rigid x-displacement of face F2 [m].  Default: 10 mm.
    face_size : float
        Square face panel side length [m].  Default: 100 mm.
    sheet_thickness : float
        Material thickness in z [m].  Bending/buckling stiffness ∝ t³.  Default: 1 mm.
    young_modulus : float
        Young's modulus [Pa].  Default: PLA 3.5 GPa.
    poisson_ratio : float
        Poisson ratio.  Default: PLA 0.36.
    yield_strength : float
        Yield stress for first-yield fraction output [Pa].  Default: 55 MPa.
    z_perturbation : float
        Initial z-offset on interior nodes to seed buckling [m].  Default: 0.1 mm.

    Returns
    -------
    dict
        strain_energy        [J]    — total SvK elastic energy across all 4 hinges
        max_von_mises_stress [Pa]   — peak von Mises stress across all elements
        max_z_displacement   [m]    — peak |z| displacement (buckling proxy)
        first_yield_fraction  []    — max_von_mises / yield_strength  (>1 → yielded)
    """
    configs = _get_hinge_configs(face_size, hinge_arm_width, hinge_fold_length)

    # Build all 4 hex meshes — pure NumPy, outside the SOFA lock
    meshes = []
    for cfg in configs:
        nat, hexes, fa_mask, fb_mask = _build_hex_mesh(
            center_xy       = cfg["center_xy"],
            arm_axis        = cfg["arm_axis"],
            arm_width       = cfg["arm_width"],
            fold_length     = cfg["fold_length"],
            sheet_thickness = sheet_thickness,
        )
        meshes.append((nat, hexes, fa_mask, fb_mask))

    with _SOFA_LOCK:
        root = Sofa.Core.Node("root")
        fems, mstates = _build_scene(
            root, configs, meshes,
            applied_displacement, young_modulus, poisson_ratio,
            z_perturbation, sheet_thickness,
        )

        Sofa.Simulation.init(root)
        for _ in range(N_STEPS):
            Sofa.Simulation.animate(root, DT)

        cur_positions = [
            np.array(ms.position.value, dtype=np.float64)
            for ms in mstates
        ]
        Sofa.Simulation.unload(root)

    # ── Compute QoIs from equilibrium positions ─────────────────────────────
    total_energy = 0.0
    all_vm       = []
    max_z        = 0.0

    for (nat, hexes, fa_mask, fb_mask), cur in zip(meshes, cur_positions):
        tets = _hex_to_5tets(hexes)

        total_energy += _svk_energy(nat, cur, tets, young_modulus, poisson_ratio)

        vm = _vm_stress_per_tet(nat, cur, tets, young_modulus, poisson_ratio)
        all_vm.extend(vm.tolist())

        interior_mask = ~fa_mask & ~fb_mask
        z_abs = np.abs(cur[interior_mask, 2])
        if len(z_abs) > 0:
            max_z = max(max_z, float(np.max(z_abs)))

    max_vm = float(np.max(all_vm)) if all_vm else 0.0

    return {
        "strain_energy":        total_energy,
        "max_von_mises_stress": max_vm,
        "max_z_displacement":   max_z,
        "first_yield_fraction": max_vm / yield_strength,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Validation
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("SOFA Unit Cell — Phase 3a Validation  (PLA, 20 cm tessellation)")
    print("=" * 70)
    hdr = (f"{'Config':<38} {'E [J]':>9} {'σ_vm [MPa]':>11}"
           f" {'z_max [mm]':>11} {'σ/σ_y':>7}")
    print(hdr)
    print("-" * len(hdr))

    cases = [
        dict(label="baseline (w=5mm, L=20mm, t=1mm, δ=10mm)",
             hinge_arm_width=0.005, hinge_fold_length=0.020,
             applied_displacement=0.010),
        dict(label="thin sheet  (t=0.5mm)",
             hinge_arm_width=0.005, hinge_fold_length=0.020,
             applied_displacement=0.010, sheet_thickness=0.0005),
        dict(label="large δ    (δ=20mm)",
             hinge_arm_width=0.005, hinge_fold_length=0.020,
             applied_displacement=0.020),
    ]

    for case in cases:
        label = case.pop("label")
        r     = evaluate_unit_cell(**case)
        E     = r["strain_energy"]
        vm    = r["max_von_mises_stress"] / 1e6   # MPa
        z     = r["max_z_displacement"]   * 1e3   # mm
        fyf   = r["first_yield_fraction"]
        print(f"  {label:<38} {E:>9.4e} {vm:>11.2f} {z:>11.4f} {fyf:>7.3f}")

    print("-" * len(hdr))
    print("\nExpected:")
    print("  thin sheet: energy ↓ ×8 (EI ∝ t³), buckling easier (z_max ↑)")
    print("  large δ:    energy ↑ ~×4 (quadratic), stress ↑ proportionally")
    print("  σ/σ_y > 1 → filament has plastically yielded")
    print("=" * 70)
