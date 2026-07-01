"""Probe 1 — the element / mesh gate for the RVE.

A slender cantilever plate (L x b x t) is clamped at x=0 and loaded transversely
(out-of-plane, -z) at x=L. We solve linear elasticity with the SAME element the RVE
uses (linear tets, a fixed number of structured layers through the thickness) and
compare the tip deflection to Euler-Bernoulli theory

    delta = P L^3 / (3 E I),   I = b t^3 / 12.

If the FEM deflection matches theory as the through-thickness layer count grows, the
element is adequate for the thin-hinge bending/buckling the surrogate must capture;
if the FEM stays much too stiff (shear locking), we need shells or higher order.

Run:  conda run -n fenicsx python nff/rve/probe_bending.py
"""

import os

import numpy as np
from mpi4py import MPI


# ── geometry / mesh ────────────────────────────────────────────────────────────

def build_plate_msh(path, L, b, t, n_through, lc):
    """Cantilever plate mesh (extruded box, n_through structured layers).

    Physical groups: volume 'plate' (1), 'clamp' (x=0, id 10), 'load' (x=L, id 20).
    """
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("plate")
        pts = [gmsh.model.geo.addPoint(x, y, 0.0, lc) for (x, y) in
               [(0, 0), (L, 0), (L, b), (0, b)]]
        lines = [gmsh.model.geo.addLine(pts[i], pts[(i + 1) % 4]) for i in range(4)]
        surf = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(lines)])
        ext = gmsh.model.geo.extrude([(2, surf)], 0, 0, t, numElements=[n_through])
        gmsh.model.geo.synchronize()
        vol = next(e for e in ext if e[0] == 3)
        clamp, load = [], []
        for (d, s) in gmsh.model.getBoundary([vol], oriented=False):
            xmin, _, _, xmax, _, _ = gmsh.model.getBoundingBox(2, s)
            if abs(0.5 * (xmin + xmax)) < 1e-6:
                clamp.append(s)
            elif abs(0.5 * (xmin + xmax) - L) < 1e-6:
                load.append(s)
        gmsh.model.setPhysicalName(3, gmsh.model.addPhysicalGroup(3, [vol[1]], 1), "plate")
        gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, clamp, 10), "clamp")
        gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, load, 20), "load")
        gmsh.model.mesh.generate(3)
        gmsh.write(path)
    finally:
        gmsh.finalize()


# ── FEM ────────────────────────────────────────────────────────────────────────

def solve_cantilever(msh_path, E, nu, P, L, b, t, degree=1):
    """Linear-elastic cantilever; returns (tip_deflection_mm, xs, uz_centerline)."""
    import ufl
    from dolfinx import fem, default_scalar_type, geometry
    from dolfinx.fem.petsc import LinearProblem
    from dolfinx.io import gmsh as dgmsh

    md = dgmsh.read_from_msh(msh_path, MPI.COMM_WORLD, gdim=3)
    domain, ft = md.mesh, md.facet_tags

    V = fem.functionspace(domain, ("Lagrange", degree, (3,)))
    mu = E / (2 * (1 + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

    def eps(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return lmbda * ufl.tr(eps(u)) * ufl.Identity(3) + 2 * mu * eps(u)

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=ft)
    trac = fem.Constant(domain, np.array([0.0, 0.0, -P / (b * t)], dtype=default_scalar_type))
    a = ufl.inner(sigma(u), eps(v)) * ufl.dx
    Lf = ufl.inner(trac, v) * ds(20)

    tdim = domain.topology.dim
    clamp_facets = ft.find(10)
    dofs = fem.locate_dofs_topological(V, tdim - 1, clamp_facets)
    bc = fem.dirichletbc(fem.Constant(domain, np.zeros(3, dtype=default_scalar_type)), dofs, V)

    problem = LinearProblem(a, Lf, bcs=[bc], petsc_options_prefix="probe_",
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()
    if not isinstance(uh, fem.Function):
        uh = uh[0]                                   # 0.11 may return (u, converged, its)

    # centerline u_z(x) at (x, b/2, t/2)
    xs = np.linspace(0.0, L, 40)
    pts = np.array([[x, b / 2, t / 2] for x in xs])
    tree = geometry.bb_tree(domain, tdim)
    cand = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cand, pts)
    cells, keep = [], []
    for i in range(len(pts)):
        c = colliding.links(i)
        if len(c) > 0:
            cells.append(c[0]); keep.append(i)
    vals = uh.eval(pts[keep], cells)
    uz = np.full(len(xs), np.nan); uz[keep] = vals[:, 2]
    return float(-uz[keep][-1]), xs, -uz


def analytical_tip(P, L, b, t, E):
    I = b * t ** 3 / 12.0
    return P * L ** 3 / (3.0 * E * I)


# ── driver ─────────────────────────────────────────────────────────────────────

def main(out="data/outputs/probe1_bending.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    E, nu = 210_000.0, 0.30           # MPa
    L, b, t = 20.0, 4.0, 1.0          # mm  (L/t = 20, slender)
    P = 10.0                          # N
    lc = 1.0                          # in-plane element size [mm]
    delta_ref = analytical_tip(P, L, b, t, E)

    layers = [1, 2, 3, 4, 6]
    tmp = "/tmp/_probe_plate.msh"
    results = {1: [], 2: []}                          # degree -> [ratio per layer]
    profile = {}                                     # degree -> (xs, uz) at 3 layers
    for nz in layers:
        build_plate_msh(tmp, L, b, t, nz, lc)
        for deg in (1, 2):
            tip, xs, uz = solve_cantilever(tmp, E, nu, P, L, b, t, degree=deg)
            results[deg].append(tip / delta_ref)
            print(f"  P{deg}  n_through={nz}: tip={tip:.4f} mm   FEM/theory={tip/delta_ref:.3f}")
            if nz == 3:
                profile[deg] = (xs, uz)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))
    axA.axhline(1.0, color="#6C757D", ls="--", lw=1, label="Euler-Bernoulli")
    axA.plot(layers, results[1], "-o", color="#D62828", lw=2, label="P1 (linear tets)")
    axA.plot(layers, results[2], "-o", color="#2A9D8F", lw=2, label="P2 (quadratic tets)")
    axA.set_xlabel("through-thickness layers"); axA.set_ylabel("FEM tip / theory")
    axA.set_title("bending-stiffness convergence"); axA.set_ylim(0, 1.2); axA.legend()

    for deg, col in ((1, "#D62828"), (2, "#2A9D8F")):
        xs, uz = profile[deg]
        axB.plot(xs, uz, "-o", color=col, lw=2, ms=3, label=f"P{deg} FEM (3 layers)")
    xs = profile[1][0]
    axB.plot(xs, delta_ref * (3 * (xs/L)**2 - (xs/L)**3) / 2, "--", color="#6C757D",
             lw=1.5, label="Euler-Bernoulli")
    axB.set_xlabel("x [mm]"); axB.set_ylabel("deflection $-u_z$ [mm]")
    axB.set_title("cantilever deflection profile (3 layers)"); axB.legend()

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    r1, r2 = results[1][-1], results[2][-1]
    print(f"\n  theory tip = {delta_ref:.4f} mm")
    print(f"  P1 finest FEM/theory = {r1:.3f}  (LOCKS)" if r1 < 0.85 else f"  P1 = {r1:.3f}")
    print(f"  P2 finest FEM/theory = {r2:.3f}  -> {'PASS' if r2 > 0.9 else 'CHECK'}")
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
