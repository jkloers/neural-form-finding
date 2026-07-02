"""Single-hinge deployment via CalculiX — use the solver's built-in mechanics.

We do NOT hand-code constitutive models. CalculiX owns the physics: finite-strain
(NLGEOM), von-Mises plasticity (*PLASTIC), automatic increment/cut-back continuation,
and rigid-body kinematics. We only (1) build the mesh, (2) write a small text deck,
(3) run ``ccx``, and (4) read energy / out-of-plane displacement / strain from the output.

Kinematics: clamp the left face-arc; on the right face-arc prescribe the rigid rotation
of each node about the pivot (inside the ligament). A small out-of-plane crest is baked
into the mesh so the ligament buckles the right way. Faces stay coplanar (z=0 on the arcs).

Mesh: quadratic 15-node wedges (C3D15) — thin extruded sheets need prisms; split tets
invert. Runs in the ``ccx`` conda env (gmsh + shapely + calculix).

Known limitation (validation-phase TODO): the rotation is applied in a single step, so the
displacement ramp is a straight line rather than the true arc — fine for small angles, to be
replaced by a multi-step (cumulative) arc path before large-angle production runs.
"""

import os
import subprocess

import numpy as np

from nff.rve.geometry import RVEParams, build_rve_domain, boundary_tag

STEEL = dict(E=210_000.0, nu=0.30, sigma_y=235.0, Et=2100.0)      # S235, hardening E/100


def _build_mesh(p: RVEParams, pivot, imp_amp, n_through, lc_min, lc_max):
    """Second-order (C3D15 prism) RVE mesh. Returns nodes, connectivity, arc node-id sets."""
    import gmsh
    dom = build_rve_domain(p)
    coords = list(dom.exterior.coords)[:-1]
    n = len(coords)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("rve")
        pts = [gmsh.model.geo.addPoint(x, y, 0.0, lc_max) for (x, y) in coords]
        lines, slit = [], []
        for i in range(n):
            a, b = coords[i], coords[(i + 1) % n]
            ln = gmsh.model.geo.addLine(pts[i], pts[(i + 1) % n]); lines.append(ln)
            mx, my = 0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])
            if boundary_tag(mx, my, p) == "free" and my < -1e-6:
                slit.append(ln)
        surf = gmsh.model.geo.addPlaneSurface([gmsh.model.geo.addCurveLoop(lines)])
        ext = gmsh.model.geo.extrude([(2, surf)], 0, 0, p.thickness,
                                     numElements=[n_through], recombine=True)   # -> prisms
        gmsh.model.geo.synchronize()
        vol = next(e for e in ext if e[0] == 3)
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "CurvesList", slit)
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", lc_min)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", lc_max)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 0.6 * p.w_lig)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 2.5 * p.w_lig)
        gmsh.model.mesh.field.setAsBackgroundMesh(2)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        # tag arcs by bounding box on the volume boundary side-surfaces
        arc = {"rigid_A": [], "rigid_B": []}
        for (d, s) in gmsh.model.getBoundary([vol], oriented=False):
            xmn, ymn, zmn, xmx, ymx, zmx = gmsh.model.getBoundingBox(2, s)
            if (zmx - zmn) < 1e-6:
                continue
            t = boundary_tag(0.5 * (xmn + xmx), 0.5 * (ymn + ymx), p)
            if t in arc:
                arc[t].append(s)
        for name, surfs in arc.items():
            gmsh.model.addPhysicalGroup(2, surfs, {"rigid_A": 1, "rigid_B": 2}[name])
        gmsh.model.addPhysicalGroup(3, [vol[1]], 1)
        gmsh.model.mesh.generate(3)
        gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)    # 15-node prisms (C3D15)
        gmsh.model.mesh.setOrder(2)

        tags, xyz, _ = gmsh.model.mesh.getNodes()
        xyz = np.asarray(xyz).reshape(-1, 3)
        remap = {int(t): i + 1 for i, t in enumerate(tags)}       # gmsh tag -> 1-based id
        # 15-node prisms (gmsh type 18); reorder to CalculiX C3D15
        g2c = [0, 1, 2, 3, 4, 5, 6, 9, 7, 12, 14, 13, 8, 10, 11]
        conn = []
        etypes, _, enodes = gmsh.model.mesh.getElements(dim=3)
        for et, en in zip(etypes, enodes):
            if int(et) == 18:                                     # 15-node prism (C3D15)
                arr = np.asarray(en, int).reshape(-1, 15)
                conn = [[remap[int(row[i])] for i in g2c] for row in arr]
        if not conn:
            raise RuntimeError("no C3D15 prisms extracted (check element order/type)")
        # fix orientation: CalculiX needs the bottom triangle CW seen from the top
        for k, row in enumerate(conn):
            v = xyz[[row[0] - 1, row[1] - 1, row[2] - 1, row[3] - 1, row[4] - 1, row[5] - 1]]
            n = np.cross(v[1] - v[0], v[2] - v[0])
            h = v[3:6].mean(0) - v[0:3].mean(0)
            if np.dot(n, h) < 0:                                  # top/bottom swapped
                conn[k] = [row[i] for i in [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8, 12, 13, 14]]
        # out-of-plane crest imperfection along x=0 over the ligament (after orientation fix)
        Py = pivot[1]; wx, wy = 2.0 * abs(Py), 1.4 * abs(Py)
        crest = np.exp(-(xyz[:, 0] / wx) ** 2) * np.exp(-((xyz[:, 1] - Py) / wy) ** 2)
        xyz[:, 2] += imp_amp * crest
        arcA = sorted({remap[int(t)] for t in gmsh.model.mesh.getNodesForPhysicalGroup(2, 1)[0]})
        arcB = sorted({remap[int(t)] for t in gmsh.model.mesh.getNodesForPhysicalGroup(2, 2)[0]})
        return xyz, conn, arcA, arcB
    finally:
        gmsh.finalize()


def _write_inp(path, xyz, conn, arcA, arcB, pivot, mat, angle, elastic_only, n_frames):
    L = ["*NODE"]
    for i, (x, y, z) in enumerate(xyz, start=1):
        L.append(f"{i}, {x:.6e}, {y:.6e}, {z:.6e}")
    L.append("*ELEMENT, TYPE=C3D15, ELSET=EALL")
    for e, row in enumerate(conn, start=1):
        L.append(f"{e}, " + ", ".join(str(v) for v in row))
    L.append("*NSET, NSET=ARCA\n" + ",\n".join(str(v) for v in arcA))
    L.append("*MATERIAL, NAME=STEEL")
    L.append("*ELASTIC")
    L.append(f"{mat['E']:.1f}, {mat['nu']:.3f}")
    if not elastic_only:
        L.append("*PLASTIC")
        L.append(f"{mat['sigma_y']:.1f}, 0.0")
        L.append(f"{mat['sigma_y'] + mat['Et'] * 0.5:.1f}, 0.5")  # hardening slope Et
    L.append("*SOLID SECTION, ELSET=EALL, MATERIAL=STEEL")
    L.append("*STEP, NLGEOM, INC=1000")
    L.append("*STATIC")
    L.append(f"{1.0/n_frames:.4f}, 1.0, 1e-6, {1.0/n_frames:.4f}")
    L.append("*BOUNDARY")
    L.append("ARCA, 1, 3, 0.0")                                  # clamp the left face
    # right face: prescribe the rigid rotation of each arc node about the pivot (z=0)
    c, s = np.cos(angle), np.sin(angle)
    Px, Py = pivot
    for nid in arcB:
        x, y = xyz[nid - 1, 0], xyz[nid - 1, 1]
        ux = Px + c * (x - Px) - s * (y - Py) - x
        uy = Py + s * (x - Px) + c * (y - Py) - y
        L.append(f"{nid}, 1, 1, {ux:.6e}")
        L.append(f"{nid}, 2, 2, {uy:.6e}")
        L.append(f"{nid}, 3, 3, 0.0")
    L.append("*EL PRINT, ELSET=EALL, TOTALS=ONLY")
    L.append("ELSE")
    L.append("*NODE FILE")
    L.append("U")
    L.append("*EL FILE")
    L.append("E")
    L.append("*END STEP")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def deploy_ccx(p, pivot, angle_deg, material=STEEL, imp_amp=None, elastic_only=False,
               n_through=1, lc_min=None, lc_max=None, n_frames=10, workdir="/tmp/ccx_job"):
    """Run one deployment to ``angle_deg``; returns the ccx job handle + run status.

    Physics is entirely CalculiX's. Outputs live in the job files: stored energy in
    ``<job>.dat`` (*EL PRINT ELSE), displacement/strain fields in ``<job>.frd``. The
    field parsers + the batch ``deploy(geometry, angle) -> {W, uz_max, strain_max, ...}``
    wrapper are built in the validation phase.
    """
    lc_min = lc_min if lc_min is not None else max(p.w_c / 2, p.w_lig / 8)
    lc_max = lc_max if lc_max is not None else p.r_win / 4
    imp_amp = imp_amp if imp_amp is not None else 0.3 * p.thickness
    os.makedirs(workdir, exist_ok=True)
    job = os.path.join(workdir, "hinge")
    xyz, conn, arcA, arcB = _build_mesh(p, pivot, imp_amp, n_through, lc_min, lc_max)
    _write_inp(job + ".inp", xyz, conn, arcA, arcB, pivot, material,
               np.radians(angle_deg), elastic_only, n_frames)
    r = subprocess.run(["ccx", "hinge"], cwd=workdir, capture_output=True, text=True, timeout=600)
    return dict(returncode=r.returncode, stdout=r.stdout[-2000:], stderr=r.stderr[-1000:],
                n_nodes=len(xyz), n_elems=len(conn), job=job)
