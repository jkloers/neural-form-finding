"""Mesh a single-hinge RVE with gmsh.

Builds the 2D domain from ``geometry.py``, extrudes it to the sheet thickness with
a fixed number of prism layers (so the through-thickness resolution is controlled
directly — important for out-of-plane bending/buckling), grades the in-plane mesh
finer at the ligament neck, and writes a ``.msh`` with physical groups the FEM
reads: surfaces ``rigid_A`` / ``rigid_B`` / ``free`` and volume ``hinge``.

Runs under any env with the gmsh Python API (kgnn_mac or fenicsx).
"""

import numpy as np

from nff.rve.geometry import RVEParams, build_rve_domain, boundary_tag


def build_rve_mesh(p: RVEParams, path: str, n_through: int = 3,
                   lc_min: float = None, lc_max: float = None, verbose: bool = False) -> dict:
    """Mesh the RVE and write ``path`` (.msh). Returns mesh statistics.

    Args:
        p: RVE parameters [mm].
        path: output .msh path.
        n_through: prism layers through the thickness (>= 2-3 for bending).
        lc_min, lc_max: in-plane element size at the neck / far field [mm].
    """
    import gmsh

    lc_min = lc_min if lc_min is not None else max(p.w_c / 2.0, p.w_lig / 12.0)
    lc_max = lc_max if lc_max is not None else p.r_win / 6.0

    dom = build_rve_domain(p)
    coords = list(dom.exterior.coords)[:-1]                       # unique, CCW
    n = len(coords)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.model.add("rve")

        pts = [gmsh.model.geo.addPoint(x, y, 0.0, lc_max) for (x, y) in coords]
        lines, slit_lines = [], []
        for i in range(n):
            a, b = coords[i], coords[(i + 1) % n]
            ln = gmsh.model.geo.addLine(pts[i], pts[(i + 1) % n])
            lines.append(ln)
            mx, my = 0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])
            if boundary_tag(mx, my, p) == "free" and my < -1e-6:  # the main-cut slit (not the top)
                slit_lines.append(ln)
        loop = gmsh.model.geo.addCurveLoop(lines)
        surf = gmsh.model.geo.addPlaneSurface([loop])
        ext = gmsh.model.geo.extrude([(2, surf)], 0.0, 0.0, p.thickness,
                                     numElements=[n_through], recombine=False)
        gmsh.model.geo.synchronize()

        top = next(e for e in ext if e[0] == 2)                   # top cap surface
        vol = next(e for e in ext if e[0] == 3)

        # in-plane grading: fine near the ligament neck (the slit), coarse far away
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "CurvesList", slit_lines)
        gmsh.model.mesh.field.setNumber(1, "Sampling", 200)
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

        # tag boundary surfaces: flat caps (z const) are free; sides by (x, y)
        groups = {"rigid_A": [], "rigid_B": [], "free": []}
        for (d, s) in gmsh.model.getBoundary([vol], oriented=False):
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, s)
            if (zmax - zmin) < 1e-6:                               # flat cap (z=0 or z=t) -> free
                groups["free"].append(s)
            else:
                groups[boundary_tag(0.5*(xmin+xmax), 0.5*(ymin+ymax), p)].append(s)
        for name, surfs in groups.items():
            gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, surfs), name)
        gmsh.model.setPhysicalName(3, gmsh.model.addPhysicalGroup(3, [vol[1]]), "hinge")

        gmsh.model.mesh.generate(3)
        gmsh.write(path)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        zc = np.asarray(node_coords, dtype=float).reshape(-1, 3)[:, 2]
        n_z_levels = len(set(np.round(zc, 6)))
        etypes, etags, _ = gmsh.model.mesh.getElements(dim=3)
        n_cells = int(sum(len(t) for t in etags))
        cell_names = [gmsh.model.mesh.getElementProperties(int(et))[0] for et in etypes]
        stats = {
            "n_nodes": int(len(node_tags)),
            "n_cells": n_cells,
            "cell_types": cell_names,
            "n_through": n_through,
            "n_z_levels": n_z_levels,            # == n_through + 1 (structured layers)
            "lc_min": lc_min, "lc_max": lc_max,
            "groups": {k: len(v) for k, v in groups.items()},
            "path": path,
        }
        return stats
    finally:
        gmsh.finalize()
