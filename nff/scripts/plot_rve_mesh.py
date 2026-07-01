"""Visualise a single-hinge RVE mesh: in-plane grading + 3D mesh coloured by the
boundary tags (rigid_A / rigid_B / free). Saves to data/outputs/rve_mesh.png.

Run: JAX_PLATFORMS=cpu conda run -n kgnn_mac python nff/scripts/plot_rve_mesh.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from nff.rve.geometry import RVEParams, build_rve_domain, classify_boundary
from nff.rve.mesh import build_rve_mesh

RED, BLUE, TEAL, STEEL = "#D62828", "#1f6feb", "#2A9D8F", "#C6CAD1"


def _extract(path):
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(path)
        ntags, ncoords, _ = gmsh.model.mesh.getNodes()
        X = np.asarray(ncoords, float).reshape(-1, 3)
        idx = {int(t): i for i, t in enumerate(ntags)}
        groups = {}
        for (dim, ptag) in gmsh.model.getPhysicalGroups(2):
            name = gmsh.model.getPhysicalName(dim, ptag)
            tris = []
            for s in gmsh.model.getEntitiesForPhysicalGroup(dim, ptag):
                etypes, _, enodes = gmsh.model.mesh.getElements(2, s)
                for et, en in zip(etypes, enodes):
                    if int(et) == 2:                                  # 3-node triangle
                        for tri in np.asarray(en, int).reshape(-1, 3):
                            tris.append([idx[int(n)] for n in tri])
            groups[name] = np.asarray(tris, int) if tris else np.zeros((0, 3), int)
        return X, groups
    finally:
        gmsh.finalize()


def main(out="data/outputs/rve_mesh.png"):
    p = RVEParams()                       # rounded Saint-Venant half-disk, coherent with the viz
    path = "/tmp/_rve_view.msh"
    stats = build_rve_mesh(p, path, n_through=3)
    X, groups = _extract(path)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 6.5))

    # ── in-plane grading: the top-cap triangles (all nodes at z = t) ──
    tmax = X[:, 2].max()
    cap = np.array([tri for tri in groups["free"]
                    if np.all(np.abs(X[tri, 2] - tmax) < 1e-6)])
    axA.triplot(Triangulation(X[:, 0], X[:, 1], cap), color="#333333", lw=0.4)
    axA.set_aspect("equal"); axA.set_xlabel("mm"); axA.set_ylabel("mm")
    axA.set_title(f"in-plane mesh, graded at the neck — {stats['n_cells']} tets, "
                  f"{stats['n_z_levels']-1} layers through t")

    # ── boundary tags on the rounded RVE (2D plan) ──
    dom = build_rve_domain(p)
    axB.fill(*dom.exterior.xy, facecolor="#EDEEF0", edgecolor="none", zorder=0)
    cols = {"rigid_A": RED, "rigid_B": BLUE, "free": TEAL}
    for tag, segs in classify_boundary(dom, p).items():
        for (a, b) in segs:
            axB.plot([a[0], b[0]], [a[1], b[1]], color=cols[tag], lw=3.5,
                     solid_capstyle="round", zorder=3)
    for tag, c in [("rigid_A (tile-A handle)", RED), ("rigid_B (tile-B handle)", BLUE),
                   ("free (secondary + main cut)", TEAL)]:
        axB.plot([], [], color=c, lw=4, label=tag)
    axB.set_aspect("equal"); axB.set_xlabel("mm"); axB.set_ylabel("mm")
    axB.legend(loc="lower center", fontsize=9)
    axB.set_title("boundary tags — imposed (a, s, θ) on the two arc handles")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {stats['n_nodes']} nodes, {stats['n_cells']} tets  ->  saved {out}")


if __name__ == "__main__":
    main()
