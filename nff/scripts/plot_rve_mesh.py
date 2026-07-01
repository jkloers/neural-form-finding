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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from nff.rve.geometry import RVEParams
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
    p = RVEParams(w_lig=8.0, w_c=0.6, alpha_deg=90.0, rho=0.8, thickness=1.0, r_win=20.0)
    path = "/tmp/_rve_view.msh"
    stats = build_rve_mesh(p, path, n_through=3)
    X, groups = _extract(path)

    fig = plt.figure(figsize=(15, 6.5))
    axA = fig.add_subplot(1, 2, 1)
    axB = fig.add_subplot(1, 2, 2, projection="3d")

    # ── in-plane grading: the top-cap triangles (all nodes at z = t) ──
    tmax = X[:, 2].max()
    cap = np.array([tri for tri in groups["free"]
                    if np.all(np.abs(X[tri, 2] - tmax) < 1e-6)])
    tri2d = Triangulation(X[:, 0], X[:, 1], cap)
    axA.triplot(tri2d, color="#333333", lw=0.4)
    axA.set_aspect("equal"); axA.set_xlabel("mm"); axA.set_ylabel("mm")
    axA.set_title(f"in-plane mesh (graded at the neck) — {stats['n_cells']} tets")

    # ── 3D boundary mesh coloured by tag ──
    def polys(tris):
        return [X[t] for t in tris]
    # caps (z const) shown as the steel faces; slit part of 'free' shown teal
    free = groups["free"]
    is_cap = np.array([np.ptp(X[t, 2]) < 1e-6 for t in free]) if len(free) else np.zeros(0, bool)
    caps, slit = free[is_cap], free[~is_cap]
    for tris, col, a in [(caps, STEEL, 0.55), (slit, TEAL, 0.9),
                         (groups["rigid_A"], RED, 0.9), (groups["rigid_B"], BLUE, 0.9)]:
        if len(tris):
            axB.add_collection3d(Poly3DCollection(polys(tris), facecolor=col,
                                                  edgecolor="#333333", linewidths=0.2, alpha=a))
    axB.set_xlim(-p.r_win, p.r_win); axB.set_ylim(-p.r_win, p.r_win); axB.set_zlim(-p.r_win, p.r_win)
    axB.set_box_aspect((2, 2, 0.6)); axB.view_init(elev=28, azim=-60)
    axB.set_xlabel("x"); axB.set_ylabel("y"); axB.set_zlabel("z")
    for c, lb in [(RED, "rigid_A"), (BLUE, "rigid_B"), (TEAL, "free (cuts)"), (STEEL, "sheet faces")]:
        axB.plot([], [], color=c, lw=6, label=lb)
    axB.legend(loc="upper left", fontsize=9)
    axB.set_title(f"RVE boundary tags — {stats['n_z_levels']-1} layers through t")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {stats['n_nodes']} nodes, {stats['n_cells']} tets  ->  saved {out}")


if __name__ == "__main__":
    main()
