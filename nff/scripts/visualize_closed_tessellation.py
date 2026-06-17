"""Visual validation of the closed-state (flat-sheet) kirigami builder.

Produces two layered figures so the high-confidence LES output can be validated
independently of the riskier panel/hinge bookkeeping:

    Layer 1 — cut-network diagram: each cut segment x_ij -> x'_ij straight from
              the LES, coloured by type, boundary points marked.
    Layer 2 — assembled tessellation: panels + corner hinges via plot_tessellation.

It also round-trips the result through CentroidalState.from_tessellation and
asserts the centroidal representation reconstructs the same panel vertices.

Run:
    JAX_PLATFORMS=cpu conda run -n kgnn_mac \
        python nff/scripts/visualize_closed_tessellation.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from nff.topology.closed_builder import (
    build_topology_matrix,
    build_square_boundary_points,
    solve_cut_vertices,
    build_closed_tessellation,
)
from nff.utils.visualization import plot_tessellation
from nff.stages.state import CentroidalState
from nff.stages.geometry import reconstruct_vertices

OUTPUT_DIR = os.path.join("data", "outputs")
TYPE_COLORS = {1: "#F58025", 2: "#457B9D"}      # horizontal=orange, vertical=blue


def plot_cut_network(T, boundary_points, cut_vertices, ax, title=None):
    """Draw every cut segment x_ij -> x'_ij with type colouring and labels."""
    M, N = T.shape
    for i in range(M):
        for j in range(N):
            x, xp = cut_vertices[(i, j)]
            ctype = abs(int(T[i, j]))
            color = TYPE_COLORS[ctype]
            ax.plot([x[0], xp[0]], [x[1], xp[1]], color=color, lw=2.5, zorder=3)
            ax.scatter([x[0], xp[0]], [x[1], xp[1]], color=color, s=18, zorder=4)
            mid = 0.5 * (x + xp)
            ax.text(mid[0], mid[1], f"{i},{j}", fontsize=6, color="#333333",
                    ha="center", va="center", zorder=5)
    if boundary_points:
        bp = np.array(list(boundary_points.values()))
        ax.scatter(bp[:, 0], bp[:, 1], facecolors="none", edgecolors="#6C757D",
                   s=90, lw=1.5, zorder=2, label="boundary points")
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title)


def tiling_check(tess) -> dict:
    """Verify panels tile without overlaps and cover the sheet (Shapely).

    Returns a dict with the worst pairwise overlap area, the coverage ratio
    (union area / bounding-box area), and the summed panel area.
    """
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union

    polys = [Polygon(tess.vertices[f.vertex_indices]) for f in tess.faces]
    sum_area = sum(p.area for p in polys)

    max_overlap = 0.0
    for a in range(len(polys)):
        for b in range(a + 1, len(polys)):
            inter = polys[a].intersection(polys[b]).area
            max_overlap = max(max_overlap, inter)

    union = unary_union(polys)
    bounds = box(*union.bounds)
    coverage = union.area / bounds.area if bounds.area > 0 else 0.0
    return {"max_overlap": max_overlap, "coverage": coverage,
            "sum_area": sum_area, "union_area": union.area}


def roundtrip_check(tessellation) -> float:
    """Build a CentroidalState and return the max vertex reconstruction error."""
    state = CentroidalState.from_tessellation(tessellation)
    recon = np.asarray(reconstruct_vertices(state.face_centroids, state.centroid_node_vectors))
    # Compare against the panel vertices grouped by face.
    max_err = 0.0
    for f_id, face in enumerate(tessellation.faces):
        original = tessellation.vertices[face.vertex_indices]
        max_err = max(max_err, float(np.max(np.abs(recon[f_id, :len(original)] - original))))
    return max_err


def run_case(M, N, r, spacing=1.0, tag="case"):
    """Build, visualize, and validate one configuration."""
    print(f"\n=== {tag}: M={M} N={N} r={r} ===")
    T = build_topology_matrix(M, N)
    boundary_points = build_square_boundary_points(T, spacing=spacing)
    cut_vertices = solve_cut_vertices(T, boundary_points, r)

    tess = build_closed_tessellation(M, N, boundary_points=boundary_points, r=r, spacing=spacing)
    n_faces = len(tess.faces)
    n_hinges = len(tess.hinges)
    n_voids = len(tess.voids)
    print(f"panels={n_faces}  hinges={n_hinges}  voids={n_voids}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor="white")
    plot_cut_network(T, boundary_points, cut_vertices, axes[0],
                     title=f"Layer 1 — cut network ({tag})")
    plot_tessellation(tess, ax=axes[1], show_target=False,
                      show_face_indices=True, show_hinge_indices=True,
                      title=f"Layer 2 — panels + hinges ({tag})")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"closed_tessellation_{tag}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")

    # Clean presentation render: orange panels, slits visible, no labels/axes.
    fig_c, ax_c = plt.subplots(figsize=(8, 8), facecolor="white")
    plot_tessellation(tess, ax=ax_c, show_target=False, show_hinges=False,
                      show_face_indices=False, show_hinge_indices=False,
                      color_faces="#F58025")
    ax_c.set_aspect("equal")
    ax_c.axis("off")
    clean_path = os.path.join(OUTPUT_DIR, f"closed_tessellation_{tag}_clean.png")
    fig_c.savefig(clean_path, dpi=200, bbox_inches="tight")
    plt.close(fig_c)
    print(f"saved {clean_path}")

    chk = tiling_check(tess)
    print(f"tiling: max pairwise overlap={chk['max_overlap']:.3e}  "
          f"coverage={chk['coverage']:.4f}  "
          f"sum_panel_area={chk['sum_area']:.4f}  union={chk['union_area']:.4f}")

    try:
        err = roundtrip_check(tess)
        print(f"CentroidalState round-trip max error: {err:.2e}")
        # JAX defaults to float32 here (x64 is only forced inside train.py), so
        # the round-trip floor is ~1e-6, not machine-zero.
        assert err < 1e-5, "round-trip reconstruction mismatch"
    except Exception as exc:                       # noqa: BLE001 — diagnostic script
        print(f"round-trip check FAILED: {exc}")


def main():
    # M, N are PANEL counts (cut grid is (M+1)x(N+1)). r=0.5 collapses cuts to
    # points (panels meet at points); r<0.5 gives the flat tiled sheet with
    # zero-area slits. 4x4 r=0.45 reproduces Fig. 2 of Dang et al.
    run_case(M=3, N=3, r=0.4, tag="3x3_r040")
    run_case(M=4, N=4, r=0.45, tag="4x4_r045_paper")
    run_case(M=6, N=6, r=0.4, tag="6x6_r040")


if __name__ == "__main__":
    main()
