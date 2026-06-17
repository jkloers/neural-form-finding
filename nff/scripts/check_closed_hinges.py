"""Audit hinge connectivity of the closed-state RDPQK tessellation.

Checks, for an M×N panel sheet (face index = col*N + row):
  1. Coincidence  — each hinge's two vertices coincide in the flat state.
  2. Edge-adjacency — each hinge connects panels that differ by 1 in exactly one
     of (col, row); never diagonal / non-adjacent.
  3. Completeness — each edge-adjacency has exactly one hinge (no missing, no
     duplicate, no extra).
  4. Connectivity — the hinge graph spans all panels.
And overlays the hinges (pivot dots + face-centroid links) on the flat sheet.

Run:
    JAX_PLATFORMS=cpu conda run -n kgnn_mac \
        python nff/scripts/check_closed_hinges.py
"""

import os
from collections import deque, Counter

import numpy as np
import matplotlib.pyplot as plt

import jax.numpy as jnp
from collections import defaultdict
from nff.topology.closed_builder import build_closed_tessellation
from nff.topology.closed_builder_jax import (
    build_deploy_structure, solve_cut_vertices_jax, boundary_points_flat,
)
from nff.utils.visualization import plot_tessellation

OUTPUT_DIR = os.path.join("data", "outputs")


def audit(M, N, r=0.45, tol=1e-6):
    tess = build_closed_tessellation(M, N, r=r)
    verts = tess.vertices
    hinges = tess.hinges
    n_faces = len(tess.faces)

    def cr(p):
        return (p // N, p % N)

    # Expected edge-adjacencies.
    expected = set()
    for col in range(M):
        for row in range(N):
            p = col * N + row
            for dc, dr in ((1, 0), (0, 1)):
                c2, r2 = col + dc, row + dr
                if c2 < M and r2 < N:
                    expected.add(frozenset({p, c2 * N + r2}))

    gaps, pair_list, nonadj = [], [], []
    for h in hinges:
        gaps.append(float(np.linalg.norm(verts[h.vertex1] - verts[h.vertex2])))
        fp = frozenset({h.face1, h.face2})
        pair_list.append(fp)
        (c1, r1), (c2, r2) = cr(h.face1), cr(h.face2)
        if (abs(c1 - c2), abs(r1 - r2)) not in ((1, 0), (0, 1)):
            nonadj.append((h.id, h.face1, h.face2))

    pair_counts = Counter(pair_list)
    hinge_set = set(pair_list)
    missing = expected - hinge_set
    extra = hinge_set - expected
    dups = {tuple(sorted(p)): c for p, c in pair_counts.items() if c > 1}

    # Topological-pivot check: each hinge pivot must equal the position of the
    # cut endpoint the two panels actually share (not a spurious boundary x==x'
    # coincidence). This catches pivots placed on the boundary that belong inside.
    struct = build_deploy_structure(M, N)
    cp = struct["corner_pid"]
    coords = np.asarray(solve_cut_vertices_jax(
        struct, jnp.array(boundary_points_flat(struct)),
        jnp.full((struct["rows"], struct["cols"]), r)))
    pt2 = defaultdict(list)
    for p in range(cp.shape[0]):
        for c in range(4):
            pt2[int(cp[p, c])].append(p)
    topo = {}
    for pid_val, ps in pt2.items():
        for a in range(len(ps)):
            for b in range(a + 1, len(ps)):
                if ps[a] != ps[b]:
                    topo[frozenset({ps[a], ps[b]})] = coords[pid_val]
    pivot_mismatch = 0
    for h in hinges:
        tp = topo.get(frozenset({h.face1, h.face2}))
        if tp is None or np.linalg.norm(verts[h.vertex1] - tp) > tol:
            pivot_mismatch += 1

    # Connectivity (BFS over the hinge graph).
    adj = {p: set() for p in range(n_faces)}
    for h in hinges:
        adj[h.face1].add(h.face2)
        adj[h.face2].add(h.face1)
    seen, q = {0}, deque([0])
    while q:
        for nb in adj[q.popleft()]:
            if nb not in seen:
                seen.add(nb)
                q.append(nb)

    print(f"\n=== {M}x{N} (panels={n_faces}, hinges={len(hinges)}, voids={len(tess.voids)}) ===")
    print(f"  expected edge-adjacencies : {len(expected)}")
    print(f"  max hinge pivot gap       : {max(gaps):.2e}  (tol {tol:g})")
    print(f"  non-edge-adjacent hinges  : {len(nonadj)}  {nonadj if nonadj else ''}")
    print(f"  missing adjacencies       : {len(missing)}  {sorted(tuple(sorted(m)) for m in missing) if missing else ''}")
    print(f"  extra hinges              : {len(extra)}  {sorted(tuple(sorted(e)) for e in extra) if extra else ''}")
    print(f"  duplicate hinges          : {len(dups)}  {dups if dups else ''}")
    print(f"  graph connected           : {len(seen) == n_faces}  ({len(seen)}/{n_faces})")
    print(f"  pivot-position mismatches : {pivot_mismatch}  (vs topological cut-point sharing)")
    ok = (max(gaps) < tol and not nonadj and not missing and not extra
          and not dups and len(seen) == n_faces and pivot_mismatch == 0)
    print(f"  VERDICT                   : {'ALL HINGES CORRECT' if ok else 'PROBLEMS FOUND'}")
    return tess, ok


def visualize(tess, M, N, tag):
    fig, ax = plt.subplots(figsize=(10, 10), facecolor="white")
    plot_tessellation(tess, ax=ax, show_target=False, show_hinges=False,
                      show_face_indices=True, show_hinge_indices=False, color_faces="#F58025")
    verts = tess.vertices
    centroids = tess.get_face_centroids()

    def cr(p):
        return (p // N, p % N)

    for h in tess.hinges:
        piv = verts[h.vertex1]
        (c1, r1), (c2, r2) = cr(h.face1), cr(h.face2)
        edge_adj = (abs(c1 - c2), abs(r1 - r2)) in ((1, 0), (0, 1))
        color = "#2A9D8F" if edge_adj else "#D62828"
        c_a, c_b = centroids[h.face1], centroids[h.face2]
        ax.plot([c_a[0], c_b[0]], [c_a[1], c_b[1]], color=color, lw=1.2, alpha=0.7, zorder=8)
        ax.scatter(*piv, s=45, color=color, zorder=9, edgecolors="black", linewidths=0.4)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Hinges on flat sheet ({tag}) — green=edge-adjacent, red=BAD")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, f"check_hinges_{tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def main():
    for M, N in ((4, 4), (6, 6)):
        tess, _ = audit(M, N)
        visualize(tess, M, N, f"{M}x{N}")


if __name__ == "__main__":
    main()
