"""3D perspective render of one deployed hinge, at TRUE scale (no out-of-plane exaggeration).

The whole CalculiX mesh (the Saint-Venant RVE) is drawn in the form-finding-lab green ``#2A9D8F``;
its true out-of-plane buckle is shown 1:1. The two adjacent kirigami faces are added in light grey
purely as CONTEXT -- they are NOT part of the finite-element model, only a visual cue for what the
hinge connects. Reads the cached frames (scratchpad/deploy_w9.py -> hinge_frames_w9.npz) and writes
one transparent PNG per rotation, ready to drop into the placeholder windows of the energy figure.

    conda run -n ccx python nff/scripts/figures/render_hinge_3d.py --theta 11 26.2 --labels 0.5 1.0
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

TEAL = "#2A9D8F"     # form-finding-lab turquoise (the hinge_overview strip colour)
INK, GREY = "#1A1A1A", "#6C757D"
# stress ramp: TURQUOISE, anchored on the hinge_overview teal #2A9D8F -- deep teal at low stress,
# light turquoise at peak. Named for reuse by the figure's von Mises scale.
STRESS_CMAP = LinearSegmentedColormap.from_list(
    "stress_teal", [(0.00, "#08332D"), (0.30, "#136B5F"), (0.58, "#2A9D8F"),
                    (0.79, "#5AC7B7"), (1.00, "#A6F0E2")])


def _vertex_normals(P, faces):
    """Area-weighted per-vertex normals, oriented toward +z (the top cap faces up)."""
    fn = np.cross(P[faces[:, 1]] - P[faces[:, 0]], P[faces[:, 2]] - P[faces[:, 0]])
    fn *= np.sign(fn[:, 2])[:, None] + 1e-30              # consistent outward (upward) winding
    N = np.zeros_like(P)
    for k in range(3):
        np.add.at(N, faces[:, k], fn)
    return N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)


def _subdivide(P, faces, fld, N, levels):
    """Loop-style 1->4 midpoint subdivision, carrying position, scalar field and vertex normal.
    Purely visual: smooths the coarse 754-triangle cap into a clean, high-resolution surface."""
    for _ in range(levels):
        e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
        uniq, inv = np.unique(e, axis=0, return_inverse=True)
        mid = len(P) + inv.reshape(3, -1).T              # (F,3) new-vertex ids for edges 01,12,20
        P = np.vstack([P, 0.5 * (P[uniq[:, 0]] + P[uniq[:, 1]])])
        fld = np.concatenate([fld, 0.5 * (fld[uniq[:, 0]] + fld[uniq[:, 1]])])
        Nm = N[uniq[:, 0]] + N[uniq[:, 1]]
        N = np.vstack([N, Nm / (np.linalg.norm(Nm, axis=1, keepdims=True) + 1e-12)])
        a, b, c = faces.T
        m0, m1, m2 = mid.T
        faces = np.concatenate([np.stack([a, m0, m2], 1), np.stack([m0, b, m1], 1),
                                np.stack([m2, m1, c], 1), np.stack([m0, m1, m2], 1)])
    return P, faces, fld, N


def _phong(normals, base, view=(0.2, -0.3, 0.9), key=(-0.45, -0.55, 0.72),
           fill=(0.55, 0.35, 0.55), ambient=0.34, k_diff=0.62, k_spec=0.20, shininess=30.0):
    """Per-face Phong shade: ambient + key/fill diffuse + a white specular sheen (light realism)."""
    V = np.asarray(view) / np.linalg.norm(view)
    bright = np.full(len(normals), ambient)
    spec = np.zeros(len(normals))
    for L, w in ((key, 1.0), (fill, 0.4)):
        L = np.asarray(L) / np.linalg.norm(L)
        bright = bright + w * k_diff * np.clip(normals @ L, 0, 1)
        H = L + V; H = H / np.linalg.norm(H)
        spec = spec + w * k_spec * np.clip(normals @ H, 0, 1) ** shininess
    rgb = base * bright[:, None] + spec[:, None]         # specular adds white highlight
    return np.clip(rgb, 0, 1)


def render(xyz, conn, disp, w_lig, thickness, r_win, out, elev=None, azim=-70, view_r=None,
           vm=None, vmax=None, vmin=None, scale=True, subdiv=2):
    top = conn[:, [3, 4, 5]] - 1                          # C3D15 top-cap triangle (local nodes 3,4,5)
    P = xyz + disp[:, :3]                                 # deformed nodes (frd DISP has a 4th col)

    lo = hi = None
    if vm is None:
        fld = np.zeros(len(P))                            # flat lab-green surface (field unused)
    else:
        fld = np.asarray(vm, float)
        lo = vmin if vmin is not None else float(np.percentile(fld[np.unique(top)], 2))
        hi = vmax if vmax is not None else float(np.percentile(fld[np.unique(top)], 99))

    N = _vertex_normals(P, top)
    P, faces, fld, N = _subdivide(P, top, fld, N, subdiv)
    tris = P[faces]
    face_N = N[faces].mean(axis=1)                        # smooth (Gouraud-like) per-face normal
    face_N /= np.linalg.norm(face_N, axis=1, keepdims=True) + 1e-12

    if vm is None:
        base = np.tile(to_rgb(TEAL), (len(faces), 1))
    else:
        fval = fld[faces].mean(axis=1)
        base = STRESS_CMAP(np.clip((fval - lo) / (hi - lo + 1e-9), 0, 1))[:, :3]
    fc = np.ones((len(faces), 4))
    fc[:, :3] = _phong(face_N, base)

    fig = plt.figure(figsize=(5.6, 4.3))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("persp")
    # antialiased=False + edgecolors=face kills matplotlib's white AA seams on a transparent bg;
    # a hairline edge in the face colour fills any sub-pixel gaps -> one clean continuous surface
    ax.add_collection3d(Poly3DCollection(tris, facecolors=fc, edgecolors=fc, linewidths=0.25,
                        antialiased=False, rasterized=True))

    # TRUE scale: z box-aspect proportional to the real out-of-plane range (no exaggeration).
    # ``view_r`` zooms to the hinge (for thin ligaments in a big Saint-Venant window); None = full mesh.
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    zlo, zhi = min(z.min(), -1), z.max() + 1
    if view_r is None:                                    # tight frame to the shape -> fills the canvas
        m = 0.04 * max(x.max() - x.min(), y.max() - y.min())   # (max resolution; meshes differ in size)
        x0, x1, y0, y1 = x.min() - m, x.max() + m, y.min() - m, y.max() + m
    else:
        x0, x1, y0, y1 = -view_r, view_r, -view_r, 0.2 * view_r
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_zlim(zlo, zhi)
    ax.set_box_aspect((x1 - x0, y1 - y0, zhi - zlo))
    # auto view: a hinge that buckles out of plane reads best at a low, dramatic angle; a nearly-flat
    # (thin-ligament) hinge reads as a sliver there, so tilt toward a clean top-down plan view instead.
    if elev is None:
        z_range = float(z.max() - z.min())
        elev = 24 if z_range > 0.06 * max(x1 - x0, y1 - y0) else 64
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

    # rigorous stress scale: a slim green colorbar mapping the von Mises field to MPa. The bar shows
    # the pure cmap (unshaded) so a colour reads back to a stress unambiguously. Suppressed with
    # ``scale=False`` when the render is a tight inset carrying a SHARED scale elsewhere in the figure.
    if lo is not None and scale:
        sm = ScalarMappable(norm=Normalize(lo, hi), cmap=STRESS_CMAP); sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.030, pad=0.0, aspect=26, shrink=0.66)
        cb.set_label(r"von Mises stress  $\sigma_\mathrm{vM}$  [MPa]", fontsize=8, color=INK)
        cb.ax.tick_params(labelsize=7, colors=INK, length=2)
        cb.outline.set_edgecolor("#D3D6DB"); cb.outline.set_linewidth(0.7)

    fig.savefig(out, dpi=400, bbox_inches="tight", transparent=True, pad_inches=0.02)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", default="data/outputs/hinge_frames_w9.npz")
    # frames chosen by ROTATION (the surrogate places the D=0.5 / D=1 dots by theta); labelled by D
    ap.add_argument("--theta", type=float, nargs="+", default=[11.0, 26.2])
    ap.add_argument("--labels", nargs="+", default=["0.5", "1.0"])
    ap.add_argument("--prefix", default="data/outputs/hinge_render_w9")
    ap.add_argument("--r-win", type=float, default=30.0, help="Saint-Venant window radius [mm]")
    ap.add_argument("--view-r", type=float, default=None,
                    help="zoom the frame to +-view_r [mm] around the hinge (thin ligaments); None = full mesh")
    ap.add_argument("--color-by", choices=["flat", "stress"], default="flat",
                    help="'stress' colours the surface by von Mises (needs 'vm' in the frames npz)")
    ap.add_argument("--vmax", type=float, default=None, help="fixed von Mises scale max [MPa] (shared)")
    ap.add_argument("--vmin", type=float, default=None, help="fixed von Mises scale min [MPa] (shared)")
    ap.add_argument("--no-scale", action="store_true",
                    help="suppress the per-render colorbar (tight inset carrying a shared scale)")
    ap.add_argument("--subdiv", type=int, default=2,
                    help="mesh subdivision levels for a smooth, high-res surface (each level x4 faces)")
    args = ap.parse_args()

    d = np.load(args.frames)
    xyz, conn, disp = d["xyz"], d["conn"], d["disp"]
    vm_all = d["vm"] if (args.color_by == "stress" and "vm" in d.files) else None
    th, w_lig, t = d["theta_deg"], float(d["w_lig"]), float(d["thickness"])
    for tgt_theta, lbl in zip(args.theta, args.labels):
        k = int(np.argmin(np.abs(th - tgt_theta)))
        out = f"{args.prefix}_D{lbl}.png"
        vm_k = vm_all[k] if vm_all is not None else None
        render(xyz, conn, disp[k], w_lig, t, args.r_win, out, view_r=args.view_r,
               vm=vm_k, vmax=args.vmax, vmin=args.vmin, scale=not args.no_scale, subdiv=args.subdiv)
        smax = f" vm_max={vm_k.max():.0f}MPa" if vm_k is not None else ""
        print(f"D={lbl}: frame {k}  theta={th[k]:.1f}deg  uz={np.abs(disp[k][:, 2]).max():.2f}{smax}  -> {out}")


if __name__ == "__main__":
    main()
