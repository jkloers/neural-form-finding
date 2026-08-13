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


def _matte(normals, base, key=(-0.45, -0.55, 0.72), ambient=0.80, k_diff=0.20):
    """Per-face shade with NO specular term and a narrow diffuse range -- a colour-faithful surface.

    ⚠ USE THIS WHENEVER THE COLOUR CARRIES DATA. A specular highlight adds white to the face colour,
    so a bright patch on the render is ambiguous between "high field value" and "facing the light":
    on a hinge that buckles out of plane the fold crest catches the key light exactly where the
    plastic strain is, and the sheen washes out the very region the figure is about.

    Ambient dominates (0.80 against 0.20 diffuse) so the ramp keeps its identity; the small
    remaining diffuse term is what still reads the 3D fold as a shape rather than a flat silhouette.
    """
    L = np.asarray(key) / np.linalg.norm(key)
    bright = ambient + k_diff * np.clip(normals @ L, 0, 1)
    return np.clip(base * bright[:, None], 0, 1)


def _depth_cue(rgb, tris, elev, azim, *, near=1.0, far=0.62):
    """Darken faces by DISTANCE FROM THE VIEWER, so self-overlapping material reads as layered.

    A hinge driven into axial compression folds back over itself, and at that point a purely
    normal-based shade cannot separate the two sheets: both flaps face the light the same way, so
    the near one and the far one come out the same colour and the fold reads as one flat smear.
    Depth is the only cue that distinguishes them, because it is the one thing that actually differs.

    Hue is untouched and the ordering is monotone in depth, so a face's colour still reads back to
    its field value up to a single brightness factor -- the ramp keeps its meaning, which a fog
    blend toward the background would have destroyed by desaturating far material.
    """
    e, a = np.radians(elev), np.radians(azim)
    view = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    d = tris.mean(axis=1) @ view                      # signed distance along the view axis
    span = float(d.max() - d.min())
    if span < 1e-9:
        return rgb
    t = (d - d.min()) / span                          # 0 = farthest, 1 = nearest
    return np.clip(rgb * (far + (near - far) * t)[:, None], 0, 1)


def _free_edges(faces):
    """Edges belonging to exactly ONE triangle: the surface's free boundary.

    On this mesh that boundary is the pair of CUT edges plus the window rim, which is precisely the
    outline of each flap. Stroking it separates material that overlaps in projection: when a hinge
    folds back on itself the two flaps are the same colour and the same brightness, so without an
    outline the near one dissolves into the far one and the fold reads as a single flat smear.
    """
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, idx, counts = np.unique(e, axis=0, return_index=True, return_counts=True)
    return e[idx[counts == 1]]


def _ligament_centre(P):
    """(x, y) of the ligament, from where the MESH IS FINEST.

    ⚠ THE LIGAMENT IS NOT AT THE ORIGIN. The RVE is a Saint-Venant window whose cut apex moves with
    ``w_lig`` and ``alpha``, so on the 5x5 pair the ligament sat at y = -35.8 mm and -18.7 mm for two
    designs of the same sheet. A window centred on (0, 0) put the first one entirely OUTSIDE the
    frame, and the render came out as blank panel with the hinge cropped off the bottom edge.

    Node density is the locator rather than the field: the mesher refines the fillet, so the density
    peak is the ligament whether or not the hinge has yielded. Keying off plastic strain instead
    would fail on exactly the designs this figure exists to show -- the ones that barely damage.
    """
    H, xe, ye = np.histogram2d(P[:, 0], P[:, 1], bins=48)
    i, j = np.unravel_index(np.argmax(H), H.shape)
    cx, cy = 0.5 * (xe[i] + xe[i + 1]), 0.5 * (ye[j] + ye[j + 1])
    # Refine: the modal bin is coarse, so recentre on the nodes it actually contains.
    r = 0.5 * max(xe[1] - xe[0], ye[1] - ye[0]) * 3.0
    near = (np.abs(P[:, 0] - cx) < r) & (np.abs(P[:, 1] - cy) < r)
    return (float(P[near, 0].mean()), float(P[near, 1].mean())) if near.sum() else (cx, cy)


def render(xyz, conn, disp, w_lig, thickness, r_win, out, elev=None, azim=-70, view_r=None,
           vm=None, vmax=None, vmin=None, scale=True, subdiv=2, dpi=400,
           cmap=None, label=None, shading="phong", outline=False, annotate=False,
           figsize=(5.6, 4.3), annotate_pt=7.0):
    """Render one deployed hinge to ``out``.

    ``vm`` is the nodal scalar painted on the surface. It is named for von Mises because that was
    the first field drawn, but any nodal field works -- ``peeq`` gives the plastic (damaged) regions
    -- so ``cmap`` and ``label`` exist to keep the colour ramp and the bar caption honest about
    which field is actually shown. Both default to the von Mises pair, so existing callers are
    unaffected.

    ``shading`` is ``"phong"`` (default, the glossy lit look) or ``"matte"``. Pass ``"matte"``
    whenever the colour is being READ rather than admired -- see :func:`_matte`.
    """
    cmap = cmap if cmap is not None else STRESS_CMAP
    label = label if label is not None else r"von Mises stress  $\sigma_\mathrm{vM}$  [MPa]"
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
        base = cmap(np.clip((fval - lo) / (hi - lo + 1e-9), 0, 1))[:, :3]
    fc = np.ones((len(faces), 4))
    fc[:, :3] = _phong(face_N, base) if shading == "phong" else _matte(face_N, base)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("persp")

    # TRUE scale: z box-aspect proportional to the real out-of-plane range (no exaggeration).
    # ``view_r`` zooms to the hinge (for thin ligaments in a big Saint-Venant window); None = full mesh.
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    zlo, zhi = min(z.min(), -1), z.max() + 1
    if view_r is None:                                    # tight frame to the shape -> fills the canvas
        m = 0.04 * max(x.max() - x.min(), y.max() - y.min())   # (max resolution; meshes differ in size)
        x0, x1, y0, y1 = x.min() - m, x.max() + m, y.min() - m, y.max() + m
    else:
        # Centred on the LIGAMENT, not the origin -- see :func:`_ligament_centre`.
        cx, cy = _ligament_centre(P)
        x0, x1, y0, y1 = cx - view_r, cx + view_r, cy - view_r, cy + view_r
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_zlim(zlo, zhi)
    ax.set_box_aspect((x1 - x0, y1 - y0, zhi - zlo))
    # auto view: a hinge that buckles out of plane reads best at a low, dramatic angle; a nearly-flat
    # (thin-ligament) hinge reads as a sliver there, so tilt toward a clean top-down plan view instead.
    if elev is None:
        z_range = float(z.max() - z.min())
        elev = 24 if z_range > 0.06 * max(x1 - x0, y1 - y0) else 64
    # Depth cue needs the RESOLVED camera, so shading is finished here rather than above.
    if shading == "depth":
        fc[:, :3] = _depth_cue(fc[:, :3], tris, elev, azim)
    # antialiased=False + edgecolors=face kills matplotlib's white AA seams on a transparent bg;
    # a hairline edge in the face colour fills any sub-pixel gaps -> one clean continuous surface
    ax.add_collection3d(Poly3DCollection(tris, facecolors=fc, edgecolors=fc, linewidths=0.25,
                        antialiased=False, rasterized=True))
    if outline:
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        ax.add_collection3d(Line3DCollection(P[_free_edges(faces)], colors=INK,
                                             linewidths=0.55, alpha=0.85, zorder=5))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

    # rigorous stress scale: a slim green colorbar mapping the von Mises field to MPa. The bar shows
    # the pure cmap (unshaded) so a colour reads back to a stress unambiguously. Suppressed with
    # ``scale=False`` when the render is a tight inset carrying a SHARED scale elsewhere in the figure.
    if lo is not None and scale:
        sm = ScalarMappable(norm=Normalize(lo, hi), cmap=cmap); sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.030, pad=0.0, aspect=26, shrink=0.66)
        cb.set_label(label, fontsize=8, color=INK)
        cb.ax.tick_params(labelsize=7, colors=INK, length=2)
        cb.outline.set_edgecolor("#D3D6DB"); cb.outline.set_linewidth(0.7)

    # ORIENTATION + SCALE, drawn in 2D over the projection. An orthographic view of a fold is
    # unreadable without them: nothing in the picture says which way is up or how big the hinge is,
    # and the two views differ ONLY in which axis was killed, which is exactly what a reader cannot
    # recover from the pixels. Placed in axes fraction so they land identically in every tile.
    if annotate:
        # Which world axes survive this projection. elev ~ 90 looks down z (plan: x-y);
        # elev ~ 0 looks along -y (elevation: x-z).
        plan = elev > 45.0
        h_lab, v_lab = ("x", "y") if plan else ("x", "z")
        # ANCHORED TO THE 3D AXES BOX, not the figure. Placed in figure fractions the glyphs land in
        # the transparent margin outside the mesh; `bbox_inches="tight"` then keeps that margin and
        # the tile comes out mostly empty with the hinge stranded in the middle.
        pos = ax.get_position()
        ov = fig.add_axes([0, 0, 1, 1], facecolor="none", zorder=10)
        ov.set_xlim(0, 1); ov.set_ylim(0, 1); ov.set_axis_off()
        L = 0.235 * pos.width
        ox, oy = pos.x0 + 0.055 * pos.width, pos.y0 + 0.10 * pos.height
        for dx, dy, lab in ((L, 0.0, h_lab), (0.0, L * pos.width / pos.height, v_lab)):
            ov.annotate("", xy=(ox + dx, oy + dy), xytext=(ox, oy),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.085 * annotate_pt,
                                        shrinkA=0, shrinkB=0, mutation_scale=0.5 * annotate_pt))
            ov.text(ox + 1.55 * dx, oy + 1.55 * dy, lab, fontsize=annotate_pt, color=INK,
                    ha="center", va="center")
        ov.text(ox - 0.012 * pos.width, oy - 0.075 * pos.height, "plan" if plan else "elevation",
                fontsize=annotate_pt, color=INK, ha="left", va="top")
        # Scale bar: a round number of mm, measured in the HORIZONTAL world axis of this view.
        world_w = x1 - x0
        nice = next((v for v in (1, 2, 5, 10, 20, 50) if v >= 0.25 * world_w), 50)
        frac = (nice / world_w) * pos.width
        bx = pos.x0 + pos.width * 0.95 - frac
        by = pos.y0 + 0.075 * pos.height
        ov.plot([bx, bx + frac], [by, by], color=INK, lw=0.085 * annotate_pt, solid_capstyle="butt")
        for e in (bx, bx + frac):
            ov.plot([e, e], [by - 0.012 * pos.height, by + 0.012 * pos.height], color=INK, lw=0.15 * annotate_pt)
        ov.text(bx + frac / 2, by + 0.022 * pos.height, f"{nice:g} mm", fontsize=annotate_pt, color=INK,
                ha="center", va="bottom")

    fig.savefig(out, dpi=dpi, bbox_inches="tight", transparent=True, pad_inches=0.02)
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
    ap.add_argument("--elev", type=float, default=None,
                    help="camera elevation [deg]. Default auto-picks a dramatic or a plan view from how "
                         "flat the hinge is -- which gives two DIFFERENT cameras when comparing a "
                         "buckled and a flat hinge. Set it explicitly for any matched comparison")
    ap.add_argument("--azim", type=float, default=-70.0, help="camera azimuth [deg]")
    ap.add_argument("--dpi", type=int, default=400, help="render resolution")
    args = ap.parse_args()

    d = np.load(args.frames)
    xyz, conn, disp = d["xyz"], d["conn"], d["disp"]
    vm_all = d["vm"] if (args.color_by == "stress" and "vm" in d.files) else None
    th, w_lig, t = d["theta_deg"], float(d["w_lig"]), float(d["thickness"])
    for tgt_theta, lbl in zip(args.theta, args.labels):
        k = int(np.argmin(np.abs(th - tgt_theta)))
        out = f"{args.prefix}_{lbl}.png"
        vm_k = vm_all[k] if vm_all is not None else None
        render(xyz, conn, disp[k], w_lig, t, args.r_win, out, elev=args.elev, azim=args.azim,
               view_r=args.view_r, vm=vm_k, vmax=args.vmax, vmin=args.vmin,
               scale=not args.no_scale, subdiv=args.subdiv, dpi=args.dpi)
        smax = f" vm_max={vm_k.max():.0f}MPa" if vm_k is not None else ""
        print(f"D={lbl}: frame {k}  theta={th[k]:.1f}deg  uz={np.abs(disp[k][:, 2]).max():.2f}{smax}  -> {out}")


if __name__ == "__main__":
    main()
