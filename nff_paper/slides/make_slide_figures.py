"""Plates the defense deck needs and the manuscript does not.

Only figures that exist nowhere in ``paper/Images/`` belong here. Anything the manuscript already
carries is used as-is: a slide-only redraw of a published plate would be a second version of a
figure with no second source of truth.

    conda run -n kgnn_mac python nff_paper/slides/make_slide_figures.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nff.utils.figstyle import use_agg                                   # noqa: E402
use_agg()

import matplotlib.pyplot as plt                                          # noqa: E402

from nff.utils.figstyle import (FS_ANNOT, GREY, INK, ORANGE, TEAL,       # noqa: E402
                                apply_charter, figsize_in, paper_rc,
                                save_paper_figure, slide_bp)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

# The deck's chrome colour. It is not in figstyle.py because no manuscript figure uses it: the
# plates keep the charter ORANGE, and this one shade darker belongs to the slide furniture.
PRINCETON = "#E77500"

# The load-capacity paragraph of paper/sn-article.tex, Section 3.2. Four strategies are compared
# from one start against one target, swept in the transverse-to-axial ratio until deployment
# degenerates. The manuscript publishes the ratio for all four and the boundary error for only
# two of them, so only those two are plotted as points: the other two enter as the reference
# levels they are stated as. An unpublished coordinate would be a number on a slide with no
# source, which is the one thing this deck promises not to carry.
#   label, boundary error [mm], rho at which deployment degenerates, colour
ARMS = [
    ("Kinematics only",            81.7, 1.80, INK),
    ("Shape through\nequilibrium", 41.9, 1.29, ORANGE),
]
LEVELS = [                          # rho only, no published boundary error
    ("uniform sheet",                       1.66, GREY),
    ("$+\ \Phi_{comp}$, $\Phi_{shear}$",     2.95, TEAL),
]


def actuation_tradeoff():
    """Accuracy against load capacity: the two are set by different terms, which is the point."""
    w_bp, h_bp = slide_bp(0.62, aspect=1.32)
    apply_charter(**paper_rc())
    fig = plt.figure(figsize=figsize_in(w_bp, h_bp), facecolor="white")
    ax = fig.add_axes([0.145, 0.155, 0.825, 0.80])

    for label, rho, colour in LEVELS:
        ax.axhline(rho, color=colour, lw=0.8, ls=(0, (4, 2.5)), zorder=1)
        # Left, not right: the +64 per cent arrow stands at x = 120 and a right-set label
        # lands on top of it.
        ax.text(22.0, rho + 0.05, label, color=colour, fontsize=FS_ANNOT,
                ha="left", va="bottom")

    for label, err, rho, colour in ARMS:
        ax.plot(err, rho, "o", color=colour, ms=7.0, mec="white", mew=0.8, zorder=3)

    # Placed in DATA coordinates: an offset rule that clears one marker runs into the other.
    ann = {"Kinematics only":             (81.7, 1.92, "center", "bottom"),
           "Shape through\nequilibrium":  (45.5, 1.29, "left",   "center")}
    for label, err, rho, colour in ARMS:
        x, y, ha, va = ann[label]
        ax.annotate(label, xy=(x, y), color=colour, fontsize=FS_ANNOT, ha=ha, va=va, zorder=4)

    # 1.80 to 2.95 is the 64 per cent the manuscript states, over the geometric pattern.
    ax.annotate("", xy=(120.0, 2.95), xytext=(120.0, 1.80),
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=0.9))
    ax.text(117.5, 2.34, "$+64\\%$", color=TEAL, fontsize=FS_ANNOT, ha="right", va="center")

    ax.set_xlabel("boundary error  [mm]")
    ax.set_ylabel(r"transverse ratio $\rho$ sustained")
    ax.set_xlim(20, 135)
    ax.set_ylim(1.0, 3.3)

    # The two directions that count, stated on the axes rather than in the caption.
    ax.annotate("", xy=(0.035, 0.055), xytext=(0.26, 0.055), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=GREY, lw=0.8))
    ax.text(0.275, 0.055, "more accurate", transform=ax.transAxes, color=GREY,
            fontsize=FS_ANNOT, ha="left", va="center")

    save_paper_figure(fig, os.path.join(OUT, "actuation_tradeoff"), w_bp, h_bp,
                      dpi=600, png=False)
    plt.close(fig)


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEPLOY = os.path.join(REPO, "data/experiments/pet45_3x3_deployment_20260814")
CLOSEUP = os.path.join(REPO, "data/experiments/pet45_3x3_hinge_closeup_20260816")


def _rotating_squares(n=3, side=1.0, theta_deg=25.0):
    """Panel outlines and hinge points of an ``n`` by ``n`` rotating-square sheet.

    The lattice pitch that keeps adjacent panels in contact is ``side*(cos t + sin t)``: at
    ``t = 0`` the panels tile the plane and at ``t = 45`` they stand corner to corner. Rotation
    alternates in sign across the checkerboard, which is what opens a void at every vertex where
    four panels meet while every panel stays rigid.

    Returns:
        (panels, hinges) with panels an ``(n*n, 4, 2)`` array of corners and hinges an
        ``(m, 2)`` array of the points at which two panels touch.
    """
    import numpy as np

    t = np.deg2rad(theta_deg)
    pitch = side * (np.cos(t) + np.sin(t))
    corners = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]]) * side

    panels, index = [], {}
    for i in range(n):
        for j in range(n):
            a = t if (i + j) % 2 == 0 else -t
            rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            index[(i, j)] = len(panels)
            panels.append(np.array([i, j]) * pitch + corners @ rot.T)
    panels = np.asarray(panels)

    # A hinge is where two neighbouring panels share a corner. Taken as the closest corner pair
    # rather than constructed, so the drawing cannot disagree with the kinematics above.
    hinges = []
    for i in range(n):
        for j in range(n):
            for di, dj in ((1, 0), (0, 1)):
                if i + di >= n or j + dj >= n:
                    continue
                a, b = panels[index[(i, j)]], panels[index[(i + di, j + dj)]]
                d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
                p, q = np.unravel_index(d.argmin(), d.shape)
                hinges.append(0.5 * (a[p] + b[q]))
    return panels, np.asarray(hinges)


def _spring_glyph(ax, xy, radius, colour, turns=1.9, lw=0.9):
    """A torsional spring drawn as a spiral, the mark for a hinge that carries a moment."""
    import numpy as np

    phi = np.linspace(0.0, turns * 2 * np.pi, 220)
    r = radius * (0.28 + 0.72 * phi / phi[-1])
    ax.plot(xy[0] + r * np.cos(phi), xy[1] + r * np.sin(phi),
            color=colour, lw=lw, solid_capstyle="round", zorder=5)


def _hinge_corners(n=3, side=1.0, theta_deg=6.0):
    """Which corner of which panel holds which, as ``(panel_a, corner_a, panel_b, corner_b)``.

    At ``theta = 0`` the panels tile and every shared edge has two coincident corner pairs, so the
    contact cannot be read off the coordinates at all. A small opening picks it out, and the
    indices then carry back to the tiled sheet, which is where the ligaments have to be drawn.
    """
    import numpy as np

    panels, _ = _rotating_squares(n=n, side=side, theta_deg=theta_deg)
    out = []
    for i in range(n):
        for j in range(n):
            for di, dj in ((1, 0), (0, 1)):
                if i + di >= n or j + dj >= n:
                    continue
                ia, ib = i * n + j, (i + di) * n + (j + dj)
                d = np.linalg.norm(panels[ia][:, None, :] - panels[ib][None, :, :], axis=-1)
                p, q = np.unravel_index(d.argmin(), d.shape)
                out.append((ia, int(p), ib, int(q)))
    return out


def _flat_cut_pattern(n=3, side=1.0, lig=0.10):
    """The sheet closed, and the slits cut in it.

    Every interior edge of the tiling is cut except for a ligament at one end, and which end is
    what the checkerboard decides. Twelve slits for a three by three sheet, one per hinge.

    Returns:
        ``(panels, slits)`` with panels the tiled outlines and slits a list of ``(start, end)``,
        ``start`` being the end the ligament holds.
    """
    import numpy as np

    panels, _ = _rotating_squares(n=n, side=side, theta_deg=0.0)

    def mid(u, v):
        return tuple(np.round(0.5 * (u + v), 6))

    slits = []
    for ia, p, ib, q in _hinge_corners(n=n, side=side):
        edges_b = {mid(panels[ib][e], panels[ib][(e + 1) % 4]) for e in range(4)}
        for e in ((p - 1) % 4, p):
            u, v = panels[ia][e], panels[ia][(e + 1) % 4]
            if mid(u, v) not in edges_b:
                continue
            if not np.allclose(u, panels[ia][p]):        # run the slit away from the ligament
                u, v = v, u
            slits.append((u + lig * side * (v - u) / np.linalg.norm(v - u), v))
            break
    return panels, slits


def _scissors_glyph(ax, xy, length, angle_deg=90.0, colour=INK, lw=1.0):
    """A pair of scissors, blades along +x before ``angle_deg`` turns them.

    The blades are filled tapered triangles opening into a V and the handles are open rings.
    Two crossed strokes read as a stick figure at this size, and two closed blades merge into one
    spike; the open V and the ring are what make the shape a pair of scissors. Both cutting edges
    start on the axis at the pivot, which is where the shear is.
    Nothing drawn here is clipped, so the glyph can hang out of the tile it belongs to.

    Returns:
        ``(x0, y0, x1, y1)`` of everything drawn, in data coordinates.
    """
    import numpy as np
    from matplotlib.patches import Circle, Polygon

    th = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    o = np.asarray(xy, dtype=float)
    L = float(length)
    ring_r, ring_c = 0.15 * L, 0.60 * L
    seen = []
    for sgn in (+1.0, -1.0):
        # the blade: a sliver with its cutting edge along the axis and its back bulging away, so
        # the two together read as an open pair rather than as a cross
        blade = np.array([[0.04 * L, 0.0],
                          [0.20 * L, sgn * 0.17 * L],
                          [0.97 * L, sgn * 0.26 * L]])
        ax.add_patch(Polygon(o + blade @ rot.T, closed=True, facecolor=colour, edgecolor=colour,
                             lw=0.3 * lw, joinstyle="miter", zorder=7, clip_on=False))
        # the handle: a shaft out of the pivot the other way, ending in a ring
        hdir = np.array([-1.0, -sgn * 0.50])
        hdir /= np.linalg.norm(hdir)
        ring = ring_c * hdir
        ax.plot(*np.column_stack([o + rot @ (0.02 * L * hdir),
                                  o + rot @ (ring - ring_r * hdir)]),
                color=colour, lw=1.5 * lw, solid_capstyle="round", zorder=7, clip_on=False)
        ax.add_patch(Circle(o + rot @ ring, ring_r, facecolor="none", edgecolor=colour,
                            lw=1.5 * lw, zorder=8, clip_on=False))
        seen += [o + rot @ np.array([0.97 * L, sgn * 0.26 * L]), o + rot @ ring]
    ax.add_patch(Circle(o, 0.055 * L, facecolor="white", edgecolor=colour, lw=0.9 * lw,
                        zorder=9, clip_on=False))
    seen = np.asarray(seen)
    return (seen[:, 0].min() - ring_r, seen[:, 1].min() - ring_r,
            seen[:, 0].max() + ring_r, seen[:, 1].max() + ring_r)


# The opening slide's row, in bp on the 408 bp slide measure. The row is WIDTH bound: a square
# sheet, a square tessellation and a landscape photograph at one height come to 3.4 heights across,
# so the height each of them gets is what is left of 408 after the two question gaps. Widening the
# gaps buys arrow shaft and costs every visual its share of the difference.
Q_GAP = 25.0
AX_M = 0.10                 # air around a sheet, in panel units
# How far the handle rings reach behind the pivot, in blade lengths, from the glyph's own
# ring_c and handle direction: 0.60 * 0.8944 + 0.15. The layout needs it before the glyph is
# drawn, so it is asserted against the bbox the glyph returns.
SCIS_BACK = 0.687
SCIS_L = 0.75               # blade reach, in panel units
SCIS_PY = -0.42             # pivot, just inside the sheet on the slit that opens onto it


def two_scales_question():
    """The cut, the mechanism it opens into, and the material it is cut from.

    Three visuals and the two questions between them. Left is a sheet with the slits cut in it and
    nothing yet claimed about where they should go. Middle is the model those slits open into:
    rigid panels meeting at points, each point one stiffness. Right is one hinge of a real sheet,
    photographed out of plane. Naming any of the three is the slide's job, not the plate's.

    The two sheets are drawn at ONE scale. At 25 degrees the footprint has grown by
    ``cos t + sin t``, so the closed sheet is 0.7525 of the open one, and drawing them to a common
    bp per panel is what makes the opening a change of size rather than two unrelated pictures.
    """
    import numpy as np
    from PIL import Image
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Rectangle

    P_flat, slits = _flat_cut_pattern(n=3, side=1.0)
    P_dep, hinges = _rotating_squares(n=3, side=1.0, theta_deg=25.0)
    photo = np.asarray(Image.open(os.path.join(OUT, "two_scales_ligament.png")))
    ph_aspect = photo.shape[1] / photo.shape[0]

    # the scissors sits on the one slit whose open mouth is on the sheet's bottom edge, so its
    # handles hang below the sheet and cost the row no width at all
    y_bot = P_flat[:, :, 1].min()
    mouth = next(s[1] for s in slits if abs(s[1][1] - y_bot) < 1e-9)

    # ── the layout, solved for the scale the three visuals share ──────────────────────────
    dep_lo = min(P_dep[:, :, 0].min(), P_dep[:, :, 1].min()) - AX_M
    dep_hi = max(P_dep[:, :, 0].max(), P_dep[:, :, 1].max()) + AX_M
    dep_ext = dep_hi - dep_lo
    fx0, fx1 = P_flat[:, :, 0].min() - AX_M, P_flat[:, :, 0].max() + AX_M
    fy0, fy1 = y_bot - AX_M, P_flat[:, :, 1].max() + AX_M

    # Three SQUARE tiles on one centre line. The two sheets keep one scale, so the closed one is
    # the smaller square, and the photograph is cropped square to sit as the third.
    W_BP = 408.0
    s = (W_BP - 2 * Q_GAP) / ((fx1 - fx0) + dep_ext * (1.0 + ph_aspect))
    wf, hf = (fx1 - fx0) * s, (fy1 - fy0) * s
    wd = hd = dep_ext * s

    # The centre line has to clear BOTH the scissors hanging under the small tile and the two
    # larger tiles, which reach lower than the small one does even before the scissors. Taking
    # only the first of those puts the deployed and photograph tiles off the bottom of the canvas.
    over = (fy0 - (SCIS_PY - SCIS_BACK * SCIS_L)) * s
    pad = 3.0
    mid = pad + max(hd / 2.0, over + hf / 2.0)
    H_BP = mid + hd / 2.0 + pad

    apply_charter(**paper_rc())
    fig = plt.figure(figsize=figsize_in(W_BP, H_BP), facecolor="white")

    def rect(x, y, w, h):
        return [x / W_BP, y / H_BP, w / W_BP, h / H_BP]

    # ── the sheet, cut ────────────────────────────────────────────────────────────────────
    ax = fig.add_axes(rect(0.0, mid - hf / 2.0, wf, hf))
    ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(Rectangle((P_flat[:, :, 0].min(), y_bot), 3.0, 3.0,
                           facecolor=to_rgba(ORANGE, 0.28), edgecolor=INK, lw=1.0,
                           zorder=1))
    for a, b in slits:
        ax.plot([a[0], b[0]], [a[1], b[1]], color=INK, lw=1.1, solid_capstyle="butt", zorder=3)
    box = _scissors_glyph(ax, (mouth[0], SCIS_PY), SCIS_L, angle_deg=90.0, colour=INK, lw=1.1)
    assert abs(box[1] - (SCIS_PY - SCIS_BACK * SCIS_L)) < 0.02 * SCIS_L, \
        f"SCIS_BACK is stale: the glyph reaches {SCIS_PY - box[1]:.3f} behind its pivot"
    ax.set_xlim(fx0, fx1); ax.set_ylim(fy0, fy1)

    # ── the sheet, opened, with a stiffness at every contact ──────────────────────────────
    axd = fig.add_axes(rect(wf + Q_GAP, mid - hd / 2.0, wd, hd))
    axd.set_aspect("equal"); axd.axis("off")
    for quad in P_dep:
        axd.fill(quad[:, 0], quad[:, 1], facecolor=ORANGE, alpha=0.28, zorder=1)
        axd.plot(np.append(quad[:, 0], quad[0, 0]), np.append(quad[:, 1], quad[0, 1]),
                 color=INK, lw=1.0, zorder=2)
    for h in hinges:
        _spring_glyph(axd, h, 0.165, TEAL)
        axd.plot(*h, "o", color=TEAL, ms=3.6, mec="white", mew=0.7, zorder=6)
    axd.set_xlim(dep_lo, dep_hi); axd.set_ylim(dep_lo, dep_hi)

    # ── the two questions ─────────────────────────────────────────────────────────────────
    for gx in (wf, wf + Q_GAP + wd):
        axq = fig.add_axes(rect(gx, mid - hd / 2.0, Q_GAP, hd))
        axq.set_xlim(0, 1); axq.set_ylim(0, 1); axq.axis("off")
        axq.annotate("", xy=(0.95, 0.42), xytext=(0.05, 0.42),
                     arrowprops=dict(arrowstyle="<->", color=PRINCETON, lw=1.1,
                                     mutation_scale=11))
        axq.text(0.5, 0.50, "?", fontsize=FS_ANNOT + 12, color=PRINCETON,
                 ha="center", va="bottom", fontweight="bold")

    # ── one hinge of it, in the material ──────────────────────────────────────────────────
    axp = fig.add_axes(rect(wf + Q_GAP + wd + Q_GAP, mid - hd / 2.0, ph_aspect * hd, hd))
    axp.axis("off")
    axp.imshow(photo)

    save_paper_figure(fig, os.path.join(OUT, "two_scales_question"), W_BP, H_BP,
                      dpi=600, png=False)
    plt.close(fig)


def photographs():
    """Crops of the prototype photographs, at the size the slide shows them.

    ``slides/figures/`` is not versioned, so anything the deck includes has to be reproducible
    from the experiment folders or the next clean build fails on a missing file.
    """
    from PIL import Image

    # The ligament, cropped SQUARE so it sits as one tile beside the two square sheets, on the
    # relief hole at the cut tip with the buckled arch either side of it. The hole lands 0.45
    # across and 0.48 down, and the window stops above the orange floor tape, which is a fiducial
    # and not part of the subject. Halved from 4032 px: the slide shows it about 100 mm across,
    # so 1000 px is still past 250 dpi.
    src = os.path.join(CLOSEUP, "raw_png/IMG_3747.png")
    lig = Image.open(src).convert("RGB").crop((1250, 400, 3250, 2400))
    lig = lig.resize((lig.width // 2, lig.height // 2), Image.LANCZOS)
    lig.save(os.path.join(OUT, "two_scales_ligament.png"), optimize=True)

    # The rectification step already desaturates the floor, so the taped tile edges, hinges and
    # target read in the charter's own orange, green and red without anything being recoloured.
    for state, name in (("Closed", "sheet_closed_photo.png"),
                        ("Deployed", "sheet_deployed_photo.png")):
        img = Image.open(os.path.join(DEPLOY, f"rectified/{state}_clean.png")).convert("RGB")
        img.save(os.path.join(OUT, name), optimize=True)


def design_thumbs(dpi=300, lighten=0.70):
    """The five optimised cut patterns as footline thumbnails, each sharp and lightened.

    The deck marks which of the five designs a frame is showing with a strip of these at the bottom
    left. Both versions are rendered here rather than drawn under a pgf opacity in the deck,
    because some projectors flatten pgf transparency to nothing, which would leave five identical
    thumbnails and no indication of progress at all.

    Geometry is the --no-headers --no-row-labels build of matrix_all: the undeployed cut pattern is
    the second drawn column, its ink spanning 96.9 to 172.6 bp, and the row pitch is 79.6 bp from
    y = 6. Measured by scanning a 150 dpi rasterisation of that file.
    """
    import subprocess
    from PIL import Image
    src = os.path.join(OUT, "matrix_all.pdf")
    if not os.path.exists(src):
        raise SystemExit(f"{src} not built yet")
    stem = os.path.join(OUT, "_matrix_raster")
    subprocess.run(["pdftoppm", "-r", str(dpi), "-png", "-singlefile", src, stem], check=True)
    im = Image.open(stem + ".png").convert("RGB")
    ppb = im.size[0] / 432.0
    x0, x1, y0, pitch = 96.9, 172.6, 6.0, 79.6
    white = Image.new("RGB", (1, 1), "white")
    for i in range(5):
        box = (round(x0 * ppb), round((y0 + i * pitch) * ppb),
               round(x1 * ppb), round((y0 + (i + 1) * pitch) * ppb))
        cell = im.crop(box)
        # Trim the air the cell leaves around the drawing, so five thumbnails of different sheets
        # still read as one row of equal-weight marks.
        inv = Image.eval(cell.convert("L"), lambda v: 255 - v)
        if inv.getbbox():
            cell = cell.crop(inv.getbbox())
        cell.save(os.path.join(OUT, f"design_thumb_{i + 1}.png"), optimize=True)
        pale = Image.blend(cell, white.resize(cell.size), lighten)
        pale.save(os.path.join(OUT, f"design_thumb_{i + 1}_grey.png"), optimize=True)
    os.remove(stem + ".png")
    print(f"wrote 5 design thumbnails, sharp and lightened, from {os.path.basename(src)}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    # design_thumbs reads matrix_all.pdf, which make_slide_figures.sh writes LATER, so it is a
    # second entry point rather than part of the default run.
    if len(sys.argv) > 1 and sys.argv[1] == "design-thumbs":
        design_thumbs()
        raise SystemExit(0)
    # actuation_tradeoff() is kept but not called: the accuracy-against-capacity slide was cut
    # from the deck, and a figure nothing includes is a file that goes stale unnoticed.
    # photographs() first: two_scales_question() reads the ligament crop it writes.
    photographs()
    two_scales_question()
    print("done")
