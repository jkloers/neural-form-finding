"""One ligament leaving the plane: where it sits, what it looks like, what the oracle predicts.

The deployment series measured the 3x3 PET sheet as a PLANAR mechanism and found a residual it
could not see: at matched extension the sheet came out 42.4 mm (2.5%) narrower than predicted,
which read as excess arc length is a 20-45 mm ripple, while the surrogate's guard put every hinge
past its trained compression floor. Both said the sheet relieves hinge compression by leaving the
plane. Neither could show it, because a homography measures only the floor projection.

Three panels, and nothing else. (a) says where we are: the deployed sheet, the two panels the
ligament joins, and the hinge itself. (b) and (c) are the comparison, the photographed hinge beside
the same hinge solved in CalculiX, drawn AT THE SAME CAMERA AND THE SAME SCALE so that any
disagreement is visible rather than argued. Every number lives in the sidecar ``.txt``.

★★★ THE CAMERA IS FITTED, NOT CHOSEN. The RVE's cut-tip fillet has radius ``0.16 * w_lig`` =
3.573 mm, so diameter 7.15 mm, which is exactly the H11 drill in the cut package -- the model and
the photograph contain the SAME circle, in the same material. Its imaged ellipse gives the viewing
direction (from the axis ratio and the major axis' bearing) and the scale (from the major axis,
the one diameter that tilt does not foreshorten). Fitting the render's camera to it turns "these
look similar" into "these are the same view".

⚠ FIT ON ABSOLUTE ANGLES, NEVER ON ROTATION-INVARIANT ONES. A first attempt matched the ellipse
ratio and the ellipse-to-cut angle, converged to a cost of 0.001, and produced a render of the
specimen MIRRORED: a reflection preserves both quantities. matplotlib keeps world +z pointing up at
``roll = 0`` and these frames were shot roughly level, so comparing bearings measured from image-up
fixes the sign. And all three residuals must be fitted together -- ratio, ellipse bearing, cut
bearing -- because with roll free, any two of them leave a one-parameter family the search happily
slides along, which shows up as the third disagreeing by 25 deg.

Run:
    python3 nff_paper/figures/plot_hinge_buckle_figure.py --variant bare
    python3 nff_paper/figures/plot_hinge_buckle_figure.py --variant taped
"""

import argparse
import hashlib
import json
import os
import shutil
import tempfile

import numpy as np

from nff.utils.figstyle import (use_agg, apply_charter, paper_rc, figsize_in, page_bp,
                                save_paper_figure, panel_letter, REAL, SIM, MUTE, INK, GREY,
                                RED, ORANGE, TEAL)

use_agg()
import matplotlib.pyplot as plt                                          # noqa: E402
from matplotlib.colors import ListedColormap, rgb_to_hsv, hsv_to_rgb     # noqa: E402
from matplotlib.patches import Polygon as MplPolygon, Circle             # noqa: E402
from scipy import ndimage as ndi                                         # noqa: E402
from shapely.geometry import Point, Polygon as ShapelyPolygon            # noqa: E402
from shapely.ops import unary_union                                      # noqa: E402

from nff_paper.lab.photo_rectify import (load_rgb, apply_h, clean_mask,       # noqa: E402
                                           colour_masks, homography_dlt, warp_to_grid)
from nff_paper.lab import photo_buckle as pb                           # noqa: E402
from nff_paper.lab import photo_geometry as pg                         # noqa: E402
from nff_paper.lab import photo_retouch as prt                         # noqa: E402
from nff_paper.figures.render_hinge_3d import render, Projector        # noqa: E402


CLOSEUP = "data/experiments/pet45_3x3_hinge_closeup_20260816"
DEPLOY = "data/experiments/pet45_3x3_deployment_20260814"
FRAMES = "data/outputs/hinge_frames_pet_pet45_3x3_mid_phys_h3.npz"

# Per shot: the hole seed and its radius in pixels, and the 35 mm equivalent focal length. These
# frames used digital zoom, so the focal differs between them and must be carried per image.
SHOTS = {
    "bare":  dict(image="IMG_3747", seed=(2228, 1358), r_guess=80,  f35=74,
                  tape=False, label="tape removed"),
    "taped": dict(image="IMG_3742", seed=(1855, 1093), r_guess=54,  f35=46,
                  tape=True,  label="as taped"),
}
HOLE_MM = 7.15                      # H11 drill; equals 2 * 0.16 * w_lig in the RVE
CACHE = "data/outputs/.hinge_buckle_cache"


def cache_key(**kw):
    """A stable digest of everything an expensive step depends on."""
    return hashlib.sha1(json.dumps(kw, sort_keys=True, default=str).encode()).hexdigest()[:16]


def cached_json(key, build, *, use_cache=True):
    """Run ``build`` once per distinct input and keep the answer on disk.

    The camera fit is a grid search over roughly 770k projections, so it costs about 45 s and is
    DETERMINISTIC in its inputs. Recomputing it on every styling tweak is pure waste, and the waste
    is what makes a figure tedious enough to stop improving.
    """
    path = os.path.join(CACHE, f"{key}.json")
    if use_cache and os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    val = build()
    os.makedirs(CACHE, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(val, fh)
    return val


def cached_png(key, build, *, use_cache=True):
    """Same, for the 3D render: it depends only on the camera, the frame and the framing."""
    path = os.path.join(CACHE, f"{key}.png")
    if not (use_cache and os.path.exists(path)):
        os.makedirs(CACHE, exist_ok=True)
        shutil.copyfile(build(), path)
    return path


def colour_correct(rgb, *, sat=1.22, lo_pct=1.0, hi_pct=99.2):
    """Grey-world white balance off the floor, a linear contrast stretch, then a saturation lift.

    ⚠ The stretch is AFFINE ON RGB, driven by luminance percentiles. Rescaling each pixel by the
    ratio of its stretched luminance to its original is not a contrast curve: it multiplies dark
    pixels by a large factor and light ones by a small one, and it drove green tape to near-black
    while leaving the floor correct.
    """
    hsv = rgb_to_hsv(np.clip(rgb, 0, 1))
    floor = (hsv[..., 1] < 0.18) & (hsv[..., 2] > 0.25)
    if floor.sum() > 1000:
        mean = rgb[floor].reshape(-1, 3).mean(axis=0)
        rgb = np.clip(rgb * (mean.mean() / np.maximum(mean, 1e-3)), 0, 1)
    lum = rgb.mean(axis=2)
    lo, hi = np.percentile(lum, lo_pct), np.percentile(lum, hi_pct)
    rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)
    hsv = rgb_to_hsv(rgb)
    hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0, 1)
    return hsv_to_rgb(hsv)


def scan_filter(rgb, *, bg_px=6.0, detail_px=3.0, core=0.10, keep=0.92,
                lo_pct=1.0, hi_pct=99.3, sat=1.14, flatten=True, gamma=1.0,
                white_balance=True):
    """A document-scanner look: even illumination, texture suppressed, edges kept.

    The mechanism is AMPLITUDE CORING, not frequency filtering, and that choice is the whole point.
    The floor's mottle and the sheet's cut lines live at the SAME spatial scale -- both a few pixels
    across -- so nothing that selects by frequency can separate them, and every attempt to do so
    either leaves the speckle or erases the only thing that renders a transparent sheet visible.
    What does separate them is contrast: the mottle is a few percent of the local mean, the cut lines
    and tape edges are tens of percent. So the detail layer is passed through a soft gate,

        detail' = detail * d^2 / (d^2 + core^2)

    which is ~0 for small ``d`` and ~1 for large, with no threshold to sit exactly on.

    ⚠ Flat-fielding ALONE makes this worse, not better. Dividing by a blurred copy evens the
    lighting but normalises the background to a constant, which RAISES the speckle's relative
    contrast; a white-point knee on top of that then clipped the sheet away while the floor grain
    survived. Flatten the illumination, yes, but core the detail in the same pass.

    ⚠ Scales are ABSOLUTE PIXELS, not fractions of the frame. Every photograph here is 4032 x 3024
    off the same camera, so absolute is the consistent choice -- and a fraction silently retunes the
    filter whenever it is handed a crop, which made the settings chosen on a test crop behave
    differently on the full frame.

    ⚠ Colour is carried by RATIOS to luminance. Filtering channels independently shifts hues
    wherever they disagree, which turns the green marking tape olive.
    """
    rgb = np.clip(np.asarray(rgb, float), 1e-4, 1.0)
    if white_balance:
        hsv0 = rgb_to_hsv(rgb)
        floor = (hsv0[..., 1] < 0.18) & (hsv0[..., 2] > 0.25)
        if floor.sum() > 1000:
            mean = rgb[floor].reshape(-1, 3).mean(axis=0)
            rgb = np.clip(rgb * (mean.mean() / np.maximum(mean, 1e-3)), 1e-4, 1.0)
    lum = rgb.mean(axis=2)
    chroma = rgb / lum[..., None]                        # hue and saturation, held aside

    if flatten:
        base = ndi.gaussian_filter(lum, max(bg_px, 1.0))
        flat = lum / np.maximum(base, 1e-3)              # even illumination, background near 1
    else:
        # ⚠ FLATTENING SUITS THE PLAN VIEW AND RUINS THE CLOSE-UP. In (a) the subject is tape and
        # cut lines, and normalising the background away is pure gain. In (b) the subject is a
        # TRANSLUCENT SHEET whose only rendering is its own body tone against the floor: flatten
        # that and the sheet becomes an outline drawing while the floor's blotches, being the same
        # size as the features, survive regardless. So here the pass is despeckle plus contrast.
        flat = lum

    low = ndi.gaussian_filter(flat, max(detail_px, 1.0))
    d = flat - low
    gate = d * d / (d * d + core * core)
    flat = low + d * (gate + (1.0 - gate) * (1.0 - keep))

    lo, hi = np.percentile(flat, lo_pct), np.percentile(flat, hi_pct)
    out_lum = np.clip((flat - lo) / max(hi - lo, 1e-6), 0.0, 1.0) ** gamma
    out = np.clip(chroma * out_lum[..., None], 0.0, 1.0)
    hsv = rgb_to_hsv(out)
    hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0, 1)
    return hsv_to_rgb(hsv)


def rotate_about_hole(img, hole_px, angle_deg, half_px):
    """Crop a square about the hole, then rotate it in place, so the hole stays at the centre.

    Rotating about the hole rather than the frame keeps the registration trivial: the feature both
    panels are aligned on does not move, so the same rotation can be applied to the photograph and
    to the render and the match survives it exactly. Camera roll IS an in-plane rotation of a
    perspective projection, which is why levelling the pair costs nothing in fidelity.
    """
    cx, cy = hole_px
    r = int(round(1.45 * half_px))                      # margin so the rotation cannot clip
    x0, y0 = int(round(cx)) - r, int(round(cy)) - r
    pad = ((max(0, -y0), max(0, y0 + 2 * r - img.shape[0])),
           (max(0, -x0), max(0, x0 + 2 * r - img.shape[1])), (0, 0))
    q = np.pad(img, pad[:2] + pad[2:] if img.ndim == 3 else pad[:2], mode="edge")
    x0, y0 = x0 + pad[1][0], y0 + pad[0][0]
    q = q[y0:y0 + 2 * r, x0:x0 + 2 * r]
    if abs(angle_deg) > 1e-6:
        q = ndi.rotate(q, angle_deg, axes=(1, 0), reshape=False, order=1, mode="nearest")
    return q, (q.shape[1] / 2.0, q.shape[0] / 2.0)


def crop_alpha(img, pad=3):
    """Crop an RGBA render to the bounding box of what it drew."""
    if img.ndim != 3 or img.shape[2] != 4:
        return img
    ys, xs = np.nonzero(img[..., 3] > 0.02)
    if ys.size == 0:
        return img
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, img.shape[0])
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, img.shape[1])
    return img[y0:y1, x0:x1]


# ── the model's own features ──────────────────────────────────────────────────────────────────

def model_features(d, k):
    """Deformed nodes, plus masks for the hole rim and the main-cut corridor."""
    ref = np.asarray(d["xyz"], float)
    P = ref + np.asarray(d["disp"][k], float)[:, :3]
    w = float(d["w_lig"])
    rho = 0.16 * w
    alpha = np.radians(float(d["alpha_deg"]))
    cap = ref[:, 2] > 0.25 * float(d["thickness"])
    tip = np.array([0.0, -w])
    r = np.hypot(ref[:, 0] - tip[0], ref[:, 1] - tip[1])
    rim = cap & (r > 0.92 * rho) & (r < 1.08 * rho)
    u = np.array([np.cos(-alpha), np.sin(-alpha)])
    n = np.array([-u[1], u[0]])
    s = (ref[:, :2] - tip) @ u
    q = np.abs((ref[:, :2] - tip) @ n)
    cut = cap & (s > 6.0) & (s < 45.0) & (q < 3.0)
    return P, rim, cut


def view_signature(project, P, rim, cut):
    """``(ellipse ratio, ellipse bearing, cut bearing)`` of a projected model, all in image space."""
    e = pb.fit_ellipse(project(P[rim]))
    c = project(P[cut])
    c = c - c.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    return e.ratio, e.angle, float(np.arctan2(vt[0][1], vt[0][0])), e


def fit_camera(P, rim, cut, target, view_r, *, focal=None,
               elev_range=(2.0, 80.0, 1.0), azim_step=1.5, roll_max=40.0, roll_step=2.0):
    """Camera whose projected model reproduces the photograph's three bearings.

    ``target`` is ``(ratio, ellipse_bearing, cut_bearing)`` measured on the photograph and already
    converted to the projector's convention, in which y runs UP.
    """
    t_ratio, t_major, t_cut = target
    pr = Projector(P, view_r=view_r, focal_length=focal)
    best = None
    for roll in np.arange(-roll_max, roll_max + 1e-9, roll_step):
        for elev in np.arange(*elev_range):
            for azim in np.arange(-180.0, 180.0, azim_step):
                def project(q, _e=elev, _a=azim, _r=roll):
                    return pr.project(q, elev=_e, azim=_a, roll=_r)
                try:
                    ratio, major, cutang, e = view_signature(project, P, rim, cut)
                except Exception:
                    continue
                dr = (ratio - t_ratio) / 0.05
                dm = np.degrees(np.mod(major - t_major + np.pi / 2, np.pi) - np.pi / 2) / 5.0
                dc = np.degrees(np.mod(cutang - t_cut + np.pi, 2 * np.pi) - np.pi) / 5.0
                cost = dr * dr + dm * dm + dc * dc
                if best is None or cost < best["cost"]:
                    best = dict(cost=float(cost), elev=float(elev), azim=float(azim),
                                roll=float(roll), ratio=float(ratio),
                                major_deg=float(np.degrees(major)),
                                cut_deg=float(np.degrees(cutang)))
    pr.close()
    return best


def render_registration(P, top_nodes, rim, cam, view_r, img_shape):
    """Where the relief hole lands in a rendered image, and that image's scale, in pixels.

    ⚠ NOT by finding the hole in the picture. At this hinge the drilled hole sits at the tip of the
    main cut, and the cut has OPENED, so the hole is continuous with the slit and the slit runs to
    the silhouette: there is no enclosed island of background to detect, and looking for one fails
    outright rather than quietly.

    Registration instead comes from the projection itself. ``render`` saves with a tight bounding
    box, so the drawn surface's extent in the projector's normalised coordinates maps affinely onto
    the cropped image's extent, and projecting the rim through that map places it exactly.
    """
    pr = Projector(P, view_r=view_r)
    kw = dict(elev=cam["elev"], azim=cam["azim"], roll=cam["roll"])
    uv = pr.project(P[top_nodes], **kw)
    rim_uv = pr.project(P[rim], **kw)
    pr.close()
    u0, u1 = uv[:, 0].min(), uv[:, 0].max()
    v0, v1 = uv[:, 1].min(), uv[:, 1].max()
    H, W = img_shape[:2]
    sx, sy = (W - 1) / (u1 - u0), (H - 1) / (v1 - v0)
    px = np.column_stack([(rim_uv[:, 0] - u0) * sx, (v1 - rim_uv[:, 1]) * sy])
    e = pb.fit_ellipse(px)
    return e, float(sx), float(sy)


def rim_planarity_mm(d, k):
    """RMS departure of the deformed hole rim from its own best-fit plane, in millimetres.

    Bounds the one assumption the camera fit makes -- that the drilled hole is still a circle -- by
    computing it instead of asserting it.
    """
    P, rim, _ = model_features(d, k)
    Q = P[rim]
    Q = Q - Q.mean(axis=0)
    _, _, vt = np.linalg.svd(Q, full_matrices=False)
    return float(np.sqrt(np.mean((Q @ vt[2]) ** 2)))


# ── panels ────────────────────────────────────────────────────────────────────────────────────

ROT_DEG = -90.0        # brings the sheet's long axis horizontal, as Fig. 10 of the manuscript does
PULL_DOF = 1           # the run applies its load on face 5 along +y of the design frame


def rot_h(deg):
    """A rigid rotation as a homography, for composing with a rectification."""
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])


def shared_corners(nodes, pairs):
    """The corner each pair of panels shares, per hinge, in millimetres.

    A hinge is where two panel corners meet, so the closest pair of corners between the two panels
    locates it. In the closed sheet those corners coincide exactly; deployed they sit a few
    millimetres apart, and their midpoint is the hinge.
    """
    out = []
    for b, a in pairs:
        d = np.linalg.norm(nodes[a][:, None, :] - nodes[b][None, :, :], axis=-1)
        i, j = np.unravel_index(d.argmin(), d.shape)
        out.append(0.5 * (nodes[a][i] + nodes[b][j]))
    return np.asarray(out)


def fill_geom(ax, geom, **kw):
    """Draw a shapely polygon or multipolygon, exteriors only."""
    if geom.is_empty:
        return
    parts = getattr(geom, "geoms", [geom])
    for g in parts:
        if g.geom_type == "Polygon" and not g.is_empty:
            ax.add_patch(MplPolygon(np.asarray(g.exterior.coords), closed=True, **kw))


def hinge_windows(tiles, hinge_pts, pairs, r_win_mm):
    """The meshed window of every hinge, as MATERIAL: the disk cut by ITS OWN TWO panels.

    ⚠ A window is not a disk. The condensation solves on the sheet inside a Saint-Venant radius, so
    the region is the disk cut by the panels' boundaries -- which is why it appears in Fig. 4 as a
    strip pinched between two cuts and not as a circle. Drawing the disk instead puts teal on the
    floor, where there is no material and nothing is solved.

    ⚠⚠ AND IT IS THE TWO PANELS THAT HINGE JOINS, not the whole sheet. Clipping against the union of
    all nine paints teal onto any third panel that happens to pass within the radius, which claims a
    window over material the RVE never contains -- ``nff/rve/geometry.py`` defines it as "the two
    tiles it joins". Two hinges on this deployed sheet sit close enough to a neighbour for the
    difference to show.
    """
    polys = [ShapelyPolygon(t) for t in tiles]
    out = []
    for p, (b, a) in zip(hinge_pts, pairs):
        local = unary_union([polys[int(a)], polys[int(b)]])
        out.append(local.intersection(Point(*p).buffer(r_win_mm, quad_segs=64)))
    return out


# Two objects lie in the deployed frame and belong to the laboratory rather than to the specimen:
# a steel weight resting on the sheet beside panel 3, and a cable crossing the upper panels.
# Positions are in MILLIMETRES OF THE ROTATED DEPLOYED FRAME, the only frame in which they are
# fixed across re-renders, and they are DECLARED rather than detected: a rule that finds a dark
# cable also finds the sheet's own cut lines, which are the subject. Each was then read off the
# rectification -- the weight is a 2412 px dark component centred at (135, -596), the cable three
# collinear fragments running from (1703, -874) to (2715, -1220) -- so the regions cover measured
# objects with a margin rather than covering a guess. ⚠ The cable's line was fitted through its
# visible run, not through the fragments' bounding boxes: a 30 mm error in the line left its frayed
# END outside a 44 mm corridor, and one surviving dark speck is as distracting as the whole cable.
LAB_OBJECTS = (
    dict(kind="disc", xy=(95.0, -590.0), r=135.0, why="steel weight lying on the sheet"),
    dict(kind="segment", a=(1670.0, -890.0), b=(2790.0, -1213.0), w=58.0, why="cable"),
)
# ⚠ The glare is BOUNDED by a declared rectangle and only then detected inside it. A percentile
# over the whole sheet cannot separate a highlight from a merely brighter panel: measured on this
# frame, one step of the threshold takes the mask from 1.6% of the panel to 46%, swallowing a whole
# panel. The rectangle is the intention, the detector supplies the shape.
GLARE_REGION = (
    dict(kind="polygon", why="the ceiling light reflected in the sheet",
         pts=[(830.0, -1400.0), (1640.0, -1400.0), (1640.0, -940.0), (830.0, -940.0)]),
)
GLARE_KW = dict(q=88.0, q_grow=62.0, min_area_px=200, close_px=7, dilate_px=7)
# The target is a rectangle of tape whose four corners were MEASURED, and 97% of its perimeter is
# visible in the frame -- one edge is 86% and the rest is complete. Repainting the whole band along
# the measured corners therefore COMPLETES a known shape rather than inventing one, and it makes a
# band that the eye reads as one rectangle instead of a broken one. Width measured off the red
# mask by distance transform: half-width 10.8 mm.
TAPE_W_MM = 21.6


def _wrap90(deg):
    """An axis angle folded into (-90, 90], where a line and its reverse are the same thing."""
    return float((float(deg) + 90.0) % 180.0 - 90.0)


def level_angle(kind="target"):
    """Extra in-plane rotation that brings the deployed frame's content level, in degrees.

    ``ROT_DEG`` squares the panel to the CLOSED sheet, which is the design frame and the frame the
    state panels compare in. But the sheet swings bodily as it deploys, and the target was taped to
    the floor at its own angle, so in a photograph of the DEPLOYED state nothing visible is level:
    the sheet reads about 8 deg off and the taped rectangle about 15 deg. On a panel that overlays
    no prediction that is an angle with no visible cause, and it reads as a careless crop.

    ⚠ Levelling hides nothing. The swing is intrinsic to this mechanism and it is a RESULT, but it
    is a result the state panels carry, where a fixed target sits beside a sheet that has turned.
    A lone photograph cannot show a rotation, having nothing to show it against.

    ⚠⚠ **Square the TARGET, not the sheet.** Levelling the tessellation's principal axis leaves the
    taped rectangle at about 8 deg, and that rectangle is the only true rectangle in the frame, so
    the eye takes it for the frame and reads the whole panel as crooked. Squaring the target puts
    its worst edge within 1.8 deg of the page; the sheet then sits inside it at +7.7 deg, which is
    the real deployment swing and reads as intentional because the surround is square. It also
    frames better: 0.00 % of the panel outside the photograph against 0.57 %.

    Args:
        kind: ``"target"`` squares the taped rectangle to the page, ``"sheet"`` levels the deployed
            sheet's principal axis instead, ``"none"`` keeps the design frame.

    Returns:
        Degrees to add to ``ROT_DEG``.
    """
    if kind in (None, "none"):
        return 0.0
    z = np.load(f"{DEPLOY}/derived/measured.npz", allow_pickle=True)
    R = rot_h(ROT_DEG)
    if kind == "sheet":
        P = np.array([apply_h(R, n) for n in z["Deployed_nodes"]]).reshape(-1, 2)
        Q = P - P.mean(axis=0)
        v = np.linalg.eigh(np.cov(Q.T))[1][:, -1]
        # An axis is a LINE, not a direction: wrap into (-90, 90] before negating, or the answer
        # comes back 180 deg out and the sheet is levelled upside down.
        return -_wrap90(np.degrees(np.arctan2(v[1], v[0])))
    if kind == "target":
        # ⚠ SQUARE ALL FOUR EDGES, not the longest one. The taped rectangle reaches the design frame
        # through a homography fitted on the SHEET, so its corners arrive as a quadrilateral a
        # couple of degrees out of square: in this deployment the four edges sit at -14.88, -14.74,
        # -17.48 and -15.91 deg. Squaring the longest edge alone puts that one dead level and
        # leaves the opposite short edge 2.74 deg off -- against the panel's own border, on the
        # side of the plate where a viewer has a straight edge to compare with, which is exactly
        # where a residual tilt is read as a crooked figure. A length-weighted least squares over
        # all four brings the worst edge to 1.8 deg and no edge is favoured over its neighbours.
        # ⚠ Level on the rectangle that is DRAWN, not on the quadrilateral that is measured. The
        # weighted least squares above is then exact rather than a compromise, and the drawn
        # rectangle lands dead square on the page instead of 1.8 deg off it.
        t = apply_h(R, pg.regularise_rectangle(np.asarray(z["red_design_corners"], float)))
        e = np.array([t[(i + 1) % 4] - t[i] for i in range(4)])
        w = np.linalg.norm(e, axis=1)
        # An edge of a rectangle is an axis mod 90 deg, so fold before averaging: the two families
        # of edges are 90 deg apart and their raw angles would average to something between them.
        a = np.degrees(np.arctan2(e[:, 1], e[:, 0]))
        return -float((w * ((a + 45.0) % 90.0 - 45.0)).sum() / w.sum())
    raise ValueError(f"unknown level kind {kind!r}")


def rotate_shapes(shapes, deg):
    """The declared regions, turned with the frame they were measured in.

    ⚠ ``LAB_OBJECTS`` and ``GLARE_REGION`` are millimetres of the ``ROT_DEG`` frame, so any extra
    levelling must carry them along or every mask lands somewhere else on the sheet.
    """
    a = np.radians(deg)
    M = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    out = []
    for s in shapes:
        t = dict(s)
        for key in ("xy", "a", "b"):
            if key in t:
                t[key] = tuple(M @ np.asarray(t[key], float))
        if "pts" in t:
            t["pts"] = (np.asarray(t["pts"], float) @ M.T)
        out.append(t)
    return out


def context_window(frame_on="target", margin=0.018, level_deg=0.0):
    """The millimetre bounds panel (a) is drawn on, and the tiles and target it holds.

    Separated from the drawing so a layout can ask for the crop's ASPECT before it places any axes.
    A caller that re-derives these bounds is a second copy of the framing rule, and the two drift.

    Returns:
        ``(x0, x1, y0, y1, tiles, target)`` in millimetres of the rotated deployed frame.
    """
    z = np.load(f"{DEPLOY}/derived/measured.npz", allow_pickle=True)
    R = rot_h(ROT_DEG + level_deg)
    tiles = np.array([apply_h(R, n) for n in z["Deployed_nodes"]])
    target = apply_h(R, z["red_design_corners"])
    pts = (tiles.reshape(-1, 2) if frame_on == "sheet"
           else np.vstack([tiles.reshape(-1, 2), target]))
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    m = margin * (x1 - x0)
    return x0 - m, x1 + m, y0 - m, y1 + m, tiles, target


def sheet_mask(gx, gy, tiles):
    """Which grid cells the specimen covers, for a search that must not wander onto the floor."""
    return prt.region_mask(gx, gy, [dict(kind="polygon", pts=t) for t in tiles])


def panel_context(ax, *, hinge=3, marker_frac=1.60, scan=True, margin=0.018, n_px=1150,
                  r_win_mm=100.0, frame_on="target", retouch=False, draw_target=True,
                  ledger=None, level_deg=0.0):
    """The deployed sheet in plan, oriented and annotated as the manuscript's Fig. 10 is.

    The photograph is rectified through the deployment's own homography and then rotated by
    ``ROT_DEG``, the same -90 deg the validation plate uses to bring the sheet's long axis
    horizontal. The rotation is RIGID, so the sheet keeps its true 1716 x 2719 mm proportions;
    an earlier version squared the target instead, which stretched it 1.58x across the short axis
    for no gain.

    Annotation follows the manuscript rather than inventing a second language: orange is the
    measured sheet, the grey panel is held, the target is a BLACK DASHED rectangle exactly as
    ``plot_deployment_validation.TARGET_KW`` draws it, and the panels are numbered.

    Args:
        frame_on: ``"target"`` frames the target rectangle as well, ``"sheet"`` frames the sheet
            alone. ⚠ The choice is what decides how much white the panel shows. An inverse warp
            fills whatever the source did not cover, and the photograph's footprint is a KEYSTONE,
            so an axis-aligned crop that reaches out to the target leaves 6.3% of the panel empty
            against 0.6% for one that stops at the sheet.
        retouch: reconstruct the empty region, the specular highlight and the lab objects, and
            neutralise the target tape. Disclosed through ``ledger``, never silently.
        draw_target: the dashed rectangle. Off under ``frame_on="sheet"``, where it would be
            clipped by the frame and read as a mistake rather than as a crop.
        ledger: a dict the retouch statistics are written into, for the provenance sidecar.
    """
    z = np.load(f"{DEPLOY}/derived/measured.npz", allow_pickle=True)
    nodes, pairs = z["Deployed_nodes"], z["face_pairs"]
    A = z["Deployed_H"]                       # image pixels -> millimetres
    R = rot_h(ROT_DEG + level_deg)
    rgb = load_rgb(f"{DEPLOY}/raw/Deployed.jpg")

    x0, x1, y0, y1, tiles, target = context_window(frame_on, margin, level_deg)
    gx = np.linspace(x0, x1, n_px)
    gy = np.linspace(y0, y1, int(round(n_px * (y1 - y0) / (x1 - x0))))

    # ⚠ GENTLE, not scanned. A hard local flatten made the plan view crisp and made it look
    # processed: the tape acquired an emboss and the sheet's reflections turned into relief. What is
    # actually wanted is the original colour with the floor lifted, so the illumination is left
    # alone, the grain is cored lightly, and the brightening is one gamma on luminance.
    img = (scan_filter(rgb, flatten=False, detail_px=2.0, core=0.05, gamma=0.62,
                       lo_pct=0.5, hi_pct=99.6, sat=1.06) if scan else colour_correct(rgb))
    if retouch:
        # The measurement is already made: the homography was fitted on the UNMODIFIED photograph
        # and this panel is drawn from it, so nothing reconstructed here can reach a number.
        warped = warp_to_grid(img, R @ A, gx, gy, fill=np.nan)
        # ⚠ clean_mask FIRST. JPEG stores chroma at half resolution, so a raw hue mask arrives with
        # a one-pixel halo and speckle, and repainting through it leaves the tape fringed. The
        # dilation then swallows the tape's own specular edge, which is white rather than red and
        # so is invisible to a hue window.
        masks = colour_masks(rgb)
        def _warp_mask(m, grow=0):
            w = warp_to_grid(clean_mask(m).astype(float), R @ A, gx, gy, order=0, fill=0.0) > 0.5
            return ndi.binary_dilation(w, np.ones((grow, grow), bool)) if grow > 1 else w
        red = _warp_mask(masks["red"], grow=5)
        # ⚠ Restricted to OUTSIDE the panels. The tape lies UNDER the sheet where the two cross, so
        # painting the band over them would put the floor on top of the specimen.
        band = prt.region_mask(gx, gy, [dict(kind="segment", a=target[i], b=target[(i + 1) % 4],
                                             w=TAPE_W_MM) for i in range(4)])
        # ⚠ Every reconstruction draws from the sheet and the floor ONLY. The glare, the cable and
        # the weight all touch orange marking tape, and a fill allowed to sample it spreads tape
        # colour across the sheet -- the reflection comes back as a brown stain instead of leaving.
        tape = red | _warp_mask(masks["orange"], grow=5) | _warp_mask(masks["green"], grow=5)
        inside = sheet_mask(gx, gy, tiles)
        band = band & ~inside
        warped, led = prt.retouch(warped, gx=gx, gy=gy, inside=inside,
                                  objects=rotate_shapes(LAB_OBJECTS, level_deg),
                                  neutral_mask=(red | band), glare=True, source=~tape,
                                  glare_where=prt.region_mask(
                                      gx, gy, rotate_shapes(GLARE_REGION, level_deg)),
                                  glare_kw=GLARE_KW)
        if ledger is not None:
            ledger.update(led)
            ledger["completed"] = prt.mask_stats(band & ~red, gx, gy)
    else:
        warped = np.clip(np.nan_to_num(warp_to_grid(img, R @ A, gx, gy, fill=1.0), nan=1.0), 0, 1)
    # ⚠ ``origin="lower"`` and a bottom-to-top extent, exactly as
    # ``plot_deployment_validation.panel_frame`` draws its rectified states. Row 0 of the warp
    # belongs to ``gy[0]``, so an upper-origin extent puts it at the top and hands the axes a
    # DESCENDING y -- which mirrors this panel about its long axis relative to every other
    # drawing of the same sheet. The photograph and the overlay stay registered either way, so
    # the flip is invisible here and only shows up against the state panels beside it.
    ax.imshow(np.clip(warped, 0, 1), extent=[gx[0], gx[-1], gy[0], gy[-1]], origin="lower")

    b, a = pairs[hinge]                                  # face_pairs is stored (high, low)
    hinges = np.array([apply_h(R, h[None])[0] for h in shared_corners(nodes, pairs)])

    def poly(i, **kw):
        ax.add_patch(MplPolygon(tiles[i], closed=True, **kw))

    # The manuscript draws every target as a BLACK DASHED line (plot_deployment_validation.py's
    # TARGET_KW); matching it exactly keeps one visual language across the figures.
    # ⚠ REGULARISED for drawing, exactly as plot_deployment_validation.geometry does it, and for
    # the same reason. ⛔ Not `target` itself: that is the measured tape, and it is still what the
    # retouch band above is painted along and what the framing is computed from. Drawing the
    # rectangle and painting the band along the quadrilateral is the point -- the band is
    # reconstructing tape that exists, the line is naming the shape the tape stands for.
    if draw_target:
        ax.add_patch(MplPolygon(pg.regularise_rectangle(target), closed=True, facecolor="none",
                                edgecolor=INK, lw=0.9, ls=(0, (3, 2))))
    for i in range(len(tiles)):
        poly(i, facecolor="none", edgecolor="white", lw=0.8, alpha=0.70)
    for i in (int(a), int(b)):
        poly(i, facecolor=ORANGE, edgecolor=ORANGE, lw=1.1, alpha=0.30)
    # The meshed window around EVERY hinge, at its true radius, teal, exactly as Fig. 4a draws it.
    # Showing all twelve says what the close-up is a sample of: the condensation treats each of
    # these windows independently, and the one ringed below is simply the one opened up.
    for win in hinge_windows(tiles, hinges, pairs, r_win_mm):
        fill_geom(ax, win, facecolor=TEAL, edgecolor=TEAL, lw=0.7, alpha=0.38, zorder=4)
    poly(3, facecolor=GREY, edgecolor=GREY, lw=0.9, alpha=0.45)
    for i in (int(a), int(b)):
        c = tiles[i].mean(axis=0)
        ax.text(c[0], c[1], str(i), fontsize=8.4, color="white", ha="center", va="center",
                fontweight="bold")
    c3 = tiles[3].mean(axis=0)
    ax.text(c3[0], c3[1], "3", fontsize=8.4, color="white", ha="center", va="center",
            fontweight="bold")

    c5 = tiles[5].mean(axis=0)
    ax.text(c5[0], c5[1], "5", fontsize=8.4, color="white", ha="center", va="center",
            fontweight="bold")

    dd = np.linalg.norm(nodes[a][:, None, :] - nodes[b][None, :, :], axis=-1)
    i, j = np.unravel_index(dd.argmin(), dd.shape)
    hp = apply_h(R, (0.5 * (nodes[a][i] + nodes[b][j]))[None])[0]
    # Sized to the WINDOW it rings, not to the panel: the mark then means "this is the one opened
    # up", and it stays correct if the sheet or the figure is ever rescaled.
    r = marker_frac * r_win_mm
    # A ring AROUND the window, not a bullseye on top of it: thin strokes, and a radius comfortably
    # larger than ``r_win`` so the whole window is visible inside with margin. Heavy strokes at a
    # snug radius read as a target symbol stamped on the hinge instead of a mark singling it out.
    ax.add_patch(Circle(hp, r, facecolor="none", edgecolor=INK, lw=2.6, alpha=0.55, zorder=6))
    ax.add_patch(Circle(hp, r, facecolor="none", edgecolor="white", lw=1.3, zorder=7))
    ax.set_xlim(gx[0], gx[-1])
    ax.set_ylim(gy[0], gy[-1])
    ax.set_aspect("equal")
    ax.set_axis_off()
    return hp, r


def downsample(img, target_px=1200):
    """Block-average an image down to about ``target_px`` on its long side.

    ⚠ THIS IS WHAT KEEPS THE PAPER PDF SMALL. matplotlib embeds an ``imshow`` array at the array's
    own resolution, not at the figure's dpi, so a 3000 px crop ships 3000 px into the PDF however
    small the panel is printed. ``paper/Images`` holds no photographs today and its typical figure is
    under 620 kB, so an unreduced plate would dominate the article. The same trick is used by
    ``plot_deployment_validation.rectified``, which resamples onto a 2.2 mm grid before drawing.

    Block-averaging rather than slicing: a stride subsample aliases the floor's speckle into moire,
    which is exactly the texture the tone curve was tuned to calm.
    """
    q = np.asarray(img, float)
    f = int(max(q.shape[:2]) // max(target_px, 1))
    if f < 2:
        return q
    h, w = (q.shape[0] // f) * f, (q.shape[1] // f) * f
    q = q[:h, :w]
    shape = (h // f, f, w // f, f) + q.shape[2:]
    return q.reshape(shape).mean(axis=(1, 3))


def show_in_mm(ax, img, hole_px, mm_per_px, view_r_mm, aspect_y=0.60, target_px=1200):
    """Draw an image in a MILLIMETRE frame centred on the relief hole.

    Both panels are placed this way, so the photograph and the render end up sharing one coordinate
    system rather than merely looking similar in size. That is what lets a reader compare them: a
    feature at the same millimetre offset lands at the same place on the page in both.
    """
    cx, cy = hole_px
    H, W = img.shape[:2]
    extent = [(0 - cx) * mm_per_px, (W - cx) * mm_per_px,
              (H - cy) * mm_per_px, (0 - cy) * mm_per_px]
    img = downsample(img, target_px)          # extent is in mm, so this cannot move anything
    ax.imshow(img, extent=extent, interpolation="bilinear")
    ax.set_xlim(-view_r_mm, view_r_mm)
    # A LANDSCAPE window, not a square one. The buckle is a shallow wide arch, so a square field
    # spends most of its height on floor in the photograph and on nothing at all in the render.
    ax.set_ylim(aspect_y * view_r_mm, -aspect_y * view_r_mm)   # y down, as in both source images
    ax.set_aspect("equal")
    ax.set_axis_off()


def panel_photo(ax, shot, view_r_mm, *, level_deg=0.0, aspect_y=0.60, scan=True,
                target_px=1200):
    """The photographed hinge, levelled and in the shared millimetre frame."""
    rgb = load_rgb(f"{CLOSEUP}/raw_png/{shot['image']}.png")
    e = pb.fit_hole(rgb.mean(2), shot["seed"], shot["r_guess"])
    mm_per_px = e.mm_per_px(HOLE_MM)
    img = (scan_filter(rgb, flatten=False, detail_px=2.0, core=0.05, gamma=0.70,
                       lo_pct=0.5, hi_pct=99.6, sat=1.06) if scan else colour_correct(rgb))
    img, hole = rotate_about_hole(img, (e.cx, e.cy), level_deg, view_r_mm / mm_per_px)
    show_in_mm(ax, img, hole, mm_per_px, view_r_mm, aspect_y=aspect_y, target_px=target_px)
    return e


def panel_render(ax, d, k, cam, view_r_mm, *, tape, dpi, focal=None, aspect_y=0.60,
                 roll=None, use_cache=True, target_px=1200):
    """The oracle's hinge, in the same millimetre frame, measured off its own relief hole.

    The render's scale is read from the hole it draws rather than from the axes limits, so the two
    panels are registered by the same physical feature and any error in that registration is an
    error the figure can be challenged on.
    """
    xyz, conn, disp = d["xyz"], d["conn"], d["disp"][k]
    kw = {}
    if tape:
        kw = dict(vm=pb.band_indicator(xyz, half_width_mm=0.5 * float(d["w_lig"])),
                  vmin=0.0, vmax=1.0, cmap=ListedColormap(["#c9ced4", "#5FAE4A"]))
    roll = cam["roll"] if roll is None else roll

    def build():
        tmp = os.path.join(tempfile.mkdtemp(), "hinge.png")
        render(xyz, conn, disp, float(d["w_lig"]), float(d["thickness"]), float(d["r_win"]), tmp,
               elev=cam["elev"], azim=cam["azim"], roll=roll, view_r=1.35 * view_r_mm,
               focal_length=focal, scale=False, subdiv=2, dpi=dpi, shading="depth",
               outline=True, annotate=False, figsize=(5.0, 5.0), **kw)
        return tmp

    key = cache_key(step="render", elev=cam["elev"], azim=cam["azim"], roll=roll, k=int(k),
                    view_r=view_r_mm, tape=bool(tape), dpi=int(dpi), focal=focal)
    img = crop_alpha(plt.imread(cached_png(key, build, use_cache=use_cache)))
    top = np.unique(np.asarray(conn)[:, [3, 4, 5]] - 1)
    _, rim, _ = model_features(d, k)
    e, sx, sy = render_registration(np.asarray(xyz) + np.asarray(disp)[:, :3], top, rim,
                                    dict(cam, roll=roll), 1.35 * view_r_mm, img.shape)
    show_in_mm(ax, img, (e.cx, e.cy), HOLE_MM / e.major, view_r_mm, aspect_y=aspect_y,
               target_px=target_px)
    return dict(hole_px=[e.cx, e.cy], major_px=e.major, ratio=e.ratio,
                bearing_deg=float(np.degrees(e.angle)), mm_per_px=HOLE_MM / e.major,
                aspect_check=float(sx / sy))


def scale_bar(ax, view_r_mm, length_mm=50.0, colour=INK, aspect_y=0.60):
    """A bar in the shared millimetre frame, so both panels carry the same physical length."""
    x0 = -view_r_mm + 0.09 * view_r_mm
    y0 = aspect_y * view_r_mm - 0.10 * view_r_mm
    ax.plot([x0, x0 + length_mm], [y0, y0], color=colour, lw=2.2, solid_capstyle="butt",
            zorder=6)
    ax.text(x0 + length_mm / 2, y0 - 0.035 * view_r_mm, f"{length_mm:g} mm", color=colour,
            fontsize=6.6, ha="center", va="bottom", zorder=6)


def fit_shot_camera(shot, frames_path=FRAMES, theta_measured=16.58, *, view_r=None,
                    use_cache=True):
    """Measure the relief hole in one photograph and fit the render's camera to it.

    The whole comparison rests on this: the hole is one circle that the photograph and the mesh
    both contain, so its imaged ellipse fixes the viewing direction, the scale and the roll. Shared
    by the standalone plate and by the merged physical-validation plate, because a second copy of
    this would be a second camera.

    Args:
        shot: one entry of :data:`SHOTS`.
        frames_path: the CalculiX replay of the ligament's own measured route.
        theta_measured: hinge opening in degrees; selects the frame to compare against.
        view_r: half-width of the field in millimetres; defaults to the photograph's own.
        use_cache: reuse the ~45 s grid search when its inputs are unchanged.

    Returns:
        dict with the loaded ``frames``, the frame index ``k``, the fitted ``camera``, ``view_r``,
        ``mm_per_px`` and the fitted ``hole`` ellipse.
    """
    d = np.load(frames_path, allow_pickle=True)
    th = d["theta_deg"]
    k = int(np.argmin(np.abs(th - theta_measured)))

    rgb = load_rgb(f"{CLOSEUP}/raw_png/{shot['image']}.png")
    e_ph = pb.fit_hole(rgb.mean(2), shot["seed"], shot["r_guess"])
    mm_per_px = e_ph.mm_per_px(HOLE_MM)
    cut_px = pb.radial_dark_peaks(rgb.mean(2), (e_ph.cx, e_ph.cy), 0.5 * e_ph.major,
                                  n_peaks=1, sector=(np.radians(20), np.radians(160)))[0][0]
    view_r = view_r or 0.5 * min(rgb.shape[:2]) * mm_per_px
    # pixel y runs down and the projector's runs up, so every bearing flips sign
    target = (e_ph.ratio, np.mod(-e_ph.angle, np.pi), -cut_px)
    P, rim, cut = model_features(d, k)
    key = cache_key(step="camera", target=[float(t) for t in target], view_r=float(view_r),
                    frames=os.path.basename(frames_path), k=int(k))
    cam = cached_json(key, lambda: fit_camera(P, rim, cut, target, view_r), use_cache=use_cache)
    cam["theta"] = float(th[k])
    return dict(frames=d, k=k, camera=cam, view_r=float(view_r), mm_per_px=float(mm_per_px),
                hole=e_ph)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(SHOTS), default="bare")
    ap.add_argument("--frames", default=FRAMES)
    ap.add_argument("--theta-measured", type=float, default=16.58)
    ap.add_argument("--theta-sim", type=float, default=17.25)
    ap.add_argument("--view-r", type=float, default=None,
                    help="physical half-width of panels (b) and (c) [mm]; default fills the frame")
    ap.add_argument("--marker-frac", type=float, default=1.60,
                    help="radius of the ring marking the studied hinge, in units of r_win")
    ap.add_argument("--width-frac", type=float, default=1.18,
                    help="page_bp width fraction. 1.18 is the house wide value; the manuscript must "
                         "wrap the graphic in \\makebox[\\textwidth][c]{...} to centre the overhang")
    ap.add_argument("--aspect", type=float, default=4.05, help="page_bp aspect, width/height")
    ap.add_argument("--target-px", type=int, default=1200,
                    help="long side each photograph is reduced to before drawing; controls PDF size")
    ap.add_argument("--out", default=None,
                    help="default ships the bare variant to paper/Images/hinge_buckle.pdf")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--no-cache", action="store_true",
                    help="redo the camera fit and the render instead of reusing the cache")
    args = ap.parse_args()

    shot = SHOTS[args.variant]
    # The shipping variant writes STRAIGHT into the manuscript's Images folder. Never iterate to a
    # temp path and copy in: a byte-identical copy makes latexmk skip its rebuild, and every check
    # downstream then passes on a stale figure.
    out = args.out or ("paper/Images/hinge_buckle.pdf" if args.variant == "bare"
                       else f"data/outputs/hinge_buckle_figure_{args.variant}.pdf")
    fit = fit_shot_camera(shot, args.frames, args.theta_measured, view_r=args.view_r,
                          use_cache=not args.no_cache)
    d, k, cam = fit["frames"], fit["k"], fit["camera"]
    view_r, mm_per_px, e_ph = fit["view_r"], fit["mm_per_px"], fit["hole"]

    # ---- the plate: three pictures on ONE LINE, all the same shape and size
    apply_charter(**paper_rc())
    w_bp, h_bp = page_bp(args.width_frac, args.aspect)
    fig = plt.figure(figsize=figsize_in(w_bp, h_bp))
    gs = fig.add_gridspec(1, 3, left=0.018, right=0.994, top=0.855, bottom=0.020, wspace=0.035)

    axa = fig.add_subplot(gs[0, 0])
    # Draw the context first: its crop aspect sets the shape of all three panels, so the row reads
    # as one strip rather than three differently proportioned pictures.
    _, _ = panel_context(axa, marker_frac=args.marker_frac,
                         r_win_mm=float(d["r_win"]))
    # (b) and (c) span 2*view_r horizontally and 2*aspect_y*view_r vertically, so their data
    # aspect IS aspect_y; matching (a)'s crop aspect makes all three panels one shape.
    aspect_y = abs(float(np.diff(axa.get_ylim())[0] / np.diff(axa.get_xlim())[0]))

    axb = fig.add_subplot(gs[0, 1])
    # Levelling: the fit says how far the camera was rolled, so rendering at roll 0 and rotating the
    # photograph by the same amount puts world-up upright in BOTH without touching the match.
    panel_photo(axb, shot, view_r, level_deg=-cam["roll"], aspect_y=aspect_y,
                target_px=args.target_px)
    scale_bar(axb, view_r, aspect_y=aspect_y)

    axc = fig.add_subplot(gs[0, 2])
    rend = panel_render(axc, d, k, cam, view_r, tape=shot["tape"], dpi=args.dpi,
                        aspect_y=aspect_y, roll=0.0, use_cache=not args.no_cache,
                        target_px=args.target_px)
    scale_bar(axc, view_r, aspect_y=aspect_y)

    # Paper-coherent naming: the manuscript pairs "measured" with "predicted" throughout, and
    # names the solver only where the method is described, never on a plate.
    titles = ["the deployed sheet",
              "measured   $\\vartheta$ = %.1f$^\\circ$" % args.theta_measured,
              "predicted   $\\vartheta$ = %.1f$^\\circ$" % cam["theta"]]
    for ax, txt in zip((axa, axb, axc), titles):
        ax.set_title(txt, fontsize=8.0, color=INK, pad=3)

    for a_, letter in zip((axa, axb, axc), "abc"):
        bb = a_.get_position()
        panel_letter(fig, letter, x=max(bb.x0 - 0.012, 0.002), y=min(bb.y1 + 0.030, 0.998))

    pdf, lo, hi = save_paper_figure(fig, out, w_bp, h_bp, dpi=args.dpi, strict_margins=True)
    plt.close(fig)
    out = pdf

    # ---- the numbers, in the sidecar rather than on the plate
    uz = d["disp"][k][:, 2]
    info = dict(variant=args.variant, image=shot["image"], out=out,
                camera=cam, view_r_mm=view_r, mm_per_px=mm_per_px,
                hole_photo=dict(ratio=e_ph.ratio, tilt_deg=e_ph.tilt_deg, major_px=e_ph.major,
                                # reported in the PROJECTOR's convention (y up), so it can be
                                # compared with the model's directly; pixel y runs the other way
                                bearing_deg=float(np.degrees(np.mod(-e_ph.angle, np.pi))),
                                bearing_px_deg=float(np.degrees(e_ph.angle))),
                render_hole=rend,
                rim_planarity_mm=rim_planarity_mm(d, k),
                model=dict(theta_deg=float(d["theta_deg"][k]), uz_max_mm=float(uz.max()),
                           uz_over_t=float(uz.max() / float(d["thickness"])),
                           w_lig_mm=float(d["w_lig"]), alpha_deg=float(d["alpha_deg"]),
                           peak_peeq=float(np.nanmax(d["peeq_nodal"][k])),
                           damage=float(d["damage"][k])))
    with open(os.path.splitext(out)[0] + ".txt", "w") as fh:
        fh.write(caption_text(info, args))
    print(json.dumps(info, indent=2))
    print(f"wrote {out} ({w_bp:.1f} x {h_bp:.1f} bp, type {lo:.1f}-{hi:.1f} pt, "
          f"{os.path.getsize(out) / 1024.0:.0f} kB)")


def caption_text(info, args):
    c, m = info["camera"], info["model"]
    return (
        "Figure. One ligament of the 3x3 PET sheet leaving the plane.\n\n"
        "(a) The sheet in plan in its deployed state, rectified through the deployment homography\n"
        "and oriented as Fig. 10 is, with panel 3 held, panel 5 the one pulled, the\n"
        "dashed rectangle the target, and the ring the ligament joining panels 1 and 2.\n"
        "(b) That ligament photographed (%s, %s). (c) The same ligament solved in the CalculiX\n"
        "hinge oracle along its own measured (a, s, theta) route, drawn at true 1:1 out-of-plane\n"
        "scale. Panels (b) and (c) share a camera and a scale; both were levelled by the same\n"
        "in-plane rotation, so the match between them is untouched by it.\n\n"
        "Ligament: w_lig %.2f mm, alpha %.1f deg, t 0.5 mm, Saint-Venant window r_win 100 mm.\n"
        "Opening: %.2f deg measured from the deployment photographs, %.2f deg simulated;\n"
        "  the render is the frame at %.2f deg.\n"
        "Predicted escape: u_z %.1f mm, %.0f times the sheet gauge. Peak PEEQ %.4f, damage %.1e,\n"
        "  so the buckle is elastic.\n\n"
        "Camera: fitted, not chosen. The relief hole is a 7.15 mm circle present in both the\n"
        "  photograph and the model. elev %.1f deg, azim %.1f deg, roll %.1f deg.\n"
        "  Hole axis ratio %.3f photographed against %.3f rendered; ellipse bearing %.1f deg\n"
        "  against %.1f deg (both measured from image-up); cut bearing %.1f deg.\n"
        "  Scale %.5f mm per pixel, from the hole's major axis, the one diameter that tilt\n"
        "  does not foreshorten.\n"
        "  Model rim departs from its own best-fit plane by %.3f mm RMS, which bounds the\n"
        "  assumption that the drilled hole is still a circle.\n\n"
        "Provenance, not part of the caption:\n"
        "  frames %s\n"
        "  photograph %s/raw_png/%s.png\n"
        "  regenerate: python3 nff_paper/figures/plot_hinge_buckle_figure.py --variant %s\n"
        % (info["image"], "tape removed" if info["variant"] == "bare" else "as taped",
           m["w_lig_mm"], m["alpha_deg"], args.theta_measured, args.theta_sim, m["theta_deg"],
           m["uz_max_mm"], m["uz_over_t"], m["peak_peeq"], m["damage"],
           c["elev"], c["azim"], c["roll"],
           info["hole_photo"]["ratio"], c["ratio"],
           info["hole_photo"]["bearing_deg"], c["major_deg"], c["cut_deg"],
           info["mm_per_px"], info["rim_planarity_mm"],
           args.frames, CLOSEUP, info["image"], info["variant"]))


if __name__ == "__main__":
    main()
