"""One ligament, followed from the force network of the sheet into its own strain history.

The compression-shift hexbin is a population picture: it reports that the strain field moved, not
what moved or where. This is the case study. A single ligament -- the one driven deepest into axial
compression by the reference design -- is located in the deployed load path and then followed
through the ``(eta_a, eta_s)`` plane, for the reference design and for the design that carries the
hinge-path objectives.

    row 1   reference design      shape objective only
    row 2   regularized design    shape and hinge-path objectives
    col 1   load transmission     tension routes, compression zones, shear planes; the monitored
                                  ligament ringed in that row's colour
    col 2   ligament strain       every trajectory of that design in that row's colour, the
                                  monitored one emphasised, with that design's projected
                                  distributions on both axes

``eta_a < 0`` is a ligament being SQUEEZED rather than pivoted: it does no opening work, and it is
the regime the hinge condensation resolves worst. No surrogate trust box is drawn -- where the
checkpoint happens to be confident is a property of the training set, not of the mechanism.

Labelled ISO-DAMAGE lines carry the surrogate's damage landscape, in multiples of the RMS training
damage, so the levels are absolute and the rows compare directly. ⚠ That field is a slice at fixed
rotation and fixed ligament geometry (see :func:`damage_field`), and it EXTRAPOLATES outside the
checkpoint's training box -- deep compression is well outside it, and the levels thin out there
because the head saturates, not because the hinge is safe.

At the tip of each monitored trajectory sits a 3D render of the RVE hinge, painted by accumulated
plastic strain, so the damaged material is visible as geometry rather than only as a contour level.

★★ THE RENDER MUST BE A REPLAY IF THE TWO ROWS ARE TO BE COMPARED. Pass ``--ref-frames`` and
``--opt-frames`` pointing at archives written by ``deploy_hinge_frames.py --paths <that design>
--hinge <k>``: the oracle is then driven along that ligament's OWN measured ``(a, s, theta)`` route
at its OWN ``w_lig`` and ``alpha``, and -- because the oracle is elastoplastic and therefore
path-dependent -- the plastic strain in the picture is the plastic strain the ligament has.

⚠ The shared ``--frames`` default is a PURE-ROTATION ray cache at a fixed ligament width, indexed by
rotation alone. It carries zero axial and zero shear, so it is blind to the components a hinge-path
objective moves: on the 5x5 pair the two rows differed 16x in ``eta_s`` and 3x in surrogate damage,
yet resolved to adjacent frames of one sweep with 13.63% vs 13.73% of nodes past yield. Use it only
to illustrate the rotation spine, never to compare two designs.

★ THE PROJECTED DISTRIBUTIONS ARE TAKEN FROM THE ARC-LENGTH RESAMPLED CLOUD (``densify``), not from
the raw ``paths.eta`` samples that ``plot_path_shift.py`` and ``plot_path_arms.py`` histogram. Raw
samples are spaced by LOAD STEP, so a ligament that stalls mid-deployment drops many samples in one
place and the distribution ends up shaded by how slowly the stepping moved. Resampling by arc length
makes the marginal a density over path length, which is the quantity the word "distribution" is
doing work for here.

Hinge indices are in ``bond_connectivity`` order, which is topological and therefore identical
across designs of the same tessellation -- ligament ``k`` is the same physical ligament in both rows.

Run:
    PYTHONPATH=$(pwd) JAX_PLATFORMS=cpu conda run -n kgnn_mac \
        python nff/scripts/figures/plot_compressed_hinge.py \
            --ref-paths data/outputs/hinge_paths/pet45_baseline \
            --ref-field data/outputs/load_paths/pet45_baseline \
            --opt-paths data/outputs/hinge_paths/pet45_cP10_dP \
            --opt-field data/outputs/load_paths/pet45_cP10_dP \
            --out data/outputs/pet45_hinge_case_study.png

``--*-field`` is a ``save_field`` STEM, without extension. Prefer a field re-extracted by
``extract_load_paths.py`` over a run's own ``load_path.npz``: archives written before ``hinge_axis``
joined ``_ARRAYS`` are zero-filled on load, which silently drops the corotated cut frame and
decomposes the interface forces in the branch frame instead.
"""

import argparse
import os

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp                                            # noqa: E402
import numpy as np                                                 # noqa: E402

from nff.utils.figstyle import (FS_ANNOT, INK, ORANGE, TEAL, TEXT_W_BP, apply_charter, paper_rc,
                                printed_pt, save_paper_figure, use_agg)
use_agg()
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D                                # noqa: E402
from matplotlib.patches import Patch                               # noqa: E402

from nff.closed.hinge_paths import load_paths                      # noqa: E402
from nff.closed.load_paths import load_field, interface_decomposition  # noqa: E402
from nff.closed.load_path_viz import (draw_load_path, zone_colours, MODE_COLOUR,  # noqa: E402
                                      CLAMP_FILL)
from nff.closed.hinge_path_binning import PROJECTIONS, densify, trajectories  # noqa: E402
from nff.models.hinge_surrogate import (load_hinge_surrogate, apply_hinge_failure,  # noqa: E402
                                        _geom_vector)


DEFAULT_SURROGATE = "data/surrogates/hinge_surrogate_pet_v3.pkl"
# PET, t = 0.5 mm, w_lig = 25 mm, theta 1.5-45 deg. The widest cached PET campaign, and therefore
# the closest to the learned ligament widths this figure reports (~19-22 mm); the 5 mm set is far
# enough away that its buckling pattern is a different regime.
DEFAULT_FRAMES = "data/outputs/hinge_frames_pet_w25.npz"
ISO_LEVELS = (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2)   # multiples of the RMS training damage
# Out-of-plane amplitude in ligament widths. The campaign's measured spread inside a w_lig band at
# these rotations is ~0.5..1.8, so these levels cross the populated part of the plane.
BUCKLE_LEVELS = (0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6)
DEFAULT_DATASET = "data/fea/hinge_dataset_pet_merged.npz"
ISO_LEGEND = {"damage": "iso-damage",
              "buckle": r"iso-buckle  $u_z/w_{\mathrm{lig}}$"}
# The contours must not read as more trajectories: same plane, same panel, and the population is
# drawn in the row's own colour. Distinguished by STYLE (dashed) as well as tone.
ISO_COLOUR = "#6A4C93"
# The plastic-strain ramp for the 3D hinge renders: near-white where the material is untouched,
# deepening through the same purple family as the iso-lines so the two damage readings in the
# figure -- the contour levels and the render -- are visibly one quantity.
DAMAGE_CMAP = LinearSegmentedColormap.from_list(
    "damage", ["#FFFFFF", "#EFEBF4", "#D8CEE6", "#B7A6CE", "#8C74B0", "#5B3E86"])
# ⚠⚠ THE CANVAS IS SIZED FOR THE PAGE, NOT FOR THE SCREEN. At the old 13.2 in width this figure was
# reduced 0.39x to reach the 372 pt (5.17 in) \textwidth, so a 10 pt label printed at 3.9 pt and the
# render annotations -- already shrunk once by embedding a 5.6 in canvas in a ~1 in inset -- printed
# near 1 pt. Author close to the printed size and every point size below IS the point size on paper.
FIG_W_IN = 7.30                     # authored width; PAGE_SCALE takes it to \textwidth
PAGE_SCALE = TEXT_W_BP / (FIG_W_IN * 72.0)   # \textwidth / authored width, both in points
def pt(printed):
    """Source point size that prints at ``printed`` pt once the figure is scaled to \textwidth.

    This plate is the one figure NOT authored at its printed size: it embeds finite-element replays
    in insets, and the canvas they need is wider than the column. The compensation is declared here
    and handed to ``paper_rc`` so the shared ladder still lands on the page at its stated sizes.
    """
    return printed_pt(printed, PAGE_SCALE)

PET_YIELD_MPA = 44.0     # first point of PET_PLASTIC in nff/rve/materials/pet.py -- the flow-curve
                         # start, i.e. where the material begins to accumulate permanent set


# RED, on the user's call: damage is the failure quantity and RED is what the charter reserves for
# failure.
#
# ⚠ THE UNDAMAGED END IS A MID GREY, NOT NEAR-WHITE. Against a near-white floor the red band had
# almost no ground to stand on and the render read as a pale wash; a mid grey gives the ramp
# somewhere to start and lets a small plastic zone register. The warm end is deepened for the same
# reason -- with matte shading there is no specular highlight left to add brightness, so the ramp
# itself has to carry the contrast it used to borrow from the light.
DAMAGE_RAMP = ("#C9C4BF", "#C9A184", "#C4703F", "#A83A1B", "#7E1810", "#4A0805")


def _yield_cmap(vmin_ratio, vmax_ratio):
    """Ramp for ``sigma_vM / sigma_yield``, spanning the PLASTIC range only.

    ⚠ The render is scaled from ``vmin_ratio`` (yield, or the lowest plastic value across the
    renders being compared) rather than from zero. Anchoring at zero spends most of the ramp on
    ELASTIC material, which is not damage at all, and leaves two hinges of genuinely different
    damage looking identical -- which is exactly what a comparison figure must not do.
    """
    return LinearSegmentedColormap.from_list("yielded", list(DAMAGE_RAMP))


REF_COLOUR = ORANGE          # shape objective only
OPT_COLOUR = TEAL            # shape and hinge-path objectives -- the project's turquoise
REF_LABEL = "shape objective only"
OPT_LABEL = "shape and hinge-path objectives"
# ★ The rows are ONE problem solved twice, and the figure has to say so. Read as two independent
# case studies, the whole argument -- that adding hinge physics to the objective moves the
# ligament strains -- is invisible. The numeral carries the sequence, the second line names what
# is held fixed, and a between-rows arrow closes it.
ROW_STEP = {REF_LABEL: ("1", "shape only"),
            OPT_LABEL: ("2", "shape + hinge physics")}
N_BINS = 70


def _zone_swatch(fields):
    """The legend patch for the compression zones: an ACTUAL tile colour from these sheets.

    Reconstructing the blend from the palette constants means the legend drifts silently every time
    the load-path figure is restyled -- it did, twice, in one afternoon. Asking ``zone_colours`` for
    the real colours and taking the most compressed tile cannot drift, because it is the same call
    the drawing makes.
    """
    best, dark = "#FFFFFF", -1.0
    for field in fields:
        for rgb in zone_colours(field, interface_decomposition(field))[0]:
            lum = 1.0 - float(np.mean(rgb))
            if lum > dark:
                best, dark = rgb, lum
    return best


class _DeployedShim:
    """Duck-typed ``Tessellation`` over a ``ForceField``, so the sheet panel can draw its supports.

    ``draw_supports_and_loads`` needs only ``.vertices`` and ``.faces[i].vertex_indices``, and a
    ``ForceField`` already carries the deployed tile polygons -- so the boundary conditions can be
    drawn from the saved field alone, with no re-solve and no rebuilt tessellation.
    """

    class _Face:
        def __init__(self, idx):
            self.vertex_indices = idx

    def __init__(self, face_nodes):
        nodes = np.asarray(face_nodes)
        n_faces, n_nodes, _ = nodes.shape
        self.vertices = nodes.reshape(-1, 2)
        self.faces = [self._Face(np.arange(f * n_nodes, (f + 1) * n_nodes)) for f in range(n_faces)]


def worst_hinge(paths) -> int:
    """Index of the ligament whose trajectory reaches deepest into axial compression."""
    return int(np.argmin(np.asarray(paths.eta)[..., 0].min(axis=0)))


def path_damage(paths, surrogate) -> np.ndarray:
    """(n_hinges,) path-MAX surrogate damage, each ligament read with its OWN w_lig and alpha.

    The quantity the damage loss acts on, so selecting on it picks the ligament the objective is
    actually about -- unlike the compression criterion, which picks the most squeezed one and can
    miss a ligament that is merely rotating itself to death.
    """
    net_params, stats, _ = surrogate
    eta = np.asarray(paths.eta)                       # (steps, hinges, 3)
    w_lig = np.asarray(paths.w_lig)[None, :]
    u = jnp.stack([jnp.asarray(eta[..., 0] * w_lig), jnp.asarray(eta[..., 1] * w_lig),
                   jnp.asarray(eta[..., 2])], axis=-1)
    g = _geom_vector(jnp.broadcast_to(jnp.asarray(w_lig), eta.shape[:2]),
                     jnp.broadcast_to(jnp.asarray(paths.alpha)[None, :], eta.shape[:2]), None, stats)
    D = np.asarray(apply_hinge_failure(net_params, u, g, stats))
    return D.max(axis=0) / float(np.asarray(stats.get("D_scale", 1.0)))


def most_damaged_hinge(paths, surrogate) -> int:
    """Index of the ligament the surrogate scores as most damaged over its whole path."""
    return int(np.argmax(path_damage(paths, surrogate)))


def traj_theta(paths, hinge) -> float:
    """The deployed hinge rotation [rad] of one ligament -- the slice the iso-damage is taken at."""
    return float(np.asarray(paths.eta)[-1, hinge, 2])


def _clouds(paths_pair, proj, n_resample):
    """Per design: ``(points, weights)`` for the projected distributions.

    ★★ THE WEIGHTS ARE LOAD-BEARING, and their absence was a real distortion.

    ``densify`` gives EVERY path the same number of points regardless of how far it travels. A
    ligament whose excursion is 6x the typical one therefore spreads its 400 points over 6x the
    distance and contributes 6x LESS density per unit length -- so the marginal SUPPRESSES exactly
    the extreme paths the eye picks out of the scatter. Measured on the 5x5 pair: the reference
    design's 1st-percentile eta_a reads -0.508 unweighted against -0.777 length-weighted (a 53%
    understatement), while the regularized design, whose paths are all of similar length, is
    unaffected (-0.1095 vs -0.1100). The error was therefore ASYMMETRIC between the two rows --
    the worst kind, because it made two very different clouds produce near-identical marginals.

    Weighting each point by its own path's arc length makes the marginal a true LINE DENSITY over
    the plane, which is what the scatter above it shows.
    """
    out = []
    for paths in paths_pair:
        pts, ids = densify(paths, proj, n_resample)
        traj = trajectories(paths, proj)
        length = np.linalg.norm(np.diff(traj, axis=1), axis=2).sum(axis=1)   # (n_hinges,)
        w = length[np.asarray(ids)] / max(n_resample, 1)
        out.append((pts, w))
    return out


def _smooth_density(values, bins, smooth=2.5, weights=None):
    """Binned density smoothed by a Gaussian kernel -> ``(bin centres, density)``.

    A kernel density estimate, done as histogram-then-convolve so it needs nothing beyond numpy.
    ``smooth`` is the kernel width in BINS, so the amount of smoothing follows the bin count rather
    than the strain scale, and the two axes are smoothed comparably without tuning each.

    Zero padding at the ends is correct here: the bin range is the plotted window, which already
    extends past the data, so the density genuinely is zero there.
    """
    counts, edges = np.histogram(values, bins=bins, density=True, weights=weights)
    radius = int(np.ceil(3.0 * smooth))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / smooth) ** 2)
    kernel /= kernel.sum()
    return 0.5 * (edges[:-1] + edges[1:]), np.convolve(counts, kernel, mode="same")


def _marginal(ax, cloud, component, bins, colour, *, horizontal, dens_lim):
    """Projected density of ONE design along one strain component, as a filled area.

    ``dens_lim`` is shared across the rows on purpose. Each row now draws only its own design, so
    without a pinned density axis every panel would autoscale to its own peak and two distributions
    of very different spread would render at the same height -- the comparison the layout exists for
    would be silently destroyed.
    """
    pts, weights = cloud
    centres, dens = _smooth_density(pts[:, component], bins, weights=weights)
    if horizontal:
        ax.fill_betweenx(centres, 0.0, dens, color=colour, alpha=0.30, lw=0.0, zorder=3)
        ax.plot(dens, centres, color=colour, lw=1.5, zorder=4)
        ax.set_xlim(0.0, dens_lim)
    else:
        ax.fill_between(centres, 0.0, dens, color=colour, alpha=0.30, lw=0.0, zorder=3)
        ax.plot(centres, dens, color=colour, lw=1.5, zorder=4)
        ax.set_ylim(0.0, dens_lim)
    zero = ax.axhline if horizontal else ax.axvline
    zero(0.0, color=INK, lw=0.9, ls="--", alpha=0.55, zorder=2)

    ax.grid(False)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    # The strain axis is SHARED with the main panel, so clearing its ticks here would clear them
    # there too -- hide the labels only. The density axis is this panel's own and can go entirely.
    if horizontal:
        ax.set_ylim(bins[0], bins[-1])
        ax.set_xticks([])
        plt.setp(ax.get_yticklabels(), visible=False)
    else:
        ax.set_xlim(bins[0], bins[-1])
        ax.set_yticks([])
        plt.setp(ax.get_xticklabels(), visible=False)


def damage_field(surrogate, w_lig, alpha, theta, xlim, ylim, n=220):
    """The surrogate's damage over this window -> ``(eta_a grid, eta_s grid, D / D_scale)``.

    ⚠ THIS IS A SLICE, and it has to be. ``D`` is a function of ``(a, s, theta; w_lig, alpha)``,
    while the plane drawn here is two-dimensional -- so the field holds ``theta`` and the ligament
    geometry FIXED at the monitored ligament's own deployed values. A trajectory crossing a level is
    therefore not reading its own damage except at the moment it reaches that ``theta``; the field
    says what the landscape looks like around the deployed state, not along the path.

    Because the panel spans the same window in both rows while ``w_lig`` is a LEARNED per-hinge
    quantity, the two rows generally carry different fields. That is the physics, not an
    inconsistency: eta = a / w_lig, so a wider ligament puts the same eta at a larger displacement.
    ``build_figure(iso_common=True)`` pins both rows to one ligament when a fixed backdrop is wanted.

    Values are in multiples of the checkpoint's RMS training damage (``D_scale``), the same
    normalisation the damage loss uses, so one is a hinge as damaged as the average training sample
    rather than a fraction of fracture.
    """
    net_params, stats, _ = surrogate
    EA, ES = np.meshgrid(np.linspace(xlim[0], xlim[1], n), np.linspace(ylim[0], ylim[1], n))
    shape = EA.shape
    u = jnp.stack([jnp.asarray(EA * w_lig), jnp.asarray(ES * w_lig),
                   jnp.full(shape, float(theta))], axis=-1)
    g = _geom_vector(jnp.full(shape, float(w_lig)), jnp.full(shape, float(alpha)), None, stats)
    field = np.asarray(apply_hinge_failure(net_params, u, g, stats))
    return EA, ES, field / float(np.asarray(stats.get("D_scale", 1.0)))


def is_replay(frames) -> bool:
    """True when this archive was driven along a MEASURED ``(a, s, theta)`` polyline.

    A ray archive is a pure-rotation sweep and is therefore blind to the axial and shear components
    -- the very ones a hinge-path objective moves. Two designs read off a ray cache render as
    neighbouring frames of one animation no matter how differently their ligaments were strained.
    """
    return str(frames.get("drive", "ray")) == "path"


def paint_field(frames, k):
    """The nodal field a render paints -> ``(values, kind)``, ``kind in {'peeq', 'yield'}``.

    Prefers accumulated plastic strain, which IS the damage measure. Falls back to
    ``sigma_vM / sigma_yield`` for archives written before ``peeq_nodal`` existed.

    ⚠ The fallback cannot separate two hinges of different damage: PET's flow curve is nearly flat,
    so von Mises SATURATES a few percent above yield while plastic strain keeps accumulating.
    """
    peeq = np.asarray(frames.get("peeq_nodal", np.zeros(0)), float)
    if peeq.size and np.isfinite(peeq).any() and np.nanmax(peeq) > 0:
        return peeq[k], "peeq"
    return np.asarray(frames["vm"])[k] / PET_YIELD_MPA, "yield"


FIELD_LABEL = {"peeq": r"plastic strain  PEEQ", "yield": r"$\sigma_\mathrm{vM}/\sigma_y$"}
# The colour key already names the field, so the title only has to say WHAT THE OBJECT IS. A replay
# is the monitored ligament itself; a ray is a stand-in at the same rotation, and must say so.
INSET_TITLE = {True: "the monitored ligament, deployed",
               False: "a generic hinge at the same rotation"}


def buckle_field(dataset, w_lig, alpha, theta, xlim, ylim, n=220, *,
                 theta_tol=4.0, w_band=3.0):
    """MEASURED out-of-plane amplitude over this window -> ``(eta_a grid, eta_s grid, uz / w_lig)``.

    ★ THIS IS DATA, NOT A MODEL. The contours are a Delaunay interpolation of the CalculiX campaign's
    own ``uz_max`` over the ``(eta_a, eta_s)`` samples near this rotation and ligament width -- so,
    unlike the surrogate's damage field, it asserts nothing outside the convex hull of jobs actually
    solved. ``LinearNDInterpolator`` returns NaN there and the contours simply stop, which is the
    honest boundary and should be left visible.

    ⚠ THE NORMALISATION IS ``uz / w_lig``, NOT ``uz / t``. Buckling amplitude scales with the
    buckling member, and the campaign swept ``w_lig`` 5-25 mm at fixed gauge. Measured by holding
    out half the jobs and interpolating from the rest (theta 41 +-4 deg, w_lig 16-20 mm):
    ``uz/w_lig`` reproduces to 2.7% median / 11.0% p90, against ``uz/t`` at 6.9% / 12.7%. Pooling a
    w_lig band under the ``uz/t`` scaling is what makes the wider band degrade to 10.9% median.

    ``alpha`` is accepted for signature symmetry with :func:`damage_field` and is deliberately not
    filtered on: the campaign's alpha spread inside a w_lig band is narrow, and adding a third
    filter empties the neighbourhood.
    """
    from scipy.interpolate import LinearNDInterpolator

    th_deg, w = np.degrees(float(theta)), float(w_lig)
    m = (np.abs(dataset["theta_deg"] - th_deg) <= theta_tol) & (np.abs(dataset["w_lig"] - w) <= w_band)
    if m.sum() < 12:
        return None
    pts = np.c_[dataset["eta_a"][m], dataset["eta_s"][m]]
    vals = dataset["uz_over_w"][m]
    EA, ES = np.meshgrid(np.linspace(xlim[0], xlim[1], n), np.linspace(ylim[0], ylim[1], n))
    field = LinearNDInterpolator(pts, vals)(np.c_[EA.ravel(), ES.ravel()]).reshape(EA.shape)
    return EA, ES, _nan_smooth(field, 0.02 * n)


def _nan_smooth(field, sigma):
    """Gaussian-smooth a field that is NaN outside the data hull, WITHOUT bleeding across the edge.

    A Delaunay interpolation is piecewise linear on the triangulation, so its contours come out as
    chains of straight facets that read as scribble rather than as iso-lines -- the eye takes the
    triangulation for structure in the data. Smoothing removes the facets while leaving the level
    values alone.

    Plain smoothing would treat the NaN exterior as zero and drag the contours inward at the hull
    boundary. Normalising by the smoothed VALIDITY MASK makes each output a weighted mean over the
    valid neighbours only, and the original NaN region is restored afterwards so the honest edge of
    the measured data stays exactly where it was.
    """
    from scipy.ndimage import gaussian_filter

    valid = np.isfinite(field)
    num = gaussian_filter(np.where(valid, field, 0.0), sigma, mode="nearest")
    den = gaussian_filter(valid.astype(float), sigma, mode="nearest")
    out = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 1e-6)
    return np.where(valid, out, np.nan)


def load_buckle_dataset(path):
    """The campaign npz reduced to what :func:`buckle_field` needs."""
    z = np.load(path)
    w = np.asarray(z["w_lig"], float)
    return {"eta_a": np.asarray(z["a"], float) / w, "eta_s": np.asarray(z["s"], float) / w,
            "theta_deg": np.asarray(z["theta_deg"], float), "w_lig": w,
            "uz_over_w": np.asarray(z["uz_max"], float) / w}


def _trim(img, tol=0.004, pad=2):
    """Crop the uniform margin off a rendered tile, so three views pack without dead space."""
    a = np.asarray(img, float)
    ink = a[..., 3] > tol if a.shape[-1] == 4 else (a[..., :3].std(axis=-1) + (1.0 - a[..., :3].mean(axis=-1))) > tol
    rows, cols = np.where(ink.any(axis=1))[0], np.where(ink.any(axis=0))[0]
    if not rows.size or not cols.size:
        return img
    r0, r1 = max(0, rows[0] - pad), min(img.shape[0], rows[-1] + 1 + pad)
    c0, c1 = max(0, cols[0] - pad), min(img.shape[1], cols[-1] + 1 + pad)
    return img[r0:r1, c0:c1]


def _stack_views(tiles, gap=10):
    """One tall strip: the oblique view on top, then the plan and the elevation beneath it.

    Each tile is trimmed and rescaled to a common width so the three read as one object seen three
    ways rather than three separate pictures at three separate scales.
    """
    cut = [_trim(t) for t in tiles]
    W = max(c.shape[1] for c in cut)
    rows = []
    for c in cut:
        if c.shape[1] != W:                      # nearest-neighbour rescale: no scipy dependency
            idx = (np.arange(W) * c.shape[1] / W).astype(int)
            h = max(1, int(round(c.shape[0] * W / c.shape[1])))
            jdx = (np.arange(h) * c.shape[0] / h).astype(int)
            c = c[jdx][:, idx]
        rows.append(c if c.shape[-1] == 4 else np.dstack([c, np.ones(c.shape[:2])]))
    sep = np.zeros((gap, W, 4))                  # transparent gutter, not a white bar
    out = [rows[0]]
    for r in rows[1:]:
        out += [sep, r]
    return np.vstack(out)


def valid_frames(frames):
    """Indices of frames the oracle actually CONVERGED, newest-usable last.

    ⚠⚠ A REPLAY CAN END ON A DIVERGED INCREMENT AND STILL LOOK COMPLETE. CalculiX writes the failed
    increment before it gives up, so the archive has one more frame than it has physics. Measured on
    ligament 23 of the shape arm: ``uz`` ran 14.9 -> 14.3 mm in ~0.2 mm steps and then jumped to
    63.74 mm, with nodal PEEQ all NaN -- and because the inset picks the frame NEAREST IN ROTATION,
    that blown-up frame is precisely the one the figure would have drawn.

    Two independent tests, because either can fire alone: a non-finite field, and an out-of-family
    jump in the buckle amplitude. Everything after the first bad increment is dropped too -- a solve
    does not recover once it has diverged.
    """
    n = int(len(np.asarray(frames["theta_deg"])))
    ok = np.ones(n, bool)
    for key in ("peeq_nodal", "vm"):
        a = np.asarray(frames.get(key, np.zeros(0)), float)
        if a.size and a.shape[0] == n:
            ok &= np.isfinite(a.reshape(n, -1)).all(axis=1)
    uz = np.asarray(frames.get("uz_max", np.zeros(0)), float)
    if uz.size == n and n > 4:
        d = np.abs(np.diff(uz))
        med = float(np.median(d[np.isfinite(d)])) if np.isfinite(d).any() else 0.0
        for b in np.where(~np.isfinite(d) | (d > max(20.0 * med, 1e-12)))[0]:
            ok[b + 1:] = False
    idx = np.where(ok)[0]
    return idx if idx.size else np.arange(n)


def hinge_render_inset(ax, frames, theta_rad, tip, colour, *, field_lim, field_kind, tmpdir,
                       size=0.38, elev=34.0, azim=-70.0):
    """A 3D render of the RVE hinge, pinned beside the trajectory tip.

    Reuses ``render_hinge_3d.render`` -- the existing renderer -- rather than drawing a second
    3D hinge.

    ★ PREFER A REPLAY ARCHIVE (``deploy_hinge_frames.py --paths ... --hinge k``). Then the frames
    were solved along this ligament's own measured ``(a, s, theta)`` route at its own ``w_lig`` and
    ``alpha``, the render is the ligament rather than a stand-in, and -- because the oracle is
    elastoplastic and therefore path-dependent -- the plastic strain in the picture is the plastic
    strain the ligament actually has.

    ⚠ WITH A RAY ARCHIVE the render is a CACHED, PURE-ROTATION RVE at a fixed ligament width, and
    the frame nearest in ROTATION is picked. It shows what a PET hinge at this rotation looks like,
    not what this ligament did; two rows differing only in shear will render identically. State the
    frame's ``w_lig`` when quoting it.

    ``field_lim`` is shared by every inset in the figure, so a colour means the same value in each.
    """
    from nff.scripts.figures.render_hinge_3d import render

    theta_deg = np.degrees(theta_rad)
    usable = valid_frames(frames)                      # never draw a diverged increment
    k = int(usable[np.argmin(np.abs(np.asarray(frames["theta_deg"])[usable] - theta_deg))])
    png = os.path.join(tmpdir, f"hinge_{id(frames)}_{k}_{int(round(theta_deg))}.png")
    field, _ = paint_field(frames, k)
    vmin, vmax = field_lim
    # ZOOM to the hinge. The RVE is a 100 mm Saint-Venant window around a ~25 mm ligament, so the
    # default full-mesh frame renders the hinge as a speck in a field of flat sheet. The window is
    # centred on the LIGAMENT by the renderer, not on the origin -- the two are tens of mm apart.
    w_lig = float(frames["w_lig"])
    # THREE CAMERAS on one state. A single oblique view cannot separate the two things this hinge
    # does at once: the plan view carries the in-plane opening, the elevation carries the
    # out-of-plane buckle (here uz reaches ~45x the sheet gauge), and the oblique view is the only
    # one in which the two read as one deformed body. Shown together they disambiguate each other --
    # material that overlaps in one projection is separated in another.
    # TWO orthogonal views, no oblique. An oblique camera shows both the opening and the buckle at
    # once but names neither: without axes the reader cannot tell which way is "up", so the fold
    # reads as an arbitrary shape. A plan and an elevation are unambiguous because each kills one
    # axis -- the plan is the in-plane mechanism, the elevation is the out-of-plane buckle.
    views = (("top", 89.0, azim), ("side", 2.0, azim))
    tiles = []
    for name, ev, az in views:
        pv = os.path.join(tmpdir, f"hinge_{id(frames)}_{k}_{name}.png")
        render(frames["xyz"], frames["conn"], frames["disp"][k], w_lig,
               float(frames["thickness"]), float(frames.get("r_win", 100.0)), pv,
               vm=field, vmin=vmin, vmax=vmax, scale=False, view_r=1.15 * w_lig,
               cmap=_yield_cmap(vmin, vmax), label=FIELD_LABEL[field_kind], shading="depth",
               elev=ev, azim=az, subdiv=2, dpi=400, outline=True, annotate=True,
               figsize=(1.80, 1.50), annotate_pt=pt(10.5))
        tiles.append(plt.imread(pv))

    # The box is sized FROM the composite: a three-view strip is tall, and an inset kept at the
    # single-view aspect would letterbox it into a sliver with white bars top and bottom.
    strip = _stack_views(tiles)
    ax_w, ax_h = ax.get_window_extent().width, ax.get_window_extent().height
    box_h = size * 1.42
    box_w = box_h * (strip.shape[1] / strip.shape[0]) * (ax_h / ax_w)
    x0, y0 = _clearest_slot(ax, box_w, box_h)
    inset = ax.inset_axes([x0, y0, box_w, box_h], zorder=20)
    inset.imshow(strip)
    inset.set_xticks([]); inset.set_yticks([]); inset.grid(False)
    # ALL FOUR spines: the project rcParams switch off top and right, which on a bare image axes
    # leaves an L-shaped bracket rather than a frame.
    for name in ("top", "right", "bottom", "left"):
        inset.spines[name].set_visible(True)
        inset.spines[name].set_edgecolor(colour)
        inset.spines[name].set_linewidth(1.3)
    inset.patch.set_facecolor("#FFFFFF")

    # A straight leader from the deployed state to the render, so the pairing survives the inset
    # being placed wherever there happens to be room rather than beside the tip.
    # BLACK, not the row colour: the leader is a pointer, not another trajectory, and in the row's
    # own colour it read as one more path leaving the cloud.
    ax.annotate("", xy=(x0 + box_w / 2, y0 + box_h * 0.5), xytext=_to_axes(ax, tip)[0],
                xycoords=ax.transAxes, textcoords=ax.transAxes, zorder=19,
                arrowprops=dict(arrowstyle="-", color=INK, lw=1.0, alpha=0.85))
    if not is_replay(frames):          # a REPLAY needs no caption: it is the row's own ligament,
        # which the leader line already states. Only the generic RAY has to disclaim itself.
        inset.set_title(INSET_TITLE[False], fontsize=pt(FS_ANNOT), color=INK, pad=3)
    return float(frames["theta_deg"][k]), w_lig, field_kind, is_replay(frames)


def _to_axes(ax, pts):
    """Data -> axes fraction, going through the axis scale itself.

    Composing transData with the inverse of transAxes, rather than using ``ax.transLimits``, which
    is a linear map built from the view limits and therefore only happens to be right on a linear
    axis. A leader line computed the other way shot off-canvas and, with ``bbox_inches='tight'``,
    dragged the whole figure with it.
    """
    return ax.transAxes.inverted().transform(ax.transData.transform(np.atleast_2d(pts)))


def _clearest_slot(ax, w, h, margin=0.02):
    """Axes-fraction ``(x0, y0)`` for a ``w x h`` box in the EMPTIEST part of the panel.

    The inset used to be pinned to a fixed corner, which put it on top of the data whenever the
    trajectories happened to run that way. Every already-drawn line and marker is projected into
    axes coordinates and each candidate slot is scored by how many of those points it would cover;
    the legend's own corner is excluded by construction.
    """
    pts = []
    for line in ax.get_lines():
        xy = line.get_xydata()
        if len(xy):
            pts.append(_to_axes(ax, xy))
    if not pts:
        return 1.0 - margin - w, 1.0 - margin - h
    P = np.concatenate(pts, axis=0)
    P = P[np.all(np.isfinite(P), axis=1)]

    best, best_score = (1.0 - margin - w, 1.0 - margin - h), np.inf
    for gx in np.linspace(margin, 1.0 - margin - w, 9):
        for gy in np.linspace(margin, 1.0 - margin - h, 9):
            inside = np.count_nonzero((P[:, 0] >= gx - 0.02) & (P[:, 0] <= gx + w + 0.02) &
                                      (P[:, 1] >= gy - 0.02) & (P[:, 1] <= gy + h + 0.02))
            # The legend sits lower-left; bias against that corner so the two never collide.
            penalty = 400 if (gx < 0.42 and gy < 0.30) else 0
            score = inside + penalty
            if score < best_score:
                best, best_score = (float(gx), float(gy)), score
    return best


def _draw_iso(ax, grid, iso_levels, fmt="%g"):
    """Iso-lines of the backdrop field over the strain plane.

    LINES ONLY. A filled wash of the same field was tried and dropped: it dominated the panel,
    and since the damage field extrapolates hard outside the checkpoint's training box, the largest,
    boldest area of colour was the part the model knows least about. Labelled contours state the
    same landscape without asserting a value at every pixel.

    ⚠ THE CONTOURS ARE THE BACKDROP AND MUST STAY BEHIND THE DATA. They were previously heavier
    than the trajectory population they sit under, so the panel read as a contour map with some
    faint lines on it rather than as a cloud of paths against a landscape. Keep them thin, pale and
    dashed; if they compete with the paths again, lighten these, do not darken the paths.

    ``np.nanmin``/``nanmax``: an interpolated MEASURED field is NaN outside the convex hull of the
    jobs actually solved, and those NaNs are the honest edge of the data.
    """
    EA, ES, field = grid
    lo, hi = float(np.nanmin(field)), float(np.nanmax(field))
    levels = [lv for lv in iso_levels if lo < lv < hi]
    if levels:
        cs = ax.contour(EA, ES, field, levels=levels, colors=ISO_COLOUR, linewidths=0.7,
                        linestyles="--", alpha=0.55, zorder=1)
        # pt(): a bare point size on this canvas is NOT a printed one. Left raw, these iso labels
        # printed at 5.3 pt, under the legibility floor.
        ax.clabel(cs, inline=True, fontsize=pt(FS_ANNOT), colors=ISO_COLOUR, fmt=fmt)
    return levels, (lo, hi)


def _smooth_traj(traj, n_out=110, window=19):
    """Resample each ``(n_steps, 2)`` curve by arc length and smooth it, endpoints pinned.

    The raw curve has one vertex per LOAD STEP, and the solver does not place those vertices evenly:
    a ligament that stalls and then snaps leaves a long straight run followed by a hard corner, so
    the path reads as a broken line when the mechanism it describes is continuous. Resampling by arc
    length puts the vertices where the curve actually travels, and a small odd-length moving average
    removes the step-to-step jitter left by the equilibrium tolerance.

    ⚠ COSMETIC ONLY. Every quantity the figure REPORTS -- compression depth, endpoints, the
    distributions on the margins -- is computed from the raw ``trajectories`` array, never from
    this. Smoothing what is measured would be a different claim than smoothing what is drawn.
    """
    out = np.empty((traj.shape[0], n_out, 2))
    for h in range(traj.shape[0]):
        c = traj[h]
        seg = np.linalg.norm(np.diff(c, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        if s[-1] <= 0:                                  # never moved: a point, not a path
            out[h] = c[0]
            continue
        g = np.linspace(0.0, s[-1], n_out)
        r = np.stack([np.interp(g, s, c[:, k]) for k in (0, 1)], axis=-1)
        if window > 1 and n_out > window:
            k = np.ones(window) / window
            pad = window // 2
            for j in (0, 1):
                pj = np.pad(r[:, j], pad, mode="edge")
                r[:, j] = np.convolve(pj, k, mode="valid")
            r[0], r[-1] = c[0], c[-1]                   # origin and deployed state are data
        out[h] = r
    return out


def _strain_panel(main, top, right, paths, monitored, colour, xlim, ylim, bins_a, bins_s,
                  cloud, dens_lims, proj, grid=None, norm=None, frames=None, field_lim=None,
                  field_kind="yield", tmpdir=None, iso_levels=ISO_LEVELS):
    """One design's trajectories, with its projected distributions on the axes."""
    traj = trajectories(paths, proj)

    iso = _draw_iso(main, grid, iso_levels) if grid is not None else ([], (0.0, 0.0))
    # The zero line survives the compression wash it used to accompany: it is the axis reference
    # separating a ligament being pulled from one being squeezed, and it is a LINE, not a region.
    main.axvline(0.0, color=INK, lw=0.9, ls="--", alpha=0.55, zorder=1.5)
    main.axhline(0.0, color=INK, lw=0.8, ls=":", alpha=0.45, zorder=1.5)

    drawn = _smooth_traj(traj)                     # for the CURVES only; endpoints stay raw below
    others = [h for h in range(traj.shape[0]) if h != monitored]
    # The population must out-read the contour backdrop it sits on. At alpha 0.28 it did not: the
    # paths are the subject and the iso-lines are the setting, and the panel said the opposite.
    for h in others:
        main.plot(drawn[h, :, 0], drawn[h, :, 1], color=colour, alpha=0.55, lw=1.25, zorder=2)
    # Every ligament's DEPLOYED state, marked. The trajectories all radiate from one origin, so
    # without endpoints there is nothing to say which end of a curve is the deployment -- and the
    # deployed states are the population the endpoint view of the compression-shift figure bins.
    ends = traj[others, -1, :]
    main.plot(ends[:, 0], ends[:, 1], 'o', ms=3.6, color=colour, alpha=0.9,
              markeredgecolor="#FFFFFF", markeredgewidth=0.6, zorder=3, linestyle='none')

    e = traj[monitored]
    main.plot(drawn[monitored, :, 0], drawn[monitored, :, 1], color=colour, lw=2.6, zorder=4,
              solid_capstyle="round")
    # ONE mark, hollow and black. The filled marker that used to sit under this ring added a white
    # collar and a second colour to a symbol whose whole job is to be unambiguous.
    main.plot(e[-1, 0], e[-1, 1], 'o', ms=8.5, markerfacecolor='none', markeredgecolor=INK,
              markeredgewidth=1.5, zorder=7)

    main.set_xlim(*xlim)
    main.set_ylim(*ylim)
    main.set_xlabel(r"$\eta_a = a\,/\,w_{\mathrm{lig}}$")
    main.set_ylabel(r"$\eta_s = s\,/\,w_{\mathrm{lig}}$")

    _marginal(top, cloud, 0, bins_a, colour, horizontal=False, dens_lim=dens_lims[0])
    _marginal(right, cloud, 1, bins_s, colour, horizontal=True, dens_lim=dens_lims[1])

    # Drawn LAST: the inset is placed from the axes limits, which must already be final.
    render_info = None
    if frames is not None:
        render_info = hinge_render_inset(main, frames, float(traj_theta(paths, monitored)),
                                         (e[-1, 0], e[-1, 1]), colour, field_lim=field_lim,
                                         field_kind=field_kind, tmpdir=tmpdir)
    return iso, render_info


def build_figure(ref_paths, ref_field, opt_paths, opt_field, out_png, *, hinge=None,
                 n_resample=400, surrogate=None, iso_common=False, frames=None,
                 ref_frames=None, opt_frames=None, iso="damage", dataset=None,
                 select='damage', ref_config=None, opt_config=None,
                 ref_tess=None, opt_tess=None):
    """The 2x2 case study. Returns ``(out_png, hinge_index, stats)``.

    Args:
        ref_paths, opt_paths: ``HingePaths`` of the two designs.
        ref_field, opt_field: their ``ForceField``s, for the load-path panels.
        hinge: index to follow; ``None`` picks it by ``select``.
        select: ``'damage'`` follows the ligament the surrogate scores as MOST DAMAGED over its
            path in the reference design -- the quantity the objective acts on. ``'compression'``
            follows the most axially squeezed one instead. Ignored when ``hinge`` is given.
        iso_common: draw the SAME iso-damage contours in both rows, taken at the reference
            design's ligament. Off by default, and the choice is not cosmetic -- see
            :func:`_iso_damage`. Per-row contours describe each design's own ligament but give no
            common yardstick; a common slice is a fixed backdrop, but then the lower row's contours
            do not describe the ligament drawn on top of them.
        frames: a loaded ``hinge_frames_*.npz`` mapping used for BOTH rows. Supplying it puts a 3D
            render of the RVE hinge at each monitored trajectory's tip.
        ref_frames, opt_frames: per-row archives, overriding ``frames`` for that row. ★ THIS IS THE
            ONE THAT MAKES THE RENDERS COMPARABLE. Point each at a REPLAY archive
            (``deploy_hinge_frames.py --paths <that design> --hinge <k>``) and each render is that
            design's own ligament driven along its own measured route. With one shared ray archive
            both rows resolve to neighbouring frames of a single pure-rotation sweep and render
            identically however differently the two ligaments were actually strained.
        ref_config, opt_config: the runs' configs. Supplying them draws the CLAMPS and the LOAD
            ARROWS on that row's sheet -- without them a reader cannot tell which tile is held and
            which is pulled. The tessellation they need is reconstructed from the force field
            (:class:`_DeployedShim`), so no re-solve is required.
        iso: which landscape the trajectories are read against. ``'damage'`` is the SURROGATE's
            damage, a model extrapolating outside its training box. ``'buckle'`` is the CalculiX
            campaign's own measured out-of-plane amplitude ``uz/w_lig``, which asserts nothing
            outside the hull of jobs actually solved -- see :func:`buckle_field`. ``'none'`` omits
            the backdrop.
        dataset: the campaign npz reduced by :func:`load_buckle_dataset`; required for
            ``iso='buckle'``.
        ref_tess, opt_tess: real ``Tessellation`` objects, if a caller happens to have them.
    """
    if ref_paths.n_hinges != opt_paths.n_hinges:
        raise ValueError(f"designs disagree on hinge count: {ref_paths.n_hinges} vs "
                         f"{opt_paths.n_hinges} -- they are not the same tessellation")
    for name, paths, field in (("reference", ref_paths, ref_field), ("optimized", opt_paths, opt_field)):
        if paths.n_hinges != np.asarray(field.hinge_xy).shape[0]:
            raise ValueError(f"{name}: hinge paths carry {paths.n_hinges} ligaments but its force "
                             f"field carries {np.asarray(field.hinge_xy).shape[0]}")

    import tempfile

    apply_charter(**paper_rc(scale=PAGE_SCALE))
    proj = PROJECTIONS['eta_a_eta_s']
    if hinge is not None:
        monitored = int(hinge)
    elif surrogate is not None and select == 'damage':
        monitored = most_damaged_hinge(ref_paths, surrogate)
    else:
        monitored = worst_hinge(ref_paths)

    # ONE window and ONE set of bins over both designs: rows that autoscale separately cannot be
    # compared, which is the only thing this layout exists to let the reader do.
    clouds = _clouds((ref_paths, opt_paths), proj, n_resample)
    allpts = np.concatenate([c[0] for c in clouds], axis=0)
    lo, hi = allpts.min(axis=0), allpts.max(axis=0)
    pad = 0.07 * (hi - lo)
    xlim = (lo[0] - pad[0], hi[0] + pad[0])
    ylim = (lo[1] - pad[1], hi[1] + pad[1])
    bins_a = np.linspace(xlim[0], xlim[1], N_BINS)
    bins_s = np.linspace(ylim[0], ylim[1], N_BINS)
    dens_lims = tuple(1.06 * max(_smooth_density(c[0][:, comp], bins, weights=c[1])[1].max()
                                 for c in clouds)
                      for comp, bins in ((0, bins_a), (1, bins_s)))

    # The landscape each row is read against. Levels are ABSOLUTE in both modes, so the rows are
    # comparable without a shared colour scale.
    grids, iso_levels = (None, None), ISO_LEVELS
    geoms = [(float(p.w_lig[monitored]), float(p.alpha[monitored]), float(traj_theta(p, monitored)))
             for p in (ref_paths, opt_paths)]
    if iso_common:
        geoms = [geoms[0], geoms[0]]
    if iso == "buckle" and dataset is not None:
        iso_levels = BUCKLE_LEVELS
        grids = tuple(buckle_field(dataset, *g, xlim, ylim) for g in geoms)
    elif iso == "damage" and surrogate is not None:
        grids = tuple(damage_field(surrogate, *g, xlim, ylim) for g in geoms)

    row_frames = (ref_frames if ref_frames is not None else frames,
                  opt_frames if opt_frames is not None else frames)

    # ONE scale over both renders, so a colour means the same thing in each.
    #
    # WHERE THE RAMP STARTS depends on WHICH FIELD is painted, and the two cases are opposite.
    # Plastic strain: zero IS undamaged, so anchoring at zero is exactly right and the blank areas
    # are a true statement. Von Mises: zero is unstressed-but-elastic, which is not damage at all --
    # anchoring there spends most of the ramp below yield and makes two different hinges look the
    # same, so that fallback starts at yield instead.
    field_lim, field_kind = None, "yield"
    # SAME frame choice as the inset, diverged increments excluded -- otherwise the shared colour
    # scale is set from a frame the render never draws, and on a NaN field `nanmax` would silently
    # take the limit from the wrong increment.
    def _pick(f, p):
        u = valid_frames(f)
        return int(u[np.argmin(np.abs(np.asarray(f["theta_deg"])[u]
                                      - np.degrees(traj_theta(p, monitored))))])
    picks = [(f, _pick(f, p))
             for f, p in zip(row_frames, (ref_paths, opt_paths)) if f is not None]
    if picks:
        vals = [paint_field(f, k) for f, k in picks]
        # If the two archives disagree on which field they carry, fall back to the one BOTH have --
        # a shared scale across two different quantities would be meaningless.
        field_kind = "peeq" if all(kind == "peeq" for _, kind in vals) else "yield"
        if field_kind == "yield":
            vals = [(np.asarray(f["vm"])[k] / PET_YIELD_MPA, "yield") for f, k in picks]
        hi = float(max(np.nanmax(v) for v, _ in vals))
        lo = 0.0 if field_kind == "peeq" else 1.0
        field_lim = (lo, max(hi, lo + 0.02))
    frames_present = field_lim is not None

    fig = plt.figure(figsize=(FIG_W_IN, FIG_W_IN * 0.94), dpi=300)
    # The sheet is much taller than it is wide, so a left column sized like the right one is mostly
    # margin; `_fit_axes` fills the height either way.
    # `bottom` leaves a clear strip for the renders' shared colour key; without it the bar lands on
    # the lower panel's x-axis label.
    outer = fig.add_gridspec(2, 2, width_ratios=[0.62, 1.0], height_ratios=[1.0, 1.0],
                             left=0.075, right=0.99, top=0.945,
                             bottom=0.105 if frames_present else 0.06,
                             wspace=0.08, hspace=0.22)

    rows = ((0, ref_paths, ref_field, REF_COLOUR, REF_LABEL, clouds[0], grids[0], ref_config,
             ref_tess, row_frames[0]),
            (1, opt_paths, opt_field, OPT_COLOUR, OPT_LABEL, clouds[1], grids[1], opt_config,
             opt_tess, row_frames[1]))
    stats, sheet_pos = {}, []
    tmp = tempfile.TemporaryDirectory()
    tmpdir = tmp.name
    for r, paths, field, colour, label, cloud, grid, cfg, tess, row_fr in rows:
        sheet = fig.add_subplot(outer[r, 0])
        stats[label] = draw_load_path(sheet, field, config=cfg,
                                      tessellation=tess or _DeployedShim(field.face_nodes),
                                      highlight=monitored, highlight_colour=INK,
                                      clamp_as_fill=True)

        # FLUSH, not spaced: a marginal separated from its panel by a gap floats, and the reader
        # has to carry the axis across the gap by eye. Sharing the axis is not enough -- the two
        # have to touch for the alignment to be visible.
        inner = outer[r, 1].subgridspec(2, 2, width_ratios=[1.0, 0.15],
                                        height_ratios=[0.15, 1.0], wspace=0.0, hspace=0.0)
        main = fig.add_subplot(inner[1, 0])
        top = fig.add_subplot(inner[0, 0], sharex=main)
        right = fig.add_subplot(inner[1, 1], sharey=main)
        (levels, rng), render_info = _strain_panel(
            main, top, right, paths, monitored, colour, xlim, ylim, bins_a, bins_s, cloud,
            dens_lims, proj, grid=grid, frames=row_fr, field_lim=field_lim,
            field_kind=field_kind, tmpdir=tmpdir, iso_levels=iso_levels)
        stats[label] = {**stats[label], 'iso_levels': levels, 'iso_range': rng,
                        'render': render_info}

        # ONE legend for the pair, on the top row only: both rows use identical encodings and differ
        # only in hue, so repeating it underneath states nothing the reader has not already learnt.
        if r == 0:
            handles = [Line2D([], [], color="#6C757D", lw=2.6, label="monitored"),
                       Line2D([], [], color="#6C757D", lw=1.25, alpha=0.55, label="others")]
            if grid is not None:
                handles.append(Line2D([], [], color=ISO_COLOUR, lw=0.85, ls="--", alpha=0.6,
                                      label=ISO_LEGEND[iso]))
            row_leg = main.legend(handles=handles, fontsize=pt(7.0), loc="lower left", frameon=True,
                                  facecolor="#FFFFFF", edgecolor="none", framealpha=1.0)
            row_leg.set_zorder(30)

        # Row identity on the left edge: a step NUMERAL plus what changed, so the pair reads as one
        # problem solved twice rather than as two unrelated case studies.
        pos = sheet.get_position()
        sheet_pos.append(pos)
        step, what = ROW_STEP[label]
        ymid = 0.5 * (pos.y0 + pos.y1)
        fig.text(0.030, ymid, what, rotation=90, va="center", ha="center",
                 fontsize=pt(8.5), color=INK, linespacing=1.4)
        fig.text(0.030, pos.y1 - 0.012, step, rotation=0, va="top", ha="center",
                 fontsize=pt(10.5), color=colour, fontweight="bold")

        if r == 0:
            sheet.set_title("Load transmission", fontsize=pt(9.0), color=INK)
            # On the MARGINAL, not the main panel: the marginal sits at the top of the cell, so a
            # title on the panel underneath it would be drawn behind it.
            top.set_title("Ligament strain", fontsize=pt(9.0), color=INK, pad=6)
            # The transmission channels are design-independent, so this one is drawn once.
            # OUTSIDE the axes, on the figure. Sized for the page these five entries cover most of
            # a sheet panel, and a key drawn on top of the tiles it describes hides the reading.
            sheet_leg = fig.legend(handles=[
                Line2D([], [], color=MODE_COLOUR['tension'], lw=2.2, label="tension route"),
                Patch(facecolor=_zone_swatch((ref_field, opt_field)), edgecolor="#E2E4E8",
                      label="compression zone"),
                Line2D([], [], color=MODE_COLOUR['shear'], lw=1.8, label="shear plane"),
                Patch(facecolor=CLAMP_FILL, edgecolor="#9CA3AB", label="clamped tile"),
                Line2D([], [], color=INK, lw=0, marker='o', ms=6, markerfacecolor='none',
                       markeredgewidth=1.6, label="monitored ligament")],
                fontsize=pt(6.5), loc="lower left", bbox_to_anchor=(0.015, 0.005), ncol=2,
                frameon=False, handlelength=1.6, columnspacing=1.2, handletextpad=0.5)
            sheet_leg.set_zorder(30)

    # ⚠ A between-rows ARROW down the left margin was tried and removed: the rotated row labels
    # already fill that column for almost the full row height, so the arrow and its "retrain" tag
    # were squeezed into the hspace gap and rendered as a glitch.
    # ⚠ NO SUPTITLE. The pair reads as one problem solved twice from the row labels alone; a banner
    # over the top only repeats them, and the caption is where that sentence belongs.

    # ⚠ NO SHARED COLOUR BAR. Sized for the page it collided with the lower row's axis label and
    # the sheet key, and on the compressed ligament it spans 0..0.05 PEEQ -- a scale whose whole
    # range is "barely yielded", which is not worth the space it was taking. The renders still carry
    # one COMMON limit (`field_lim`), so the two rows remain directly comparable; the numbers are
    # printed on every run instead of drawn.
    # No bbox_inches: it crops to content, and the emitted page then stops being the authored
    # canvas that PAGE_SCALE is computed from.
    pdf, lo, hi = save_paper_figure(fig, out_png, FIG_W_IN * 72.0, fig.get_size_inches()[1] * 72.0,
                                    scale=PAGE_SCALE)
    print(f"  wrote {pdf} (type {lo:.1f}-{hi:.1f} pt printed)")
    plt.close(fig)
    tmp.cleanup()
    return out_png, monitored, stats


def _cfg(path):
    """Load a run config, or ``None``. Kept lazy: the figure works without one."""
    if not path:
        return None
    from nff.config.experiment import load_and_parse_config
    return load_and_parse_config(path)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref-paths", required=True, help="hinge-path dir of the reference design")
    p.add_argument("--ref-field", required=True, help="save_field STEM of the same design")
    p.add_argument("--opt-paths", required=True, help="hinge-path dir of the regularized design")
    p.add_argument("--opt-field", required=True, help="save_field STEM of the same design")
    p.add_argument("--hinge", type=int, default=None,
                   help="ligament index (default: the deepest-compression one in the reference)")
    p.add_argument("--resample", type=int, default=400)
    p.add_argument("--surrogate", default=DEFAULT_SURROGATE,
                   help="checkpoint for the iso-damage contours")
    p.add_argument("--no-iso", action="store_true", help="omit the contour backdrop entirely")
    p.add_argument("--iso", choices=("damage", "buckle", "none"), default="damage",
                   help="which landscape the trajectories are read against. 'damage' is the "
                        "SURROGATE's damage, extrapolating outside its training box. 'buckle' is "
                        "the CalculiX campaign's own MEASURED out-of-plane amplitude uz/w_lig, "
                        "interpolated only inside the hull of jobs actually solved.")
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help="campaign npz supplying the measured field for --iso buckle")
    p.add_argument("--frames", default=DEFAULT_FRAMES,
                   help="cached RVE hinge frames for the 3D inset at each trajectory tip, used for "
                        "BOTH rows. A pure-rotation RAY archive: the frame nearest in rotation is "
                        "drawn and its ligament width is the campaign's, NOT the monitored "
                        "ligament's -- so it cannot show a difference that lives in shear.")
    p.add_argument("--ref-frames", default=None,
                   help="per-row frames for the reference design, overriding --frames. Point this "
                        "at a REPLAY archive (deploy_hinge_frames.py --paths ... --hinge k) and the "
                        "render becomes that ligament's own route at its own w_lig and alpha.")
    p.add_argument("--opt-frames", default=None, help="likewise for the regularized design")
    p.add_argument("--no-render", action="store_true", help="omit the 3D hinge insets")
    p.add_argument("--select", choices=("damage", "compression"), default="damage",
                   help="how to pick the monitored ligament from the reference design")
    p.add_argument("--ref-config", default=None,
                   help="the reference run's config.yaml -> draws clamps and load arrows")
    p.add_argument("--opt-config", default=None, help="likewise for the regularized run")
    p.add_argument("--iso-common", action="store_true",
                   help="draw the SAME contours in both rows, at the reference design's ligament, "
                        "instead of each row's own. A common backdrop rather than each design's "
                        "own damage landscape -- read the caveat in _iso_damage before using it.")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ref_paths = load_paths(os.path.join(args.ref_paths, "hinge_paths"))
    opt_paths = load_paths(os.path.join(args.opt_paths, "hinge_paths"))
    iso = "none" if args.no_iso else args.iso
    # ⚠ The surrogate is loaded whenever it is needed for EITHER job, because it also SELECTS the
    # monitored ligament (`select='damage'`). Tying it to `iso` alone silently changed which
    # ligament the figure follows the moment the backdrop was switched to buckling.
    need_surrogate = iso == "damage" or args.select == "damage"
    surrogate = load_hinge_surrogate(args.surrogate) if need_surrogate else None
    dataset = load_buckle_dataset(args.dataset) if iso == "buckle" else None
    def _frames(path):
        return None if args.no_render or not path else dict(np.load(path))

    out, hinge, stats = build_figure(ref_paths, load_field(args.ref_field),
                                     opt_paths, load_field(args.opt_field), args.out,
                                     hinge=args.hinge, n_resample=args.resample,
                                     surrogate=surrogate, iso_common=args.iso_common,
                                     frames=_frames(args.frames),
                                     ref_frames=_frames(args.ref_frames),
                                     opt_frames=_frames(args.opt_frames),
                                     iso=iso, dataset=dataset, select=args.select,
                                     ref_config=_cfg(args.ref_config),
                                     opt_config=_cfg(args.opt_config))

    print(f"  monitored ligament {hinge} of {ref_paths.n_hinges}")
    for label in (REF_LABEL, OPT_LABEL):
        info = stats[label].get('render')
        if info:
            theta_k, w_lig, kind, replay = info
            # The provenance is the whole difference between a render that can carry the claim and
            # one that cannot, so it is printed on every run rather than left to the caller.
            print(f"  {label:38s} 3D render: {'REPLAY of this ligament' if replay else 'RAY (pure rotation, generic)'}"
                  f", frame at theta {theta_k:.1f} deg, w_lig {w_lig:.2f} mm"
                  f"{'' if replay else ' (NOT the learned width)'}, painted with {kind}")
    if iso != "none":
        unit = {"damage": "D/D_scale", "buckle": "uz/w_lig"}[iso]
        for label, paths in ((REF_LABEL, ref_paths), (OPT_LABEL, opt_paths)):
            lo, hi = stats[label]['iso_range']
            levels = stats[label]['iso_levels']
            print(f"  {label:38s} iso-{iso} slice at w_lig {paths.w_lig[hinge]:.2f} mm, "
                  f"alpha {np.degrees(paths.alpha[hinge]):.1f} deg, "
                  f"theta {np.degrees(traj_theta(paths, hinge)):.1f} deg   "
                  f"{unit} over the window {lo:.3f}..{hi:.3f}   levels drawn {levels}")
            if iso == "buckle" and not levels:
                print(f"  {'':38s} ⚠ no contours: too few campaign jobs near this "
                      f"(theta, w_lig) to interpolate")
    for label, paths in ((REF_LABEL, ref_paths), (OPT_LABEL, opt_paths)):
        e = np.asarray(paths.eta)[:, hinge, :]
        eta = np.asarray(paths.eta)
        print(f"  {label:38s} min eta_a {e[:, 0].min():+.4f} at load fraction "
              f"{paths.load_fraction[int(np.argmin(e[:, 0]))]:.2f}   endpoint {e[-1, 0]:+.4f}   "
              f"population min {eta[..., 0].min():+.4f}   routes "
              f"{stats[label]['tension_routes']}  shaded {stats[label]['shaded_tiles']}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
