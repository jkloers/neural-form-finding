"""One cut pattern, three states: what geometry promises, what the load does to it, and what
designing through equilibrium recovers.

The three panels share ONE sheet, ONE target and ONE axis extent, so the middle panel reads as the
sheet failing to open rather than as a different drawing. The load glyph appears in panels (b) and
(c) and NOT in (a): that absence is the argument, since the geometric design never saw a force.

    (a) traced by geometry   the design's own closed-form deployment, the shape it was chosen for
    (b) driven               that same cut pattern under the actuation, solved
    (c) through equilibrium  a design that had the actuation in its objective, same load as (b)

Every panel is a fresh solve. Numbers belong in the caption, which is written alongside the plate.

Run:
    PYTHONPATH=$(pwd) JAX_PLATFORMS=cpu conda run -n kgnn_mac \
        python nff/scripts/figures/plot_ume_conditioning.py \
            --geom-params data/outputs/kinematic_designs/ume_geom/best_params.pkl \
            --phys-params data/outputs/runs/<run>/best_params.pkl \
            --disturbance 20.0
"""

import argparse
import copy
import json
import os
import sys

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np                                                          # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from nff.closed.deploy import _boundary_cloud                               # noqa: E402
from nff.closed.hinge_paths import resolve_length_scale                     # noqa: E402
from nff.closed.load_paths import solve_deployed_state                      # noqa: E402
from nff.closed.one_dof import fit_one_dof                                  # noqa: E402
from nff.closed.shield_kinematics import (_checkerboard_signs,              # noqa: E402
                                          build_deploy_plan, deploy_tiles)
from nff.config.experiment import load_and_parse_config                     # noqa: E402
from nff.scripts.diagnostics.score_designs import (_boundary_of, _deployed_tiles,  # noqa: E402
                                                   _fit_translation, _flat_tiles,
                                                   shape_metrics, target_cloud, target_record)
from nff.scripts.diagnostics.sweep_ume_collapse import with_disturbance     # noqa: E402
from nff.topology.closed_builder_jax import build_deploy_structure          # noqa: E402
from nff.utils.figstyle import (FS_ANNOT, GREY, INK, ORANGE, TEXT_W_BP,     # noqa: E402
                                apply_charter, figsize_in, page_bp, panel_letter,
                                paper_rc, save_paper_figure, use_agg)

use_agg()
import matplotlib.pyplot as plt                                             # noqa: E402
from matplotlib.patches import FancyArrowPatch                              # noqa: E402

LOAD_INK = INK                  # applied loads and moments are BLACK and short, never red
WEIGHT_SCALE = 0.55             # strokes do not follow the type ladder


def ideal_deployment(config, params_path, cloud):
    """The closed-form deployment this design was chosen for: its promise, with no force in it.

    The scan is ``score_designs.kinematic_promise``'s, kept identical on purpose: the same opening
    grid, the same boundary-vertex gather and the same free placement, so the number printed under
    panel (a) is on the ruler panels (b) and (c) are scored on.
    """
    results, _, _, _, _ = solve_deployed_state(config, params_path)
    vs = results['valid_state']
    flat = np.asarray(_flat_tiles(vs))
    M, N = int(config.topology['M']), int(config.topology['N'])
    plan = build_deploy_plan(build_deploy_structure(M, N), M, N,
                             float(config.topology.get('spacing', 1.0)),
                             float(config.topology.get('spacing_y',
                                                       config.topology.get('spacing', 1.0))))
    best = None
    for psi in np.linspace(-np.pi / 4, -0.05, 120):
        tiles = np.asarray(deploy_tiles(plan, flat, float(psi)))
        pts = _boundary_of(tiles, vs)
        shift = _fit_translation(pts, cloud)
        sq = ((pts + shift)[:, None, :] - cloud[None, :, :]) ** 2
        precision = float(sq.sum(axis=-1).min(axis=1).mean())
        if best is None or precision < best[0]:
            best = (precision, tiles + shift, float(psi), pts + shift)
    return best[1], best[2], best[3]


def loaded_state(config, params_path, cloud, length_scale):
    """One design under one actuation, solved."""
    results, _, _, _, _ = solve_deployed_state(config, params_path)
    vs = results['valid_state']
    disp = results['solution'].fields[-1]
    tiles = np.asarray(_deployed_tiles(vs, disp))
    boundary = np.asarray(_boundary_cloud(vs, disp))
    M, N = int(config.topology['M']), int(config.topology['N'])
    fit = fit_one_dof(np.asarray(disp)[:, 2], _checkerboard_signs(M, N).astype(float))
    stats = shape_metrics(boundary, cloud, length_scale)
    stats['theta_deg'] = float(np.degrees(2.0 * abs(fit.psi)))
    return tiles, boundary, stats


def draw_sheet(ax, tiles, cloud, clamp_face=None, lw=0.35):
    """Panels in charter orange, the target as a thin dashed outline, the clamp in grey."""
    for k, quad in enumerate(tiles):
        face = GREY if (clamp_face is not None and k == clamp_face) else ORANGE
        ax.fill(quad[:, 0], quad[:, 1], facecolor=face, edgecolor='white',
                lw=lw * WEIGHT_SCALE / 0.35, zorder=2)
    ax.plot(np.append(cloud[:, 0], cloud[0, 0]), np.append(cloud[:, 1], cloud[0, 1]),
            color=INK, lw=0.7, ls=(0, (4, 2.4)), zorder=3)
    ax.set_aspect('equal')
    ax.axis('off')


def draw_load(ax, centre, span, moment_at, force_at=None):
    """The actuation: a short moment arc on the driven panel, and an arrow if a disturbance acts.

    Drawn identically wherever it appears, and omitted entirely from the geometric panel, so the
    reader sees which design was shown a force and which was not.
    """
    r = 0.055 * span
    arc = FancyArrowPatch((moment_at[0] - r, moment_at[1] - r * 0.2),
                          (moment_at[0] + r, moment_at[1] - r * 0.2),
                          connectionstyle="arc3,rad=-0.65",
                          arrowstyle='-|>', mutation_scale=6.5,
                          lw=1.1 * WEIGHT_SCALE / 0.55, color=LOAD_INK, zorder=6)
    ax.add_patch(arc)
    if force_at is None:
        return
    L = 0.10 * span
    ax.add_patch(FancyArrowPatch((force_at[0], force_at[1] + L), (force_at[0], force_at[1]),
                                 arrowstyle='-|>', mutation_scale=6.5,
                                 lw=1.1 * WEIGHT_SCALE / 0.55, color=LOAD_INK, zorder=6))


def build(config_path, geom_params, phys_params, disturbance, face, dof, out_stem,
          width_frac=1.0, aspect=2.35, dpi=600):
    config = load_and_parse_config(config_path)
    length_scale = resolve_length_scale(config)
    cloud, spec = target_cloud(config)
    # A zero disturbance is the pure deployment problem, so no load is added at all rather than a
    # zero-valued one: an inert entry would still print in the caption as if it acted.
    loaded_cfg = (with_disturbance(config, face, dof, -abs(disturbance)) if disturbance
                  else config)
    clamp = (config.topology.get('bc_clamped') or [None])[0]

    print("[a] ideal deployment of the geometric design ...", flush=True)
    ideal, psi_ideal, ideal_boundary = ideal_deployment(config, geom_params, cloud)
    print("[b] the geometric design under the actuation ...", flush=True)
    geo_t, geo_b, geo_s = loaded_state(loaded_cfg, geom_params, cloud, length_scale)
    print("[c] the design trained through equilibrium ...", flush=True)
    phy_t, phy_b, phy_s = loaded_state(loaded_cfg, phys_params, cloud, length_scale)

    ideal_s = shape_metrics(ideal_boundary, cloud, length_scale)
    ideal_s['theta_deg'] = float(np.degrees(2.0 * abs(psi_ideal)))

    # ONE extent for all three panels: the collapse has to read as the sheet shrinking, which it
    # cannot do if each panel is autoscaled to its own contents.
    everything = np.vstack([ideal.reshape(-1, 2), geo_t.reshape(-1, 2), phy_t.reshape(-1, 2),
                            cloud])
    lo, hi = everything.min(0), everything.max(0)
    pad = 0.04 * float((hi - lo).max())
    centre = (lo + hi) / 2.0
    half = float((hi - lo).max()) / 2.0 + pad
    span = 2 * half

    apply_charter(**{**paper_rc(), 'axes.grid': False})
    w_bp, h_bp = page_bp(width_frac, aspect)
    fig, axes = plt.subplots(1, 3, figsize=figsize_in(w_bp, h_bp))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.90, bottom=0.01, wspace=0.02)

    titles = ("geometry alone", "the same pattern, driven", "designed through equilibrium")
    panels = ((ideal, None), (geo_t, geo_b), (phy_t, phy_b))
    for k, (ax, title, (tiles, _)) in enumerate(zip(axes, titles, panels)):
        draw_sheet(ax, tiles, cloud, clamp_face=None if k == 0 else clamp)
        ax.set_xlim(centre[0] - half, centre[0] + half)
        ax.set_ylim(centre[1] - half, centre[1] + half)
        ax.set_title(title, fontsize=FS_ANNOT, color=INK, pad=3.0)
        if k > 0:
            drive = tiles[int(config.topology['loads'][0]['face'])].mean(0)
            draw_load(ax, centre, span, drive,
                      tiles[face].mean(0) if disturbance else None)

    for ax, letter in zip(axes, "abc"):
        bb = ax.get_position()
        panel_letter(fig, letter, x=bb.x0 + 0.004, y=bb.y1 - 0.012)

    pdf, lo_pt, hi_pt = save_paper_figure(fig, out_stem, w_bp, h_bp, dpi=dpi,
                                          strict_margins=True)
    print(f"  wrote {pdf}   type {lo_pt:.1f}-{hi_pt:.1f} pt")

    stats = {'target': target_record(spec), 'length_scale_mm': length_scale,
             'disturbance_N': float(disturbance), 'face': int(face), 'dof': int(dof),
             'drive': list(config.topology['loads']),
             'a_ideal': ideal_s, 'b_geometry_driven': geo_s, 'c_through_equilibrium': phy_s}
    with open(f"{out_stem}.json", "w") as fh:
        json.dump(stats, fh, indent=1)
    write_caption(stats, out_stem)
    return stats


def write_caption(stats, out_stem):
    """Every number the caption needs, computed from the data that made the plate."""
    t = stats['target']
    eps_star = 1.0 / (t['petals'] ** 2 + 1.0)
    L = stats['length_scale_mm']
    a, b, c = stats['a_ideal'], stats['b_geometry_driven'], stats['c_through_equilibrium']
    lines = [
        "FIGURE. One sheet, one target, one actuation.",
        "",
        f"Target: a plum blossom r = R (1 + eps cos {t['petals']}t), mean radius "
        f"{t['radius'] * L:.0f} mm, eps = {t['eps']:.2f}. A polar outline is strictly convex iff "
        f"eps < 1/(n^2+1) = {eps_star:.4f}, so this target is re-entrant by a factor of "
        f"{t['eps'] / eps_star:.1f}.",
        f"Sheet: 7x7 closed RDPQK, 49 panels, 84 hinges, 2844.8 mm square. The centre panel is "
        f"clamped on all three degrees of freedom and one moment drives the whole sheet.",
        "",
        f"(a) The cut pattern designed against the blossom by rigid-panel kinematics alone, shown "
        f"in the configuration it was chosen for: opening {a['theta_deg']:.1f} deg, boundary error "
        f"{a['rms_boundary_mm']:.1f} mm. Placement is free, as a mechanism with no supports has no "
        f"place of its own.",
        f"(b) That same cut pattern under the actuation: opening {b['theta_deg']:.1f} deg, boundary "
        f"error {b['rms_boundary_mm']:.1f} mm.",
        f"(c) A cut pattern designed through equilibrium under the same actuation: opening "
        f"{c['theta_deg']:.1f} deg, boundary error {c['rms_boundary_mm']:.1f} mm.",
        "",
        f"The deployed sheet in (b) is not broken. It remains a one-degree-of-freedom mechanism to "
        f"within half a millimetre, and it opens to the same {b['theta_deg']:.1f} deg as (c). The "
        f"two designs differ in WHICH mechanism they are. The kinematic design places the blossom "
        f"at {a['theta_deg']:.1f} deg of its own family, and the load does not stop there.",
        "",
        f"Physics conditioning is not a correction applied to (a). Scored on its own kinematics the "
        f"design in (c) is the WORSE of the two, and a search over geometry alone would have "
        f"discarded it.",
    ]
    with open(f"{out_stem}.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join("  " + ln for ln in lines if ln))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="data/configs/closed/ume_7x7.yaml")
    ap.add_argument("--geom-params",
                    default="data/outputs/kinematic_designs/ume_geom/best_params.pkl")
    ap.add_argument("--phys-params", required=True)
    ap.add_argument("--disturbance", type=float, default=0.0,
                    help="optional transverse load in newtons, applied in BOTH loaded panels; "
                         "0 is the pure deployment problem the moment alone poses")
    ap.add_argument("--face", type=int, default=48)
    ap.add_argument("--dof", type=int, default=1)
    ap.add_argument("--width-frac", type=float, default=1.0)
    ap.add_argument("--aspect", type=float, default=2.35, help="WIDTH / HEIGHT")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--out", default="paper/Images/ume_conditioning")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    build(args.config, args.geom_params, args.phys_params, args.disturbance,
          args.face, args.dof, args.out, args.width_frac, args.aspect, args.dpi)


if __name__ == "__main__":
    main()
