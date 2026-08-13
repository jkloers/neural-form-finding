"""What equilibrium does to the solution set: two families become one, and stay wide.

Deliberately shows nothing that Figure ``fig:solution-set`` already shows. That figure owns the
kinematic cloud, its starting patterns, its optimisation paths and its two cut patterns. This one
shows only where the three sets sit, how big they are, and the one deployment quantity that
explains the merge.

Sizes are the honest part of panel a and the scatter is not. Any plane retains most of the
kinematic spread, which lies along a single direction, and about a fifth of the equilibrium
spread, which is shared over roughly twelve. Each set therefore carries a ring whose radius is
its root-mean-square spread measured in the FULL design space, in the same millimetres as the
axes.

    JAX_PLATFORMS=cpu conda run -n kgnn_mac python \
        nff/scripts/figures/plot_physics_comparison.py --paper-dir paper
"""

import argparse
import glob
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np                                                   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from nff.utils.figstyle import (FS_ANNOT, FS_BASE, INK, MUTE, ORANGE, TEAL, apply_charter,
                                figsize_in, page_bp, panel_letter, paper_rc,
                                save_paper_figure, use_agg)

use_agg()
import matplotlib.pyplot as plt                                       # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# The shared plane
# ══════════════════════════════════════════════════════════════════════════════

def flatten_mm(coords_list):
    """Design vertices as vectors whose Euclidean distance IS the RMS displacement in mm.

    Dividing by the square root of the vertex count turns the plain norm on the flattened
    array into the same millimetre quantity ``canonical_distance`` reports, so a length read
    off the plane is directly comparable to the spreads quoted in the text.
    """
    X = np.stack([np.asarray(c, dtype=float).reshape(-1) for c in coords_list])
    return X / np.sqrt(len(np.asarray(coords_list[0])))


def shared_plane(groups):
    """Two leading principal directions over the converged designs of both arms."""
    X = np.vstack([flatten_mm(g) for g in groups if len(g)])
    mean = X.mean(axis=0)
    _, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    var = (S ** 2) / max(len(X) - 1, 1)
    return {"mean": mean, "basis": Vt[:2], "explained": float(var[:2].sum() / var.sum())}


def project(coords_list, plane):
    return (flatten_mm(coords_list) - plane["mean"]) @ plane["basis"].T


def rms_spread(coords_list):
    """Root-mean-square distance from the set's own centre, in mm, in the full space."""
    V = flatten_mm(coords_list)
    return float(np.sqrt(np.mean(np.sum((V - V.mean(0)) ** 2, axis=1))))


def load_designs(run):
    """Converged and starting cut vertices. Trajectories are not needed and not read."""
    ends, starts = [], []
    for path in sorted(glob.glob(os.path.join(run, "designs", "seed*.npz"))):
        with np.load(path) as z:
            ends.append(np.asarray(z["coords"]))
            starts.append(np.asarray(z["coords_start"]))
    return ends, starts


# ══════════════════════════════════════════════════════════════════════════════
# Panels
# ══════════════════════════════════════════════════════════════════════════════

def panel_sets(ax, sets, frame):
    """Every solution set at once: its designs, and a ring for its measured extent."""
    for s in sets:
        if s["dots"] is not None:
            ax.scatter(s["dots"][:, 0], s["dots"][:, 1], s=7, c=s["colour"],
                       edgecolors=INK, linewidths=0.22, zorder=s["z"], label=None)
    for s in sets:
        ax.add_patch(plt.Circle(s["centre"], s["radius"], fill=False, edgecolor=s["colour"],
                                lw=0.8, ls=(0, (3, 2)), zorder=s["z"] + 1))
        ax.plot([], [], color=s["colour"], lw=0.8, ls=(0, (3, 2)), label=s["label"])

    ax.set_xlim(frame[0], frame[1])
    ax.set_ylim(frame[2], frame[3])
    # Millimetres must be the same length on both axes, or a ring draws as an ellipse and
    # reads as an anisotropy the data does not have.
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("PC1 (mm)")
    ax.set_ylabel("PC2 (mm)")
    ax.grid(False)
    ax.tick_params(length=2.4, pad=1.5)


def panel_angle(ax, kin_deg, phy_deg, scale=1.0):
    """The deployment quantity the two families differed in, measured the same way for both.

    Relative rotation across a hinge: the difference between the deployed rotations of the two
    tiles it joins, averaged over the sheet. It must be RELATIVE. The two arms sit in different
    frames, because the clamp pins one sublattice under equilibrium while the closed-form map
    is symmetric about zero, so no absolute tile rotation compares across them.
    """
    bins = np.arange(30.0, 96.0, 1.0)
    ax.hist(kin_deg, bins=bins, color=TEAL, histtype="stepfilled", alpha=0.55,
            edgecolor=TEAL, lw=0.8, label="kinematic")
    ax.hist(phy_deg, bins=bins, color=ORANGE, histtype="stepfilled", alpha=0.55,
            edgecolor=ORANGE, lw=0.8, label="equilibrium")
    ax.set_xlabel("opening angle (deg)")
    ax.set_ylabel("designs")
    ax.legend(loc="upper left", fontsize=FS_ANNOT / scale, handletextpad=0.5,
              borderpad=0.3, labelspacing=0.25, framealpha=0.0)
    ax.tick_params(length=2.4, pad=1.5)


def caption_text(plane, radii):
    """The figure's caption, written beside it so the manuscript never restates it by hand."""
    return (
        "Effect of equilibrium on the solution set. Two hundred random starting patterns were "
        "solved onto one target, first by the closed-form kinematic map and then by Stage-2 "
        "equilibrium under a moment calibrated to open the sheet fully. \\textbf{(a)}~The "
        "converged designs of both arms in one plane, spanned by their two leading principal "
        "components. Each dashed ring is the root-mean-square spread of a set measured in the "
        "full design space, in the millimetres of the axes. The rings may be compared. The "
        "scatter may not, because a plane retains most of the kinematic spread and about a "
        "fifth of the equilibrium spread. "
        "\\textbf{(b)}~Distribution of pairwise design distance within each ensemble, "
        "quotiented by the symmetry of the cut family. \\textbf{(c)}~Opening angle of the "
        "deployed sheet, the mean rotation across a hinge between the two tiles it joins, "
        "measured identically for both arms.")


def main():
    from nff.closed.design_ensemble import (build_symmetry, circle_problem, distance_matrix,
                                            problem_statics, split_families)

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kinematic", default="data/outputs/design_cloud/circle_7x7_n200_s1")
    ap.add_argument("--physics", default="data/outputs/design_cloud/physics_n200_e150_s1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--paper-dir", default=None)
    args = ap.parse_args()

    cfg = circle_problem()
    group = build_symmetry(cfg, problem_statics(cfg))
    print(f"  symmetry group: {sorted(group)}")

    kin_end, kin_start = load_designs(args.kinematic)
    phy_end, phy_start = load_designs(args.physics)
    print(f"  kinematic {len(kin_end)} designs, physics {len(phy_end)} designs")

    dist = {}
    for name, ends in (("kinematic", kin_end), ("physics", phy_end)):
        D = distance_matrix(ends, group)
        labels, fam = split_families(D)
        off = ~np.eye(len(ends), dtype=bool)
        dist[name] = {"D": D, "labels": labels, "fam": fam, "pairs": D[off]}
        print(f"  {name:9s} families {fam['sizes']}  separation {fam['separation_ratio']:.2f}")

    angles = {}
    for name, run in (("kinematic", args.kinematic), ("physics", args.physics)):
        path = os.path.join(run, "opening_angle.npz")
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}; run the opening-angle measurement first")
        with np.load(path) as z:
            angles[name] = np.asarray(z["theta_deg"])
        a = angles[name]
        print(f"  {name:9s} opening angle {a.mean():.2f} +- {a.std():.2f} deg  "
              f"[{a.min():.2f}, {a.max():.2f}]")

    plane = shared_plane([kin_end, phy_end])
    print(f"  plane holds {100 * plane['explained']:.0f}% of the converged-design variance")

    kin_lab = dist["kinematic"]["labels"]
    groups = []
    for c in np.unique(kin_lab):
        members = np.flatnonzero(kin_lab == c)
        if len(members) < 5:
            continue
        sub = [kin_end[i] for i in members]
        groups.append({"tag": f"kin{c}", "coords": sub, "colour": TEAL, "z": 6,
                       "label": "kinematic" if len(groups) == 0 else None})
    groups.append({"tag": "phy", "coords": phy_end, "colour": ORANGE, "z": 4,
                   "label": "equilibrium"})

    sets = []
    for g in groups:
        xy = project(g["coords"], plane)
        sets.append({"dots": xy, "centre": xy.mean(0), "radius": rms_spread(g["coords"]),
                     "colour": g["colour"], "z": g["z"], "label": g["label"], "tag": g["tag"]})

    # The random patterns are drawn as an extent only. Four hundred start dots would be a
    # second copy of Figure fig:solution-set panel a, which is the thing this plate avoids.
    starts = kin_start + phy_start
    sets.insert(0, {"dots": None, "centre": project(starts, plane).mean(0),
                    "radius": rms_spread(starts), "colour": MUTE, "z": 2,
                    "label": "random patterns", "tag": "start"})
    radii = {"kin_a": sets[1]["radius"], "kin_b": sets[2]["radius"],
             "phy": sets[3]["radius"], "start": sets[0]["radius"]}
    for s in sets:
        print(f"  ring {s['tag']:5s} radius {s['radius']:7.1f} mm")

    scale = 1.0
    w_bp, h_bp = page_bp(1.0, aspect=3.05)
    apply_charter(**paper_rc(scale))
    fig = plt.figure(figsize=figsize_in(w_bp, h_bp))
    ax_a = fig.add_axes([0.108, 0.235, 0.280, 0.640])
    ax_b = fig.add_axes([0.455, 0.235, 0.205, 0.640])
    ax_c = fig.add_axes([0.775, 0.235, 0.205, 0.640])

    # The frame is padded to the AXES box aspect, so ``set_aspect("equal")`` costs nothing.
    # Letting matplotlib enforce equal scaling instead would shrink panel a inside its box
    # and leave it shorter than its neighbours.
    box = ax_a.get_position()
    box_aspect = (box.width * w_bp) / (box.height * h_bp)
    ext = np.vstack([np.array([[s["centre"][0] - s["radius"], s["centre"][1] - s["radius"]],
                               [s["centre"][0] + s["radius"], s["centre"][1] + s["radius"]]])
                     for s in sets])
    cx, cy = 0.5 * (ext[:, 0].min() + ext[:, 0].max()), 0.5 * (ext[:, 1].min() + ext[:, 1].max())
    hx, hy = 0.58 * np.ptp(ext[:, 0]), 0.60 * np.ptp(ext[:, 1])
    hx, hy = max(hx, hy * box_aspect), max(hy, hx / box_aspect)
    panel_sets(ax_a, sets, (cx - hx, cx + hx, cy - hy, cy + hy))
    ax_a.legend(loc="lower left", fontsize=FS_ANNOT / scale, handletextpad=0.5,
                borderpad=0.3, labelspacing=0.25, framealpha=0.0)

    # The claim that depends on no projection at all: a bimodal distance distribution against
    # a unimodal one.
    for name, label, colour in (("kinematic", "kinematic", TEAL),
                                ("physics", "equilibrium", ORANGE)):
        ax_b.hist(dist[name]["pairs"], bins=48, density=True, histtype="step", color=colour,
                  lw=1.1, label=label)
    ax_b.set_xlabel("design distance (mm)")
    ax_b.set_ylabel("density")
    ax_b.legend(loc="upper right", fontsize=FS_ANNOT / scale, handletextpad=0.5,
                borderpad=0.3, labelspacing=0.25, framealpha=0.0)
    ax_b.tick_params(length=2.4, pad=1.5)

    panel_angle(ax_c, angles["kinematic"], angles["physics"], scale)

    for ax, letter in ((ax_a, "a"), (ax_b, "b"), (ax_c, "c")):
        panel_letter(ax, letter, x=-0.215, y=1.07, scale=scale)

    caption = caption_text(plane, radii)
    out_path = args.out or os.path.join(args.physics, "physics_comparison.pdf")
    pdf, lo, hi = save_paper_figure(fig, out_path, w_bp, h_bp, dpi=600, scale=scale)
    plt.close(fig)
    print(f"  wrote {pdf}   type [{lo}, {hi}] pt")

    with open(os.path.splitext(pdf)[0] + ".txt", "w") as fh:
        fh.write(caption + "\n")

    if args.paper_dir:
        import shutil
        images = os.path.join(args.paper_dir, "Images")
        os.makedirs(images, exist_ok=True)
        for ext_ in (".pdf", ".png", ".txt"):
            shutil.copyfile(os.path.splitext(pdf)[0] + ext_,
                            os.path.join(images, "physics_comparison" + ext_))
        print(f"  wrote {images}/physics_comparison.pdf, .png and .txt")


if __name__ == "__main__":
    main()
