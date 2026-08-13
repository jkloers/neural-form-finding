"""What the calibrated PET law predicts, and what survives condensation into the surrogate.

Three panels, none of them a calibration:

    (a) the law on a single ligament it was never fitted to,
    (b) the range of the law the oracle campaign that trains the surrogate exercises,
    (c) the error each stage of the chain contributes.

The law itself is an appendix object and is drawn by ``plot_material_law.py``.

The canvas is authored at the size the figure prints, 1.18\\textwidth = 437.3 bp, so an authored
point size is a printed point size. ``bbox_inches`` is therefore never passed to ``savefig``: it
crops to content and the LaTeX scale factor becomes unknown.

    JAX_PLATFORMS=cpu conda run -n kgnn_mac --no-capture-output \
        python nff/scripts/figures/plot_material_test.py
"""
import argparse
import csv
import json
import os
import pickle

import numpy as np

from nff.rve.materials.pet import PET_PLASTIC, PETIsotropic
from nff.utils.figstyle import (FAIL, FS_ANNOT, FS_BASE, FS_PANEL, FS_TICK, INK, MUTE, REAL,
                                SIM, apply_charter, figsize_in, panel_letter, paper_rc,
                                save_paper_figure, use_agg)
use_agg()
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch         # noqa: E402

WIDTH_BP, HEIGHT_BP = 1.18 * 370.6, 175.0
# The type ladder and the legibility floor come from nff/utils/figstyle.py, so a change to the
# policy reaches this plate instead of stopping at a private copy of the numbers.

PROCESSED = "data/experiments/processed"
# The measured single-ligament opening, and the UNTUNED prediction of it. Do not substitute
# pet_w18_open_calibrated.npz: that run carries the x0.757 heat-affected-zone knock-down, which
# reproduces the measurement to 0.2 per cent and was rejected on principle. The ligament experiment
# validates the material law; it does not calibrate it.
OPEN_MEASURED = PROCESSED + "/real_w18_open_trueaxis.csv"
OPEN_SIMULATED = PROCESSED + "/pet_w18_open_multipoint.npz"
DATASET = "data/fea/hinge_dataset_pet_merged"
SURROGATE = "data/surrogates/hinge_surrogate_pet_v3"

BOX = dict(boxstyle="round,pad=0.26", fc="white", ec="#D3D6DB", lw=0.7, alpha=0.93)

# Errors quoted on the arrows of the budget panel. Every one is established elsewhere in the
# manuscript; this panel collects them and does not re-derive them.
LIGAMENT_PCT = 11.0       # untuned single-ligament opening, 319 N against 287 N
# Worst paired-route gap measured in Sec. sec:path-dependence: one hinge driven to the same endpoint
# along its deployment route and along a straight ray, at the widest separation the trust region
# admits (0.118 ligament widths). Provenance data/surrogates/path_dependence_pet_v3.json, 20 pairs.
# ⚠ This REPLACES an earlier 43.3, which was a relRMSE on a second route family the manuscript no
# longer carries, and which had a median of 5.4 per cent. See [[feedback-relrmse-is-not-a-typical-error]].
OFF_ROUTE_PCT = 6.9


def _corner(ax, text, loc="upper left", fs=FS_ANNOT):
    x, y, va, ha = {"upper left": (0.035, 0.965, "top", "left"),
                    "upper right": (0.965, 0.965, "top", "right"),
                    "lower left": (0.035, 0.035, "bottom", "left"),
                    "lower right": (0.965, 0.035, "bottom", "right")}[loc]
    ax.text(x, y, text, transform=ax.transAxes, va=va, ha=ha, fontsize=fs, color=INK,
            bbox=BOX, zorder=6)


def load_opening():
    """-> ((a, F) measured, (a, F) simulated) for the w_lig = 18 mm ligament in opening."""
    with open(OPEN_MEASURED) as f:
        meas = np.array([[float(a), float(b)] for a, b in list(csv.reader(f))[1:]], float)
    with np.load(OPEN_SIMULATED) as npz:
        sim = np.column_stack([npz["a"], npz["F_a"]])
    return meas, sim


def load_law_reach():
    """-> (peeq_lig of the trained rows, n_rows, n_jobs). The trainer's own filters, in its order.

    ``load_dataset`` is used rather than a raw npz read because it back-fills ``stop_code`` from the
    sidecar manifest. Without it ``converged_mask`` returns all-True and the population comes out as
    11 919 instead of the 11 571 the checkpoint was fitted on.
    """
    from nff.scripts.train_hinge_surrogate import converged_mask, load_dataset
    from nff.utils.routes import route_family_mask
    with open(SURROGATE + ".pkl", "rb") as f:
        meta = pickle.load(f).get("meta", {}) or {}
    data, _, _ = load_dataset(DATASET, w_lig_max=meta.get("w_lig_max", 25.0))
    keep = converged_mask(data)
    if meta.get("route_family", "all") != "all":
        keep &= route_family_mask(data["u"][:, 0], data["u"][:, 1], data["u"][:, 2],
                                  data["g"][:, 0], data["job_id"], meta["route_family"])
    return data["peeq_lig"][keep], int(keep.sum()), len(np.unique(data["job_id"][keep]))


def panel_ligament(ax, meas, sim):
    """(a) The identified law on a single ligament it was never fitted to."""
    ax.plot(meas[:, 0], meas[:, 1], color=REAL, lw=1.2, label="measured")
    ax.plot(sim[:, 0], sim[:, 1], color=SIM, lw=1.2, label="predicted")
    pm, ps = float(meas[:, 1].max()), float(sim[:, 1].max())
    for x, y, c in ((meas[np.argmax(meas[:, 1]), 0], pm, REAL),
                    (sim[np.argmax(sim[:, 1]), 0], ps, SIM)):
        ax.plot([x], [y], "o", ms=3.4, color=c, zorder=5)
    ax.annotate("", xy=(0.30, ps), xytext=(0.30, pm),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.7, shrinkA=0, shrinkB=0))
    ax.text(0.42, ps + 0.05 * ps, f"$+{100 * (ps / pm - 1):.0f}$%", fontsize=FS_ANNOT,
            color=INK, va="center", ha="left")
    ax.set_xlim(0, 3.75)
    ax.set_ylim(0, 1.16 * max(pm, ps))
    ax.set_xlabel("axial opening $a$ [mm]")
    ax.set_ylabel("opening force $F_a$ [N]")
    ax.legend(loc="center right", fontsize=FS_ANNOT, handlelength=1.5, borderpad=0.32,
              labelspacing=0.28, bbox_to_anchor=(1.0, 0.55))
    _corner(ax, f"{ps:.0f} N predicted\n{pm:.0f} N measured\nno parameter adjusted",
            "lower right")
    return pm, ps


def panel_reach(ax, peeq, n_rows, n_jobs, eps_meas, eps_f):
    """(b) The range of the law the campaign that trains the surrogate exercises."""
    ax.axvspan(eps_meas, eps_f, color=MUTE, alpha=0.20, lw=0, zorder=0)
    ax.hist(peeq, bins=np.linspace(0.0, 1.9, 77), color=SIM, alpha=0.85, lw=0, zorder=2)
    ax.axvline(eps_meas, color=SIM, lw=0.9, ls="--", zorder=3)
    ax.axvline(eps_f, color=FAIL, lw=0.9, ls="--", zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(0.7, 1.5e5)
    ax.set_xlim(-0.03, 1.95)
    ax.set_xlabel(r"peak ligament strain $\varepsilon_p$")
    ax.set_ylabel("oracle samples")
    below = float((peeq <= eps_meas).mean())
    ax.text(1.487, 25, "inter-\npolated", ha="center", va="center", fontsize=FS_ANNOT, color=INK)
    ax.text(eps_f + 0.04, 30, r"$\varepsilon_f$", ha="left", va="center", rotation=90,
            fontsize=FS_ANNOT, color=FAIL)
    _corner(ax, f"{100 * below:.2f}% below $\\varepsilon_p={eps_meas}$\n"
                f"99th percentile {np.percentile(peeq, 99):.2f}\n"
                f"{n_rows} rows, {n_jobs} hinges", "upper right")
    return below, float(np.percentile(peeq, 99)), float(peeq.max())


def panel_budget(ax, metrics):
    """(c) The error each stage of the chain contributes."""
    ax.axis("off")
    stages = ["material law", "hinge response", r"$\mathcal{W}$ surrogate",
              "deployed sheet"]
    # A backslash-escaped per cent sign renders literally in matplotlib, so the sign is written
    # bare. The labels are terse because the panel is a third of the plate wide; the full wording
    # is in the body paragraph.
    arrows = [(f"material model  {LIGAMENT_PCT:.0f}%", SIM, False),
              (f"surrogate  {metrics[0]:.2f}%,\n{metrics[1]:.1f}%, {metrics[2]:.1f}%", SIM, False),
              (f"off route  {OFF_ROUTE_PCT:.1f}%", MUTE, True)]
    ys = np.linspace(0.93, 0.07, len(stages))
    for i, (label, y) in enumerate(zip(stages, ys)):
        faded = i == len(stages) - 1
        ax.add_patch(FancyBboxPatch((0.0, y - 0.055), 0.52, 0.11, transform=ax.transAxes,
                                    boxstyle="round,pad=0.006", lw=0.8,
                                    ec=MUTE if faded else SIM,
                                    fc="#F4F6F8" if faded else "white", zorder=2))
        ax.text(0.26, y, label, transform=ax.transAxes, ha="center", va="center",
                fontsize=FS_ANNOT, color=MUTE if faded else INK, zorder=3)
    for (label, colour, dashed), y0, y1 in zip(arrows, ys[:-1], ys[1:]):
        ax.add_patch(FancyArrowPatch((0.26, y0 - 0.058), (0.26, y1 + 0.058),
                                     transform=ax.transAxes, arrowstyle="-|>", lw=0.8,
                                     mutation_scale=6, color=colour, zorder=1,
                                     linestyle=(0, (2.5, 2)) if dashed else "solid"))
        ax.text(0.56, 0.5 * (y0 + y1), label, transform=ax.transAxes, ha="left", va="center",
                fontsize=FS_ANNOT, color=colour if dashed else INK, zorder=3)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="paper/Images/material_test.png")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    eps_f = PETIsotropic().eps_f
    eps_meas = max(e for _, e in PET_PLASTIC if e <= np.log(3.4))
    meas, sim = load_opening()
    peeq, n_rows, n_jobs = load_law_reach()
    with open(SURROGATE + ".json") as f:
        te = json.load(f)["test_metrics"]
    metrics = (100 * te["energy_rel"], 100 * te["force_rel"], 100 * te["damage_rel"])

    apply_charter(**paper_rc())
    fig, axes = plt.subplots(1, 3, figsize=figsize_in(WIDTH_BP, HEIGHT_BP),
                             gridspec_kw={"width_ratios": [1.0, 1.0, 1.32]})

    peak_meas, peak_sim = panel_ligament(axes[0], meas, sim)
    below, p99, pmax = panel_reach(axes[1], peeq, n_rows, n_jobs, eps_meas, eps_f)
    panel_budget(axes[2], metrics)

    # Letters BEFORE tight_layout: they sit above the axes, and tight_layout only reserves room for
    # artists that already exist when it runs. Added afterwards they are simply cropped off the top.
    for ax, letter in zip(axes, "abc"):
        panel_letter(ax, letter, x=-0.20, y=1.06)
    fig.tight_layout(w_pad=1.4)

    pdf, lo, hi = save_paper_figure(fig, args.out, WIDTH_BP, HEIGHT_BP, dpi=args.dpi)
    print(f"  wrote {pdf} ({WIDTH_BP:.1f} x {HEIGHT_BP:.1f} bp, type {lo:.1f}-{hi:.1f} pt)")
    plt.close(fig)

    prov = dict(peak_meas=peak_meas, peak_sim=peak_sim, below=below, p99=p99, pmax=pmax,
                n_rows=n_rows, n_jobs=n_jobs, eps_meas=eps_meas, eps_f=eps_f, metrics=metrics)
    _caption(os.path.splitext(args.out)[0] + ".txt", prov)
    print(f"  wrote {args.out}\n  wrote {pdf}")
    for k, v in prov.items():
        print(f"  {k}: {v}")


def _caption(path, p):
    with open(path, "w") as f:
        f.write(
            "What the calibrated PET law predicts, and what survives condensation.\n"
            "Generated by nff/scripts/figures/plot_material_test.py.\n\n"
            "(a) The law on a single 18 mm ligament in opening, a geometry the identification "
            f"never saw. The prediction is {p['peak_sim']:.0f} N against {p['peak_meas']:.0f} N "
            "measured, with no parameter adjusted.\n"
            "(b) The peak plastic strain reached in the ligament over the oracle campaign that "
            f"trains the surrogate, {p['n_rows']} samples on {p['n_jobs']} hinge geometries, "
            f"against the reach of the measured hardening table. {100 * p['below']:.2f} per cent "
            f"lie below its last coupon-derived point.\n"
            "(c) The error each stage of the chain contributes.\n\n"
            "Provenance, not part of the caption.\n"
            "Inputs: data/experiments/processed/real_w18_open_trueaxis.csv (measured opening), "
            "data/experiments/processed/pet_w18_open_multipoint.npz (predicted opening), "
            "data/fea/hinge_dataset_pet_merged (campaign), "
            "data/surrogates/hinge_surrogate_pet_v3 (checkpoint).\n"
            "TRAP: pet_w18_open_calibrated.npz peaks at 287.4 N and must NOT be used in panel (a). "
            "It carries the x0.757 heat-affected-zone knock-down, which reproduces the measurement "
            "almost exactly and was rejected on principle. The ligament experiment validates the "
            "material law; it does not calibrate it.\n"
            f"Law reach: 99th percentile {p['p99']:.3f}, maximum {p['pmax']:.3f}. No sample in the "
            f"campaign reaches the fracture strain {p['eps_f']}, consistent with the campaign "
            "recording no torn job. The population is the trainer's own: route family origin_ray, "
            "unconverged jobs quarantined, w_lig at most 25 mm.\n"
            f"Surrogate held-out error, read from the checkpoint sidecar: "
            f"{p['metrics'][0]:.2f} per cent on the energy, {p['metrics'][1]:.1f} on the "
            f"generalised forces, {p['metrics'][2]:.2f} on the damage.\n"
            "Regenerate: JAX_PLATFORMS=cpu conda run -n kgnn_mac --no-capture-output "
            "python nff/scripts/figures/plot_material_test.py\n")


if __name__ == "__main__":
    main()
