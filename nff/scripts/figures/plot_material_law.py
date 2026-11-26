"""The PET constitutive law identified from tensile coupons, as handed to the finite-element solver.

An appendix figure: it documents the calibration, not a result. The hardening table is read from
``nff.rve.materials.pet`` rather than retyped, so the picture cannot drift from the card the solver
receives.

The canvas is authored at the size the figure prints, ``\\textwidth`` = 370.6 bp, so an authored
point size is a printed point size. ``bbox_inches`` is therefore never passed to ``savefig``: it
crops to content and the LaTeX scale factor becomes unknown.

    JAX_PLATFORMS=cpu conda run -n kgnn_mac --no-capture-output \
        python nff/scripts/figures/plot_material_law.py
"""
import argparse
import csv
import os

import numpy as np

from nff.calibration.summary_io import load_summary
from nff.rve.materials.pet import PET, PET_PLASTIC, PETIsotropic
from nff.utils.figstyle import (FAIL, FS_ANNOT, FS_BASE, FS_TICK, INK, MUTE, REAL, SIM,
                                apply_charter, figsize_in, paper_rc, save_paper_figure, use_agg)
use_agg()
import matplotlib.pyplot as plt                                        # noqa: E402

WIDTH_BP, HEIGHT_BP = 370.6, 223.0
# The type ladder and the legibility floor come from nff/utils/figstyle.py, so a change to the
# policy reaches this plate instead of stopping at a private copy of the numbers.

COUPON_DIR = "data/experiments/raw/kirigami_20260723"
SUMMARY = COUPON_DIR + "/Kirigami experiments - Tensile test (safe).csv"
PROCESSED = "data/experiments/processed"


def load_coupons(decimate=70):
    """-> (engineering curves, mean draw ratio, median plateau, median upper-yield peak).

    The curves are returned in ENGINEERING stress against crosshead strain, which is what the frame
    measures. The strain carries the machine and grip compliance and is therefore not the material's
    strain. These curves are used for the two stress levels that anchor the law, which need no
    strain measurement at all; the modulus is nominal and is not read from a coupon.
    """
    rows = [r for r in load_summary(SUMMARY) if r.included and r.lam is not None]
    curves, plateaus, peaks = [], [], []
    for r in rows:
        with open(f"{PROCESSED}/{r.name}_stress_strain.csv") as f:
            raw = np.array([[float(a), float(b)] for a, b in list(csv.reader(f))[1:]], float)
        eps_eng, sig_eng = raw[:, 0], raw[:, 1]
        peak = int(np.argmax(sig_eng))
        peaks.append(float(sig_eng[peak]))
        # Draw plateau: the flat neck-propagation stress. Skip the yield drop that follows the peak,
        # then take the median of what remains.
        start = peak + int(0.3 * (len(sig_eng) - peak))
        plateaus.append(float(np.median(sig_eng[start:])))
        curves.append((r.name, eps_eng[::decimate], sig_eng[::decimate]))
    return (curves, float(np.mean([r.lam for r in rows])),
            float(np.median(plateaus)), float(np.median(peaks)))


def draw(ax, curves, lam, plateau_meas, eps_meas, eps_f):
    """The hardening table handed to the solver, and the coupons it is built from."""
    E = PET["E"]
    measured = [p for p in PET_PLASTIC if p[1] <= eps_meas]
    interpolated = [p for p in PET_PLASTIC if p[1] > eps_meas]

    ax.axvspan(eps_meas, eps_f, color=MUTE, alpha=0.20, lw=0, zorder=0)

    # Neck-propagation locus: at a constant plateau force each shoulder point carries
    # sigma_true = sigma_plateau * exp(eps_true), by plastic incompressibility, up to eps = ln(lam).
    anchor_sig, anchor_eps = measured[-1]
    sig_plateau = anchor_sig / lam
    eps_t = np.linspace(0.02, np.log(lam), 200)
    sig_locus = sig_plateau * np.exp(eps_t)
    ax.plot(eps_t - sig_locus / E, sig_locus, color=REAL, lw=1.0, ls=(0, (3, 2)), zorder=3,
            label="drawing locus")

    tab = np.array(PET_PLASTIC, float)
    ax.plot(tab[:, 1], tab[:, 0], color=SIM, lw=1.3, zorder=4, label="hardening table")
    ax.plot([p[1] for p in measured], [p[0] for p in measured], "o", ms=2.8, color=SIM, zorder=5)
    ax.plot([p[1] for p in interpolated], [p[0] for p in interpolated], "o", ms=2.8, mfc="white",
            mec=SIM, mew=0.9, zorder=5)
    ax.plot([anchor_eps], [anchor_sig], "D", ms=4.4, mfc="white", mec=REAL, mew=1.2, zorder=6)
    ax.annotate(f"draw anchor\n$\\Lambda={lam:.2f}$", (anchor_eps, anchor_sig),
                textcoords="offset points", xytext=(13, -6), ha="left", va="top",
                fontsize=FS_ANNOT, color=INK, zorder=6,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.6, shrinkA=0, shrinkB=4))

    ax.axvline(eps_f, color=FAIL, lw=0.9, ls="--", zorder=3)
    ax.text(eps_f + 0.035, 110, r"$\varepsilon_f$", ha="left", va="center", rotation=90,
            fontsize=FS_ANNOT, color=FAIL)
    ax.text(1.487, 196, "interpolated", ha="center", va="top", fontsize=FS_ANNOT, color=INK)

    ax.set_xlim(-0.03, 1.9)
    ax.set_ylim(0, 218)
    ax.set_xlabel(r"true plastic strain $\varepsilon_p$")
    ax.set_ylabel("true stress [MPa]")
    ax.legend(loc="lower right", fontsize=FS_ANNOT, handlelength=1.7, borderpad=0.35,
              labelspacing=0.3, borderaxespad=0.4)

    # The coupons on the axis the frame actually measures. Their strain carries the machine
    # compliance; the two stress levels they fix do not. Placed above the hardening curve, which
    # stays below 100 MPa over the left half of the panel.
    ins = ax.inset_axes([0.135, 0.545, 0.43, 0.375])
    for _, e, s in curves:
        ins.plot(e, s, color=REAL, lw=0.5, alpha=0.75)
    ins.axhline(plateau_meas, color=INK, lw=0.6, ls=":")
    ins.set_xlim(0, 0.21)
    ins.set_ylim(0, 62)
    ins.set_xticks([0, 0.1, 0.2])
    ins.set_yticks([0, 25, 50])
    ins.tick_params(labelsize=FS_ANNOT - 0.4, pad=1.2, length=2)
    ins.set_xlabel("crosshead strain", fontsize=FS_ANNOT - 0.4, labelpad=1.0)
    ins.grid(False)
    ins.text(0.20, 55, f"{len(curves)} coupons", ha="right", va="top", fontsize=FS_ANNOT - 0.4,
             color=REAL)
    ins.text(0.20, plateau_meas - 3, f"{plateau_meas:.1f} MPa", ha="right", va="top",
             fontsize=FS_ANNOT - 0.4, color=INK)
    return sig_plateau


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="paper/Images/material_law.png")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    eps_f = PETIsotropic().eps_f
    eps_meas = max(e for _, e in PET_PLASTIC if e <= np.log(3.4))   # last coupon-derived point
    curves, lam, plateau_meas, yield_peak = load_coupons()

    apply_charter(**paper_rc())
    fig, ax = plt.subplots(figsize=figsize_in(WIDTH_BP, HEIGHT_BP))
    sig_plateau = draw(ax, curves, lam, plateau_meas, eps_meas, eps_f)
    fig.tight_layout()
    pdf, lo, hi = save_paper_figure(fig, args.out, WIDTH_BP, HEIGHT_BP, dpi=args.dpi)
    print(f"  wrote {pdf} ({WIDTH_BP:.1f} x {HEIGHT_BP:.1f} bp, type {lo:.1f}-{hi:.1f} pt)")
    plt.close(fig)

    prov = dict(n_coupons=len(curves), lam=lam, sig_plateau=sig_plateau,
                plateau_measured=plateau_meas, yield_peak=yield_peak, eps_meas=eps_meas,
                eps_f=eps_f, n_points=len(PET_PLASTIC), E=PET["E"], nu=PET["nu"])
    _caption(os.path.splitext(args.out)[0] + ".txt", prov)
    print(f"  wrote {args.out}\n  wrote {pdf}")
    for k, v in prov.items():
        print(f"  {k}: {v}")


def _caption(path, p):
    with open(path, "w") as f:
        f.write(
            "The PET constitutive law identified from tensile coupons.\n"
            "Generated by nff/scripts/figures/plot_material_law.py.\n\n"
            f"The {p['n_points']}-point hardening table handed to the solver, in true stress "
            "against true plastic strain. Filled markers are coupon-derived and open markers "
            "interpolated. The measured natural draw ratio "
            f"{p['lam']:.2f} places the anchor at {p['eps_meas']}, and the table follows the "
            "neck-propagation locus to it. Beyond that point the table is interpolated to the "
            f"fracture strain {p['eps_f']}, which is an interpolation between two measured "
            f"endpoints rather than four measurements. Inset: the {p['n_coupons']} coupons that "
            "drew stably, as the frame records them. Their strain axis carries the machine and "
            "grip compliance, so it is not the material's strain. The two "
            f"stress levels they fix, an upper yield near {p['yield_peak']:.0f} MPa and a draw "
            f"plateau at {p['plateau_measured']:.1f} MPa, need no strain measurement.\n\n"
            "Provenance, not part of the caption.\n"
            "Inputs: data/experiments/raw/kirigami_20260723 (per-specimen curves and the operator "
            "summary sheet), nff/rve/materials/pet.py (the law itself; the table is read from the "
            "code and never retyped).\n"
            f"Elastic constants carried by the same object: E = {p['E']:.0f} MPa, nu = {p['nu']}.\n"
            f"Draw plateau: {p['sig_plateau']:.1f} MPa implied by the anchor and the draw ratio, "
            f"against {p['plateau_measured']:.1f} MPa measured as the median engineering stress "
            "over the interior of the draw. The two agree to 1.3 per cent, so the locus is not a "
            "fit.\n"
            "Regenerate: JAX_PLATFORMS=cpu conda run -n kgnn_mac --no-capture-output "
            "python nff/scripts/figures/plot_material_law.py\n")


if __name__ == "__main__":
    main()
