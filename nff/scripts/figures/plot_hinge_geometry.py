"""Visualize the single-hinge RVE: (A) the to-scale cut pattern, (B) the deformed final state.

Panel A draws the RVE domain (the mechanically-active core: two tiles + ligament + the two cuts)
to scale in mm, annotated with the cut dimensions — what you laser-cut (extend the tiles + add
grip tabs beyond the window). Panel B parses the CalculiX .frd and shows the deformed shape,
colored by out-of-plane displacement uz — the ligament buckling.
"""
from __future__ import annotations

import argparse
import os
import re

import numpy as np

from nff.rve.geometry import RVEParams, build_rve_domain

INK, MUTE, MAT, ACC = "#212529", "#868e96", "#dbe4ee", "#E8590C"
EFLOAT = re.compile(r"[-+]?\d*\.\d+E[-+]\d+")


def parse_frd(path):
    """Return (coords[N,3], last-increment disp[N,3]) keyed by row order of the node block."""
    with open(path, encoding="latin-1", errors="ignore") as f:
        lines = f.readlines()
    # node block: between '2C' and '3C'
    nodes, order = {}, []
    i = 0
    while i < len(lines) and "2C" not in lines[i][:6]:
        i += 1
    i += 1
    while i < len(lines) and "3C" not in lines[i][:6]:
        if lines[i].startswith(" -1"):
            nid = int(lines[i][3:13])
            xyz = [float(v) for v in EFLOAT.findall(lines[i][13:])][:3]
            if len(xyz) == 3:
                nodes[nid] = xyz
                order.append(nid)
        i += 1
    # last DISP block
    disp_starts = [k for k, ln in enumerate(lines) if ln.startswith(" -4  DISP")]
    disp = {}
    if disp_starts:
        k = disp_starts[-1] + 1
        while k < len(lines) and lines[k].startswith(" -5"):
            k += 1
        while k < len(lines) and lines[k].startswith(" -1"):
            nid = int(lines[k][3:13])
            d = [float(v) for v in EFLOAT.findall(lines[k][13:])][:3]
            if len(d) == 3:
                disp[nid] = d
            k += 1
    ids = [n for n in order if n in disp]
    coords = np.array([nodes[n] for n in ids])
    d = np.array([disp[n] for n in ids])
    return coords, d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--w-lig", type=float, default=15.0)
    ap.add_argument("--fillet", type=float, default=0.16)
    ap.add_argument("--frd", default="/tmp/hinge/w015.000_a0090.0_f0.160_t/hinge.frd")
    ap.add_argument("--out", default="data/experiments/processed/pet_hinge_geometry.png")
    args = ap.parse_args()

    w = args.w_lig
    p = RVEParams(w_lig=w, w_c=0.2, alpha_deg=90.0, rho=args.fillet * w,
                  thickness=0.5, r_win=2.4 * w)
    dom = build_rve_domain(p)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "text.color": INK})
    fig = plt.figure(figsize=(13, 6))
    axA = fig.add_subplot(1, 2, 1)
    axB = fig.add_subplot(1, 2, 2, projection="3d")

    # ---- Panel A: cut pattern to scale ---------------------------------
    xs, ys = dom.exterior.xy
    axA.fill(xs, ys, color=MAT, ec=INK, lw=1.4, zorder=2)
    axA.plot([-p.r_win, p.r_win], [0, 0], color=ACC, lw=2.5, zorder=3)  # secondary cut (y=0)
    axA.annotate("secondary cut", (p.r_win * 0.45, 0), (p.r_win * 0.25, p.r_win * 0.28),
                 color=ACC, fontsize=9, arrowprops=dict(arrowstyle="->", color=ACC, lw=1))
    axA.annotate("main cut\n(slit + rounded tip)", (0, -w - 3), (-p.r_win * 0.9, -w - 10),
                 color=ACC, fontsize=9, arrowprops=dict(arrowstyle="->", color=ACC, lw=1))
    # ligament dimension
    axA.annotate("", (p.r_win * 0.14, 0), (p.r_win * 0.14, -w),
                 arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2))
    axA.text(p.r_win * 0.16, -w / 2, f"w_lig = {w:.0f} mm", fontsize=10, va="center", color=INK)
    axA.text(w * 0.25, -w - 1.5, f"fillet ρ = {args.fillet*w:.1f} mm", fontsize=8.5, color=MUTE)
    axA.text(4, -w * 0.5 - 6, "α = 90°\n(cut-to-cut)", fontsize=8.5, color=MUTE)
    axA.text(0, -p.r_win * 0.72, f"deformation domain  Ø ≈ {2*p.r_win:.0f} mm\n(tiles rigid beyond this)",
             fontsize=8.5, color=MUTE, ha="center")
    axA.set_aspect("equal")
    axA.set_xlabel("x  [mm]"); axA.set_ylabel("y  [mm]")
    axA.set_title("Cut pattern (to scale) — extend tiles + tabs beyond this",
                  fontsize=11, loc="left", color=INK)
    axA.grid(True, color="#eef1f4", lw=0.8)

    # ---- Panel B: deformed final state from .frd -----------------------
    if os.path.exists(args.frd):
        coords, disp = parse_frd(args.frd)
        defo = coords + disp
        uz = disp[:, 2]
        sc = axB.scatter(defo[:, 0], defo[:, 1], defo[:, 2], c=uz, cmap="RdBu_r",
                         s=3, vmin=-np.abs(uz).max(), vmax=np.abs(uz).max())
        cb = fig.colorbar(sc, ax=axB, shrink=0.6, pad=0.08)
        cb.set_label("out-of-plane uz [mm]", fontsize=9)
        # equal aspect
        rng = np.ptp(defo, axis=0); c = defo.mean(0); r = rng.max() / 2
        axB.set_xlim(c[0]-r, c[0]+r); axB.set_ylim(c[1]-r, c[1]+r); axB.set_zlim(c[2]-r, c[2]+r)
        axB.set_xlabel("x [mm]"); axB.set_ylabel("y [mm]"); axB.set_zlabel("z [mm]")
        axB.set_title(f"Deformed final state — ligament buckles {np.abs(uz).max():.0f} mm out of plane",
                      fontsize=11, loc="left", color=INK)
        axB.view_init(elev=22, azim=-60)
    else:
        axB.text2D(0.5, 0.5, "frd not found", transform=axB.transAxes, ha="center")

    fig.suptitle(f"PET single-hinge — Experiment 1 geometry & deformation (w_lig = {w:.0f} mm, t = 0.5 mm)",
                 fontsize=12.5, fontweight="bold", x=0.02, ha="left", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor="white")
    print(f"figure -> {args.out}")


if __name__ == "__main__":
    main()
