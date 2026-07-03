"""Strain energy W vs each kinematic (theta, shear s, stretch a), with the recalibrated
linear ROM overlaid and the elastic / plastic / fracture regimes shaded from the data.

Follows the project charter (nff/scripts/validate_hinge.py palette + clean 1x3 style); nothing
generic is hardcoded. Run in an env with matplotlib (e.g. the ``ccx`` env):

    conda run -n ccx python nff/scripts/plot_hinge_energy_regimes.py \
        --data sofa/output/hinge_dataset --out data/outputs/hinge_energy_regimes.png

The dispersion band is the 10-90th percentile of W across all OTHER variables (geometry w_lig,
alpha, and the other two kinematics) -- so it is wide, dominated by w_lig. The ROM is the linear
spring model W ~ k_s a^2 + k_b theta^2 + k_tau s^2, recalibrated on the near-origin (elastic)
samples; the FEA mean peels BELOW it once plasticity/buckling softens the response.
"""

import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── project charter ───────────────────────────────────────────────────────────────
ORANGE, TEAL, RED, GREY, INK = "#F58025", "#2A9D8F", "#D62828", "#6C757D", "#1A1A1A"
ZONE = {"elastic": "#EAF2F8", "plastic": "#FCF3E6", "fracture": "#FBEAEA"}   # light, non-obtrusive


def apply_charter():
    plt.rcParams.update({
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.6,
        "axes.edgecolor": INK, "axes.linewidth": 0.9, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": GREY, "ytick.color": GREY,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "legend.frameon": True, "legend.framealpha": 0.9, "legend.edgecolor": "#D3D6DB",
    })   # note: no font-family override -- inherit the project/user default


# (key, display transform, x-label, ROM term key, term label, symmetric-about-0?)
AXES = [
    ("theta", np.degrees,    r"rotation $\theta$ [deg]", "b", r"bending $k_b\,\theta^2$",  False),
    ("s",     lambda v: v,   r"shear $s$ [mm]",          "t", r"shear $k_\tau\,s^2$",      True),
    ("a",     lambda v: v,   r"stretch $a$ [mm]",        "s", r"stretch $k_s\,a^2$",       False),
]


def fit_rom(a, s, theta, W, theta_fit=0.05):
    """Recalibrate W ~ k_s a^2 + k_b theta^2 + k_tau s^2 on small-deformation samples
    (theta < theta_fit rad, the elastic branch); non-negative stiffnesses."""
    m = theta < theta_fit
    A = np.stack([a[m] ** 2, theta[m] ** 2, s[m] ** 2], axis=1)
    k, *_ = np.linalg.lstsq(A, W[m], rcond=None)
    k_s, k_b, k_t = np.clip(k, 0.0, None)
    return dict(s=k_s, b=k_b, t=k_t)


def bin_axis(x, a, s, theta, W, peeq, n_bins=26, min_n=15):
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    cols = ("center", "meanW", "p10", "p90", "a2", "s2", "th2", "peeq")
    out = {k: [] for k in cols}
    for b in range(n_bins):
        m = idx == b
        out["center"].append(0.5 * (edges[b] + edges[b + 1]))
        if m.sum() < min_n:
            for k in cols[1:]:
                out[k].append(np.nan)
            continue
        out["meanW"].append(W[m].mean())
        out["p10"].append(np.percentile(W[m], 10)); out["p90"].append(np.percentile(W[m], 90))
        out["a2"].append((a[m] ** 2).mean()); out["s2"].append((s[m] ** 2).mean())
        out["th2"].append((theta[m] ** 2).mean()); out["peeq"].append(np.median(peeq[m]))
    out["edges"] = edges
    return {k: np.asarray(v) for k, v in out.items()}


def zone_boundaries(xc, peeq, eps_f, tol=1e-3):
    """Monotonic elastic->plastic->fracture boundaries in d=|x|: return (d_yield, d_frac)."""
    d = np.abs(xc)
    o = np.argsort(d)
    pm = np.maximum.accumulate(np.nan_to_num(peeq[o]))   # monotone in d -> single crossings
    def cross(thr):
        hit = np.where(pm >= thr)[0]
        return float(d[o][hit[0]]) if len(hit) else np.inf
    return cross(tol), cross(eps_f)


def plot_axis(ax, b, xform, xlabel, term_key, term_label, sym, rom, eps_f):
    xc = xform(b["center"])
    ex = xform(b["edges"])
    mean, p10, p90 = b["meanW"], b["p10"], b["p90"]
    good = np.isfinite(mean)
    ycap = 1.4 * np.nanmax(p90[good])

    # 1. regime background from monotonic boundaries (clean, no per-bin flicker)
    d_y, d_f = zone_boundaries(xc, b["peeq"], eps_f)
    d = np.abs(xc)
    zone = np.where(d < d_y, "elastic", np.where(d < d_f, "plastic", "fracture"))
    for i in range(len(xc)):
        ax.axvspan(ex[i], ex[i + 1], color=ZONE[zone[i]], alpha=0.6, lw=0, zorder=0)

    # 2. FEA: 10-90% dispersion band + mean
    ax.fill_between(xc[good], p10[good], p90[good], color=ORANGE, alpha=0.15, lw=0,
                    label="FEA 10–90%", zorder=2)
    ax.plot(xc[good], mean[good], "-", color=ORANGE, lw=2.3, label="FEA mean $W$", zorder=6)

    # 3. recalibrated ROM: all terms + the isolated term for this axis
    rom_full = rom["s"] * b["a2"] + rom["b"] * b["th2"] + rom["t"] * b["s2"]
    rom_iso = {"s": rom["s"] * b["a2"], "b": rom["b"] * b["th2"], "t": rom["t"] * b["s2"]}[term_key]
    ax.plot(xc[good], rom_full[good], "-", color=TEAL, lw=1.9, label="ROM (all terms)", zorder=4)
    ax.plot(xc[good], rom_iso[good], "--", color=TEAL, lw=1.6, label=f"ROM {term_label}", zorder=4)

    # 4. deviation: first bin (by increasing |x|) where the ROM over-predicts FEA by >10%
    o = np.argsort(d)
    fl = 0.03 * np.nanmax(mean[good])
    cand = [i for i in o if good[i] and mean[i] > fl and rom_full[i] > mean[i] * 1.10]
    if cand:
        i = cand[0]
        for xv in ([xc[i], -xc[i]] if sym else [xc[i]]):
            ax.axvline(xv, color=RED, ls=":", lw=1.4, zorder=5)
        ax.plot(xc[i], mean[i], "o", color=RED, ms=6, zorder=7)
        ax.annotate("FEA departs ROM", (xc[i], mean[i]), textcoords="offset points",
                    xytext=(10, 12), fontsize=8.5, color=RED, weight="bold", zorder=8)

    # 5. one label per (wide-enough) zone, from the boundaries -- transparent bbox, at the top
    xmax = np.abs(xc).max(); span = ex[-1] - ex[0]
    def label_zone(name, xpos, wide):
        if wide > 0.07 * span:
            ax.text(xpos, ycap * 0.955, name, ha="center", va="top", fontsize=9.5, color=GREY,
                    bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", boxstyle="round,pad=0.2"))
    label_zone("Elastic", 0.0 if sym else d_y / 2, d_y * (1 if not sym else 2))
    if np.isfinite(d_f):
        label_zone("Plastic", (d_y + d_f) / 2, d_f - d_y)
        label_zone("Fracture", (d_f + xmax) / 2, xmax - d_f)
    else:
        label_zone("Plastic", (d_y + xmax) / 2, xmax - d_y)

    ax.set_xlabel(xlabel); ax.set_ylabel(r"strain energy $W$ [N$\cdot$mm]")
    ax.set_xlim(ex[0], ex[-1]); ax.set_ylim(0, ycap)
    ax.legend(loc="lower right" if not sym else "upper center", fontsize=8, ncol=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="sofa/output/hinge_dataset")
    ap.add_argument("--out", default="data/outputs/hinge_energy_regimes.png")
    args = ap.parse_args()

    d = np.load(args.data + ".npz")
    eps_f = json.load(open(args.data + ".json"))["const"].get("eps_f", 0.25)
    a, s, theta, W, peeq = d["a"], d["s"], d["theta"], d["W"], d["peeq_p99"]
    rom = fit_rom(a, s, theta, W)

    apply_charter()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.9))
    for ax, (key, xform, xlabel, tk, tl, sym) in zip(axes, AXES):
        bb = bin_axis({"theta": theta, "s": s, "a": a}[key], a, s, theta, W, peeq)
        plot_axis(ax, bb, xform, xlabel, tk, tl, sym, rom, eps_f)
    fig.suptitle("Hinge strain energy across kinematic regimes  ·  FEA vs recalibrated linear ROM",
                 fontsize=13, color=INK, y=1.03)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"ROM: k_s={rom['s']:.1f}  k_b={rom['b']:.1f}  k_tau={rom['t']:.1f}  ->  {args.out}")


if __name__ == "__main__":
    main()
