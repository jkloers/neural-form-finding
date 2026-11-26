"""What the interaction range of the sheet depends on.

Four panels from the three arms of ``sweep_pocket.py``:

    a  how the disturbance falls off with distance, at several deployments
    b  the drop to the first neighbours against deployment, the headline dependence
    c  the same against probe magnitude, where linearity ends
    d  the same against the stiffness of the two stiff springs, the causal test

Run:
    JAX_PLATFORMS=cpu PYTHONPATH=$(pwd) python legacy/experiments/figures/plot_interaction_range.py \
        --sweep data/outputs/pocket/sweep --out paper/Images/interaction_range.pdf
"""

import argparse
import json
import os

import numpy as np

from nff.utils import figstyle as FS


def _load(sweep_dir, arm):
    path = os.path.join(sweep_dir, f"{arm}.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _kept(rows):
    """Levels that passed the stationarity and round-off gates."""
    return [r for r in rows if "rejected" not in r]


def _first_ratio(row, key="own"):
    return float(row[key]["ratios"][0])


def _decay(row, key="own"):
    return float(row[key]["decay_length"])


def plot(sweep_dir, out, scale=1.0):
    import matplotlib
    FS.use_agg()
    import matplotlib.pyplot as plt

    dep = _load(sweep_dir, "deployment")
    stf = _load(sweep_dir, "stiffness")
    mag = _load(sweep_dir, "magnitude")
    if dep is None:
        raise SystemExit(f"no deployment arm in {sweep_dir}; run sweep_pocket.py first")

    rows = _kept(dep["rows"])
    thetas = np.array([r["theta"]["mean"] for r in rows])
    w_bp, h_bp = FS.page_bp(width_frac=1.0, aspect=1.55)

    with matplotlib.rc_context(FS.paper_rc(scale=scale)):
        fig, axes = plt.subplots(2, 2, figsize=FS.figsize_in(w_bp, h_bp))

        # ── a: the fall-off itself, at several deployments ────────────────────
        ax = axes[0, 0]
        picks = np.linspace(0, len(rows) - 1, min(5, len(rows))).astype(int)
        cmap = plt.get_cmap("viridis")
        for j, i in enumerate(picks):
            shells = np.array(rows[i]["own"]["shells"], dtype=float)
            shells = shells / shells[0]
            hops = np.arange(shells.size)
            ax.semilogy(hops, shells, marker="o", ms=2.0, lw=0.9,
                        color=cmap(j / max(len(picks) - 1, 1)),
                        label=f"{thetas[i]:.0f}")
        ax.set_xlabel("distance from the load (panels)")
        ax.set_ylabel("relative stored energy")
        ax.set_xlim(0, 8)
        leg = ax.legend(frameon=False, fontsize=FS.printed_pt(FS.FS_ANNOT, scale),
                        title="hinge rotation (deg)",
                        title_fontsize=FS.printed_pt(FS.FS_ANNOT, scale), ncol=2)
        leg._legend_box.align = "left"
        FS.panel_letter(ax, "a", scale=scale)

        # ── b: against deployment ─────────────────────────────────────────────
        # Both series are drops per panel, so they share one axis, and sharing it is the point:
        # the first step climbs by a factor of seven across the range while every later step does
        # not move. One of the two numbers belongs to the sheet and the other to the deployment.
        ax = axes[0, 1]
        ax.semilogy(thetas, [_first_ratio(r) for r in rows], marker="o", ms=3.0, lw=1.0,
                    color=FS.INK, label="load to its neighbours")
        ax.semilogy(thetas, [float(r["own"]["ratios"][1]) for r in rows], marker="s", ms=2.5,
                    lw=0.9, color=FS.ORANGE, ls="--", label="each panel after")
        ax.set_xlabel("hinge rotation (deg)")
        ax.set_ylabel("drop per panel")
        ax.set_ylim(1.5, 40)
        ax.legend(frameon=False, fontsize=FS.printed_pt(FS.FS_ANNOT, scale), loc="upper left")
        FS.panel_letter(ax, "b", scale=scale)

        # ── c: against probe magnitude ────────────────────────────────────────
        ax = axes[1, 0]
        if mag is not None:
            levels = mag["levels"]
            x = np.array([r["probe_over_reference"] for r in levels])
            y = np.array([_first_ratio(r) for r in levels])
            ax.loglog(x, y, marker="o", ms=3.0, lw=1.0, color=FS.INK)
            ax.axhline(_first_ratio(mag["reference"]), color=FS.GREY, lw=0.8, ls=":")
            ax.set_xlabel("load (relative)")
            ax.set_ylabel("drop to the first neighbours")
        else:
            ax.axis("off")
        FS.panel_letter(ax, "c", scale=scale)

        # ── d: against the stiffness of the stiff springs ─────────────────────
        # Same two series and the same axis range as (b), because the comparison between the two
        # panels IS the result: opening the sheet moves the first drop by a factor of seven, and
        # stiffening the ligaments by two decades moves neither series at all. Left to itself the
        # axis would span 8.70 to 8.72 and draw a 0.3% wobble as a collapse.
        ax = axes[1, 1]
        if stf is not None:
            f = np.array([r["stiffness_factor"] for r in stf["rows"]])
            ax.loglog(f, [_first_ratio(r) for r in stf["rows"]], marker="o", ms=3.0, lw=1.0,
                      color=FS.INK, label="load to its neighbours")
            ax.loglog(f, [float(r["own"]["ratios"][1]) for r in stf["rows"]], marker="s", ms=2.5,
                      lw=0.9, color=FS.ORANGE, ls="--", label="each panel after")
            ax.set_xlabel("ligament stiffness (relative)")
            ax.set_ylabel("drop per panel")
            ax.set_ylim(1.5, 40)
        else:
            ax.axis("off")
        FS.panel_letter(ax, "d", scale=scale)

        fig.tight_layout(pad=0.5)
        FS.save_paper_figure(fig, out, w_bp, h_bp, scale=scale)
    print(f"[interaction-range] wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", default="data/outputs/pocket/sweep")
    ap.add_argument("--out", default="paper/Images/interaction_range.pdf")
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()
    plot(args.sweep, args.out, args.scale)


if __name__ == "__main__":
    main()
