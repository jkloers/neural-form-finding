"""Plot a load sweep from ``sweep_localisation.py``: kinematics, load path, and voids together.

Three rows, one per question. The point of drawing them on one shared load axis is that the three
departures are not independent -- the sheet leaves the 1-DOF family, concentrates its force, and
bends its voids out of parallel at the same time -- so the rows are meant to be read down a column.

Run:
    JAX_PLATFORMS=cpu PYTHONPATH=$(pwd) python nff/scripts/figures/plot_localisation_sweep.py \
        --run data/outputs/localisation/open_squares
"""

import argparse
import json
import os

import numpy as np

from nff.utils import figstyle as FST


def _load(run_dir: str):
    with open(os.path.join(run_dir, "summary.json")) as fh:
        meta = json.load(fh)
    npz = np.load(os.path.join(run_dir, "sweep.npz"))
    return meta, npz


def _series(rows, *path, default=np.nan):
    """Pull a possibly-nested key out of every row, as a float array."""
    out = []
    for r in rows:
        v = r
        for k in path:
            v = v.get(k, {}) if isinstance(v, dict) else {}
        out.append(float(v) if isinstance(v, (int, float)) else default)
    return np.asarray(out, dtype=float)


def _xlabel(axis: str) -> str:
    return "opening force (N)" if axis == "open" else "cantilever / opening force"


def plot_curves(meta, npz, out, scale=1.0):
    import matplotlib.pyplot as plt

    rows = meta["rows"]
    axis = meta["axis"]
    x = np.asarray([r["level"] for r in rows], dtype=float)
    deg = np.degrees

    w_bp, h_bp = FST.page_bp(1.0, aspect=0.95)
    with plt.rc_context(FST.paper_rc(scale=scale)):
        fig, ax = plt.subplots(3, 3, figsize=FST.figsize_in(w_bp, h_bp))

        # ── row 1: is the deployment 1-DOF? ─────────────────────────────────────────────
        a = ax[0, 0]
        a.plot(x, deg(np.abs(_series(rows, "psi"))), "o-", color=FST.ORANGE, ms=2.4, lw=1.0,
               label=r"$|\psi|$ (mechanism)")
        a.plot(x, deg(_series(rows, "theta_rel_max")), "-", color=FST.INK, lw=0.9,
               label=r"$\vartheta_{\max}$")
        a.plot(x, deg(_series(rows, "theta_rel_min")), "-", color=FST.GREY, lw=0.9,
               label=r"$\vartheta_{\min}$")
        a.set_ylabel("angle (deg)")
        a.legend(frameon=False, fontsize=FST.FS_ANNOT)

        a = ax[0, 1]
        a.plot(x, 1.0 - _series(rows, "r2"), "o-", color=FST.TEAL, ms=2.4, lw=1.0)
        a.set_yscale("log")
        a.set_ylabel(r"$1-R^2$ of the 1-DOF fit")

        a = ax[0, 2]
        a.plot(x, _series(rows, "localisation"), "o-", color=FST.RED, ms=2.4, lw=1.0,
               label=r"$\vartheta_{\max}/2|\psi|$")
        a.axhline(1.0, color=FST.GREY, lw=0.6, ls=":")
        a.set_ylabel("localisation ratio")
        a.legend(frameon=False, fontsize=FST.FS_ANNOT)

        # ── row 2: how is the force organised? ──────────────────────────────────────────
        a = ax[1, 0]
        for key, col, lab in (("tension_share", FST.ORANGE, "tension"),
                              ("compression_share", FST.TEAL, "compression"),
                              ("shear_share", FST.PURPLE, "shear")):
            a.plot(x, _series(rows, "budget", key), "-", color=col, lw=1.0, label=lab)
        a.set_ylabel("share of transmitted force")
        a.legend(frameon=False, fontsize=FST.FS_ANNOT)

        a = ax[1, 1]
        a.plot(x, _series(rows, "budget", "gini"), "o-", color=FST.INK, ms=2.4, lw=1.0,
               label="Gini")
        a.plot(x, _series(rows, "budget", "active_fraction"), "-", color=FST.GREEN, lw=1.0,
               label="active fraction")
        a.set_ylabel("concentration")
        a.legend(frameon=False, fontsize=FST.FS_ANNOT)

        a = ax[1, 2]
        a.plot(x, _series(rows, "checkerboard", "corr_detrended"), "o-", color=FST.PURPLE,
               ms=2.4, lw=1.0, label="detrended")
        a.plot(x, _series(rows, "checkerboard", "corr_raw"), "-", color=FST.MUTE, lw=1.0,
               label="raw")
        a.axhline(0.0, color=FST.GREY, lw=0.6, ls=":")
        a.set_ylabel("checkerboard projection")
        a.legend(frameon=False, fontsize=FST.FS_ANNOT)

        # ── row 3: how far from a parallelogram are the voids? ──────────────────────────
        a = ax[2, 0]
        a.plot(x, _series(rows, "para_mean"), "o-", color=FST.ORANGE, ms=2.4, lw=1.0, label="mean")
        a.plot(x, _series(rows, "para_max"), "-", color=FST.RED, lw=1.0, label="max")
        a.set_yscale("log")
        a.set_ylabel("void parallelogram defect")
        a.legend(frameon=False, fontsize=FST.FS_ANNOT)

        a = ax[2, 1]
        a.plot(x, _series(rows, "gap_mean"), "o-", color=FST.TEAL, ms=2.4, lw=1.0, label="mean")
        a.plot(x, _series(rows, "gap_max"), "-", color=FST.RED, lw=1.0, label="max")
        a.set_yscale("log")
        a.set_ylabel("pin gap (mm)")
        a.legend(frameon=False, fontsize=FST.FS_ANNOT)

        # The link: the void defect against the 1-DOF residual, both per load level.
        a = ax[2, 2]
        rms = _series(rows, "rms_over_psi")
        para = _series(rows, "para_mean")
        ok = np.isfinite(rms) & np.isfinite(para) & (rms > 0) & (para > 0)
        sc = a.scatter(rms[ok], para[ok], c=np.log10(x[ok]), cmap=FST.charter_diverging(),
                       s=9, zorder=3)
        if ok.sum() > 2:
            rho = np.corrcoef(np.log(rms[ok]), np.log(para[ok]))[0, 1]
            a.set_title(f"log-log r = {rho:+.3f}", fontsize=FST.FS_ANNOT)
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel(r"1-DOF residual  rms$(\rho)/|\psi|$")
        a.set_ylabel("void parallelogram defect")
        cb = fig.colorbar(sc, ax=a, pad=0.02)
        cb.set_label(r"$\log_{10}$ load", fontsize=FST.FS_ANNOT)

        for a in ax.ravel()[:8]:
            a.set_xscale("log")
            a.set_xlabel(_xlabel(axis))
        for i, a in enumerate(ax.ravel()):
            FST.panel_letter(a, "abcdefghi"[i], scale=scale)

        fig.suptitle(f"{meta['arm']}  /  {axis} sweep", fontsize=FST.FS_ANNOT)
        fig.tight_layout()
        FST.save_paper_figure(fig, out, w_bp, h_bp, scale=scale)
    print(f"[plot] wrote {out}.pdf / .png")


def plot_maps(meta, npz, out, scale=1.0, picks=(0, None, -1)):
    """Spatial maps at three load levels: angle residual, force, void defect."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    from nff.closed.void_geometry import build_void_index
    from nff.topology.closed_builder_jax import build_deploy_structure

    rows = meta["rows"]
    n = len(rows)
    idx_picks = [0, n // 2 if picks[1] is None else picks[1], n - 1]
    struct = build_deploy_structure(int(meta.get("M", 15)), int(meta.get("N", 10)))
    vidx = build_void_index(struct)

    w_bp, h_bp = FST.page_bp(1.0, aspect=0.62)
    with plt.rc_context(FST.paper_rc(scale=scale)):
        fig, ax = plt.subplots(3, 3, figsize=FST.figsize_in(w_bp, h_bp))
        for c, i in enumerate(idx_picks):
            resid = npz[f"residual__{i}"]
            dist = npz[f"distance__{i}"]
            para = npz[f"para_defect__{i}"]
            lvl = rows[i]["level"]

            a = ax[0, c]
            a.scatter(dist, np.degrees(np.abs(resid)), s=5, color=FST.TEAL)
            a.set_yscale("log")
            a.set_xlabel("hops from the loaded tile")
            if c == 0:
                a.set_ylabel(r"$|\rho|$ (deg)")
            a.set_title(f"level = {lvl:.3g}", fontsize=FST.FS_ANNOT)

            a = ax[1, c]
            f_mag = npz[f"f_mag__{i}"]
            fp = npz[f"face_pairs__{i}"]
            hd = np.minimum(dist[fp[:, 0]], dist[fp[:, 1]])
            a.scatter(hd, f_mag, s=5, color=FST.ORANGE)
            a.set_yscale("log")
            a.set_xlabel("hops from the loaded tile")
            if c == 0:
                a.set_ylabel("interface force")

            a = ax[2, c]
            vd = np.mean(dist[vidx.owner_face.reshape(vidx.n_voids, -1)], axis=1)
            a.scatter(vd, para, s=5, color=FST.PURPLE)
            a.set_yscale("log")
            a.set_xlabel("hops from the loaded tile")
            if c == 0:
                a.set_ylabel("void defect")

        for i, a in enumerate(ax.ravel()):
            FST.panel_letter(a, "abcdefghi"[i], scale=scale)
        fig.tight_layout()
        FST.save_paper_figure(fig, out, w_bp, h_bp, scale=scale)
    print(f"[plot] wrote {out}.pdf / .png")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--which", default="both", choices=("curves", "maps", "both"))
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()

    FST.use_agg()
    FST.apply_charter()
    meta, npz = _load(args.run)
    base = args.out or os.path.join(args.run, "sweep")
    if args.which in ("curves", "both"):
        plot_curves(meta, npz, base + "_curves", scale=args.scale)
    if args.which in ("maps", "both"):
        plot_maps(meta, npz, base + "_maps", scale=args.scale)


if __name__ == "__main__":
    main()
