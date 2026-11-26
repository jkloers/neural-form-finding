"""Turn a load sweep into the three answers, with the suddenness tested rather than asserted.

"Is the transition sudden?" is the question the eye answers wrongly most often, so every series
gets the same three readings, and it matters which one licenses which word:

    bent   a continuous two-segment fit against a single line, compared by BIC.
           ⚠⚠ This tests ONLY "is the series something other than a straight line". Every
           U-shaped series passes it trivially, so `bent = YES` is NOT evidence of a transition
           and must never be reported as a knee on its own. It is here to locate curvature.
    jump   the largest change between ADJACENT levels over the median change. This is the
           discriminator: ~1 for a smooth series, and large only where the series steps.
    mono   whether the series is monotone. A non-monotone series has an interior optimum, which
           is a different finding from a transition and wants different words.

The decisive test is not in this file: it is whether a jump SURVIVES REFINING the sweep. A steep
but continuous stretch sampled coarsely produces a large jump ratio that collapses toward 1 when
the levels are packed closer; a genuine discontinuity keeps its ratio. Run a fine sweep across the
candidate interval and compare.

Run:
    JAX_PLATFORMS=cpu PYTHONPATH=$(pwd) python legacy/experiments/diagnostics/report_localisation.py \
        --run data/outputs/localisation/open_squares
"""

import argparse
import json
import os

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Shape tests on one series
# ══════════════════════════════════════════════════════════════════════════════

def _bic(rss: float, n: int, k: int) -> float:
    return n * np.log(max(rss, 1e-300) / n) + k * np.log(n)


def knee_test(x: np.ndarray, y: np.ndarray, min_side: int = 4) -> dict:
    """Continuous two-segment linear fit against one line, compared by BIC.

    ``x`` should already be in the variable the sweep is uniform in (log load here). Returns
    ``segmented=False`` when the single line wins, which is the honest default.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = x.size
    if n < 2 * min_side + 1:
        return {"segmented": False, "reason": "too few points", "n": int(n)}

    p1 = np.polyfit(x, y, 1)
    rss1 = float(np.sum((y - np.polyval(p1, x)) ** 2))
    bic1 = _bic(rss1, n, 2)

    best = None
    for i in range(min_side, n - min_side):
        xb = x[i]
        # continuous hinge basis: y = a + b1 * x + b2 * relu(x - xb)
        A = np.column_stack([np.ones(n), x, np.maximum(x - xb, 0.0)])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        rss = float(np.sum((y - A @ coef) ** 2))
        if best is None or rss < best[0]:
            best = (rss, xb, coef, i)
    rss2, xb, coef, i = best
    bic2 = _bic(rss2, n, 4)

    return {"segmented": bool(bic2 < bic1), "bic_line": float(bic1), "bic_segmented": float(bic2),
            "delta_bic": float(bic1 - bic2), "breakpoint_x": float(xb),
            "slope_before": float(coef[1]), "slope_after": float(coef[1] + coef[2]),
            "rss_line": rss1, "rss_segmented": rss2, "n": int(n),
            "resolution_x": float(np.median(np.diff(x))) if n > 1 else float("nan")}


def jump_test(x: np.ndarray, y: np.ndarray) -> dict:
    """Largest adjacent-level change relative to the median change.

    A smooth series has a ratio near 1. A limit point under force control shows up as a single
    adjacent pair carrying tens of times the typical change.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return {"ratio": float("nan"), "n": int(x.size)}
    d = np.abs(np.diff(y))
    med = float(np.median(d))
    i = int(np.argmax(d))
    return {"ratio": float(d[i] / med) if med > 0 else float("inf"),
            "at_x_lo": float(x[i]), "at_x_hi": float(x[i + 1]),
            "delta": float(d[i]), "median_delta": med, "n": int(x.size)}


def _series(rows, *path):
    out = []
    for r in rows:
        v = r
        for k in path:
            v = v.get(k, {}) if isinstance(v, dict) else {}
        out.append(float(v) if isinstance(v, (int, float)) else np.nan)
    return np.asarray(out, dtype=float)


SERIES = [
    ("Q1 log10(1-R2) of 1-DOF fit",   ("r2",),                         "one_minus"),
    ("Q1 log10 rms(rho)/|psi|",       ("rms_over_psi",),               "log"),
    ("Q1 log10 localisation ratio",   ("localisation",),               "log"),
    ("Q1 |swing| (deg)",              ("swing",),                      "abs_deg"),
    ("Q1 |psi| (deg)",                ("psi",),                        "abs_deg"),
    ("Q1 decay 1/lambda",             ("inv_decay_theta",),            "raw"),
    ("Q2 compression share",          ("budget", "compression_share"), "raw"),
    ("Q2 shear share",                ("budget", "shear_share"),       "raw"),
    ("Q2 tension share",              ("budget", "tension_share"),     "raw"),
    ("Q2 Gini",                       ("budget", "gini"),              "raw"),
    ("Q2 active fraction",            ("budget", "active_fraction"),   "raw"),
    ("Q2 checkerboard detrended",     ("checkerboard", "corr_detrended"), "raw"),
    ("Q2 clustering ratio",           ("clustering", "ratio"),         "raw"),
    ("Q3 void defect mean",           ("para_mean",),                  "log"),
    ("Q3 void defect max",            ("para_max",),                   "log"),
    ("Q3 pin gap mean (mm)",          ("gap_mean",),                   "log"),
]


def transform(y, kind):
    if kind == "one_minus":
        return np.log10(np.maximum(1.0 - y, 1e-16))
    if kind == "log":
        return np.log10(np.maximum(y, 1e-300))
    if kind == "abs_deg":
        return np.degrees(np.abs(y))
    return y


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(os.path.join(args.run, "summary.json")) as fh:
        meta = json.load(fh)
    rows = meta["rows"]
    lvl = np.asarray([r["level"] for r in rows], dtype=float)
    x = np.log10(lvl)

    conv = [bool(r.get("checks", {}).get("converged", False)) for r in rows]
    print(f"\n=== {meta['arm']} / {meta['axis']} sweep : {len(rows)} levels "
          f"{lvl[0]:.4g}..{lvl[-1]:.4g}, {sum(conv)}/{len(conv)} converged ===")
    if not all(conv):
        bad = [f"{lvl[i]:.4g}" for i, c in enumerate(conv) if not c]
        print(f"  !! NOT CONVERGED at levels: {', '.join(bad)}  (these are not reportable)")

    print(f"\n{'series':32} {'start':>11} {'end':>11} {'mono':>5} {'bent':>5} "
          f"{'curv at':>9} {'JUMP x':>7} {'jump at':>12}")
    out = {}
    for name, path, kind in SERIES:
        y_raw = _series(rows, *path)
        y = transform(y_raw, kind)
        k = knee_test(x, y)
        j = jump_test(x, y)
        good = np.isfinite(y)
        d = np.diff(y[good])
        mono = bool(np.all(d >= 0) or np.all(d <= 0)) if d.size else True
        brk = 10.0 ** k["breakpoint_x"] if k.get("segmented") else np.nan
        lo = j.get("at_x_lo")
        hi = j.get("at_x_hi")
        where = (f"{10.0 ** lo:.3g}-{10.0 ** hi:.3g}"
                 if lo is not None and np.isfinite(lo) else "-")
        print(f"{name:32} {y[0]:11.4g} {y[-1]:11.4g} {'yes' if mono else 'NO':>5} "
              f"{'yes' if k.get('segmented') else 'no':>5} "
              f"{brk:9.4g} {j['ratio']:7.1f} {where:>12}")
        out[name] = {"bent": k, "jump": j, "monotone": mono,
                     "start": float(y[0]), "end": float(y[-1]),
                     "raw_start": float(y_raw[0]), "raw_end": float(y_raw[-1]),
                     "transform": kind}

    # The jump ratios only mean something against the sweep's own resolution, so state it.
    grid = float(np.median(np.diff(lvl) / lvl[:-1])) if lvl.size > 1 else float("nan")
    print(f"\nsweep resolution: adjacent levels differ by {100 * grid:.1f}% on average. A large "
          f"JUMP is\nonly a discontinuity if it SURVIVES refining that spacing -- test it.")

    dest = args.out or os.path.join(args.run, "report.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"\n[report] wrote {dest}")


if __name__ == "__main__":
    main()
