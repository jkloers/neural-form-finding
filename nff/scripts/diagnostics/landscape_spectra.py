"""Spectral and geometric diagnostics of the design landscape.

Three instruments on the same object, each answering a question the others cannot.

  E  FREQUENCY ANALYSIS. Sample L on a UNIFORM grid along a transect and take the power spectral
     density. The decay exponent is diagnostic of the regularity class, and the correspondence is
     exact: a step discontinuity has Fourier transform ~1/f so PSD ~ f^-2; a C^1 kink gives f^-4; a
     smooth landscape decays faster than any power. Independent of the bracket-shrinking test that
     first found the discontinuity, so agreement between the two is genuine corroboration.
     ⚠ Uniform sampling is mandatory. A log grid destroys the transform.

  F  THE GEOMETRY THE CHART INDUCES. The pullback metric G = J^T J describes how the chart stretches
     design space; its spectrum is the anisotropy of the parameterisation itself, before any
     objective. The departure rate ||B dy|| / ||dy|| for a unit parameter step measures how fast a
     straight step in the design variables leaves the feasible set -- identically zero for a chart
     whose tangent image lies in ker(B), order one for free vertices.

  G  HESSIAN SPECTRAL DENSITY. Conditioning drawn as a distribution rather than a single ratio.
     A bare condition number was destroyed by Cauchy interlacing; a full spectrum shown against the
     null distribution of random subspaces of the same dimension cannot be.

    JAX_PLATFORMS=cpu conda run -n kgnn_mac python nff/scripts/diagnostics/landscape_spectra.py
"""

import argparse
import os

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from nff.analysis.chart_ablation import (arm_validity_cfg, build_geom_loss_fn, chart_basis,
                                       chart_jacobian, closed_start_geometry, geom_shape,
                                       hinge_gap_operator)
from nff.config.experiment import load_and_parse_config


def psd_of_transect(lf, y0, d, half_width, n):
    """Loss on a uniform transect, its PSD, and the fitted decay exponent."""
    t = np.linspace(-half_width, half_width, n)
    v = np.array([float(lf(jnp.asarray(y0 + tt * d))) for tt in t])
    v = v - v.mean()
    w = np.hanning(n)                       # taper: a finite window otherwise leaks f^-2 of its own
    F = np.fft.rfft(v * w)
    psd = (np.abs(F) ** 2)[1:]
    freq = np.fft.rfftfreq(n, d=t[1] - t[0])[1:]
    band = (freq > freq[2]) & (freq < freq[-1] * 0.5)      # drop DC-adjacent and Nyquist-adjacent
    alpha = -np.polyfit(np.log10(freq[band]), np.log10(psd[band]), 1)[0]
    return freq, psd, float(alpha)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="closed/sheet_4x8ft_rom_28deg")
    ap.add_argument("--out", default="data/outputs/chart_ablation_28deg/landscape_spectra.npz")
    ap.add_argument("--n-samp", type=int, default=512)
    ap.add_argument("--half-width", type=float, default=4e-3)
    args = ap.parse_args()

    cfg = load_and_parse_config(f"data/configs/{args.config}.yaml")
    y0, _, _, state, _ = closed_start_geometry(cfg)
    y0 = np.asarray(y0)
    F_, K = geom_shape(state)
    B = hinge_gap_operator(state)
    J, _, _, _ = chart_jacobian(cfg)
    Q_A, k_A, sv = chart_basis(J)
    rng = np.random.default_rng(0)

    # ── E: frequency analysis ────────────────────────────────────────────────────────────
    P_row = B.T @ np.linalg.solve(B @ B.T, B)
    v = rng.normal(size=y0.size)
    dirs = {'transverse (crosses the shell)': P_row @ v,
            'tangent to the shell': v - P_row @ v,
            'closed-chart direction': Q_A[:, 1]}
    loss_fn, _, _, _ = build_geom_loss_fn(cfg)
    lf = jax.jit(loss_fn)
    psd = {}
    print("  E · power spectral density of the loss along uniform transects")
    print(f"    {'direction':<32}{'fitted alpha':>14}   regularity implied")
    for name, d in dirs.items():
        d = d / np.linalg.norm(d)
        f, p, a = psd_of_transect(lf, y0, d, args.half_width, args.n_samp)
        psd[name] = (f, p, a)
        implied = ('jump  (f^-2)' if a < 3 else 'kink  (f^-4)' if a < 5 else 'smooth (steeper)')
        print(f"    {name:<32}{a:14.2f}   {implied}")

    # ── F: the geometry the chart induces ───────────────────────────────────────────────
    G = J.T @ J
    gev = np.linalg.eigvalsh(G)
    gev = gev[gev > gev.max() * 1e-12]
    dep_chart = np.linalg.norm(B @ Q_A) / np.linalg.norm(Q_A)
    R = np.linalg.qr(rng.normal(size=(y0.size, k_A)))[0]
    dep_free = np.linalg.norm(B @ R) / np.linalg.norm(R)
    print(f"\n  F · pullback metric G = J^T J   ({gev.size} live directions)")
    print(f"    eigenvalue range {gev.min():.3e} .. {gev.max():.3e}"
          f"   anisotropy {gev.max()/gev.min():.1f}")
    print(f"    departure rate ||B d||/||d|| for a unit design step:")
    print(f"      closed chart      {dep_chart:.3e}   <- leaves the feasible set at machine zero")
    print(f"      random 12-frame   {dep_free:.3e}")

    # ── G: Hessian spectral density ─────────────────────────────────────────────────────
    nm = np.load("data/outputs/chart_ablation_28deg/null_model.npz")
    H, QA = nm["H"], nm["Q_A"]
    ev_full = np.linalg.eigvalsh(H)
    ev_chart = np.linalg.eigvalsh(QA.T @ H @ QA)
    null_ev = []
    for _ in range(60):
        Q = np.linalg.qr(rng.normal(size=(H.shape[0], QA.shape[1])))[0]
        null_ev.append(np.linalg.eigvalsh(Q.T @ H @ Q))
    null_ev = np.concatenate(null_ev)
    print(f"\n  G · Hessian spectral density")
    print(f"    ambient (72)      |lambda| {np.abs(ev_full).min():.2e} .. {np.abs(ev_full).max():.2e}"
          f"   negative modes {(ev_full < 0).sum()}")
    print(f"    closed chart (12) |lambda| {np.abs(ev_chart).min():.2e} .. "
          f"{np.abs(ev_chart).max():.2e}   negative modes {(ev_chart < 0).sum()}")
    print(f"    random 12-dim     |lambda| {np.abs(null_ev).min():.2e} .. "
          f"{np.abs(null_ev).max():.2e}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, gev=gev, sv=sv, dep_chart=dep_chart, dep_free=dep_free,
                        ev_full=ev_full, ev_chart=ev_chart, null_ev=null_ev,
                        **{f"psd_freq__{k}": v[0] for k, v in psd.items()},
                        **{f"psd_val__{k}": v[1] for k, v in psd.items()},
                        **{f"psd_alpha__{k}": v[2] for k, v in psd.items()})
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
