"""Best-case calibration of the linear-spring reference model, Eq. (rom).

The manuscript scores eq:rom with the constants condensed at w_lig = 18 mm, which is the
calibration a user of that model would actually have. This script answers the obvious reply:
what if it were calibrated as well as it possibly could be? It fits k_a, k_s and k_theta by
non-negative least squares to the SAME held-out states the condensed constants are scored on,
in sample, so the result is a floor on the quadratic form rather than a fit that could be
tightened. Non-negativity keeps the fitted form a valid positive-semidefinite energy, which
eq:total-energy requires.

    PYTHONPATH=$(pwd) JAX_PLATFORMS=cpu conda run -n kgnn_mac \
        python nff_paper/analysis/rom_best_fit.py
"""
import numpy as np
from scipy.optimize import nnls
from nff_paper.figures._surrogate_eval import (load_scored, rom_energy, rel_rmse, ROM_BAND,
                                                 ROM_K_A, ROM_K_S, ROM_K_THETA)

sc = load_scored()
te = sc.test
band = te & sc.band()
u, W = sc.u[band], sc.data["W"][band]
n = int(band.sum())
print(f"n states = {n}   w_lig band = {ROM_BAND}   (paper says 205)")

floor = 1e-9
def med_rel(pred, true):
    return float(np.median(np.abs(pred - true) / np.maximum(np.abs(true), floor)))

# --- 1. condensed constants, as the paper reports ---
Wc = rom_energy(u)
print(f"\ncondensed  k=({ROM_K_A},{ROM_K_S},{ROM_K_THETA})")
print(f"  relRMSE = {rel_rmse(Wc, W)*100:.1f}%   median = {med_rel(Wc, W)*100:.1f}%")

# --- 2. best case: non-negative least squares on these same rows ---
A = np.stack([0.5*u[:,0]**2, 0.5*u[:,1]**2, 0.5*u[:,2]**2], axis=1)
k, _ = nnls(A, W)
Wb = A @ k
print(f"\nbest-case (NNLS, in-sample on the scored states)")
print(f"  k_a={k[0]:.1f}  k_s={k[1]:.1f}  k_theta={k[2]:.1f}  [N/mm, N/mm, N mm/rad^2]")
print(f"  relRMSE = {rel_rmse(Wb, W)*100:.1f}%   median = {med_rel(Wb, W)*100:.1f}%")

# --- 3. surrogate on the same rows, for reference ---
Ws = sc.W_pred[band]
print(f"\nsurrogate  relRMSE = {rel_rmse(Ws, W)*100:.2f}%   median = {med_rel(Ws, W)*100:.2f}%")

# --- 4. rotation-only best case (k_theta alone), since rotation carries the energy ---
At = (0.5*u[:,2]**2).reshape(-1,1)
kt, _ = nnls(At, W)
print(f"\nrotation-only best case: k_theta={kt[0]:.1f}"
      f"   relRMSE = {rel_rmse(At@kt, W)*100:.1f}%   median = {med_rel(At@kt, W)*100:.1f}%")
print(f"\ntheta range on these states: {np.degrees(u[:,2]).min():.1f} to {np.degrees(u[:,2]).max():.1f} deg")
