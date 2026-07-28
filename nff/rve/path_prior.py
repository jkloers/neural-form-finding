"""Aim the oracle campaign at the region the deployed sheet actually visits.

The harvest (``nff.closed.path_dataset``) records every hinge's displacement path under the
linear-spring ROM. Two things about it are trustworthy and one is not:

* trustworthy -- the **envelope**: how far the hinge translates and rotates, and in which
  directions. That is what told us 40% of measured points are COMPRESSIVE (``a < 0``), a region the
  oracle's legacy box ``eta_a in [0, 1]`` excluded outright.
* trustworthy -- the fact that ``(a, s)`` in **mm** is the transferable quantity. The harvests on
  disk store ``eta = (a,s)/w_lig`` at ``w_lig = 5 mm`` while the standard is 18 mm, so any campaign
  that sampled eta directly would be silently rescaled by 3.6x. The net's own features are physical
  (``a, s, theta, log w_lig, alpha``), so sampling in mm and DERIVING eta is also the natural choice.
* NOT trustworthy -- the **density**. Paths are minimisers of the ROM's energy, not the true
  hinge's, and the ROM's linear ``k_rot`` under-prices the first few degrees of rotation (59 vs
  146 N.mm at 7 deg), so it produces rotation-first routes that cost 65% more than the real hinge
  would pay. Sampling that density would inherit the bias.

⟹ take the envelope as a REGION and explore it uniformly (user-directed 2026-07-28), rather than
resampling measured endpoints.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from nff.rve.hinge_function import DeploymentRay, HingeGeometry


@dataclass(frozen=True)
class Envelope:
    """Per-axis bounds of the measured hinge motion, in PHYSICAL units."""
    a: tuple[float, float]                            # axial [mm]; negative = compression
    s: tuple[float, float]                            # shear [mm]
    theta_deg: tuple[float, float]
    alpha_deg: tuple[float, float]
    n_points: int
    source: str

    THETA_MAX_DEG = 90.0                              # the mechanism's own limit, not a margin

    def inflate(self, factor: float) -> "Envelope":
        """Grow the translation axes about zero -- the extrapolation margin the gradient needs.

        The surrogate's GRADIENT drives the optimizer, which leaves the measured manifold
        immediately, so a net trained only inside the envelope extrapolates blind on its first step.

        Rotation is NOT inflated past 90 deg, and is floored at zero. Both bounds are physical
        rather than statistical: negative theta is adjacent tiles passing through each other, and
        beyond 90 deg the rotating-tile mechanism has closed on itself (the user discarded every
        harvested path above 90 deg for the same reason).
        """
        f = float(factor)
        return Envelope(a=(self.a[0] * f, self.a[1] * f), s=(self.s[0] * f, self.s[1] * f),
                        theta_deg=(self.theta_deg[0],
                                   min(self.theta_deg[1] * f, self.THETA_MAX_DEG)),
                        alpha_deg=self.alpha_deg, n_points=self.n_points,
                        source=f"{self.source} x{f:g}")

    def as_dict(self) -> dict:
        return dict(a_mm=list(self.a), s_mm=list(self.s), theta_deg=list(self.theta_deg),
                    alpha_deg=list(self.alpha_deg), n_points=self.n_points, source=self.source)


def measure_envelope(prior_dir: str, q: float = 0.5) -> Envelope:
    """Quantile-trimmed bounds of a harvest, de-normalised to millimetres.

    Args:
        prior_dir: a ``data/fea/path_priors/<name>/`` directory.
        q: percent trimmed from each tail (0.5 -> the p0.5..p99.5 range). Trimming matters: a
           single diverged deployment once reached ``eta_a = -1.1e42``, which is finite and would
           otherwise define the box.

    Returns:
        The measured :class:`Envelope`.
    """
    npz = np.load(os.path.join(prior_dir, "paths.npz"))
    A = {k: npz[k] for k in npz.files}                 # materialise once: npz re-decompresses
    manifest = json.load(open(os.path.join(prior_dir, "manifest.json")))
    w_lig_harvest = float(manifest["w_lig_mm"])

    eta, alpha = A["eta"], A["alpha"]                  # (E, T+1, H, 3), (E, H)
    mask = A["hinge_mask"] if "hinge_mask" in A else np.ones(alpha.shape, bool)
    pts = eta[np.broadcast_to(mask[:, None, :], eta.shape[:3])]      # (n_points, 3)
    a_mm, s_mm = pts[:, 0] * w_lig_harvest, pts[:, 1] * w_lig_harvest
    th_deg = np.degrees(pts[:, 2])
    al_deg = np.degrees(alpha[mask])

    lo, hi = q, 100.0 - q
    return Envelope(a=tuple(np.percentile(a_mm, [lo, hi])),
                    s=tuple(np.percentile(s_mm, [lo, hi])),
                    theta_deg=(0.0, float(np.percentile(th_deg, hi))),
                    alpha_deg=tuple(np.percentile(al_deg, [lo, hi])),
                    n_points=int(pts.shape[0]),
                    source=os.path.basename(prior_dir.rstrip("/")))


def _lhs(n, d, seed):
    """Stratified Latin hypercube in [0,1)^d -- even coverage without a grid's aliasing."""
    rng = np.random.default_rng(seed)
    u = (np.arange(n)[:, None] + rng.random((n, d))) / max(n, 1)
    for j in range(d):
        rng.shuffle(u[:, j])
    return u


def _lerp(u, lo, hi):
    return lo + u * (hi - lo)


def sample_campaign_jobs(n, envelope, *, seed=0, w_lig=(5.0, 50.0), fillet_ratio=0.16,
                         n_steps=30, spine_frac=0.25, inflate_frac=0.30, inflate=1.5,
                         eta_cap=1.2):
    """Sample the campaign -> list of ``(HingeGeometry, DeploymentRay)``.

    Three groups, all sharing one geometry sampler (``w_lig`` log-uniform over the DESIGN range,
    ``alpha`` over the measured range):

    * **spine** (``spine_frac``) -- theta prescribed, ``a`` and ``s`` LEFT FREE. The solver finds
      the opening that minimises energy at each rotation, so these are least-energy routes generated
      by the physics rather than inherited from the ROM. Measured: at a 28 deg fold the solver picks
      ``a = -4.10 mm`` and pays 23% less than a forced small opening -- the natural route is
      COMPRESSIVE, which the legacy box could not represent at all.
    * **fan, in-envelope** (the rest, minus ``inflate_frac``) -- a full 3-D ``(a, s, theta)`` point
      drawn by LHS across the measured envelope and driven as a straight ray. This is what covers
      the compression-, tension- and shear-dominated corners; the spine alone is a 1-D curve and
      would leave the surrogate blind off it.
    * **fan, inflated** (``inflate_frac``) -- the same, over an ``inflate``x envelope.

    ``eta_cap`` rejects combinations whose derived ``|a|/w_lig`` or ``|s|/w_lig`` exceeds it: the
    envelope is in mm and ``w_lig`` is drawn independently, so a 15 mm translation on a 5 mm ligament
    is a guaranteed tear that will not converge and carries no information.

    Returns:
        Shuffled ``[(geo, ray), ...]`` so a partial run stays representative of the whole design.
    """
    n_spine = int(round(spine_frac * n))
    n_fan = n - n_spine
    n_infl = int(round(inflate_frac * n_fan))
    jobs = []

    us = _lhs(n_spine, 3, seed)
    for i in range(n_spine):
        geo = HingeGeometry(float(np.exp(_lerp(us[i, 0], *np.log(w_lig)))),
                            float(_lerp(us[i, 1], *envelope.alpha_deg)), fillet_ratio)
        jobs.append((geo, DeploymentRay(float(_lerp(us[i, 2], *envelope.theta_deg)),
                                        0.0, 0.0, n_steps, f"sp{i:05d}", free_dofs=("a", "s"))))

    for grp, (count, env, tag) in enumerate([(n_fan - n_infl, envelope, "fn"),
                                             (n_infl, envelope.inflate(inflate), "fx")]):
        uf = _lhs(count, 5, seed + 1 + grp)
        for i in range(count):
            wl = float(np.exp(_lerp(uf[i, 0], *np.log(w_lig))))
            a = float(_lerp(uf[i, 2], *env.a))
            sh = float(_lerp(uf[i, 3], *env.s))
            a = float(np.clip(a, -eta_cap * wl, eta_cap * wl))      # keep eta physical at small w_lig
            sh = float(np.clip(sh, -eta_cap * wl, eta_cap * wl))
            geo = HingeGeometry(wl, float(_lerp(uf[i, 1], *env.alpha_deg)), fillet_ratio)
            jobs.append((geo, DeploymentRay(float(_lerp(uf[i, 4], *env.theta_deg)),
                                            a / wl, sh / wl, n_steps, f"{tag}{i:05d}")))

    np.random.default_rng(seed + 7).shuffle(jobs)
    return jobs
