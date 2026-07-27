"""What DISTRIBUTION of ``(a, s, theta)`` paths does a tessellation ride, over random starts?

``nff.closed.hinge_paths`` measures the paths of ONE deployment. That is a single sample of a
random variable: the design (where the cuts sit, hence where the tiles start) is one particular
point, and the twelve hinge paths it produces are correlated by that shared geometry. Training the
surrogate on it would aim the oracle at one sheet rather than at the family.

This module runs the SAME setup -- same loads, same clamps, same target, same hinge model -- from
many randomized starting designs and pools the result. The randomization is the one already built
into ``nff.closed.setup.init_closed_les_params``: Gaussian noise of scale ``init_noise`` on ``z``
(the per-cut aspect-ratio logits, ``r = sigmoid(z)``) and on ``bnd_logits`` (the boundary sliders).
``apply_closed_les_mapping`` rebuilds every face centroid and node vector from those two arrays, so
perturbing them genuinely moves the tiles -- and the parameterization keeps the flat sheet valid by
construction (``r`` stays in (0,1); the ordered softmax keeps the boundary convex), so every sample
is a real sheet, not a broken one.

WHAT COMES OUT. ``ensemble_statistics`` reports the pooled marginals, the component correlations,
and a variance decomposition that answers "does the random start actually change the distribution,
or is the spread just hinge-to-hinge within any one sheet?". ``sampling_spec`` then writes the
measured distribution down in ``nff.rve.dataset.sample_jobs``' OWN keyword names, so it can be
handed to the oracle directly once the surrogate is ready to be retrained.

CAVEAT -- READ BEFORE TRUSTING ANY NUMBER. The distribution inherits the physics that produced it.
Under the ROM those are linear springs whose ``k_stretch/k_shear/k_rot`` are not yet calibrated, and
the target shape is not yet the one we mean to match. So the SHAPE of the answer (which directions
are correlated, how much of the sampled box is visited, whether the random start matters) is the
durable part; the absolute ranges must be re-measured once the springs are calibrated and the
target is fixed. ``sampling_spec`` stamps its own provenance with this in mind.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import copy
import json
import os

import numpy as np

from nff.closed.hinge_paths import (HingePaths, build_hinge_geometry_fn, extract_hinge_paths,
                                    path_diagnostics, straightness, save_paths)
from nff.models.hinge_surrogate import DOMAIN


# ── the ensemble ──────────────────────────────────────────────────────────────────

@dataclass
class EnsembleSample:
    """One deployment from one random starting design."""
    seed: int
    noise: float
    paths: HingePaths


@dataclass
class PathEnsemble:
    """Many deployments of the same setup from randomized starting designs."""
    samples: List[EnsembleSample] = field(default_factory=list)
    failures: List[dict] = field(default_factory=list)
    config_path: str = ""
    length_scale: float = 1.0
    n_load_steps: int = 0

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def n_hinges(self) -> int:
        return self.samples[0].paths.n_hinges if self.samples else 0

    def endpoints(self) -> np.ndarray:
        """(n_samples * n_hinges, 3) final ``(eta_a, eta_s, theta)`` -- one per hinge per design.

        This is the quantity a ``DeploymentRay`` is defined by, so it is the distribution the
        oracle's ``(theta1_deg, eta_a, eta_s)`` sampling should match.
        """
        return np.concatenate([s.paths.eta[-1] for s in self.samples], axis=0)

    def all_points(self) -> np.ndarray:
        """(n_samples * n_steps * n_hinges, 3) EVERY point of every path.

        The surrogate has to be accurate all the way along a path, not only at its endpoint, so the
        occupied region is measured from this rather than from ``endpoints``.
        """
        return np.concatenate([s.paths.eta.reshape(-1, 3) for s in self.samples], axis=0)

    def alphas(self) -> np.ndarray:
        """(n_samples * n_hinges,) per-hinge RVE-frame opening angle [rad] across the ensemble."""
        return np.concatenate([s.paths.alpha for s in self.samples], axis=0)

    def per_sample(self, fn) -> np.ndarray:
        return np.asarray([fn(s.paths) for s in self.samples])


def run_path_ensemble(config, *, n_samples: int = 24, noise: float = 0.5, seed0: int = 0,
                      n_load_steps: Optional[int] = None, load_scale: float = 1.0,
                      config_path: str = "", verbose: bool = True) -> PathEnsemble:
    """Deploy the same setup from ``n_samples`` randomized starting designs and extract every path.

    Args:
        config: parsed ``ExperimentConfig`` (deep-copied; the caller's object is untouched).
        n_samples: number of random starting designs.
        noise: ``init_noise`` -- Gaussian sigma on the design logits. 0 gives the deterministic
            uniform start ``n_samples`` times over, which is only useful as a degeneracy check.
        seed0: ``init_seed`` of the first sample; subsequent samples use ``seed0 + i``.
        n_load_steps: path resolution (overrides ``physics.num_load_steps``).
        load_scale: multiply every load by this. Applied BEFORE the state is built, because loads
            are baked into ``initial_state`` -- editing ``load_specs`` afterwards silently does
            nothing.
        verbose: print a line per sample.

    Returns:
        ``PathEnsemble``. Designs whose solve raises are recorded in ``.failures`` rather than
        aborting the sweep -- a random start can be geometrically valid yet numerically awkward,
        and losing the whole ensemble to one of them would be worse than reporting the attrition.
    """
    from nff.closed.setup import (build_closed_initial_state, init_closed_les_params,
                                  build_surrogate_energy)
    from nff.stages.pipeline import forward_pipeline

    config = copy.deepcopy(config)
    if n_load_steps is not None:
        config.physics.num_load_steps = int(n_load_steps)
    if load_scale != 1.0:
        config.topology['loads'] = [{**dict(l), 'value': float(l['value']) * load_scale}
                                    for l in config.topology.get('loads', [])]

    # The state carries only topology, material, clamps and loads -- all built at the uniform
    # r_init and independent of the design -- so it is built ONCE and shared by every sample. The
    # randomized design enters through Stage 0 (`apply_closed_les_mapping`), which rebuilds the
    # geometry from `z`/`bnd_logits` on every forward pass.
    initial_state, _ = build_closed_initial_state(config)
    config.topology['init_noise'] = float(noise)
    config.topology['init_seed'] = int(seed0)
    params0, static_features = init_closed_les_params(config)
    bond_energy, _, geometry_fn_sur, _, w_lig_logit0 = build_surrogate_energy(
        config, static_features, initial_state, params0)
    geometry_from_design = build_hinge_geometry_fn(config, static_features, initial_state)
    length_scale = resolve_length_scale(config)

    ens = PathEnsemble(config_path=config_path, length_scale=length_scale,
                       n_load_steps=int(config.physics.num_load_steps))
    load_specs = config.topology.get('loads', [])
    for i in range(n_samples):
        seed = int(seed0) + i
        config.topology['init_seed'] = seed
        params, _ = init_closed_les_params(config)
        if w_lig_logit0 is not None:
            params = {**params, 'w_lig_logit': w_lig_logit0}
        try:
            result = forward_pipeline(initial_state, config.target, config.validity, config.physics,
                                      map_type=config.mapping.type, map_params=params,
                                      static_features=static_features, load_specs=load_specs,
                                      bond_energy_fn=bond_energy,
                                      hinge_geometry=(geometry_fn_sur(params)
                                                      if geometry_fn_sur else None))
            paths = extract_hinge_paths(result['solution'], result['valid_state'],
                                        geometry_from_design(params), length_scale,
                                        reference_bond_vectors=result.get('reference_bond_vectors'))
            if not np.all(np.isfinite(paths.eta)):
                raise FloatingPointError("non-finite path (the solve diverged)")
        except Exception as e:                                  # noqa: BLE001 - report, don't abort
            ens.failures.append({'seed': seed, 'error': f"{type(e).__name__}: {e}"})
            if verbose:
                print(f"  [{i + 1}/{n_samples}] seed {seed}: FAILED  {type(e).__name__}: {e}")
            continue
        ens.samples.append(EnsembleSample(seed=seed, noise=float(noise), paths=paths))
        if verbose:
            e = paths.eta
            print(f"  [{i + 1}/{n_samples}] seed {seed}: "
                  f"eta_a {e[..., 0].min():+.3f}..{e[..., 0].max():+.3f}  "
                  f"eta_s {e[..., 1].min():+.3f}..{e[..., 1].max():+.3f}  "
                  f"theta {np.degrees(e[..., 2]).max():5.1f}deg")
    return ens


def resolve_length_scale(config) -> float:
    """mm per pipeline length-unit -- a purely GEOMETRIC bridge, meaningful even on the ROM branch.

    ``hinge_model.length_scale`` when declared, else the real panel pitch implied by
    ``sheet_width_mm``, else 1.0 (dimensionless).
    """
    hm = getattr(config, 'hinge_model', None)
    ls = float(getattr(hm, 'length_scale', 0.0) or 0.0) if hm is not None else 0.0
    if ls > 0.0:
        return ls
    sheet_w = float(config.topology.get('sheet_width_mm', 0.0) or 0.0)
    if sheet_w > 0:
        return sheet_w / (int(config.topology['M']) * float(config.topology.get('spacing', 1.0)))
    return 1.0


# ── the distribution ──────────────────────────────────────────────────────────────

_QUANTILES = (1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0)
_COMPONENTS = ('eta_a', 'eta_s', 'theta')


def _marginal(x: np.ndarray, degrees: bool = False) -> dict:
    v = np.degrees(x) if degrees else x
    return {'mean': float(np.mean(v)), 'std': float(np.std(v)),
            'min': float(np.min(v)), 'max': float(np.max(v)),
            'q': {f"p{q:g}": float(np.percentile(v, q)) for q in _QUANTILES}}


def _variance_decomposition(ens: PathEnsemble) -> dict:
    """Does the random START move the distribution, or is the spread within-sheet anyway?

    For each component, split the variance of the per-hinge endpoints into a BETWEEN-design part
    (variance of each design's mean) and a WITHIN-design part (mean of each design's variance). The
    reported ratio is the intraclass correlation ``between / (between + within)``:

      ~0  the random start barely matters -- one sheet already samples the whole distribution, so
          a handful of designs is enough to characterize it;
      ~1  each start lands in its own cluster -- the oracle must be aimed at the FAMILY, and a
          single deployment would badly under-report the spread.
    """
    out = {}
    for k, comp in enumerate(_COMPONENTS):
        per = np.stack([s.paths.eta[-1, :, k] for s in ens.samples])       # (n_samples, n_hinges)
        between = float(np.var(per.mean(axis=1)))
        within = float(np.mean(per.var(axis=1)))
        tot = between + within
        out[comp] = {'between_design_var': between, 'within_design_var': within,
                     'intraclass_corr': float(between / tot) if tot > 1e-30 else 0.0}
    return out


def ensemble_statistics(ens: PathEnsemble, domain: dict = DOMAIN) -> dict:
    """Pooled marginals, correlations, variance decomposition and box occupancy for the ensemble."""
    if not ens.samples:
        raise ValueError("empty ensemble -- every design failed; check .failures")
    ends, pts, alphas = ens.endpoints(), ens.all_points(), ens.alphas()

    corr = np.corrcoef(ends.T)
    strn = np.concatenate([[straightness(s.paths.eta[:, h, :]) for h in range(s.paths.n_hinges)]
                           for s in ens.samples])
    per_diag = [path_diagnostics(s.paths, domain) for s in ens.samples]

    # rotation-dominated = in-plane motion small next to rotation, in the ray's own units. This is
    # what `sample_jobs`' `spine_frac` (eta_a = eta_s = 0) is meant to cover.
    inplane = np.linalg.norm(ends[:, :2], axis=1)
    rot = np.abs(ends[:, 2])
    rot_dominated = float(np.mean(inplane <= 0.1 * np.maximum(rot, 1e-9)))

    in_box = ((pts[:, 0] >= 0.0) & (pts[:, 0] <= domain['eta_a_max']) &
              (np.abs(pts[:, 1]) <= domain['eta_s_max']) &
              (np.abs(pts[:, 2]) <= domain['theta_max']))
    return {
        'n_designs': ens.n_samples,
        'n_failed': len(ens.failures),
        'n_hinges': ens.n_hinges,
        'n_load_steps': ens.n_load_steps,
        'n_endpoints': int(ends.shape[0]),
        'n_path_points': int(pts.shape[0]),
        'endpoint_marginals': {'eta_a': _marginal(ends[:, 0]), 'eta_s': _marginal(ends[:, 1]),
                               'theta_deg': _marginal(ends[:, 2], degrees=True)},
        'path_marginals': {'eta_a': _marginal(pts[:, 0]), 'eta_s': _marginal(pts[:, 1]),
                           'theta_deg': _marginal(pts[:, 2], degrees=True)},
        'alpha_deg': _marginal(alphas, degrees=True),
        'endpoint_correlation': {'order': list(_COMPONENTS),
                                 'matrix': [[float(v) for v in row] for row in corr]},
        'variance_decomposition': _variance_decomposition(ens),
        'straightness': _marginal(strn),
        'rotation_dominated_frac': rot_dominated,
        'monotonic_frac': {c: float(np.mean([d['monotonic_frac'][c] for d in per_diag]))
                           for c in _COMPONENTS},
        'compression': {
            'design_frac_with_any': float(np.mean([d['compression']['n_hinges_compressive'] > 0
                                                   for d in per_diag])),
            'hinge_frac': float(np.mean([d['compression']['n_hinges_compressive'] / d['n_hinges']
                                         for d in per_diag])),
            'min_eta_a': float(pts[:, 0].min()),
            'sample_frac': float(np.mean(pts[:, 0] < -1e-6)),
        },
        'in_domain_frac': float(np.mean(in_box)),
        'box_occupancy': {
            'eta_a': float(pts[:, 0].max() / domain['eta_a_max']) if domain['eta_a_max'] else 0.0,
            'eta_s': float(np.abs(pts[:, 1]).max() / domain['eta_s_max']) if domain['eta_s_max'] else 0.0,
            'theta': float(np.abs(pts[:, 2]).max() / domain['theta_max']) if domain['theta_max'] else 0.0,
        },
        'domain': dict(domain),
    }


# ── writing the distribution down for the oracle ──────────────────────────────────

def sampling_spec(ens: PathEnsemble, *, q_lo: float = 1.0, q_hi: float = 99.0, pad: float = 0.15,
                  domain: dict = DOMAIN) -> dict:
    """Turn the measured distribution into ``sample_jobs`` keyword ranges.

    The returned ``sample_jobs_kwargs`` can be splatted straight into
    ``nff.rve.dataset.sample_jobs`` -- same names, same ``(lo, hi)`` tuple convention -- so the
    oracle can be aimed at the observed manifold without any translation step.

    Ranges are taken from ALL path points, not just endpoints, because the surrogate is evaluated at
    every increment along a deployment; then padded by ``pad`` (fraction of the range) so the
    trained region has margin at its edges rather than the pipeline running along an extrapolation
    boundary. ``q_lo``/``q_hi`` trim the tails so one outlier design cannot re-inflate the box back
    to the guessed one.

    ``w_lig`` is deliberately NOT emitted: the ROM has no learnable ligament width, so every hinge
    in the ensemble carries the same manufactured value and the measurement would be degenerate.
    Keep the manufacturing range there.
    """
    pts, ends, alphas = ens.all_points(), ens.endpoints(), ens.alphas()

    def rng(v, lo_floor=None):
        lo, hi = float(np.percentile(v, q_lo)), float(np.percentile(v, q_hi))
        m = pad * max(hi - lo, 1e-9)
        lo, hi = lo - m, hi + m
        return [max(lo, lo_floor) if lo_floor is not None else lo, hi]

    # Every path starts at the origin, so eta_a's low quantile sits at ~0 and blind padding would
    # push it negative on purely tensile data -- inventing a compressive requirement that was never
    # observed. Only let the lower bound cross zero when the measurement actually did.
    eta_a = rng(pts[:, 0], lo_floor=None if pts[:, 0].min() < 0.0 else 0.0)
    eta_s = rng(pts[:, 1])
    theta1 = rng(np.degrees(ends[:, 2]), lo_floor=0.0)      # a ray is named by its ENDPOINT rotation
    alpha_deg = rng(np.degrees(alphas))

    # how much smaller is the measured box than the one currently sampled?
    def vol(a, s, t):
        return max(a[1] - a[0], 0.0) * max(s[1] - s[0], 0.0) * max(t[1] - t[0], 0.0)
    cur = vol([0.0, domain['eta_a_max']], [-domain['eta_s_max'], domain['eta_s_max']],
              [0.0, float(np.degrees(domain['theta_max']))])
    new = vol(eta_a, eta_s, theta1)

    return {
        'sample_jobs_kwargs': {
            'alpha_deg': [round(v, 3) for v in alpha_deg],
            'theta1_deg': [round(v, 3) for v in theta1],
            'eta_a': [round(v, 4) for v in eta_a],
            'eta_s': [round(v, 4) for v in eta_s],
            'spine_frac': round(float(np.clip(
                np.mean(np.linalg.norm(ends[:, :2], axis=1) <=
                        0.1 * np.maximum(np.abs(ends[:, 2]), 1e-9)), 0.0, 1.0)), 3),
        },
        'w_lig': 'NOT MEASURED -- the ROM has no learnable ligament width, so the ensemble is '
                 'degenerate in w_lig. Keep the manufacturing range, e.g. (1.0, 10.0).',
        'requires_negative_eta_a': bool(eta_a[0] < 0.0),
        'box_volume_vs_current': float(new / cur) if cur > 0 else None,
        'derivation': {
            'source': 'all path points (endpoints for theta1/spine_frac)',
            'quantiles': [q_lo, q_hi], 'pad_frac': pad,
            'n_designs': ens.n_samples, 'n_path_points': int(pts.shape[0]),
        },
        'PROVISIONAL': 'Measured under an UNCALIBRATED ROM (linear springs, k_* not yet fitted to '
                       'coupon data) against a target shape that is not yet the one we mean to '
                       'match. Re-measure before spending oracle budget on it.',
    }


def save_ensemble(ens: PathEnsemble, out_dir: str, meta: Optional[dict] = None) -> dict:
    """Write per-design paths plus the pooled distribution + oracle spec.

    Layout::
        <out_dir>/designs/seed<NNN>.{npz,json}   one per design (plain ``save_paths`` output)
        <out_dir>/distribution.json              pooled statistics + sample_jobs kwargs
    """
    os.makedirs(os.path.join(out_dir, "designs"), exist_ok=True)
    for s in ens.samples:
        save_paths(s.paths, os.path.join(out_dir, "designs", f"seed{s.seed:03d}"),
                   meta={'seed': s.seed, 'init_noise': s.noise})
    summary = {'config': ens.config_path, 'length_scale_mm_per_unit': ens.length_scale,
               'seeds': [s.seed for s in ens.samples],
               'init_noise': ens.samples[0].noise if ens.samples else None,
               'failures': ens.failures,
               'statistics': ensemble_statistics(ens),
               'sampling_spec': sampling_spec(ens),
               **(meta or {})}
    with open(os.path.join(out_dir, "distribution.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def load_ensemble(out_dir: str) -> PathEnsemble:
    """Inverse of ``save_ensemble`` (paths only; statistics are recomputed on demand)."""
    from nff.closed.hinge_paths import load_paths
    with open(os.path.join(out_dir, "distribution.json")) as f:
        summary = json.load(f)
    ens = PathEnsemble(config_path=summary.get('config', ''),
                       length_scale=float(summary.get('length_scale_mm_per_unit', 1.0)))
    noise = summary.get('init_noise') or 0.0
    for seed in summary.get('seeds', []):
        paths = load_paths(os.path.join(out_dir, "designs", f"seed{int(seed):03d}"))
        ens.samples.append(EnsembleSample(seed=int(seed), noise=float(noise), paths=paths))
    ens.n_load_steps = ens.samples[0].paths.n_steps if ens.samples else 0
    ens.failures = summary.get('failures', [])
    return ens
