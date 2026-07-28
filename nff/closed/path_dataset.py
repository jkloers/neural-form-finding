"""Harvest a dataset of REAL hinge displacement paths from random tessellations.

The surrogate's training set used to be drawn from a hand-guessed box in ``(eta_a, eta_s, theta)``,
and when we finally measured what a deployment actually rides we found the two barely overlap --
most conspicuously, the box forbids ``a < 0`` while over half of all hinges go compressive. This
module exists so the next campaign is aimed at measured paths instead of guessed ones.

WHAT IS SAMPLED. One *example* = one random flat tessellation, deployed once. The randomness is
exactly the design vector the optimizer itself moves:

    z           per-cut void aspect ratios, r = sigmoid(z)
    bnd_logits  boundary points (ordered softmax -> the sliders)

Both stay valid by construction -- ``r`` cannot leave (0, 1) and the ordered boundary stays convex
-- so every sampled design is a legal sheet, with no rejection step.

WHAT IS STORED. The whole polyline, not just where it ended. The oracle is elastoplastic, so ``W``
is PATH-DEPENDENT: a state function ``W(a, s, theta)`` is only a legitimate surrogate if the paths
it was trained along resemble the paths a deployment rides. Endpoints alone cannot express that, so
we keep every load step. ``nff/rve/ccx_solver.py::_write_deck`` already emits one ``*STEP`` per
kinematic state and accepts an arbitrary list of them, so a stored polyline can later be replayed
into the oracle verbatim.

The deployed geometry of every example is stored too -- cheap, and it makes the contact-sheet
figure possible without re-solving.
"""

import copy
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from nff.closed.hinge_paths import extract_hinge_paths, build_hinge_geometry_fn
from nff.closed.hinge_path_ensemble import resolve_length_scale


@dataclass
class PathExample:
    """One random tessellation, deployed.

    Attrs:
        seed:      the design seed (reproduces z / bnd_logits exactly)
        eta:       (n_steps+1, n_hinges, 3) path in (eta_a, eta_s, theta[rad]); row 0 is the origin
        alpha:     (n_hinges,) hinge opening angle [rad], the RVE's geometric descriptor
        w_lig:     (n_hinges,) ligament width [mm] used to normalise eta
        load_frac: (n_steps+1,) fraction of the full load at each row
        verts:     (n_vertices, 2) DEPLOYED global vertex positions
        flat:      (n_vertices, 2) the flat design it deployed from
        z, bnd:    the design vector itself
        clamped_face / loaded_face: WHICH tile was held and which was pulled
        load_value: the force applied [N]
    """
    seed: int
    eta: np.ndarray
    alpha: np.ndarray
    w_lig: np.ndarray
    load_frac: np.ndarray
    verts: np.ndarray
    flat: np.ndarray
    z: np.ndarray
    bnd: np.ndarray
    clamped_face: int = -1
    loaded_face: int = -1
    load_value: float = 0.0
    # (n_hinges,) bool -- False marks a hinge whose path was discarded (see filter_dataset).
    # A mask rather than deletion so every example keeps the same shape and the stacked arrays
    # stay rectangular; the accessors below apply it.
    hinge_mask: Optional[np.ndarray] = None

    def mask(self) -> np.ndarray:
        return (np.ones(len(self.alpha), bool) if self.hinge_mask is None
                else np.asarray(self.hinge_mask, bool))


@dataclass
class PathDataset:
    examples: List[PathExample] = field(default_factory=list)
    failures: List[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def n_examples(self) -> int:
        return len(self.examples)

    def polylines(self) -> np.ndarray:
        """(n_examples * n_hinges, n_steps+1, 3) -- every path, as a path."""
        return np.concatenate([np.transpose(e.eta[:, e.mask()], (1, 0, 2))
                               for e in self.examples], axis=0)

    def endpoints(self) -> np.ndarray:
        """(n_examples * n_hinges, 3) -- where each path ended."""
        return np.concatenate([e.eta[-1][e.mask()] for e in self.examples], axis=0)

    def alphas(self) -> np.ndarray:
        return np.concatenate([e.alpha[e.mask()] for e in self.examples], axis=0)

    def w_ligs(self) -> np.ndarray:
        return np.concatenate([e.w_lig[e.mask()] for e in self.examples], axis=0)

    def grips(self) -> np.ndarray:
        """(n_examples, 2) the (clamped, loaded) tile pair of each example."""
        return np.array([[e.clamped_face, e.loaded_face] for e in self.examples])

    def loads(self) -> np.ndarray:
        """(n_examples,) the force each example was pulled with [N]."""
        return np.array([e.load_value for e in self.examples])

    def per_path(self, values: np.ndarray) -> np.ndarray:
        """Broadcast one value per EXAMPLE out to one value per PATH."""
        return np.repeat(np.asarray(values), [int(e.mask().sum()) for e in self.examples])

    def design_index(self) -> np.ndarray:
        """(n_paths,) which example each path came from -- so held-out splits are BY DESIGN.

        Splitting by sample would leak: the 12 hinges of one sheet are far from independent, and a
        surrogate that memorised one of them would score well on its siblings.
        """
        return np.concatenate([np.full(int(e.mask().sum()), i)
                               for i, e in enumerate(self.examples)])


def _overlap_fraction(verts: np.ndarray, face_ids) -> float:
    """Fraction of total face area that is double-covered -- i.e. tiles passing through each other."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    polys = [Polygon(verts[f]) for f in face_ids]
    polys = [p if p.is_valid else p.buffer(0) for p in polys]
    tot = sum(p.area for p in polys)
    return float((tot - unary_union(polys).area) / max(tot, 1e-12))


def grip_pairs(config) -> List[tuple]:
    """Every (clamped tile, pulled tile) pair with the two on OPPOSITE sides of the sheet.

    Faces are indexed ``column * N + row``, so the bottom row is ``i*N`` and the top row is
    ``i*N + (N-1)``. For the 3x3 standard that is 3 x 3 = 9 pairs.
    """
    M, N = int(config.topology['M']), int(config.topology['N'])
    bottom = [i * N for i in range(M)]
    top = [i * N + (N - 1) for i in range(M)]
    return [(int(c), int(l)) for c in bottom for l in top]


def calibrate_grip_loads(config, grips, *, angle_max_deg: float = 50.0, eta_max: float = 1.0,
                         n_load_steps: int = 8, max_iter: int = 18, verbose: bool = True) -> dict:
    """Force at which each (clamp, pull) grip first hits a physical limit.

    Grips are NOT equally compliant -- on the 3x3 standard, 0->5 reaches 48 deg at 19 N while 3->8
    manages 9 deg at 15 N, a spread of about 5x. A single absolute force therefore deploys some
    grips barely at all and tears others apart, so each grip is normalised by its own ceiling and
    "force" then means the same thing everywhere: a fraction of full deployment.

    TWO limits, and the binding one wins:

        rotation  theta_max reaches ``angle_max_deg``      -- the fold ceiling
        strain    max(|eta_a|, |eta_s|) reaches ``eta_max`` -- the ligament's neck-strain limit

    Calibrating on rotation ALONE is wrong and was measured to be actively harmful: for grips whose
    rotation saturates (the short axis locks at 53.1 deg), the bisection keeps raising the force
    trying to reach the angle, and at 11 kN we saw eta_a = 14.6 -- a ligament stretched fifteen
    times its own width while the angle sat below target. Rotation saturates; strain does not.

    The ceiling is measured on the UNIFORM design, so a random design may still overshoot it. It is
    a normaliser, not a clamp; per-example angles and forces are recorded so overshoot can be
    filtered later.
    """
    from nff.closed.setup import build_closed_initial_state, init_closed_les_params
    from nff.stages.pipeline import forward_pipeline

    cfg = copy.deepcopy(config)
    cfg.physics.num_load_steps = int(n_load_steps)
    cfg.topology['init_noise'] = 0.0
    cfg.topology['init_seed'] = 0
    dof = int(cfg.topology['loads'][0]['dof'])
    length_scale = resolve_length_scale(cfg)
    ref_state, _ = build_closed_initial_state(cfg)
    pr0, sf0 = init_closed_les_params(cfg)
    geom_fn = build_hinge_geometry_fn(cfg, sf0, ref_state)      # topology-only, hoisted

    def utilisation(clamp_f, load_f, force):
        """max(theta/theta_ceiling, eta/eta_ceiling) -- 1.0 means "at the first physical limit"."""
        cfg.topology['bc_clamped'] = [int(clamp_f)]
        cfg.topology['loads'] = [{'face': int(load_f), 'dof': dof, 'value': float(force)}]
        st, _ = build_closed_initial_state(cfg)
        pr, sf = init_closed_les_params(cfg)
        res = forward_pipeline(st, cfg.target, cfg.validity, cfg.physics, map_type=cfg.mapping.type,
                               map_params=pr, static_features=sf,
                               load_specs=cfg.topology['loads'])
        paths = extract_hinge_paths(res['solution'], res['valid_state'], geom_fn(pr), length_scale,
                                    reference_bond_vectors=res.get('reference_bond_vectors'))
        e = np.asarray(paths.eta)
        th = float(np.degrees(np.abs(e[..., 2])).max())
        et = float(np.maximum(np.abs(e[..., 0]), np.abs(e[..., 1])).max())
        return max(th / angle_max_deg, et / eta_max), th, et

    out = {}
    for clamp_f, load_f in grips:
        lo, hi = 1.0, 100.0
        u, th, et = utilisation(clamp_f, load_f, hi)
        while u < 1.0 and hi < 1e7:
            hi *= 5.0
            u, th, et = utilisation(clamp_f, load_f, hi)
        mid = hi
        for _ in range(max_iter):
            mid = float(np.sqrt(lo * hi))               # geometric: the force spans decades
            u, th, et = utilisation(clamp_f, load_f, mid)
            if abs(u - 1.0) < 0.02:
                break
            lo, hi = (mid, hi) if u < 1.0 else (lo, mid)
        out[(int(clamp_f), int(load_f))] = float(mid)
        if verbose:
            binds = "rotation" if th / angle_max_deg >= et / eta_max else "strain"
            print(f"    grip {clamp_f}->{load_f}: {mid:9.1f} N   theta_max {th:5.1f} deg  "
                  f"eta_max {et:5.2f}   ({binds} binds)")
    return out


def harvest_paths(config, *, n_examples: int = 64, noise: float = 0.5, seed0: int = 0,
                  n_load_steps: Optional[int] = None, config_path: str = "",
                  grips=None, load_range=None, grip_loads=None, verbose: bool = True,
                  progress_every: int = 1, clear_cache_every: int = 20,
                  max_overlap: float = 5e-4, max_eta: float = 20.0,
                  theta_bounds_deg=(-2.5, 130.0)) -> PathDataset:
    """Deploy ``n_examples`` random tessellations and record every hinge's full path.

    Three things vary independently, so the ensemble spans design, drive strength and load path:

        DESIGN   z (void aspect ratios) and bnd_logits (boundary points), via init_noise
        FORCE    log-uniform over ``load_range`` -- log, not linear, because the deployment angle
                 is roughly logarithmic in load (15 N -> 18 deg, 60 -> 23, 240 -> 26, 3840 -> 39),
                 so a linear draw would spend most of its samples in the saturated tail
        GRIP     which tile is held and which is pulled, from ``grips``

    Args:
        config: parsed ``ExperimentConfig`` -- deep-copied, the caller's object is untouched.
        n_examples: number of random designs.
        noise: ``init_noise``, the Gaussian sigma on the design logits.
        seed0: first ``init_seed``; example i uses ``seed0 + i``.
        n_load_steps: path resolution (overrides ``physics.num_load_steps``).
        grips: list of ``(clamped_face, loaded_face)``. ``None`` -> the config's own single pair
            (no grip variation); ``'all'`` -> every opposite-side pair from :func:`grip_pairs`.
        grip_loads: ``{(clamp, pull): full_deployment_force}`` from :func:`calibrate_grip_loads`.
            When given, ``load_range`` is read as a FRACTION of each grip's own ceiling, so
            "force" means the same fraction of full deployment for every grip.
        load_range: ``(lo, hi)``, log-uniform. Newtons if ``grip_loads`` is None, otherwise a
            fraction of the grip ceiling. ``None`` -> the config's value / (0.02, 1.0).
        progress_every: print every k examples (0 = silent).
        max_overlap: reject any deployment whose tiles overlap by more than this fraction of total
            face area. Contact is a BARRIER, not a constraint, and it only acts on bonded pairs, so
            this gate is what actually keeps unphysical sheets out of the dataset.
        max_eta: reject a path whose ``|eta|`` exceeds this. ``isfinite`` alone is not enough -- a
            diverged solve produced eta_a = -1.1e42, which is finite and passed straight through.
        theta_bounds_deg: admissible relative hinge rotation. The lower bound is the contact
            barrier's tolerance band (min_angle = -2 deg) plus a little; anything below it escaped
            the barrier. The upper bound is above the x-axis kinematic lock at 126.9 deg, so real
            deployment is never clipped -- it only catches spinning tiles (measured: two examples
            reached -299 and -272 deg, both at trivial 2-7 N loads, i.e. failed solves).
        clear_cache_every: call ``jax.clear_caches()` every k examples (0 = never). REQUIRED for
            long harvests: ``forward_pipeline`` builds a fresh jitted solver closure on every call,
            so JAX caches a new compiled executable each time and never evicts it. Measured
            unmitigated growth is ~55 MB PER EXAMPLE, reaching 9.1 GB by 160 examples -- a 2700
            example run was killed by the OOM killer (exit 137) before this was added.

    Returns:
        ``PathDataset``. A design whose solve diverges is recorded in ``.failures`` and skipped
        rather than aborting the harvest.
    """
    from nff.closed.setup import (build_closed_initial_state, init_closed_les_params,
                                  build_surrogate_energy)
    from nff.stages.pipeline import forward_pipeline
    from nff.stages.geometry import deformed_vertices, reconstruct_vertices
    from nff.closed.deploy import _global_verts

    config = copy.deepcopy(config)
    if n_load_steps is not None:
        config.physics.num_load_steps = int(n_load_steps)
    base_loads = [dict(l) for l in config.topology.get('loads', [])]
    base_dof = int(base_loads[0]['dof']) if base_loads else 1
    base_value = float(base_loads[0]['value']) if base_loads else 60.0
    if grips == 'all':
        grips = grip_pairs(config)
    elif grips is None:
        grips = [(int(config.topology['bc_clamped'][0]), int(base_loads[0]['face']))]
    grips = [(int(c), int(l)) for c, l in grips]

    # Reference state + hoisted static work. Both depend only on TOPOLOGY -- the descriptor and the
    # bond-order permutation cost ~1.5 s and are identical for every grip and every force, so they
    # are built once even though the state itself is rebuilt per example (clamps and loads are
    # baked into the state at build time, so a new grip or force means a new state).
    ref_state, tessellation = build_closed_initial_state(config)
    config.topology['init_noise'] = float(noise)
    config.topology['init_seed'] = int(seed0)
    params0, static_features = init_closed_les_params(config)
    bond_energy, _, geometry_fn_sur, _, w_lig_logit0 = build_surrogate_energy(
        config, static_features, ref_state, params0)
    geometry_from_design = build_hinge_geometry_fn(config, static_features, ref_state)
    length_scale = resolve_length_scale(config)

    rng = np.random.default_rng(seed0)
    # `load_range` is a FRACTION of each grip's own full-deployment force when grip_loads is given,
    # and an absolute newton range otherwise.
    if grip_loads is not None:
        flo, fhi = (float(load_range[0]), float(load_range[1])) if load_range else (0.02, 1.0)
        frac = np.exp(rng.uniform(np.log(flo), np.log(fhi), size=n_examples))
        lo, hi = flo, fhi
    else:
        lo, hi = ((float(load_range[0]), float(load_range[1])) if load_range
                  else (base_value, base_value))
        frac = (np.exp(rng.uniform(np.log(lo), np.log(hi), size=n_examples)) if hi > lo
                else np.full(n_examples, lo))
    # Assign grips round-robin then SORT, so all examples sharing a grip run consecutively.
    # `load_specs` is a static Python object to the jitted pipeline, so a change of loaded face
    # retraces; grouping keeps that to one trace per grip instead of one per example.
    gidx = np.sort(np.arange(n_examples) % len(grips))

    face_ids = [np.asarray(f.vertex_indices, int) for f in tessellation.faces]
    ds = PathDataset(meta={
        'config_path': config_path,
        'max_overlap': float(max_overlap),
        'max_eta': float(max_eta),
        'theta_bounds_deg': [float(theta_bounds_deg[0]), float(theta_bounds_deg[1])],
        'use_contact': bool(getattr(config.physics, 'use_contact', False)),
        'n_requested': int(n_examples),
        'noise': float(noise),
        'seed0': int(seed0),
        'n_load_steps': int(config.physics.num_load_steps),
        'length_scale_mm_per_unit': float(length_scale),
        # w_lig is the normaliser for eta = a / w_lig, so it is recorded explicitly rather than
        # left implicit -- a dataset whose eta scale is unknown cannot be handed to the oracle.
        'w_lig_mm': float(getattr(getattr(config, 'hinge_model', None), 'w_lig_mm', 5.0)),
        'clamped_dofs': list(config.topology.get('clamped_dofs', []) or []),
        'grips': [list(g) for g in grips],
        'load_range': [float(lo), float(hi)],
        'load_range_is_fraction_of_grip_ceiling': grip_loads is not None,
        'grip_ceiling_N': ({f"{c}->{l}": v for (c, l), v in grip_loads.items()}
                           if grip_loads is not None else None),
        'load_dof': base_dof,
        'face_vertex_ids': [[int(v) for v in f.vertex_indices] for f in tessellation.faces],
    })

    import jax
    for i in range(n_examples):
        if clear_cache_every and i and (i % clear_cache_every == 0):
            jax.clear_caches()          # see clear_cache_every: without this the run OOMs
        seed = int(seed0) + i
        clamp_f, load_f = grips[int(gidx[i])]
        force = (float(frac[i]) * grip_loads[(clamp_f, load_f)] if grip_loads is not None
                 else float(frac[i]))
        # clamps and loads are baked into the state at BUILD time -- set them before building,
        # never after (mutating load_specs afterwards silently does nothing)
        config.topology['bc_clamped'] = [clamp_f]
        config.topology['loads'] = [{'face': load_f, 'dof': base_dof, 'value': force}]
        load_specs = config.topology['loads']
        initial_state, _ = build_closed_initial_state(config)
        config.topology['init_seed'] = seed
        params, _ = init_closed_les_params(config)
        if w_lig_logit0 is not None:
            params = {**params, 'w_lig_logit': w_lig_logit0}
        try:
            res = forward_pipeline(initial_state, config.target, config.validity, config.physics,
                                   map_type=config.mapping.type, map_params=params,
                                   static_features=static_features, load_specs=load_specs,
                                   bond_energy_fn=bond_energy,
                                   hinge_geometry=(geometry_fn_sur(params)
                                                   if geometry_fn_sur else None))
            paths = extract_hinge_paths(res['solution'], res['valid_state'],
                                        geometry_from_design(params), length_scale,
                                        reference_bond_vectors=res.get('reference_bond_vectors'))
            if not np.all(np.isfinite(paths.eta)):
                raise FloatingPointError("non-finite path (the solve diverged)")
            # isfinite is NOT enough: a diverged solve produced eta_a = -1.1e42, which is finite
            # and sailed straight through into a released dataset. Bound the magnitude too.
            if np.abs(paths.eta[..., :2]).max() > max_eta:
                raise FloatingPointError(
                    f"path left the physical range (|eta| up to "
                    f"{np.abs(paths.eta[..., :2]).max():.3g} > {max_eta})")
            th_deg = np.degrees(paths.eta[..., 2])
            if th_deg.min() < theta_bounds_deg[0] or th_deg.max() > theta_bounds_deg[1]:
                raise FloatingPointError(
                    f"hinge rotation out of bounds ({th_deg.min():.1f}..{th_deg.max():.1f} deg, "
                    f"allowed {theta_bounds_deg[0]}..{theta_bounds_deg[1]})")
            vs = res['valid_state']
            disp = res['solution'].fields[-1]
            ms = res['mapped_state']
            verts = _global_verts(tessellation, np.asarray(deformed_vertices(vs, disp)))
            flat = _global_verts(tessellation, np.asarray(reconstruct_vertices(
                ms.face_centroids, ms.centroid_node_vectors)))
            # Hard validity gate. Contact should make this unreachable, but a deployment with
            # overlapping tiles is not physical and must never reach the surrogate -- and trusting
            # the sign of theta to detect it is exactly the mistake that contaminated the first
            # harvest. Measure the geometry instead.
            ov = _overlap_fraction(verts, face_ids)
            if ov > max_overlap:
                raise ValueError(f"tiles interpenetrate: {ov:.3%} of face area overlaps")
        except Exception as e:                                  # noqa: BLE001 - report, don't abort
            ds.failures.append({'seed': seed, 'grip': [clamp_f, load_f], 'force': force,
                                'error': f"{type(e).__name__}: {e}"})
            if verbose:
                print(f"  [{i + 1}/{n_examples}] seed {seed} grip {clamp_f}->{load_f} "
                      f"{force:.0f}N: FAILED  {type(e).__name__}: {e}")
            continue
        ds.meta.setdefault('n_hinges', int(np.asarray(paths.alpha).shape[0]))
        ds.examples.append(PathExample(
            seed=seed, eta=np.asarray(paths.eta), alpha=np.asarray(paths.alpha),
            w_lig=np.asarray(paths.w_lig), load_frac=np.asarray(paths.load_fraction),
            verts=np.asarray(verts), flat=np.asarray(flat),
            z=np.asarray(params['z']), bnd=np.asarray(params['bnd_logits']),
            clamped_face=clamp_f, loaded_face=load_f, load_value=force))
        if verbose and progress_every and ((i + 1) % progress_every == 0):
            e = paths.eta
            print(f"  [{i + 1}/{n_examples}] {clamp_f}->{load_f} {force:7.1f}N  "
                  f"eta_a {e[..., 0].min():+.3f}..{e[..., 0].max():+.3f}  "
                  f"eta_s {e[..., 1].min():+.3f}..{e[..., 1].max():+.3f}  "
                  f"theta {np.degrees(e[..., 2]).min():+6.1f}..{np.degrees(e[..., 2]).max():+6.1f}deg")
    return ds


# ── persistence ───────────────────────────────────────────────────────────────────

def save_dataset(ds: PathDataset, out_dir: str) -> str:
    """Write one ``paths.npz`` (stacked arrays) + ``manifest.json`` (provenance).

    Stacked rather than one file per example: the whole point is to resample across examples, and
    a single contiguous array is what both the prior and the figure want.
    """
    os.makedirs(out_dir, exist_ok=True)
    if not ds.examples:
        raise ValueError("refusing to save an empty dataset")
    npz = os.path.join(out_dir, "paths.npz")
    np.savez_compressed(
        npz,
        eta=np.stack([e.eta for e in ds.examples]),            # (E, T+1, H, 3)
        alpha=np.stack([e.alpha for e in ds.examples]),        # (E, H)
        w_lig=np.stack([e.w_lig for e in ds.examples]),
        load_frac=np.stack([e.load_frac for e in ds.examples]),
        verts=np.stack([e.verts for e in ds.examples]),
        flat=np.stack([e.flat for e in ds.examples]),
        z=np.stack([e.z for e in ds.examples]),
        bnd=np.stack([e.bnd for e in ds.examples]),
        seed=np.array([e.seed for e in ds.examples]),
        hinge_mask=np.stack([e.mask() for e in ds.examples]),
        clamped_face=np.array([e.clamped_face for e in ds.examples]),
        loaded_face=np.array([e.loaded_face for e in ds.examples]),
        load_value=np.array([e.load_value for e in ds.examples]),
    )
    meta = dict(ds.meta)
    meta.update({'n_examples': ds.n_examples, 'n_failures': len(ds.failures),
                 'failures': ds.failures[:50],
                 'n_hinges': int(ds.examples[0].alpha.shape[0]),
                 'n_path_points': int(ds.examples[0].eta.shape[0])})
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return npz


def filter_dataset(ds: PathDataset, *, max_eta: float = 20.0,
                   theta_bounds_deg=(-2.5, 130.0), max_overlap: float = 5e-4,
                   theta_noise_deg: float = 0.01, theta_keep_max_deg: float = 90.0) -> PathDataset:
    """Apply the validity gates to an ALREADY HARVESTED dataset, in place of re-deploying it.

    Every gate is computable from what is stored, so a gate added after the fact does not cost
    another harvest. Returns a new dataset; the rejected examples are recorded in ``.failures`` so
    the attrition stays visible rather than silently shrinking the count.
    """
    # NEGATIVE THETA comes in two flavours and they must be handled differently. Measured over
    # 36228 paths: 4.62% dip below zero, but the MEDIAN negative value is -0.000 deg and p1 is
    # -0.014 -- almost all of it is floating-point noise sitting on zero, not an excursion. Only
    # 0.25% of paths go below -0.01 deg. So: clamp the noise to exactly zero, and DISCARD the
    # genuine outliers (a hinge that really rotated closed is unphysical and must not be replayed
    # into the oracle). Discarding every path that is negative by any amount would throw away 1674
    # perfectly good paths to remove ~92 bad ones.
    fids = [np.asarray(f, int) for f in ds.meta.get('face_vertex_ids', [])]
    out = PathDataset(meta=dict(ds.meta))
    n_dropped_paths = 0
    out.failures.extend(ds.failures)
    for e in ds.examples:
        th = np.degrees(e.eta[..., 2])
        why = None
        if np.abs(e.eta[..., :2]).max() > max_eta:
            why = f"|eta| {np.abs(e.eta[..., :2]).max():.3g}"
        elif th.min() < theta_bounds_deg[0] or th.max() > theta_bounds_deg[1]:
            why = f"theta {th.min():.1f}..{th.max():.1f} deg"
        elif fids and _overlap_fraction(e.verts, fids) > max_overlap:
            why = f"overlap {_overlap_fraction(e.verts, fids):.3%}"
        if why:
            out.failures.append({'seed': int(e.seed), 'grip': [e.clamped_face, e.loaded_face],
                                 'force': float(e.load_value), 'error': f"filtered: {why}"})
        else:
            th_deg = np.degrees(e.eta[..., 2])                      # (T+1, H)
            # INTERSECT with any existing mask -- filters must compose. Re-running this on an
            # already-filtered dataset must not resurrect what a previous pass dropped, and it
            # would: the theta clamp below rewrites eta for every hinge, masked-out ones included,
            # so a fresh criterion computed from the clamped data would let them all back in.
            keep = e.mask() & (th_deg.min(axis=0) >= -abs(theta_noise_deg))
            keep &= th_deg.max(axis=0) <= theta_keep_max_deg        # per hinge
            n_dropped_paths += int((~keep).sum())
            if not keep.any():
                out.failures.append({'seed': int(e.seed),
                                     'grip': [e.clamped_face, e.loaded_face],
                                     'force': float(e.load_value),
                                     'error': "filtered: every hinge went negative"})
                continue
            e.eta[..., 2] = np.maximum(e.eta[..., 2], 0.0)          # clamp the noise to exact zero
            e.hinge_mask = keep
            out.examples.append(e)
    out.meta.update({'n_examples': out.n_examples, 'filtered': True,
                     'n_dropped_paths': n_dropped_paths,
                     'theta_noise_deg': float(theta_noise_deg),
                     'theta_keep_max_deg': float(theta_keep_max_deg),
                     'max_eta': float(max_eta), 'max_overlap': float(max_overlap),
                     'theta_bounds_deg': [float(theta_bounds_deg[0]), float(theta_bounds_deg[1])]})
    return out


def merge_datasets(dirs, meta_extra: Optional[dict] = None) -> PathDataset:
    """Concatenate shard directories into one dataset.

    Long harvests are run as a sequence of SUBPROCESSES, one per shard, because
    ``forward_pipeline`` builds a fresh jitted solver on every call and JAX never evicts the
    resulting executables -- ``jax.clear_caches()`` slows the leak but does not stop it (4.5 GB by
    240 examples even with clearing). A process boundary returns the memory unconditionally, and it
    has the second virtue that a killed run keeps every shard it had already written.
    """
    merged = None
    for d in dirs:
        ds = load_dataset(d)
        if merged is None:
            merged = PathDataset(meta=dict(ds.meta))
        merged.examples.extend(ds.examples)
        merged.failures.extend(ds.failures)
    if merged is None:
        raise ValueError("no shards to merge")
    merged.meta['n_examples'] = merged.n_examples
    merged.meta['n_shards'] = len(list(dirs))
    if meta_extra:
        merged.meta.update(meta_extra)
    return merged


def load_dataset(out_dir: str) -> PathDataset:
    npz = np.load(os.path.join(out_dir, "paths.npz"))
    # ⚠ Materialise each array ONCE. Indexing `npz['eta'][i]` inside the loop re-decompresses the
    # WHOLE array on every access -- for 3019 examples that is 3019 full decompressions of a 27 MB
    # array, which is how loading a merged dataset kept getting OOM-killed.
    d = {k: npz[k] for k in npz.files}
    with open(os.path.join(out_dir, "manifest.json")) as f:
        meta = json.load(f)
    ds = PathDataset(meta=meta)
    has_grip = 'clamped_face' in d                       # datasets harvested before grip variation
    for i in range(d['eta'].shape[0]):
        ds.examples.append(PathExample(
            seed=int(d['seed'][i]), eta=d['eta'][i], alpha=d['alpha'][i], w_lig=d['w_lig'][i],
            load_frac=d['load_frac'][i], verts=d['verts'][i], flat=d['flat'][i],
            z=d['z'][i], bnd=d['bnd'][i],
            hinge_mask=(d['hinge_mask'][i] if 'hinge_mask' in d else None),
            clamped_face=int(d['clamped_face'][i]) if has_grip else -1,
            loaded_face=int(d['loaded_face'][i]) if has_grip else -1,
            load_value=float(d['load_value'][i]) if has_grip else 0.0))
    return ds
