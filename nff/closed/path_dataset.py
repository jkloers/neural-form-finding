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
        verts:     (n_vertices, 2) DEPLOYED global vertex positions, for the contact sheet
        flat:      (n_vertices, 2) the flat design it deployed from
        z, bnd:    the design vector itself
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
        return np.concatenate([np.transpose(e.eta, (1, 0, 2)) for e in self.examples], axis=0)

    def endpoints(self) -> np.ndarray:
        """(n_examples * n_hinges, 3) -- where each path ended."""
        return np.concatenate([e.eta[-1] for e in self.examples], axis=0)

    def alphas(self) -> np.ndarray:
        return np.concatenate([e.alpha for e in self.examples], axis=0)

    def w_ligs(self) -> np.ndarray:
        return np.concatenate([e.w_lig for e in self.examples], axis=0)

    def design_index(self) -> np.ndarray:
        """(n_paths,) which example each path came from -- so held-out splits are BY DESIGN.

        Splitting by sample would leak: the 12 hinges of one sheet are far from independent, and a
        surrogate that memorised one of them would score well on its siblings.
        """
        return np.concatenate([np.full(len(e.alpha), i) for i, e in enumerate(self.examples)])


def harvest_paths(config, *, n_examples: int = 64, noise: float = 0.5, seed0: int = 0,
                  n_load_steps: Optional[int] = None, config_path: str = "",
                  verbose: bool = True, progress_every: int = 1) -> PathDataset:
    """Deploy ``n_examples`` random tessellations and record every hinge's full path.

    Args:
        config: parsed ``ExperimentConfig`` -- deep-copied, the caller's object is untouched.
            Its boundary conditions are used AS GIVEN; the standard harvest config holds ONE tile
            and pulls ONE tile on the opposite side.
        n_examples: number of random designs.
        noise: ``init_noise``, the Gaussian sigma on the design logits.
        seed0: first ``init_seed``; example i uses ``seed0 + i``.
        n_load_steps: path resolution (overrides ``physics.num_load_steps``). More steps = a finer
            polyline for the oracle to replay later, at linear cost in the deploy.
        progress_every: print every k examples (0 = silent).

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

    # The state carries only topology, material, clamps and loads -- all built at the uniform
    # r_init and design-independent -- so it is built ONCE. The random design enters through
    # Stage 0, which rebuilds the geometry from z / bnd_logits on every forward pass.
    initial_state, tessellation = build_closed_initial_state(config)
    config.topology['init_noise'] = float(noise)
    config.topology['init_seed'] = int(seed0)
    params0, static_features = init_closed_les_params(config)
    bond_energy, _, geometry_fn_sur, _, w_lig_logit0 = build_surrogate_energy(
        config, static_features, initial_state, params0)
    geometry_from_design = build_hinge_geometry_fn(config, static_features, initial_state)
    length_scale = resolve_length_scale(config)
    load_specs = config.topology.get('loads', [])

    ds = PathDataset(meta={
        'config_path': config_path,
        'n_requested': int(n_examples),
        'noise': float(noise),
        'seed0': int(seed0),
        'n_load_steps': int(config.physics.num_load_steps),
        'length_scale_mm_per_unit': float(length_scale),
        # w_lig is the normaliser for eta = a / w_lig, so it is recorded explicitly rather than
        # left implicit -- a dataset whose eta scale is unknown cannot be handed to the oracle.
        'w_lig_mm': float(getattr(getattr(config, 'hinge_model', None), 'w_lig_mm', 5.0)),
        'clamped_faces': list(config.topology.get('bc_clamped', [])),
        'clamped_dofs': list(config.topology.get('clamped_dofs', []) or []),
        'loads': [dict(l) for l in load_specs],
        # Face -> global vertex indices. Topology is design-independent, so one copy describes
        # every example, and the contact sheet can draw real panels instead of guessing at them.
        'face_vertex_ids': [[int(v) for v in f.vertex_indices] for f in tessellation.faces],
    })

    for i in range(n_examples):
        seed = int(seed0) + i
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
            vs = res['valid_state']
            disp = res['solution'].fields[-1]
            ms = res['mapped_state']
            verts = _global_verts(tessellation, np.asarray(deformed_vertices(vs, disp)))
            flat = _global_verts(tessellation, np.asarray(reconstruct_vertices(
                ms.face_centroids, ms.centroid_node_vectors)))
        except Exception as e:                                  # noqa: BLE001 - report, don't abort
            ds.failures.append({'seed': seed, 'error': f"{type(e).__name__}: {e}"})
            if verbose:
                print(f"  [{i + 1}/{n_examples}] seed {seed}: FAILED  {type(e).__name__}: {e}")
            continue
        ds.meta.setdefault('n_hinges', int(np.asarray(paths.alpha).shape[0]))
        ds.examples.append(PathExample(
            seed=seed, eta=np.asarray(paths.eta), alpha=np.asarray(paths.alpha),
            w_lig=np.asarray(paths.w_lig), load_frac=np.asarray(paths.load_fraction),
            verts=np.asarray(verts), flat=np.asarray(flat),
            z=np.asarray(params['z']), bnd=np.asarray(params['bnd_logits'])))
        if verbose and progress_every and ((i + 1) % progress_every == 0):
            e = paths.eta
            print(f"  [{i + 1}/{n_examples}] seed {seed}: "
                  f"eta_a {e[..., 0].min():+.3f}..{e[..., 0].max():+.3f}  "
                  f"eta_s {e[..., 1].min():+.3f}..{e[..., 1].max():+.3f}  "
                  f"theta max {np.degrees(np.abs(e[..., 2])).max():5.1f}deg")
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
    )
    meta = dict(ds.meta)
    meta.update({'n_examples': ds.n_examples, 'n_failures': len(ds.failures),
                 'failures': ds.failures[:50],
                 'n_hinges': int(ds.examples[0].alpha.shape[0]),
                 'n_path_points': int(ds.examples[0].eta.shape[0])})
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return npz


def load_dataset(out_dir: str) -> PathDataset:
    d = np.load(os.path.join(out_dir, "paths.npz"))
    with open(os.path.join(out_dir, "manifest.json")) as f:
        meta = json.load(f)
    ds = PathDataset(meta=meta)
    for i in range(d['eta'].shape[0]):
        ds.examples.append(PathExample(
            seed=int(d['seed'][i]), eta=d['eta'][i], alpha=d['alpha'][i], w_lig=d['w_lig'][i],
            load_frac=d['load_frac'][i], verts=d['verts'][i], flat=d['flat'][i],
            z=d['z'][i], bnd=d['bnd'][i]))
    return ds
