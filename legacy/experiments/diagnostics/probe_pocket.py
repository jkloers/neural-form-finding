"""How far a local load reaches on a large sheet, measured away from whatever holds it.

A concentrated load raises a pocket of deformation around itself. Every earlier sweep in this
package loaded an edge and held the opposite one, so the pocket around the load was mixed with the
pocket around the supports -- and the supports were measured to be the wider disturbance. This
driver takes the other arrangement: a large homogeneous sheet, the probe in the middle, and a single
corner tile clamped only to fix the coordinate frame.

Three things make the measurement mean something, and each was arrived at by getting it wrong first.

The probe is SELF-EQUILIBRATED. Its net force and net moment are removed before it is applied, so a
three-DOF gauge, being statically determinate, reacts with exactly zero rather than merely little.
The support therefore raises no pocket of its own.

The background has NO ACTUATOR. A single moment does deploy the sheet, since it is a one-DOF
mechanism, but it leaves a stress concentration; the probe changes the deployment slightly and that
concentration reappears in the difference. Measured, it made the pocket attributed to a central
probe peak at the actuator corner, identically for every probe direction. A moment on every tile,
alternating with the checkerboard and centred to zero net moment, removes the actuator instead of
correcting for it: uniformity 0.999 against 0.862.

The pocket is a FORCE field, not a displacement field. The tangent stiffness has a whole BAND of
soft global modes -- eigenvalues 0.054, 4.06, 49.2, 119 at 11x11 -- so the displacement response to
a local probe stays global whatever is projected out, and the remainder is just the next mode. The
stiff springs are ~4 orders stiffer than the rotational one, the soft modes store almost nothing in
them, and so the stiff-spring energy localises with no projection at all.

Two layers:

    --layer linear     Assemble the tangent stiffness once at the background deployment. The
                       response is then built modally, ``sum_i (v_i . f) / lam_i * v_i`` over the
                       stiff modes, so probe direction and magnitude are nearly free and the soft
                       band is excluded by construction rather than subtracted afterwards.
                       ``--soft-scan`` shows the answer stops moving once the band is out.
    --layer nonlinear  Full incremental Stage-2 solves, the pocket being the difference between the
                       probed and unprobed equilibria. Says where linear response stops describing
                       the sheet.

⚠ Every shape measure is r^2 weighted, so taken over the whole sheet it reports the sheet. The
reported shape is restricted to the pocket core; the whole-sheet version is kept under ``_far_``
because the contrast is the point. Read the threshold radius, never the radius of gyration.

⚠ Loading is PROPORTIONAL, not deploy-then-probe: the incremental solver ramps every load together
from flat (``pipeline.py`` starts from zero displacement). Each level's ENDPOINT is the requested
equilibrium; the path to it is not "deploy, then push".

Run:
    JAX_PLATFORMS=cpu PYTHONPATH=$(pwd) python legacy/experiments/diagnostics/probe_pocket.py \
        --config pocket_11 --layer linear
"""

import argparse
import copy
import json
import os
import time

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np

from nff.config.experiment import load_and_parse_config
from nff_paper.closed_support.one_dof import decay_length, fit_one_dof, graph_distance, relative_rotations
from nff_paper.closed_support.perturbation import perturbation_radius
from nff.analysis.force_stats import energy_split
from nff_paper.closed_support.pocket import (angular_profile, centre_face, decay_ratios, decomposition_precision,
                               energy_decomposition, load_specs, mechanism_split, pocket_shape,
                               rigid_resultants, scatter_hinge_to_tiles, scatter_load,
                               self_equilibrate, subspace_split, tile_adjacency)
from nff.analysis.load_paths import deployed_nodes
from nff_paper.closed_support.shield_kinematics import _checkerboard_signs
from legacy.experiments.diagnostics.diagnose_conditioning import _build_energy_over_free_dofs
from legacy.experiments.diagnostics.sweep_localisation import solve_at
from nff.stages.geometry import build_reference_bond_vectors

CONFIG_DIR = "data/configs/closed"


# ══════════════════════════════════════════════════════════════════════════════
# Sheet bookkeeping
# ══════════════════════════════════════════════════════════════════════════════

def resolve_config(name: str) -> str:
    """Accept either a bare config name or a path."""
    if os.path.exists(name):
        return name
    path = os.path.join(CONFIG_DIR, name if name.endswith(".yaml") else f"{name}.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no config at {name} or {path}")
    return path


def checkerboard_background(sigma: np.ndarray, magnitude: float) -> list:
    """A moment on EVERY tile, alternating with the checkerboard, carrying no net resultant.

    A single moment does deploy the whole sheet, because the sheet is a one-DOF mechanism, but it
    does so from one tile and leaves a stress concentration there. Every later difference then
    measures that concentration responding to a slightly different deployment, which swamps the
    probe: measured on the 11x11 sheet, the disturbance attributed to a central probe peaked at the
    actuator tile until six modes had been removed by hand.

    Driving every tile removes the actuator instead of correcting for it. Net force is zero because
    these are pure moments, and the net moment is removed by centring, which needs no positions and
    so is exact whatever the deployment. The sheet is then deployed while being held nowhere.
    """
    m = float(magnitude) * (np.asarray(sigma, dtype=float) - float(np.mean(sigma)))
    return [{'face': int(k), 'dof': 2, 'value': float(v)} for k, v in enumerate(m)]


class Sheet:
    """One background solve plus everything derived from it that the probes reuse."""

    def __init__(self, config, n_steps: int, background_loads=None, pair_axis: str = "x"):
        self.config = config
        self.M, self.N = int(config.topology['M']), int(config.topology['N'])
        self.background_loads = (list(background_loads) if background_loads is not None
                                 else list(config.topology['loads']))
        self.centre = centre_face(self.M, self.N)
        # Faces are numbered col * N + row, so the next column is +N and the next row is +1. Which
        # neighbour carries the other half of the probe is a lattice direction, and swapping it is
        # the test that separates a pocket aimed by the LOAD from one aimed by the TILING.
        self.pair_axis = pair_axis
        self.partner = self.centre + (self.N if pair_axis == "x" else 1)
        self.faces = [self.centre, self.partner]
        self.sigma = _checkerboard_signs(self.M, self.N).astype(float)

        res, c, params, sf, be, hg, _ = solve_at(config, None, self.background_loads, n_steps)
        self.results, self.parsed = res, c
        self.bond_energy, self.hinge_geometry = be, hg
        self.valid_state = res['valid_state']
        self.displacement = np.asarray(res['solution'].fields[-1])
        self.n_faces = self.displacement.shape[0]

        n_nodes = np.asarray(self.valid_state.centroid_node_vectors).shape[1]
        self.pairs = tile_adjacency(np.asarray(self.valid_state.bond_connectivity), n_nodes)
        # Tile centroids in the DEPLOYED state: the pocket is a shape on the deployed sheet.
        self.centroids = np.asarray(self.valid_state.face_centroids) + self.displacement[:, :2]
        # Panel CORNERS too, so a figure can draw the sheet without re-solving it.
        self.nodes = np.asarray(deployed_nodes(self.valid_state.face_centroids,
                                               self.valid_state.centroid_node_vectors,
                                               self.displacement))
        self.origin = self.centroids[self.faces].mean(axis=0)

        rbv = res.get('reference_bond_vectors')
        self.reference_bonds = np.asarray(rbv if rbv is not None
                                          else build_reference_bond_vectors(self.valid_state))
        self.stiffness = dict(k_stretch=self.valid_state.k_stretch,
                              k_shear=self.valid_state.k_shear, k_rot=self.valid_state.k_rot)
        self.linearized = bool(c.physics.linearized_strains)
        self.background_slip = self.slip_energy(self.displacement)

        self.fit = fit_one_dof(self.displacement[:, 2], self.sigma)
        rel = np.abs(relative_rotations(self.displacement[:, 2], self.pairs))
        self.theta = dict(min=float(np.degrees(rel.min())), max=float(np.degrees(rel.max())),
                          mean=float(np.degrees(rel.mean())),
                          uniformity=float(rel.min() / rel.max()))

    def slip_energy(self, displacement: np.ndarray) -> np.ndarray:
        """Per-hinge energy stored in the two STIFF springs, at a given displacement field.

        The rotational spring is left out, and that omission is the whole point. The mechanism
        stores its work there, spread over every hinge at once, so a rotational reading is global by
        construction. The axial and shear springs are ~4 orders stiffer and are what a load has to
        work against locally, so this is the field that can have a radius at all.

        ⚠ The axial and shear channels are not separately trustworthy on a closed sheet, where the
        reference bond frame is degenerate and swaps them on roughly half the hinges. Their SUM is,
        and only the sum is used.
        """
        parts = energy_split(self.valid_state, displacement, self.reference_bonds,
                             linearized=self.linearized, **self.stiffness)['per_hinge']
        return np.asarray(parts['axial']) + np.asarray(parts['shear'])

    def force_pocket(self, displacement: np.ndarray) -> np.ndarray:
        """Per-tile change in stiff-spring energy against the background, the pocket of force."""
        return scatter_hinge_to_tiles(
            np.abs(self.slip_energy(displacement) - self.background_slip),
            self.pairs, self.n_faces)

    def probe_fields(self, delta: np.ndarray) -> dict:
        """The probe's own stress field, its redistribution of the background, and their sum.

        The energy is evaluated at ``background - delta`` as well, which is not an equilibrium of
        anything. It does not need to be: the energy is a function of the displacement field, and
        reflecting the probe is only how the even and odd parts are separated.
        """
        plus = self.slip_energy(self.displacement + delta)
        minus = self.slip_energy(self.displacement - delta)
        own, redist = energy_decomposition(plus, self.background_slip, minus)

        def to_tiles(v):
            return scatter_hinge_to_tiles(np.abs(v), self.pairs, self.n_faces)

        return {"own": to_tiles(own), "redist": to_tiles(redist),
                "total": to_tiles(plus - self.background_slip),
                "precision": decomposition_precision(plus, self.background_slip, minus)}

    def summary(self) -> dict:
        return {"M": self.M, "N": self.N, "n_faces": self.n_faces, "n_hinges": int(self.pairs.shape[0]),
                "centre_face": int(self.centre), "partner_face": int(self.partner),
                "n_background_loads": len(self.background_loads),
                "psi_deg": float(np.degrees(self.fit.psi)), "one_dof_r2": float(self.fit.r2),
                "theta_deg": self.theta}


# ══════════════════════════════════════════════════════════════════════════════
# The probe
# ══════════════════════════════════════════════════════════════════════════════

def build_probe(sheet: Sheet, angle: float, magnitude: float,
                balanced: bool = True) -> np.ndarray:
    """An opposed in-plane force pair on the centre tile and its neighbour, made resultant-free.

    ``angle`` is measured from +x in the deployed frame. At 0 the pair pinches along the line
    joining the two tiles; at 90 degrees it shears across that line, which carries a net moment
    until ``self_equilibrate`` pays for it with a couple on each tile.

    With ``balanced`` false the pair becomes a single force and the resultants are left in place, so
    the gauge has to react. That is the control for the claim that a support disturbs the sheet
    further than a load does: the two differ by the reaction and by nothing else.
    """
    direction = np.array([np.cos(angle), np.sin(angle)])
    raw = np.zeros((2, 3))
    raw[0, :2] = magnitude * direction
    if not balanced:
        return raw
    raw[1, :2] = -magnitude * direction
    return self_equilibrate(raw, sheet.centroids[sheet.faces], origin=sheet.origin)


# ══════════════════════════════════════════════════════════════════════════════
# Measuring one response
# ══════════════════════════════════════════════════════════════════════════════

def measure(sheet: Sheet, delta: np.ndarray, total: np.ndarray, global_fields: np.ndarray,
            mode_field: np.ndarray, tag: str) -> dict:
    """Every pocket readout for one response, from the force field and from the displacement field.

    ``force_*`` is the primary answer and needs no projection at all. The sheet's soft modes are
    soft precisely because they store almost no energy in the stiff springs, so the stiff-spring
    energy is blind to them and localises on its own.

    The displacement readouts are kept because they say something the force field cannot, but they
    require the global content to be removed first and they do not fully succeed at it: measured on
    the 11x11 sheet the eigenvalues run 0.056, 3.37, 41.5, so there is a BAND of soft global modes
    rather than a single mechanism, and whatever is left after projecting out two of them is still
    the third. ``global_share`` reports how much had to be removed; when it is ~0.9997 the
    displacement radius that follows is measuring the leftovers, not the probe.
    """
    out = {"tag": tag}

    parts = sheet.probe_fields(delta)
    force, own, redist = parts["total"], parts["own"], parts["redist"]
    out["precision"] = parts["precision"]

    split = subspace_split(delta.reshape(-1), global_fields.reshape(global_fields.shape[0], -1))
    pocket_field = split.pocket.reshape(delta.shape)
    out["global_share"] = split.share
    out["global_amplitudes"] = [float(a) for a in split.amplitudes]
    out["mode_mechanism_share"] = mechanism_split(delta.reshape(-1), mode_field.reshape(-1)).share

    # Route 2: the checkerboard fit on the rotation channel, independent of the Hessian.
    fit = fit_one_dof(delta[:, 2], sheet.sigma)
    out["fit_psi"] = float(fit.psi)
    out["fit_swing"] = float(fit.swing)
    out["fit_r2"] = float(fit.r2)

    rot = np.abs(pocket_field[:, 2])
    disp = np.hypot(pocket_field[:, 0], pocket_field[:, 1])
    resid = np.abs(fit.residual)

    dist = graph_distance(sheet.pairs, sheet.n_faces, [sheet.centre])

    for name, field in (("own", own), ("redist", redist), ("force", force),
                        ("rot", rot), ("disp", disp), ("resid", resid)):
        if field.max() <= 0:
            continue
        radius = float("nan")
        try:
            rr = perturbation_radius(field, sheet.pairs, [sheet.centre])
            radius = float(rr.radius_interp[sheet.centre])
            out[f"{name}_radius"] = radius
            out[f"{name}_baseline"] = float(rr.baseline)
        except ValueError as exc:                      # sheet too small for an interior baseline
            out[f"{name}_radius"] = float("nan")
            out[f"{name}_radius_error"] = str(exc)

        # ⚠ The second moment is r^2 weighted, so taken over the whole sheet it is dominated by the
        # far field and reports the sheet, not the pocket. Measured: the whole-sheet axis sat near
        # 58 deg for EVERY probe direction and barely moved when the probe pair was rotated 90 deg,
        # which reads as a pocket the load cannot aim. Restricted to the core it tracks the load
        # (pair at 22.5 deg -> axis 26-47; pair at 112.5 deg -> axis 104-117). The core is primary;
        # the whole-sheet version is kept as `_far_` so the contrast stays visible.
        hops = int(np.ceil(radius)) if np.isfinite(radius) and radius >= 2 else 3
        core = np.where(dist <= hops, field, 0.0)
        for k, v in pocket_shape(core, sheet.centroids, sheet.origin).as_dict().items():
            out[f"{name}_{k}"] = v
        for k, v in pocket_shape(field, sheet.centroids, sheet.origin).as_dict().items():
            out[f"{name}_far_{k}"] = v
        out[f"{name}_core_hops"] = hops
        out[f"{name}_peak"] = float(field.max())
    for name, field in (("own", own), ("redist", redist), ("force", force)):
        out[f"{name}_decay"] = decay_ratios(field, dist).as_dict()
    out["decay"] = decay_length(force, dist)
    prof = angular_profile(force, sheet.centroids, sheet.origin, n_sectors=12)
    out["profile_angle"] = prof["angle"].tolist()
    out["profile_reach"] = np.nan_to_num(prof["reach"], nan=0.0).tolist()
    return out, pocket_field, parts


# ══════════════════════════════════════════════════════════════════════════════
# Layer 1: linear response
# ══════════════════════════════════════════════════════════════════════════════

def linear_layer(sheet: Sheet, args) -> dict:
    """Assemble the tangent stiffness once, then every probe is a back-substitution."""
    t0 = time.time()
    energy_free_fn, free_ids, F_bg = _build_energy_over_free_dofs(
        sheet.valid_state, sheet.parsed.physics, sheet.background_loads,
        bool(sheet.parsed.physics.use_contact),
        bond_energy_fn=sheet.bond_energy, hinge_geometry=sheet.hinge_geometry)
    free_ids = np.asarray(free_ids)
    final_free = sheet.displacement.reshape(-1)[free_ids]

    grad = np.asarray(jax.grad(energy_free_fn)(final_free))
    K = np.asarray(jax.hessian(energy_free_fn)(final_free))
    K = 0.5 * (K + K.T)
    evals, evecs = np.linalg.eigh(K)
    grad_norm = float(np.linalg.norm(grad))
    print(f"[linear] K {K.shape[0]}x{K.shape[0]} assembled in {time.time() - t0:.1f}s   "
          f"grad {grad_norm:.2e}   lam0 {evals[0]:.4e}   gap {evals[1] / evals[0]:.1f}", flush=True)
    # A Hessian is only a tangent stiffness AT an equilibrium. A background solve that stopped short
    # leaves a gradient, and everything below would then be a Green's function of nothing.
    if grad_norm > float(args.grad_gate):
        raise RuntimeError(
            f"background state is not stationary (|grad| = {grad_norm:.3e} > {args.grad_gate:.1e}); "
            "raise --maxiter or --steps rather than reading this spectrum")

    # The softest mode should BE the mechanism. Its rotation channel is checked against the
    # checkerboard rather than assumed, because everything downstream leans on it.
    mode_full = np.zeros(sheet.n_faces * 3)
    mode_full[free_ids] = evecs[:, 0]
    mode_field = mode_full.reshape(sheet.n_faces, 3)
    rot_mode = mode_field[:, 2]
    overlap = float(abs(rot_mode @ sheet.sigma) / (np.linalg.norm(rot_mode) * np.linalg.norm(sheet.sigma)))
    print(f"[linear] softest mode vs checkerboard: |cos| = {overlap:.4f}   "
          f"(rotation share of the mode {np.sum(rot_mode**2) / np.sum(mode_field**2):.3f})", flush=True)

    if evals[0] <= 0:
        raise RuntimeError(f"tangent stiffness is not positive definite (lam0 = {evals[0]:.3e}); "
                           "the background state is not a minimum and no Green's function exists")

    # The sheet's response to its OWN actuation. A probe partly just deploys the sheet further, and
    # that component carries the actuator's boundary layer with it; without removing this direction
    # the measured pocket sits at the actuator corner for every probe alike.
    bg_free = np.linalg.solve(K, np.asarray(F_bg))
    bg_full = np.zeros(sheet.n_faces * 3)
    bg_full[free_ids] = bg_free
    bg_field = bg_full.reshape(sheet.n_faces, 3)
    cos_bg_mode = float(abs(bg_free @ evecs[:, 0]) / np.linalg.norm(bg_free))
    print(f"[linear] background response vs softest mode: |cos| = {cos_bg_mode:.4f}", flush=True)
    global_fields = np.stack([bg_field, mode_field])

    # The response is built in the COMPLEMENT of the soft band rather than projected afterwards.
    # delta = sum_i (v_i . f) / lam_i * v_i, and the softest few terms carry a 1/lam that is orders
    # larger than the rest, so they dominate any field built from the full sum. Dropping them by
    # construction is the same projection done consistently, and `--soft-scan` shows that the answer
    # stops moving once the band is out.
    def response(f_free: np.ndarray, n_soft: int) -> np.ndarray:
        coef = evecs.T @ f_free
        return evecs[:, n_soft:] @ (coef[n_soft:] / evals[n_soft:])

    def scatter(free_vec: np.ndarray) -> np.ndarray:
        full = np.zeros(sheet.n_faces * 3)
        full[free_ids] = free_vec
        return full.reshape(sheet.n_faces, 3)

    if args.soft_scan:
        probe = build_probe(sheet, 0.0, float(args.probe))
        f_free = scatter_load(probe, sheet.faces, sheet.n_faces).reshape(-1)[free_ids]
        print("[soft-scan] how many soft modes have to leave before the pocket stops moving")
        print(f"{'n_soft':>7} {'lam_next':>10} {'radius':>8} {'gyration':>9} {'aspect':>7} "
              f"{'peak tile':>10} {'at probe':>9}")
        for k in range(0, min(8, evals.size)):
            delta = scatter(response(f_free, k))
            force = sheet.force_pocket(sheet.displacement + delta)
            try:
                rr = perturbation_radius(force, sheet.pairs, [sheet.centre])
                rad = rr.radius_interp[sheet.centre]
            except ValueError:
                rad = float("nan")
            shape = pocket_shape(force, sheet.centroids, sheet.origin)
            peak = int(np.argmax(force))
            print(f"{k:7d} {evals[k]:10.3e} {rad:8.2f} {shape.gyration:9.3f} {shape.aspect:7.2f} "
                  f"{peak:10d} {'YES' if peak in sheet.faces else 'no':>9}", flush=True)
        print(f"[soft-scan] probe tiles are {sheet.faces}; the sheet centre is {sheet.centre}")

    rows, fields = [], {}
    angles = np.linspace(0.0, np.pi, int(args.directions), endpoint=False)
    for i, ang in enumerate(angles):
        probe = build_probe(sheet, float(ang), float(args.probe))
        res_norm = rigid_resultants(probe, sheet.centroids[sheet.faces], origin=sheet.origin)
        f_full = scatter_load(probe, sheet.faces, sheet.n_faces).reshape(-1)
        delta = scatter(response(f_full[free_ids], int(args.n_soft)))

        row, pocket_field, parts = measure(sheet, delta, sheet.displacement + delta, global_fields,
                                           mode_field, f"linear_{i}")
        row["angle"] = float(ang)
        row["angle_deg"] = float(np.degrees(ang))
        row["probe"] = float(args.probe)
        # Statically determinate gauge: its reaction is the negated resultant, so this IS the check.
        row["resultants_rel"] = (np.abs(res_norm) / max(args.probe, 1e-30)).tolist()
        rows.append(row)
        fields[f"delta__{i}"] = delta
        fields[f"pocket__{i}"] = pocket_field
        fields[f"force__{i}"] = parts["total"]
        fields[f"own__{i}"] = parts["own"]
        fields[f"redist__{i}"] = parts["redist"]
        print(f"  dir {np.degrees(ang):6.1f} deg   force radius "
              f"{row.get('force_radius', float('nan')):5.2f} tiles   "
              f"aspect {row.get('force_aspect', float('nan')):5.2f}   "
              f"axis {np.degrees(row.get('force_angle', np.nan)):6.1f} deg   "
              f"gyration {row.get('force_gyration', float('nan')):5.2f}   "
              f"resultants {max(row['resultants_rel']):.1e}", flush=True)

    # The same probe with its resultants LEFT IN, so the gauge must react. One extra
    # back-substitution, and it is what makes "a support reaches further than a load" a statement
    # about one sheet and one quantity rather than a comparison across two setups.
    net = build_probe(sheet, 0.0, float(args.probe), balanced=False)
    f_net = scatter_load(net, sheet.faces, sheet.n_faces).reshape(-1)
    delta_net = scatter(response(f_net[free_ids], int(args.n_soft)))
    parts_net = sheet.probe_fields(delta_net)
    dist_net = graph_distance(sheet.pairs, sheet.n_faces, [sheet.centre])
    net_decay = decay_ratios(parts_net["own"], dist_net).as_dict()
    print(f"[net load] resultants {np.abs(rigid_resultants(net, sheet.centroids[sheet.faces], origin=sheet.origin)).max() / max(args.probe, 1e-30):.2f} "
          f"of the probe   own r1 {net_decay['ratios'][0]:.2f}   "
          f"lam {net_decay['decay_length']:.2f}", flush=True)
    fields["net_own"] = parts_net["own"]
    fields["net_total"] = parts_net["total"]

    fields["mode"] = mode_field
    fields["bg_response"] = bg_field
    fields["evals_low"] = evals[:20]
    return {"rows": rows, "fields": fields, "global_fields": global_fields,
            "net_decay": net_decay,
            "lam0": float(evals[0]), "lam1": float(evals[1]),
            "mode_checkerboard_overlap": overlap, "bg_vs_mode_cos": cos_bg_mode,
            "grad_norm": float(np.linalg.norm(grad)), "n_free": int(free_ids.size)}


# ══════════════════════════════════════════════════════════════════════════════
# Layer 2: nonlinear response
# ══════════════════════════════════════════════════════════════════════════════

def nonlinear_layer(sheet: Sheet, args, global_fields: np.ndarray, mode_field: np.ndarray) -> dict:
    """Full solves at growing probe magnitude; the pocket is the difference of two equilibria."""
    rows, fields = [], {}
    magnitudes = np.geomspace(float(args.mag_min), float(args.mag_max), int(args.n_mags))
    for i, mag in enumerate(magnitudes):
        t0 = time.time()
        probe = build_probe(sheet, float(args.angle), float(mag))
        loads = sheet.background_loads + load_specs(probe, sheet.faces)
        res, *_ = solve_at(sheet.config, None, loads, int(args.steps))
        disp = np.asarray(res['solution'].fields[-1])
        delta = disp - sheet.displacement

        row, pocket_field, parts = measure(sheet, delta, disp, global_fields, mode_field,
                                           f"nonlinear_{i}")
        row["magnitude"] = float(mag)
        row["angle"] = float(args.angle)
        row["wall_s"] = time.time() - t0
        rows.append(row)
        fields[f"nl_delta__{i}"] = delta
        fields[f"nl_force__{i}"] = parts["total"]
        fields[f"nl_own__{i}"] = parts["own"]
        print(f"  mag {mag:9.4g} N   force radius {row.get('force_radius', float('nan')):5.2f} "
              f"tiles   aspect {row.get('force_aspect', float('nan')):5.2f}   "
              f"axis {np.degrees(row.get('force_angle', np.nan)):6.1f} deg   "
              f"peak {row.get('force_peak', float('nan')):.3e}   {row['wall_s']:5.1f}s", flush=True)
        jax.clear_caches()
    fields["nl_magnitudes"] = magnitudes
    return {"rows": rows, "fields": fields}


# ══════════════════════════════════════════════════════════════════════════════
# Background calibration
# ══════════════════════════════════════════════════════════════════════════════

def calibrate_background(config, sigma, base: float, target_deg: float, n_steps: int,
                         lo: float = 0.1, hi: float = 5.0) -> float:
    """Bisect a SCALE on the background moment until the mean hinge rotation hits ``target_deg``.

    Bisecting the scale rather than the moment keeps the sign fixed. The sign matters: the two
    signs are not mirror images, because one drives the sheet open and the other drives it into the
    contact barrier, where it stops at zero and every magnitude looks identical.
    """

    def theta_mean(scale: float) -> float:
        loads = checkerboard_background(sigma, base * scale)
        res, *_ = solve_at(config, None, loads, n_steps)
        vs = res['valid_state']
        disp = np.asarray(res['solution'].fields[-1])
        n_nodes = np.asarray(vs.centroid_node_vectors).shape[1]
        pairs = tile_adjacency(np.asarray(vs.bond_connectivity), n_nodes)
        rel = np.abs(relative_rotations(disp[:, 2], pairs))
        jax.clear_caches()
        return float(np.degrees(rel.mean()))

    for _ in range(12):
        mid = 0.5 * (lo + hi)
        t = theta_mean(mid)
        print(f"  M = {base * mid:9.4f}  ->  theta mean {t:7.3f} deg", flush=True)
        if abs(t - target_deg) < 0.25:
            return base * mid
        lo, hi = (mid, hi) if t < target_deg else (lo, mid)
    return base * 0.5 * (lo + hi)


# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="pocket_11")
    ap.add_argument("--layer", default="linear", choices=("linear", "nonlinear", "both"))
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--maxiter", type=int, default=None, help="override physics.solver_maxiter")
    ap.add_argument("--tol", type=float, default=None, help="override physics.solver_tol")
    ap.add_argument("--probe", type=float, default=1.0, help="linear probe magnitude, newtons")
    ap.add_argument("--directions", type=int, default=8, help="probe directions over [0, pi)")
    ap.add_argument("--n-soft", type=int, default=2,
                    help="soft modes excluded when building the response; see --soft-scan")
    ap.add_argument("--grad-gate", type=float, default=1e-6,
                    help="reject the background state if its gradient norm exceeds this")
    ap.add_argument("--soft-scan", action="store_true",
                    help="report the pocket against the number of excluded soft modes")
    ap.add_argument("--angle", type=float, default=0.0, help="probe direction for the nonlinear layer")
    ap.add_argument("--mag-min", type=float, default=1.0)
    ap.add_argument("--mag-max", type=float, default=1000.0)
    ap.add_argument("--n-mags", type=int, default=8)
    ap.add_argument("--bg-mode", default="checkerboard", choices=("checkerboard", "point"),
                    help="checkerboard drives every tile and leaves no actuator to contaminate the "
                         "difference; point reproduces the single-moment actuation for comparison")
    ap.add_argument("--bg-moment", type=float, default=-3.42,
                    help="per-tile background moment (checkerboard) or the single moment (point). "
                         "⚠ the SIGN selects opening or closing, and the closing sign is stopped "
                         "dead by the contact barrier at theta = 0")
    ap.add_argument("--gauge", type=int, default=None, help="override the clamped gauge tile")
    ap.add_argument("--pair-axis", default="x", choices=("x", "y"),
                    help="which neighbour carries the other half of the probe; swapping it "
                         "separates a pocket aimed by the load from one aimed by the tiling")
    ap.add_argument("--calibrate-background", action="store_true")
    ap.add_argument("--target-angle", type=float, default=45.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    path = resolve_config(args.config)
    config = copy.deepcopy(load_and_parse_config(path))
    if args.gauge is not None:
        config.topology['bc_clamped'] = [int(args.gauge)]
    if args.maxiter is not None:
        config.physics.solver_maxiter = int(args.maxiter)
    if args.tol is not None:
        config.physics.solver_tol = float(args.tol)
    M, N = int(config.topology['M']), int(config.topology['N'])
    sigma = _checkerboard_signs(M, N).astype(float)

    if args.bg_mode == "checkerboard":
        background = checkerboard_background(sigma, float(args.bg_moment))
    else:
        background = list(config.topology['loads'])
        if args.bg_moment is not None:
            background = [{'face': 1, 'dof': 2, 'value': float(args.bg_moment)}]

    if args.calibrate_background:
        print(f"[calibrate] bisecting the {args.bg_mode} background for theta mean = "
              f"{args.target_angle} deg", flush=True)
        value = calibrate_background(config, sigma, float(args.bg_moment), args.target_angle,
                                     args.steps)
        print(f"[calibrate] use  --bg-moment {value:.4f}")
        return

    out_dir = args.out or os.path.join("data/outputs/pocket", os.path.basename(path)[:-5])
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    sheet = Sheet(config, args.steps, background_loads=background, pair_axis=args.pair_axis)
    print(f"[sheet] {sheet.M}x{sheet.N}  {sheet.n_faces} faces  {sheet.pairs.shape[0]} hinges   "
          f"background solve {time.time() - t0:.1f}s", flush=True)
    print(f"[sheet] psi {np.degrees(sheet.fit.psi):.3f} deg   1-DOF r2 {sheet.fit.r2:.6f}   "
          f"theta {sheet.theta['min']:.2f}..{sheet.theta['max']:.2f} deg "
          f"(uniformity {sheet.theta['uniformity']:.4f})", flush=True)
    pair_vec = sheet.centroids[sheet.partner] - sheet.centroids[sheet.centre]
    print(f"[sheet] gauge tile {config.topology['bc_clamped']}   probe on faces {sheet.faces}   "
          f"pair axis '{sheet.pair_axis}' lies at "
          f"{np.degrees(np.arctan2(pair_vec[1], pair_vec[0])) % 180:.1f} deg", flush=True)

    store, out = {}, {"config": path, "sheet": sheet.summary(), "args": vars(args)}

    lin = linear_layer(sheet, args)
    out["linear"] = {k: v for k, v in lin.items() if k not in ("fields", "global_fields")}
    store.update(lin["fields"])
    mode_field, global_fields = lin["fields"]["mode"], lin["global_fields"]

    if args.layer in ("nonlinear", "both"):
        print("[nonlinear] full solves, pocket = probed minus unprobed", flush=True)
        nl = nonlinear_layer(sheet, args, global_fields, mode_field)
        out["nonlinear"] = {"rows": nl["rows"]}
        store.update(nl["fields"])

    store["centroids"] = sheet.centroids
    store["nodes"] = sheet.nodes
    store["face_pairs"] = sheet.pairs
    store["sigma"] = sheet.sigma
    store["background_displacement"] = sheet.displacement
    np.savez_compressed(os.path.join(out_dir, "pocket.npz"), **store)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"[pocket] wrote {out_dir}/pocket.npz + summary.json", flush=True)


if __name__ == "__main__":
    main()
