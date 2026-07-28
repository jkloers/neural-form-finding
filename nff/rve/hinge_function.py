"""The hinge as a function.

A hinge is a constitutive map

    hinge : (geometry g, kinematics u) -> (energy W, generalized force F = dW/du, validity)

where ``u = (a, s, theta)`` are the three relative-tile DOF (axial, shear, rotation)
and ``F = (F_a, F_s, M_theta)`` is the reaction work-conjugate to them. Failure is a
function of the *whole* ``u``, not of ``theta`` alone.

Two implementations share this signature:

  - the ORACLE (here): CalculiX evaluates the map along a monotonic deployment ray,
    returning a whole path of ``(u, W, F, validity)`` samples. Expensive, exact.
  - the SURROGATE (later, JAX side): a learned, differentiable ``W_theta(u; g)`` with
    the identical ``(g, u) -> (W, dW/du)`` contract. Cheap, pointwise.

This module is the single source of truth for how a hinge geometry + a deployment ray
become an evaluation of that map. It is numpy-only (runs in the ``ccx`` env); it does
not import JAX. The ``ligament_strains`` reparametrization used by the JAX pipeline is a
fixed diffeomorphism of ``(a, s, theta)`` and is applied downstream at integration time
(the "shared kinematic map", brief section 10) -- the oracle coordinates ``(a, s, theta)``
are the ones the reactions are conjugate to, so the Sobolev labels are exact here.

Frame convention (load-bearing -- this is what the Phase-3 point-ROM bridge must match).
The kinematics ``u = (a, s, theta)`` are the motion of tile B relative to tile A, expressed
in the HINGE-LOCAL frame set by the cut geometry:
  - pivot   = the main-cut tip ``(0, -w_lig)`` (the rotation centre),
  - a       = translation along the SECONDARY cut (the +x axis of the RVE),
  - s       = translation PERPENDICULAR to the secondary cut (+y),
  - theta   = relative rotation about the pivot.
The point ROM (l0 ~ 0) has no such frame until we give ``reference_bond_vectors`` the cut
orientation; the bridge projects the ROM's relative face DOF onto exactly these axes.

Units: SI-ish -- lengths mm, angle rad (theta) / deg (theta_deg), energy N.mm, force N,
moment N.mm.
"""

from dataclasses import dataclass, field, replace

import numpy as np

from nff.rve.geometry import RVEParams
from nff.rve.ccx_solver import STEEL, deploy
from nff.rve.materials import coerce_material


# ── validity regimes ────────────────────────────────────────────────────────────
ELASTIC, PLASTIC, FAILED = 0, 1, 2
REGIME_NAME = {ELASTIC: "elastic", PLASTIC: "plastic", FAILED: "failed"}
_PLASTIC_TOL = 1e-6                                  # PEEQ above this => yielded


@dataclass(frozen=True)
class HingeConstants:
    """The fixed manufacturing + material context every hinge shares in a campaign.

    These are NOT swept: one steel, one gauge, one kerf, one window, one fillet law.
    Only ``HingeGeometry`` (w_lig, alpha) varies across the geometry space.
    """
    thickness: float = 1.0                           # laser-cuttable mild-steel gauge [mm]
    w_c: float = 0.2                                  # cut kerf [mm]
    r_win: float = 30.0                               # Saint-Venant window radius [mm]
    fillet_ratio: float = 0.16                        # rho = fillet_ratio * w_lig (set by rehearsal)
    material: object = field(default_factory=lambda: dict(STEEL))   # Material, params dict, or name
    eps_f: float | None = None                        # fracture strain; None -> the material's own
    n_through: int = 2                                # elements through thickness (>=2 for plastic bending)
    lc_fillet_frac: float = 0.4                       # resolve the fillet: lc_min = frac * rho
    lc_min_floor: float = 0.06                        # never mesh finer than this [mm]
    imp_amp: float | None = None                      # out-of-plane buckle seed [mm]; None -> 0.3*t
    min_inc: float = 1e-3                             # min load increment (smaller babies through snaps)
    stabilize: float | None = None                    # *STATIC,STABILIZE damping past buckling (uz-only)
    arcB_uz_free: bool = False                        # leave the driven arc free out of plane
    field_freq: int = 1000                            # *NODE/EL FILE FREQUENCY -> 1 frame per state
    el_fields: str | None = None                      # *EL FILE list; None -> the material's own
    stop_at_fracture: bool = True                     # poll the .frd and kill ccx past eps_f
    # ^ worth it for a brittle material, where it skips the deep-plastic grind past rupture. For a
    # ductile one it is pure cost: PET folds peak near PEEQ 0.4 against eps_f 1.784, so the poll can
    # never fire, yet it re-parses the whole (growing) .frd every few seconds in Python under the
    # GIL, once per worker. Set False for a PET campaign; `eps_f` still labels the regime.

    def __post_init__(self):
        # normalise the material once, and take eps_f from it unless explicitly overridden -- so a
        # PET campaign cannot silently early-stop on steel's 0.25.
        mat = coerce_material(self.material)
        object.__setattr__(self, "material", mat)
        if self.eps_f is None:
            object.__setattr__(self, "eps_f", mat.eps_f)


@dataclass(frozen=True)
class HingeGeometry:
    """The hinge's identity: the shape DOF that vary across the campaign.

    ``fillet_ratio`` (the rounded 'little circle' at the cut tip, ``rho = fillet_ratio * w_lig``) is
    now a per-hinge DOF -- the main stress-relief lever -- not a fixed campaign constant. Defaults to
    the legacy 0.16 so 2-DOF callers/datasets are unchanged.
    """
    w_lig: float                                      # ligament gap (main-cut tip -> secondary) [mm]
    alpha_deg: float                                  # angle between the two cuts [deg]
    fillet_ratio: float = 0.16                        # rho = fillet_ratio * w_lig (cut-tip fillet)

    @property
    def tag(self) -> str:
        return f"w{self.w_lig:07.3f}_a{self.alpha_deg:06.1f}_f{self.fillet_ratio:05.3f}"

    def rho(self, const: HingeConstants = None) -> float:
        """Cut-tip fillet radius [mm]. ``const`` is ignored (kept for call-site compatibility)."""
        return self.fillet_ratio * self.w_lig


@dataclass(frozen=True)
class DeploymentRay:
    """A monotonic, proportional path ``u(lambda) = lambda * u1``, ``lambda in (0, 1]``.

    Samples ALL three DOF. Translations are set as fractions of the ligament width -- the
    neck-strain scale, since ductile failure is governed by ``a/w_lig``:

        a1 = eta_a * w_lig      (axial opening; eta_a >= 0 -> deployment / tension only)
        s1 = eta_s * w_lig      (shear; either sign)

    ``eta_a, eta_s`` are dimensionless neck-strain ratios, DECOUPLED from theta1. Out-of-plane
    buckling relieves membrane strain, so a buckled ligament can accommodate large in-plane
    shear/stretch -- these are sampled GENEROUSLY (eta ~ O(1)) and each ray runs to its own
    PEEQ-flagged failure. ``eta = 0`` is the pure-rotation spine the deployed pipeline rides.
    """
    theta1_deg: float                                 # full-deployment rotation [deg]
    eta_a: float = 0.0                                # axial neck-strain ratio: a1 = eta_a * w_lig
    eta_s: float = 0.0                                # shear neck-strain ratio: s1 = eta_s * w_lig
    n_steps: int = 20
    tag: str = ""

    def targets(self, geo: HingeGeometry):
        """(a1, s1, theta1_deg) full-deployment handle motion for this geometry."""
        return self.eta_a * geo.w_lig, self.eta_s * geo.w_lig, self.theta1_deg

    def states(self, geo: HingeGeometry) -> np.ndarray:
        """(n_steps, 3) kinematic states (a, s, theta[rad]) driven into the oracle."""
        a1, s1, theta1_deg = self.targets(geo)
        dth = np.radians(theta1_deg) / self.n_steps
        return np.array([(a1 * (k + 1) / self.n_steps, s1 * (k + 1) / self.n_steps, (k + 1) * dth)
                         for k in range(self.n_steps)], float)


@dataclass(frozen=True)
class DeploymentPath:
    """An arbitrary measured polyline ``u(k) = (a, s, theta)``, replayed state by state.

    The oracle is elastoplastic, so ``W`` is a path-dependent work rather than a potential: two
    routes to the same endpoint do not carry the same energy or the same accumulated damage. A
    surrogate fitted as a state function ``W(a, s, theta)`` is therefore only legitimate if it is
    trained along paths resembling the ones the deployed sheet actually rides -- hence replay of
    the harvested polylines (``nff.closed.path_dataset``) instead of proportional rays.

    ``polyline`` is PHYSICAL: (a, s) in mm, theta in rad, one row per state, origin NOT included
    (the undeformed state is implicit at t = 0).
    """
    polyline: np.ndarray                              # (n_steps, 3) = (a[mm], s[mm], theta[rad])
    tag: str = ""

    def __post_init__(self):
        object.__setattr__(self, "polyline", np.asarray(self.polyline, float).reshape(-1, 3))

    @property
    def n_steps(self) -> int:
        return len(self.polyline)

    @property
    def theta1_deg(self) -> float:
        """Peak rotation along the path [deg] -- the campaign's rotation-depth descriptor."""
        return float(np.degrees(np.max(self.polyline[:, 2])))

    def targets(self, geo: HingeGeometry):
        """(a, s, theta_deg) of the FINAL state -- the endpoint this path arrives at."""
        a1, s1, th1 = self.polyline[-1]
        return float(a1), float(s1), float(np.degrees(th1))

    def states(self, geo: HingeGeometry) -> np.ndarray:
        """(n_steps, 3) kinematic states -- the measured polyline, verbatim."""
        return self.polyline


@dataclass
class HingeResponse:
    """A path of samples of the constitutive map for one (geometry, ray).

    All arrays are aligned and length ``n_samples`` (one per solved increment).
    Kinematics ``u = (a, s, theta)``; conjugate forces ``F = (F_a, F_s, M_theta)``.
    """
    geo: HingeGeometry
    ray: DeploymentRay | DeploymentPath
    const: HingeConstants
    # kinematics u
    a: np.ndarray
    s: np.ndarray
    theta: np.ndarray                                 # [rad]
    theta_deg: np.ndarray
    # response
    W: np.ndarray                                     # stored energy [N.mm]
    F_a: np.ndarray
    F_s: np.ndarray
    M_theta: np.ndarray                               # dW/du by the envelope theorem
    damage: np.ndarray                                # Delta = <PEEQ>_lig / eps_f (nff.rve.damage)
    peeq_lig: np.ndarray                              # hottest ligament element -- the tear indicator
    eta_mean_lig: np.ndarray                          # <eta> over the ligament (diagnostic)
    uz_max: np.ndarray                                # out-of-plane amplitude [mm]
    regime: np.ndarray                                # ELASTIC / PLASTIC / FAILED per sample
    # provenance / summary
    n_elems: int
    ok: bool
    failure_theta_deg: float                          # first theta with regime==FAILED (nan if survives)

    @property
    def n_samples(self) -> int:
        return len(self.theta_deg)

    @property
    def damage_at_tear(self) -> float:
        """``Delta_tear``: the damage reading at the first torn frame; NaN if the hinge survives.

        The campaign's calibrated fracture line -- a reported number and a line on plots, not a
        second damage measure and not a term in any loss.
        """
        failed = self.regime == FAILED
        return float(self.damage[np.argmax(failed)]) if failed.any() else float("nan")


def to_rve_params(geo: HingeGeometry, const: HingeConstants) -> RVEParams:
    """Geometry + fixed context -> the meshable RVE parameters."""
    return RVEParams(w_lig=geo.w_lig, w_c=const.w_c, alpha_deg=geo.alpha_deg,
                     rho=geo.rho(const), thickness=const.thickness, r_win=const.r_win)


def descriptor(geo: HingeGeometry, const: HingeConstants) -> dict:
    """The per-hinge descriptor carried with every dataset row (dimensional + ratios)."""
    return dict(w_lig=geo.w_lig, alpha_deg=geo.alpha_deg, fillet_ratio=geo.fillet_ratio,
                t_over_wlig=const.thickness / geo.w_lig,
                wc_over_wlig=const.w_c / geo.w_lig,
                rho_over_wlig=geo.fillet_ratio,
                sigy_over_E=coerce_material(const.material).yield_strain)


def solver_kwargs(geo: HingeGeometry, ray: DeploymentRay, const: HingeConstants) -> dict:
    """The single place that turns (geometry, ray, context) into oracle call kwargs.

    Used by BOTH ``evaluate_hinge`` (serial) and the parallel campaign, so there is one
    definition of how a hinge is driven.
    """
    a1, s1, theta1_deg = ray.targets(geo)
    rho = geo.rho(const)
    # Always drive the oracle from an explicit state list, so a DeploymentRay and a replayed
    # DeploymentPath take the SAME code path (the ray's states are the proportional ramp).
    return dict(angle_deg=theta1_deg, n_steps=ray.n_steps, a=a1, s=s1, states=ray.states(geo),
                n_through=const.n_through, material=const.material,
                imp_amp=const.imp_amp, min_inc=const.min_inc, stabilize=const.stabilize,
                arcB_uz_free=const.arcB_uz_free, field_freq=const.field_freq,
                el_fields=const.el_fields,
                lc_min=max(const.lc_fillet_frac * rho, const.lc_min_floor),
                lc_max=const.r_win / 5.0)


def classify_regime(peeq_lig: np.ndarray, eps_f: float) -> np.ndarray:
    """elastic (no plasticity) / plastic (yielded, intact) / failed (torn).

    Keyed on the SAME ligament peak that stops the solve, so the dataset's ``FAILED`` label and
    the solver's stop-at-fracture agree by construction -- there is no second failure criterion.
    """
    p = np.asarray(peeq_lig, float)
    return np.where(p >= eps_f, FAILED, np.where(p > _PLASTIC_TOL, PLASTIC, ELASTIC))


def assemble_response(geo, ray, const, parsed) -> HingeResponse:
    """Turn a parsed CalculiX result into aligned samples of the constitutive map.

    ``(a, s)`` come from the states actually imposed (``parse_job`` interpolates the driven state
    list onto the solved increments), so a replayed polyline is labelled by its real route rather
    than by a proportionality that only holds for a straight ray.
    """
    theta_deg = np.asarray(parsed["theta_deg"], float)
    # align every field to the common number of solved increments (guards ragged parses)
    fields = ["W", "F_a", "F_s", "M_theta", "damage", "peeq_lig", "eta_mean_lig", "uz_max"]
    n = min([len(theta_deg)] + [len(np.asarray(parsed[k])) for k in fields])
    theta_deg = theta_deg[:n]
    W, F_a, F_s, M_theta, damage, peeq_lig, eta_mean_lig, uz_max = (
        np.asarray(parsed[k], float)[:n] for k in fields)

    if parsed.get("a") is not None:
        a, s = np.asarray(parsed["a"], float)[:n], np.asarray(parsed["s"], float)[:n]
    else:                                             # legacy parse without imposed-state labels
        a1, s1, theta1_deg = ray.targets(geo)
        frac = theta_deg / theta1_deg if theta1_deg else np.zeros(n)
        a, s = a1 * frac, s1 * frac
    theta = np.radians(theta_deg)

    regime = classify_regime(peeq_lig, const.eps_f)
    failed = regime == FAILED
    failure_theta = float(theta_deg[np.argmax(failed)]) if failed.any() else float("nan")

    return HingeResponse(geo=geo, ray=ray, const=const, a=a, s=s, theta=theta,
                         theta_deg=theta_deg, W=W, F_a=F_a, F_s=F_s, M_theta=M_theta,
                         damage=damage, peeq_lig=peeq_lig, eta_mean_lig=eta_mean_lig,
                         uz_max=uz_max, regime=regime,
                         n_elems=int(parsed.get("n_elems", 0)), ok=bool(parsed.get("ok", False)),
                         failure_theta_deg=failure_theta)


def evaluate_hinge(geo: HingeGeometry, ray: DeploymentRay,
                   const: HingeConstants = HingeConstants(),
                   *, timeout: float = 900, workdir: str = None,
                   ncpus: int = 1) -> HingeResponse:
    """Evaluate the constitutive map along one deployment ray (serial, one hinge).

    This IS the hinge-as-a-function: (geometry, ray) -> path of (u, W, dW/du, validity).
    The dataset campaign (``nff.rve.dataset``) is nothing but this, phased over many
    hinges for speed; it reuses ``solver_kwargs`` and ``assemble_response`` verbatim.
    """
    parsed = deploy(to_rve_params(geo, const), timeout=timeout,
                    eps_f=const.eps_f if const.stop_at_fracture else None,
                    ncpus=ncpus,
                    workdir=workdir or f"/tmp/hinge/{geo.tag}_{ray.tag or 't'}",
                    **solver_kwargs(geo, ray, const))
    return assemble_response(geo, ray, const, parsed)
