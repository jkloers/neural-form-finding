"""Problem setup for the closed-state (closed_les) inverse-design pipeline.

Builds the flat closed-sheet tessellation (initial state, clamps, loads) and the differentiable
design parameters from a parsed config. The hinge energy is built separately, by
``nff.hinge.build_hinge_model``.

The flat sheet is the Dang et al. (2021) RDPQK construction; the deployed shape
is produced later by Stage-2 physics. ``map_type: closed_les`` optimizes the
per-cut aspect ratios and the boundary-slider positions through the pipeline.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from nff.topology.closed_builder import build_closed_tessellation
from nff.topology.closed_builder_jax import build_deploy_structure, build_boundary_edges
from nff.config.conditions import configure_tessellation
from nff.stages.state import CentroidalState


def build_closed_initial_state(config):
    """Build a flat closed-state RDPQK tessellation + CentroidalState from config.

    Geometry comes from the closed builder (M×N panels); material, clamps, and
    loads are applied with the standard config machinery.
    """
    topo = config.topology
    M = int(topo['M'])
    N = int(topo['N'])
    r_init = float(topo.get('r_init', 0.45))
    spacing = float(topo.get('spacing', 1.0))
    spacing_y = float(topo.get('spacing_y', spacing))

    tessellation = build_closed_tessellation(M, N, r=r_init, spacing=spacing, spacing_y=spacing_y)
    configure_tessellation(tessellation, SimpleNamespace(**topo))  # material, clamps, loads

    # Optional: restrict which DOFs the clamp fixes (default all 3 = [x, y, theta]).
    # e.g. clamped_dofs: [0, 1] pins translation but leaves rotation free.
    clamp_dofs = topo.get('clamped_dofs', None)
    clamped = topo.get('bc_clamped')
    if clamp_dofs is not None and isinstance(clamped, list):
        for f in clamped:
            tessellation.set_face_dofs(int(f), list(clamp_dofs))
    # Additionally pin y (DOF 1) on specific faces — e.g. anchor one corner so a
    # net vertical load has support while the rest of the edge stays x-only.
    for f in topo.get('y_pin_faces', []) or []:
        tessellation.set_face_dofs(int(f), [0, 1])
    # y-roller: pin y ONLY (DOF 1) so the face slides freely in x under an x-load. Removes the
    # rigid rotation mode of a single-tile pinch without blocking the pull (an x-force face can
    # still be y-rollered — different DOFs).
    for f in topo.get('y_roller_faces', []) or []:
        tessellation.set_face_dofs(int(f), [1])
    # x-roller: the transpose — pin x ONLY (DOF 0) so the face slides freely in y.
    # ⚠ For the standard 1x2 sheet under a +y pull this is the WRONG anti-swing support, and the
    # intuition that it should be the right one is a trap worth recording. Both rollers kill the
    # swing equally well (tilt -10.2 deg -> about -1 deg either way), but the restraint costs what it
    # BLOCKS, and what it blocks is set by the mechanism's anisotropy, not by the load direction:
    # this sheet expands lam_x = 1.48 against lam_y = 1.10, so x is the dominant deployment axis even
    # though the pull is along y. Measured, single-tile grip c0->p8 at 6.49 N:
    #     y-roller  theta_max 16.90 deg,  deployed width 3.741 u
    #     x-roller  theta_max  5.81 deg,  deployed width 3.189 u   <- 3x less opening
    # Use the y-roller for a y-pull on an x-dominant sheet. This option exists for the transposed
    # case (an x-pull, or a sheet whose long expansion axis is y).
    for f in topo.get('x_roller_faces', []) or []:
        tessellation.set_face_dofs(int(f), [0])

    state = CentroidalState.from_tessellation(tessellation, target_cfg=config.target)
    return state, tessellation


def init_closed_les_params(config):
    """Init design params and static LES structure for the closed_les map type.

    Params: {'z': (rows, cols) latent for r = sigmoid(z), init r_init;
             'bnd_logits': (n_logits,) per-edge logits -> ordered boundary sliders}.
    The ordered-boundary + r∈(0,1) parameterization guarantees a valid (non-self-
    intersecting) flat tessellation, so no validity loss is needed.

    Optional RANDOM starting design (topology ``init_noise`` > 0, seeded by ``init_seed``): adds
    Gaussian noise to z and bnd_logits. Still valid by construction (r = sigmoid stays in (0,1);
    ordered-softmax boundary stays convex), so different random starts probe whether the optimizer
    is init-sensitive / stuck at the uniform symmetric attractor. ``init_noise`` = 0 (default) is the
    deterministic uniform init -- backward-compatible.
    """
    topo = config.topology
    M, N = int(topo['M']), int(topo['N'])
    r_init = float(topo.get('r_init', 0.45))
    spacing = float(topo.get('spacing', 1.0))
    spacing_y = float(topo.get('spacing_y', spacing))

    struct = build_deploy_structure(M, N)
    sliders = build_boundary_edges(struct, spacing=spacing, spacing_y=spacing_y)

    z_init = float(np.log(r_init / (1.0 - r_init)))         # sigmoid(z_init) = r_init
    z = np.full((struct['rows'], struct['cols']), z_init)
    bnd = np.asarray(sliders['init_logits'], dtype=float)
    noise = float(topo.get('init_noise', 0.0))
    if noise > 0.0:
        rng = np.random.default_rng(int(topo.get('init_seed', 0)))
        z = z + rng.normal(0.0, noise, size=z.shape)
        bnd = bnd + rng.normal(0.0, noise, size=bnd.shape)
    params = {'z': jnp.asarray(z, dtype=float), 'bnd_logits': jnp.asarray(bnd, dtype=float)}
    static_features = {'struct': struct, 'sliders': sliders, 'closed_les': True}
    return params, static_features
