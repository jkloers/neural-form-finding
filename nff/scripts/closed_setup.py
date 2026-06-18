"""Shared setup for the closed-state (closed_les) inverse-design pipeline.

Builds the flat closed-sheet tessellation and the differentiable design
parameters from a parsed config. Imported by both ``train.py`` (generic
dispatch) and ``run_closed.py`` (the closed-state driver) so neither script
imports the other.

The flat sheet is the Dang et al. (2021) RDPQK construction; the deployed shape
is produced later by Stage-2 physics. ``map_type: closed_les`` optimizes the
per-cut aspect ratios and the boundary-slider positions through the pipeline.
"""

from types import SimpleNamespace

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

    tessellation = build_closed_tessellation(M, N, r=r_init, spacing=spacing)
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

    state = CentroidalState.from_tessellation(tessellation, target_cfg=config.target)
    return state, tessellation


def init_closed_les_params(config):
    """Init design params and static LES structure for the closed_les map type.

    Params: {'z': (rows, cols) latent for r = sigmoid(z), init r_init;
             'bnd_logits': (n_logits,) per-edge logits -> ordered boundary sliders}.
    The ordered-boundary + r∈(0,1) parameterization guarantees a valid (non-self-
    intersecting) flat tessellation, so no validity loss is needed.
    """
    topo = config.topology
    M, N = int(topo['M']), int(topo['N'])
    r_init = float(topo.get('r_init', 0.45))
    spacing = float(topo.get('spacing', 1.0))

    struct = build_deploy_structure(M, N)
    sliders = build_boundary_edges(struct, spacing=spacing)

    z_init = float(np.log(r_init / (1.0 - r_init)))         # sigmoid(z_init) = r_init
    params = {
        'z': jnp.full((struct['rows'], struct['cols']), z_init),
        'bnd_logits': jnp.asarray(sliders['init_logits'], dtype=float),
    }
    static_features = {'struct': struct, 'sliders': sliders, 'closed_les': True}
    return params, static_features
