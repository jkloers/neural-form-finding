"""
Stage 0 — Initial mapping in centroidal coordinates.

Mapping engines, all pure JAX with the interface CentroidalState -> CentroidalState:

1. Closed chart (apply_closed_les_mapping) — the main pipeline.
   Per-cut aspect ratios r = sigmoid(z) and ordered boundary sliders are solved through the
   collinearity LES into a flat closed sheet; valid by construction, so Stage 1 is bypassed.

2. Free vertices (apply_direct_mapping) — the ablation baseline.
   Every panel vertex is a free parameter; validity must be restored by Stage-1 alternating
   projections. Used only by the chart ablation (nff/analysis/chart_ablation.py) that measures what
   the closed chart buys. The analytic open-pattern maps and direct_transform are archived in
   legacy/open_pipeline/mapping_with_open_maps.py; the GNN engines in legacy/gnn/.
"""
import jax
import jax.numpy as jnp
from typing import Union, Dict
from nff.stages.state import CentroidalState
from nff.topology.closed_builder_jax import (
    solve_cut_vertices_jax, boundary_flat_from_logits,
)

def parse_map_params(raw_params: Union[Dict, jnp.ndarray, list]) -> Union[Dict, jnp.ndarray]:
    """Converts raw mapping parameters (from YAML/dict/list) into JAX-compatible format.
    
    Dictionaries are preserved but values are converted to jnp.arrays.
    Lists/arrays are converted to jnp.float64 arrays.
    """
    if isinstance(raw_params, dict):
        return {
            k: v if isinstance(v, bool) else jnp.array(v, dtype=float)
            for k, v in raw_params.items()
        }
    return jnp.array(raw_params, dtype=float)


def apply_direct_mapping(state: CentroidalState, map_params: dict) -> CentroidalState:
    """Stage 0 direct parameterization.

    map_params is the learnable PyTree itself:
        {'face_centroids': (n_faces, 2), 'centroid_node_vectors': (n_faces, max_nodes, 2)}

    No analytical function, no Jacobian — the parameters ARE the geometry.
    All downstream constraints (connectivity, non-intersection, physics) flow
    gradients back through this identity-like pass.
    """
    return state._replace(
        face_centroids=map_params['face_centroids'],
        centroid_node_vectors=map_params['centroid_node_vectors'],
    )


def apply_closed_les_mapping(state: CentroidalState, map_params: dict,
                             static_features: dict) -> CentroidalState:
    """Stage 0 for the closed-state RDPQK builder.

    Builds the flat closed-sheet geometry from the design parameters via the
    differentiable collinearity LES, replacing only the optimizable fields. All
    topology (hinges, BCs, boundary ids, stiffnesses) is carried unchanged from
    ``state``. Gradients flow back to the design parameters through the linear
    solve and the boundary-slider parameterization.

    Validity is guaranteed by construction: r = sigmoid(z) in (0, 1) gives
    positive barycentric weights, and the boundary logits keep the edge sliders
    strictly ordered (convex boundary) — together the Tutte-embedding conditions,
    so the flat tessellation is non-self-intersecting with no validity penalty.

    Args:
        state: CentroidalState carrying the static topology (built once from the
            closed tessellation).
        map_params: {'z': (rows, cols), 'bnd_logits': (n_logits,)} — design vars.
            Aspect ratios r = sigmoid(z) in (0, 1); boundary positions from
            per-edge cumulative-softmax of bnd_logits.
        static_features: {'struct': build_deploy_structure(...),
            'sliders': build_boundary_edges(...)}.

    Returns:
        CentroidalState with flat geometry from the current design parameters.
    """
    struct = static_features['struct']
    sliders = static_features['sliders']

    r = jax.nn.sigmoid(map_params['z'])
    boundary_flat = boundary_flat_from_logits(sliders, map_params['bnd_logits'])
    coords = solve_cut_vertices_jax(struct, boundary_flat, r)      # (P, 2) flat
    panel_flat = coords[struct['corner_pid']]                      # (n_faces, 4, 2)
    face_centroids = panel_flat.mean(axis=1)
    cnv = panel_flat - face_centroids[:, None, :]

    return state._replace(face_centroids=face_centroids, centroid_node_vectors=cnv)


