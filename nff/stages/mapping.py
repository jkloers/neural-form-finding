"""
Stage 0 — Initial mapping in centroidal coordinates.

Two mapping engines are available:

1. Analytical (apply_mapping):
   Continuous parametric functions (conformal polynomials, asymmetric roots, …).
   Centroids are displaced by the map; CNVs are deformed by its exact Jacobian
   computed via jax.jacfwd. The Jacobian is fully differentiable.

2. GNN (apply_gnn_mapping):
   A graph neural network (EGNN, MPNN, or other) predicts new centroid positions
   and a per-face 2×2 local transformation matrix. No Jacobian is involved;
   the network has full control over local geometry.

Both engines share the same interface: CentroidalState → CentroidalState, pure JAX.
Adding a new mapping engine means implementing a new branch inside apply_gnn_mapping
or a new factory in build_mapping_fn — nothing else in the pipeline needs to change.
"""
import jax
import jax.numpy as jnp
from typing import Callable, Any, Union, Dict
from nff.stages.state import CentroidalState
from nff.stages.geometry import reconstruct_vertices
from nff.config.targets import get_target_points

def preprocess_to_complex(p_restricted, context, params=None):
    """Normalization and Shirley-Chiu projection."""
    if isinstance(params, dict):
        use_sc = params.get('use_shirley_chiu', context.get('use_shirley_chiu', True))
    else:
        use_sc = context.get('use_shirley_chiu', True)

    h_sizes = jnp.where(context['half_sizes'] == 0.0, 1.0, context['half_sizes'])
    normalized = (p_restricted - context['box_center']) / h_sizes
    u, v = normalized[0], normalized[1]
    
    x_disk = jnp.where(use_sc, u * jnp.sqrt(jnp.maximum(0.0, 1.0 - (v ** 2) / 2.0)), u)
    y_disk = jnp.where(use_sc, v * jnp.sqrt(jnp.maximum(0.0, 1.0 - (u ** 2) / 2.0)), v)
    
    return x_disk + 1j * y_disk

def postprocess_radial_fit(w, context, params=None):
    """Radial scaling, target adaptation, and translation."""
    tx, ty = 0.0, 0.0
    if isinstance(params, dict):
        tx = params.get('tx', 0.0)
        ty = params.get('ty', 0.0)
        
    w_final = w
    
    # Adaptation to target shape
    w_angle = jnp.angle(w_final)
    
    # Flags: strict_boundary_fit
    strict_fit = context.get('strict_boundary_fit', True)
    
    target_rad = jnp.where(
        strict_fit & (jnp.max(context['b_radii']) > 0.0),
        jnp.interp(w_angle, context['b_angles'], context['b_radii']),
        context.get('base_initial_radius', context['radius'])
    )

    # Application of learnable global scale
    scale_multiplier = 1.0
    if isinstance(params, dict) and 'log_scale' in params:
        scale_multiplier = jnp.exp(params['log_scale'])

    x_new = jnp.real(w_final) * target_rad * scale_multiplier + context['center'][0] + tx
    y_new = jnp.imag(w_final) * target_rad * scale_multiplier + context['center'][1] + ty
    
    return jnp.array([x_new, y_new])

def map_elliptical_grip(p_restricted, params, context):
    z = preprocess_to_complex(p_restricted, context, params)
    # Elliptical grip is essentially Shirley-Chiu with target radius
    return jnp.array([jnp.real(z), jnp.imag(z)]) * context['radius'] + context['center']

def map_homothetic(p_restricted, params, context):
    offset = p_restricted - context['box_center']
    return context['center'] + offset

def map_conformal_polynomial(z, params):
    """Pure mathematical core for conformal polynomial."""
    # Support both dictionary (new) and array (legacy) map_params
    if isinstance(params, dict):
        c_val = params.get('c_val', jnp.zeros(1))
    else:
        # Fallback to legacy array format (offset 4: tx, ty, theta, s_val)
        if params is None or params.shape[0] < 4:
            c_val = jnp.zeros(1)
        else:
            c_val = params[4:]
    
    w = z
    for k in range(c_val.shape[0]):
        power = 4 * (k + 1) + 1
        w = w + c_val[k] * (z ** power)
    return w

def map_asymmetric_roots(z, params):
    """Pure mathematical core for asymmetric roots mapping."""
    if params is None or (isinstance(params, jnp.ndarray) and params.shape[0] < 6):
        roots_flat = jnp.array([10.0, 0.0])
        weights = jnp.array([1.0])
    elif isinstance(params, dict):
        roots_flat = params.get('roots', jnp.array([10.0, 0.0]))
        weights = params.get('weights', jnp.ones(roots_flat.shape[0] // 2))
    else:
        # Legacy offset 4
        rem = params.shape[0] - 4
        if rem % 3 == 0:
            n_roots = rem // 3
            roots_flat = params[4 : 4 + 2 * n_roots]
            weights = params[4 + 2 * n_roots : ]
        else:
            n_roots = rem // 2
            roots_flat = params[4:]
            weights = jnp.ones(n_roots)
            
    roots = roots_flat[0::2] + 1j * roots_flat[1::2]
    
    coeffs = jnp.array([1.0 + 0j])
    for i in range(roots.shape[0]):
        shifted = jnp.pad(coeffs, (1, 0))
        coeffs = jnp.pad(coeffs, (0, 1)) - (weights[i] / roots[i]) * shifted
        
    k = jnp.arange(1, len(coeffs) + 1)
    integrated_coeffs = coeffs / k
    integrated_coeffs = jnp.pad(integrated_coeffs, (1, 0)) 
    
    return jnp.polyval(integrated_coeffs[::-1], z)

def map_boundary_projection(p_restricted, params, context):
    offset = p_restricted - context['box_center']
    p_angle = jnp.arctan2(offset[1], offset[0])
    p_norm = jnp.linalg.norm(offset)
    
    w_box = context['half_sizes'][0]
    h_box = context['half_sizes'][1]
    norm_max_tess = 1.0 / jnp.maximum(
        jnp.abs(jnp.cos(p_angle)) / w_box,
        jnp.abs(jnp.sin(p_angle)) / h_box)
        
    target_boundary_radius = jnp.interp(p_angle, context['b_angles'], context['b_radii'])
    scale_rad = jnp.where(p_norm > 0, target_boundary_radius / norm_max_tess, 0.0)
    
    mapped = context['shape_center'] + offset * scale_rad
    return mapped

def build_mapping_fn(
        state: CentroidalState,
        target_params: dict,
        map_type: str = 'elliptical_grip',
        domain_restriction: float = 0.8,
        use_shirley_chiu: bool = True,
        strict_boundary_fit: bool = True) -> Callable:
    """Factory function to build a modular JAX mapping pipeline."""
    c = state.face_centroids
    s = state.centroid_node_vectors
    n_faces, max_nodes, dim = s.shape

    # 1. Gather global context for the mapping (bounding box)
    all_vertices = reconstruct_vertices(c, s)
    vertices_flat = all_vertices.reshape(-1, dim)
    
    min_xy = jnp.min(vertices_flat, axis=0)
    max_xy = jnp.max(vertices_flat, axis=0)
    box_center = (min_xy + max_xy) / 2.0
    half_sizes = (max_xy - min_xy) / 2.0
    max_half_size = jnp.maximum(jnp.max(half_sizes), 1e-6)

    shape_type = target_params.get('type', 'circle')
    center = jnp.asarray(target_params.get('center', [0.0, 0.0]), dtype=float)
    radius = float(target_params.get('radius', 1.0))

    # Pre-computation for boundary_projection or shape adaptation
    boundary_pts = jnp.asarray(get_target_points(target_params, n_points=500), dtype=float)
    shape_center = jnp.mean(boundary_pts, axis=0)
    boundary_vec = boundary_pts - shape_center
    boundary_angles = jnp.arctan2(boundary_vec[:, 1], boundary_vec[:, 0])
    boundary_radii = jnp.linalg.norm(boundary_vec, axis=1)

    order = jnp.argsort(boundary_angles)
    b_angles = boundary_angles[order]
    b_radii = boundary_radii[order]
    b_angles = jnp.concatenate([b_angles - 2 * jnp.pi, b_angles, b_angles + 2 * jnp.pi])
    b_radii = jnp.tile(b_radii, 3)

    # 2. Context dictionary passed to pure maps
    context = {
        'box_center': box_center,
        'half_sizes': half_sizes,
        'max_half_size': max_half_size,
        'center': center,
        'radius': radius,
        'b_angles': b_angles,
        'b_radii': b_radii,
        'shape_center': shape_center,
        'use_shirley_chiu': use_shirley_chiu,
        'strict_boundary_fit': strict_boundary_fit,
        'base_initial_radius': jnp.mean(boundary_radii) if len(boundary_radii) > 0 else radius
    }

    # 3. Create the generic wrapper
    def mapping_fn(p, map_params=None):
        if map_params is None:
            map_params = {}
            
        # A. Pre-processing: Domain restriction
        p_restricted = context['box_center'] + (p - context['box_center']) * domain_restriction
        
        # B. Core logic branching
        if map_type in ['conformal_polynomial', 'asymmetric_roots']:
            # Pipeline: Pre-process -> Complex Core -> Post-process
            z = preprocess_to_complex(p_restricted, context, map_params)
            
            if map_type == 'conformal_polynomial':
                w = map_conformal_polynomial(z, map_params)
            else:
                w = map_asymmetric_roots(z, map_params)
                
            mapped_p = postprocess_radial_fit(w, context, map_params)
        else:
            # Classic leaf-style maps
            if map_type == 'elliptical_grip':
                mapped_p = map_elliptical_grip(p_restricted, map_params, context)
            elif map_type == 'boundary_projection':
                mapped_p = map_boundary_projection(p_restricted, map_params, context)
            elif map_type == 'homothetic':
                mapped_p = map_homothetic(p_restricted, map_params, context)
            else:
                mapped_p = p_restricted
        
        # C. Global scale (learn_global_scale = True): differentiable zoom/shrink
        # centered on the target shape center.  When log_scale is absent (or 0)
        # this is an exact no-op (exp(0) = 1).
        if 'log_scale' in map_params:
            log_s = map_params['log_scale']
            shape_center = context['center']
            mapped_p = shape_center + jnp.exp(log_s) * (mapped_p - shape_center)

        return mapped_p

    return mapping_fn

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


def apply_gnn_mapping(
        state: CentroidalState,
        gnn_params: dict,
        static_features: dict,
        map_type: str = 'gnn_dummy',
) -> CentroidalState:
    """Apply a GNN-based mapping to a CentroidalState.

    Unlike the analytical mapping (which uses an explicit Jacobian), the GNN
    directly controls the full geometry. It predicts:
      1. New centroid positions (x, y).
      2. A full 2×2 per-face transformation matrix.

    The CNVs are transformed by this matrix, allowing the network to stretch and
    shear tiles asymmetrically to close gaps.

    Args:
        state:           Flat CentroidalState (initial tessellation).
        gnn_params:      GNN weight dict PyTree.
        static_features: Dict returned by build_static_graph_features.
        map_type:        'gnn_dummy', 'gnn_egnn', or 'gnn_mpnn'.

    Returns:
        CentroidalState with updated face_centroids and centroid_node_vectors.
    """
    from nff.models.graph_builder import state_to_graph

    graph = state_to_graph(state, static_features)
    h = graph.nodes['h']
    x = graph.nodes['x']
    senders_np   = static_features['senders']
    receivers_np = static_features['receivers']
    n_faces      = static_features['n_nodes']

    if map_type == 'gnn_egnn':
        from nff.models.egnn import apply_egnn
        num_layers = static_features.get('num_layers') or sum(
            1 for k in gnn_params if k.endswith('_phi_x_W'))
        new_centroids, _, local_transform = apply_egnn(
            gnn_params, h, x, senders_np, receivers_np, n_faces, num_layers)
    elif map_type == 'gnn_mpnn':
        from nff.models.mpnn import apply_mpnn
        num_layers  = static_features.get('num_layers') or sum(
            1 for k in gnn_params if k.endswith('_phi_e_W1'))
        inner_depth = static_features.get('inner_depth', 2)
        new_centroids, _, local_transform = apply_mpnn(
            gnn_params, h, x, senders_np, receivers_np, n_faces, num_layers, inner_depth)
    else:  # gnn_dummy (default) — identity-like pass-through
        from nff.models.dummy import apply_dummy_gnn
        new_centroids = apply_dummy_gnn(
            gnn_params,
            h=h,
            x=x,
            edges=graph.edges,
            senders_np=senders_np,
            receivers_np=receivers_np,
            n_faces=n_faces,
        )
        return state._replace(face_centroids=new_centroids)

    # ── Transformation des CNVs (Centroid Node Vectors) ───────────────────────────
    # Le GNN prédit directement la matrice de transformation 2x2 pour chaque face.
    # Cette matrice gère la rotation, l'étirement (scale) asymétrique et le cisaillement.
    # À l'initialisation, cette matrice vaut l'Identité, donc les tuiles ne sont pas déformées.
    cnvs = state.centroid_node_vectors  # (n_faces, max_nodes, 2)
    new_cnvs = jnp.einsum('fab,fnb->fna', local_transform, cnvs)

    return state._replace(face_centroids=new_centroids, centroid_node_vectors=new_cnvs)


def apply_direct_transform_mapping(state: CentroidalState, map_params: dict) -> CentroidalState:
    """Stage 0 direct-transform parameterization.

    Mirrors the GNN output structure without message passing:
        map_params = {
            'face_centroids':   (n_faces, 2),     — directly optimized positions
            'local_transforms': (n_faces, 2, 2),  — per-face linear deformation of initial shapes
        }

    CNVs are derived as local_transforms[f] @ initial_cnvs[f, n], identical to
    the GNN line:  new_cnvs = einsum('fab,fnb->fna', local_transform, cnvs)

    Initialized to scale*I so the output equals the scaled flat tessellation.
    Adjacent faces receive correlated gradients, keeping hinge gaps small.
    """
    new_cnvs = jnp.einsum('fab,fnb->fna', map_params['local_transforms'],
                           state.centroid_node_vectors)
    return state._replace(
        face_centroids=map_params['face_centroids'],
        centroid_node_vectors=new_cnvs,
    )


def init_direct_transform_params(state: CentroidalState, target_params: dict) -> dict:
    """Initialize direct-transform parameters from the flat tessellation.

    Produces the same initial output as init_direct_vertices_params:
    centered/scaled centroids and local_transforms = scale*I so that
    new_cnvs = scale * initial_cnvs.

    Args:
        state:         CentroidalState of the centered flat tessellation.
        target_params: dict with keys 'center' and 'radius'.

    Returns:
        dict with 'face_centroids' (n_faces, 2) and 'local_transforms' (n_faces, 2, 2).
    """
    center = jnp.asarray(target_params.get('center', [0.0, 0.0]), dtype=float)
    radius = float(target_params.get('radius', 1.0))
    n_faces = state.face_centroids.shape[0]

    tess_centroid = jnp.mean(state.face_centroids, axis=0)
    all_vertices = reconstruct_vertices(state.face_centroids, state.centroid_node_vectors)
    vertices_flat = all_vertices.reshape(-1, 2)
    current_max_dist = jnp.max(jnp.linalg.norm(vertices_flat - tess_centroid, axis=-1))
    scale = radius / jnp.maximum(current_max_dist, 1e-6)

    centered_centroids = (state.face_centroids - tess_centroid) * scale + center
    # scale*I: initial output = scale * initial_cnvs, matching direct_vertices init
    identity_transforms = jnp.tile(jnp.eye(2, dtype=float) * scale, (n_faces, 1, 1))

    return {
        'face_centroids':   centered_centroids,
        'local_transforms': identity_transforms,
    }


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


def init_direct_vertices_params(state: CentroidalState, target_params: dict) -> dict:
    """Initialize direct-vertex parameters from the flat tessellation.

    Centers the tessellation on the target shape and scales it so the
    bounding radius matches the target radius — the same initialization
    used for GNN-based maps.

    Args:
        state:         CentroidalState of the flat (undeformed) tessellation.
        target_params: dict with keys 'center' (2-vector) and 'radius' (scalar).

    Returns:
        dict with 'face_centroids' and 'centroid_node_vectors' as JAX arrays.
    """
    center = jnp.asarray(target_params.get('center', [0.0, 0.0]), dtype=float)
    radius = float(target_params.get('radius', 1.0))

    tess_centroid = jnp.mean(state.face_centroids, axis=0)

    all_vertices = reconstruct_vertices(state.face_centroids, state.centroid_node_vectors)
    vertices_flat = all_vertices.reshape(-1, 2)
    current_max_dist = jnp.max(jnp.linalg.norm(vertices_flat - tess_centroid, axis=-1))
    scale = radius / jnp.maximum(current_max_dist, 1e-6)

    centered_centroids = (state.face_centroids - tess_centroid) * scale + center
    scaled_cnvs = state.centroid_node_vectors * scale

    return {
        'face_centroids': centered_centroids,
        'centroid_node_vectors': scaled_cnvs,
    }


def apply_mapping(
        state: CentroidalState,
        mapping_fn: Callable,
        map_params: Any = None) -> CentroidalState:
    """Apply a generic mapping function to a CentroidalState using Rigid-Face generalized mapping.
    
    Args:
        state: CentroidalState with flat tessellation geometry.
        mapping_fn: callable f(p, params) that maps a point R^2 -> R^2.
        map_params: parameters for the parameterized map (e.g. polynomial coefficients).
        
    Returns:
        CentroidalState with updated (face_centroids, centroid_node_vectors).
    """
    c = state.face_centroids
    s = state.centroid_node_vectors
    
    # 0. Parse parameters if they are in raw format
    params = parse_map_params(map_params) if map_params is not None else None

    # 1. Bind parameters to create a function purely of p
    f_point_fn = lambda p: mapping_fn(p, params)

    # 2. Compute Jacobian matrix function using JAX
    jac_f_fn = jax.jacfwd(f_point_fn)
    
    # 3. Vectorize across all centroids
    f_vmap_fn = jax.vmap(f_point_fn)
    jac_vmap_fn = jax.vmap(jac_f_fn)
    
    # 4. Map centroids
    c_new = f_vmap_fn(c)
    
    # 5. Transform CNVs using the Jacobian
    jac_matrices = jac_vmap_fn(c)
    s_new = jnp.einsum('fab,fnb->fna', jac_matrices, s)

    return state._replace(
        face_centroids=c_new,
        centroid_node_vectors=s_new,
    )
