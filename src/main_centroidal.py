"""
main_centroidal.py — Unified centroidal form-finding pipeline.

Pipeline:
    Tessellation (numpy)
        → to_centroidal_state()
        → CentroidalState (JAX)
        → Stage 0: Initial Mapping (→ GNN later)
        → Stage 1: Geometric Validity Optimization
        → Stage 2: Static Physics Solver
        → SolutionData
"""

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
jax.config.update("jax_enable_x64", True)

import sys
sys.path.append(os.path.abspath('.'))

import jax.numpy as jnp
from jax import vmap
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass

# Topology
from topology.unit_patterns import unit_RDQK_D
from topology.builder import build_tessellation

# Target shape
from geometry.target_shape import get_target_points, DEFAULT_TARGET

# Centroidal pipeline
from jax_backend.centroidal.state import CentroidalState
from jax_backend.centroidal.geometry import reconstruct_vertices
from jax_backend.centroidal.pipeline import forward_pipeline
from jax_backend.physics_solver.kinematics import rotation_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CentroidalConfig:
    # Tessellation
    width: int = 2
    height: int = 2
    pattern: callable = unit_RDQK_D

    # Target shape
    target_type: str = DEFAULT_TARGET['type']
    target_center: tuple = (0.0, 0.0)
    target_radius: float = DEFAULT_TARGET['radius']

    # Material properties
    k_stretch: float = 10.0
    k_shear: float = 5.0
    k_rot: float = 1.0
    density: float = 1.0

    # Initial mapping
    map_type: str = 'elliptical_grip'
    scale_factor: float = 1.0

    # Geometric validity weights
    w_connectivity: float = 700.0
    w_non_intersection: float = 1000.0
    w_target: float = 1.0
    w_arm_symmetry: float = 1.0

    # Physics
    use_contact: bool = True
    linearized_strains: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    config = CentroidalConfig()

    target_params = {
        'type': config.target_type,
        'center': list(config.target_center),
        'radius': config.target_radius,
    }

    # ══════════════════════════════════════════════════════════════════════════
    # 1. Build tessellation (numpy)
    # ══════════════════════════════════════════════════════════════════════════
    print(f"Building tessellation ({config.width}x{config.height})...")
    tessellation = build_tessellation(config.pattern, config.width, config.height)
    print(f"  {len(tessellation.vertices)} vertices, "
          f"{len(tessellation.faces)} faces, "
          f"{len(tessellation.hinges)} hinges, "
          f"{len(tessellation.voids)} voids.")

    # ══════════════════════════════════════════════════════════════════════════
    # 2. Configure boundary conditions & material properties
    # ══════════════════════════════════════════════════════════════════════════
    # Material
    tessellation.set_hinge_properties(
        k_stretch=config.k_stretch,
        k_shear=config.k_shear,
        k_rot=config.k_rot)
    tessellation.set_all_faces_properties(density=config.density)

    # Dirichlet BCs: clamp boundary faces
    clamped_ids = tessellation.clamp_boundary_faces()
    print(f"  Clamped {len(clamped_ids)} boundary faces.")

    # Neumann BCs: example — downward force on a central interior face
    all_ids = set(range(len(tessellation.faces)))
    interior_ids = sorted(all_ids - set(clamped_ids))
    if interior_ids:
        loaded_face = interior_ids[len(interior_ids) // 2]
        tessellation.set_face_load(loaded_face, dof_id=1, value=-1.0)
        print(f"  Applied F_y = -1.0 on face {loaded_face}.")

    # ══════════════════════════════════════════════════════════════════════════
    # 3. Export to CentroidalState
    # ══════════════════════════════════════════════════════════════════════════
    print("\nExporting to CentroidalState...")
    cs_dict = tessellation.to_centroidal_state()
    initial_state = CentroidalState(**{k: jnp.array(v) for k, v in cs_dict.items()})

    n_faces = initial_state.face_centroids.shape[0]
    n_hinges = initial_state.hinge_face_pairs.shape[0] // 2  # 2 entries per hinge
    print(f"  n_faces = {n_faces}")
    print(f"  n_hinges = {n_hinges}")
    print(f"  centroid_node_vectors: {initial_state.centroid_node_vectors.shape}")
    print(f"  hinge_node_pairs:      {initial_state.hinge_node_pairs.shape}")
    print(f"  hinge_adj_info:        {initial_state.hinge_adj_info.shape}")
    print(f"  boundary_face_nodes:   {initial_state.boundary_face_node_ids.shape}")
    print(f"  constrained DOFs:      {initial_state.constrained_face_DOF_pairs.shape[0]}")
    print(f"  loaded DOFs:           {initial_state.loaded_face_DOF_pairs.shape[0]}")

    # ══════════════════════════════════════════════════════════════════════════
    # 4. Run the full pipeline
    # ══════════════════════════════════════════════════════════════════════════
    geom_weights = {
        'connectivity': config.w_connectivity,
        'non_intersection': config.w_non_intersection,
        'target': config.w_target,
        'arm_symmetry': config.w_arm_symmetry,
    }

    print("\n" + "=" * 60)
    print("CENTROIDAL PIPELINE")
    print("  Stage 0: Initial Mapping")
    print("  Stage 1: Geometric Validity")
    print("  Stage 2: Static Physics Solver")
    print("=" * 60)

    result = forward_pipeline(
        initial_state=initial_state,
        target_params=target_params,
        map_type=config.map_type,
        scale_factor=config.scale_factor,
        geom_weights=geom_weights,
        use_contact=config.use_contact,
        linearized_strains=config.linearized_strains,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 5. Results & Visualization
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 60)
    print("RESULTS VISUALIZATION")
    print("-" * 60)

    from utils.visualization import plot_tessellation
    import copy

    def plot_stage(state, title):
        # 1. Reconstruct vertices from centroidal state
        c = state.face_centroids
        s = state.centroid_node_vectors
        verts_rec = reconstruct_vertices(c, s) # (n_faces, max_nodes, 2)
        
        # 2. Update a copy of the tessellation
        tess_copy = copy.deepcopy(tessellation)
        new_verts = np.zeros_like(tess_copy.vertices)
        for i, face in enumerate(tess_copy.faces):
            for j, v_idx in enumerate(face.vertex_indices):
                new_verts[v_idx] = verts_rec[i, j]
        tess_copy.update_vertices(new_verts)
        
        # 3. Plot
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_tessellation(tess_copy, ax=ax, title=title, 
                          show_target=True, target_params=target_params)
        plt.show()

    # Stage 0
    print("Displaying Stage 0: Initial Mapping...")
    plot_stage(result['mapped_state'], "Stage 0: Initial Mapping")

    # Stage 1
    print("Displaying Stage 1: Geometric Validity...")
    plot_stage(result['valid_state'], "Stage 1: Geometric Validity")

    # Stage 2
    print("Displaying Stage 2: Static Equilibrium...")
    # Reconstruct equilibrium state from solution
    sol = result['solution']
    valid_state = result['valid_state']
    
    # centroids_eq = centroids_valid + displacement
    c_eq = valid_state.face_centroids + sol.fields[:, :2]
    # s_eq = rotate(s_valid, theta)
    R = vmap(rotation_matrix)(sol.fields[:, 2])
    s_eq = jnp.einsum('nij, nkj -> nki', R, valid_state.centroid_node_vectors)
    
    equilibrium_state = valid_state._replace(face_centroids=c_eq, centroid_node_vectors=s_eq)
    plot_stage(equilibrium_state, "Stage 2: Static Equilibrium")

    print("\nPipeline complete.")
