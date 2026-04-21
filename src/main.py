import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
jax.config.update("jax_enable_x64", True)

import sys
sys.path.append(os.path.abspath('.'))

import matplotlib.pyplot as plt
from dataclasses import dataclass

# Imports locaux
from topology.unit_patterns import unit_RDQK_D, unit_RDQK_0
from topology.builder import build_tessellation
from geometry.make_initial_map import compute_initial_map
from utils.visualization import plot_tessellation
from jax_backend.pytrees import create_jax_state
from optimization.solver import solve_form_finding_deployed, solve_form_finding_contracted
from geometry.target_shape import DEFAULT_TARGET
from geometry.get_contracted_shape import get_contracted_shape

# --- Configuration ---
@dataclass
class ExperimentConfig:
    width: int = 2
    height: int = 2
    pattern: callable = unit_RDQK_D
    initial_map_type: str = 'elliptical_grip'
    fix_contracted_boundary: bool = True
    rectangular_ratio: float = 1.0
    target_boundary: tuple = (DEFAULT_TARGET['type'], DEFAULT_TARGET['center'], DEFAULT_TARGET['radius'])


if __name__ == "__main__":

    config = ExperimentConfig()

    print(f"Building tessellation ({config.width}x{config.height})...")
    tessellation = build_tessellation(config.pattern, config.width, config.height)
    print(f"-> {len(tessellation.vertices)} vertices, {len(tessellation.faces)} faces, {len(tessellation.hinges)} hinges, {len(tessellation.voids)} voids.")


    # Plot the initial tessellation
    plot_tessellation(tessellation, 
        title="Initial Tessellation", 
        show_target=False,
        show_indices=False, 
        show_vertices=False, 
        show_hinges=False)
    plt.show()


    print(f"Applying initial map: {config.initial_map_type}...")
    mapped_tessellation = compute_initial_map(
        tessellation, 
        config.target_boundary, 
        map_type=config.initial_map_type, 
        scale_factor=1.0)

        # Plot the initial tessellation
    plot_tessellation(mapped_tessellation, 
        title="Mapped Tessellation", 
        show_target=True,
        show_indices=False, 
        show_vertices=False, 
        show_hinges=False)
    plt.show()


    # JAX PyTree state for optimization
    print("\nCreating JAX PyTree state for optimization...")
    tess_dict = mapped_tessellation.to_jax_state()
    tessellation_state = create_jax_state(tess_dict)
    print("Tessellation State Dimensions:")
    print("Vertices (X):", tessellation_state.X.shape)
    print("Face Indices (F_idx):", tessellation_state.F_idx.shape)
    print("Hinge Adjacent Vertices (E_adjacent):", tessellation_state.E_adjacent.shape)
    print("Rest Angles (A_rest):", tessellation_state.A_rest.shape)
    print("Hinge Angular Stiffness (H_ang):", tessellation_state.H_angular_stiffness.shape)
    print("Hinge Linear Stiffness (H_lin):", tessellation_state.H_linear_stiffness.shape)
    print("Hinge Vertex Connections (V_connect):", tessellation_state.V_connect.shape)
    print("boundary Indices (Boundary_indices):", tessellation_state.Boundary_indices.shape)
    print("Opposite Edges (E_opp):", tessellation_state.E_opp.shape)

    # Définition des paramètres cibles basés sur la config centrale
    target_params = {
        'radius': DEFAULT_TARGET['radius']
    }

    print("\nStarting optimization...")
    # Optimisation
    optimized_state, result = solve_form_finding_deployed(tessellation_state, target_params, max_iter=500)
    mapped_tessellation.update_vertices(optimized_state.X)
    print("Optimization finished.")

    # Visualisation de la forme déployée (optimisée)
    plot_tessellation(mapped_tessellation, 
        title="Deployed Shape", 
        show_target=True, 
        show_indices=False, 
        show_vertices=False, 
        show_hinges=False,
        color_faces='#2ECC71')
    plt.show()

    print("\nCalculating contracted shape using optimization...")
    # On repart de l'état déployé pour trouver la forme contractée
    contracted_state, result = solve_form_finding_contracted(optimized_state, target_params, max_iter=500)
    
    contracted_tessellation = mapped_tessellation.copy()
    contracted_tessellation.update_vertices(contracted_state.X)
    print("Contraction finished.")

    # Visualisation de la forme contractée (plus épurée)
    plot_tessellation(contracted_tessellation, 
        title="Contracted Shape", 
        show_target=False, 
        show_indices=False, 
        show_vertices=False, 
        show_hinges=False,
        color_faces='#2ECC71')
    plt.show()
