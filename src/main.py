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
from optimization.solver import solve_form_finding

# --- Configuration ---
@dataclass
class ExperimentConfig:
    width: int = 2
    height: int = 2
    pattern: callable = unit_RDQK_D
    initial_map_type: str = 'elliptical_grip'
    fix_contracted_boundary: bool = True
    rectangular_ratio: float = 1.0
    target_boundary: tuple = ('circle', [0.0, 0.0], 1.0)

# --- Builder ---
def setup_kirigami_problem(config: ExperimentConfig):

    print(f"Building tessellation ({config.width}x{config.height})...")
    tessellation = build_tessellation(config.pattern, config.width, config.height)
    print(f"-> {len(tessellation.vertices)} vertices, {len(tessellation.faces)} faces, {len(tessellation.hinges)} hinges, {len(tessellation.voids)} voids.")

    print(f"Applying initial map: {config.initial_map_type}...")
    mapped_tessellation = compute_initial_map(
        tessellation, 
        config.target_boundary, 
        map_type=config.initial_map_type, 
        scale_factor=1.0
    )
    return mapped_tessellation


if __name__ == "__main__":
    # Initialisation de la configuration
    config = ExperimentConfig()

    # Génération
    mapped_tessellation = setup_kirigami_problem(config)

    # Conversion JAX
    print("\nCreating JAX PyTree state for optimization...")
    tess_dict = mapped_tessellation.to_jax_state()
    tessellation_state = create_jax_state(tess_dict)
    print("Tessellation State Dimensions:")
    print("Vertices (X):", tessellation_state.X.shape)
    print("Face Indices (F_idx):", tessellation_state.F_idx.shape)
    print("Hinge Adjacent Vertices (E_adjacent):", tessellation_state.E_adjacent.shape)
    print("Rest Angles (A_rest):", tessellation_state.A_rest.shape)
    print("Hinge Stiffness (H_stiffness):", tessellation_state.H_stiffness.shape)
    print("Hinge Vertex Connections (V_connect):", tessellation_state.V_connect.shape)
    print("Anchor Indices (Anch_indices):", tessellation_state.Anch_indices.shape)
    print("Opposite Edges (E_opp):", tessellation_state.E_opp.shape)

    # Définition des paramètres cibles
    target_params = {
        'radius': 1.0
    }

    print("\nStarting optimization...")
    # Optimisation
    optimized_state, result = solve_form_finding(tessellation_state, target_params, max_iter=500)
    print("Optimization finished.")
    
    # Visualization
    plt.figure(figsize=(10, 10))
    # On met à jour l'objet Tessellation avec les nouvelles positions optimisées
    mapped_tessellation.update_vertices(optimized_state.X)
    plot_tessellation(mapped_tessellation)
    plt.show()
    
    # Prochaines étapes :
    # energy = compute_total_energy(state)
    # optimized_state = run_optimization(state)
