import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
jax.config.update("jax_enable_x64", True)

import sys
sys.path.append(os.path.abspath('.'))

import matplotlib.pyplot as plt
from dataclasses import dataclass

# Local imports
from topology.unit_patterns import unit_RDQK_D, unit_RDQK_0
from topology.builder import build_tessellation
from geometry.make_initial_map import compute_initial_map
from utils.visualization import plot_tessellation, animate_tessellation
from jax_backend.pytrees import create_jax_state
from optimization.solver import solve_form_finding_deployed, solve_form_finding_contracted
from geometry.target_shape import DEFAULT_TARGET
from geometry.get_contracted_shape import get_contracted_shape

# --- Configuration ---
@dataclass
class ExperimentConfig:
    width: int = 4
    height: int = 4
    pattern: callable = unit_RDQK_D
    initial_map_type: str = 'elliptical_grip'
    fix_contracted_boundary: bool = True
    rectangular_ratio: float = 1.0
    target_boundary: tuple = (DEFAULT_TARGET['type'], DEFAULT_TARGET['center'], DEFAULT_TARGET['radius'])
    animate_contraction: bool = True


if __name__ == "__main__":

    config = ExperimentConfig()

    target_params = {
        'type': config.target_boundary[0],
        'center': config.target_boundary[1],
        'radius': config.target_boundary[2]
    }

    print(f"Building tessellation ({config.width}x{config.height})...")
    tessellation = build_tessellation(config.pattern, config.width, config.height)
    print(f"-> {len(tessellation.vertices)} vertices, {len(tessellation.faces)} faces, {len(tessellation.hinges)} hinges, {len(tessellation.voids)} voids.")
    # Plot the initial tessellation
    # plot_tessellation(tessellation, 
    #     title="Initial Tessellation", 
    #     show_target=False,
    #     show_indices=False, 
    #     show_vertices=False, 
    #     show_hinges=False)
    # plt.show()


    print(f"Applying initial map: {config.initial_map_type}...")
    mapped_tessellation = compute_initial_map(
        tessellation, 
        config.target_boundary, 
        map_type=config.initial_map_type, 
        scale_factor=1.0)
    # Plot the mapped tessellation
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
    
    # Calculate border edge rest lengths from the UNMAPPED initial tessellation
    # Proportionality rule: global scaling is inversely proportional 
    # to grid size so the mesh fits in the same target shape.
    # Calibrated with base_alpha (e.g. 1.0 gives 0.5 for a 2x2 grid)
    base_alpha = 0.5
    alpha_border = base_alpha / max(config.width, config.height)
    print(f"Dynamically calculated alpha_border: {alpha_border} (base_alpha={base_alpha}, grid={config.width}x{config.height})")
    
    tess_dict['border_edges_rest_lengths_sq'] = tessellation.compute_border_edges_lengths_sq(alpha=alpha_border)
    
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

    # Define target parameters based on the central configuration
    print("\nStarting optimization...")
    # Save the initial mapped tessellation for comparison
    initial_map_tessellation = mapped_tessellation.copy()
    
    # Optimization
    optimized_state, result = solve_form_finding_deployed(tessellation_state, target_params, max_iter=500)
    mapped_tessellation.update_vertices(optimized_state.X)
    print("Optimization finished.")

    from utils.metrics import compute_tessellation_differences
    diff_areas, diff_ratios = compute_tessellation_differences(initial_map_tessellation, mapped_tessellation)

    print("\n--- Optimization Evaluation ---")
    avg_diff_area = sum(diff_areas) / len(diff_areas)
    avg_diff_ratio = sum(diff_ratios) / len(diff_ratios)
    print(f"Average Face Area Difference: {avg_diff_area:.2f}%")
    print(f"Average Face Ratio Difference: {avg_diff_ratio:.2f}%")
    print(f"Max Face Area Difference: {max(diff_areas):.2f}%")
    print(f"Max Face Ratio Difference: {max(diff_ratios):.2f}%")
    print("-------------------------------")

    # Visualization of the deployed shape (optimized) with deformations
    from utils.visualization import plot_tessellation_differences
    plot_tessellation_differences(
        mapped_tessellation,
        diff_areas,
        title="Deployed Shape (Area Deformation)",
        show_target=True,
        show_indices=False,
        show_vertices=False,
        show_hinges=False,
        target_params=target_params
    )
    plt.show()


    print("\nCalculating contracted shape using optimization...")
    # VERY IMPORTANT: Update face rest lengths so that 
    # the contracted solver tries to maintain the DEPLOYED shape, not the mapped shape.
    from jax_backend.pytrees import compute_face_lengths_sq
    new_F_rest = compute_face_lengths_sq(optimized_state.X, optimized_state.F_idx)
    optimized_state = optimized_state._replace(F_rest_lengths_sq=new_F_rest)
    
    contracted_state, result, history = solve_form_finding_contracted(optimized_state, target_params, max_iter=500)
    contracted_tessellation = mapped_tessellation.copy()
    contracted_tessellation.update_vertices(contracted_state.X)
    print("Contraction finished.")

    if config.animate_contraction:
        print("\nCreating animation of the closing process...")
        animate_tessellation(
            contracted_tessellation, 
            history['states'], 
            filepath="closing_animation.gif", 
            fps=15,
            show_target=True, 
            show_indices=False, 
            show_vertices=False, 
            show_hinges=False,
            target_params=target_params,
            color_faces='orange'
        )

    # Visualization of the contracted shape
    plot_tessellation(contracted_tessellation, 
        title="Contracted Shape", 
        show_target=True, 
        show_indices=False, 
        show_vertices=False, 
        show_hinges=False,
        target_params=target_params,
        color_faces='orange')
    plt.show()

    # Plotting Energy History
    plt.figure(figsize=(8, 5))
    plt.plot(history['energy'], label='Internal Energy', color='orange', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Objective / Energy')
    plt.title('Internal Energy of the Tessellation during Closing')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
