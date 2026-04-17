import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize
import sys
import os

from topology.core import Tessellation
from topology.unit_patterns import unit_RDQK_D 
from topology.builder import build_tessellation
from geometry.make_initial_map import compute_initial_map
from utils.visualization import plot_tessellation


sys.path.append(os.path.abspath('.'))

# --- Parameters ---
# Kirigami pattern and target shape parameters
width = 4
height = 4
initial_map_type = 'elliptical_grip' # Options: 'scaled', 'rescaled', 'elliptical_grip'
fix_contracted_boundary_shape = True
rectangular_ratio = 1 # 1 = carré

# --- Building Tessellation ---
pattern = unit_RDQK_D
tessellation = build_tessellation(pattern, width, height)
print(f"Tessellation built with {len(tessellation.vertices)} vertices, {len(tessellation.faces)} faces, and {len(tessellation.hinges)} hinges.")

target_boundary = ('circle', [0.0, 0.0], 1.0)
mapped_tessellation = compute_initial_map(tessellation, target_boundary, map_type=initial_map_type, scale_factor= 1.0)
print("Initial mapping completed.")

plt.figure(figsize=(10, 10))
plot_tessellation(mapped_tessellation)
plt.show(block=False)

# --- Optimisation sous contraintes ---
# Note: L'optimisation nécessite l'adaptation des fonctions objectives et contraintes
# pour travailler avec la nouvelle structure Tessellation.
# Pour l'instant, nous gardons la logique mais commentons l'optimisation.

# # Préparation des données pour l'optimisation
# # (Adapter find_boundary_points, etc., pour Tessellation)
# # ...

# # Fonctions objectif et contraintes (à adapter)
# def objective(x):
#     # Adapter OBJ_regularization pour Tessellation
#     pass

# def constraints(x):
#     # Adapter all_constraint_residual_and_jacobian
#     pass

# # Lancement de l'optimisation
# # x0 = compose_v(mapped_tessellation.vertices)
# # res = minimize(objective, x0, method='trust-constr', constraints=cons, ...)
# # solved_pointsD = decompose_v(res.x)

# # Obtention de la structure contractée finale
# # solved_points0 = get_contracted_shape(solved_pointsD, ...)

# # Sauvegarde des résultats
# # write_mesh_generic(...)

