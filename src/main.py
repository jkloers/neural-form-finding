import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize
import sys
import os

from geometry.make_tessalation_generic import make_tessellation_generic
from geometry.make_initial_map import compute_initial_map


# Ajout du répertoire courant au path
sys.path.append(os.path.abspath('.'))

# Définition de la taille du motif de kirigami
width = 4
height = 4

# Choix de la forme 2D cible déployée depuis une bibliothèque prédéfinie
shapes = ['circle', 'egg', 'rainbow', 'anvil', 'shear', 'wedge', 'star', 'wavy', 'rect']
shape_name = shapes[0]

# Définition du type de carte initiale pour l'optimisation
# 1: configuration déployée standard
# 2: configuration déployée standard avec redimensionnement (paramètre optionnel scale_factor)
# 3: carte conforme (Schwarz-Christoffel)
# 4: carte de Teichmuller (Meng et al., SIIMS 2016)
initial_map_type = 3
scale_factor = None 

# Exiger que le motif contracté soit rectangulaire ? 0 (non) / 1 (oui)
fix_contracted_boundary_shape = 1

# Spécifier le ratio largeur/hauteur si le motif doit être rectangulaire
# 0 (non applicable) / autre nombre positif (ratio prescrit)
rectangular_ratio = 1 # 1 = carré

# --- Construction de la tessellation et estimation initiale ---
unit_dir = 'unit_cell_scripts/square/'

# Construction de la tessellation (Nécessitera l'adaptation des indices 1-based de MATLAB en 0-based)

tesselation = make_tessellation_generic(unit_dir, width, height, False)

# (pointsD_standard, edgesD, edge_pairsD, anglesD, ringsD, face_setsD, free,
#  path_adjs, intervals, Dto0, unitfacesD, cornersD, overlapD, points0) = make_tessellation_generic(unit_dir, width, height, False)

# Création des ensembles de faces initiaux
face_sets0 = [Dto0[face] for face in face_setsD]

# Identification des points et angles sur les frontières
boundR, boundT, boundL, boundB = find_boundary_points(pointsD_standard, free)

if fix_contracted_boundary_shape:
    boundary_rings = find_boundary_rings(pointsD_standard, anglesD, Dto0, free, cornersD)
else:
    boundary_rings = []
    rectangular_ratio = 0

# Recherche de la frontière réelle
ymin = np.min(points0[:, 1])
xval = np.sort(np.unique(points0[:, 0]))
xmax = xval[-1]

boundRD = np.where(points0[Dto0, 0] == xmax)[0]
boundBD = np.where(points0[Dto0, 1] == ymin)[0]

# Recherche des arêtes de frontière dans edgesD
mask_bottom = np.isin(edgesD[:, 0], boundBD) & np.isin(edgesD[:, 1], boundBD)
edges_bottom = edgesD[mask_bottom]

mask_right = np.isin(edgesD[:, 0], boundRD) & np.isin(edgesD[:, 1], boundRD)
edges_right = edgesD[mask_right]

# Chargement de la forme cible (Supposons que les fonctions soient importées ou définies localement)
shape_func = globals()[shape_name]
spline_boundR, spline_boundT, spline_boundL, spline_boundB = shape_func()

# Construction de l'estimation initiale dans l'espace déployé
pointsD = compute_initial_map(pointsD_standard, shape_name, initial_map_type,
                              scale_factor, boundR, boundT, boundL, boundB)

# Affichage de l'estimation initiale
plt.figure(4)
plt.axis('equal')
plt.axis('off')
plot_faces_generic(pointsD, face_setsD, 4)

plt.plot(pointsD[boundR, 0], pointsD[boundR, 1], 'or')
plt.plot(pointsD[boundT, 0], pointsD[boundT, 1], 'og')
plt.plot(pointsD[boundL, 0], pointsD[boundL, 1], 'ob')
plt.plot(pointsD[boundB, 0], pointsD[boundB, 1], 'oy')
plt.title('Initial guess in the deployed space')
plt.show(block=False)

# --- Optimisation sous contraintes ---

same_face_adjs = find_smoothing_faces(unitfacesD, width, height)

boundary_nodes_cell = [boundR, boundT, boundL, boundB]
boundary_target_splines_cell = [spline_boundR, spline_boundT, spline_boundL, spline_boundB]

# Préparation de l'optimisation (L'équivalent de fmincon nécessite une formulation spécifique des contraintes)
x0 = compose_v(pointsD)

def objective(x):
    return OBJ_regularization(decompose_v(x), face_setsD, same_face_adjs)

def constraints(x):
    return all_constraint_residual_and_jacobian(
        decompose_v(x), edgesD, edge_pairsD, anglesD, ringsD, boundary_rings,
        boundary_nodes_cell, boundary_target_splines_cell, overlapD, [],
        rectangular_ratio, edges_bottom, edges_right)

# En Python, les contraintes doivent être packagées dans des dictionnaires ou des objets NonlinearConstraint
# Le bloc ci-dessous est conceptuel et dépendra de la structure exacte retournée par all_constraint_residual_and_jacobian
cons = {'type': 'eq', 'fun': constraints} 

print("Lancement de l'optimisation...")
start_time = time.time()
res = minimize(objective, x0, method='trust-constr', constraints=cons,
               options={'disp': True, 'maxiter': 250, 'gtol': 1e-6})
print(f"Terminé en {time.time() - start_time:.2f} secondes")

solved_pointsD = decompose_v(res.x)

# Obtention de la structure contractée finale
solved_points0 = get_contracted_shape(solved_pointsD, face_setsD, Dto0)

# Rotation optimale
solved_pointsD_temp = solved_pointsD - np.mean(solved_pointsD, axis=0)
solved_points0 = solved_points0 - np.mean(solved_points0, axis=0)
corners = [boundB[0], boundB[-1], boundT[0], boundT[-1]]

U, _, _ = Kabsch(solved_points0[Dto0[corners], :].T, solved_pointsD_temp[corners, :].T)
solved_points0 = (U @ solved_points0.T).T

if fix_contracted_boundary_shape:
    boundB_vec = solved_points0[Dto0[boundB[-1]], :] - solved_points0[Dto0[boundB[0]], :]
    angle = -np.angle(complex(boundB_vec[0], boundB_vec[1]))
    R = twoD_rotation(angle)
    solved_points0 = (R @ solved_points0.T).T

# --- Sauvegarde des résultats ---
name = f"results/quad_{shape_name}_w{width}_h{height}_i{initial_map_type}_f{fix_contracted_boundary_shape}_r{rectangular_ratio}"
write_mesh_generic(f"{name}_contracted.obj", solved_points0[Dto0, :], face_setsD)
write_mesh_generic(f"{name}_deployed.obj", solved_pointsD, face_setsD)