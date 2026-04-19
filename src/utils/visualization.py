from topology.core import Tessellation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from topology.unit_patterns import unit_RDQK_D
from topology.builder import build_tessellation
from matplotlib.patches import Polygon

from geometry.make_initial_map import compute_initial_map


def plot_tessellation(tessellation, ax=None):
    """
    Plots the tessellation using a discrete architectural palette,
    procedural hatching based on face geometry, and emphasized topological hinges.
    """

    print(f"Plotting tessellation with {len(tessellation.vertices)} vertices, {len(tessellation.faces)} faces, and {len(tessellation.hinges)} hinges.")

    if ax is None:
        # Utilisation d'un fond "papier" (off-white) pour un style plus graphique
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='#F4F4F0')
        ax.set_facecolor('#F4F4F0')

    # Palette discrète (ex: Minimaliste / Constructiviste)
    colors = ['#E63946', '#457B9D', '#1D3557', '#A8DADC']
    hatches = ['////', '\\\\\\\\', '....', '']

    for i, face in enumerate(tessellation.faces):
        vertices = tessellation.vertices[face.vertex_indices]
        
        # 1. Calcul de l'aire via la formule du lacet pour une attribution déterministe
        x, y = vertices[:, 0], vertices[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        
        # 2. Assignation de la couleur et de la texture basée sur la géométrie
        color_idx = int(area) % len(colors)
        hatch_idx = i % len(hatches)
        
        polygon = Polygon(
            vertices, 
            closed=True, 
            facecolor='orange', 
            edgecolor='#111111',
            linewidth=1.2,
            alpha=0.85
        )
        ax.add_patch(polygon)

        # Affichage du numéro de la face
        centroid = vertices.mean(axis=0)
        ax.text(centroid[0], centroid[1], str(i), color='black', fontsize=12, ha='center', va='center', weight='bold', zorder=20)

    # # Affichage des numéros des sommets
    # for i, v in enumerate(tessellation.vertices):
    #     ax.text(v[0], v[1], f"v{i}", color='red', fontsize=10, ha='right', va='bottom', weight='bold', zorder=21)

    # 3. Représentation mécanique/topologique des charnières
    num_faces = len(tessellation.faces)
    for i, hinge in enumerate(tessellation.hinges):
        # Vérification des indices de face valides
        if hinge.face1 < 0 or hinge.face1 >= num_faces or hinge.face2 < 0 or hinge.face2 >= num_faces:
            print(f"Warning: Hinge with face indices out of bounds: face1={hinge.face1}, face2={hinge.face2}, num_faces={num_faces}")
            continue
        
        # Coordonnées des sommets de la charnière
        v1 = tessellation.vertices[hinge.vertex1]
        v2 = tessellation.vertices[hinge.vertex2]
        v1_adj = tessellation.vertices[hinge.vertex_adjacent1]
        v2_adj = tessellation.vertices[hinge.vertex_adjacent2]

        
        # Vecteurs de la charnière
        hinge_vector1 = 0.4 * (v1_adj - v1)
        hinge_vector2 = 0.4 * (v2_adj - v2)
        # Point milieu de la charnière
        midpoint = (v1 + v2) / 2

        #afficchage des vecteurs
        ax.arrow(midpoint[0], midpoint[1], hinge_vector1[0], hinge_vector1[1], 
                 head_width=0.02, head_length=0.02, fc="#000000", ec="#000000", zorder=12)
        ax.arrow(midpoint[0], midpoint[1], hinge_vector2[0], hinge_vector2[1], 
                 head_width=0.02, head_length=0.02, fc="#000000", ec="#000000", zorder=12)
        
        # Marqueur au centre de la charnière
        ax.scatter(midpoint[0], midpoint[1], color='#F1FAEE', edgecolor="#000000", s=50, linewidth=2.0, zorder=14)

        # Affichage du numéro de la charnière
        ax.text(midpoint[0], midpoint[1] + 0.05, f"h{i}", color='blue', fontsize=10, ha='center', va='bottom', weight='bold', zorder=22)

    #plot target shape for reference
    circle = plt.Circle((0.0, 0.0), 1.0, color="#D6A400", fill=False, linestyle='--', linewidth=2.0, zorder=5)
    ax.add_patch(circle)

    ax.set_aspect('equal')
    ax.axis('off') # La suppression des axes renforce l'aspect purement artistique

    return ax