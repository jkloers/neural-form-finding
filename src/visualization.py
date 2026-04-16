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
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        
        # 2. Assignation de la couleur et de la texture basée sur la géométrie
        # Le hachage de l'aire arrondie assure une distribution reproductible
        color_idx = int(area) % len(colors)
        hatch_idx = i % len(hatches)
        
        polygon = Polygon(
            vertices, 
            closed=True, 
            facecolor='orange', 
            edgecolor='#111111', # Noir très profond pour les arêtes
            linewidth=1.2,
            alpha=0.85 # Légère transparence pour adoucir le contraste
        )
        ax.add_patch(polygon)

    # 3. Représentation mécanique/topologique des charnières
    for hinge in tessellation.hinges:
        v1 = tessellation.vertices[hinge.vertex1]
        v2 = tessellation.vertices[hinge.vertex2]
        midpoint = (v1 + v2) / 2
        
        # Remplacement du 'scatter' par un marqueur carré évidé (liaison formelle)
        ax.plot(midpoint[0], midpoint[1], marker='s', color='#F4F4F0', 
                markeredgecolor='#111111', markersize=5, markeredgewidth=1.2, zorder=10)

    ax.set_aspect('equal')
    ax.axis('off') # La suppression des axes renforce l'aspect purement artistique
    return ax

if __name__ == "__main__":

    initial_tessellation = build_tessellation(unit_RDQK_D, nx=3, ny=3)
    mapped_tessellation = compute_initial_map(initial_tessellation, ('circle', [0.0, 0.0], 1.0), map_type='elliptical_grip')

    plot_tessellation(mapped_tessellation)

    plt.show()