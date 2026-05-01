from topology.core import Tessellation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from problem.targets import DEFAULT_TARGET
from problem.targets import get_target_points

def plot_tessellation(tessellation, ax=None, 
                      show_faces=True, 
                      show_hinges=True, 
                      show_vertices=False, 
                      show_indices=True, 
                      show_target=True,
                      target_params=None,
                      show_border_edges=False,
                      title=None,
                      color_faces='orange'):
    """
    Plots the tessellation with configurable visibility for topological elements.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='#FFFFFF')
        ax.set_facecolor('#FFFFFF')

    # 1. Dessin des Faces
    if show_faces:
        is_color_list = isinstance(color_faces, list) and len(color_faces) == len(tessellation.faces)
        for i, face in enumerate(tessellation.faces):
            vertices = tessellation.vertices[face.vertex_indices]
            
            current_color = color_faces[i] if is_color_list else color_faces
            
            polygon = Polygon(
                vertices, 
                closed=True, 
                facecolor=current_color, 
                edgecolor='#111111',
                linewidth=1.2,
                alpha=0.85,
                zorder=10
            )
            ax.add_patch(polygon)

            if show_indices:
                centroid = vertices.mean(axis=0)
                ax.text(centroid[0], centroid[1], str(i), color='black', fontsize=10, 
                        ha='center', va='center', weight='bold', zorder=20)

    # 2. Dessin des Sommets
    if show_vertices:
        X = tessellation.vertices
        ax.scatter(X[:, 0], X[:, 1], color='#E63946', s=20, zorder=25)
        if show_indices:
            for i, v in enumerate(X):
                ax.text(v[0], v[1], f"v{i}", color='#E63946', fontsize=8, ha='right', zorder=26)

    # 3. Dessin des Charnières (Vecteurs et points milieux)
    if show_hinges:
        num_faces = len(tessellation.faces)
        for i, hinge in enumerate(tessellation.hinges):
            if hinge.face1 < 0 or hinge.face2 < 0: continue
            
            v1 = tessellation.vertices[hinge.vertex1]
            v2 = tessellation.vertices[hinge.vertex2]
            v1_adj = tessellation.vertices[hinge.vertex_adjacent1]
            v2_adj = tessellation.vertices[hinge.vertex_adjacent2]

            midpoint = (v1 + v2) / 2
            
            # Vecteurs de déploiement/contraction
            hinge_vector1 = 0.3 * (v1_adj - v1)
            hinge_vector2 = 0.3 * (v2_adj - v2)

            ax.arrow(midpoint[0], midpoint[1], hinge_vector1[0], hinge_vector1[1], 
                     head_width=0.015, head_length=0.015, fc="#000000", ec="#000000", zorder=12)
            ax.arrow(midpoint[0], midpoint[1], hinge_vector2[0], hinge_vector2[1], 
                     head_width=0.015, head_length=0.015, fc="#000000", ec="#000000", zorder=12)
            
            ax.scatter(midpoint[0], midpoint[1], color='#F1FAEE', edgecolor="#000000", s=30, linewidth=1.5, zorder=14)
            
            if show_indices:
                ax.text(midpoint[0], midpoint[1] + 0.02, f"h{i}", color='blue', fontsize=8, 
                        ha='center', va='bottom', weight='bold', zorder=22)

    # 4. Target Shape (Cloud representation)
    if show_target:
        target_pts = get_target_points(target_params, n_points=200)
        if len(target_pts) > 0:
            # On boucle pour fermer la ligne
            plot_pts = np.vstack([target_pts, target_pts[0]])
            ax.plot(plot_pts[:, 0], plot_pts[:, 1], color="#D6A400", linestyle='--', linewidth=2.0, zorder=5)

    if title:
        ax.set_title(title, fontsize=16, weight='bold', pad=20)

    # Centrage automatique de la vue
    X = tessellation.vertices
    if len(X) > 0:
        x_min, y_min = X.min(axis=0)
        x_max, y_max = X.max(axis=0)
        
        if show_target and 'target_pts' in locals() and len(target_pts) > 0:
            x_min = min(x_min, target_pts[:, 0].min())
            x_max = max(x_max, target_pts[:, 0].max())
            y_min = min(y_min, target_pts[:, 1].min())
            y_max = max(y_max, target_pts[:, 1].max())
        
        center_x = (x_max + x_min) / 2
        center_y = (y_max + y_min) / 2
        
        # On définit une portée carrée basée sur la plus grande dimension
        delta_x = (x_max - x_min)
        delta_y = (y_max - y_min)
        max_range = max(delta_x, delta_y) * 1.1 # +10% de marge
        
        ax.set_xlim(center_x - max_range/2, center_x + max_range/2)
        ax.set_ylim(center_y - max_range/2, center_y + max_range/2)

    ax.set_aspect('equal')
    ax.axis('off')
    return ax

def animate_tessellation(tessellation, state_history, filepath="closing_animation.gif", fps=15, target_params=None, **plot_kwargs):
    """
    Animates the tessellation process given a history of states and saves it to a file.
    """
    if not state_history:
        print("Warning: state_history is empty, cannot animate.")
        return

    fig, ax = plt.subplots(figsize=(10, 10), facecolor='#FFFFFF')
    
    # Compute fixed bounds so the camera doesn't jitter
    all_X = np.concatenate(state_history)
    x_min, y_min = all_X.min(axis=0)
    x_max, y_max = all_X.max(axis=0)
    center_x = (x_max + x_min) / 2
    center_y = (y_max + y_min) / 2
    delta_x = (x_max - x_min)
    delta_y = (y_max - y_min)
    max_range = max(delta_x, delta_y) * 1.1

    def update(frame):
        ax.clear()
        tessellation.update_vertices(state_history[frame])
        
        # We reuse the existing plotting function
        plot_tessellation(tessellation, ax=ax, title=f"Closing Process", target_params=target_params, **plot_kwargs)
        
        # Enforce fixed bounds over the automatic ones computed in plot_tessellation
        ax.set_xlim(center_x - max_range/2, center_x + max_range/2)
        ax.set_ylim(center_y - max_range/2, center_y + max_range/2)

    print(f"Generating animation with {len(state_history)} frames...")
    ani = animation.FuncAnimation(fig, update, frames=len(state_history), blit=False)
    
    # Use pillow for GIF, otherwise default
    writer = 'pillow' if filepath.endswith('.gif') else None
    ani.save(filepath, writer=writer, fps=fps)
    plt.close(fig)
    print(f"Animation successfully saved to {filepath}")

def plot_tessellation_differences(tessellation, diff_values, ax=None, 
                                  title="Deformation Map",
                                  cmap_name='YlOrRd',
                                  **kwargs):
    """
    Plots the tessellation where faces are colored based on a difference metric.
    """
    if not diff_values:
        print("Warning: diff_values is empty.")
        return ax
        
    min_val = min(diff_values)
    max_val = max(diff_values)
    
    if max_val > min_val:
        norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
    else:
        norm = mcolors.Normalize(vmin=min_val - 0.1, vmax=max_val + 0.1)
        
    cmap = cm.get_cmap(cmap_name)
    face_colors = [cmap(norm(val)) for val in diff_values]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='#FFFFFF')
    else:
        fig = ax.figure
        
    ax = plot_tessellation(tessellation, ax=ax, title=title, color_faces=face_colors, **kwargs)
    
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7)
    cbar.set_label('Deformation (%)', rotation=270, labelpad=15)
    
    return ax