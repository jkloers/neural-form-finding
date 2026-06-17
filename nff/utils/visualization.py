import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, Circle
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from nff.config.targets import get_target_points


def plot_cuts_in_sheet(tessellation, ax=None, filepath=None, hinge_frac=0.10,
                       paper_color="#F58025", cut_color="#1A1A1A", title=None):
    """Render the flat sheet as a kirigami cut pattern.

    Shows the solid sheet with the cuts (slits) drawn as black lines along the
    panel boundaries, and the small remaining-paper bridges (hinges) as paper-
    coloured discs at the pivot points. Long cuts ⟺ small ``hinge_frac``.

    Args:
        tessellation: flat closed Tessellation (Stage-0 geometry).
        hinge_frac: hinge bridge radius as a fraction of the mean panel edge.
    """
    from shapely.geometry import Polygon as _ShPoly
    from shapely.ops import unary_union

    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(9, 9), facecolor="white")
    verts = tessellation.vertices

    polys = []
    for f in tessellation.faces:
        p = verts[f.vertex_indices]
        ax.add_patch(Polygon(p, closed=True, facecolor=paper_color,
                             edgecolor=cut_color, lw=1.6, zorder=2))
        polys.append(_ShPoly(p))

    # Solid (uncut) sheet outline.
    sheet = unary_union([p if p.is_valid else p.buffer(0) for p in polys])
    geoms = getattr(sheet, "geoms", [sheet])
    for g in geoms:
        xs, ys = g.exterior.xy
        ax.plot(xs, ys, color=cut_color, lw=3.0, zorder=4)

    # Remaining-paper bridges (hinges) at the pivots.
    edge_len = np.mean([np.linalg.norm(verts[f.vertex_indices[0]] - verts[f.vertex_indices[1]])
                        for f in tessellation.faces])
    radius = hinge_frac * float(edge_len)
    for h in tessellation.hinges:
        ax.add_patch(Circle(verts[h.vertex1], radius, facecolor=paper_color,
                            edgecolor="none", zorder=3))

    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title)
    if own and filepath:
        plt.savefig(filepath, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved cut pattern to {filepath}")

def plot_tessellation(tessellation, ax=None, 
                      show_faces=True, 
                      show_hinges=True, 
                      show_vertices=False, 
                      show_face_indices=True,
                      show_hinge_indices=True,
                      show_external_forces=False,
                      show_kinematic_blocks=False,
                      show_target=True,
                      target_params=None,
                      show_border_edges=False,
                      title=None,
                      color_faces='#F58025',
                      mapping_fn=None,
                      map_params=None,
                      original_vertices=None):
    """
    Plots the tessellation with configurable visibility for topological elements.
    """
    if ax is None:
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='#FFFFFF')
        ax.set_facecolor('#FFFFFF')

    # 0. Transformation Grid (if mapping_fn is provided)
    if mapping_fn is not None and original_vertices is not None:
        import jax.numpy as jnp
        import jax
        
        min_xy = np.min(original_vertices, axis=0)
        max_xy = np.max(original_vertices, axis=0)
        
        num_lines = 25
        pts_per_line = 100
        x_starts = np.linspace(min_xy[0], max_xy[0], num_lines)
        y_starts = np.linspace(min_xy[1], max_xy[1], num_lines)
        
        y_range = jnp.linspace(min_xy[1], max_xy[1], pts_per_line)
        x_range = jnp.linspace(min_xy[0], max_xy[0], pts_per_line)
        
        f_point = lambda p: mapping_fn(p, map_params)
        f_vmap = jax.vmap(f_point)
        
        # Vertical lines
        for x in x_starts:
            pts = jnp.column_stack([jnp.full(pts_per_line, x), y_range])
            mapped_pts = f_vmap(pts)
            ax.plot(mapped_pts[:, 0], mapped_pts[:, 1], color='#457B9D', alpha=0.4, linewidth=1.0, zorder=6)
            
        # Horizontal lines
        for y in y_starts:
            pts = jnp.column_stack([x_range, jnp.full(pts_per_line, y)])
            mapped_pts = f_vmap(pts)
            ax.plot(mapped_pts[:, 0], mapped_pts[:, 1], color='#457B9D', alpha=0.4, linewidth=1.0, zorder=6)

    # 1. Faces
    if show_faces:
        is_color_list = isinstance(color_faces, list) and len(color_faces) == len(tessellation.faces)
        for i, face in enumerate(tessellation.faces):
            vertices = tessellation.vertices[face.vertex_indices]
            
            current_color = color_faces[i] if is_color_list else color_faces
            
            # Color kinematic blocks differently
            if show_kinematic_blocks and hasattr(face, 'dofs') and len(face.dofs) > 0:
                current_color = '#6C757D'  # gray for clamped faces
            
            polygon = Polygon(
                vertices, 
                closed=True, 
                facecolor=current_color, 
                edgecolor='#1A1A1A',
                linewidth=1.2,
                alpha=0.85,
                zorder=10
            )
            ax.add_patch(polygon)

            if show_face_indices:
                centroid = vertices.mean(axis=0)
                ax.text(centroid[0], centroid[1], str(i), color='black', fontsize=10, 
                        ha='center', va='center', weight='bold', zorder=20)

    # 2. Vertices
    if show_vertices:
        X = tessellation.vertices
        ax.scatter(X[:, 0], X[:, 1], color='#E63946', s=20, zorder=25)
        for i, v in enumerate(X):
            ax.text(v[0], v[1], f"v{i}", color='#E63946', fontsize=8, ha='right', zorder=26)

    # 3. Hinges (vectors and midpoints)
    if show_hinges:
        num_faces = len(tessellation.faces)
        for i, hinge in enumerate(tessellation.hinges):
            if hinge.face1 < 0 or hinge.face2 < 0: continue
            
            v1 = tessellation.vertices[hinge.vertex1]
            v2 = tessellation.vertices[hinge.vertex2]
            v1_adj = tessellation.vertices[hinge.vertex_adjacent1]
            v2_adj = tessellation.vertices[hinge.vertex_adjacent2]

            midpoint = (v1 + v2) / 2
            
            # Deployment/contraction vectors
            hinge_vector1 = 0.3 * (v1_adj - v1)
            hinge_vector2 = 0.3 * (v2_adj - v2)

            ax.arrow(midpoint[0], midpoint[1], hinge_vector1[0], hinge_vector1[1], 
                     head_width=0.015, head_length=0.015, fc="#000000", ec="#000000", zorder=12)
            ax.arrow(midpoint[0], midpoint[1], hinge_vector2[0], hinge_vector2[1], 
                     head_width=0.015, head_length=0.015, fc="#000000", ec="#000000", zorder=12)
            
            ax.scatter(midpoint[0], midpoint[1], color='#F1FAEE', edgecolor="#000000", s=30, linewidth=1.5, zorder=14)
            
            if show_hinge_indices:
                ax.text(midpoint[0], midpoint[1] + 0.02, f"h{i}", color='#457B9D', fontsize=8, 
                        ha='center', va='bottom', weight='bold', zorder=22)

    # 4. External Forces & Moments
    if show_external_forces:
        for face in tessellation.faces:
            if hasattr(face, 'loads') and face.loads:
                vertices = tessellation.vertices[face.vertex_indices]
                centroid = vertices.mean(axis=0)
                
                fx = face.loads.get(0, 0.0)
                fy = face.loads.get(1, 0.0)
                moment = face.loads.get(2, 0.0)
                
                # Draw force vector
                if fx != 0 or fy != 0:
                    force_len = np.sqrt(fx**2 + fy**2)
                    scale = min(0.2 / force_len, 0.05) if force_len > 0 else 0.05
                    dx = fx * scale
                    dy = fy * scale
                    
                    # Offset start slightly so we don't cover the centroid index
                    ax.arrow(centroid[0], centroid[1], dx, dy, 
                             head_width=0.03, head_length=0.03, fc="#D62828", ec="#D62828", 
                             zorder=30, width=0.005)
                             
                # Draw moment
                if moment != 0:
                    r = 0.08
                    if moment > 0:
                        # Counter-clockwise
                        start = (centroid[0] + r, centroid[1] - r/2)
                        end = (centroid[0] - r/2, centroid[1] + r)
                        rad = 0.6
                    else:
                        # Clockwise
                        start = (centroid[0] - r, centroid[1] - r/2)
                        end = (centroid[0] + r/2, centroid[1] + r)
                        rad = -0.6
                        
                    arrow = FancyArrowPatch(start, end, connectionstyle=f"arc3,rad={rad}", 
                                            color="#D62828", arrowstyle="Simple, tail_width=1.5, head_width=6, head_length=8",
                                            zorder=30)
                    ax.add_patch(arrow)

    # 4. Target Shape
    if show_target:
        target_pts = get_target_points(target_params, n_points=200)
        if len(target_pts) > 0:
            plot_pts = np.vstack([target_pts, target_pts[0]])
            ax.plot(plot_pts[:, 0], plot_pts[:, 1], color="#009900", linestyle='--', linewidth=2.5, zorder=5)

    if title:
        ax.set_title(title, fontsize=16, weight='bold', color='black', pad=20)

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

        delta_x = (x_max - x_min)
        delta_y = (y_max - y_min)
        max_range = max(delta_x, delta_y) * 1.1  # +10% margin
        
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
    
    if filepath is None:
        print("No filepath provided, skipping animation generation.")
        return

    plt.style.use('default')
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
    
    if filepath is not None:
        writer = 'pillow' if filepath.endswith('.gif') else None
        ani.save(filepath, writer=writer, fps=fps)
        print(f"Animation successfully saved to {filepath}")
    else:
        # If no filepath is provided, do nothing or display it (if in a notebook)
        pass
    plt.close(fig)

def plot_tessellation_differences(tessellation, diff_values, ax=None, 
                                  title="Deformation Map",
                                  cmap_name='YlOrRd',
                                  **kwargs):
    """
    Plots the tessellation where faces are colored based on a difference metric.
    """
    if diff_values is None or len(diff_values) == 0:
        print("Warning: diff_values is empty.")
        return ax
        
    min_val = jnp.min(diff_values)
    max_val = jnp.max(diff_values)
    
    if max_val > min_val:
        norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
    else:
        norm = mcolors.Normalize(vmin=min_val - 0.1, vmax=max_val + 0.1)
        
    cmap = cm.get_cmap(cmap_name)
    face_colors = [cmap(norm(val)) for val in diff_values]
    
    if ax is None:
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='#FFFFFF')
    else:
        fig = ax.figure
        
    ax = plot_tessellation(tessellation, ax=ax, title=title, color_faces=face_colors, **kwargs)
    
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7)
    cbar.set_label('Deformation (%)', rotation=270, labelpad=15)
    
    return ax