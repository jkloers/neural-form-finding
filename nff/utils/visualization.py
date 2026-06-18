import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, Circle, Rectangle
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from nff.config.targets import get_target_points


def plot_cut_pattern(coords, T, cols, ax=None, filepath=None, hinge_margin=0.06,
                     paper_color="#F58025", cut_color="#1A1A1A", lw=2.4, title=None):
    """Render the true kirigami cut pattern of the flat sheet.

    Each interior cut C_{i,j} is the long slit that runs from one hinge to its
    opposite hinge across the void — i.e. between the collinear neighbour vertices
    the LES links (horizontal: x_{i-1,j} -> x'_{i+1,j}; vertical: x'_{i,j-1} ->
    x_{i,j+1}), with the cut's own vertices x_{i,j}, x'_{i,j} lying in between. The
    slit is drawn as long as possible, retracted by ``hinge_margin`` at interior
    ends (leaving the small uncut hinge ligaments) and run all the way to the sheet
    border when an endpoint is a boundary vertex.

    Args:
        coords: (P, 2) cut endpoint positions, P = 2 * rows * cols.
                Endpoint s of cut (i, j) is index 2*(i*cols + j) + s.
        T: (rows, cols) topology matrix (sign = interior/boundary, |.| = h/v).
        cols: number of cut-grid columns (T.shape[1]).
        hinge_margin: uncut ligament length left at each interior cut end.
    """
    from shapely.geometry import MultiPoint
    coords = np.asarray(coords)
    rows = T.shape[0]

    def pid(i, j, s):
        return 2 * (i * cols + j) + s

    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(9, 9), facecolor="white")

    # Solid sheet = convex hull of the boundary-cut positions.
    bpos = [coords[pid(i, j, 0)] for i in range(rows) for j in range(cols) if T[i, j] < 0]
    hull = MultiPoint(bpos).convex_hull
    hx, hy = hull.exterior.xy
    ax.fill(hx, hy, facecolor=paper_color, edgecolor=cut_color, lw=2.8, zorder=1)

    # Interior cuts: long slit from hinge to opposite hinge across the void.
    for i in range(rows):
        for j in range(cols):
            if T[i, j] <= 0:
                continue
            if abs(int(T[i, j])) == 1:                  # horizontal cut
                (ai, aj, as_), (bi, bj, bs) = (i - 1, j, 0), (i + 1, j, 1)
            else:                                       # vertical cut
                (ai, aj, as_), (bi, bj, bs) = (i, j - 1, 1), (i, j + 1, 0)
            A, B = coords[pid(ai, aj, as_)], coords[pid(bi, bj, bs)]
            d = B - A
            L = float(np.linalg.norm(d))
            if L < 1e-9:
                continue
            u = d / L
            # Retract at interior ends (leave a hinge); run to the border otherwise.
            p0 = A + (0.0 if T[ai, aj] < 0 else hinge_margin) * u
            p1 = B - (0.0 if T[bi, bj] < 0 else hinge_margin) * u
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=cut_color, lw=lw,
                    solid_capstyle="round", zorder=2)

    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title)
    if own and filepath:
        plt.savefig(filepath, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved cut pattern to {filepath}")


def write_deformed_into(tessellation, node_positions):
    """Return a copy of ``tessellation`` with vertices set to deformed positions.

    Args:
        tessellation: reference Tessellation.
        node_positions: (n_faces, max_nodes, 2) per-face deformed node positions
            (e.g. from ``deformed_vertices``).
    """
    deformed = tessellation.copy()
    new_vertices = np.array(deformed.vertices, dtype=float)
    for f_id, face in enumerate(deformed.faces):
        for local, gv in enumerate(face.vertex_indices):
            new_vertices[gv] = node_positions[f_id, local]
    deformed.update_vertices(new_vertices)
    return deformed


def plot_loading_diagram(tessellation, clamped_faces, load_specs, filepath,
                         title="Loading"):
    """One clean schematic of the boundary conditions and applied loads.

    Clamped faces are greyed with a hatched fixed-support wall; a distributed edge
    pull (dof 0) is drawn as a comb of uniform arrows just outside the loaded edge;
    point loads (dof 1) are single bold arrows at the loaded tile. No per-face arrow
    clutter — exactly one legible loading picture.
    """
    clamped = {int(f) for f in (clamped_faces or [])}
    colors = ["#9AA3AB" if i in clamped else "#F7C59F" for i in range(len(tessellation.faces))]
    fig, ax = plt.subplots(figsize=(11, 8.5), facecolor="white")
    plot_tessellation(tessellation, ax=ax, show_target=False, show_hinges=False,
                      show_face_indices=False, show_hinge_indices=False,
                      show_external_forces=False, color_faces=colors)

    allv = np.asarray(tessellation.vertices, dtype=float)
    x0, y0 = allv.min(axis=0)
    x1, y1 = allv.max(axis=0)
    span = float(max(x1 - x0, y1 - y0))
    L = 0.11 * span
    RED = "#D62828"

    def fv(fi):
        return np.asarray(tessellation.vertices[tessellation.faces[int(fi)].vertex_indices], dtype=float)

    # Distributed pull (dof 0) — comb of uniform arrows + a tail bracket.
    pull = [s for s in (load_specs or []) if int(s.get('dof', -1)) == 0 and float(s.get('value', 0.0)) != 0.0]
    if pull:
        xa = max(fv(s['face'])[:, 0].max() for s in pull) + 0.03 * span
        ys = [fv(s['face'])[:, 1].mean() for s in pull]
        for cy in ys:
            ax.annotate('', xy=(xa + L, cy), xytext=(xa, cy),
                        arrowprops=dict(arrowstyle='-|>', color=RED, lw=2.0, mutation_scale=14), zorder=30)
        ax.plot([xa, xa], [min(ys), max(ys)], color=RED, lw=2.5, zorder=29)

    # Point loads (dof 1) — single bold arrow per loaded tile.
    for s in (load_specs or []):
        if int(s.get('dof', -1)) == 1 and float(s.get('value', 0.0)) != 0.0:
            p = fv(s['face'])
            cx = p[:, 0].mean()
            if float(s['value']) < 0:           # downward
                y_anchor = p[:, 1].max() + 0.02 * span
                ax.annotate('', xy=(cx, y_anchor), xytext=(cx, y_anchor + 1.7 * L),
                            arrowprops=dict(arrowstyle='-|>', color=RED, lw=3.2, mutation_scale=22), zorder=31)
            else:                               # upward
                y_anchor = p[:, 1].min() - 0.02 * span
                ax.annotate('', xy=(cx, y_anchor), xytext=(cx, y_anchor - 1.7 * L),
                            arrowprops=dict(arrowstyle='-|>', color=RED, lw=3.2, mutation_scale=22), zorder=31)

    # Fixed-support wall on the clamped edge.
    if clamped:
        wx = min(fv(i)[:, 0].min() for i in clamped)
        ax.add_patch(Rectangle((wx - 0.06 * span, y0), 0.05 * span, y1 - y0,
                               facecolor="#6C757D", edgecolor="#6C757D", hatch="////", lw=0, alpha=0.55, zorder=2))

    ax.set_xlim(x0 - 0.14 * span, x1 + 0.28 * span)
    ax.set_ylim(y0 - 0.16 * span, y1 + 0.22 * span)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=15, weight="bold")
    fig.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved loading diagram to {filepath}")


def plot_area_change(tess_flat, tess_deployed, rel, filepath,
                     titles=("Trained design (closed)", "Deployed")):
    """Two-panel figure colouring each panel by its area change vs the initial design.

    Args:
        tess_flat, tess_deployed: trained-design tessellations (flat / deployed).
        rel: (n_faces,) relative area change (trained / initial - 1).
        filepath: output PNG path.
    """
    from matplotlib.ticker import FuncFormatter
    cmap = plt.get_cmap("coolwarm")          # blue = shrunk, red = enlarged
    norm = mcolors.TwoSlopeNorm(vmin=min(float(rel.min()), -1e-3), vcenter=0.0,
                                vmax=max(float(rel.max()), 1e-3))
    area_colors = [cmap(norm(r)) for r in rel]

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.6), facecolor="white")
    for ax, t, ttl in ((axes[0], tess_flat, titles[0]), (axes[1], tess_deployed, titles[1])):
        plot_tessellation(t, ax=ax, show_target=False, show_hinges=False,
                          show_face_indices=False, show_hinge_indices=False, color_faces=area_colors)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(ttl)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("panel area change vs initial design", fontsize=12)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.0%}"))
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def animate_closed_evolution(tessellation, frames, add_target, filepath,
                             title="Opening process", fps=6):
    """Animate the closed-state design morphing over training epochs (flat | deployed).

    Args:
        tessellation: reference Tessellation (copied internally for each panel).
        frames: list of (epoch, flat_verts, deployed_verts, boundary_cloud), where
            *_verts are (n_global_vertices, 2) global vertex arrays.
        add_target: callable (ax, boundary_cloud) -> None drawing the target on the
            deployed panel.
        filepath: output GIF path.
    """
    allf = np.concatenate([f[1] for f in frames])
    alld = np.concatenate([f[2] for f in frames])
    fb = (allf[:, 0].min() - .5, allf[:, 0].max() + .5, allf[:, 1].min() - .5, allf[:, 1].max() + .5)
    db = (alld[:, 0].min() - .5, alld[:, 0].max() + .5, alld[:, 1].min() - .5, alld[:, 1].max() + .5)
    tfl, tdp = tessellation.copy(), tessellation.copy()
    fig, (axf, axd) = plt.subplots(1, 2, figsize=(15, 7.6), facecolor="white")
    fig.suptitle(title, fontsize=16, weight="bold")

    def draw(k):
        ep, flat, dep, cloud = frames[k]
        axf.clear(); axd.clear()
        tfl.update_vertices(flat)
        plot_tessellation(tfl, ax=axf, show_target=False, show_hinges=False,
                          show_face_indices=False, show_hinge_indices=False, color_faces="#F58025")
        axf.set_xlim(fb[0], fb[1]); axf.set_ylim(fb[2], fb[3]); axf.set_aspect("equal"); axf.axis("off")
        axf.set_title(f"Closed design — epoch {ep}")
        tdp.update_vertices(dep)
        plot_tessellation(tdp, ax=axd, show_target=False, show_hinges=False,
                          show_face_indices=False, show_hinge_indices=False, color_faces="#F58025")
        add_target(axd, cloud)
        axd.set_xlim(db[0], db[1]); axd.set_ylim(db[2], db[3]); axd.set_aspect("equal"); axd.axis("off")
        axd.set_title(f"Deploy — epoch {ep}")

    animation.FuncAnimation(fig, draw, frames=len(frames), blit=False).save(
        filepath, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"  Saved training-evolution animation to {filepath}")


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
        # Size arrows relative to the domain and to the largest load in the scene,
        # so the biggest force is a clearly visible fraction of the geometry and the
        # rest are drawn in proportion (independent of absolute force magnitude).
        X_all = tessellation.vertices
        domain_scale = float(np.linalg.norm(X_all.max(axis=0) - X_all.min(axis=0))) if len(X_all) else 1.0
        max_force = 0.0
        max_moment = 0.0
        for face in tessellation.faces:
            if hasattr(face, 'loads') and face.loads:
                max_force = max(max_force, float(np.hypot(face.loads.get(0, 0.0), face.loads.get(1, 0.0))))
                max_moment = max(max_moment, abs(float(face.loads.get(2, 0.0))))
        ref_len = 0.16 * domain_scale          # length of the largest force arrow
        head = 0.045 * domain_scale
        shaft = 0.012 * domain_scale

        for face in tessellation.faces:
            if not (hasattr(face, 'loads') and face.loads):
                continue
            vertices = tessellation.vertices[face.vertex_indices]
            centroid = vertices.mean(axis=0)

            fx = face.loads.get(0, 0.0)
            fy = face.loads.get(1, 0.0)
            moment = face.loads.get(2, 0.0)

            # Draw force vector — length proportional to |F| / max|F|.
            if (fx != 0 or fy != 0) and max_force > 0:
                fmag = np.hypot(fx, fy)
                length = ref_len * fmag / max_force
                dx, dy = fx / fmag * length, fy / fmag * length
                ax.arrow(centroid[0], centroid[1], dx, dy,
                         head_width=head, head_length=head, width=shaft,
                         length_includes_head=True, fc="#D62828", ec="#8B0000",
                         linewidth=0.8, alpha=0.95, zorder=30)

            # Draw moment as a curved arrow, sized to the domain.
            if moment != 0:
                r = 0.07 * domain_scale
                if moment > 0:                  # counter-clockwise
                    start = (centroid[0] + r, centroid[1] - r / 2)
                    end = (centroid[0] - r / 2, centroid[1] + r)
                    rad = 0.6
                else:                           # clockwise
                    start = (centroid[0] - r, centroid[1] - r / 2)
                    end = (centroid[0] + r / 2, centroid[1] + r)
                    rad = -0.6
                arrow = FancyArrowPatch(start, end, connectionstyle=f"arc3,rad={rad}",
                                        color="#D62828",
                                        arrowstyle="Simple, tail_width=1.5, head_width=6, head_length=8",
                                        mutation_scale=max(10.0, 0.6 * domain_scale), zorder=30)
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
        plot_tessellation(tessellation, ax=ax, title="Opening process", target_params=target_params, **plot_kwargs)
        
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