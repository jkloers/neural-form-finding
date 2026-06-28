"""
nff/sofa/mesh_builder_gmsh.py — gmsh-based mesh builder for kirigami unit cells.

Each hinge is a curved strip connecting two face edges.  The strip is bounded by:
  - Two cubic Bézier arcs: upper (p0_top → p1_top) and lower (p0_bot → p1_bot)
  - Two straight closing edges along the face boundaries (shared with face surfaces)

The two faces of a corner hinge are rigidly translated apart by ``gap`` along the
centroid-to-centroid axis, opening a real void that the Bézier hinge spans.  The
four arc endpoints slide along each (translated) face's own edges via the reach
parameters, so they always lie exactly on the face boundaries.

Parametrisation per hinge
--------------------------
  gap             — rigid separation of the two faces along ê_arm [m]
  s0_top, s0_bot  — reach from face-i corner along its upper/lower edges [m]
  s1_top, s1_bot  — reach from face-k corner along its upper/lower edges [m]
  bc_up_xy — single interior CP for the upper (quadratic) arc, absolute XY [m]
  bc_lo_xy — single interior CP for the lower (quadratic) arc, absolute XY [m]

Workflow
--------
  1. Resolve face polygons + hinge anchors via compute_hinge_geometry
     (faces translated apart by gap; endpoints anchored on the face edges).
  2. Build conforming 2D domain in gmsh: faces + hinge share boundary edges.
  3. Extrude by sheet_thickness → prism elements → split each prism into 3 tetrahedra.
  4. Classify nodes into clamped / loaded via 2D point-in-polygon.

Returns
-------
  nodes    : (N, 3) float64
  tets     : (M, 4) int32   (0-indexed)
  bc_masks : {'clamped': bool (N,), 'loaded': bool (N,)}
"""

from __future__ import annotations
import numpy as np
from matplotlib.path import Path as _MplPath

# compute_hinge_geometry lives in the sibling geometry module. This file runs in
# two import contexts — as a package submodule (nff.sofa.*) on the JAX/viz side,
# and as a bare top-level module inside the SOFA Docker container — so try the
# package path first and fall back to the flat one.
try:
    from nff.sofa.hinge_geometry import compute_hinge_geometry
except ImportError:
    from hinge_geometry import compute_hinge_geometry


# ── prism → tet splitting ─────────────────────────────────────────────────────

def _prisms_to_tets(prisms: np.ndarray) -> np.ndarray:
    """Split (N, 6) prism connectivity into (3N, 4) tetrahedra.

    Prism node layout (gmsh default after straight extrusion of a triangle):
      bottom: v0, v1, v2   top: v3, v4, v5  (v3 above v0, etc.)
    """
    v = prisms
    return np.vstack([
        np.column_stack([v[:, 0], v[:, 1], v[:, 2], v[:, 3]]),
        np.column_stack([v[:, 1], v[:, 3], v[:, 4], v[:, 2]]),
        np.column_stack([v[:, 2], v[:, 3], v[:, 4], v[:, 5]]),
    ]).astype(np.int32)


# ── BC classification ─────────────────────────────────────────────────────────

def _nodes_in_polygon(nodes_3d: np.ndarray, poly_xy: np.ndarray) -> np.ndarray:
    return _MplPath(poly_xy).contains_points(nodes_3d[:, :2], radius=1e-9)


# ── bezier sampling ───────────────────────────────────────────────────────────

def _bezier_sample(p0: np.ndarray, c: np.ndarray, p2: np.ndarray,
                   n_segs: int) -> np.ndarray:
    """Return n_segs+1 uniformly-parameterised points along quadratic Bézier p0,c,p2.

    One interior control point → a single convex arc (no inflections), which gives
    more regular hinge boundaries than a cubic and fewer design parameters.
    """
    t = np.linspace(0.0, 1.0, n_segs + 1)[:, None]
    return (1-t)**2 * p0 + 2*(1-t)*t * c + t**2 * p2


# ── main builder ──────────────────────────────────────────────────────────────

def _extract_nodes_and_tets(gmsh) -> tuple:
    """Pull node coordinates and 3D elements from a meshed gmsh model.

    Prisms (6-node, from extruded triangles) are split into tets; native
    tets (4-node) pass through.

    Returns:
        (coords (N, 3) float64, tets (M, 4) int32)
    """
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = np.array(coords, dtype=np.float64).reshape(-1, 3)
    tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

    etypes, _, enode_tags_list = gmsh.model.mesh.getElements(dim=3)
    el_nodes_per_type = {4: 4, 6: 6}
    tet_blocks = []
    for etype, enodes in zip(etypes, enode_tags_list):
        npe = el_nodes_per_type.get(int(etype))
        if npe is None:
            continue
        rows = np.array(enodes, dtype=np.int64).reshape(-1, npe)
        idx  = np.array([[tag_to_idx[int(t)] for t in r] for r in rows], np.int32)
        tet_blocks.append(_prisms_to_tets(idx) if npe == 6 else idx)

    if not tet_blocks:
        raise RuntimeError("gmsh produced no 3D elements.")
    return coords, np.vstack(tet_blocks).astype(np.int32)


def _compute_bc_masks(coords, face_polys_2d, clamped_faces, loaded_faces) -> dict:
    """Classify mesh nodes into clamped/loaded by point-in-face-polygon test.

    Geometric (not index-based), so it stays correct even though the mesh is
    rebuilt from scratch on every call.
    """
    bc_masks = {
        'clamped': np.zeros(coords.shape[0], bool),
        'loaded':  np.zeros(coords.shape[0], bool),
    }
    for fi_idx, poly_xy in enumerate(face_polys_2d):
        mask = _nodes_in_polygon(coords, poly_xy)
        if fi_idx in clamped_faces:
            bc_masks['clamped'] |= mask
        if fi_idx in loaded_faces:
            bc_masks['loaded'] |= mask
    return bc_masks


def build_mesh_gmsh(
    cs,
    gap: float = 0.003,
    sheet_thickness: float = 0.001,
    bezier_params: dict = None,
    mesh_size_face: float = None,
    mesh_size_hinge: float = None,
    mesh_refine: float = 1.0,
    n_z_layers: int = 2,
    verbose: bool = False,
) -> tuple:
    """
    Build a tetrahedral FEM mesh for a kirigami unit cell using gmsh.

    Parameters
    ----------
    cs                  : CentroidalState (or SimpleNamespace from build_physical_cs)
    gap                 : rigid separation of the two faces along ê_arm [m].
                          Translates whole faces apart, opening the hinge void.
    sheet_thickness     : extrusion depth in z [m]
    bezier_params       : dict controlling hinge shape (all keys optional):
                            's0_top', 's0_bot' — reach along face-i edges [m]
                            's1_top', 's1_bot' — reach along face-k edges [m]
                            'bc_up_xy' — single upper-arc CP, absolute XY [m]
                            'bc_lo_xy' — single lower-arc CP, absolute XY [m]
                          Missing reach keys default to gap; missing CPs default to
                          symmetrically bowed arcs (gap away from the chord axis).
    mesh_size_face      : target element size in face bodies [m]
    mesh_size_hinge     : target element size inside the hinge [m]
    mesh_refine         : >1 → finer in-plane elements (smoother FD gradients, slower)
    n_z_layers          : prism layers through thickness
    verbose             : show gmsh output

    Returns
    -------
    nodes    : (N, 3) float64
    tets     : (M, 4) int32
    bc_masks : {'clamped': bool (N,), 'loaded': bool (N,)}
    """
    import gmsh

    # ── 1-3. Face geometry + hinge strip endpoints ───────────────────────────
    geo = compute_hinge_geometry(cs, gap=gap, bezier_params=bezier_params)
    face_verts    = geo['face_verts']
    hinge_data    = geo['hinge_data']
    clamped_faces = geo['clamped_faces']
    loaded_faces  = geo['loaded_faces']
    n_faces       = face_verts.shape[0]

    # Mesh element sizing — a discretization concern, derived from the resolved
    # geometry: face size from the panel span, hinge size from the gap, both
    # refined by mesh_refine (>1 → finer elements, smoother FD gradients, slower).
    if mesh_size_face is None:
        span = float(np.ptp(face_verts.reshape(-1, 2), axis=0).max())
        mesh_size_face = span / 8.0
    if mesh_size_hinge is None:
        mesh_size_hinge = max(gap * 1.5, 1e-5)
    r = max(float(mesh_refine), 1e-3)
    lc_f, lc_h = mesh_size_face / r, mesh_size_hinge / r

    # ── 4. gmsh session ───────────────────────────────────────────────────────
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 3 if verbose else 0)
    gmsh.model.add("kirigami")

    # Point registry: rounded XY key → gmsh tag
    _pt_cache: dict[tuple, int] = {}

    def _add_point(xy: np.ndarray, lc: float) -> int:
        key = (round(float(xy[0]), 12), round(float(xy[1]), 12))
        if key not in _pt_cache:
            _pt_cache[key] = gmsh.model.geo.addPoint(float(xy[0]), float(xy[1]), 0.0, lc)
        return _pt_cache[key]

    # Shared-edge registry: frozenset{pt_a, pt_b} → (canonical_tag, from_pt)
    _edge_cache: dict[frozenset, tuple] = {}

    def _get_line(pt_a: int, pt_b: int) -> int:
        """Return signed curve tag for pt_a → pt_b, creating if needed."""
        key = frozenset([pt_a, pt_b])
        if key in _edge_cache:
            ctag, from_pt = _edge_cache[key]
            return ctag if from_pt == pt_a else -ctag
        ctag = gmsh.model.geo.addLine(pt_a, pt_b)
        _edge_cache[key] = (ctag, pt_a)
        return ctag

    # ── 5. Build hinge surfaces ───────────────────────────────────────────────
    # CCW loop: p0_top → p0_bot (face fi edge)
    #         → p1_bot (lower arc forward: p0_bot → p1_bot)
    #         → p1_top (face fk edge)
    #         → p0_top (upper arc reversed: p1_top → p0_top)
    hinge_surf_tags = []
    hinge_curve_data = []

    for hd in hinge_data:
        pt_p0_top = _add_point(hd['p0_top'], lc_h)
        pt_p0_bot = _add_point(hd['p0_bot'], lc_h)
        pt_p1_top = _add_point(hd['p1_top'], lc_h)
        pt_p1_bot = _add_point(hd['p1_bot'], lc_h)

        # Segment counts proportional to arc length
        def _n_segs(pa, c, pb):
            bow = float(np.linalg.norm(c - 0.5*(pa+pb)))
            arc = float(np.linalg.norm(pb - pa)) + 2.0 * bow
            return max(8, int(np.ceil(arc / lc_h)))

        n_up = _n_segs(hd['p0_top'], hd['bc_up'], hd['p1_top'])
        n_lo = _n_segs(hd['p0_bot'], hd['bc_lo'], hd['p1_bot'])

        # Upper arc: p0_top → p1_top
        pts_up = _bezier_sample(hd['p0_top'], hd['bc_up'], hd['p1_top'], n_up)
        tags_up = ([pt_p0_top]
                   + [_add_point(pts_up[k], lc_h) for k in range(1, n_up)]
                   + [pt_p1_top])
        bez_up = [gmsh.model.geo.addLine(tags_up[k], tags_up[k+1]) for k in range(n_up)]

        # Lower arc: p0_bot → p1_bot
        pts_lo = _bezier_sample(hd['p0_bot'], hd['bc_lo'], hd['p1_bot'], n_lo)
        tags_lo = ([pt_p0_bot]
                   + [_add_point(pts_lo[k], lc_h) for k in range(1, n_lo)]
                   + [pt_p1_bot])
        bez_lo = [gmsh.model.geo.addLine(tags_lo[k], tags_lo[k+1]) for k in range(n_lo)]

        # Closing edges along face boundaries (shared with face outlines via _get_line)
        edge_fi = _get_line(pt_p0_top, pt_p0_bot)   # face fi side: p0_top → p0_bot
        edge_fk = _get_line(pt_p1_bot, pt_p1_top)   # face fk side: p1_bot → p1_top

        # CCW loop: edge_fi + bez_lo + edge_fk + (bez_up reversed)
        hinge_loop = gmsh.model.geo.addCurveLoop(
            [edge_fi]
            + bez_lo
            + [edge_fk]
            + [-l for l in reversed(bez_up)]
        )
        hinge_surf_tags.append(gmsh.model.geo.addPlaneSurface([hinge_loop]))

        hinge_curve_data.append({
            'fi': hd['fi'], 'lj': hd['lj'],
            'fk': hd['fk'], 'll': hd['ll'],
            'pt_p0_top': pt_p0_top, 'pt_p0_bot': pt_p0_bot,
            'pt_p1_top': pt_p1_top, 'pt_p1_bot': pt_p1_bot,
            'p0_top': hd['p0_top'], 'p0_bot': hd['p0_bot'],
            'p1_top': hd['p1_top'], 'p1_bot': hd['p1_bot'],
            'up_is_prev_fi': hd['up_is_prev_fi'],
            'up_is_prev_fk': hd['up_is_prev_fk'],
        })

    # Lookup: (face_idx, corner_lj) → (role, hcd)
    corner_hinge: dict[tuple, tuple] = {}
    for hcd in hinge_curve_data:
        corner_hinge[(hcd['fi'], hcd['lj'])] = ('fi', hcd)
        corner_hinge[(hcd['fk'], hcd['ll'])] = ('fk', hcd)

    # ── 6. Build face surfaces ────────────────────────────────────────────────
    # Each hinge corner lj on a face is replaced by TWO outline points (p_top, p_bot
    # in CCW order).  The closing edge between them is shared with the hinge surface.
    face_surf_tags = []
    face_polys_2d  = []

    for fi in range(n_faces):
        fv = face_verts[fi]   # (4, 2) CCW

        outline: list[tuple[int, np.ndarray]] = []

        for lj in range(4):
            if (fi, lj) in corner_hinge:
                role, hcd = corner_hinge[(fi, lj)]
                # Insert the two strip endpoints in CCW face order.
                # "first" = endpoint on the incoming edge (from vertex (lj-1)%4).
                if role == 'fi':
                    if hcd['up_is_prev_fi']:
                        # adj_up is at (lj-1)%4 → p0_top is on the incoming edge
                        outline.append((hcd['pt_p0_top'], hcd['p0_top']))
                        outline.append((hcd['pt_p0_bot'], hcd['p0_bot']))
                    else:
                        # adj_dn is at (lj-1)%4 → p0_bot is on the incoming edge
                        outline.append((hcd['pt_p0_bot'], hcd['p0_bot']))
                        outline.append((hcd['pt_p0_top'], hcd['p0_top']))
                else:  # 'fk'
                    if hcd['up_is_prev_fk']:
                        outline.append((hcd['pt_p1_top'], hcd['p1_top']))
                        outline.append((hcd['pt_p1_bot'], hcd['p1_bot']))
                    else:
                        outline.append((hcd['pt_p1_bot'], hcd['p1_bot']))
                        outline.append((hcd['pt_p1_top'], hcd['p1_top']))
            else:
                tag = _add_point(fv[lj], lc_f)
                outline.append((tag, fv[lj]))

        outline_pt_tags = [t for t, _ in outline]
        face_polys_2d.append(np.array([xy for _, xy in outline]))

        n_out = len(outline_pt_tags)
        curves = [_get_line(outline_pt_tags[k], outline_pt_tags[(k + 1) % n_out])
                  for k in range(n_out)]

        face_loop = gmsh.model.geo.addCurveLoop(curves)
        face_surf_tags.append(gmsh.model.geo.addPlaneSurface([face_loop]))

    # ── 7. Mesh 2D ────────────────────────────────────────────────────────────
    gmsh.model.geo.synchronize()
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc_h * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc_f * 2.0)
    gmsh.model.mesh.generate(2)

    # ── 8. Extrude ────────────────────────────────────────────────────────────
    all_surfs = face_surf_tags + hinge_surf_tags
    gmsh.model.geo.extrude(
        [(2, tag) for tag in all_surfs],
        0.0, 0.0, float(sheet_thickness),
        numElements=[n_z_layers],
        recombine=False,
    )
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)

    # ── 9-10. Extract nodes + 3D elements (prisms split into tets) ────────────
    coords, tets = _extract_nodes_and_tets(gmsh)
    gmsh.finalize()

    # ── 11. BC masks (geometric: point-in-face-polygon, robust to re-meshing) ─
    bc_masks = _compute_bc_masks(coords, face_polys_2d, clamped_faces, loaded_faces)

    return coords, tets, bc_masks
