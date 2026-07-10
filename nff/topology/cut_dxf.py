"""DXF export of the flat kirigami cut pattern — true 1:1, units = mm, for laser cutting.

Consumes the SAME Shapely objects as the PNG / A4 renderers (:func:`nff.topology.cut_pattern.
build_cut_geometry`), so the DXF is a vector twin of the printed pattern: what we cut == what we
simulated. Two toolpath conventions:

- ``outline`` (default, faithful): the boundary of ``cut_sheet(geom)`` — every kerf slot, tip
  fillet and the sheet edge as exact closed loops. The laser follows the outline and the thin
  ``w_c`` slots drop out. Reproduces the exact simulated geometry (do kerf compensation in the
  laser software).
- ``centerline`` (thin-kerf): the clean sheet outline plus each slot's centreline as a single open
  line (the laser's own kerf gives the width), with a small relief circle of radius ``rho`` at each
  retracted hinge tip. Faster to cut; drops the exact slot width.

Pure geometry + ezdxf — no ``nff`` runtime dependencies beyond the sibling ``cut_pattern`` module.
"""

import numpy as np
import ezdxf

from nff.topology.cut_pattern import cut_sheet


def _iter_polygons(geom):
    """Yield the non-empty Polygon parts of a Polygon / MultiPolygon."""
    for g in getattr(geom, "geoms", [geom]):
        if not g.is_empty and g.geom_type == "Polygon":
            yield g


def _add_ring(msp, ring, layer):
    """Add one closed ring as an LWPOLYLINE; returns 1 if written, 0 if degenerate."""
    pts = [(float(x), float(y)) for x, y in ring.coords]
    if pts and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 2:
        return 0
    msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": layer})
    return 1


def export_cut_geometry_dxf(geom, out_path, *, mode="outline", cut_layer="CUT",
                            perimeter_layer="PERIMETER", frame_layer="FRAME", note_layer="NOTES",
                            add_frame=True, frame_margin_mm=10.0, note=True, dxfversion="R2010"):
    """Write the flat cut pattern to a 1:1 DXF (units = mm).

    Args:
        geom: output of :func:`nff.topology.cut_pattern.build_cut_geometry` (mm).
        out_path: destination ``.dxf`` path.
        mode: ``"outline"`` (faithful slot/fillet loops) or ``"centerline"`` (thin-kerf lines +
            relief circles).
        cut_layer: layer for the interior cuts (voids / slots / relief circles).
        perimeter_layer: layer for the outer sheet perimeter (cut last so the part stays registered).
        frame_layer: layer for the raw-stock reference rectangle (not a cut).
        note_layer: layer for the size / scale text note.
        add_frame: draw a stock-outline rectangle ``frame_margin_mm`` beyond the pattern bounds.
        frame_margin_mm: margin of the stock rectangle around the pattern [mm].
        note: write a size + "1:1, units mm" text note under the pattern.
        dxfversion: ezdxf document version.

    Returns:
        dict: ``path``, ``mode``, ``n_loops`` (closed loops written), ``n_lines`` (open lines),
        ``sheet_mm`` (w, h), ``bounds_mm`` (minx, miny, maxx, maxy).
    """
    doc = ezdxf.new(dxfversion, setup=True)
    doc.units = ezdxf.units.MM                       # $INSUNITS = 4 -> CAD reads the drawing as mm
    msp = doc.modelspace()
    doc.layers.add(perimeter_layer, color=1)         # red — outer perimeter (all cut lines are red)
    doc.layers.add(cut_layer, color=1)               # red — interior cuts (separate layer for cut order)
    doc.layers.add(frame_layer, color=8)             # gray — stock reference (do NOT cut)
    doc.layers.add(note_layer, color=4)              # cyan — annotation (not a cut)

    minx, miny, maxx, maxy = geom["sheet"].bounds
    sheet_w, sheet_h = maxx - minx, maxy - miny
    n_loops = n_lines = 0

    if mode == "outline":
        # Boundary of (sheet - kerf slots): exact closed loops. Largest polygon's exterior = the
        # outer perimeter; every other ring (voids, smaller pieces) is an interior cut.
        part = cut_sheet(geom)
        polys = sorted(_iter_polygons(part), key=lambda p: p.area, reverse=True)
        for k, poly in enumerate(polys):
            n_loops += _add_ring(msp, poly.exterior, perimeter_layer if k == 0 else cut_layer)
            for ring in poly.interiors:
                n_loops += _add_ring(msp, ring, cut_layer)
    elif mode == "centerline":
        # Clean sheet outline + each slot centreline (single stroke) + a relief circle at each
        # retracted hinge tip (radius rho). The laser's kerf supplies the slot width.
        n_loops += _add_ring(msp, geom["sheet"].exterior, perimeter_layer)
        for (p0, p1) in geom["centerlines"]:
            msp.add_line((float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1])),
                         dxfattribs={"layer": cut_layer})
            n_lines += 1
        for h in geom["hinge_info"]:
            rho = float(h["rho"])
            if rho > 1e-9:
                tip = np.asarray(h["tip"], float)
                msp.add_circle((float(tip[0]), float(tip[1])), rho, dxfattribs={"layer": cut_layer})
                n_loops += 1
    else:
        raise ValueError(f"mode must be 'outline' or 'centerline', got {mode!r}")

    if add_frame:
        m = float(frame_margin_mm)
        fx0, fy0, fx1, fy1 = minx - m, miny - m, maxx + m, maxy + m
        msp.add_lwpolyline([(fx0, fy0), (fx1, fy0), (fx1, fy1), (fx0, fy1)],
                           close=True, dxfattribs={"layer": frame_layer})

    if note:
        txt = f"kirigami cut pattern  {sheet_w:.1f} x {sheet_h:.1f} mm  |  1:1 scale, units = mm  |  {mode}"
        h_txt = max(2.0, 0.02 * max(sheet_w, sheet_h))
        msp.add_text(txt, height=h_txt,
                     dxfattribs={"layer": note_layer}).set_placement((minx, miny - 2.5 * h_txt))

    doc.saveas(out_path)
    return dict(path=out_path, mode=mode, n_loops=n_loops, n_lines=n_lines,
                sheet_mm=(sheet_w, sheet_h), bounds_mm=(minx, miny, maxx, maxy))


def render_dxf_png(dxf_path, png_path, *, dpi=200):
    """Render a DXF to PNG (layer colours preserved) for a quick on-screen check of the toolpath."""
    import matplotlib.pyplot as plt
    from ezdxf.addons.drawing import RenderContext, Frontend
    from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

    doc = ezdxf.readfile(dxf_path)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_aspect("equal")
    Frontend(RenderContext(doc), MatplotlibBackend(ax)).draw_layout(doc.modelspace(), finalize=True)
    fig.savefig(png_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return png_path
