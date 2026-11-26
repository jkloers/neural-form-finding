"""Exact SVG writer for line/quadratic-Bezier paths, at 1:1 millimetre scale.

Hand-rolled because the environment has no SVG writer package, and because nothing else is needed:
the geometry arrives as exact control points and leaves as ``M``/``L``/``Q``/``C``/``Z`` commands,
so the only quantisation anywhere is the decimal precision of the emitted numbers (6 decimals at mm
scale is 1 nm).

One user unit is one millimetre, matching the convention ``nff.topology.cut_dxf`` already uses. The
project works y-up and SVG is y-down, which is handled by a single wrapping transform rather than by
negating coordinates -- an isometry maps a quadratic Bezier's control points to the image curve's
control points, so nothing is resampled and the numbers in the file stay readable as world values.

Pure numpy; this module imports nothing from ``nff``.
"""

import numpy as np

from nff_paper.topology_art.quad_warp import BezierPath, CompoundPath

FILL_STYLE = {"fill": "#1A1A1A", "fill-rule": "evenodd", "stroke": "none"}

DEFAULT_STYLE = {
    "fill": "none",
    "stroke": "#1A1A1A",
    "stroke-width": "0.4",
    "stroke-linecap": "round",
    "stroke-linejoin": "round",
}


def _fmt(value, precision):
    text = f"{float(value):.{precision}f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def _pair(point, precision):
    return f"{_fmt(point[0], precision)},{_fmt(point[1], precision)}"


def path_data_fn(path: BezierPath, *, precision: int = 6, curve: str = "q") -> str:
    """SVG ``d`` attribute for one path.

    Args:
        path: the exact path.
        precision: decimals emitted; at mm scale 6 is nanometre resolution.
        curve: ``"q"`` emits quadratics directly; ``"c"`` degree-elevates each to its EXACT cubic
            (``c1 = b0 + 2/3*(b1-b0)``, ``c2 = b2 + 2/3*(b1-b2)``) for consumers without ``Q``.

    Returns:
        The ``d`` string.
    """
    if curve not in ("q", "c"):
        raise ValueError(f"curve must be 'q' or 'c', got {curve!r}")
    if isinstance(path, CompoundPath):
        if not all(sub.closed for sub in path.subpaths):
            raise ValueError("every subpath of a filled compound path must be closed")
        return " ".join(path_data_fn(sub, precision=precision, curve=curve)
                        for sub in path.subpaths)
    out = [f"M{_pair(path.start, precision)}"]
    current = np.asarray(path.start, float)
    for kind, points in path.segments:
        if kind == "L":
            out.append(f"L{_pair(points[0], precision)}")
            current = np.asarray(points[0], float)
        elif kind == "A":
            (rx, ry), (large_arc, sweep), end = points
            out.append(f"A{_fmt(rx, precision)},{_fmt(ry, precision)} 0 "
                       f"{int(large_arc)},{int(sweep)} {_pair(end, precision)}")
            current = np.asarray(end, float)
        elif kind == "Q":
            control, end = np.asarray(points[0], float), np.asarray(points[1], float)
            if curve == "q":
                out.append(f"Q{_pair(control, precision)} {_pair(end, precision)}")
            else:
                c1 = current + (2.0 / 3.0) * (control - current)
                c2 = end + (2.0 / 3.0) * (control - end)
                out.append(
                    f"C{_pair(c1, precision)} {_pair(c2, precision)} {_pair(end, precision)}"
                )
            current = end
        else:
            raise ValueError(f"unknown segment kind {kind!r}")
    if path.closed:
        out.append("Z")
    return " ".join(out)


def write_svg_fn(
    out_path: str,
    layers: dict[str, list[BezierPath]],
    *,
    width_mm: float,
    height_mm: float,
    origin: tuple[float, float] = (0.0, 0.0),
    styles: dict[str, dict] | None = None,
    y_up: bool = True,
    background: str | None = None,
    curve: str = "q",
    precision: int = 6,
    title: str | None = None,
) -> dict:
    """Write layered paths to a 1:1 millimetre SVG.

    Args:
        out_path: destination ``.svg``.
        layers: ordered mapping of layer id to its paths; each becomes one ``<g>``.
        width_mm, height_mm: page size.
        origin: world position placed at the page's lower-left corner.
        styles: per-layer SVG presentation attributes, merged over ``DEFAULT_STYLE``.
        y_up: wrap the content so world y points up the page.
        background: page fill colour, or ``None`` for a transparent page.
        curve: ``"q"`` or ``"c"``; see ``path_data_fn``.
        precision: decimals emitted.
        title: optional ``<title>``.

    Returns:
        Report dict with ``path``, ``n_paths``, ``n_line``, ``n_quad``, ``page_mm``.
    """
    styles = styles or {}
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
        f'width="{_fmt(width_mm, 4)}mm" height="{_fmt(height_mm, 4)}mm" '
        f'viewBox="0 0 {_fmt(width_mm, 4)} {_fmt(height_mm, 4)}">',
    ]
    if title:
        lines.append(f"  <title>{title}</title>")
    if background:
        lines.append(f'  <rect x="0" y="0" width="{_fmt(width_mm, 4)}" '
                     f'height="{_fmt(height_mm, 4)}" fill="{background}"/>')

    if y_up:
        transform = (f'translate({_fmt(-origin[0], precision)},'
                     f'{_fmt(height_mm + origin[1], precision)}) scale(1,-1)')
    else:
        transform = f'translate({_fmt(-origin[0], precision)},{_fmt(-origin[1], precision)})'
    lines.append(f'  <g transform="{transform}">')

    n_paths = n_line = n_quad = n_arc = 0
    for name, paths in layers.items():
        if not paths:
            continue
        style = {**DEFAULT_STYLE, **styles.get(name, {})}
        attrs = " ".join(f'{k}="{v}"' for k, v in style.items())
        lines.append(f'    <g id="{name}" {attrs}>')
        for item in paths:
            # An item may carry its own style, so elements that must be painted in one order but
            # inked differently -- a leaf and the disc on it -- can share a layer. Splitting them
            # into two layers would put every disc on top of every leaf.
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], dict):
                path, override = item
            else:
                path, override = item, {}
            extra = "".join(f' {k}="{v}"' for k, v in override.items())
            lines.append(
                f'      <path d="{path_data_fn(path, precision=precision, curve=curve)}"{extra}/>')
            n_paths += 1
            subpaths = path.subpaths if isinstance(path, CompoundPath) else (path,)
            for sub in subpaths:
                for kind, _ in sub.segments:
                    if kind == "L":
                        n_line += 1
                    elif kind == "A":
                        n_arc += 1
                    else:
                        n_quad += 1
        lines.append("    </g>")

    lines.append("  </g>")
    lines.append("</svg>")

    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    return {
        "path": out_path,
        "n_paths": n_paths,
        "n_line": n_line,
        "n_quad": n_quad,
        "n_arc": n_arc,
        "page_mm": (float(width_mm), float(height_mm)),
    }


def fit_to_page_fn(
    bounds: tuple[float, float, float, float],
    page_mm: tuple[float, float],
    *,
    margin_mm: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Aspect-preserving similarity taking a world bounding box onto a page.

    Returns ``(matrix, offset)`` for ``nff_paper.topology_art.quad_warp.transform_path_fn``. A similarity is
    exact on Bezier control points, so page fitting costs nothing in precision.
    """
    min_x, min_y, max_x, max_y = (float(v) for v in bounds)
    page_w, page_h = (float(v) for v in page_mm)
    span_x, span_y = max(max_x - min_x, 1e-12), max(max_y - min_y, 1e-12)
    scale = min((page_w - 2 * margin_mm) / span_x, (page_h - 2 * margin_mm) / span_y)
    offset = np.array([
        0.5 * (page_w - scale * span_x) - scale * min_x,
        0.5 * (page_h - scale * span_y) - scale * min_y,
    ])
    return scale * np.eye(2), offset
