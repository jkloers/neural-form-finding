"""Audit every figure the manuscript includes, as it will actually appear on the page.

A figure's type size is not a property of the figure. It is the size it was authored at times the
factor LaTeX scales the graphic by, and that factor lives in the ``.tex``, not in the plotting
script. This reads both ends: the width each graphic is included at, and the page size the file
actually carries, and reports the product.

Two things fail a figure:

    scale != 1.000   the canvas is not the size it prints at, so no authored point size means what
                     it says. The exception is a plate that declares a compensation (see PLATES),
                     and a figure carrying no type at all, which has nothing to shrink.
    Type 3 fonts     text drawn as filled outlines. Not selectable, not searchable, and rejected at
                     submission by Springer Nature.

The ``min pt`` column is REPORTED, not enforced. It is the smallest printed glyph box, and a
mathtext subscript is legitimately smaller than the size it is set in, so a low value here is
usually a subscript rather than a defect. The 6 pt floor is enforced where nominal sizes are still
known: ``figstyle.audit_type``, which every ``save_paper_figure`` call runs before it writes.

Run:
    python nff/scripts/figures/audit_paper_type.py
    python nff/scripts/figures/audit_paper_type.py --tex paper/sn-article.tex
"""

import argparse
import os
import re
import subprocess

from nff.utils.figstyle import TEXT_W_BP

# Plates whose content forces a canvas wider than the column. The value is the declared printed
# width over the authored width; a figure listed here is judged against ITS factor, not against 1.
PLATES = {"hinge_path_design": TEXT_W_BP / (7.30 * 72.0)}

_INCLUDE = re.compile(r"\\includegraphics\[width=([^\]]*)\]\{Images/([^}]+)\}")
_FRAC = re.compile(r"^\s*([0-9.]*)\s*\\(?:textwidth|linewidth)\s*$")


def _run(cmd):
    """Stdout of ``cmd``, or an empty string if the tool is missing or fails."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout
    except FileNotFoundError:
        return ""


def printed_width_bp(expr):
    """The width in points a ``width=`` expression resolves to, or None if it is not recognised.

    ``\\linewidth`` equals ``\\textwidth`` in this document's one-column body, so both map to the
    same number; a ``\\dimexpr`` subtracting a fixed length is handled because Figure 1 uses one.
    """
    m = _FRAC.match(expr)
    if m:
        return (float(m.group(1)) if m.group(1) else 1.0) * TEXT_W_BP
    m = re.match(r"\s*\\dimexpr\s*\\(?:textwidth|linewidth)\s*-\s*([0-9.]+)pt\s*\\relax\s*", expr)
    if m:
        return TEXT_W_BP - float(m.group(1)) * 72.0 / 72.27       # TeX points to PostScript points
    return None


def page_bp_of(path):
    """``(width, height)`` of a PDF's first page in points, or None for a raster file."""
    out = _run(["pdfinfo", path])
    m = re.search(r"Page size:\s+([0-9.]+) x ([0-9.]+)", out)
    return (float(m.group(1)), float(m.group(2))) if m else None


def font_types(path):
    """The distinct font types embedded in a PDF, e.g. ``{'CID TrueType'}``."""
    lines = _run(["pdffonts", path]).splitlines()[2:]
    return {ln[37:54].strip() for ln in lines if ln.strip()}


def min_glyph_pt(path, scale):
    """Smallest PRINTED glyph height in the document, from the real text boxes.

    ``pdftotext -bbox`` reports a box per word in the file's own points; multiplying by the page
    scale gives what the reader sees. Cap height runs about 0.72 em in DejaVu Sans, so the box is
    divided back out to report a point size rather than a height.
    """
    out = _run(["pdftotext", "-bbox", path, "-"])
    heights = [float(b) - float(t) for t, b in
               re.findall(r'yMin="([0-9.]+)"[^>]*yMax="([0-9.]+)"', out)]
    heights = [h for h in heights if h > 0.5]
    return min(heights) * scale / 0.72 if heights else float("nan")


def audit(tex_path):
    """Report every included figure. Returns the number that failed."""
    with open(tex_path) as fh:
        includes = _INCLUDE.findall(fh.read())
    root = os.path.join(os.path.dirname(tex_path), "Images")

    print(f"{'figure':<26}{'page bp':>16}{'printed':>10}{'scale':>8}{'min pt':>8}  fonts")
    print("-" * 96)
    failed = 0
    for expr, name in includes:
        path = os.path.join(root, name)
        stem = os.path.splitext(name)[0]
        want = printed_width_bp(expr)
        if not os.path.exists(path):
            print(f"{stem:<26}{'MISSING':>16}"); failed += 1; continue
        if not name.endswith(".pdf"):
            print(f"{stem:<26}{'raster ' + name.rsplit('.', 1)[1]:>16}"
                  f"{want:>10.1f}{'--':>8}{'--':>8}  not vector"); failed += 1; continue

        page = page_bp_of(path)
        scale = (want / page[0]) if (page and want) else float("nan")
        expect = PLATES.get(stem, 1.0)
        types = font_types(path) or {"none"}
        lo = min_glyph_pt(path, scale)

        has_type = types != {"none"}
        bad = []
        # A figure with no text has no type to shrink, so the scale rule does not apply to it.
        if has_type and abs(scale - expect) > 0.005:
            bad.append(f"scale {scale:.3f} != {expect:.3f}")
        if any("Type 3" in t for t in types):
            bad.append("Type 3")
        failed += bool(bad)

        print(f"{stem:<26}{page[0]:>7.1f} x{page[1]:>7.1f}{want:>10.1f}{scale:>8.3f}"
              f"{lo:>8.1f}  {', '.join(sorted(types))}"
              + (f"   <-- {'; '.join(bad)}" if bad else ""))
    print("-" * 96)
    print(f"{len(includes) - failed}/{len(includes)} figures on policy"
          + (f"; {failed} to fix" if failed else ""))
    return failed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tex", default="paper/sn-article.tex")
    raise SystemExit(1 if audit(ap.parse_args().tex) else 0)


if __name__ == "__main__":
    main()
