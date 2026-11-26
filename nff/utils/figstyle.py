"""Shared figure styling: one charter, one palette, one backend switch.

Every plotting script (``nff_paper/figures/`` and the core diagnostics) draws from here. Before this module the rcParams
block was copy-pasted verbatim into 16 scripts, ``apply_charter`` was defined four times, and two
palettes disagreed on what "ink" means (``#1A1A1A`` vs ``#212529``). The charter value wins.

Import order matters for the backend: ``use_agg()`` must run before the first ``pyplot`` import in
a process, which is why it is a function here rather than an import side effect.
"""

# ── Princeton charter palette ────────────────────────────────────────────────────────────────
ORANGE = "#F58025"   # faces / paper / cut pattern
TEAL   = "#2A9D8F"   # hinges / physical loss
RED    = "#D62828"   # loads / target / failure
GREY   = "#6C757D"   # clamps / secondary annotation
INK    = "#1A1A1A"   # text / edges / axes
PURPLE = "#6A4C93"   # boundary sliders
GREEN  = "#1F8A4C"   # deployed boundary points / geometric loss

# ── Experiment-vs-model palette (physical calibration figures) ───────────────────────────────
# Distinct from the charter on purpose: these figures contrast MEASURED against SIMULATED, a
# distinction the charter has no colour for.
SIM   = "#08519c"    # simulated / model prediction
REAL  = "#E8590C"    # measured / experimental
MUTE  = "#adb5bd"    # de-emphasised background series
LIT   = GREY         # literature / reference value
FAIL  = RED          # fracture, tear, out-of-range
TRAIN = "#C9CED4"    # faint training-set cloud behind a held-out split

_RC = {
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.6,
    "axes.edgecolor": INK, "axes.linewidth": 0.9, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": GREY, "ytick.color": GREY,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "legend.frameon": True, "legend.framealpha": 0.9, "legend.edgecolor": "#D3D6DB",
}


def use_agg():
    """Select the headless backend. Call BEFORE importing pyplot, then import it."""
    import matplotlib
    matplotlib.use("Agg")


def apply_charter(**overrides):
    """Apply the project rcParams. Extra keyword pairs override individual entries."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({**_RC, **overrides})


# ── Page geometry and the printed type ladder ────────────────────────────────────────────────
# A figure's type size is decided by the canvas it is authored on, not by the size it is exported
# at: LaTeX scales the whole graphic to the requested width, so a canvas twice as wide as it prints
# halves every label. Authoring at the printed size makes an authored point size the printed point
# size, and the ladder below then means the same thing in every figure of the paper.

TEXT_W_BP = 370.6     # \textwidth = 31pc (paper/sn-jnl.cls) in PostScript points

FS_PANEL = 8.0        # panel letters (bold)
FS_BASE  = 8.0        # axis labels, panel titles, body text
FS_TICK  = 7.2        # tick labels
FS_ANNOT = 6.6        # in-plot annotation, legends, scale bars, colorbar labels
FS_MICRO = 6.0        # hard floor: below this the page is unreadable, so it is an error


def page_bp(width_frac=1.0, aspect=1.0):
    """Canvas size in points for a graphic included at ``width_frac`` of ``\\textwidth``.

    Args:
        width_frac: the multiplier in ``\\includegraphics[width=...\\textwidth]``.
        aspect: width / height of the intended canvas.

    Returns:
        ``(width_bp, height_bp)``.
    """
    w = float(width_frac) * TEXT_W_BP
    return w, w / float(aspect)


SLIDE_W_BP = 408.0    # 16:9 beamer text block at aspectratio=169: 144 mm, in PostScript points


def slide_bp(width_frac=1.0, aspect=1.0):
    """Canvas size in points for a graphic shown at ``width_frac`` of a 16:9 slide's text block.

    The counterpart of ``page_bp`` for the other medium this project publishes into. A slide is
    both wider and far shallower than a column, so a plate authored for the page is scaled down to
    fit a slide's height and its type shrinks with it; authoring against this width instead keeps
    an authored point size a projected point size, exactly as ``page_bp`` does for print.

    Args:
        width_frac: fraction of the slide text block the graphic is shown at.
        aspect: width / height of the intended canvas.

    Returns:
        ``(width_bp, height_bp)``.
    """
    w = float(width_frac) * SLIDE_W_BP
    return w, w / float(aspect)


def figsize_in(w_bp, h_bp):
    """Points to the inches matplotlib wants for ``figsize``."""
    return (w_bp / 72.0, h_bp / 72.0)


def printed_pt(printed, scale=1.0):
    """Authored point size that prints at ``printed`` pt once the figure is scaled by ``scale``.

    ``scale`` is printed width / authored width. At 1.0 this is the identity, which is the case
    every figure should be in; it exists for the one plate whose content forces a larger canvas.
    """
    return printed / float(scale)


def paper_rc(scale=1.0, **overrides):
    """rcParams for a manuscript figure: one family, the ladder, and embeddable fonts.

    The family is named outright rather than left to matplotlib's default, so no host with a
    different font path can substitute silently. Type 42 is not optional: Type 3 text is drawn as
    filled outlines, is neither selectable nor searchable, and is rejected at submission.

    Args:
        scale: printed width / authored width, for a canvas that cannot be authored at print size.
        **overrides: extra rcParams applied last.

    Returns:
        A dict to splat into ``apply_charter``.
    """
    fs = lambda printed: printed_pt(printed, scale)
    return {"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
            "font.size": fs(FS_BASE), "axes.labelsize": fs(FS_BASE),
            "axes.titlesize": fs(FS_BASE), "figure.titlesize": fs(FS_BASE),
            "xtick.labelsize": fs(FS_TICK), "ytick.labelsize": fs(FS_TICK),
            "legend.fontsize": fs(FS_ANNOT),
            "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
            **overrides}


def text_width_bp(text, fontsize, weight="normal", family="sans-serif"):
    """Rendered width of ``text`` in points, from the real glyph outlines.

    A layout that reserves space for a label by counting characters and multiplying by a constant
    is silently tied to the font size that constant was tuned at. Measuring instead keeps a gutter
    or a column correct when the type ladder changes.
    """
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    prop = FontProperties(family=family, size=fontsize, weight=weight)
    return float(TextPath((0, 0), text, prop=prop).get_extents().width)


PANEL_LETTERS = None    # None -> read the environment on every call


def _panel_letters_on():
    """Whether ``panel_letter`` draws anything.

    A panel letter is print furniture: it is there so running prose can write "see (b)". A speaker
    names the panel out loud instead, so a deck build exports ``NFF_SLIDE_FIGURES=1`` and every
    generator that goes through ``panel_letter`` loses its letters without needing a flag of its
    own. Assign a bool to ``PANEL_LETTERS`` to decide it in process, for a notebook or a test.
    """
    if PANEL_LETTERS is not None:
        return bool(PANEL_LETTERS)
    import os
    return os.environ.get("NFF_SLIDE_FIGURES", "") != "1"


def panel_letters_on():
    """Whether ``panel_letter`` will draw. For a caller that reserves space for a letter."""
    return _panel_letters_on()


def panel_letter(target, letter, x=None, y=None, scale=1.0, color=INK, **kwargs):
    """Draw a panel letter in the house form: bare, bold, ``FS_PANEL``.

    Bare rather than parenthesised on the plate itself; the caption keeps ``\\textbf{(a)}`` because
    there the parentheses do syntactic work in running prose.

    Drawn as an empty string when ``_panel_letters_on()`` is false, rather than skipped: the Text
    artist still exists, so ``audit_type``, ``audit_margins`` and any caller that keeps the return
    value behave exactly as they do with letters on.

    Args:
        target: an Axes (letter placed in axes coordinates) or a Figure (figure coordinates).
        letter: the label, normally a single lowercase character.
        x, y: position in the target's coordinates. Defaults suit an Axes with visible spines.
        scale: printed width / authored width, as in ``paper_rc``.
        color: letter colour.
        **kwargs: forwarded to ``text``.

    Returns:
        The Text artist.
    """
    on_axes = hasattr(target, "transAxes")
    transform = target.transAxes if on_axes else target.transFigure
    if x is None:
        x = -0.20 if on_axes else 0.02
    if y is None:
        y = 1.06 if on_axes else 0.98
    if not _panel_letters_on():
        letter = ""
    return target.text(x, y, letter, transform=transform, color=color,
                       fontsize=printed_pt(FS_PANEL, scale), fontweight="bold",
                       ha=kwargs.pop("ha", "left"), va=kwargs.pop("va", "baseline"), **kwargs)


def audit_type(fig, floor=FS_MICRO, scale=1.0):
    """Every visible text artist in ``fig``, as the point size it will print at.

    Args:
        fig: the Figure to walk.
        floor: printed point size below which the figure is rejected.
        scale: printed width / authored width, as in ``paper_rc``.

    Returns:
        ``(min_pt, max_pt, offenders)`` with offenders as ``(printed_pt, text)``, largest first.

    Raises:
        ValueError: if any visible text prints below ``floor``.
    """
    from matplotlib.text import Text
    sizes = [(t.get_fontsize() * scale, t.get_text()) for t in fig.findobj(Text)
             if t.get_visible() and t.get_text().strip()]
    if not sizes:
        return (float("nan"), float("nan"), [])
    offenders = sorted([s for s in sizes if s[0] < floor - 1e-9], reverse=True)
    if offenders:
        listed = "; ".join(f"{pt:.2f} pt {txt!r}" for pt, txt in offenders[:8])
        raise ValueError(f"{len(offenders)} text artists print below the {floor} pt floor: {listed}")
    return (min(s[0] for s in sizes), max(s[0] for s in sizes), offenders)


def audit_margins(fig, pad_bp=0.25):
    """Ink that will be CUT by the page edge, since nothing here crops to content.

    ``save_paper_figure`` deliberately never passes ``bbox_inches``, so a label that overruns its
    gutter is silently guillotined rather than growing the page. This is the check for that.

    Measured with one ``Figure.get_tightbbox`` rather than by walking ``Text`` artists: a log axis
    keeps tick-label artists for ticks outside the view interval, and they report ``get_visible()``
    with stale positions, so a text walk is mostly false positives. ``get_tightbbox`` sees only what
    is drawn, handles rotated and mathtext extents, covers non-text artists at no extra cost, and
    honours ``set_in_layout(False)`` -- which is the escape hatch for anything deliberately outside.

    Args:
        fig: the Figure to measure, already drawn or drawable.
        pad_bp: overflow tolerated before this is an error. Not zero: the measurement runs at the
            figure's own dpi while the save is at 600, and glyph metrics move slightly with both dpi
            and FreeType build, so an exact floor fails on another machine for no visible reason.

    Returns:
        ``(worst_bp, worst_edge, offenders)``; ``worst_bp`` is negative when everything is inside,
        and offenders are ``(overflow_bp, edge, label)``, worst first.

    Raises:
        ValueError: if any edge is overrun by more than ``pad_bp``.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer() if hasattr(fig.canvas, "get_renderer") else None
    w_in, h_in = fig.get_size_inches()

    def _edges(bb):
        """Overflow past each page edge, in points; positive means outside.

        ⚠ ``bb`` must be in INCHES. ``Figure.get_tightbbox`` returns inches but
        ``Artist.get_tightbbox`` returns display pixels, so a child's box has to be converted or
        every number comes out 100/72 too large.
        """
        return {"left": -bb.x0 * 72.0, "bottom": -bb.y0 * 72.0,
                "right": (bb.x1 - w_in) * 72.0, "top": (bb.y1 - h_in) * 72.0}

    total = _edges(fig.get_tightbbox(renderer))
    worst_edge = max(total, key=total.get)
    worst = total[worst_edge]
    if worst <= pad_bp:
        return worst, worst_edge, []

    # Attribute the overflow to a child, because the fix is that child's margin rather than the
    # offending glyph. Only the edges the figure as a whole overruns are worth reporting.
    offenders = []
    for child in fig.get_children():
        if not getattr(child, "get_in_layout", lambda: True)():
            continue
        try:
            bb = child.get_tightbbox(renderer)
        except (AttributeError, TypeError, ValueError):
            continue
        if bb is None:
            continue
        bb = bb.transformed(fig.dpi_scale_trans.inverted())          # display px -> inches
        for edge, over in _edges(bb).items():
            if over > pad_bp and total[edge] > pad_bp:
                offenders.append((float(over), edge, _artist_label(child)))
    offenders.sort(reverse=True)
    listed = "; ".join(f"{over:.2f} bp past the {edge} edge by {lab}"
                       for over, edge, lab in offenders[:6]) or "no single child accounts for it"
    raise ValueError(f"ink falls outside the page by up to {worst:.2f} bp and will be cut, since "
                     f"the figure is written without bbox_inches: {listed}")


def _artist_label(artist):
    """Something a reader can act on: the axes' y-label, or the artist's class."""
    for attr in ("get_ylabel", "get_xlabel", "get_text"):
        if hasattr(artist, attr):
            got = getattr(artist, attr)()
            if got and got.strip():
                return f"{type(artist).__name__} {got.strip()!r}"
    return type(artist).__name__


def _pdf_page_bp(path):
    """The MediaBox of a single-page PDF, or None if it cannot be read.

    Read from the bytes rather than trusted from ``figsize``, because that is exactly the step
    ``bbox_inches`` breaks: it crops to content, so the emitted page is not the authored size and
    the LaTeX scale factor silently becomes unknown.
    """
    import re
    with open(path, "rb") as fh:
        m = re.search(rb"/MediaBox\s*\[([^\]]*)\]", fh.read())
    if not m:
        return None
    v = [float(x) for x in m.group(1).split()]
    return (v[2] - v[0], v[3] - v[1]) if len(v) == 4 else None


def save_paper_figure(fig, out, w_bp, h_bp, dpi=600, scale=1.0, floor=FS_MICRO, png=True,
                      strict_margins=False):
    """Audit the type, write a vector PDF, and prove the emitted page is the authored size.

    Args:
        fig: the Figure to write.
        out: output path; the extension is replaced to produce the PDF and the PNG.
        w_bp, h_bp: the authored canvas, in points.
        dpi: raster resolution for the PNG and for any rasterised artist in the PDF.
        scale: printed width / authored width, as in ``paper_rc``.
        floor: printed point size below which the figure is rejected.
        png: also write a PNG companion, for viewing.
        strict_margins: reject clipped ink instead of reporting it. Default off only because
            switching every existing plate to strict at once would fail some of them at save time,
            after their compute; opt in per figure, then make it the default.

    Returns:
        ``(pdf_path, min_pt, max_pt)``.

    Raises:
        ValueError: if the type audit fails, ink is clipped under ``strict_margins``, or the emitted
            page is not the authored size.
    """
    import os
    lo, hi, _ = audit_type(fig, floor=floor, scale=scale)
    try:
        over, edge, _ = audit_margins(fig)
        print(f"[margin] closest ink is {-over:.2f} bp inside the {edge} edge")
    except ValueError as exc:
        if strict_margins:
            raise
        print(f"[margin] WARNING {exc}")
    stem = os.path.splitext(out)[0]
    if os.path.dirname(stem):
        os.makedirs(os.path.dirname(stem), exist_ok=True)
    pdf = stem + ".pdf"
    fig.savefig(pdf, dpi=dpi)                       # NO bbox_inches: it would crop away the scale
    if png:
        fig.savefig(stem + ".png", dpi=dpi)
    got = _pdf_page_bp(pdf)
    if got is not None and (abs(got[0] - w_bp) > 0.5 or abs(got[1] - h_bp) > 0.5):
        raise ValueError(f"{pdf} is {got[0]:.1f} x {got[1]:.1f} bp, authored "
                         f"{w_bp:.1f} x {h_bp:.1f} bp -- the print scale is not what it claims")
    return pdf, lo, hi


def charter_diverging(low=TEAL, high=ORANGE, name="charter_div"):
    """Diverging colormap built from the charter, for a SIGNED field about zero.

    Defined here rather than in each figure so a signed quantity looks the same everywhere and no
    script reaches for matplotlib's built-ins. ``coolwarm`` and friends are outside the palette:
    their blue has no meaning in this project, and next to a charter figure they read as a
    different document.

    Pair it with a ``TwoSlopeNorm`` centred on 0 so the white midpoint really is zero.
    """
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, [low, "#FFFFFF", high])


def finite_sorted(x, *ys):
    """Drop non-finite samples across all arrays, then sort every array by ``x``.

    Args:
        x:  (n,) sort key.
        *ys: any number of (n,) companion arrays.

    Returns:
        Tuple ``(x, *ys)`` filtered and sorted, same length in every entry.
    """
    import numpy as np
    x = np.asarray(x, float)
    mask = np.isfinite(x)
    ys = tuple(np.asarray(y, float) for y in ys)
    for y in ys:
        mask &= np.isfinite(y)
    order = np.argsort(x[mask], kind="stable")
    return (x[mask][order],) + tuple(y[mask][order] for y in ys)
