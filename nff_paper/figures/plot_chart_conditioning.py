"""The Stage-0 chart ablation plate.

One figure, three bands: what the chart removes (geometry, no physics), how that shows up in the
conditioning, and whether independent starts reach the same design. Reads only the ``.npz`` files
written by ``diagnose_chart.py`` and ``ensemble_charts.py``, so every panel is regenerable
without re-running any physics.

    JAX_PLATFORMS=cpu conda run -n kgnn_mac python nff_paper/figures/plot_chart_conditioning.py
"""

import argparse
import os

import numpy as np

from nff.utils.figstyle import GREY, INK, ORANGE, RED, TEAL, apply_charter, use_agg

use_agg()
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.gridspec import GridSpec                               # noqa: E402

C_A, C_B, C_C = TEAL, RED, ORANGE
LABEL = {'A': 'closed chart', 'B': 'free vertices', 'C': 'free vertices + penalty'}


def _binned(rows, col, edges):
    """Mean of ``col`` per bin of column 0, with empty bins dropped."""
    x, y = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
        if m.sum() >= 3:
            x.append(np.sqrt(lo * hi))
            y.append(rows[m, col].mean())
    return np.array(x), np.array(y)


def _panel_label(ax, letter):
    ax.text(-0.14, 1.06, letter, transform=ax.transAxes, fontsize=12, fontweight='bold',
            color=INK, va='bottom', ha='left')


def panel_a_removed(ax, p0):
    """The flat sheet, and the stiffest direction the chart cannot express."""
    y0 = p0['y0']
    B, Q_A = p0['B'], p0['Q_A']
    n_faces = int(p0['n_faces']); max_nodes = int(p0['max_nodes'])
    _, _, Vt = np.linalg.svd(B, full_matrices=False)
    u = Vt[0] - Q_A @ (Q_A.T @ Vt[0])            # a tearing mode, orthogonal to the chart image
    u /= np.linalg.norm(u)
    amp = 0.022 * np.sqrt(y0.size)               # small enough that the panels stay readable

    base = y0.reshape(n_faces, max_nodes, 2)
    for f in range(n_faces):
        ax.fill(base[f, :, 0], base[f, :, 1], facecolor='#EEE7DF', edgecolor=GREY,
                lw=0.8, zorder=1)
    torn = (y0 + amp * u).reshape(n_faces, max_nodes, 2)
    for f in range(n_faces):
        ax.fill(torn[f, :, 0], torn[f, :, 1], facecolor='none', edgecolor=RED,
                lw=1.3, zorder=2)
    yt = y0 + amp * u
    gaps = (B @ yt).reshape(-1, 2)
    # each x-row of B is (+1 at node1_x, -1 at node2_x): argmax recovers node1's flat index
    idx = np.array([int(np.argmax(B[2 * h])) for h in range(B.shape[0] // 2)])
    ax.scatter(yt[idx], yt[idx + 1], s=26, facecolor='none', edgecolor=RED, lw=1.2, zorder=3)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('what the chart cannot express', fontsize=9, color=INK)
    ax.text(0.5, -0.04, f'hinges open by up to {np.abs(gaps).max() * 406.4:.0f} mm',
            transform=ax.transAxes, ha='center', fontsize=7, color=RED)


def panel_b_validity(ax, p0):
    """Flat-sheet validity vs geometric step. The hero panel."""
    h_max = float(p0['h_max_mm'])
    edges = np.logspace(np.log10(0.8), np.log10(h_max), 14)
    for key, colour, lab in (('valid_A', C_A, LABEL['A']), ('valid_BC', C_B, 'free vertices (B, C)')):
        rows = p0[key]
        for col, ls, tag in ((1, '-', 'connected'), (3, '--', 'non-inverted')):
            x, y = _binned(rows, col, edges)
            ax.plot(x, y, ls, color=colour, lw=1.9 if ls == '-' else 1.2,
                    label=f"{lab} — {tag}" if ls == '-' else None)
            if ls == '--':
                ax.plot(x, y, ls, color=colour, lw=1.2, alpha=0.55)
    ax.set_xscale('log'); ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel('geometric step  [mm]'); ax.set_ylabel('fraction still valid')
    ax.legend(fontsize=6.5, loc='center left')
    ax.set_title('flat-sheet validity', fontsize=9, color=INK)


def panel_c_gap(ax, p0):
    """Max hinge gap vs step -- threshold-free, and the widest separation in the plate."""
    h_max = float(p0['h_max_mm'])
    edges = np.logspace(np.log10(0.8), np.log10(h_max), 14)
    for key, colour, lab in (('valid_A', C_A, LABEL['A']), ('valid_BC', C_B, 'free vertices')):
        rows = p0[key]
        x, y = _binned(rows, 4, edges)
        ax.plot(x, np.maximum(y, 1e-16), 'o-', ms=3, color=colour, lw=1.8, label=lab)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('geometric step  [mm]'); ax.set_ylabel('max hinge gap  [mm]')
    ax.legend(fontsize=7); ax.set_title('the sheet comes apart', fontsize=9, color=INK)


def panel_d_scaling(ax):
    """Analytic dimension scaling -- free, no physics."""
    from nff.analysis.chart_ablation import chart_dims
    grids = [(m, m) for m in (3, 4, 5, 6, 8, 10, 14, 20)]
    d = [chart_dims(m, n) for m, n in grids]
    x = [m for m, _ in grids]
    ax.plot(x, [q['hinge_gap_fraction'] for q in d], 'o-', color=C_B, lw=1.8,
            label='directions that tear')
    ax.axhline(0.5, color=GREY, ls=':', lw=1.0)
    ax.text(14, 0.505, 'limit 1/2', fontsize=6.5, color=GREY, va='bottom')
    ax2 = ax.twinx()
    ax2.plot(x, [q['compression'] for q in d], 's--', color=C_A, lw=1.5, ms=4,
             label='compression')
    ax2.set_ylabel('chart compression  ×', color=C_A, fontsize=8)
    ax2.tick_params(axis='y', labelcolor=C_A); ax2.grid(False); ax2.set_ylim(0, 9)
    ax.scatter([3], [chart_dims(3, 3)['hinge_gap_fraction']], s=70, facecolor='none',
               edgecolor=INK, lw=1.3, zorder=5)
    ax.set_xlabel('tiles per side'); ax.set_ylabel('tearing fraction', color=C_B)
    ax.set_ylim(0.25, 0.55); ax.legend(fontsize=6.5, loc='center right')
    ax.set_title('scaling (analytic)', fontsize=9, color=INK)


def panel_e_spectrum(ax, p1):
    """Geometric Hessian spectrum, full vs restricted to the chart image."""
    H, Q_A = p1['H_y'], p1['Q_A']
    for M, colour, lab in ((H, C_B, f"free vertices ({H.shape[0]} dof)"),
                           (Q_A.T @ H @ Q_A, C_A, f"closed chart ({Q_A.shape[1]} dof)")):
        w = np.linalg.eigvalsh(M)
        o = np.argsort(-np.abs(w))
        pos = w[o] > 0
        idx = np.arange(len(w))
        ax.scatter(idx[pos], np.abs(w[o])[pos], s=13, color=colour, label=lab)
        ax.scatter(idx[~pos], np.abs(w[o])[~pos], s=13, facecolor='none', edgecolor=colour, lw=0.9)
    ax.set_yscale('log'); ax.set_xlabel('mode index'); ax.set_ylabel(r'$|\lambda|$')
    ax.legend(fontsize=6.5); ax.set_title('curvature spectrum (open = negative)',
                                          fontsize=9, color=INK)


def panel_f_transects(ax, p2):
    """Loss along random directions, on the shared millimetre axis."""
    for key, colour, lab in (('transect_A', C_A, LABEL['A']), ('transect_B', C_B, 'free vertices')):
        a = p2[key]
        n = 22
        for k in range(0, len(a) - n + 1, n):
            seg = a[k:k + n]
            o = np.argsort(seg[:, 0])
            ax.plot(seg[o, 0], seg[o, 1], color=colour, alpha=0.35, lw=0.9)
        ax.plot([], [], color=colour, lw=1.6, label=lab)
    hi = float(np.nanmax(p2['transect_B'][:, 1]))
    ax.set_yscale('log'); ax.set_ylim(5e-2, 1e6)
    ax.set_xlabel('signed geometric step  [mm]')
    ax.set_ylabel('design loss'); ax.legend(fontsize=7, loc='upper right')
    ax.text(0.02, 0.95, f'free-vertex peak {hi:.0e}\n(off scale)', transform=ax.transAxes,
            fontsize=6.5, color=RED, va='top')
    ax.set_title('landscape transects', fontsize=9, color=INK)


def panel_g_gradient(ax, p2):
    """Analytic vs finite-difference directional derivative: empirical gradient bias."""
    eps_vals = (1e-4, 1e-3, 1e-2)
    w = 0.34
    for i, (key, colour, lab) in enumerate((('grad_A', C_A, LABEL['A']),
                                            ('grad_B', C_B, 'free vertices'))):
        a = p2[key]
        for j, eps in enumerate(eps_vals):
            m = a[:, 0] == eps
            vals = np.log10(np.maximum(a[m, 3], 1e-12))
            parts = ax.violinplot([vals], positions=[j + (i - 0.5) * w], widths=w * 0.9,
                                  showextrema=False, showmedians=True)
            for b in parts['bodies']:
                b.set_facecolor(colour); b.set_alpha(0.55); b.set_edgecolor(colour)
            parts['cmedians'].set_color(INK); parts['cmedians'].set_linewidth(1.1)
        ax.plot([], [], color=colour, lw=4, alpha=0.6, label=lab)
    ax.set_xticks(range(len(eps_vals)))
    ax.set_xticklabels([f"{e:.0e}" for e in eps_vals])
    ax.set_xlabel('finite-difference step'); ax.set_ylabel(r'$\log_{10}$ relative error')
    ax.legend(fontsize=6.5); ax.set_title('gradient reliability', fontsize=9, color=INK)


def panel_h_pareto(ax, p1):
    """The penalty cannot buy conditioning -- nine decades of it."""
    par = p1['pareto']
    ax.plot(par[:, 0], par[:, 1], 'o-', color=C_C, lw=1.8, label='free vertices + penalty')
    ax.axhline(float(p1['cond_chart']), color=C_A, ls='--', lw=1.6,
               label='closed chart (no penalty)')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('hinge-gap penalty weight'); ax.set_ylabel(r'cond$_{\rm eff}$')
    ax.legend(fontsize=6.5); ax.set_title('the penalty steelman', fontsize=9, color=INK)


def panel_loss_spread(ax, ens, arms=('A', 'B', 'C')):
    """Final loss per seed. Torn designs are drawn hollow: a lower loss reached by a sheet
    that came apart is not a better design, and the panel must not read as if it were."""
    meta = ens['meta']
    for i, arm in enumerate(arms):
        m = meta[:, 0] == 'ABC'.index(arm)
        v, ok = meta[m, 3], meta[m, 8] > 0.5
        jit = np.random.default_rng(i).normal(0, 0.05, len(v))
        ax.scatter(np.full(len(v), i)[ok] + jit[ok], v[ok], s=20,
                   color=(C_A, C_B, C_C)[i], alpha=0.9, label=None)
        ax.scatter(np.full(len(v), i)[~ok] + jit[~ok], v[~ok], s=22, facecolor='none',
                   edgecolor=(C_A, C_B, C_C)[i], lw=1.1)
        ax.hlines(np.median(v), i - 0.2, i + 0.2, color=INK, lw=1.6)
    ax.set_yscale('log'); ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([LABEL[a].replace(' + ', '\n+ ') for a in arms], fontsize=6.5)
    ax.set_ylabel('final loss')
    ax.text(0.03, 0.03, 'hollow = sheet came apart', transform=ax.transAxes,
            fontsize=6.5, color=GREY)
    ax.set_title('lower loss, torn sheet', fontsize=9, color=INK)


def panel_design_spread(ax, ens, arms=('A', 'B', 'C')):
    meta, Y = ens['meta'], ens['y_end']
    ls = float(ens['length_scale'])
    n_nodes = Y.shape[1] // 2
    for i, arm in enumerate(arms):
        m = np.where(meta[:, 0] == 'ABC'.index(arm))[0]
        pw = [np.linalg.norm(Y[a] - Y[b]) / np.sqrt(n_nodes) * ls
              for ai, a in enumerate(m) for b in m[ai + 1:]]
        if not pw:
            continue
        parts = ax.violinplot([np.log10(np.maximum(pw, 1e-6))], positions=[i], widths=0.7,
                              showextrema=False, showmedians=True)
        for b in parts['bodies']:
            b.set_facecolor((C_A, C_B, C_C)[i]); b.set_alpha(0.6)
        parts['cmedians'].set_color(INK)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([LABEL[a].replace(' + ', '\n+ ') for a in arms], fontsize=6.5)
    ax.set_ylabel(r'$\log_{10}$ pairwise design distance [mm]')
    ax.set_title('...and the same design?', fontsize=9, color=INK)


def panel_success(ax, ens, arms=('A', 'B', 'C')):
    meta = ens['meta']
    for i, arm in enumerate(arms):
        m = meta[:, 0] == 'ABC'.index(arm)
        n = max(int(m.sum()), 1)
        conv = float(meta[m, 4].sum()) / n
        torn = float((meta[m, 8] == 0).sum()) / n
        ax.bar(i, conv, color=(C_A, C_B, C_C)[i], width=0.6)
        ax.bar(i, torn, bottom=conv, color=GREY, width=0.6, alpha=0.55)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([LABEL[a].replace(' + ', '\n+ ') for a in arms], fontsize=6.5)
    ax.set_ylim(0, 1.05); ax.set_ylabel('fraction of seeds')
    ax.set_title('converged-valid (grey: torn)', fontsize=9, color=INK)


def panel_basin(ax, basin, arms=('A', 'B', 'C')):
    """Median loss against how far the start was displaced. The validity version of this
    panel is a constant (100/0/0) and carries no information, so quality is plotted instead."""
    meta = basin['meta']
    for i, arm in enumerate(arms):
        m = meta[:, 0] == 'ABC'.index(arm)
        if not m.sum():
            continue
        xs, ys = [], []
        for nz in np.unique(meta[m, 2]):
            mm = m & (meta[:, 2] == nz)
            xs.append(np.median(meta[mm, 6]))
            ys.append(np.median(meta[mm, 3]))
        style = 'o-' if arm == 'A' else 'o--'
        ax.plot(xs, ys, style, color=(C_A, C_B, C_C)[i], lw=1.8, ms=4,
                label=LABEL[arm] + ('' if arm == 'A' else ' (torn)'),
                markerfacecolor=(C_A if arm == 'A' else 'none'))
    ax.set_yscale('log')
    ax.set_xlabel('start perturbation  [mm]'); ax.set_ylabel('median final loss')
    ax.legend(fontsize=6.0, loc='upper left')
    ax.set_title('basin: quality vs start', fontsize=9, color=INK)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/outputs/chart_ablation")
    ap.add_argument("--out", default="paper/Images/chart_conditioning.png")
    args = ap.parse_args()
    apply_charter()

    L = lambda n: np.load(os.path.join(args.data, n), allow_pickle=True)     # noqa: E731
    p0, p1, p2 = L("phase0.npz"), L("phase1.npz"), L("phase2.npz")
    p1 = dict(p1)
    p1['cond_chart'] = float(np.linalg.cond(p1['Q_A'].T @ p1['H_y'] @ p1['Q_A']))
    ens = L("ensemble_main.npz") if os.path.exists(
        os.path.join(args.data, "ensemble_main.npz")) else None
    basin = L("ensemble_basin.npz") if os.path.exists(
        os.path.join(args.data, "ensemble_basin.npz")) else None

    fig = plt.figure(figsize=(13.5, 11.0))
    gs = GridSpec(3, 4, figure=fig, hspace=0.42, wspace=0.34,
                  height_ratios=[1.0, 1.0, 1.0])

    axes = [fig.add_subplot(gs[0, j]) for j in range(4)]
    panel_a_removed(axes[0], p0); panel_b_validity(axes[1], p0)
    panel_c_gap(axes[2], p0); panel_d_scaling(axes[3])

    axes2 = [fig.add_subplot(gs[1, j]) for j in range(4)]
    panel_e_spectrum(axes2[0], p1); panel_f_transects(axes2[1], p2)
    panel_g_gradient(axes2[2], p2); panel_h_pareto(axes2[3], p1)

    axes3 = [fig.add_subplot(gs[2, j]) for j in range(4)]
    if ens is not None:
        panel_loss_spread(axes3[0], ens); panel_design_spread(axes3[1], ens)
        panel_success(axes3[2], ens)
    if basin is not None:
        panel_basin(axes3[3], basin)

    for ax, letter in zip(axes + axes2 + axes3, 'abcdefghijkl'):
        _panel_label(ax, letter)

    fig.text(0.5, 0.055,
             'All arms share one closed 3×3 sheet, one target, one physics model and the same '
             'starting geometry per seed; only the Stage-0 chart differs. Curvature and step size '
             'are measured in RMS panel-vertex displacement, so the charts are comparable.',
             ha='center', fontsize=7.5, color=GREY)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
