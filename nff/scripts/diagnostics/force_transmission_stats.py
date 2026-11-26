"""How each design carries its load, as a table -- the numbers behind the manuscript's Section 3.2.

Reads the ``load_path.npz`` already written into each run directory, so nothing is re-solved and
the statistics describe exactly the force field the figure draws. A run without one is skipped and
named.

Every quantity printed here is defined in :mod:`nff.closed.force_stats`. Two are easy to get
backwards and both are computed the safe way there: the checkerboard amplitude is measured on the
DETRENDED tile field, and the cut-field autocorrelation is reported for a sub-block as well as the
whole lattice.

Run:
    PYTHONPATH=$(pwd) JAX_PLATFORMS=cpu conda run -n kgnn_mac \
        python nff/scripts/diagnostics/force_transmission_stats.py \
            --runs data/outputs/runs/run_*_figmat_* \
            --out data/outputs/force_transmission.json
"""

import argparse
import json
import os
import pickle

import jax
jax.config.update("jax_enable_x64", True)   # before any solve; float32 shifts the energy split by
                                            # ~0.1%, which is enough to trip this tool's own check

import numpy as np                                                   # noqa: E402

from nff.closed.force_stats import (channel_concentration, checkerboard_amplitude,
                                    cut_family_split, elongation_alternation, energy_split,
                                    force_clustering, lattice_autocorrelation, mode_boundary_shear,
                                    mode_intensity, pressure_profile, region_mask,
                                    rotation_departure, route_directness, support_band_pressure,
                                    tile_pressure, tile_shape)
from nff.closed.load_paths import interface_decomposition, load_field


def _energy_split(run_dir, field):
    """The three-spring split of the stored hinge energy, which needs one solve per run.

    Kept behind a flag because every other statistic here reads the archives only. The solve is
    seconds; the reason to make it optional is that the tool should stay usable on a run whose
    config or parameters have moved on.
    """
    from nff.config.experiment import load_and_parse_config
    from nff.closed.load_paths import solve_deployed_state

    config_path = os.path.join(run_dir, "config.yaml")
    config = load_and_parse_config(config_path)
    import yaml
    raw = yaml.safe_load(open(config_path)) or {}
    material = raw.get('material', {}) or {}
    linearized = bool((raw.get('physics', {}) or {}).get('linearized_strains', False))

    results, _, _, _, _ = solve_deployed_state(config, os.path.join(run_dir, "best_params.pkl"))
    split = energy_split(results['valid_state'],
                         results['solution'].fields[-1],
                         results['reference_bond_vectors'],
                         k_stretch=float(material['k_stretch']),
                         k_shear=float(material['k_shear']),
                         k_rot=float(material['k_rot']),
                         linearized=linearized)

    # The split must BE the solve's own energy, not a plausible reconstruction of it.
    stored = float(np.asarray(field.energy).sum())
    residual = abs(split['total'] - stored) / max(abs(stored), 1e-30)
    if residual > 1e-6:
        raise SystemExit(
            f"{run_dir}: the three energy components sum to {split['total']:.6g} against the "
            f"{stored:.6g} stored in load_path.npz, a relative gap of {residual:.2%}. The split "
            f"does not describe this solve. Check physics.linearized_strains, which is "
            f"{linearized} here, and the material stiffnesses.")
    return {'shares': split['shares'], 'totals': split['totals'],
            'linearized': linearized, 'sum_check': residual,
            'per_hinge': split['per_hinge']}


SUPPORT_KEYS = ('bc_clamped', 'clamped_faces', 'x_roller_faces', 'y_roller_faces')


def _support_faces(config_path):
    """Clamped and roller tiles, read straight from the run's own config.

    ⚠ Every section is scanned rather than one named block. A run's saved ``config.yaml`` is the
    YAML as written, so the clamp sits under ``boundary_conditions.clamped_faces`` and the roller
    under ``tessellation.x_roller_faces``. The parsed in-memory name ``bc_clamped`` never appears
    in the file, and looking for it alone silently returns nothing.
    """
    if not os.path.exists(config_path):
        return []
    import yaml
    with open(config_path) as handle:
        raw = yaml.safe_load(handle) or {}

    faces = []
    for section in raw.values():
        if not isinstance(section, dict):
            continue
        for key in SUPPORT_KEYS:
            value = section.get(key)
            if isinstance(value, (list, tuple)):
                faces.extend(int(v) for v in value)
    return faces


def _cut_field(run_dir):
    """``sigmoid(z)`` on the cut lattice, or None when the run saved no parameters."""
    path = os.path.join(run_dir, "best_params.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as handle:
        params = pickle.load(handle)
    if not isinstance(params, dict) or 'z' not in params:
        return None
    z = np.asarray(params['z'], dtype=float)
    return 1.0 / (1.0 + np.exp(-z))


def analyse(run_dir, *, left_fraction=1 / 3, block_rows=5, resolve=False):
    """Every statistic for one run, as a plain dict ready for JSON."""
    stem = os.path.join(run_dir, "load_path")
    if not os.path.exists(stem + ".npz"):
        return None

    field = load_field(stem)
    dec = interface_decomposition(field)
    pressure = tile_pressure(field)
    compression, shear = mode_intensity(field, dec)
    aspect, _ = tile_shape(field.face_nodes)

    budget = {}
    if os.path.exists(stem + ".json"):
        with open(stem + ".json") as handle:
            budget = (json.load(handle) or {}).get('budget', {})

    weak = dec['f_norm'] < dec['f_norm'].mean()
    left = region_mask(field, (0.0, left_fraction))
    right = region_mask(field, (1.0 - left_fraction, 1.0))

    out = {
        'run': os.path.basename(run_dir),
        'n_faces': int(field.n_faces),
        'n_hinges': int(len(dec['f_n'])),
        'frame': dec['frame'],
        'f0': float(field.f0),
        'budget': budget,
        'shear_dominated_count': int(dec['shear_dominated'].sum()),
        'tension_count': int(((dec['f_n'] > 0) & ~dec['shear_dominated']).sum()),
        'compression_count': int(((dec['f_n'] <= 0) & ~dec['shear_dominated']).sum()),
        'concentration': channel_concentration(dec),
        'mode_boundary': mode_boundary_shear(field, dec),
        'routes': route_directness(field, dec),
        'checkerboard': checkerboard_amplitude(field, pressure),
        'cut_family_all': cut_family_split(field, dec),
        'cut_family_weak': cut_family_split(field, dec, mask=weak),
        'support_band': support_band_pressure(field, _support_faces(os.path.join(run_dir,
                                                                                "config.yaml"))),
        'pressure_profile': pressure_profile(field),
        'compression_shear_corr': float(np.corrcoef(compression, shear)[0, 1]),
        'aspect_median_left': float(np.median(aspect[left])),
        'aspect_median_right': float(np.median(aspect[right])),
        'elongation_left': elongation_alternation(field, region=left),
    }

    # Moment travels through enlarged tiles, or it does not. A correlation over all tiles is washed
    # out by the many carrying nothing, so the top decile is compared with the median instead.
    moment = np.zeros(field.n_faces)
    pairs = np.asarray(field.face_pairs)
    for column in (0, 1):
        np.add.at(moment, pairs[:, column], np.abs(dec['m_e']))
    top = moment >= np.percentile(moment, 90)
    areas = np.asarray(field.face_areas)
    out['moment_tile_area_ratio'] = float(areas[top].mean() / np.median(areas))

    out['clustering'] = force_clustering(field, dec)

    if resolve:
        energy = _energy_split(run_dir, field)
        out['departure'] = rotation_departure(field, dec, energy.pop('per_hinge'))
        out['energy'] = energy

    cut = _cut_field(run_dir)
    if cut is not None and cut.ndim == 2:
        out['cut_field'] = {
            'shape': [int(s) for s in cut.shape],
            'range': [float(cut.min()), float(cut.max())],
            'whole': lattice_autocorrelation(cut),
            'left_block': lattice_autocorrelation(cut, block=(slice(0, block_rows), slice(None))),
        }
    return out


def _fmt(value, width=8, decimals=3):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-".rjust(width)
    return f"{value:{width}.{decimals}f}"


def report(results):
    """The tables that back the manuscript, in the order the section makes its claims."""
    if any('departure' in r for r in results):
        print("\n── Leaving pure rotation: non-rotational %% of a hinge's energy, by its force ──")
        print(f"{'run':22s}{'Q1 weak':>9s}{'Q2':>8s}{'Q3':>8s}{'Q4 strong':>11s}"
              f"{'ratio':>8s}{'spearman':>10s}")
        for r in results:
            d = r.get('departure')
            if not d:
                continue
            q = [100 * v for v in d['share_by_force_quantile']]
            print(f"{r['run'][-18:]:22s}{q[0]:9.2f}{q[1]:8.2f}{q[2]:8.2f}{q[3]:11.2f}"
                  f"{d['ratio']:8.0f}{d['spearman']:+10.3f}")

    print("\n── Are the strongly loaded tiles compact? ────────────────────────────────────")
    print(f"{'run':22s}{'observed':>10s}{'chance':>8s}{'ratio':>7s}")
    for r in results:
        c = r['clustering']
        print(f"{r['run'][-18:]:22s}{c['observed']:10.3f}{c['chance']:8.2f}{c['ratio']:7.2f}")

    if any('energy' in r for r in results):
        print("\n── Where the deployment energy is stored (%% of the stored hinge energy) ──────")
        print(f"{'run':22s}{'rotation':>10s}{'axial':>8s}{'shear':>8s}{'sum check':>11s}")
        for r in results:
            e = r.get('energy')
            if not e:
                continue
            s = e['shares']
            print(f"{r['run'][-18:]:22s}{100*s['rotation']:10.1f}{100*s['axial']:8.1f}"
                  f"{100*s['shear']:8.1f}{e['sum_check']:11.1e}")

    print("\n── Force budget and how concentrated each channel is ─────────────────────────")
    print(f"{'run':22s}{'tens':>7s}{'shear':>7s}{'comp':>7s}{'shrdom':>8s}"
          f"{'Neff_T':>8s}{'Neff_C':>8s}{'Neff_S':>8s}")
    for r in results:
        b, c = r['budget'], r['concentration']
        print(f"{r['run'][-18:]:22s}{_fmt(b.get('tension_share'), 7)}{_fmt(b.get('shear_share'), 7)}"
              f"{_fmt(b.get('compression_share'), 7)}{r['shear_dominated_count']:8d}"
              f"{_fmt(c['tension']['n_eff'], 8, 1)}{_fmt(c['compression']['n_eff'], 8, 1)}"
              f"{_fmt(c['shear']['n_eff'], 8, 1)}")

    print("\n── Does shear sit on the tension/compression boundary? ───────────────────────")
    print(f"{'run':22s}{'enrich':>8s}{'z':>7s}{'strat':>7s}{'famA':>7s}{'famB':>7s}"
          f"{'cover':>8s}{'cut z':>8s}")
    for r in results:
        m = r['mode_boundary']
        print(f"{r['run'][-18:]:22s}{_fmt(m['enrichment'], 8, 2)}{_fmt(m['z'], 7, 2)}"
              f"{_fmt(m['enrichment_stratified'], 7, 2)}"
              f"{_fmt(m['per_family']['a']['enrichment'], 7, 2)}"
              f"{_fmt(m['per_family']['b']['enrichment'], 7, 2)}"
              f"{_fmt(m['coverage'], 8, 3)}{_fmt(m['cutset_z'], 8, 2)}")
    print("  enrich > 1 with z > 2: shear is over-represented on the boundary. cut z near zero:")
    print("  deleting the shear interfaces leaves no purer a partition than deleting as many at")
    print("  random, so shear does NOT separate the two regions. Quote the one that fits the claim.")

    print("\n── Tension routes: are they direct? ──────────────────────────────────────────")
    print(f"{'run':22s}{'routes':>7s}{'tiles':>6s}{'longest':>8s}{'median':>8s}{'weighted':>9s}"
          f"{'principal':>10s}{'looping':>8s}")
    for r in results:
        rt = r['routes']
        longest = rt['route_sizes'][0] if rt['route_sizes'] else 0
        print(f"{r['run'][-18:]:22s}{rt['n_routes']:7d}{rt['tiles_on_routes']:6d}{longest:8d}"
              f"{_fmt(rt['directness_median'], 8)}{_fmt(rt['directness_weighted'], 9)}"
              f"{_fmt(rt['directness_principal'], 10)}{rt['n_looping']:8d}")

    print("\n── The slack band between the supports ───────────────────────────────────────")
    print(f"{'run':22s}{'n':>5s}{'<p>band':>9s}{'<p>rest':>9s}{'comp band':>11s}{'comp rest':>11s}")
    for r in results:
        s = r['support_band']
        print(f"{r['run'][-18:]:22s}{s.get('n_band', 0):5d}{_fmt(s.get('pressure_band'), 9, 2)}"
              f"{_fmt(s.get('pressure_rest'), 9, 2)}{_fmt(s.get('compressive_fraction_band'), 11, 2)}"
              f"{_fmt(s.get('compressive_fraction_rest'), 11, 2)}")

    print("\n── The two scales: smooth field, and the checkerboard on top of it ───────────")
    print(f"{'run':22s}{'raw':>8s}{'detrend':>9s}{'var %':>8s}{'rNN raw':>9s}{'rNN det':>9s}")
    for r in results:
        k = r['checkerboard']
        print(f"{r['run'][-18:]:22s}{_fmt(k['corr_raw'], 8)}{_fmt(k['corr_detrended'], 9)}"
              f"{_fmt(100 * k['variance_share'], 8, 1)}{_fmt(k['neighbour_corr_raw'], 9)}"
              f"{_fmt(k['neighbour_corr_detrended'], 9)}")

    print("\n── The two cut families, over the WEAK interfaces ────────────────────────────")
    print(f"{'run':22s}{'<fn>A':>9s}{'<fn>B':>9s}{'ratio':>8s}{'%tensA':>8s}{'%tensB':>8s}")
    for r in results:
        f = r['cut_family_weak']
        print(f"{r['run'][-18:]:22s}{_fmt(f['a']['mean_f_n'], 9, 4)}{_fmt(f['b']['mean_f_n'], 9, 4)}"
              f"{_fmt(f['normal_force_ratio'], 8, 2)}"
              f"{_fmt(100 * f['a']['tension_fraction'], 8, 1)}"
              f"{_fmt(100 * f['b']['tension_fraction'], 8, 1)}")

    print("\n── Tile shape, and where the moment goes ─────────────────────────────────────")
    print(f"{'run':22s}{'AR left':>9s}{'AR right':>9s}{'>1.5 L':>8s}{'nbr ang':>9s}"
          f"{'comp-shear':>11s}{'A_moment':>10s}")
    for r in results:
        e = r['elongation_left']
        print(f"{r['run'][-18:]:22s}{_fmt(r['aspect_median_left'], 9, 2)}"
              f"{_fmt(r['aspect_median_right'], 9, 2)}"
              f"{_fmt(100 * e['fraction_above_1p5'], 8, 0)}"
              f"{_fmt(e.get('neighbour_angle_median'), 9, 0)}"
              f"{_fmt(r['compression_shear_corr'], 11)}"
              f"{_fmt(r['moment_tile_area_ratio'], 10, 2)}")

    print("\n── Cut field: whole lattice against the left block ───────────────────────────")
    print(f"{'run':22s}{'whole l1':>10s}{'whole l2':>10s}{'left l1':>9s}{'left l2':>9s}")
    for r in results:
        cut = r.get('cut_field')
        if not cut:
            continue
        w, b = cut['whole']['axis_0'], cut['left_block']['axis_0']
        print(f"{r['run'][-18:]:22s}{_fmt(w['lag_1'], 10, 2)}{_fmt(w['lag_2'], 10, 2)}"
              f"{_fmt(b['lag_1'], 9, 2)}{_fmt(b['lag_2'], 9, 2)}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="run directories holding load_path.npz")
    p.add_argument("--out", default=None, help="write the full statistics here as JSON")
    p.add_argument("--left-fraction", type=float, default=1 / 3,
                   help="fraction of the sheet width counted as the left region")
    p.add_argument("--block-rows", type=int, default=5,
                   help="rows of the cut lattice forming the left block")
    p.add_argument("--resolve", action="store_true",
                   help="re-solve each run to split the stored hinge energy into its three "
                        "springs; the only statistic here that needs more than the archives")
    args = p.parse_args()

    results, skipped = [], []
    for run_dir in args.runs:
        result = analyse(run_dir, left_fraction=args.left_fraction, block_rows=args.block_rows,
                         resolve=args.resolve)
        if result is None:
            skipped.append(run_dir)
        else:
            results.append(result)

    if skipped:
        print(f"skipped {len(skipped)} run(s) with no load_path.npz:")
        for run_dir in skipped:
            print(f"  {run_dir}")
    if not results:
        raise SystemExit("no run directory carried a load_path.npz")

    report(results)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
