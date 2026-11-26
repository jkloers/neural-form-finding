"""One tidy table of every unit cell of every design, its family, and what it carries.

The inverse-design matrix already classifies each design's unit cells into families, and the
force-transmission table already gives each actuation a force fingerprint. Neither has ever been
read against the other. This tool builds the join: one row per (design, unit cell), carrying the
cell's two-parameter geometry, its family, and the response quantities of the four panels and four
hinges it owns.

⚠ THE TWO BLOCKS ARE NOT THE SAME EXPERIMENT, and the distinction governs what may be claimed.

    figmat  five actuations, ONE target rectangle held byte-identical across all five, one uniform
            start. The actuation is the only thing that varies, so this block alone licenses
            "the actuation selects the family".
    grip    ten grips, each with a DIFFERENT target: these configs carry no ``target_half_w``, so
            run_closed auto-fits the target to the untrained deployed bounding box anchored at the
            CLAMP centroid. Measured over the ten, the target centre moves by 5.86 units in x and
            3.24 in y on a sheet of half-width 11, and the half-width and half-height themselves
            spread by 7% and 18%. Actuation and target are therefore entangled by construction.

So the grip block must never be pooled into a statement of the form "actuation X recruits family
Y". It is admissible for anything measured WITHIN a sheet -- whether a family carries more
compression than another family of the same sheet is unaffected by where that sheet's target sat --
and there it is worth ten independent sheets rather than none.

For the same reason the families are FITTED on the figmat block and the grips are LABELLED with
that fitted model, so a family means one shape everywhere and the family definition is anchored on
the block whose target is controlled.

    JAX_PLATFORMS=cpu conda run -n kgnn_mac python \\
        nff/scripts/diagnostics/cell_family_corpus.py --figmat --grips
"""

import argparse
import csv
import glob
import json
import os

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np                                                       # noqa: E402

from nff.closed import force_stats as FS                                 # noqa: E402
from nff.closed import one_dof as OD                                     # noqa: E402
from nff.closed.cell_families import (cell_descriptors, cell_face_values,  # noqa: E402
                                      cell_lattice, family_map, fit_families)
from nff.closed.load_paths import (deployed_nodes, extract_force_field,  # noqa: E402
                                   force_budget, interface_decomposition,
                                   solve_deployed_state)
from nff.closed.shield_kinematics import _checkerboard_signs             # noqa: E402
from nff.closed.void_geometry import (build_void_index, pin_gaps,        # noqa: E402
                                      polygon_areas, void_octagons)
from nff.config.experiment import load_and_parse_config                  # noqa: E402
from nff.topology.closed_builder_jax import build_deploy_structure       # noqa: E402

FIGMAT = [
    ("data/outputs/runs/run_20260806_161121_figmat_1_deploy",     "I",   "Concentrated axial load"),
    ("data/outputs/runs/run_20260806_115352_figmat_2_cantilever", "II",  "Axial and transverse load"),
    ("data/outputs/runs/run_20260806_161121_figmat_3_moment",     "III", "Axial load and couple"),
    ("data/outputs/runs/run_20260806_161121_figmat_4_asym",       "IV",  "Eccentric axial load"),
    ("data/outputs/runs/run_20260806_115352_figmat_5_vertical",   "V",   "Self-equilibrated shear couple"),
]

CELL_COLUMNS = [
    "run", "label", "block", "cell", "row", "col",
    "long_mm", "short_mm", "span_mm", "aspect", "gain", "opening_deg",
    "family", "in_cluster",
    "theta_mean", "theta_absmax", "resid_absmax", "pressure_mean", "couple_mean",
    "f_n_mean", "f_n_absmax", "f_t_mean", "f_t_absmax", "f_mag_mean", "shear_frac",
    "energy_sum", "area_mm2", "pin_gap_max",
]


def _grip_runs():
    """The newest run folder of each ``gripNN_shape`` config."""
    out = []
    for i in range(10):
        hits = sorted(glob.glob(f"data/outputs/runs/*_grip{i:02d}_shape"))
        if hits:
            out.append((hits[-1], f"g{i:02d}", f"grip {i:02d}"))
    return out


def _actuation(config) -> dict:
    """What the run applies and how it is held -- the axis this whole corpus is indexed by."""
    topo = config.topology
    loads = topo.get('loads', []) or []
    faces = [int(l['face']) for l in loads]
    values = [float(l['value']) for l in loads]
    fx = sum(v for l, v in zip(loads, values) if int(l['dof']) == 0)
    fy = sum(v for l, v in zip(loads, values) if int(l['dof']) == 1)
    moment = sum(v for l, v in zip(loads, values) if int(l['dof']) == 2)
    return {
        'loads': loads, 'load_faces': faces,
        'n_loaded': len(loads),
        'sum_fx': fx, 'sum_fy': fy, 'sum_moment': moment,
        'resultant': float(np.hypot(fx, fy)),
        'drive': float(sum(abs(v) for v in values)),
        'self_equilibrated': bool(np.hypot(fx, fy) < 1e-9 * max(sum(abs(v) for v in values), 1.0)),
        'clamped': topo.get('bc_clamped') or topo.get('clamped_faces'),
        'x_roller': topo.get('x_roller_faces'), 'y_roller': topo.get('y_roller_faces'),
        'r_init': float(topo.get('r_init', float('nan'))),
        'init_noise': float(topo.get('init_noise', 0.0)),
        'target_fixed': 'target_half_w' in topo,
    }


def measure_sheet(run_dir: str, void_index, lattice, length_scale: float) -> dict:
    """Re-solve one design and reduce every field it produces onto its unit cells."""
    config = load_and_parse_config(os.path.join(run_dir, "config.yaml"))
    results, _tess, bond_energy, hinge_geometry, cut_axis = solve_deployed_state(
        config, os.path.join(run_dir, "best_params.pkl"))

    valid = results['valid_state']
    disp = results['solution'].fields[-1]
    theta = np.asarray(disp[:, 2])
    nodes = np.asarray(deployed_nodes(valid.face_centroids, valid.centroid_node_vectors, disp))

    field = extract_force_field(results, config.physics, bond_energy_fn=bond_energy,
                                hinge_geometry=hinge_geometry, cut_axis=cut_axis, step=-1)
    dec = interface_decomposition(field)
    face_pairs = np.asarray(field.face_pairs)

    M, N = int(config.topology['M']), int(config.topology['N'])
    fit = OD.fit_one_dof(theta, _checkerboard_signs(M, N).astype(float))

    # A hinge belongs to a cell only when BOTH its panels do. The cells are disjoint, so this is
    # unambiguous; a hinge between two cells belongs to neither and is left out of the reduction.
    owner = lattice.owner_face
    cell_of_face = np.full(M * N, -1, dtype=int)
    for c, faces in enumerate(owner):
        cell_of_face[faces] = c
    hinge_cell = np.where(cell_of_face[face_pairs[:, 0]] == cell_of_face[face_pairs[:, 1]],
                          cell_of_face[face_pairs[:, 0]], -1)

    def per_cell_hinge(values, reduce="mean"):
        values = np.asarray(values, dtype=float)
        out = np.zeros(lattice.n_cells)
        for c in range(lattice.n_cells):
            v = values[hinge_cell == c]
            if v.size == 0:
                out[c] = np.nan
            elif reduce == "mean":
                out[c] = v.mean()
            elif reduce == "sum":
                out[c] = v.sum()
            elif reduce == "absmax":
                out[c] = v[np.abs(v).argmax()]
            elif reduce == "frac":
                out[c] = float(v.mean())
        return out

    desc = cell_descriptors(void_index, lattice, nodes, length_scale)
    octagon_area = np.abs(polygon_areas(void_octagons(void_index, nodes)))[lattice.keep]
    gaps = pin_gaps(void_index, nodes)[lattice.keep]

    return {
        'config': config, 'run_dir': run_dir, 'descriptors': desc,
        'actuation': _actuation(config),
        'budget': force_budget(field),
        'psi': float(fit.psi), 'swing': float(fit.swing), 'rms_over_psi': float(fit.rms_over_psi),
        'cells': {
            'theta_mean': cell_face_values(theta, lattice, "mean"),
            'theta_absmax': cell_face_values(theta, lattice, "absmax"),
            'resid_absmax': cell_face_values(fit.residual, lattice, "absmax"),
            'pressure_mean': cell_face_values(FS.tile_pressure(field), lattice, "mean"),
            'couple_mean': cell_face_values(FS.tile_couple(field), lattice, "mean"),
            'f_n_mean': per_cell_hinge(dec['f_n']),
            'f_n_absmax': per_cell_hinge(dec['f_n'], "absmax"),
            'f_t_mean': per_cell_hinge(dec['f_t']),
            'f_t_absmax': per_cell_hinge(dec['f_t'], "absmax"),
            'f_mag_mean': per_cell_hinge(dec['f_mag']),
            'shear_frac': per_cell_hinge(dec['shear_dominated'].astype(float), "frac"),
            'energy_sum': per_cell_hinge(np.asarray(field.energy), "sum"),
            'area_mm2': octagon_area * length_scale ** 2,
            'pin_gap_max': gaps.max(axis=1) * length_scale,
        },
        'n_internal_hinges': int((hinge_cell >= 0).sum()),
        'n_hinges': int(len(face_pairs)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figmat", action="store_true", help="the five controlled-target actuations")
    ap.add_argument("--grips", action="store_true", help="the ten grips (each its own target)")
    ap.add_argument("--run", action="append", default=[],
                    help="RUN_DIR:LABEL:BLOCK, repeatable")
    ap.add_argument("--n-families", type=int, default=3)
    ap.add_argument("--min-size", type=int, default=3)
    ap.add_argument("--out", default="data/outputs/cell_families")
    args = ap.parse_args()

    sheets = []
    if args.figmat:
        sheets += [(d, l, "figmat", n) for d, l, n in FIGMAT]
    if args.grips:
        sheets += [(d, l, "grip", n) for d, l, n in _grip_runs()]
    for spec in args.run:
        run_dir, label, block = spec.split(":")
        sheets.append((run_dir, label, block, label))
    if not sheets:
        raise SystemExit("nothing to do: pass --figmat, --grips or --run")

    first = load_and_parse_config(os.path.join(sheets[0][0], "config.yaml"))
    M, N = int(first.topology['M']), int(first.topology['N'])
    length_scale = float(getattr(getattr(first, 'hinge_model', None), 'length_scale', 406.4))
    void_index = build_void_index(build_deploy_structure(M, N))
    lattice = cell_lattice(void_index)
    print(f"  sheet {M}x{N}: {lattice.shape[0]} x {lattice.shape[1]} disjoint unit cells "
          f"({lattice.n_cells} per sheet, {lattice.owner_face.size}/{M * N} panels covered)")

    measured = []
    for run_dir, label, block, name in sheets:
        cfg = load_and_parse_config(os.path.join(run_dir, "config.yaml"))
        if (int(cfg.topology['M']), int(cfg.topology['N'])) != (M, N):
            raise SystemExit(f"{run_dir} is posed on a {cfg.topology['M']}x{cfg.topology['N']} "
                             f"sheet, not {M}x{N}; the unit-cell lattice would not correspond")
        m = measure_sheet(run_dir, void_index, lattice, length_scale)
        m.update(label=label, block=block, name=name)
        measured.append(m)
        print(f"  solved {label:<5} [{block}] {name}", flush=True)

    # ── Families are fitted on the controlled-target block and the rest are LABELLED with it. ──
    fit_rows = [m for m in measured if m['block'] == 'figmat'] or measured
    fit_pool = np.vstack([m['descriptors'].sides for m in fit_rows])
    model, _ = fit_families(fit_pool, args.n_families)
    print(f"  families fitted on {len(fit_rows)} sheet(s) of block "
          f"'{fit_rows[0]['block']}', {len(fit_pool)} cells:")
    for c in range(model.n_families):
        print(f"    family {c}: {model.centroids[c, 0]:.0f} x {model.centroids[c, 1]:.0f} mm")

    for m in measured:
        labels = model.label(m['descriptors'].sides)
        m['labels'] = labels
        m['map'] = family_map(labels, lattice, min_size=args.min_size)

    os.makedirs(args.out, exist_ok=True)
    rows = []
    for m in measured:
        d, cells, grid = m['descriptors'], m['cells'], m['map']
        for c in range(lattice.n_cells):
            row, col = divmod(c, lattice.shape[1])
            rows.append({
                'run': m['run_dir'], 'label': m['label'], 'block': m['block'],
                'cell': c, 'row': row, 'col': col,
                'long_mm': d.long_side[c], 'short_mm': d.short_side[c], 'span_mm': d.span[c],
                'aspect': d.aspect[c], 'gain': d.gain[c], 'opening_deg': d.opening_deg[c],
                'family': int(m['labels'][c]), 'in_cluster': int(grid['mask'].ravel()[c]),
                **{k: cells[k][c] for k in CELL_COLUMNS[14:]},
            })
    cells_path = os.path.join(args.out, "cells.csv")
    with open(cells_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CELL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    runs = {}
    for m in measured:
        counts = np.bincount(m['labels'], minlength=model.n_families).tolist()
        runs[m['label']] = {
            'run': m['run_dir'], 'block': m['block'], 'name': m['name'],
            'actuation': {k: v for k, v in m['actuation'].items() if k != 'loads'},
            'loads': m['actuation']['loads'],
            'psi_deg': float(np.degrees(m['psi'])), 'swing_deg': float(np.degrees(m['swing'])),
            'rms_over_psi': m['rms_over_psi'],
            'composition': counts,
            'contiguity': float(m['map']['contiguity']['ratio']),
            'n_clustered': m['map']['n_clustered'],
            'opening_spread_deg': float(np.ptp(m['descriptors'].opening_deg)),
            'budget': {k: float(v) for k, v in m['budget'].items()
                       if isinstance(v, (int, float, np.floating))},
            'n_internal_hinges': m['n_internal_hinges'], 'n_hinges': m['n_hinges'],
        }
    meta = {
        'sheet': [M, N], 'lattice': list(lattice.shape), 'n_cells': lattice.n_cells,
        'length_scale_mm': length_scale, 'n_families': model.n_families,
        'family_centroids_mm': model.centroids.tolist(),
        'fitted_on': [m['label'] for m in fit_rows],
        'runs': runs,
    }
    json.dump(meta, open(os.path.join(args.out, "runs.json"), "w"), indent=1)

    print(f"\n  wrote {cells_path} ({len(rows)} cells) and runs.json")
    print("\n  composition (cells per family) and contiguity:")
    for label, r in runs.items():
        print(f"    {label:<5} [{r['block']:<6}] " + " ".join(f"{n:3d}" for n in r['composition'])
              + f"   {r['contiguity']:.2f}x chance   {r['n_clustered']:2d}/{lattice.n_cells}"
              + f"   psi={r['psi_deg']:+.1f} deg")


if __name__ == "__main__":
    main()
