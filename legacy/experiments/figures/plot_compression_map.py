"""Where the sheet SQUEEZES its hinges, drawn on the deployed tessellation.

The rotating-squares mechanism opens by ROTATION. A hinge driven into axial compression
(``eta_a = a / w_lig < 0``) is a ligament being squeezed rather than pivoted, so the work spent on
it never becomes opening. The path diagnostics already report compression as numbers; this puts it
back on the sheet, which is the only way to see WHERE it happens and whether it follows the load
path from the pulled tile to the supports.

Per hinge, the score is the PATH MINIMUM of ``eta_a`` over the whole deployment, not its endpoint.
Compression is transient -- a hinge squeezed at mid-deployment has already been squeezed whatever
the final state says -- and this matches ``nff/training/loss.py::_term_compression``, which takes a
softmax-min over load steps, so the picture and the penalty score the same thing.

Colour is diverging about ZERO, which is the only meaningful pivot: negative is compression, and the
two signs are physically different regimes rather than two ends of one scale.

Run:
    PYTHONPATH=$(pwd) JAX_PLATFORMS=cpu conda run -n kgnn_mac \
        python legacy/experiments/figures/plot_compression_map.py \
            --config data/configs/closed/sheet_15x10_rom_maxdeploy.yaml \
            --params <run_dir>/best_params.pkl --out <run_dir>/compression_map.png
"""

import argparse
import copy
import os
import pickle

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np

from nff.utils.figstyle import use_agg, apply_charter, INK, GREY
use_agg()
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.colors import TwoSlopeNorm                         # noqa: E402

from nff.config.experiment import load_and_parse_config            # noqa: E402
from nff.closed.setup import (build_closed_initial_state, init_closed_les_params,  # noqa: E402
                              build_surrogate_energy)
from nff.closed.hinge_paths import (build_hinge_geometry, extract_hinge_paths,      # noqa: E402
                                    resolve_length_scale)
from nff.closed.deploy import _deployed_hinge_xy                   # noqa: E402
from nff.stages.pipeline import forward_pipeline                   # noqa: E402
from nff.utils.visualization import write_deformed_into, plot_tessellation  # noqa: E402


def deployed_node_positions(valid_state, displacement):
    """(n_faces, max_nodes, 2) deployed node positions from the rigid-tile kinematics."""
    centroids = np.asarray(valid_state.face_centroids) + np.asarray(displacement[:, :2])
    theta = np.asarray(displacement[:, 2])
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cnv = np.asarray(valid_state.centroid_node_vectors)
    rot_x = cos_t[:, None] * cnv[:, :, 0] - sin_t[:, None] * cnv[:, :, 1]
    rot_y = sin_t[:, None] * cnv[:, :, 0] + cos_t[:, None] * cnv[:, :, 1]
    return centroids[:, None, :] + np.stack([rot_x, rot_y], axis=-1)


def build_compression_figure(tessellation, valid_state, displacement, eta_a_path, out_png,
                             label=""):
    """Deployed sheet with each hinge dotted by its path-minimum ``eta_a``.

    Args:
        eta_a_path: (n_steps, n_hinges) signed axial ratio over the deployment.

    Returns:
        The written path.
    """
    apply_charter()
    worst = np.asarray(eta_a_path).min(axis=0)                     # (n_hinges,) most-compressive
    hinge_xy = _deployed_hinge_xy(valid_state, displacement)
    deformed = write_deformed_into(tessellation, deployed_node_positions(valid_state, displacement))

    fig, ax = plt.subplots(figsize=(13.0, 9.0), dpi=170)
    plot_tessellation(deformed, ax=ax, show_faces=True, show_hinges=False, show_vertices=False,
                      show_face_indices=False, show_hinge_indices=False, show_target=False,
                      color_faces="#EDEDED", title=None)

    # Diverging about 0. If nothing is compressive the norm would be degenerate, so fall back to a
    # symmetric range rather than crashing on an all-tension sheet.
    lo, hi = float(worst.min()), float(worst.max())
    norm = (TwoSlopeNorm(vcenter=0.0, vmin=min(lo, -1e-6), vmax=max(hi, 1e-6))
            if lo < 0.0 < hi else None)
    sc = ax.scatter(hinge_xy[:, 0], hinge_xy[:, 1], c=worst, cmap="coolwarm_r", norm=norm,
                    s=46, edgecolors=INK, linewidths=0.35, zorder=6)
    cb = fig.colorbar(sc, ax=ax, fraction=0.030, pad=0.02)
    cb.set_label(r"path-minimum  $\eta_a = a / w_{lig}$      (negative = COMPRESSION)")

    n_comp = int((worst < -1e-6).sum())
    ax.set_title(f"Hinge axial compression over the deployment{'  —  ' + label if label else ''}\n"
                 f"{worst.size} hinges, {n_comp} ever compressive "
                 f"({100.0 * n_comp / worst.size:.1f}%),  worst {worst.min():+.4f}",
                 fontsize=11, color=INK)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--params", default=None, help="best_params.pkl; omit for the untrained design")
    p.add_argument("--steps", type=int, default=None,
                   help="load steps (default: the config's physics.num_load_steps)")
    p.add_argument("--out", default=None)
    p.add_argument("--label", default=None)
    args = p.parse_args()

    config = copy.deepcopy(load_and_parse_config(args.config))
    if args.steps:
        # num_load_steps IS the sampling resolution of the trajectory; the design, target and BCs
        # are untouched, so this is the same deployment observed more often. The config dataclasses
        # are frozen, hence the explicit bypass.
        object.__setattr__(config.physics, 'num_load_steps', int(args.steps))
    state, tessellation = build_closed_initial_state(config)
    config.topology['init_noise'] = 0.0
    config.topology['init_seed'] = 0
    params, static_features = init_closed_les_params(config)

    bond_energy, _, geometry_fn, _, w_lig_logit0 = build_surrogate_energy(
        config, static_features, state, params)
    if w_lig_logit0 is not None:
        params = {**params, 'w_lig_logit': w_lig_logit0}
    if args.params:
        with open(args.params, 'rb') as fh:
            params = {k: np.asarray(v) for k, v in pickle.load(fh).items()}

    results = forward_pipeline(
        state, config.target, config.validity, config.physics, map_type=config.mapping.type,
        map_params=params, static_features=static_features, load_specs=config.topology['loads'],
        bond_energy_fn=bond_energy,
        hinge_geometry=(geometry_fn(params)
                        if (geometry_fn is not None and bond_energy is not None) else None))

    paths = extract_hinge_paths(results['solution'], results['valid_state'],
                                build_hinge_geometry(config, static_features, state, params),
                                resolve_length_scale(config), results['reference_bond_vectors'])

    displacement = np.asarray(results['solution'].fields[-1])
    out = args.out or os.path.join(os.path.dirname(args.params or '.') or '.', "compression_map.png")
    label = args.label or ("trained" if args.params else "untrained")
    written = build_compression_figure(tessellation, results['valid_state'], displacement,
                                       paths.eta[..., 0], out, label=label)

    worst = paths.eta[..., 0].min(axis=0)
    print(f"  {paths.n_hinges} hinges x {paths.n_steps} steps   [{label}]")
    print(f"  eta_a  min {paths.eta[..., 0].min():+.4f}  max {paths.eta[..., 0].max():+.4f}")
    print(f"  hinges ever compressive: {int((worst < -1e-6).sum())} / {worst.size}")
    print(f"  compressive path samples: {100.0 * float((paths.eta[..., 0] < -1e-6).mean()):.1f}%")
    print(f"  wrote {written}")


if __name__ == "__main__":
    main()
