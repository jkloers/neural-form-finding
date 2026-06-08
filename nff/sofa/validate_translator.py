"""
nff/sofa/validate_translator.py
================================
Smoke-test: verify that the CS-derived mesh builder matches the reference
geometry for the deployed 1×1 unit_RDQK_0 cell.

The reference is build_unified_mesh(face_size, arm_width, fold_length) from
sofa/geometry.py.  The CS-derived mesh is built by
nff/sofa/mesh_builder.build_mesh_from_centroidal_state using a deployed
CentroidalState (with per-face ARM_WIDTH shifts applied), so node positions
and BC masks should match identically.

Run from repo root:
    conda run -n kgnn_mac python nff/sofa/validate_translator.py
"""

import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import sys
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)

# sofa/geometry.py is a SOFA-env module; import it by path since it doesn't
# belong to any conda package (it runs under Homebrew Python normally).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sofa'))
from geometry import build_unified_mesh

from nff.sofa.geometry_translator import translate_rdqk_unit_cell

FACE_SIZE       = 0.100   # m — 100 mm (must match ARM_WIDTH scale)
ARM_WIDTH       = 0.010   # m — 10 mm
FOLD_LENGTH     = 0.003   # m — 3 mm
SHEET_THICKNESS = 0.001   # m — 1 mm


def main():
    # Reference mesh: build_unified_mesh uses the deployed geometry directly.
    ref_nodes, ref_hexes, ref_masks = build_unified_mesh(
        FACE_SIZE, ARM_WIDTH, FOLD_LENGTH, SHEET_THICKNESS)

    # CS-derived mesh: translate_rdqk_unit_cell builds a deployed CentroidalState
    # (per-face ARM_WIDTH shifts) then calls build_mesh_from_centroidal_state.
    tr_nodes, tr_hexes, tr_masks = translate_rdqk_unit_cell(
        face_size       = FACE_SIZE,
        arm_width       = ARM_WIDTH,
        fold_length     = FOLD_LENGTH,
        sheet_thickness = SHEET_THICKNESS,
        clamp_face      = 0,
        load_face       = 1,
    )

    print(f"Reference   : {len(ref_nodes):5d} nodes, {len(ref_hexes):5d} hexes")
    print(f"Translator  : {len(tr_nodes):5d} nodes, {len(tr_hexes):5d} hexes")

    assert len(ref_nodes) == len(tr_nodes), (
        f"Node count mismatch: {len(ref_nodes)} vs {len(tr_nodes)}")
    assert len(ref_hexes) == len(tr_hexes), (
        f"Hex count mismatch: {len(ref_hexes)} vs {len(tr_hexes)}")

    # Node positions should match after sorting (mesh builder may order differently).
    max_node_err = np.max(np.abs(np.sort(ref_nodes, axis=0)
                                 - np.sort(tr_nodes, axis=0)))
    print(f"Max node pos error  : {max_node_err:.2e} m")
    assert max_node_err < 1e-10, f"Node positions differ: {max_node_err}"

    # Face masks ('f0'..'f3' in ref, both 'fi' and 'face_i' in translator).
    for i in range(4):
        ref_m = ref_masks[f'f{i}']
        tr_m  = tr_masks[f'face_{i}']
        match = np.array_equal(ref_m, tr_m)
        print(f"  face_{i} mask match : {match}  "
              f"(ref={ref_m.sum()}, tr={tr_m.sum()} nodes)")
        assert match, f"face_{i} mask mismatch (ref={ref_m.sum()}, tr={tr_m.sum()})"

    # 'fi' alias must equal 'face_i'.
    for i in range(4):
        assert np.array_equal(tr_masks[f'f{i}'], tr_masks[f'face_{i}']), \
            f"'f{i}' alias differs from 'face_{i}'"
    print("  fi alias checks    : OK")

    # Clamped mask: face_0 should be clamped.
    n_clamped = tr_masks['clamped'].sum()
    n_f0      = tr_masks['face_0'].sum()
    assert n_clamped == n_f0, (
        f"Clamped mask ({n_clamped}) should equal face_0 mask ({n_f0})")
    print(f"Clamped mask OK     : {n_clamped} nodes")

    # Loaded mask: face_1 should be loaded.
    n_loaded = tr_masks['loaded'].sum()
    n_f1     = tr_masks['face_1'].sum()
    assert n_loaded == n_f1, (
        f"Loaded mask ({n_loaded}) should equal face_1 mask ({n_f1})")
    print(f"Loaded mask OK      : {n_loaded} nodes")

    print("\nAll checks passed.")


if __name__ == '__main__':
    main()
