"""The chart ablation's free-vertex arm still runs through the live pipeline.

The validity-ladder and conditioning results compare the closed chart (arm A) against free panel
vertices repaired by Stage-1 alternating projections (arm D). Arm D is the only remaining user of
`map_type: direct_vertices` and `validity_method: alternating_projection`, so this guards that path
after the open-pipeline code was archived.
"""
import os

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

CONFIG = "data/configs/closed/sheet_5x5_circle.yaml"      # the validity ladder's problem, 5x5


@pytest.mark.skipif(not os.path.exists(CONFIG), reason="closed config not present")
def test_free_vertex_arm_projects_and_deploys():
    from nff.config.experiment import load_and_parse_config
    from nff.analysis.chart_ablation import (
        arm_validity_cfg, deploy_validity, init_direct_from_closed, project_if_needed)
    from nff.closed.setup import build_closed_initial_state, init_closed_les_params
    from nff.stages.constraints import hinge_connectivity
    from nff.stages.mapping import apply_direct_mapping

    cfg = load_and_parse_config(CONFIG)

    # Arm A: the closed chart deploys a valid sheet.
    state_a, _ = build_closed_initial_state(cfg)
    params_a, sf = init_closed_les_params(cfg)
    audit_a = deploy_validity(cfg, params_a, state_a, "closed_les", static_features=sf)
    assert audit_a["finite"] and audit_a["valid"]

    # Arm D: open the hinges by perturbing free vertices, then let Stage 1 close them.
    params_d, state_d, _ = init_direct_from_closed(cfg)
    rng = np.random.default_rng(0)
    cnv = np.asarray(params_d["centroid_node_vectors"])
    params_d = {**params_d, "centroid_node_vectors":
                cnv + 1e-2 * rng.standard_normal(cnv.shape) * (np.abs(cnv) > 0)}
    cfg_d = arm_validity_cfg(cfg, "D")
    assert cfg_d.validity.validity_method == "alternating_projection"

    mapped = apply_direct_mapping(state_d, params_d)
    projected = project_if_needed(mapped, cfg_d)

    def gap(s):
        return float(hinge_connectivity(s.face_centroids, s.centroid_node_vectors,
                                        s.hinge_node_pairs))

    assert gap(mapped) > 1e-6
    assert gap(projected) < 1e-3 * gap(mapped)

    audit_d = deploy_validity(cfg_d, params_d, state_d, "direct_vertices")
    assert audit_d["finite"]


def test_archived_map_types_raise_a_pointer_to_legacy():
    from nff.config.experiment import load_and_parse_config
    from nff.closed.setup import build_closed_initial_state, init_closed_les_params
    from nff.stages.pipeline import forward_pipeline

    if not os.path.exists(CONFIG):
        pytest.skip("closed config not present")
    cfg = load_and_parse_config(CONFIG)
    state, _ = build_closed_initial_state(cfg)
    params, _ = init_closed_les_params(cfg)
    for map_type in ("gnn_mpnn", "conformal_polynomial", "direct_transform"):
        with pytest.raises(ValueError, match="legacy"):
            forward_pipeline(state, cfg.target, cfg.validity, cfg.physics,
                             map_type=map_type, map_params=params)
