"""The payload builder (oracle_payload) — the client/oracle contract.

Imports JAX (via build_physical_cs) but needs no running oracle.
"""
import pathlib
import yaml
from nff.sofa import oracle_payload as op

REPO = pathlib.Path(__file__).resolve().parents[1]
CFG = REPO / "data/configs/sofa/hinge_opt_2face.yaml"


def test_param_names_are_the_nine_knobs():
    assert len(op.PARAM_NAMES) == 9
    assert op.POS_NAMES + op.FREE_NAMES == op.PARAM_NAMES


def test_build_payload_has_topology_and_all_knobs():
    cfg = yaml.safe_load(open(CFG))
    cs = op.build_physical_cs(cfg)
    phys = {n: 0.003 for n in op.PARAM_NAMES}
    payload = op.build_payload(cs, phys, cfg, [0], [1])
    for key in ("face_centroids", "centroid_node_vectors",
                "hinge_node_pairs", "hinge_adj_info"):
        assert key in payload
    for knob in op.PARAM_NAMES:
        assert knob in payload
