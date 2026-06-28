"""The geometry resolver (hinge_geometry) — no gmsh, no oracle."""
import pathlib
import numpy as np
import pytest
import yaml
from nff.sofa import oracle_payload as op
from nff.sofa.hinge_geometry import compute_hinge_geometry

REPO = pathlib.Path(__file__).resolve().parents[1]
CFG = REPO / "data/configs/sofa/hinge_opt_2face.yaml"


def _cs():
    return op.build_physical_cs(yaml.safe_load(open(CFG)))


def test_geometry_returns_only_geometry_keys():
    geo = compute_hinge_geometry(_cs(), gap=0.003)
    assert set(geo) == {"face_verts", "hinge_data", "clamped_faces", "loaded_faces"}


def test_face_verts_and_hinge_anchors_are_finite():
    geo = compute_hinge_geometry(_cs(), gap=0.003)
    assert geo["face_verts"].shape[1:] == (4, 2)
    hd = geo["hinge_data"][0]
    for k in ("p0_top", "p0_bot", "p1_top", "p1_bot", "bc_up", "bc_lo"):
        assert k in hd and np.all(np.isfinite(hd[k]))


def test_nonpositive_gap_is_rejected():
    with pytest.raises(ValueError):
        compute_hinge_geometry(_cs(), gap=0.0)
