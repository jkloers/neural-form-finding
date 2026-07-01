"""Tests for the gmsh RVE mesher (nff.rve.mesh). Needs the gmsh Python API.

Run: JAX_PLATFORMS=cpu conda run -n kgnn_mac python -m pytest tests/test_rve_mesh.py -q
"""
import os

import pytest

from nff.rve.geometry import RVEParams

gmsh = pytest.importorskip("gmsh")
from nff.rve.mesh import build_rve_mesh                          # noqa: E402


@pytest.fixture(scope="module")
def stats(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("rve") / "rve.msh")
    p = RVEParams(w_lig=8.0, w_c=0.6, alpha_deg=90.0, rho=0.8, thickness=1.0, r_win=20.0)
    return build_rve_mesh(p, path, n_through=3)


def test_mesh_nonempty_and_written(stats):
    assert stats["n_nodes"] > 100
    assert stats["n_cells"] > 100
    assert os.path.exists(stats["path"]) and os.path.getsize(stats["path"]) > 0


def test_through_thickness_layers(stats):
    assert stats["n_z_levels"] == stats["n_through"] + 1          # structured layers


def test_all_boundary_groups_present(stats):
    for g in ("rigid_A", "rigid_B", "free"):
        assert stats["groups"][g] > 0, g
    # the two rigid handles are the tile far edges (few surfaces); free is the rest
    assert stats["groups"]["rigid_A"] >= 1 and stats["groups"]["rigid_B"] >= 1
    assert stats["groups"]["free"] > stats["groups"]["rigid_A"]


@pytest.mark.parametrize("n_through", [2, 4])
def test_layer_count_follows_request(tmp_path, n_through):
    path = str(tmp_path / f"rve_{n_through}.msh")
    st = build_rve_mesh(RVEParams(), path, n_through=n_through)
    assert st["n_z_levels"] == n_through + 1
