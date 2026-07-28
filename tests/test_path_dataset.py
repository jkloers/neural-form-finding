"""Harvested path dataset (``nff.closed.path_dataset``).

The dataset is the hand-off to the next surrogate campaign, so the properties worth pinning are the
ones that would silently corrupt that hand-off: the FULL polyline must survive a save/load round
trip (endpoints alone cannot express a path-dependent plastic history), the eta normaliser must be
recorded rather than implied, and the design index must group hinges by sheet so a held-out split
cannot leak siblings.
"""
import os

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)   # the pipeline runs in x64; keep this file self-sufficient
import pytest

from nff.closed.path_dataset import (PathExample, PathDataset, save_dataset, load_dataset,
                                     harvest_paths)


CFG = "data/configs/closed/sheet_4x8ft_rom_1tile.yaml"


def _example(seed, endpoints, n_steps=5, n_verts=8):
    endpoints = np.asarray(endpoints, dtype=float)
    lam = np.linspace(0.0, 1.0, n_steps + 1)
    eta = lam[:, None, None] * endpoints[None, :, :]
    h = len(endpoints)
    return PathExample(seed=seed, eta=eta, alpha=np.full(h, 1.5), w_lig=np.full(h, 5.0),
                       load_frac=lam, verts=np.zeros((n_verts, 2)) + seed,
                       flat=np.zeros((n_verts, 2)), z=np.zeros((3, 3)), bnd=np.zeros(4))


def _dataset(per_design):
    ds = PathDataset(meta={'w_lig_mm': 5.0, 'length_scale_mm_per_unit': 406.4,
                           'face_vertex_ids': [[0, 1, 2, 3]]})
    for i, ends in enumerate(per_design):
        ds.examples.append(_example(i, ends))
    return ds


# ── the hand-off to the oracle ────────────────────────────────────────────────────

def test_polylines_are_kept_whole_not_reduced_to_endpoints():
    """The oracle replays paths, so the path must survive as a path."""
    ds = _dataset([[[1.0, 0.2, 0.5], [0.4, -0.1, 0.3]]])
    p = ds.polylines()
    assert p.shape == (2, 6, 3)                      # (n_paths, n_steps+1, 3)
    assert np.allclose(p[:, 0, :], 0.0)              # every path starts at the origin
    assert np.allclose(p[0, -1], [1.0, 0.2, 0.5])    # ...and ends where it ended
    assert np.allclose(ds.endpoints()[0], p[0, -1])


def test_design_index_groups_hinges_by_sheet():
    """Held-out splits must be BY DESIGN: one sheet's hinges are not independent samples."""
    ds = _dataset([[[1.0, 0.0, 0.5]] * 3, [[2.0, 0.0, 0.6]] * 2])
    gi = ds.design_index()
    assert gi.tolist() == [0, 0, 0, 1, 1]
    assert len(gi) == len(ds.endpoints())


def test_roundtrip_preserves_the_paths_and_the_eta_normaliser(tmp_path):
    """A dataset whose eta scale is unknown cannot be handed to the oracle at all."""
    ds = _dataset([[[1.0, 0.2, 0.5], [0.4, -0.1, 0.3]], [[0.7, 0.3, 0.4], [1.1, 0.0, 0.6]]])
    save_dataset(ds, str(tmp_path))
    back = load_dataset(str(tmp_path))
    assert back.n_examples == 2
    assert np.allclose(back.polylines(), ds.polylines())
    assert np.allclose(back.alphas(), ds.alphas())
    assert back.meta['w_lig_mm'] == 5.0
    assert back.meta['length_scale_mm_per_unit'] == 406.4
    assert [e.seed for e in back.examples] == [0, 1]


def test_roundtrip_keeps_geometry_for_the_contact_sheet(tmp_path):
    ds = _dataset([[[1.0, 0.0, 0.5]], [[2.0, 0.0, 0.6]]])
    save_dataset(ds, str(tmp_path))
    back = load_dataset(str(tmp_path))
    assert back.examples[1].verts.shape == ds.examples[1].verts.shape
    assert np.allclose(back.examples[1].verts, 1.0)         # seed-1 example was filled with 1.0
    assert back.meta['face_vertex_ids'] == [[0, 1, 2, 3]]


def test_refuses_to_save_an_empty_dataset(tmp_path):
    with pytest.raises(ValueError, match="empty dataset"):
        save_dataset(PathDataset(), str(tmp_path))


# ── against the real pipeline ─────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cfg():
    if not os.path.exists(CFG):
        pytest.skip(f"{CFG} not present (data/ is gitignored)")
    from nff.config.experiment import load_and_parse_config
    return load_and_parse_config(CFG)


def test_harvest_records_the_single_tile_grip_it_actually_ran(cfg):
    """The BCs are provenance: a path dataset is only meaningful with the grip that produced it."""
    ds = harvest_paths(cfg, n_examples=2, noise=0.5, seed0=0, n_load_steps=3, verbose=False)
    assert ds.n_examples == 2, ds.failures
    assert ds.meta['clamped_faces'] == [3]
    assert [l['face'] for l in ds.meta['loads']] == [5]
    assert ds.meta['clamped_dofs'] == [0, 1, 2]      # a lone clamped tile must fix all three
    assert ds.meta['w_lig_mm'] > 0.0


def test_harvest_paths_start_at_the_origin_and_differ_between_designs(cfg):
    ds = harvest_paths(cfg, n_examples=2, noise=0.5, seed0=0, n_load_steps=3, verbose=False)
    p = ds.polylines()
    assert np.allclose(p[:, 0, :], 0.0, atol=1e-12)          # undeployed sheet is the origin
    assert np.all(np.isfinite(p))
    a, b = ds.examples[0].eta[-1], ds.examples[1].eta[-1]
    assert not np.allclose(a, b), "different random designs produced identical paths"
