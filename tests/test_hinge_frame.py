"""Golden tests for the RVE-consistent hinge frame (compute_hinge_frame / _hinge_frame_from_points).

The convention is pinned to build_rve_domain (secondary=+x, material below, main at -alpha) and to the
deployment projection hinge_kinematics (axial=sec_dir, shear=perp_CCW(sec_dir)). These tests verify:
  1. synthetic round-trip: construct a hinge at KNOWN alpha in the RVE frame -> recover it exactly,
     for every endpoint ordering and every global rotation (rotation-equivariance);
  2. on a real random tessellation the new alpha is in (0, pi), lands material-consistently, and
     corrects the old descriptor's per-hinge supplement flips;
  3. alpha is differentiable in the design (finite jax.grad through solve_cut_vertices_jax).
"""
import numpy as np
import jax, jax.numpy as jnp
import pytest

from nff.topology.hinge_descriptor import (
    _hinge_frame_from_points, compute_hinge_frame, compute_hinge_descriptors,
    build_hinge_descriptor_structure)
from nff.topology.closed_builder import build_square_boundary_points, solve_cut_vertices, _assemble_panels
from nff.topology.closed_builder_jax import boundary_points_flat, solve_cut_vertices_jax


def _rot(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


@pytest.mark.parametrize("alpha_deg", [20, 45, 64, 90, 116, 150])
@pytest.mark.parametrize("sec_ord", [+1, -1])
@pytest.mark.parametrize("main_ord", [+1, -1])
@pytest.mark.parametrize("rot_deg", [0, 37, 155, -80])
def test_roundtrip(alpha_deg, sec_ord, main_ord, rot_deg):
    """Construct a hinge at known alpha in the RVE frame, rotated globally -> recover alpha + frame."""
    a = np.radians(alpha_deg); R = _rot(np.radians(rot_deg))
    pivot = np.zeros(2)
    sec = [np.array([-1.0, 0.0]), np.array([1.0, 0.0])][::sec_ord]     # through-cut, pivot interior
    main = [pivot, np.array([np.cos(-a), np.sin(-a)])][::main_ord]     # terminating, into y<0 (material)
    P = R @ pivot
    s0, s1 = R @ sec[0], R @ sec[1]
    m0, m1 = R @ main[0], R @ main[1]
    sec_dir, shear_dir, alpha = _hinge_frame_from_points(
        jnp.asarray(P), jnp.asarray(m0), jnp.asarray(m1), jnp.asarray(s0), jnp.asarray(s1))
    sec_dir, shear_dir, alpha = np.asarray(sec_dir), np.asarray(shear_dir), float(alpha)

    assert abs(np.degrees(alpha) - alpha_deg) < 1e-4                    # alpha recovered
    assert np.dot(sec_dir, R @ np.array([1.0, 0.0])) > 0.999           # axial = RVE +x
    assert np.dot(shear_dir, R @ np.array([0.0, 1.0])) > 0.999         # shear = RVE +y (void side)
    assert abs(np.dot(sec_dir, shear_dir)) < 1e-6                       # orthonormal frame


def _random_design(M=8, N=8, seed=3):
    hs = build_hinge_descriptor_structure(M, N, ref_r=0.45)
    ds = hs["deploy_struct"]
    bf = jnp.asarray(boundary_points_flat(ds, 1.0))
    r = jnp.asarray(np.random.default_rng(seed).uniform(0.28, 0.64, size=(ds["rows"], ds["cols"])))
    coords = solve_cut_vertices_jax(ds, bf, r)
    return hs, ds, bf, r, coords


def test_real_tessellation_material_consistent():
    """New alpha in (0,pi) and the material side (main ray) agrees with the two hinged faces."""
    hs, ds, bf, r, coords = _random_design()
    coords_np = np.asarray(coords)
    fr = compute_hinge_frame(hs, coords)
    alpha = np.degrees(np.asarray(fr["alpha"])); sec = np.asarray(fr["sec_dir"])
    interior = np.asarray(hs["is_interior"])
    assert np.all((alpha[interior] > 0) & (alpha[interior] < 180))

    # independent material check: the two hinged FACES must lie on the material (main-ray) side
    _, faces, corner_pid = _assemble_panels(ds["T"], solve_cut_vertices(
        ds["T"], build_square_boundary_points(ds["T"], spacing=1.0), 0.45))
    bad = 0
    for hid in np.where(interior)[0]:
        P = coords_np[hs["pivot_pid"][hid]]
        perp = np.array([-sec[hid, 1], sec[hid, 0]])                    # void side by construction
        f1, f2 = hs["face_pairs"][hid]
        cmid = 0.5 * (coords_np[corner_pid[f1]].mean(0) + coords_np[corner_pid[f2]].mean(0))
        if np.dot(cmid - P, perp) > 1e-6:                              # faces on the VOID side => wrong
            bad += 1
    assert bad == 0, f"{bad} hinges have their faces on the void side (material misidentified)"


def test_corrects_old_supplement():
    """The frame corrects the OLD edge-based alpha (which picked the wrong branch for most hinges)."""
    hs, ds, bf, r, coords = _random_design()
    c = np.asarray(coords)
    # the ORIGINAL buggy formula, inline: angle between the two _build_hinges CCW panel edges
    piv = c[hs["pivot_pid"]]
    e1 = c[hs["edge_pid"][:, 0]] - piv; e2 = c[hs["edge_pid"][:, 1]] - piv
    cos = np.sum(e1 * e2, 1) / (np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1) + 1e-12)
    old = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    new = np.degrees(np.asarray(compute_hinge_frame(hs, coords)["alpha"]))
    interior = np.asarray(hs["is_interior"])
    # hinge 25 was the audited case: old edge-based 64.3, RVE-correct 115.7
    assert abs(new[25] - 115.7) < 2.0 and abs(old[25] - 64.3) < 2.0
    # every hinge: new == old or new == 180-old (same cut line, correctly resolved branch)
    same = np.abs(new - old) < 1.0
    supp = np.abs(new - (180 - old)) < 1.0
    assert np.all(same[interior] | supp[interior])
    # the old formula was on the WRONG branch for the large majority of hinges
    assert np.sum((np.abs(new - old) > 1.0) & interior) >= 40


def test_alpha_differentiable_in_design():
    """d(alpha)/d(r) is finite and non-trivial (frame flows gradients to the design)."""
    hs, ds, bf, r, coords = _random_design()

    def mean_alpha(rr):
        c = solve_cut_vertices_jax(ds, bf, rr)
        a = compute_hinge_frame(hs, c)["alpha"]
        return jnp.mean(jnp.where(jnp.asarray(hs["is_interior"]), a, 0.0))

    g = jax.grad(mean_alpha)(r)
    assert np.all(np.isfinite(np.asarray(g)))
    assert np.linalg.norm(np.asarray(g)) > 1e-6


if __name__ == "__main__":
    hs, ds, bf, r, coords = _random_design()
    old = np.degrees(np.asarray(compute_hinge_descriptors(hs, coords)["alpha"]))
    new = np.degrees(np.asarray(compute_hinge_frame(hs, coords)["alpha"]))
    interior = np.asarray(hs["is_interior"])
    flipped = np.sum((np.abs(new - old) > 1.0) & interior)
    print(f"interior hinges: {int(interior.sum())} | old->new supplement-corrected: {flipped}")
    for hid in [25, 10, 40, 30, 12]:
        print(f"  hid {hid:3d}: old {old[hid]:6.1f}  new {new[hid]:6.1f}")
    print("run `pytest tests/test_hinge_frame.py -q` for the full golden suite")
