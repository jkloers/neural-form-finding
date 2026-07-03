"""Verify the Phase-3 point-ROM <-> physical-hinge frame bridge (Gap 1).

Checks, with no FEM:
  1. rigid-body motion (global translate + rotate) => ligament_strains ~ 0  (=> W = 0),
  2. relative OPENING along the axial (secondary-cut) axis => axial > 0, shear ~ 0,
  3. relative SHEAR perpendicular to it     => shear != 0, axial ~ 0,
  4. build_reference_bond_vectors, given a degenerate (closed) hinge + an axial direction,
     returns a nominal vector ALONG that axis oriented face_i->face_k (opening-positive),
     instead of the arbitrary unit-x fallback.
"""
import types
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from nff.stages.physics.energy import ligament_strains
from nff.stages.geometry import build_reference_bond_vectors

ok = True
def check(name, cond):
    global ok; ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


# ── 1. rigid-body invariance => zero strain ────────────────────────────────────────
l0 = 1e-2
ref = jnp.array([[0.8, 0.6]]) * l0                       # arbitrary cut-oriented reference
A = jnp.array([[1.0, 2.0]]); B = A + ref
phi = 0.4                                                 # global rigid rotation [rad]
R = jnp.array([[jnp.cos(phi), -jnp.sin(phi)], [jnp.sin(phi), jnp.cos(phi)]])
T = jnp.array([3.0, -1.0])
dispA = (A @ R.T - A) + T                                 # node displacement under the rigid motion
dispB = (B @ R.T - B) + T
DOFs1 = jnp.concatenate([dispA, jnp.array([[phi]])], axis=1)
DOFs2 = jnp.concatenate([dispB, jnp.array([[phi]])], axis=1)
ax, sh, dr = (float(v[0]) for v in ligament_strains(DOFs1, DOFs2, reference_vector=ref[0]))
print("1. rigid-body motion => strains")
check("axial ~ 0", abs(ax) < 1e-6)   # ~5e-9, the eps=1e-12 regularization floor
check("shear ~ 0", abs(sh) < 1e-9)
check("dRot ~ 0",  abs(dr) < 1e-9)

# ── 2/3. opening vs shear along the frame ──────────────────────────────────────────
# NOTE for step 3.1: ligament_strains decomposes by length (axial) + angle (shear), so a finite
# perpendicular offset gives a 2nd-order axial ~1/2 (eps/l0)^2. The RVE imposed (a,s) as LINEAR
# translations, so the surrogate adapter must PROJECT dU onto (uhat, vhat), not reuse these.
uhat = ref[0] / jnp.linalg.norm(ref[0]); vhat = jnp.array([-uhat[1], uhat[0]])
eps = 0.02 * l0
z = jnp.zeros((1, 3))
ax_o, sh_o, _ = (float(v[0]) for v in ligament_strains(z, jnp.array([[*(eps * uhat), 0.0]]), reference_vector=ref[0]))
ax_s, sh_s, _ = (float(v[0]) for v in ligament_strains(z, jnp.array([[*(eps * vhat), 0.0]]), reference_vector=ref[0]))
print("2. opening along axial axis")
check("axial > 0", ax_o > 1e-6)
check("shear ~ 0", abs(sh_o) < 1e-6)
print("3. shear perpendicular to it")
check("shear != 0", abs(sh_s) > 1e-3)
check("axial ~ 0", abs(ax_s) < 1e-3)

# ── 4. build_reference_bond_vectors reorients a closed hinge along the given axis ───
# face_0 at origin, face_1 at (1,0); their shared vertex coincides at (0.5, 0) -> degenerate bond.
state = types.SimpleNamespace(
    face_centroids=jnp.array([[0.0, 0.0], [1.0, 0.0]]),
    centroid_node_vectors=jnp.array([[[0.5, 0.0]], [[-0.5, 0.0]]]),
    hinge_node_pairs=jnp.array([[[0, 0], [1, 0]]]),
)
axial_dirs = jnp.array([[-1.0, 0.0]])                     # points face_1->face_0 (against A->B=+x)
rv_old = build_reference_bond_vectors(state)              # backward-compat: unit-x fallback
rv_new = build_reference_bond_vectors(state, hinge_axial_dirs=axial_dirs, l0_nom=1e-3)
print("4. reference-bond reorientation (closed hinge)")
check("old fallback is +x (unchanged default)", bool(rv_old[0, 0] > 0 and abs(rv_old[0, 1]) < 1e-12))
check("new vector lies on the given axis (x)", abs(float(rv_new[0, 1])) < 1e-12)
check("new vector oriented opening-positive face_i->face_k (+x)", float(rv_new[0, 0]) > 0)
check("new vector has nominal length", abs(float(jnp.linalg.norm(rv_new[0])) - 1e-3) < 1e-9)

# ── 5. surrogate bond-energy adapter (3.1/3.3/3.4): projection + scale + W(0)=0 ────
from nff.models.hinge_surrogate import (init_hinge_surrogate, compute_norm_stats,
    apply_hinge_energy, build_hinge_bond_energy_fn)
rng = np.random.default_rng(0); nh = 4
al = jnp.asarray(rng.uniform(0.6, 2.4, nh)); wl = jnp.asarray(rng.uniform(1, 10, nh))
ang = rng.uniform(0, 2 * np.pi, nh); sec = jnp.stack([jnp.cos(ang), jnp.sin(ang)], -1)
stats = compute_norm_stats(rng.uniform(-1, 1, 512), rng.uniform(-1, 1, 512), rng.uniform(0, 0.6, 512),
                           rng.uniform(1, 10, 512), rng.uniform(0.6, 2.4, 512), rng.uniform(0, 2000, 512))
net = init_hinge_surrogate(jax.random.PRNGKey(1))
LS, ES = 3.0, 0.5
be = build_hinge_bond_energy_fn(net, stats, alpha=al, w_lig=wl, sec_dir=sec, length_scale=LS, energy_scale=ES)

print("5. surrogate bond-energy adapter")
D1 = jnp.asarray(rng.uniform(-1, 1, (nh, 3)))
Wrig = be((D1, D1))                                          # identical DOFs = rigid => u=0 => W=0
check("rigid (dU=0, dRot=0) => W = 0", float(jnp.abs(Wrig).max()) < 1e-12)
delta = 0.07
D2 = D1.at[:, :2].add(delta * sec).at[:, 2].set(D1[:, 2])   # pure opening along axial, no dRot
# expected: a = delta*LS (projection is corotated by mean_rot=D1[:,2]; opening is along sec in the
# ROTATED frame, so displace by delta*R(mean_rot)@sec to be purely axial):
c_, s_ = jnp.cos(D1[:, 2]), jnp.sin(D1[:, 2])
rot_sec = jnp.stack([c_ * sec[:, 0] - s_ * sec[:, 1], s_ * sec[:, 0] + c_ * sec[:, 1]], -1)
D2 = D1.at[:, :2].add(delta * rot_sec)
u_exp = jnp.stack([jnp.full(nh, delta * LS), jnp.zeros(nh), jnp.zeros(nh)], -1)
g = jnp.stack([wl, al], -1)
W_be = be((D1, D2)); W_ref = ES * apply_hinge_energy(net, u_exp, g, stats)
check("opening projects to a=delta*length_scale, s~0 (matches apply_hinge_energy)",
      float(jnp.abs(W_be - W_ref).max()) < 1e-9)

print("\n" + ("ALL CHECKS PASS" if ok else "SOME CHECKS FAILED"))
