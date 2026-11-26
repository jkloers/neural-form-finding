"""Shared scoring for the surrogate-validity figures: one reconstruction of the checkpoint's split.

Every held-out number in this repository is only as honest as the split it is measured on, and the
split is easy to get subtly wrong in three separate ways:

1. The trainer splits THREE ways (``val_frac`` and ``test_frac`` both 0.15). A two-way split scores
   the VALIDATION group, which model selection already saw 85 times, and silently hides the
   untouched test rows inside the "train" cloud.
2. A checkpoint trained with ``--route-family`` was split over that family ALONE. Splitting the full
   dataset partitions differently and leaks training rows into the held-out group.
3. The unconverged quarantine is applied AFTER the split, so job membership does not depend on the
   policy.

``load_scored`` performs all three in the trainer's order and asserts the resulting sizes against
the checkpoint's own sidecar, so a future drift in the split logic fails loudly instead of quietly
reporting a flattering number.

The rows of the route family the checkpoint was NOT trained on are never split: none of them were
ever seen, so all of them are legitimately held out. They are the closest thing to a deployment
error bar that costs no new finite-element work.
"""
import json
import os
import pickle

import jax
jax.config.update("jax_enable_x64", True)   # before load_hinge_surrogate, whose float64 upcast is
                                            # silently truncated to float32 otherwise
import jax.numpy as jnp
import numpy as np

from nff.models.hinge_surrogate import (apply_hinge_energy, apply_hinge_failure, apply_hinge_force,
                                        load_hinge_surrogate)
from nff.scripts.surrogate.score_deployment_ood import _whiten, load_eta
from nff.scripts.train_hinge_surrogate import check_force_sign, converged_mask, load_dataset
from nff.utils.routes import FREE_PATH, ORIGIN_RAY, route_family_mask
from nff.utils.splits import split_by_job

SURROGATE = "data/surrogates/hinge_surrogate_pet_v3"
DATASET = "data/fea/hinge_dataset_pet_merged"

# The reduced-order model of Eq. (rom), condensed at w_lig = 18 mm for PET t = 0.5 mm.
# Provenance: data/configs/closed/sheet_4x8ft_rom.yaml:136-160. k_theta is the BUCKLED SECANT at
# 20 degrees, not the elastic tangent, so this is the reference model at its most generous.
ROM_K_A, ROM_K_S, ROM_K_THETA = 238.1, 127.4, 487.0
ROM_BAND = (16.0, 20.0)     # w_lig window bracketing the width the springs were condensed at


def rel_rmse(pred, true):
    """Relative RMS error, the metric the trainer reports."""
    return float(np.sqrt(np.mean((pred - true) ** 2)) / (np.sqrt(np.mean(true ** 2)) + 1e-12))


def r2(pred, true):
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-30)


def rom_energy(u):
    """Stored energy of the linear-spring reference model [N mm].

    A quadratic in each kinematic argument, blind to w_lig by construction. Used ONLY as a control
    scored against the same oracle rows: it shows what a second-order truncation misses, and is
    never a target the surrogate should approach.
    """
    a, s, theta = u[:, 0], u[:, 1], u[:, 2]
    return 0.5 * ROM_K_A * a ** 2 + 0.5 * ROM_K_S * s ** 2 + 0.5 * ROM_K_THETA * theta ** 2


class Scored:
    """Predictions and masks for one checkpoint over one campaign."""

    def __init__(self, data, meta, stats, W_pred, F_pred, D_pred, masks):
        self.data, self.meta, self.stats = data, meta, stats
        self.W_pred, self.F_pred, self.D_pred = W_pred, F_pred, D_pred
        self.train, self.val, self.test, self.unseen_family = masks

    @property
    def u(self):
        return self.data["u"]

    @property
    def w_lig(self):
        return self.data["g"][:, 0]

    @property
    def alpha(self):
        return self.data["g"][:, 1]

    def band(self, lo=ROM_BAND[0], hi=ROM_BAND[1]):
        return (self.w_lig >= lo) & (self.w_lig < hi)


def load_scored(surrogate=SURROGATE, dataset=DATASET):
    """Reconstruct the checkpoint's own split and score every row. See the module docstring."""
    with open(surrogate + ".pkl", "rb") as f:
        meta = pickle.load(f).get("meta", {}) or {}
    params, stats, _ = load_hinge_surrogate(surrogate + ".pkl")

    data, _, _ = load_dataset(dataset, w_lig_max=meta.get("w_lig_max", 25.0))
    sign, _, _ = check_force_sign(data)
    data["F"] = data["F"] * sign

    family = meta.get("route_family", "all")
    in_family = route_family_mask(data["u"][:, 0], data["u"][:, 1], data["u"][:, 2],
                                  data["g"][:, 0], data["job_id"], family) \
        if family != "all" else np.ones(len(data["W"]), bool)

    # Split over the trained family ALONE, exactly as the trainer did.
    sub = {k: (v[in_family] if getattr(v, "shape", (0,))[:1] == in_family.shape else v)
           for k, v in data.items()}
    idx = np.where(in_family)[0]
    tr_s, va_s, te_s = split_by_job(sub, 0.15, meta.get("seed", 0), test_frac=0.15)

    n = len(data["W"])
    train, val, test = (np.zeros(n, bool) for _ in range(3))
    train[idx[tr_s]], val[idx[va_s]], test[idx[te_s]] = True, True, True

    # Quarantine AFTER the split, so job membership does not depend on the policy.
    policy = meta.get("exclude_unconverged", "none")
    if policy != "none":
        ok = converged_mask(data)
        val, test = val & ok, test & ok
        if policy == "both":
            train = train & ok
    unseen_family = ~in_family
    if policy != "none":
        unseen_family = unseen_family & converged_mask(data)

    _assert_split(surrogate, train, val, test)

    u, g = jnp.asarray(data["u"]), jnp.asarray(data["g"])
    W_pred = np.asarray(apply_hinge_energy(params, u, g, stats))
    F_pred = np.asarray(apply_hinge_force(params, u, g, stats))
    D_pred = np.asarray(apply_hinge_failure(params, u, g, stats))
    return Scored(data, meta, stats, W_pred, F_pred, D_pred, (train, val, test, unseen_family))


def _assert_split(surrogate, train, val, test):
    """Fail loudly if the reconstruction drifts from what the checkpoint recorded."""
    side = surrogate + ".json"
    if not os.path.exists(side):
        return
    with open(side) as f:
        rec = json.load(f)
    want = (rec.get("n_train"), rec.get("n_val"), rec.get("n_test"))
    if None in want:
        return
    got = (int(train.sum()), int(val.sum()), int(test.sum()))
    if got != tuple(want):
        raise SystemExit(f"split reconstruction mismatch: got {got}, checkpoint recorded {want}. "
                         "The held-out numbers would not be the ones the model was validated on.")


def recorded_metrics(surrogate=SURROGATE):
    """The test metrics the trainer itself wrote, or {} if the sidecar is absent.

    Scoring here must agree with them. Where a figure and the checkpoint disagree the figure is
    wrong by definition, so anything quoted in the paper is checked against this rather than
    recomputed on a convention of the figure's own choosing.
    """
    side = surrogate + ".json"
    if not os.path.exists(side):
        return {}
    with open(side) as f:
        return json.load(f).get("test_metrics", {}) or {}


def force_parity(sc, mask):
    """-> (true, pred, per-component R2, pooled R2, pooled relRMSE), each component RMS-normalized.

    The three components differ by an order of magnitude in units, so a raw pooled parity collapses
    the two forces onto the origin and reports the moment alone. Normalizing each by its own true
    RMS is what makes one dimensionless panel legitimate, and BOTH pooled statistics are computed
    after that normalization so they describe the same quantity.
    """
    Ft, Fp = sc.data["F"][mask], sc.F_pred[mask]
    rms = np.sqrt(np.mean(Ft ** 2, axis=0))
    Ft_n, Fp_n = Ft / rms, Fp / rms
    per = [r2(Fp_n[:, j], Ft_n[:, j]) for j in range(3)]
    return Ft_n, Fp_n, per, r2(Fp_n.ravel(), Ft_n.ravel()), rel_rmse(Fp_n.ravel(), Ft_n.ravel())


def deployment_states(prefix, stats):
    """-> whitened deployment hinge-states and their raw eta, from an extract_hinge_paths npz."""
    eta, _, _, geom = load_eta(prefix)
    w_lig, alpha = geom[:, 0], geom[:, 1]
    X = _whiten(eta[:, 0] * w_lig, eta[:, 1] * w_lig, eta[:, 2], w_lig, alpha, stats)
    return X, eta
