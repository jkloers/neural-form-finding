"""Dataset splits shared by training and diagnostics. NumPy only -- deliberately no JAX.

``split_by_job`` used to live in ``nff/scripts/train_hinge_surrogate.py``, whose module body imports
``jax`` and ``optax``. Any diagnostic that only wanted to reproduce the trainer's split therefore had
to drag the whole training stack in, and a script that reads a raw ``.npz`` could not run at all in
an environment without them. It is six lines of numpy; it belongs where anything can import it.

The trainer re-exports it, so ``from nff.scripts.train_hinge_surrogate import split_by_job``
(``nff/scripts/figures/plot_surrogate_parity.py``) keeps working unchanged.
"""

import numpy as np


def split_by_job(data, val_frac, seed, test_frac=0.0):
    """Group split by job = unseen GEOMETRY. Returns (train, val), or (train, val, test).

    ``test_frac > 0`` carves a third group that is scored exactly once at the end. Model selection
    happens on ``val``, so reporting on ``val`` too is optimistic by the selection bias over every
    evaluated epoch. The default 2-tuple keeps existing callers (plot_surrogate_parity) unchanged.

    Grouping is by ``job_id`` rather than by row because every row in a job shares one geometry and
    one deployment path -- they are nowhere near independent, so a row-wise split would leak the
    answer across the boundary and report a generalisation number that is not one.
    """
    jobs = np.unique(data["job_id"])
    rng = np.random.default_rng(seed); rng.shuffle(jobs)
    n_val = int(val_frac * len(jobs))
    val = np.isin(data["job_id"], jobs[:n_val].tolist())
    if test_frac <= 0.0:
        return ~val, val
    n_test = int(test_frac * len(jobs))
    test = np.isin(data["job_id"], jobs[n_val:n_val + n_test].tolist())
    return ~(val | test), val, test
