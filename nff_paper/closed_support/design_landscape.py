"""Local structure of the design objective at a converged pattern.

The ensemble measures where independent starts LAND. This measures why: the curvature of
the objective in design space says how many directions the target actually constrains, and
whether the directions the ensemble spreads along are the ones it leaves flat.

The chart carries no continuous gauge -- the flat sheet is a fixed axis-aligned rectangle,
so no translation, rotation or scale freedom sits in the parameters -- and the outline is
compared after re-centring. A near-null direction of the shape Hessian is therefore a real
degeneracy of the inverse problem, not a coordinate artefact.
"""

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float


def flatten(params) -> tuple:
    """Design parameters as one vector, with the map back."""
    return jax.flatten_util.ravel_pytree(params)


def hessian_spectrum(scalar_fn, params) -> dict:
    """Eigen-decomposition of ``scalar_fn`` at ``params``, in the flattened design space.

    The design vector is ~100 scalars, so the dense Hessian is exact and cheap; nothing here
    needs a Lanczos approximation.

    Returns:
        ``evals`` ascending, ``evecs`` with matching columns, and the gradient norm -- a
        spectrum is only meaningful at a stationary point.
    """
    vec, unravel = flatten(params)

    def f(v):
        return scalar_fn(unravel(v))

    H = np.asarray(jax.hessian(f)(vec))
    H = 0.5 * (H + H.T)                       # symmetrise the last ulp of asymmetry
    evals, evecs = np.linalg.eigh(H)
    grad = np.asarray(jax.grad(f)(vec))
    return {"evals": evals, "evecs": evecs, "grad_norm": float(np.linalg.norm(grad)),
            "unravel": unravel, "vec": np.asarray(vec)}


def flat_subspace(spectrum: dict, rel_tol: float = 1e-6) -> Float[np.ndarray, "n k"]:
    """Eigenvectors whose curvature is negligible against the stiffest direction.

    ``rel_tol`` is a ratio, not an absolute: the objective is dimensionless and its overall
    scale carries no meaning, only the spread of the spectrum does.
    """
    evals = spectrum["evals"]
    cut = rel_tol * float(np.max(np.abs(evals)))
    return spectrum["evecs"][:, np.abs(evals) < cut]


def effective_dimension(evals) -> float:
    """Participation ratio of a spectrum: how many directions genuinely carry it."""
    w = np.abs(np.asarray(evals, dtype=float))
    if w.sum() <= 0:
        return 0.0
    p = w / w.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-300))))


def principal_angles(U: Float[np.ndarray, "n a"], V: Float[np.ndarray, "n b"]):
    """Angles in degrees between two subspaces, ascending.

    Zero degrees means a direction shared exactly; ninety means orthogonal. This is what
    turns "the spread looks like the flat directions" into a measurement.
    """
    qu, _ = np.linalg.qr(U)
    qv, _ = np.linalg.qr(V)
    s = np.linalg.svd(qu.T @ qv, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


def coords_jacobian(forward_fn, params, centre) -> Float[np.ndarray, "2n p"]:
    """d(flat cut vertices) / d(design parameters), flattened both ways.

    Parameter-space directions are not comparable across analyses -- they are chart
    coordinates. Pushing them through this Jacobian expresses every direction as a physical
    displacement field on the sheet, in mm, which is the common ground the ensemble spread
    also lives in.
    """
    vec, unravel = flatten(params)

    def coords_of(v):
        return (statics_coords(forward_fn, unravel(v)) - jnp.asarray(centre)).reshape(-1)

    return np.asarray(jax.jacfwd(coords_of)(vec))


def statics_coords(forward_fn, params):
    """Flat cut vertices for a design, as a JAX array."""
    return forward_fn(params)["coords"]


def push_to_geometry(directions: Float[np.ndarray, "p k"],
                     jacobian: Float[np.ndarray, "2n p"], return_mask: bool = False):
    """Parameter directions as unit displacement fields on the sheet.

    Directions the geometry does not move at all are dropped, so a caller that indexes the
    result against the eigenvalues MUST take ``return_mask`` and filter them the same way.
    """
    G = jacobian @ directions
    norms = np.linalg.norm(G, axis=0)
    keep = norms > 1e-12
    out = G[:, keep] / norms[keep]
    return (out, keep) if return_mask else out


def walk_mode(params, direction: Float[np.ndarray, "p"], amplitudes, unravel):
    """Designs obtained by moving along one direction in design space."""
    vec, _ = flatten(params)
    vec = np.asarray(vec)
    return [unravel(jnp.asarray(vec + a * direction)) for a in amplitudes]
