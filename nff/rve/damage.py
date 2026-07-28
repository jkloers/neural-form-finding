"""The hinge damage measure: normalized plastic dissipation in the ligament.

There is exactly ONE damage definition for an elasto-plastic hinge material:

    Delta = <PEEQ>_lig / eps_f          volume-weighted mean over the ligament

``eps_f`` is the material's measured fracture strain (PET: 1.784, ``ln(A0/A_f)``, run-to-break
2026-07-27). Because PET's flow stress is nearly flat over the cold-draw plateau,
``sigma_y * <PEEQ>_lig * V_lig`` is the plastic work dissipated in the ligament, so ``Delta`` is
that dissipation divided by the work needed to consume the material's full ductility.

Why this measure and not a peak:

* **It measures irreversibility**, which is what the design loss is for. PEEQ is monotone
  non-decreasing along ANY path, including the shear reversals seen in 83% of deployed hinges, so
  it never un-damages when the state returns -- the property a state-function surrogate needs.
* **It is intensive.** Bending thickness ``t`` through ``theta`` over ligament width ``w`` gives
  ``PEEQ ~ theta*t/(4w)`` over a volume ``w*t*L``, so the TOTAL plastic work
  ``~ sigma_y*theta*t^2*L/4`` is independent of ``w`` and useless as an objective, while the volume
  AVERAGE ``~ theta*t/(4w)`` rises as the hinge narrows. Narrow hinges folded hard score as damaged.
* **It is smooth and mesh-convergent.** A volume integral, not an order statistic: no rank-swap
  discontinuities, no single-element gradient concentration, and it converges under refinement
  where a p99 keeps drifting.

Fracture is NOT a second measure. It is a calibrated VALUE of ``Delta``, measured once per campaign
as the ``Delta`` at which the hottest ligament point first reaches ``eps_f`` (see
:func:`peak_peeq`). That number is reported and plotted; it enters no loss term.

Stress triaxiality is deliberately absent from the law. The Johnson-Cook / Rice-Tracey locus models
void nucleation and growth in ductile metals; PET crazes, cold-draws and fibrillates instead, and no
measured ``k`` exists for it. :func:`mean_triaxiality` records ``<eta>`` as a dataset column so the
assumption is measured every campaign rather than assumed.
"""

import numpy as np

# The ligament = the region the mesher actually resolves (the gmsh refinement ball in
# ``ccx_solver._build_mesh``): an in-plane disc centred a half-width below the cut tip, taken
# through the full thickness. Reusing the mesher's own numbers means the averaging domain and the
# resolved domain are the same set by construction -- there is no second geometric constant.
LIG_CENTER_FRAC = -0.5      # y-centre = LIG_CENTER_FRAC * w_lig
LIG_RADIUS_FRAC = 0.75      # in-plane radius = LIG_RADIUS_FRAC * w_lig


def stress_triaxiality(S: np.ndarray) -> np.ndarray:
    """Stress triaxiality ``eta = sigma_m / sigma_vm`` per node.

    Args:
        S: (N, 6) Cauchy stress ``[sxx, syy, szz, sxy, syz, szx]`` (CalculiX STRESS order).

    Returns:
        (N,) triaxiality; +ve tension, -ve compression, ~0 pure shear.
    """
    sxx, syy, szz, sxy, syz, szx = np.asarray(S, float).T
    sigma_m = (sxx + syy + szz) / 3.0
    sigma_vm = np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                       + 3.0 * (sxy ** 2 + syz ** 2 + szx ** 2))
    return sigma_m / np.maximum(sigma_vm, 1e-9)


def max_principal_strain(tostrain: np.ndarray) -> np.ndarray:
    """Per-node maximum principal (tensile) strain from a TOSTRAIN block.

    A strain invariant, not a damage measure -- used for the material-agnostic ``strain_max``
    diagnostic column and by the paper tear criterion.

    Args:
        tostrain: (N, 6) total strain ``[exx, eyy, ezz, exy, eyz, ezx]`` (CalculiX order).

    Returns:
        (N,) largest eigenvalue of the strain tensor per node.
    """
    e = np.asarray(tostrain, float)
    out = np.zeros(len(e))
    for i, (xx, yy, zz, xy, yz, zx) in enumerate(e):
        T = np.array([[xx, xy, zx], [xy, yy, yz], [zx, yz, zz]])
        out[i] = np.linalg.eigvalsh(T)[-1]
    return out


def element_volumes(xyz: np.ndarray, conn) -> np.ndarray:
    """Reference volume of each C3D15 wedge, from its six corner nodes.

    The wedge is split into three tetrahedra ``(0,1,2,3), (1,2,3,4), (2,3,4,5)``. Mid-side nodes are
    ignored, so a curved element's volume is approximated by its straight-edged hull -- adequate for
    a quadrature weight, and consistent under refinement.

    Args:
        xyz:  (n_nodes, 3) reference node coordinates.
        conn: (n_elems, 15) 1-based CalculiX C3D15 connectivity.

    Returns:
        (n_elems,) positive volumes.
    """
    c = np.asarray(conn, int)[:, :6] - 1                       # 1-based -> 0-based corners
    p = np.asarray(xyz, float)[c]                              # (n_elems, 6, 3)
    vol = np.zeros(len(p))
    for a, b, cc, d in ((0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5)):
        vol += np.abs(np.einsum("ij,ij->i",
                                np.cross(p[:, b] - p[:, a], p[:, cc] - p[:, a]),
                                p[:, d] - p[:, a])) / 6.0
    return vol


def ligament_elements(xyz: np.ndarray, conn, w_lig: float) -> np.ndarray:
    """Boolean mask of the elements whose reference centroid lies in the ligament disc.

    In-plane distance only: the disc is taken through the FULL thickness, so the average spans the
    bending gradient (both yielded surface fibres and the elastic mid-plane) -- which is what makes
    it a dissipation per unit volume.

    Args:
        xyz:   (n_nodes, 3) reference node coordinates.
        conn:  (n_elems, 15) 1-based connectivity.
        w_lig: ligament width [mm].

    Returns:
        (n_elems,) bool mask.
    """
    c = np.asarray(conn, int)[:, :6] - 1
    cen = np.asarray(xyz, float)[c].mean(axis=1)               # corner centroid per element
    dy = cen[:, 1] - LIG_CENTER_FRAC * w_lig
    return (cen[:, 0] ** 2 + dy ** 2) <= (LIG_RADIUS_FRAC * w_lig) ** 2


def _nodal_peeq(frame: dict):
    """Per-node equivalent plastic strain from a parsed frame; None when absent."""
    for key in ("PEEQ", "PE"):
        arr = frame.get(key)
        if arr is not None and np.size(arr):
            arr = np.abs(np.asarray(arr, float))
            return arr[:, 0] if arr.ndim > 1 else arr
    return None


def _element_mean(nodal: np.ndarray, conn) -> np.ndarray:
    """Average a per-node field over each element's 15 nodes."""
    return np.asarray(nodal, float)[np.asarray(conn, int) - 1].mean(axis=1)


def _ligament_average(nodal: np.ndarray, xyz: np.ndarray, conn, w_lig: float) -> float:
    """Volume-weighted mean of a per-node field over the ligament elements."""
    mask = ligament_elements(xyz, conn, w_lig)
    if not mask.any():
        return float("nan")
    vol = element_volumes(xyz, conn)[mask]
    total = vol.sum()
    if not total > 0:
        return float("nan")
    return float(np.dot(vol, _element_mean(nodal, conn)[mask]) / total)


def plastic_damage(frame: dict, xyz: np.ndarray, conn, w_lig: float, eps_f: float) -> float:
    """Normalized plastic dissipation ``Delta = <PEEQ>_lig / eps_f`` for one parsed frame.

    The single damage measure for an elasto-plastic hinge material. Returns NaN when the frame
    carries no plastic-strain field (an elastic-only solve, or a material that never yields).

    Args:
        frame: parsed ``.frd`` frame; needs ``PEEQ`` (or ``PE``).
        xyz:   (n_nodes, 3) reference node coordinates.
        conn:  (n_elems, 15) 1-based C3D15 connectivity.
        w_lig: ligament width [mm], setting the averaging disc.
        eps_f: the material's fracture strain -- the only damage constant.

    Returns:
        ``Delta >= 0``; 0 = fully elastic. See the module docstring for why 1 is not the tear line.
    """
    peeq = _nodal_peeq(frame)
    if peeq is None or len(peeq) != len(xyz):
        return float("nan")
    return _ligament_average(peeq, xyz, conn, w_lig) / eps_f


def peak_peeq(frame: dict, xyz: np.ndarray, conn, w_lig: float) -> float:
    """Highest element-mean PEEQ inside the ligament -- the tear indicator.

    Calibrates ``Delta_tear`` (the value of :func:`plastic_damage` at the frame where this first
    reaches ``eps_f``) and stops the solve at fracture. Element-mean rather than raw nodal max, so a
    single extrapolated corner singularity cannot trigger it.
    """
    peeq = _nodal_peeq(frame)
    if peeq is None or len(peeq) != len(xyz):
        return float("nan")
    mask = ligament_elements(xyz, conn, w_lig)
    return float(_element_mean(peeq, conn)[mask].max()) if mask.any() else float("nan")


def mean_triaxiality(frame: dict, xyz: np.ndarray, conn, w_lig: float) -> float:
    """Plastic-work-weighted mean stress triaxiality over the ligament -- diagnostic only.

    Recorded so that dropping the triaxiality-dependent fracture locus stays an AUDITED choice: if a
    campaign reports ``<eta>`` far from uniaxial tension (1/3), the constant-``eps_f`` assumption
    deserves re-examination.

    Weighted by ``V * PEEQ``, not by volume alone. A fold is a bending field, so its outer fibre is
    in tension and its inner fibre in compression; a plain volume average of ``eta`` cancels to ~0
    regardless of the stress state and would audit nothing. Weighting by plastic strain reports the
    triaxiality WHERE THE DAMAGE ACTUALLY ACCUMULATES, which is the state ``eps_f`` has to cover.
    Falls back to the volume average when the frame carries no plastic strain.
    """
    S = frame.get("STRESS")
    if S is None or not np.size(S):
        return float("nan")
    S = np.asarray(S, float)
    if S.ndim != 2 or S.shape[1] < 6 or S.shape[0] != len(xyz):
        return float("nan")
    eta = stress_triaxiality(S[:, :6])

    peeq = _nodal_peeq(frame)
    if peeq is None or len(peeq) != len(xyz):
        return _ligament_average(eta, xyz, conn, w_lig)

    mask = ligament_elements(xyz, conn, w_lig)
    if not mask.any():
        return float("nan")
    w = element_volumes(xyz, conn)[mask] * _element_mean(peeq, conn)[mask]
    total = w.sum()
    if not total > 0:                                   # nothing has yielded yet
        return _ligament_average(eta, xyz, conn, w_lig)
    return float(np.dot(w, _element_mean(eta, conn)[mask]) / total)
