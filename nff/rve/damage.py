"""Continuous ductile-damage measure for the hinge RVE.

Replaces the binary ``peeq_p99 >= eps_f`` fracture flag with a physically-grounded, continuous
damage ``D in [0, 1]`` (0 = undamaged, 1 = fracture), computed from the SAME CalculiX output
already recorded per frame (PEEQ + the stress tensor S) -- no re-solve, no new physics.

The physics the old measure ignored: ductile fracture strain depends on STRESS TRIAXIALITY
``eta = sigma_m / sigma_vm``. A hinge fold is shear/bending-dominated (low/moderate eta), where
steel tolerates far more plastic strain than in tension -- so a constant ``eps_f = 0.25`` was
needlessly conservative exactly where the mechanism operates. Because the RVE loading is a
proportional monotonic ramp, eta is ~constant along a ray, so the deformation-theory (memoryless)
approximation ``D = PEEQ / eps_f(eta)`` at the current state is single-valued -- matching how the
surrogate reads ``(a, s, theta)``.

    eta        = sigma_m / sigma_vm                       (+tension / -compression)
    eps_f(eta) = eps_f0 * exp(-k * (eta - 1/3))           (Johnson-Cook-like; = eps_f0 at uniaxial tension)
    D_elem     = PEEQ_elem / max(eps_f(eta_elem), floor)  (>=1 => that element has fractured)

The scalar margin is a robust high percentile of the per-element D (singularity-insensitive, like
the old p99), so a lone hot fiber does not condemn the whole ligament.
"""

import numpy as np


def stress_triaxiality(S: np.ndarray) -> np.ndarray:
    """Stress triaxiality ``eta = sigma_m / sigma_vm`` per element.

    Args:
        S: (N, 6) Cauchy stress ``[sxx, syy, szz, sxy, syz, szx]`` (CalculiX STRESS order).

    Returns:
        (N,) triaxiality; +ve tension, -ve compression, ~0 pure shear.
    """
    sxx, syy, szz, sxy, syz, szx = S.T
    sigma_m = (sxx + syy + szz) / 3.0
    sigma_vm = np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                       + 3.0 * (sxy ** 2 + syz ** 2 + szx ** 2))
    return sigma_m / np.maximum(sigma_vm, 1e-9)


def fracture_locus(eta: np.ndarray, eps_f0: float = 0.25, k: float = 1.5,
                   eta_floor: float = -1.0 / 3.0, eps_f_cap: float = 3.0) -> np.ndarray:
    """Triaxiality-dependent fracture strain, calibrated so ``eps_f(1/3) = eps_f0`` (uniaxial tension).

    ``k`` sets how fast fracture strain falls with tension / rises with shear-compression (a
    material property; ~1.5 is typical for mild steel). ``eta_floor`` is the compression cutoff
    (below ~ -1/3 ductile damage effectively stops); ``eps_f_cap`` bounds the shear/compression rise.
    """
    eta_c = np.maximum(np.asarray(eta, float), eta_floor)
    return np.minimum(eps_f0 * np.exp(-k * (eta_c - 1.0 / 3.0)), eps_f_cap)


def max_principal_strain(tostrain: np.ndarray) -> np.ndarray:
    """Per-element maximum principal (tensile) strain from a TOSTRAIN block.

    Args:
        tostrain: (N, 6) total strain ``[exx, eyy, ezz, exy, eyz, ezx]`` (CalculiX order).

    Returns:
        (N,) largest eigenvalue of the strain tensor per element (the tensile principal strain).
    """
    e = np.asarray(tostrain, float)
    out = np.zeros(len(e))
    for i, (xx, yy, zz, xy, yz, zx) in enumerate(e):
        T = np.array([[xx, xy, zx], [xy, yy, yz], [zx, yz, zz]])
        out[i] = np.linalg.eigvalsh(T)[-1]
    return out


def _triax_tear_margin(principal: np.ndarray, S: np.ndarray | None, eps_tear0: float,
                       k: float, q: float) -> float:
    """Robust percentile of the triaxiality-aware tear damage ``D = principal / eps_f(eta)``.

    The tolerable tensile strain rises in shear/compression and falls in biaxial tension via the
    same Johnson-Cook-like locus used for steel ductile damage (``eps_f(eta)=eps_tear0*exp(-k(eta-
    1/3))``, so ``eps_f=eps_tear0`` at uniaxial tension). When no stress is supplied the locus
    collapses to the constant ``eps_tear0`` (uniaxial criterion).
    """
    if S is not None and np.size(S):
        S = np.asarray(S, float)
        if S.ndim == 2 and S.shape[1] >= 6 and S.shape[0] == principal.shape[0]:
            eps_f = fracture_locus(stress_triaxiality(S[:, :6]), eps_f0=eps_tear0, k=k)
            return float(np.percentile(principal / eps_f, q))
    return float(np.percentile(principal, q)) / eps_tear0


def _column_average(coords: np.ndarray, arrays, tol: float = 0.05):
    """Average per-node ``arrays`` over through-thickness columns (nodes sharing an in-plane x,y).

    Groups nodes by their reference in-plane position (rounded to ``tol`` mm) — one group per
    extruded column — and returns each array averaged within its column, one row per column. This
    removes the bending strain gradient (tension outer / compression inner), leaving the membrane
    (mid-surface) response.
    """
    key = np.round(coords[:, :2] / tol).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    n = int(inv.max()) + 1
    cnt = np.bincount(inv, minlength=n)
    out = []
    for A in arrays:
        A = np.asarray(A, float)
        acc = np.stack([np.bincount(inv, weights=A[:, c], minlength=n) for c in range(A.shape[1])],
                       axis=1)
        out.append(acc / cnt[:, None])
    return out


def tear_from_frame(frame: dict, *, eps_tear0: float = 0.03, k: float = 1.5,
                    q: float = 99.0) -> float:
    """Surface tensile-tear margin ``D`` (>=1 => tear), triaxiality-aware.

    Uses the per-node max principal strain and (when the stress field is present) the
    triaxiality-dependent fracture locus. This still sees the full bending surface strain; for
    the bending-aware measure use :func:`membrane_tear_from_frame`.
    """
    E = frame.get("TOSTRAIN")
    if E is None or not np.size(E):
        return float("nan")
    principal = max_principal_strain(np.asarray(E, float))
    return _triax_tear_margin(principal, frame.get("STRESS"), eps_tear0, k, q)


def membrane_tear_from_frame(frame: dict, coords: np.ndarray, *, eps_tear0: float = 0.03,
                             k: float = 1.5, q: float = 99.0) -> float:
    """Bending- AND triaxiality-aware tear margin ``D`` (>=1 => tear).

    Averages the strain and stress tensors through the thickness (per in-plane column) so the
    bending gradient cancels and the membrane (mid-surface) state remains, then applies the
    triaxiality-dependent fracture locus. This is the physically-faithful paper criterion: a pure
    fold (no membrane stretch) accumulates little membrane strain and does not tear regardless of
    fold angle, matching that paper creases to 180 deg without membrane-tearing.
    """
    E = frame.get("TOSTRAIN")
    if E is None or not np.size(E) or coords is None:
        return tear_from_frame(frame, eps_tear0=eps_tear0, k=k, q=q)
    coords = np.asarray(coords, float)
    E = np.asarray(E, float)
    S = frame.get("STRESS")
    if S is not None and np.size(S) and np.asarray(S).shape == E.shape:
        E_col, S_col = _column_average(coords, [E, np.asarray(S, float)])
    else:
        (E_col,) = _column_average(coords, [E]); S_col = None
    return _triax_tear_margin(max_principal_strain(E_col), S_col, eps_tear0, k, q)


def damage_from_frame(frame: dict, *, eps_f0: float = 0.25, k: float = 1.5, q: float = 99.0) -> float:
    """Robust ductile-damage percentile ``D`` from a parsed CalculiX frame (PEEQ + STRESS).

    Single source of truth for turning a raw .frd frame into the scalar failure margin, shared
    by the parser and by :meth:`SteelJ2.failure`. Returns NaN if either field is absent or the
    element counts disagree. ``D >= 1`` => fracture.

    Args:
        frame: parsed frame dict with keys ``PEEQ``/``PE`` (per-element plastic strain) and
            ``STRESS`` (N x 6 Cauchy tensor).
    """
    peeq = None
    for key in ("PEEQ", "PE"):
        if key in frame and np.size(frame[key]):
            arr = np.abs(np.asarray(frame[key], float))
            peeq = arr[:, 0] if arr.ndim > 1 else arr
            break
    S = frame.get("STRESS")
    if peeq is None or S is None or not np.size(S):
        return float("nan")
    S = np.asarray(S, float)
    if S.ndim != 2 or S.shape[1] < 6 or S.shape[0] != peeq.shape[0]:
        return float("nan")
    _, D_p99, _ = ductile_damage(peeq, S[:, :6], eps_f0=eps_f0, k=k, q=q)
    return D_p99


def ductile_damage(peeq: np.ndarray, S: np.ndarray, *, eps_f0: float = 0.25, k: float = 1.5,
                   q: float = 99.0):
    """Continuous per-element damage and a robust scalar margin.

    Args:
        peeq: (N,) equivalent plastic strain per element.
        S:    (N, 6) stress tensor per element (same ordering/elements as ``peeq``).
        eps_f0: fracture strain at uniaxial tension (the old constant; anchors the locus).
        k:    triaxiality sensitivity of the fracture locus.
        q:    percentile for the robust scalar aggregate.

    Returns:
        (D_per_elem, D_margin, eta_per_elem): ``D_margin`` >= 1 => fracture.
    """
    peeq = np.abs(np.asarray(peeq, float))
    eta = stress_triaxiality(np.asarray(S, float))
    eps_f = fracture_locus(eta, eps_f0=eps_f0, k=k)
    D = peeq / eps_f
    return D, float(np.percentile(D, q)), eta
