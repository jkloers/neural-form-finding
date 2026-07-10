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
