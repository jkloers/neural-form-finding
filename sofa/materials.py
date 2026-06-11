"""
materials.py — SvK energy and von Mises stress for tetrahedral sub-meshes.

These are analytical post-processing functions: SOFA solves the FEM equilibrium,
then we compute energies and stresses from the equilibrium node positions.
"""

import numpy as np


def svk_energy(pos_nat: np.ndarray, pos_cur: np.ndarray,
               tets: np.ndarray, young: float, nu: float) -> float:
    """
    Saint Venant-Kirchhoff total strain energy over a tet mesh [J].

    W = (λ/2)(tr E)² + μ‖E‖²_F,   E = ½(FᵀF − I),   F = dx · dX⁻¹
    """
    lam = young * nu / ((1 + nu) * (1 - 2*nu))
    mu  = young / (2 * (1 + nu))
    total = 0.0
    for tet in tets:
        dX = (pos_nat[tet[1:]] - pos_nat[tet[0]]).T
        dx = (pos_cur[tet[1:]] - pos_cur[tet[0]]).T
        det_dX = np.linalg.det(dX)
        if abs(det_dX) < 1e-30:
            continue
        F = dx @ np.linalg.inv(dX)
        E = 0.5 * (F.T @ F - np.eye(3))
        trE = np.trace(E)
        total += (lam/2 * trE**2 + mu * np.sum(E**2)) * abs(det_dX) / 6.0
    return total


def vm_stress_per_tet(pos_nat: np.ndarray, pos_cur: np.ndarray,
                      tets: np.ndarray, young: float, nu: float) -> np.ndarray:
    """Per-element von Mises (Cauchy) stress [Pa]. SvK constitutive law."""
    lam = young * nu / ((1 + nu) * (1 - 2*nu))
    mu  = young / (2 * (1 + nu))
    vm_list = []
    for tet in tets:
        dX = (pos_nat[tet[1:]] - pos_nat[tet[0]]).T
        dx = (pos_cur[tet[1:]] - pos_cur[tet[0]]).T
        det_dX = np.linalg.det(dX)
        if abs(det_dX) < 1e-30:
            continue
        F = dx @ np.linalg.inv(dX)
        J = np.linalg.det(F)
        if abs(J) < 1e-10:
            continue
        E = 0.5 * (F.T @ F - np.eye(3))
        S = lam * np.trace(E) * np.eye(3) + 2*mu * E
        sigma = F @ S @ F.T / J
        s = sigma - np.trace(sigma)/3 * np.eye(3)
        vm_list.append(float(np.sqrt(1.5 * np.sum(s**2))))
    return np.array(vm_list) if vm_list else np.array([0.0])
