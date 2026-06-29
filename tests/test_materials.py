"""SvK energy / strain post-processing — pure NumPy, no oracle needed."""
import numpy as np
from materials import svk_energy, max_principal_strain_per_tet, vm_stress_per_tet


def _unit_tet():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    tets = np.array([[0, 1, 2, 3]], dtype=int)
    return nodes, tets


def test_undeformed_has_zero_energy_and_strain():
    n, t = _unit_tet()
    assert abs(svk_energy(n, n, t, 3.5e9, 0.36)) < 1e-9
    assert abs(max_principal_strain_per_tet(n, n, t)[0]) < 1e-9


def test_stretched_tet_has_positive_energy_and_strain():
    n, t = _unit_tet()
    cur = n.copy()
    cur[1, 0] = 1.2   # stretch along x
    assert svk_energy(n, cur, t, 3.5e9, 0.36) > 0.0
    assert max_principal_strain_per_tet(n, cur, t)[0] > 0.0
    assert vm_stress_per_tet(n, cur, t, 3.5e9, 0.36)[0] > 0.0
