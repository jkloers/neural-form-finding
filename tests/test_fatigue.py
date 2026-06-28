"""Coffin-Manson fatigue model — pure NumPy, no oracle needed."""
import pytest
from nff.sofa.fatigue import plastic_strain, cycles_to_failure


def test_no_plastic_strain_below_yield():
    assert plastic_strain(0.05, 0.10) == 0.0


def test_plastic_strain_above_yield():
    assert plastic_strain(0.15, 0.10) == pytest.approx(0.05)


def test_elastic_design_has_infinite_life():
    assert cycles_to_failure(0.0, 0.05, -0.6) == float("inf")


def test_plastic_design_has_finite_life():
    n = cycles_to_failure(0.05, 0.05, -0.6)
    assert 0.0 < n < float("inf")


def test_more_plastic_strain_means_fewer_cycles():
    assert (cycles_to_failure(0.10, 0.05, -0.6)
            < cycles_to_failure(0.02, 0.05, -0.6))
