"""The replayed-polyline path must not disturb the proportional ray it generalises."""
import numpy as np

from nff.rve.ccx_solver import _states_at_times
from nff.rve.hinge_function import DeploymentPath, DeploymentRay, HingeGeometry

GEO = HingeGeometry(w_lig=18.0, alpha_deg=90.0, fillet_ratio=0.16)


def test_ray_increment_labels_match_the_old_proportional_formula():
    """parse_job used to label theta as t*angle/n_steps and (a,s) as a1*theta/theta1."""
    ray = DeploymentRay(theta1_deg=60.0, eta_a=0.4, eta_s=-0.2, n_steps=20)
    a1, s1, th1 = ray.targets(GEO)
    times = np.sort(np.concatenate([np.arange(1, 21.0),
                                    np.random.default_rng(0).uniform(0, 20, 64)]))
    u = _states_at_times(ray.states(GEO), times)

    old_theta = times * 60.0 / 20
    assert np.allclose(np.degrees(u[:, 2]), old_theta, atol=1e-12)
    assert np.allclose(u[:, 0], a1 * old_theta / th1, atol=1e-12)
    assert np.allclose(u[:, 1], s1 * old_theta / th1, atol=1e-12)


def test_replay_labels_follow_the_polyline_not_a_straight_ray():
    """A curved path and a straight ray to the same endpoint must label (a,s) differently."""
    curved = np.array([[0.0, 0.0, 0.2], [0.1, 0.9, 0.4], [1.0, 1.0, 0.6]])
    path = DeploymentPath(polyline=curved)
    ray = np.outer(np.arange(1, 4) / 3.0, curved[-1])

    t = np.array([0.5, 1.5, 2.5, 3.0])
    u_path = _states_at_times(path.states(GEO), t)
    u_ray = _states_at_times(ray, t)

    assert np.allclose(u_path[-1], curved[-1])                 # same destination
    assert np.allclose(u_ray[-1], curved[-1])
    assert not np.allclose(u_path[1], u_ray[1])                # different route
    assert path.n_steps == 3
    assert np.isclose(path.theta1_deg, np.degrees(0.6))


def test_states_are_clamped_to_the_driven_range():
    """A solver time past the last step must not extrapolate the imposed motion."""
    poly = np.array([[1.0, 0.0, 0.1], [2.0, 0.0, 0.2]])
    u = _states_at_times(poly, [0.0, 2.0, 5.0])
    assert np.allclose(u[0], 0.0)                              # origin at t = 0
    assert np.allclose(u[1], poly[-1])
    assert np.allclose(u[2], poly[-1])
