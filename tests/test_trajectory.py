import json

import numpy as np
import pytest
from scipy.integrate import simpson

from manipulator_planning.trajectory import JointSplineTrajectory, fit_path


@pytest.fixture
def trajectory():
    return fit_path(
        np.array(
            [
                [-0.8, 0.0, 0.1, 0.0, -0.2, 0.0],
                [-0.4, 0.8, 0.3, 0.2, -0.3, 0.2],
                [0.4, 0.8, 0.2, -0.3, 0.0, 0.1],
                [0.8, 0.0, 0.0, 0.0, 0.2, 0.0],
            ]
        ),
        n_control_points=18,
        duration=4,
    )


def test_fit_uses_actual_coefficients_and_rest_endpoints(trajectory):
    np.testing.assert_allclose(trajectory.evaluate(0), trajectory.control_points[0], atol=1e-12)
    np.testing.assert_allclose(trajectory.evaluate(1), trajectory.control_points[-1], atol=1e-12)
    for order in (1, 2):
        np.testing.assert_allclose(trajectory.evaluate([0, 1], order), 0, atol=1e-11)
    # Сглаженная траектория должна сохранить обход препятствия через верхнюю точку.
    assert trajectory.evaluate(0.5)[1] > 0.75
    internal = trajectory.knots[trajectory.degree + 1 : -trajectory.degree - 1]
    assert np.all(np.diff(internal) > 0)


def test_analytic_derivatives_match_finite_differences(trajectory):
    u = np.array([0.2, 0.43, 0.7])
    h = 1e-6
    for order in (1, 2, 3):
        numerical = (
            trajectory.evaluate(u + h, order - 1) - trajectory.evaluate(u - h, order - 1)
        ) / (2 * h * trajectory.duration)
        np.testing.assert_allclose(trajectory.evaluate(u, order), numerical, rtol=1e-5, atol=1e-5)


def test_real_time_scaling_and_integrated_jerk(trajectory):
    slower = trajectory.with_duration(trajectory.duration * 2)
    u = np.linspace(0, 1, 2001)
    np.testing.assert_allclose(slower.evaluate(u), trajectory.evaluate(u))
    for order in (1, 2, 3):
        np.testing.assert_allclose(
            slower.evaluate(u, order), trajectory.evaluate(u, order) / 2**order
        )
    original_jerk = simpson(
        np.sum(trajectory.evaluate(u, 3) ** 2, axis=1), x=u * trajectory.duration
    )
    slower_jerk = simpson(np.sum(slower.evaluate(u, 3) ** 2, axis=1), x=u * slower.duration)
    assert slower_jerk == pytest.approx(original_jerk / 2**5)


def test_global_derivative_bounds_are_conservative(trajectory):
    u = np.linspace(0, 1, 3001)
    for order in (1, 2, 3):
        observed = (
            np.max(np.abs(trajectory.evaluate(u, order)), axis=0) * trajectory.duration**order
        )
        assert np.all(observed <= trajectory.derivative_bounds(order) + 1e-10)


def test_serialization_roundtrip_and_sampling(trajectory):
    restored = JointSplineTrajectory.from_dict(json.loads(json.dumps(trajectory.to_dict())))
    sample = restored.sample(101)
    assert sample["q"].shape == (101, 6)
    assert sample["time"][-1] == trajectory.duration
    np.testing.assert_array_equal(restored.sample(101)["q"], trajectory.sample(101)["q"])


def test_stationary_path_and_input_validation():
    trajectory = fit_path(np.ones((3, 6)))
    np.testing.assert_allclose(trajectory.sample()["q"], 1)
    np.testing.assert_allclose(trajectory.sample()["qddd"], 0, atol=1e-10)
    with pytest.raises(ValueError, match="control points"):
        fit_path(np.zeros((2, 6)), n_control_points=5)
    with pytest.raises(ValueError, match="duration"):
        trajectory.with_duration(0)
    with pytest.raises(ValueError, match="interval"):
        trajectory.evaluate(1.01)
