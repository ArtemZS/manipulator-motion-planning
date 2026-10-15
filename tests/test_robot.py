import numpy as np
import pytest

from manipulator_planning.robot import RobotModel


def test_forward_kinematics_and_validation():
    robot = RobotModel()
    anchors = robot.joint_positions(np.zeros(6))
    assert anchors.shape == (7, 3)
    np.testing.assert_allclose(anchors[-1], [1.18, 0, 0.4], atol=1e-12)
    np.testing.assert_allclose(robot.tool_position(np.zeros(6)), anchors[-1])
    with pytest.raises(ValueError):
        robot.set_configuration([0, 0, 0])
    with pytest.raises(ValueError):
        robot.set_configuration([0, 0, 0, 0, 0, np.nan])


def test_gravity_matches_potential_gradient_and_mass_is_positive():
    robot = RobotModel()
    q = np.array([0.3, -0.5, 0.8, 0.4, -0.7, 0.2])
    matrix = robot.mass_matrix(q)
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(matrix)) > 0
    gravity = robot.inverse_dynamics(q, np.zeros(6), np.zeros(6))

    def potential(value):
        robot.set_configuration(value)
        return -np.sum(robot.model.body_mass[:, None] * robot.data.xipos * robot.model.opt.gravity)

    eps = 1e-6
    gradient = np.array(
        [(potential(q + eps * axis) - potential(q - eps * axis)) / (2 * eps) for axis in np.eye(6)]
    )
    np.testing.assert_allclose(gravity, gradient, atol=2e-8)
    qdd = np.array([0.4, -0.2, 0.6, -0.1, 0.5, 0.8])
    np.testing.assert_allclose(
        robot.inverse_dynamics(q, np.zeros(6), qdd), gravity + matrix @ qdd, atol=1e-12
    )


def test_motion_bound_covers_observed_displacements():
    robot = RobotModel()
    rng = np.random.default_rng(7)
    for _ in range(40):
        q = rng.uniform(robot.joint_lower, robot.joint_upper)
        dq = rng.uniform(-0.3, 0.3, 6)
        before = robot.joint_positions(q)
        after = robot.joint_positions(q + dq)
        assert np.max(np.linalg.norm(after - before, axis=1)) <= robot.motion_bound(dq)
