from pathlib import Path

import numpy as np
import pytest
import yaml

from manipulator_planning.collision import CollisionChecker
from manipulator_planning.objectives import evaluate_objectives
from manipulator_planning.robot import RobotModel
from manipulator_planning.trajectory import JointSplineTrajectory, fit_path
from manipulator_planning.validation import certify_collision_free, validate_trajectory


@pytest.fixture
def lifted_case():
    """Поднятая часть известного обхода сферы с фиксированными концами."""
    scene_path = Path(__file__).parents[1] / "configs/simple.yaml"
    scene = yaml.safe_load(scene_path.read_text(encoding="utf-8"))
    scene["start"], scene["goal"] = scene["diagnostic_detour"][1:3]
    robot = RobotModel(scene)
    checker = CollisionChecker(robot, scene["safety_margin"])
    trajectory = fit_path([scene["start"], scene["goal"]], n_control_points=12, duration=6)
    return robot, checker, trajectory


def test_continuous_clearance_bound_is_below_independent_dense_samples(lifted_case):
    robot, checker, trajectory = lifted_case
    certified, sampled, lower_bound, intervals = certify_collision_free(trajectory, checker)
    assert certified and intervals > 1
    assert checker.safety_margin <= lower_bound <= sampled
    independent_minimum = min(
        checker.minimum_clearance(q) for q in trajectory.evaluate(np.linspace(0, 1, 401))
    )
    assert lower_bound <= independent_minimum + 1e-12
    assert validate_trajectory(trajectory, robot, checker).feasible
    # Геометрическая гарантия не зависит от длительности прохождения сплайна.
    assert certify_collision_free(trajectory.with_duration(0.1), checker) == (
        certified,
        sampled,
        lower_bound,
        intervals,
    )


def test_collision_between_dynamic_samples_is_rejected():
    scene_path = Path(__file__).parents[1] / "configs/simple.yaml"
    scene = yaml.safe_load(scene_path.read_text(encoding="utf-8"))
    robot = RobotModel(scene)
    checker = CollisionChecker(robot, scene["safety_margin"])
    trajectory = fit_path([scene["start"], scene["goal"]], n_control_points=6, duration=8)
    # Оба узла грубой динамической сетки допустимы, но середина пересекает сферу.
    assert all(checker.is_valid(q) for q in trajectory.sample(2)["q"])
    assert checker.minimum_clearance(trajectory.evaluate(0.5)) < 0
    report = validate_trajectory(trajectory, robot, checker, samples=2)
    assert not report.feasible and not report.collision_certified
    assert report.minimum_sampled_clearance < 0
    assert "collision" in report.reason


def test_time_scaling_rejects_excess_velocity_and_acceleration(lifted_case):
    robot, checker, trajectory = lifted_case
    report = validate_trajectory(trajectory.with_duration(0.1), robot, checker, samples=21)
    assert report.collision_certified
    assert report.velocity_ratio > 1 and report.acceleration_ratio > 1
    assert not report.feasible
    assert "velocity" in report.reason and "acceleration" in report.reason


@pytest.mark.parametrize("coefficient", [0, 1])
def test_modified_endpoint_or_endpoint_derivative_is_rejected(lifted_case, coefficient):
    robot, checker, trajectory = lifted_case
    points = trajectory.control_points.copy()
    points[coefficient, 0] += 0.02
    tampered = JointSplineTrajectory(
        points, trajectory.knots, trajectory.degree, trajectory.duration
    )
    report = validate_trajectory(tampered, robot, checker, samples=31)
    if coefficient == 0:
        assert report.endpoint_position_error == pytest.approx(0.02)
    else:
        assert report.endpoint_position_error < 1e-12
        assert report.endpoint_velocity_error > 0.01
    assert not report.feasible
    assert "endpoint" in report.reason


def test_effort_of_stationary_arm_equals_gravity_squared_times_duration():
    robot = RobotModel()
    q = np.array([0.2, -0.4, 0.3, 0.1, -0.2, 0.3])
    trajectory = fit_path([q, q], n_control_points=6, duration=3)
    gravity = robot.inverse_dynamics(q, np.zeros(6), np.zeros(6))
    objectives, torques = evaluate_objectives(trajectory, robot, quadrature_order=6)
    np.testing.assert_allclose(objectives, [3 * gravity @ gravity, 0, 3], atol=1e-12)
    np.testing.assert_allclose(torques, np.broadcast_to(gravity, torques.shape), atol=1e-12)
    doubled, _ = evaluate_objectives(trajectory.with_duration(6), robot)
    np.testing.assert_allclose(doubled, objectives * 2, atol=1e-12)


def test_quadrature_converges_and_jerk_uses_physical_time(lifted_case):
    robot, _, trajectory = lifted_case
    coarse, _ = evaluate_objectives(trajectory, robot, quadrature_order=2)
    medium, _ = evaluate_objectives(trajectory, robot, quadrature_order=6)
    refined, _ = evaluate_objectives(trajectory, robot, quadrature_order=12)
    assert abs(medium[0] - refined[0]) < abs(coarse[0] - refined[0])
    np.testing.assert_allclose(medium, refined, rtol=1e-9)
    # При удвоении T интеграл квадрата третьей производной уменьшается в 2**5 раз.
    doubled, _ = evaluate_objectives(trajectory.with_duration(2 * trajectory.duration), robot)
    assert doubled[1] == pytest.approx(medium[1] / 32, rel=1e-12)
    samples = trajectory.sample(2001)
    torque = np.array(
        [
            robot.inverse_dynamics(q, qd, qdd)
            for q, qd, qdd in zip(samples["q"], samples["qd"], samples["qdd"], strict=True)
        ]
    )
    independent_effort = np.trapezoid(np.sum(torque**2, axis=1), samples["time"])
    assert medium[0] == pytest.approx(independent_effort, rel=2e-7)


def test_sampled_torque_limit_is_reported_separately():
    robot = RobotModel()
    robot.torque_limits = np.full(6, 0.01)
    checker = CollisionChecker(robot)
    q = np.zeros(6)
    trajectory = fit_path([q, q], n_control_points=6, duration=6)
    report = validate_trajectory(trajectory, robot, checker, samples=17)
    assert report.collision_certified and report.velocity_ratio == 0
    assert report.torque_ratio > 1 and not report.feasible
    assert report.dynamic_samples == 17
    assert "sampled torque" in report.reason
