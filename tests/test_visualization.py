"""Проверка геометрии и входных данных воспроизводимого отчёта."""

from pathlib import Path

import numpy as np
import pytest

from manipulator_planning.robot import RobotModel
from manipulator_planning.trajectory import fit_path
from manipulator_planning.visualization import _capsule_faces, _joint_figure, create_report


def test_capsule_mesh_uses_link_endpoints_and_physical_radius() -> None:
    a = np.array([0.4, -0.3, 0.2])
    b = np.array([0.4, -0.3, 1.0])
    mesh = np.concatenate(_capsule_faces(a, b, radius=0.05))
    np.testing.assert_allclose(mesh.min(axis=0), [0.35, -0.35, 0.15], atol=1e-12)
    np.testing.assert_allclose(mesh.max(axis=0), [0.45, -0.25, 1.05], atol=1e-12)
    assert np.isfinite(mesh).all()


def test_report_rejects_inconsistent_or_nonfinite_objectives(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="objectives"):
        create_report(tmp_path, None, {}, [], np.zeros((1, 3)), 0)
    with pytest.raises(ValueError, match="objectives"):
        create_report(tmp_path, None, {}, [None], np.array([[np.nan, 2, 3]]), 0)
    with pytest.raises(ValueError, match="selected_index"):
        create_report(tmp_path, None, {}, [], np.empty((0, 3)), 0)


def test_joint_figure_keeps_six_joint_curves_and_shows_actual_limit_usage() -> None:
    robot = RobotModel()
    trajectory = fit_path([np.zeros(6), np.array([0.5, -0.3, 0.2, 0.1, -0.1, 0.2])])
    figure = _joint_figure(robot, trajectory)
    try:
        assert len(figure.axes) == 6
        assert all(len(ax.lines) == 6 for ax in figure.axes[:5])
        usage_ax = figure.axes[-1]
        assert usage_ax.get_ylabel() == "Доля предела"
        assert len(usage_ax.lines) == 4
        values = trajectory.sample(301)
        expected_velocity = np.max(np.abs(values["qd"]) / robot.velocity_limits, axis=1)
        expected_acceleration = np.max(np.abs(values["qdd"]) / robot.acceleration_limits, axis=1)
        expected_torque = np.max(
            np.abs(
                [
                    robot.inverse_dynamics(q, qd, qdd)
                    for q, qd, qdd in zip(values["q"], values["qd"], values["qdd"], strict=True)
                ]
            )
            / robot.torque_limits,
            axis=1,
        )
        for line, expected in zip(
            usage_ax.lines[:3],
            (expected_velocity, expected_acceleration, expected_torque),
            strict=True,
        ):
            np.testing.assert_allclose(line.get_ydata(), expected)
        np.testing.assert_array_equal(usage_ax.lines[-1].get_ydata(), [1.0, 1.0])
        assert usage_ax.lines[-1].get_linestyle() == "--"
        assert usage_ax.get_ylim()[1] > 1.0
    finally:
        figure.clear()
