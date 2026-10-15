"""Проверка геометрии и входных данных воспроизводимого отчёта."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from manipulator_planning.robot import RobotModel
from manipulator_planning.trajectory import fit_path
from manipulator_planning.variants import interpolate_rrt_path, rrt_tool_path, select_variants
from manipulator_planning.visualization import (
    _VIEW_ANGLES,
    _capsule_faces,
    _joint_figure,
    _save_animation,
    _scene_figure,
    create_report,
)


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


def test_scene_presets_and_fixed_goal_coordinates() -> None:
    robot = RobotModel()
    goal = np.array([0.6, -0.2, 0.15, 0.0, -0.1, 0.0])
    trajectory = fit_path([np.zeros(6), goal])
    for view, (elevation, azimuth, _) in _VIEW_ANGLES.items():
        figure = _scene_figure(robot, {"goal": goal}, trajectory, None, view)
        try:
            ax = figure.axes[0]
            assert ax.elev == elevation
            assert ax.azim == azimuth
            expected = robot.tool_position(goal)
            label = next(text.get_text() for text in ax.texts if "ЦЕЛЬ:" in text.get_text())
            assert all(f"{coordinate:.3f}" in label for coordinate in expected)
            assert "ЦЕЛЬ" in ax.get_legend_handles_labels()[1]
        finally:
            figure.clear()


def test_report_generates_requested_gif_views_and_self_contained_html(tmp_path: Path) -> None:
    from PIL import Image

    robot = RobotModel()
    goal = np.array([0.6, -0.2, 0.15, 0.0, -0.1, 0.0])
    trajectory = fit_path([np.zeros(6), goal])
    artifacts = create_report(
        tmp_path,
        robot,
        {"goal": goal},
        [trajectory],
        np.array([[10, 20, 4]]),
        0,
        views=("front", "top"),
        frame_count=3,
        rrt_path=[np.zeros(6), goal],
    )
    assert (
        Path(artifacts["animation"]).read_bytes() == Path(artifacts["animation_front"]).read_bytes()
    )
    assert (
        Path(artifacts["animation_top"]).read_bytes() != Path(artifacts["animation"]).read_bytes()
    )
    for key in ("animation_front", "animation_top"):
        with Image.open(artifacts[key]) as gif:
            assert gif.n_frames == 3
    assert Path(artifacts["scene_top"]).exists()
    html = Path(artifacts["viewer"]).read_text(encoding="utf-8")
    assert "Plotly.addFrames" in html
    assert "data:image/gif;base64," in html
    assert Path(artifacts["interactive"]).exists()
    assert Path(artifacts["comparison"]).exists()
    records = json.loads(Path(artifacts["variants_json"]).read_text(encoding="utf-8"))["variants"]
    assert [record["key"] for record in records] == [
        "min_time",
        "min_effort",
        "min_jerk",
        "balanced",
    ]
    assert all(record["solution_index"] == 0 for record in records)
    assert all(len(record["same_solution_as"]) == 3 for record in records)
    with Path(artifacts["variants_csv"]).open(encoding="utf-8-sig", newline="") as source:
        csv_records = list(csv.DictReader(source))
    assert len(csv_records) == 4
    assert float(csv_records[-1]["T"]) == trajectory.duration
    for key in ("viewer", "interactive"):
        content = Path(artifacts[key]).read_text(encoding="utf-8")
        assert "Четыре представителя фронта Парето" in content
        assert "Максимальная плавность" in content
        assert "Источник анимации: сбалансированный B-сплайн" in content
    with pytest.raises(ValueError, match="views"):
        create_report(tmp_path, robot, {}, [trajectory], np.zeros((1, 3)), 0, views=("bad",))


def test_representatives_select_correct_criteria_and_keep_duplicate_indices() -> None:
    objectives = np.array([[10, 50, 3], [5, 200, 5], [20, 10, 7], [9, 40, 4]])
    records = select_variants(objectives, 3)
    assert [record["solution_index"] for record in records] == [0, 1, 2, 3]
    assert all(record["same_solution_as"] == [] for record in records)
    for record in records:
        np.testing.assert_array_equal(
            [record["J_tau"], record["J_jerk"], record["T"]],
            objectives[record["solution_index"]],
        )
    duplicate_records = select_variants(objectives, 0)
    assert duplicate_records[0]["same_solution_as"] == ["balanced"]
    assert duplicate_records[-1]["same_solution_as"] == ["min_time"]


def test_rrt_workspace_curve_comes_from_dense_joint_interpolation() -> None:
    robot = RobotModel()
    start, goal = np.zeros(6), np.array([np.pi / 2, 0, 0, 0, 0, 0])
    samples = interpolate_rrt_path([start, goal])
    np.testing.assert_array_equal(samples[0], start)
    np.testing.assert_array_equal(samples[-1], goal)
    assert np.max(np.abs(np.diff(samples, axis=0))) <= 0.02 + 1e-12
    tool = rrt_tool_path(robot, [start, goal])
    assert len(tool) > 50
    # Поворот первого сустава описывает дугу окружности, а не хорду между FK узлов.
    np.testing.assert_allclose(np.linalg.norm(tool[:, :2], axis=1), np.linalg.norm(tool[0, :2]))
    assert np.linalg.norm(tool[len(tool) // 2] - (tool[0] + tool[-1]) / 2) > 0.2


def test_gif_configurations_are_samples_of_selected_bspline(tmp_path: Path, monkeypatch) -> None:
    from PIL import Image

    robot = RobotModel()
    start = np.array([-0.4, -0.2, 0.1, 0.1, 0, 0])
    goal = np.array([0.5, -0.4, 0.3, 0, 0.2, 0.1])
    trajectory = fit_path([start, goal], duration=3.7)
    original = robot.joint_positions
    observed = []

    def record_configuration(q):
        observed.append(np.array(q, copy=True))
        return original(q)

    monkeypatch.setattr(robot, "joint_positions", record_configuration)
    destination = tmp_path / "balanced.gif"
    _save_animation(destination, robot, {"goal": goal}, trajectory, frame_count=5)
    np.testing.assert_allclose(observed[:5], trajectory.evaluate(np.linspace(0, 1, 5)))
    with Image.open(destination) as gif:
        assert gif.n_frames == 5
