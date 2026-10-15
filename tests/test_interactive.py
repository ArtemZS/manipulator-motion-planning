"""Проверка данных объёмной анимации и полностью автономного HTML."""

import re
from pathlib import Path

import numpy as np
import pytest

from manipulator_planning.interactive import (
    CAMERA_PRESETS,
    build_interactive_figure,
    interactive_fragment,
    write_interactive_viewer,
)
from manipulator_planning.robot import RobotModel
from manipulator_planning.trajectory import fit_path
from manipulator_planning.variants import select_variants
from manipulator_planning.visualization import _capsule_faces


def test_goal_stays_at_scene_target_and_frames_follow_original_spline() -> None:
    robot = RobotModel()
    start = np.array([-0.6, -0.2, 0.25, 0.1, 0.0, 0.1])
    end = np.array([0.5, -0.3, 0.2, 0.1, -0.1, 0.2])
    # Намеренно отличающаяся цель обнаруживает подмену цели концом нарисованного пути.
    target = np.array([0.8, -0.4, 0.1, 0.0, 0.2, 0.0])
    trajectory = fit_path([start, end], duration=3.25)
    figure = build_interactive_figure(robot, {"goal": target}, trajectory, frame_count=7)
    expected_goal = robot.tool_position(target)
    np.testing.assert_allclose(figure.layout.meta["goal"], expected_goal)
    goal_trace = next(trace for trace in figure.data if trace.name == "ЦЕЛЬ — фиксированная точка")
    np.testing.assert_allclose([goal_trace.x[0], goal_trace.y[0], goal_trace.z[0]], expected_goal)
    assert len(figure.frames) == 7
    expected_tool = np.array([robot.tool_position(q) for q in trajectory.sample(7)["q"]])
    for frame, position in zip(figure.frames, expected_tool, strict=True):
        tool = frame.data[-1]
        np.testing.assert_allclose([tool.x[0], tool.y[0], tool.z[0]], position)
        assert len(frame.data) == 8
        assert all(trace.type == "mesh3d" for trace in frame.data[:6])
        assert frame.layout.scene.camera.eye.x is None
    steps = figure.layout.sliders[0].steps
    assert steps[-1].label == "3.25"
    assert figure.layout.scene.dragmode == "orbit"
    assert figure.layout.scene.aspectmode == "cube"
    assert figure.layout.uirevision
    assert len(figure.layout.updatemenus[1].buttons) == len(CAMERA_PRESETS)
    assert figure.layout.title.y == 0.99
    assert figure.layout.title.yanchor == "top"
    play_menu, camera_menu = figure.layout.updatemenus
    assert play_menu.x == camera_menu.x == 0
    assert play_menu.y == 1.22
    assert camera_menu.y == 1.10
    assert camera_menu.type == "dropdown"
    assert camera_menu.direction == "down"
    assert play_menu.yanchor == camera_menu.yanchor == "top"
    assert figure.layout.margin.t == 175


def test_meshes_include_box_sphere_and_distinct_preset_cameras() -> None:
    robot = RobotModel()
    trajectory = fit_path([np.zeros(6), [0.4, -0.3, 0.2, 0.0, 0.1, 0.0]])
    scene = {
        "obstacles": [
            {
                "name": "Куб",
                "kind": "box",
                "position": [0.3, 0.4, 0.2],
                "half_size": [0.1, 0.2, 0.15],
            },
            {"name": "Шар", "kind": "sphere", "position": [0.7, -0.2, 0.4], "radius": 0.1},
        ]
    }
    figure = build_interactive_figure(robot, scene, trajectory, frame_count=2, view="top")
    assert figure.layout.scene.camera.eye.to_plotly_json() == CAMERA_PRESETS["top"]["eye"]
    box = next(trace for trace in figure.data if trace.name == "Куб")
    np.testing.assert_allclose([min(box.x), min(box.y), min(box.z)], [0.2, 0.2, 0.05])
    np.testing.assert_allclose([max(box.x), max(box.y), max(box.z)], [0.4, 0.6, 0.35])
    sphere = next(trace for trace in figure.data if trace.name == "Шар")
    np.testing.assert_allclose([min(sphere.x), min(sphere.y), min(sphere.z)], [0.6, -0.3, 0.3])
    assert len(sphere.i) > 100
    assert len({tuple(camera["eye"].values()) for camera in CAMERA_PRESETS.values()}) == 4


def test_html_embeds_plotly_and_animation_without_external_script(tmp_path: Path) -> None:
    robot = RobotModel()
    trajectory = fit_path([np.zeros(6), [0.4, -0.3, 0.2, 0, 0, 0]])
    fragment = interactive_fragment(robot, {}, trajectory, frame_count=2)
    destination = tmp_path / "interactive.html"
    write_interactive_viewer(destination, fragment)
    html = destination.read_text(encoding="utf-8")
    assert "Plotly.newPlot" in html
    assert "Plotly.addFrames" in html
    assert "plotly.js v" in html
    assert not re.search(r"<script[^>]+src=", html)
    assert 'id="manipulator-3d"' in html
    assert "Жёлтый маркер" in html
    assert "ЦЕЛЬ — фиксированное" in html
    assert '"scrollZoom": true' in html
    assert "Вращаемая 3D-сцена" in html
    with pytest.raises(ValueError, match="ракурс"):
        build_interactive_figure(robot, {}, trajectory, frame_count=2, view="unknown")


def test_comparison_paths_do_not_replace_balanced_bspline_animation() -> None:
    robot = RobotModel()
    start = np.array([-0.3, -0.2, 0.2, 0.1, 0, 0])
    goal = np.array([0.4, -0.4, 0.3, -0.1, 0.1, 0])
    base = fit_path([start, goal])
    trajectories = [base.with_duration(duration) for duration in (2.0, 3.0, 5.0, 4.0)]
    records = select_variants(np.array([[10, 50, 2], [5, 200, 3], [20, 10, 5], [9, 40, 4]]), 3)
    detour = np.array([-1.2, -0.8, 0.1, 0, 0.2, 0.2])
    figure = build_interactive_figure(
        robot,
        {"goal": goal},
        trajectories[3],
        base,
        frame_count=5,
        trajectories=trajectories,
        records=records,
        rrt_path=[start, detour, goal],
    )
    for record in records:
        assert any(trace.name.startswith(record["label"]) for trace in figure.data)
    rrt_trace = next(trace for trace in figure.data if trace.name.startswith("RRT*"))
    assert len(rrt_trace.x) > 50
    assert rrt_trace.line.dash == "dot"
    assert any(trace.name == "B-сплайн без NSGA-II" for trace in figure.data)
    assert figure.layout.meta["animation_source"] == "balanced_bspline"
    assert figure.layout.meta["animation_solution_index"] == 3
    np.testing.assert_allclose(figure.layout.meta["sample_times"], np.linspace(0, 4, 5))
    # Каждая объёмная поза берётся из выбранного B-сплайна, независимо от линии RRT*.
    for frame, q in zip(figure.frames, trajectories[3].sample(5)["q"], strict=True):
        positions = robot.joint_positions(q)
        for index, mesh in enumerate(frame.data[:6]):
            expected = np.concatenate(
                _capsule_faces(positions[index], positions[index + 1], robot.link_radii[index])
            )
            np.testing.assert_allclose(np.column_stack((mesh.x, mesh.y, mesh.z)), expected)
