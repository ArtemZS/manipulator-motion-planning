"""Физический проход через окно, складывание локтя и независимый поиск маршрута."""

import numpy as np

from manipulator_planning.collision import CollisionChecker
from manipulator_planning.pipeline import PipelineOptions, _build_baseline
from manipulator_planning.planning import RRTStar
from manipulator_planning.robot import RobotModel
from manipulator_planning.scenes import load_named_scene


def _check_plane_crossings(robot, scene, configurations):
    """Проверить настоящие пересечения осей звеньев плоскостью окна."""
    passage = scene["passage"]
    coordinate = passage["coordinate"]
    margin = scene["safety_margin"]
    crossings = []
    tools = []
    for q in configurations:
        points = robot.joint_positions(q)
        tools.append(points[-1])
        for index, (a, b) in enumerate(zip(points[:-1], points[1:], strict=True)):
            delta = b[0] - a[0]
            if abs(delta) < 1e-12 or (a[0] - coordinate) * (b[0] - coordinate) > 0:
                continue
            point = a + (coordinate - a[0]) / delta * (b - a)
            for axis, axis_index in (("y", 1), ("z", 2)):
                lower, upper = passage["bounds"][axis]
                allowance = robot.link_radii[index] + margin
                assert lower + allowance <= point[axis_index] <= upper - allowance
            crossings.append((point, abs(q[2])))
    assert crossings
    assert max(angle for _, angle in crossings) >= np.deg2rad(60)
    tools = np.asarray(tools)
    assert tools[0, 0] < coordinate - passage["thickness"] / 2
    assert tools[-1, 0] > coordinate + passage["thickness"] / 2


def test_frame_closes_every_external_route_within_robot_reach():
    scene = load_named_scene("narrow_passage")
    robot = RobotModel(scene)
    links = list(robot.link_geom_ids)
    # Сумма длин всех капсул и наибольшего радиуса ограничивает полную досягаемость.
    reach = 2 * np.sum(robot.model.geom_size[links, 1]) + max(robot.link_radii)
    wall = scene["passage"]["wall_bounds"]
    assert wall["y"][0] < -reach and wall["y"][1] > reach
    assert wall["z"][0] == 0
    assert wall["z"][1] > robot.joint_positions(scene["start"])[0, 2] + reach
    assert abs(scene["start"][2]) < np.deg2rad(10)
    assert abs(scene["goal"][2]) > np.deg2rad(70)


def test_diagnostic_motion_crosses_the_actual_opening_with_folded_elbow():
    scene = load_named_scene("narrow_passage")
    robot = RobotModel(scene)
    checker = CollisionChecker(robot, scene["safety_margin"])
    detour = np.asarray(scene["diagnostic_detour"])
    assert all(checker.is_motion_valid(a, b) for a, b in zip(detour[:-1], detour[1:], strict=True))
    samples = np.concatenate(
        [np.linspace(a, b, 151) for a, b in zip(detour[:-1], detour[1:], strict=True)]
    )
    _check_plane_crossings(robot, scene, samples)


def test_rrt_and_spline_find_passage_without_diagnostic_route():
    scene = load_named_scene("narrow_passage")
    scene.pop("diagnostic_detour")
    robot = RobotModel(scene)
    checker = CollisionChecker(robot, scene["safety_margin"])
    planner = RRTStar(
        robot,
        checker,
        seed=42,
        step_size=0.10,
        goal_bias=0.25,
        local_sample_rate=0.45,
        local_sample_scale=0.12,
    )
    result = planner.plan(scene["start"], scene["goal"], max_iterations=800, stop_on_first=True)
    assert result.success
    trajectory, validation, _ = _build_baseline(result.path, robot, checker, PipelineOptions())
    assert validation.feasible and validation.collision_certified
    sampled = trajectory.sample(501)["q"]
    assert np.ptp(sampled[:, 2]) > np.deg2rad(65)
    _check_plane_crossings(robot, scene, sampled)
