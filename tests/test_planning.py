"""Геометрические проверки RRT*, не зависящие от физического движка."""

from importlib.resources import files
from itertools import pairwise

import numpy as np
import pytest
import yaml

from manipulator_planning.collision import CollisionChecker
from manipulator_planning.planning import RRTStar, _Node, simplify_path
from manipulator_planning.robot import RobotModel
from manipulator_planning.trajectory import fit_path
from manipulator_planning.validation import certify_collision_free


class BoundedRobot:
    joint_lower = -np.ones(6)
    joint_upper = np.ones(6)


class BoxChecker:
    """Прямоугольное препятствие в первых двух суставных координатах."""

    lower = np.array([-0.16, -0.6])
    upper = np.array([0.16, 0.6])

    def is_valid(self, q):
        q = np.asarray(q)
        return bool(
            np.all(q >= -1)
            and np.all(q <= 1)
            and not np.all((q[:2] >= self.lower) & (q[:2] <= self.upper))
        )

    def is_motion_valid(self, a, b, max_depth=12):
        a, b = np.asarray(a), np.asarray(b)
        if not self.is_valid(a) or not self.is_valid(b):
            return False
        enter, leave = 0.0, 1.0
        for axis in range(2):
            delta = b[axis] - a[axis]
            if abs(delta) < 1e-14:
                if a[axis] < self.lower[axis] or a[axis] > self.upper[axis]:
                    return True
                continue
            first, last = sorted(
                [
                    (self.lower[axis] - a[axis]) / delta,
                    (self.upper[axis] - a[axis]) / delta,
                ]
            )
            enter, leave = max(enter, first), min(leave, last)
            if enter > leave:
                return True
        return enter > leave


def test_rrt_reaches_exact_goal_and_every_edge_is_valid():
    checker = BoxChecker()
    planner = RRTStar(BoundedRobot(), checker, seed=7)
    start, goal = np.zeros(6), np.zeros(6)
    start[0], goal[0] = -0.8, 0.8
    result = planner.plan(start, goal, max_iterations=700)
    assert result.success, result.reason
    np.testing.assert_array_equal(result.path[0], start)
    np.testing.assert_array_equal(result.path[-1], goal)
    assert all(
        checker.is_motion_valid(a, b)
        for a, b in zip(result.path[:-1], result.path[1:], strict=True)
    )
    actual_cost = sum(
        planner.distance(a, b) for a, b in zip(result.path[:-1], result.path[1:], strict=True)
    )
    assert result.cost == pytest.approx(actual_cost)
    assert result.iterations == 700
    assert all(
        b[1] < a[1] for a, b in zip(result.cost_history[:-1], result.cost_history[1:], strict=True)
    )
    simplified = simplify_path(result.path, checker)
    assert len(simplified) <= len(result.path)
    assert all(checker.is_motion_valid(a, b) for a, b in pairwise(simplified))


def test_rewire_propagates_to_all_descendants_and_rejects_cycles():
    planner = RRTStar(BoundedRobot(), BoxChecker())
    locations = [
        np.array([x, y, 0, 0, 0, 0], dtype=float)
        for x, y in [
            (-0.8, 0.8),
            (0.8, 0.8),
            (0.6, 0.8),
            (0.4, 0.8),
            (-0.4, 0.8),
        ]
    ]
    planner.nodes = [
        _Node(locations[0], None, 0, {1, 4}),
        _Node(locations[1], 0, 0.8, {2}),
        _Node(locations[2], 1, 0.9, {3}),
        _Node(locations[3], 2, 1.0),
        _Node(locations[4], 0, 0.2),
    ]
    planner._reparent(2, 4)
    assert planner.nodes[2].cost == pytest.approx(0.7)
    assert planner.nodes[3].cost == pytest.approx(0.8)
    assert 2 not in planner.nodes[1].children
    assert 2 in planner.nodes[4].children
    with pytest.raises(ValueError, match="cycle"):
        planner._reparent(4, 3)


def test_search_failure_cannot_return_a_partial_path_as_success():
    class NoEdges:
        def is_valid(self, q):
            return True

        def is_motion_valid(self, a, b, max_depth=12):
            return bool(np.array_equal(a, b))

    planner = RRTStar(BoundedRobot(), NoEdges())
    result = planner.plan(-np.ones(6) * 0.5, np.ones(6) * 0.5, max_iterations=30)
    assert not result.success
    assert result.path is None
    assert np.isinf(result.cost)
    assert result.node_count == 1


def test_invalid_goal_and_joint_space_are_reported():
    planner = RRTStar(BoundedRobot(), BoxChecker())
    start = np.array([-0.8, 0, 0, 0, 0, 0])
    result = planner.plan(start, np.zeros(6))
    assert not result.success
    assert "goal" in result.reason.lower()
    with pytest.raises(ValueError, match="finite vector"):
        planner.plan(start, [0, 0])


def test_direct_connection_has_exact_normalized_cost():
    planner = RRTStar(BoundedRobot(), BoxChecker())
    start = np.array([-0.8, 0.8, 0, 0, 0, 0])
    goal = np.array([0.8, 0.8, 0, 0, 0, 0])
    result = planner.plan(start, goal)
    assert result.success
    assert result.cost == pytest.approx(0.8)
    assert result.iterations == 0


def test_real_6r_scene_reaches_goal_and_has_certified_spline():
    """Проверить путь и сплайн на настоящей 3D-сцене без подсказанного обхода."""
    source = files("manipulator_planning").joinpath("assets/simple.yaml")
    scene = yaml.safe_load(source.read_text(encoding="utf-8"))
    scene.pop("diagnostic_detour", None)
    robot = RobotModel(scene)
    checker = CollisionChecker(robot, scene["safety_margin"])
    assert not checker.is_motion_valid(scene["start"], scene["goal"])
    planner = RRTStar(robot, checker, seed=42, step_size=0.10, goal_bias=0.45)
    result = planner.plan(scene["start"], scene["goal"], max_iterations=180)
    assert result.success, result.reason
    np.testing.assert_array_equal(result.path[0], scene["start"])
    np.testing.assert_array_equal(result.path[-1], scene["goal"])
    assert all(checker.is_motion_valid(a, b) for a, b in pairwise(result.path))
    fitted = fit_path(simplify_path(result.path, checker), n_control_points=12)
    certified, _, lower_bound, _ = certify_collision_free(fitted, checker)
    assert certified
    assert lower_bound >= checker.safety_margin
