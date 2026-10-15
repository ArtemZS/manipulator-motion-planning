"""Смешанная выборка сохраняет воспроизводимость и исследует трудные seed."""

import numpy as np
import pytest

from manipulator_planning.collision import CollisionChecker
from manipulator_planning.pipeline import load_scene
from manipulator_planning.planning import RRTStar
from manipulator_planning.robot import RobotModel


def test_local_mixture_solves_previously_stalling_seed_reproducibly():
    scene = load_scene()
    robot = RobotModel(scene)
    checker = CollisionChecker(robot)
    paths = []
    for _ in range(2):
        planner = RRTStar(
            robot, checker, seed=1, step_size=0.1, goal_bias=0.25, local_sample_rate=0.45
        )
        result = planner.plan(scene["start"], scene["goal"], max_iterations=100, stop_on_first=True)
        assert result.success
        paths.append(result.path)
    np.testing.assert_array_equal(*paths)
    assert not checker.is_motion_valid(scene["start"], scene["goal"])


def test_invalid_sampling_probabilities_are_rejected():
    robot = RobotModel()
    checker = CollisionChecker(robot)
    with pytest.raises(ValueError):
        RRTStar(robot, checker, goal_bias=0.8, local_sample_rate=0.3)
    with pytest.raises(ValueError):
        RRTStar(robot, checker, local_sample_scale=np.nan)
