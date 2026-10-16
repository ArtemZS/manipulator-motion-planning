from pathlib import Path

import mujoco
import numpy as np
import yaml

from manipulator_planning.collision import CollisionChecker
from manipulator_planning.robot import RobotModel


def test_scenario_requires_detour_and_has_one():
    scene = yaml.safe_load((Path(__file__).parents[1] / "configs/simple.yaml").read_text())
    robot = RobotModel(scene)
    checker = CollisionChecker(robot, scene["safety_margin"])
    assert checker.is_valid(scene["start"])
    assert checker.is_valid(scene["goal"])
    assert not checker.is_motion_valid(scene["start"], scene["goal"])
    detour = scene["diagnostic_detour"]
    assert all(checker.is_motion_valid(a, b) for a, b in zip(detour, detour[1:], strict=False))
    assert checker.check_count > 0


def test_every_link_is_checked_against_world_and_nonadjacent_links():
    robot = RobotModel()
    checker = CollisionChecker(robot)
    base = robot.model.geom("base").id
    floor = robot.model.geom("floor").id
    for i, link in enumerate(robot.link_geom_ids):
        assert (link, floor) in checker.pairs
        assert ((link, base) in checker.pairs) == (i != 0)
        for j in range(i + 2, 6):
            assert (link, robot.link_geom_ids[j]) in checker.pairs
    # Опущенная рука пересекает основание или пол.
    assert not checker.is_valid([0, 1.5, 0, 0, 0, 0])
    # Сложенные локоть и запястье вводят капсулу инструмента в капсулу основания.
    folded = [-0.89, -0.49, 2.57, 0.17, 2.22, 0.084]
    robot.set_configuration(folded)
    assert (
        mujoco.mj_geomDistance(
            robot.model, robot.data, robot.link_geom_ids[0], robot.link_geom_ids[5], 100.0, None
        )
        < 0
    )
    assert not checker.is_valid(folded)


def test_interval_certification_rejects_thin_unsampled_obstacle():
    q = np.zeros(6)
    robot = RobotModel(
        {
            "obstacles": [
                {"kind": "sphere", "name": "tiny", "position": [0.88, 0, 0.4], "radius": 0.005}
            ]
        }
    )
    checker = CollisionChecker(robot, 0.001)
    a, b = q.copy(), q.copy()
    a[0], b[0] = -0.15, 0.15
    assert checker.is_valid(a) and checker.is_valid(b)
    assert not checker.is_motion_valid(a, b)
    assert not checker.is_valid([4, 0, 0, 0, 0, 0])
    # Неразрешённость при заданной глубине ведёт к консервативному отклонению.
    empty = CollisionChecker(RobotModel())
    assert not empty.is_motion_valid(a, b, max_depth=0)
