"""Проверки каталога и существования безопасных обходов пространственных сцен."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from manipulator_planning.collision import CollisionChecker
from manipulator_planning.robot import RobotModel
from manipulator_planning.scenes import list_scenes, load_named_scene


def test_catalog_describes_all_packaged_scenes():
    catalog = list_scenes()
    assert [item["name"] for item in catalog] == [
        "simple",
        "scattered",
        "gate",
        "shelves",
        "cluttered",
    ]
    for item in catalog:
        assert item["description"]
        assert item["obstacle_count"] == len(load_named_scene(item["name"])["obstacles"])
        if item["name"] != "simple":
            assert item["obstacle_count"] >= 6


@pytest.mark.parametrize("name", [item["name"] for item in list_scenes()])
def test_scenes_have_valid_endpoints_and_checked_detour(name):
    scene = load_named_scene(name)
    robot = RobotModel(scene)
    checker = CollisionChecker(robot, scene["safety_margin"])
    assert checker.is_valid(scene["start"])
    assert checker.is_valid(scene["goal"])
    assert not checker.is_motion_valid(scene["start"], scene["goal"])
    detour = scene["diagnostic_detour"]
    np.testing.assert_array_equal(detour[0], scene["start"])
    np.testing.assert_array_equal(detour[-1], scene["goal"])
    assert all(checker.is_motion_valid(a, b) for a, b in zip(detour[:-1], detour[1:], strict=True))


@pytest.mark.parametrize("name", ["scattered", "gate", "shelves", "cluttered"])
def test_editable_scene_examples_match_package(name):
    source = Path(__file__).resolve().parents[1] / "configs" / "scenes" / f"{name}.yaml"
    assert yaml.safe_load(source.read_text(encoding="utf-8")) == load_named_scene(name)


def test_loading_does_not_share_mutable_scene_data():
    first = load_named_scene("gate")
    first["obstacles"][0]["position"][0] = 100
    assert load_named_scene("gate")["obstacles"][0]["position"][0] != 100
    with pytest.raises(ValueError, match="Неизвестная сцена"):
        load_named_scene("../simple")
