"""Проверки переносимых входов и отказа от публикации устаревших расчётов."""

import copy
import importlib
import json
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def pages(monkeypatch):
    """Загрузить тот же сборщик, который запускается в GitHub Actions."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("build_pages")


def saved_gallery(directory, pages, name="fixture"):
    """Создать небольшой набор файлов без запуска численного пайплайна."""
    run = directory / name
    run.mkdir(parents=True)
    for filename in pages._RESULT_FILES:
        (run / filename).write_bytes(b"saved numerical input\n")
    (run / "config.yaml").write_text("obstacles: []\n", encoding="utf-8")
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "frame_count": 7,
                "scenes": [
                    {
                        "scene": name,
                        "seed": 42,
                        "views": ["isometric"],
                        "source_run": "private/local/results",
                        "source_series": "private/series",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return run


def test_snapshot_keeps_numerical_inputs_without_private_paths(tmp_path, pages):
    source, inputs = tmp_path / "source", tmp_path / "inputs"
    run = saved_gallery(source, pages)
    (run / "interactive.html").write_text("generated HTML")
    (run / "animation_side.gif").write_bytes(b"generated GIF")
    original = (run / "trajectories.json").read_bytes()
    pages.snapshot_showcase(source, inputs)
    manifest = pages.read_manifest(inputs)
    assert manifest["scenes"][0]["seed"] == 42
    assert manifest["scenes"][0]["views"] == ["side"]
    assert "private" not in json.dumps(manifest)
    assert (inputs / "fixture/trajectories.json").read_bytes() == original
    assert sorted(path.name for path in (inputs / "fixture").iterdir()) == sorted(
        pages._RESULT_FILES
    )


def test_manifest_refuses_path_outside_inputs(tmp_path, pages):
    (tmp_path / "manifest.json").write_text(json.dumps({"scenes": [{"scene": "../outside"}]}))
    with pytest.raises(ValueError, match="имя сцены"):
        pages.read_manifest(tmp_path)


def test_scene_statistics_keeps_its_budget_and_original_numbers(pages):
    summary = {
        "scenes": ["simple", "gate"],
        "seeds": [0, 1],
        "options": {},
        "protocol": {"workers": 2},
        "options_by_scene": {"simple": {"population": 20}, "gate": {"population": 32}},
        "normalization_by_scene": {"simple": [1, 2], "gate": [3, 4]},
        "methods": [
            {"scene": name, "method": method, "succeeded": 2}
            for name in ("simple", "gate")
            for method in ("baseline", "full")
        ],
        "paired": [{"scene": "simple", "change": -10}, {"scene": "gate", "change": -25}],
    }
    before = copy.deepcopy(summary)
    result = pages.scene_statistics(summary, "gate")
    assert result["options"] == {"population": 32}
    assert result["paired"] == [{"scene": "gate", "change": -25}]
    assert result["seeds"] == [0, 1]
    assert summary == before


def test_changed_geometry_requires_updated_local_calculation(tmp_path, pages):
    inputs, project = tmp_path / "inputs", tmp_path / "project"
    saved_gallery(inputs, pages, "simple")
    config = project / "configs/simple.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("obstacles: []\n", encoding="utf-8")
    pages.validate_scene_snapshots(inputs, project)
    config.write_text(yaml.safe_dump({"obstacles": [{"radius": 0.5}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="just pages-snapshot"):
        pages.validate_scene_snapshots(inputs, project)


def test_failed_trajectory_check_does_not_create_published_site(tmp_path, pages, monkeypatch):
    inputs, statistics, output = tmp_path / "inputs", tmp_path / "statistics", tmp_path / "site"
    run = saved_gallery(inputs, pages)
    original = (run / "trajectories.json").read_bytes()
    statistics.mkdir()
    (statistics / "summary.json").write_text("{}")
    (statistics / "runs.csv").write_text("scene,seed,method,success\n")

    def invalid_trajectory(*args, **kwargs):
        raise RuntimeError("Сохранённая траектория недопустима")

    monkeypatch.setattr(pages, "replay_results", invalid_trajectory)
    with pytest.raises(RuntimeError, match="недопустима"):
        pages.build_pages(inputs, statistics, output)
    assert not output.exists()
    assert (run / "trajectories.json").read_bytes() == original
