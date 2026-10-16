"""Галерея выбирает успешные запуски и сохраняет происхождение их чисел."""

import csv
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from manipulator_planning import experiments
from manipulator_planning.pipeline import PipelineOptions
from manipulator_planning.showcase import (
    _RESULT_FILES,
    _STATISTICS_FILES,
    add_showcase_run,
    add_showcase_statistics,
    create_showcase,
)


def test_gallery_uses_successful_seed_and_preserves_saved_spline(tmp_path, monkeypatch):
    series = tmp_path / "series"
    series.mkdir()
    (series / "summary.json").write_text(
        json.dumps({"scenes": ["gate", "simple"], "seeds": [0, 1]})
    )
    for name, seed, success in (
        ("gate", 0, False),
        ("gate", 1, True),
        ("gate", 99, True),
        ("simple", 0, True),
    ):
        run = series / "runs" / name / f"seed_{seed}"
        run.mkdir(parents=True)
        pair = {
            "completed": True,
            "scene": name,
            "seed": seed,
            "fingerprint": f"{name}-{seed}",
            "report": {"status": "success" if success else "failed"},
        }
        (run / "pair.json").write_text(json.dumps(pair))
        for filename in _RESULT_FILES:
            (run / filename).write_text(f"{name}-{seed}-{filename}")
    rendered = []

    def replay(directory, *, views, frame_count):
        rendered.append((directory.name, views, frame_count))
        (directory / "variants.csv").write_text("variant,duration_s\nbalanced,4\n")

    monkeypatch.setattr("manipulator_planning.showcase.replay_results", replay)
    output = tmp_path / "gallery"
    # Предпочтительный seed 99 существует на диске, но отсутствует в текущей серии.
    result = create_showcase(series, output, frame_count=17, preferred_seed=99)
    assert [(item["scene"], item["seed"]) for item in result["scenes"]] == [
        ("gate", 1),
        ("simple", 0),
    ]
    assert rendered == [("gate", ("isometric", "top"), 17), ("simple", ("isometric", "top"), 17)]
    assert (output / "gate" / "trajectories.json").read_text() == "gate-1-trajectories.json"
    assert (output / "gate" / "origin.json").exists()
    assert "gate/view.html" in (output / "index.html").read_text(encoding="utf-8")
    assert "../series/index.html" in (output / "index.html").read_text(encoding="utf-8")
    assert len((output / "variants.csv").read_text(encoding="utf-8-sig").splitlines()) == 3


def _saved_demo(directory: Path, seed: int = 37) -> None:
    """Минимальный численный запуск для проверки добавления без реального рендера."""
    directory.mkdir(parents=True)
    for filename in _RESULT_FILES:
        (directory / filename).write_text(f"saved-{filename}", encoding="utf-8")
    (directory / "report.json").write_text(
        json.dumps({"status": "success", "scene": "narrow_passage", "options": {"seed": seed}}),
        encoding="utf-8",
    )
    (directory / "config.yaml").write_text(
        "title: Узкий проход\ndescription: Локоть огибает верхнюю грань препятствия.\n",
        encoding="utf-8",
    )


def _five_scene_gallery(tmp_path: Path, monkeypatch) -> tuple[Path, Path, list]:
    """Пять готовых анимаций с выбором успешного seed вместо неудачного нулевого."""
    series, output = tmp_path / "series", tmp_path / "gallery"
    series.mkdir()
    names = ["simple", "scattered", "gate", "shelves", "cluttered"]
    (series / "summary.json").write_text(json.dumps({"scenes": names, "seeds": [0, 1]}))
    for name in names:
        for seed in (0, 1):
            run = series / "runs" / name / f"seed_{seed}"
            run.mkdir(parents=True)
            pair = {
                "completed": True,
                "scene": name,
                "seed": seed,
                "fingerprint": f"{name}-{seed}",
                "report": {"status": "success" if seed == 1 else "failed"},
            }
            (run / "pair.json").write_text(json.dumps(pair))
            for filename in _RESULT_FILES:
                (run / filename).write_text(f"{name}-{seed}-{filename}")
    rendered = []

    def replay(directory, *, views, frame_count):
        rendered.append((directory.name, views, frame_count))
        (directory / "variants.csv").write_text("variant,duration_s\nbalanced,4\n")
        for name in ("animation.gif", *(f"animation_{view}.gif" for view in views)):
            (directory / name).write_bytes(f"{directory.name}-{name}".encode())

    monkeypatch.setattr("manipulator_planning.showcase.replay_results", replay)
    create_showcase(series, output, frame_count=17)
    return series, output, rendered


def test_append_renders_only_new_demo_and_preserves_five_existing_seeds_and_animations(
    tmp_path, monkeypatch
):
    series, output, rendered = _five_scene_gallery(tmp_path, monkeypatch)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["custom_provenance"] = {"preserve": True}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    preserved = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for entry in manifest["scenes"]
        for path in (output / entry["scene"]).iterdir()
    }
    source_summary = (series / "summary.json").read_bytes()
    source_pairs = {path: path.read_bytes() for path in series.glob("runs/*/seed_*/pair.json")}
    source = tmp_path / "narrow_run"
    _saved_demo(source)
    rendered.clear()
    result = add_showcase_run(source, output, frame_count=29)
    assert result["added"] is result["rendered"] is True
    assert rendered == [("narrow_passage", ("isometric", "side"), 29)]
    updated = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert updated["scenes"][:5] == manifest["scenes"]
    assert [entry["seed"] for entry in updated["scenes"][:5]] == [1] * 5
    assert updated["custom_provenance"] == {"preserve": True}
    assert updated["source_series"] == str(series.resolve())
    added = updated["scenes"][-1]
    assert added["scene"] == "narrow_passage"
    assert added["seed"] == 37
    assert added["purpose"] == "demo"
    assert added["description"] == "Локоть огибает верхнюю грань препятствия."
    assert len(added["source_fingerprint"]) == 64
    assert json.loads((output / "narrow_passage/origin.json").read_text(encoding="utf-8")) == added
    for path, original in preserved.items():
        assert (path.read_bytes(), path.stat().st_mtime_ns) == original
    assert (series / "summary.json").read_bytes() == source_summary
    assert all(path.read_bytes() == content for path, content in source_pairs.items())
    with (output / "variants.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6
    assert [row["purpose"] for row in rows] == ["series_example"] * 5 + ["demo"]
    page = (output / "index.html").read_text(encoding="utf-8")
    assert "5 сцен × 2 seed = 10 пар" in page
    assert "Дополнительная демонстрация" not in page
    assert "narrow_passage/pareto.png" in page
    assert "Локоть огибает верхнюю грань препятствия." in page
    assert "Пять сцен" not in page
    assert "Все 20 строк" not in page
    rendered.clear()
    repeated = add_showcase_run(source, output, frame_count=89)
    assert repeated["added"] is repeated["rendered"] is False
    assert not rendered
    assert len(repeated["scenes"]) == 6


def test_append_rejects_conflicting_names_and_failed_runs_without_changing_manifest(
    tmp_path, monkeypatch
):
    _, output, rendered = _five_scene_gallery(tmp_path, monkeypatch)
    source = tmp_path / "new_run"
    _saved_demo(source)
    original_manifest = (output / "manifest.json").read_bytes()
    rendered.clear()
    with pytest.raises(ValueError, match="уже представлена"):
        add_showcase_run(source, output, "simple")
    with pytest.raises(ValueError, match="Имя сцены"):
        add_showcase_run(source, output, "../outside")
    report = {"status": "failed", "options": {"seed": 37}}
    (source / "report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="успешный"):
        add_showcase_run(source, output)
    assert (output / "manifest.json").read_bytes() == original_manifest
    assert not rendered


def _statistics_series(directory: Path) -> dict:
    """Серия содержит общий отказ поиска и отдельный отказ после получения базы."""
    directory.mkdir()

    def solution(values):
        return {
            "objectives": values,
            "validation": {"feasible": True, "collision_clearance_lower_bound": 0.02},
        }

    success = {
        "status": "success",
        "baseline": solution([6, 6, 6]),
        "solutions": [solution([9, 2, 4]), solution([2, 9, 4])],
        "selected_index": 0,
        "timings_seconds": {"planning": 1, "smoothing": 2, "optimization_and_validation": 7},
    }
    pairs = [
        {"scene": "gate", "seed": 0, "report": success},
        {
            "scene": "gate",
            "seed": 1,
            "report": {
                "status": "failed",
                "failure_stage": "planning",
                "reason": "Нет пути <script>",
            },
        },
        {
            "scene": "gate",
            "seed": 2,
            "report": {
                **success,
                "status": "failed",
                "failure_stage": "optimization",
                "reason": "Отказ",
            },
        },
        {"scene": "gate", "seed": 3, "report": success},
    ]
    rows = [
        row
        for pair in pairs
        for row in experiments._paired_rows("gate", pair["seed"], pair["report"], "runs/test")
    ]
    summary = {
        **experiments.summarize_experiments(pairs, rows),
        "scenes": ["gate"],
        "seeds": list(range(4)),
        "options": asdict(PipelineOptions()),
        "protocol": {"workers": 2},
    }
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    experiments._write_csv(directory / "runs.csv", rows)
    for filename in _STATISTICS_FILES:
        if not (directory / filename).exists():
            (directory / filename).write_text(f"snapshot-{filename}", encoding="utf-8")
    return summary


def test_statistics_snapshot_shows_paired_results_and_failures_without_touching_animations(
    tmp_path, monkeypatch
):
    _, output, rendered = _five_scene_gallery(tmp_path, monkeypatch)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    preserved = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for entry in manifest["scenes"]
        for path in (output / entry["scene"]).iterdir()
    }
    series = tmp_path / "statistics"
    _statistics_series(series)
    source = {path.name: path.read_bytes() for path in series.iterdir()}
    rendered.clear()
    result = add_showcase_statistics(series, output)
    assert not rendered
    assert result["statistics"]["pairs"] == 4
    assert len(result["statistics"]["source_fingerprint"]) == 64
    page = (output / "index.html").read_text(encoding="utf-8")
    assert 'id="statistics-gate"' in page
    assert "Статистика по 4 seed" in page
    assert "3/4" in page and "2/4" in page
    assert "95% бутстреп-ДИ" in page
    assert "50 [50; 50]" in page
    assert "-66.67 [-66.67; -66.67]" in page
    assert "Нет пути &lt;script&gt;" in page and "Нет пути <script>" not in page
    assert "planning" in page and "optimization" in page
    assert 'href="gate/statistics/runs.csv"' in page
    for path, original in preserved.items():
        assert (path.read_bytes(), path.stat().st_mtime_ns) == original
    for filename, original in source.items():
        assert (output / "gate/statistics" / filename).read_bytes() == original
        assert (series / filename).read_bytes() == original
    updated = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert updated["series_design"] == manifest["series_design"]
    for entry, original in zip(updated["scenes"], manifest["scenes"], strict=True):
        assert {key: value for key, value in entry.items() if key != "statistics"} == original
    add_showcase_statistics(series, output)
    assert len(json.loads((output / "manifest.json").read_text(encoding="utf-8"))["scenes"]) == 5
    assert not rendered


def test_statistics_rejects_incomplete_pairs_before_changing_gallery(tmp_path, monkeypatch):
    _, output, _ = _five_scene_gallery(tmp_path, monkeypatch)
    series = tmp_path / "statistics"
    _statistics_series(series)
    original_manifest = (output / "manifest.json").read_bytes()
    original_index = (output / "index.html").read_bytes()
    with (series / "runs.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    experiments._write_csv(series / "runs.csv", rows[:-1])
    with pytest.raises(ValueError, match="ровно два метода"):
        add_showcase_statistics(series, output)
    assert (output / "manifest.json").read_bytes() == original_manifest
    assert (output / "index.html").read_bytes() == original_index
    assert not (output / "gate/statistics").exists()
