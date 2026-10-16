"""Подготовка входных данных и сборка Pages из сохранённых численных результатов."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import shutil
from pathlib import Path

import yaml
from export_showcase import export_showcase

from manipulator_planning.experiments import (
    _create_report,
    _flatten_summary,
    _write_csv,
)
from manipulator_planning.io import write_json
from manipulator_planning.pipeline import replay_results
from manipulator_planning.showcase import _RESULT_FILES, _series_design, _write_gallery


def read_manifest(directory: Path) -> dict:
    """Прочитать переносимый список сцен, исключив выход за входной каталог."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    names = [item["scene"] for item in manifest["scenes"]]
    if not names or len(names) != len(set(names)):
        raise ValueError("Нужен непустой список сцен без повторов")
    if any(re.fullmatch(r"[a-z0-9_]+", name) is None for name in names):
        raise ValueError("Некорректное имя сцены")
    return manifest


def snapshot_showcase(source: Path, inputs: Path) -> dict:
    """Сохранить только численные входы галереи, без HTML, GIF и локальных путей."""
    source, inputs = source.resolve(), inputs.resolve()
    if source == inputs or inputs.is_relative_to(source) or source.is_relative_to(inputs):
        raise ValueError("Входы публикации должны храниться отдельно от локальной галереи")
    manifest = read_manifest(source)
    entries = []
    for original in manifest["scenes"]:
        name = original["scene"]
        for filename in _RESULT_FILES:
            if not (source / name / filename).is_file():
                raise FileNotFoundError(source / name / filename)
        entry = {
            key: original[key]
            for key in (
                "scene",
                "seed",
                "purpose",
                "selection",
                "animation",
                "title",
                "description",
                "notes",
            )
            if key in original
        }
        entry["views"] = [
            view
            for view in ("isometric", "front", "side", "top")
            if (source / name / f"animation_{view}.gif").is_file()
        ]
        entry["views"] = entry["views"] or ["isometric", "side"]
        entries.append(entry)
    for entry in entries:
        target = inputs / entry["scene"]
        target.mkdir(parents=True, exist_ok=True)
        for filename in _RESULT_FILES:
            shutil.copyfile(source / entry["scene"] / filename, target / filename)
    snapshot = {"schema_version": 1, "frame_count": 121, "scenes": entries}
    write_json(inputs / "manifest.json", snapshot)
    return {"scenes": [entry["scene"] for entry in entries], "inputs": str(inputs)}


def scene_statistics(summary: dict, name: str) -> dict:
    """Выделить сцену из общей сводки, сохранив её исходные числа и бюджет."""
    result = copy.deepcopy(summary)
    result["scenes"] = [name]
    for key in ("methods", "paired"):
        result[key] = [item for item in result[key] if item["scene"] == name]
    if len(result["methods"]) != 2:
        raise ValueError(f"Нет статистики обоих методов для сцены {name}")
    for key in ("normalization_by_scene", "options_by_scene", "protocol_by_scene"):
        if key in result:
            result[key] = {name: result[key][name]}
    result["options"] = result.get("options_by_scene", {}).get(name, result["options"])
    result["protocol"] = result.get("protocol_by_scene", {}).get(name, result["protocol"])
    return result


def validate_scene_snapshots(inputs: Path, project: Path) -> None:
    """Не допускать публикацию старых расчётов под изменившейся геометрией."""
    for entry in read_manifest(inputs)["scenes"]:
        name = entry["scene"]
        saved = yaml.safe_load((inputs / name / "config.yaml").read_text(encoding="utf-8"))
        paths = [
            project / "configs" / ("simple.yaml" if name == "simple" else f"scenes/{name}.yaml"),
            project
            / "src/manipulator_planning/assets"
            / ("simple.yaml" if name == "simple" else f"scenes/{name}.yaml"),
        ]
        for path in paths:
            if path.is_file() and yaml.safe_load(path.read_text(encoding="utf-8")) != saved:
                raise ValueError(
                    f"Сцена {name} изменена в {path}. Пересчитайте её локально, "
                    "обновите статистику и выполните just pages-snapshot перед push"
                )


def build_pages(inputs: Path, statistics: Path, output: Path, *, frames: int | None = None) -> dict:
    """Проверить сплайны и пересобрать все иллюстрации в чистом рабочем каталоге."""
    inputs, statistics, output = inputs.resolve(), statistics.resolve(), output.resolve()
    work = output.with_name(output.name + "-work")
    for path in (output, work):
        if path.exists() and any(path.iterdir()):
            raise ValueError(f"Требуется пустой выходной каталог: {path}")
    manifest = read_manifest(inputs)
    validate_scene_snapshots(inputs, Path(__file__).resolve().parents[1])
    summary = json.loads((statistics / "summary.json").read_text(encoding="utf-8"))
    with (statistics / "runs.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    # CSV содержит строки, а научные графики ожидают исходные логические и числовые типы.
    for row in rows:
        row["success"] = row["success"] == "True"
        for metric in ("J_tau", "J_jerk", "T", "runtime_numerical_s", "hypervolume"):
            row[metric] = float(row[metric]) if row[metric] else None
    entries = []
    for saved in manifest["scenes"]:
        entry = copy.deepcopy(saved)
        name = entry["scene"]
        target = work / name
        target.mkdir(parents=True, exist_ok=True)
        for filename in _RESULT_FILES:
            shutil.copyfile(inputs / name / filename, target / filename)
        print(f"Пересборка сцены {name}, seed={entry['seed']}", flush=True)
        replay_results(
            target,
            views=tuple(entry["views"]),
            frame_count=frames if frames is not None else manifest["frame_count"],
        )
        scene = yaml.safe_load((target / "config.yaml").read_text(encoding="utf-8"))
        for key in ("title", "description", "notes"):
            if key in scene:
                entry[key] = scene[key]
        # Отдельные разделы всегда производны от общей сводки, без второй копии выборки.
        detail = target / "statistics"
        detail.mkdir()
        filtered = scene_statistics(summary, name)
        scene_rows = [row for row in rows if row["scene"] == name]
        write_json(detail / "summary.json", filtered)
        _write_csv(detail / "runs.csv", scene_rows)
        _write_csv(detail / "summary.csv", _flatten_summary(filtered))
        _create_report(filtered, scene_rows, detail)
        entry["statistics"] = {"directory": f"{name}/statistics"}
        entries.append(entry)
    _write_gallery(work, entries, statistics, _series_design(summary))
    write_json(work / "manifest.json", {"scenes": entries})
    result = export_showcase(work, statistics, output)
    result["statistics_pairs"] = len(rows) // 2
    write_json(
        output / "build-info.json",
        {
            "revision": os.environ.get("GITHUB_SHA"),
            "run_id": os.environ.get("GITHUB_RUN_ID"),
            "scenes": result["scenes"],
            "statistics_pairs": result["statistics_pairs"],
        },
    )
    result["files"] += 1
    return result


def main() -> None:
    """Выбрать сохранение входов или сборку сайта для GitHub Actions."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    snapshot = commands.add_parser("snapshot", help="Сохранить входы локальной галереи в Git")
    snapshot.add_argument("--source", type=Path, default=Path("results/showcase"))
    snapshot.add_argument("--inputs", type=Path, default=Path("docs/showcase"))
    build = commands.add_parser("build", help="Собрать сайт из версионируемых результатов")
    build.add_argument("--inputs", type=Path, default=Path("docs/showcase"))
    build.add_argument("--statistics", type=Path, default=Path("docs/experiments"))
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--frames", type=int, help="Уменьшить число кадров при локальной проверке")
    args = parser.parse_args()
    result = (
        snapshot_showcase(args.source, args.inputs)
        if args.command == "snapshot"
        else build_pages(args.inputs, args.statistics, args.output, frames=args.frames)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
