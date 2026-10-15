"""Наглядные отчёты всех сцен по уже рассчитанной серии экспериментов."""

import csv
import html
import json
import os
import shutil
from pathlib import Path

from .pipeline import replay_results, write_json
from .variants import variants_html

_RESULT_FILES = (
    "config.yaml",
    "report.json",
    "baseline.json",
    "trajectories.json",
    "trajectories.npz",
    "pareto.csv",
)


def create_showcase(
    series_dir: Path,
    output_dir: Path,
    *,
    views: tuple[str, ...] = ("isometric", "top"),
    frame_count: int = 121,
    preferred_seed: int = 0,
) -> dict:
    """Повторно проверить и показать по одному успешному запуску каждой сцены.

    Выбор seed служит демонстрации и не меняет статистическую выборку серии.
    При отказе предпочтительного seed берётся первый успешный по номеру.
    Планирование и оптимизация здесь не выполняются.
    """
    series_dir, output_dir = series_dir.resolve(), output_dir.resolve()
    summary = json.loads((series_dir / "summary.json").read_text(encoding="utf-8"))
    if output_dir == series_dir or series_dir.is_relative_to(output_dir):
        raise ValueError("Галерея должна находиться отдельно от исходной серии")
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed_seeds = set(summary["seeds"])
    entries = []
    for scene_name in summary["scenes"]:
        candidates = []
        for pair_path in (series_dir / "runs" / scene_name).glob("seed_*/pair.json"):
            pair = json.loads(pair_path.read_text(encoding="utf-8"))
            if (
                pair.get("seed") in allowed_seeds
                and pair.get("scene") == scene_name
                and pair.get("completed")
                and pair.get("report", {}).get("status") == "success"
            ):
                candidates.append((int(pair["seed"]), pair_path.parent, pair))
        candidates.sort(key=lambda item: (item[0] != preferred_seed, item[0]))
        if not candidates:
            raise ValueError(f"Нет успешного запуска для сцены {scene_name}")
        seed, source, pair = candidates[0]
        target = output_dir / scene_name
        if target == source:
            raise ValueError("Нельзя перезаписать исходный запуск галереей")
        target.mkdir(parents=True, exist_ok=True)
        for filename in _RESULT_FILES:
            shutil.copyfile(source / filename, target / filename)
        origin = {
            "scene": scene_name,
            "seed": seed,
            "source_run": str(source),
            "source_fingerprint": pair["fingerprint"],
            "selection": "Предпочтительный seed, затем первый успешный; только демонстрация",
            "animation": "Сбалансированный сохранённый B-сплайн после NSGA-II",
        }
        write_json(target / "origin.json", origin)
        print(f"Галерея: {scene_name}, seed={seed}, ракурсы={', '.join(views)}", flush=True)
        replay_results(target, views=views, frame_count=frame_count)
        entries.append(origin)
    _write_gallery(output_dir, entries, series_dir)
    write_json(output_dir / "manifest.json", {"source_series": str(series_dir), "scenes": entries})
    return {"viewer": str(output_dir / "index.html"), "scenes": entries}


def _write_gallery(output_dir: Path, entries: list[dict], series_dir: Path) -> None:
    """Собрать обзор сцен и единую таблицу четырёх представителей каждого фронта."""
    statistics_link = Path(os.path.relpath(series_dir / "index.html", output_dir)).as_posix()
    sections = []
    combined = []
    for entry in entries:
        name = entry["scene"]
        variants_file = output_dir / name / "variants.csv"
        table = ""
        if variants_file.exists():
            with variants_file.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            combined.extend({"scene": name, "seed": entry["seed"], **row} for row in rows)
        variants_json = output_dir / name / "variants.json"
        if variants_json.exists():
            records = json.loads(variants_json.read_text(encoding="utf-8"))["variants"]
            table = variants_html(records)
        safe = html.escape(name, quote=True)
        sections.append(
            f"<section><h2>{safe} · seed {entry['seed']}</h2>"
            f'<p><a href="{safe}/view.html">Открыть сцену, анимации и параметры</a> · '
            f'<a href="{safe}/interactive.html">Вращаемая 3D-сцена</a> · '
            f'<a href="{safe}/variants.csv">Таблица CSV</a></p>'
            f'<a href="{safe}/view.html"><img src="{safe}/comparison.png" '
            f'alt="Четыре траектории, RRT* и базовый сплайн: {safe}"></a>' + table + "</section>"
        )
    if combined:
        with (output_dir / "variants.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(combined[0]))
            writer.writeheader()
            writer.writerows(combined)
    page = (
        '<!doctype html><html lang="ru"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Пять сцен: движение манипулятора 6R</title><style>"
        "body{font:16px/1.5 system-ui,sans-serif;max-width:1200px;margin:auto;padding:24px;"
        "background:#f3f6fa;color:#203448}section{background:white;padding:24px;margin:24px 0;"
        "border-radius:12px}img{width:100%;height:auto}a{color:#225ba8}.table{overflow:auto}"
        "table{border-collapse:collapse;font-size:13px}th,td{padding:6px 10px;text-align:right;"
        "border-bottom:1px solid #ddd}</style><h1>Пять сцен: движение манипулятора 6R</h1>"
        "<p>Для каждой сцены показаны минимальное время, минимальные затраты усилий, "
        "минимальный рывок (максимальная плавность) и сбалансированный вариант. "
        "Это представители найденного приближённого фронта. Совпадение номеров решений "
        "означает совпадение траекторий. Анимации воспроизводят сбалансированный B-сплайн.</p>"
        "<p>Здесь по одному демонстрационному seed; статистика использует все 20 seed "
        f'каждой сцены. <a href="{html.escape(statistics_link, quote=True)}">'
        'Открыть статистику серии</a>. <a href="variants.csv">Все 20 строк параметров CSV</a>.</p>'
        + "".join(sections)
        + "</html>"
    )
    (output_dir / "index.html").write_text(page, encoding="utf-8")
