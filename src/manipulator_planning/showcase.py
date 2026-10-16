"""Наглядные отчёты всех сцен по уже рассчитанной серии экспериментов."""

import csv
import hashlib
import html
import json
import os
import re
import shutil
from pathlib import Path

import yaml

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

_STATISTICS_FILES = (
    "summary.json",
    "summary.csv",
    "runs.csv",
    "REPORT.md",
    "index.html",
    "statistics.png",
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
            "purpose": "series_example",
        }
        write_json(target / "origin.json", origin)
        print(f"Галерея: {scene_name}, seed={seed}, ракурсы={', '.join(views)}", flush=True)
        replay_results(target, views=views, frame_count=frame_count)
        entries.append(origin)
    design = _series_design(summary)
    _write_gallery(output_dir, entries, series_dir, design)
    write_json(
        output_dir / "manifest.json",
        {"source_series": str(series_dir), "series_design": design, "scenes": entries},
    )
    return {"viewer": str(output_dir / "index.html"), "scenes": entries}


def _series_design(summary: dict) -> dict:
    """Зафиксировать состав основной серии отдельно от демонстрационной галереи."""
    scenes, seeds = list(summary.get("scenes", [])), list(summary.get("seeds", []))
    return {"scenes": scenes, "seeds": seeds, "pairs": len(scenes) * len(seeds)}


def _run_fingerprint(directory: Path) -> str:
    """Хешировать исходные численные файлы, включая точные коэффициенты сплайнов."""
    digest = hashlib.sha256()
    for filename in _RESULT_FILES:
        digest.update(filename.encode("utf-8") + b"\0")
        digest.update((directory / filename).read_bytes())
    return digest.hexdigest()


def add_showcase_run(
    run_dir: Path,
    output_dir: Path,
    scene_name: str | None = None,
    *,
    views: tuple[str, ...] = ("isometric", "side"),
    frame_count: int = 121,
) -> dict:
    """Добавить готовый запуск, отрисовав только его и обновив общий обзор.

    Старые сцены, их seed и происхождение сохраняются без изменений. Новый
    запуск имеет назначение ``demo`` и не расширяет статистическую выборку.
    Повторное добавление того же источника под тем же именем не запускает рендер.
    """
    run_dir, output_dir = Path(run_dir).resolve(), Path(output_dir).resolve()
    manifest_path = output_dir / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {"source_series": None, "scenes": []}
    )
    entries = list(manifest["scenes"])
    for filename in _RESULT_FILES:
        if not (run_dir / filename).is_file():
            raise ValueError(f"В исходном запуске отсутствует файл {filename}")
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "success":
        raise ValueError("Для галереи требуется успешный сохранённый запуск")
    name = str(scene_name if scene_name is not None else report.get("scene", run_dir.name))
    if not re.fullmatch(r"[\w-]+", name) or name in {".", ".."}:
        raise ValueError("Имя сцены должно содержать только буквы, цифры, '_' и '-'")
    seed = report.get("options", {}).get("seed", report.get("seeds", {}).get("rrt"))
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("В отчёте запуска отсутствует целочисленный seed планировщика")
    fingerprint = _run_fingerprint(run_dir)
    source_series = manifest.get("source_series")
    series_dir = Path(source_series) if source_series else None
    design = manifest.get("series_design")
    if design is None and series_dir is not None and (series_dir / "summary.json").is_file():
        design = _series_design(
            json.loads((series_dir / "summary.json").read_text(encoding="utf-8"))
        )
    target = (output_dir / name).resolve()
    if not target.is_relative_to(output_dir):
        raise ValueError("Каталог сцены должен находиться внутри галереи")
    existing = next((entry for entry in entries if entry["scene"] == name), None)
    if existing is not None:
        if (
            existing.get("purpose") == "demo"
            and existing.get("source_fingerprint") == fingerprint
            and target.is_dir()
            and _run_fingerprint(target) == fingerprint
        ):
            _write_gallery(output_dir, entries, series_dir, design)
            return {
                "viewer": str(output_dir / "index.html"),
                "scenes": entries,
                "added": False,
                "rendered": False,
                "scene_directory": str(target),
            }
        raise ValueError(f"Сцена {name} уже представлена другим запуском; выберите новое имя")
    if target == run_dir or run_dir.is_relative_to(target):
        raise ValueError("Исходный запуск должен находиться отдельно от новой сцены галереи")
    if target.exists() and any(target.iterdir()):
        origin_path = target / "origin.json"
        previous = (
            json.loads(origin_path.read_text(encoding="utf-8")) if origin_path.is_file() else {}
        )
        if previous.get("purpose") != "demo" or previous.get("source_fingerprint") != fingerprint:
            raise ValueError("Каталог новой сцены уже содержит другие файлы")
    scene = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")) or {}
    if not isinstance(scene, dict):
        raise ValueError("Описание сцены должно быть словарём YAML")
    origin = {
        "scene": name,
        "seed": seed,
        "source_run": str(run_dir),
        "source_fingerprint": fingerprint,
        "purpose": "demo",
        "selection": "Явно переданный готовый запуск; seed сохранён без повторного выбора",
        "animation": "Сбалансированный сохранённый B-сплайн после NSGA-II",
    }
    for key in ("title", "description", "notes"):
        if scene.get(key):
            origin[key] = scene[key]
    output_dir.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=True, exist_ok=True)
    for filename in _RESULT_FILES:
        shutil.copyfile(run_dir / filename, target / filename)
    write_json(target / "origin.json", origin)
    print(
        f"Дополнительная демонстрация: {name}, seed={seed}, ракурсы={', '.join(views)}", flush=True
    )
    replay_results(target, views=views, frame_count=frame_count)
    entries.append(origin)
    _write_gallery(output_dir, entries, series_dir, design)
    updated = {**manifest, "scenes": entries}
    if design is not None:
        updated["series_design"] = design
    write_json(manifest_path, updated)
    return {
        "viewer": str(output_dir / "index.html"),
        "scenes": entries,
        "added": True,
        "rendered": True,
        "scene_directory": str(target),
    }


def add_showcase_statistics(series_dir: Path, output_dir: Path) -> dict:
    """Добавить снимок парной серии одной сцены, сохранив демонстрационные траектории.

    Сводки и рисунок копируются без пересчёта статистики. Численные запуски
    остаются в исходной серии, связанной со снимком хешем и абсолютным путём.
    """
    series_dir, output_dir = Path(series_dir).resolve(), Path(output_dir).resolve()
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for filename in _STATISTICS_FILES:
        if not (series_dir / filename).is_file():
            raise ValueError(f"В серии отсутствует файл {filename}")
    summary = json.loads((series_dir / "summary.json").read_text(encoding="utf-8"))
    names, seeds = summary.get("scenes", []), summary.get("seeds", [])
    if len(names) != 1:
        raise ValueError("Для карточки нужна отдельная серия одной сцены")
    name = names[0]
    if not any(entry["scene"] == name for entry in manifest["scenes"]):
        raise ValueError(f"Сначала добавьте демонстрацию сцены {name} в галерею")
    if (
        len(seeds) < 2
        or any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Для статистики нужны разные неотрицательные seed, не менее двух")
    with (series_dir / "runs.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = {(name, str(seed), method) for seed in seeds for method in ("baseline", "full")}
    actual = [(row["scene"], row["seed"], row["method"]) for row in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Таблица запусков не содержит ровно два метода для каждого seed")
    methods = summary.get("methods", [])
    if len(methods) != 2 or {item["method"] for item in methods} != {"baseline", "full"}:
        raise ValueError("В сводке должны присутствовать оба сравниваемых метода")
    for item in methods:
        succeeded = sum(
            row["method"] == item["method"] and row["success"] == "True" for row in rows
        )
        if (
            item["scene"] != name
            or item["attempted"] != len(seeds)
            or item["succeeded"] != succeeded
            or item["failed"] != len(seeds) - succeeded
        ):
            raise ValueError("Успешность в сводке не соответствует таблице запусков")
    digest = hashlib.sha256()
    for filename in _STATISTICS_FILES:
        digest.update(filename.encode("utf-8") + b"\0")
        digest.update((series_dir / filename).read_bytes())
    target = (output_dir / name / "statistics").resolve()
    if not target.is_relative_to(output_dir) or target == series_dir:
        raise ValueError("Серия должна находиться отдельно от снимка статистики в галерее")
    origin = {
        "source_series": str(series_dir),
        "source_fingerprint": digest.hexdigest(),
        "directory": target.relative_to(output_dir).as_posix(),
        "scenes": names,
        "seeds": seeds,
        "pairs": len(seeds),
    }
    target.mkdir(parents=True, exist_ok=True)
    for filename in _STATISTICS_FILES:
        shutil.copyfile(series_dir / filename, target / filename)
    write_json(target / "origin.json", origin)
    entries = [
        {**entry, "statistics": origin} if entry["scene"] == name else entry
        for entry in manifest["scenes"]
    ]
    source_series = manifest.get("source_series")
    _write_gallery(
        output_dir,
        entries,
        Path(source_series) if source_series else None,
        manifest.get("series_design"),
    )
    write_json(manifest_path, {**manifest, "scenes": entries})
    return {"viewer": str(output_dir / "index.html"), "statistics": origin}


def _statistics_html(output_dir: Path, entry: dict) -> str:
    """Показать успешность, критерии, парные изменения и отказы рядом со сценой."""
    origin = entry.get("statistics")
    if not origin:
        return ""
    directory = output_dir / origin["directory"]
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))

    def number(value):
        return "—" if value is None else f"{value:.4g}"

    def table(headers, rows):
        return (
            '<div class="table"><table><thead><tr>'
            + "".join(f"<th>{html.escape(str(cell))}</th>" for cell in headers)
            + "</tr></thead><tbody>"
            + "".join(
                "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>"
                for row in rows
            )
            + "</tbody></table></div>"
        )

    metrics = ("J_tau", "J_jerk", "T", "runtime_numerical_s", "hypervolume")
    methods = {item["method"]: item for item in summary["methods"]}
    rows = []
    for key, label in (("baseline", "Без NSGA-II"), ("full", "Полный пайплайн")):
        item = methods[key]
        rows.append(
            [label, f"{item['succeeded']}/{item['attempted']}"]
            + [number(item["metrics_successful_runs"][metric]["median"]) for metric in metrics]
        )
    strategy_labels = {
        "balanced": "Сбалансированный",
        "min_time": "Минимум времени",
        "min_effort": "Минимум усилий",
        "min_jerk": "Минимум рывка",
    }
    changes = []
    for key, label in strategy_labels.items():
        item = next(
            item
            for item in summary["paired"]
            if item["scene"] == entry["scene"] and item["selection"] == key
        )
        cells = [label, item["paired_successes"]]
        for metric in metrics[:3]:
            stats = item["changes_full_minus_baseline"][metric]["percent"]
            interval = stats["median_bootstrap_95_ci"]
            cells.append(
                number(stats["median"])
                + (
                    f" [{number(interval[0])}; {number(interval[1])}]"
                    if interval is not None
                    else " [нет оценки ДИ]"
                )
            )
        changes.append(cells)
    with (directory / "runs.csv").open(encoding="utf-8-sig", newline="") as stream:
        failures = [row for row in csv.DictReader(stream) if row["success"] != "True"]
    failure_rows = [
        [
            row["seed"],
            "Без NSGA-II" if row["method"] == "baseline" else "Полный пайплайн",
            row["failure_stage"],
            row["reason"],
        ]
        for row in failures
    ]
    options = summary["options"]
    href = html.escape(origin["directory"], quote=True)
    seed_text = ", ".join(map(str, summary["seeds"]))
    return (
        f'<div id="statistics-{html.escape(entry["scene"], quote=True)}">'
        f"<h3>Статистика по {len(summary['seeds'])} seed: с NSGA-II и без него</h3>"
        f"<p>Seed: {html.escape(seed_text)}. RRT*: {options['rrt_iterations']} итераций; "
        f"NSGA-II: популяция {options['population']}, {options['generations']} поколений. "
        "Один общий путь RRT* и базовый B-сплайн в каждой паре. "
        f"Анимация сцены относится к seed {entry['seed']}.</p>"
        "<p>Ниже медианы по успешным запускам. Полный метод представлен "
        "сбалансированным решением; HV учитывает весь доступный фронт и базовый сплайн. "
        "Длительность движения T и время расчёта — разные величины. "
        f"Расчёт выполнялся в {summary['protocol']['workers']} процессах.</p>"
        + table(
            ["Метод", "Успех", "Jτ, (Н·м)²·с", "Рывок, рад²/с⁵", "T, с", "Расчёт, с", "HV"],
            rows,
        )
        + f'<img src="{href}/statistics.png" loading="lazy" '
        'alt="Распределения критериев и успешность парных запусков сцены">'
        "<h4>Парные изменения относительно варианта без NSGA-II</h4>"
        "<p>Медиана изменения в процентах [95% бутстреп-ДИ]; отрицательное значение "
        "означает уменьшение. Учитываются только пары, где успешны оба метода. "
        "Каждый представитель выбирается отдельно; улучшение одного критерия "
        "может сопровождаться ухудшением других.</p>"
        + table(
            ["Представитель", "Пар", "ΔJτ, % [95% ДИ]", "Δрывка, % [95% ДИ]", "ΔT, % [95% ДИ]"],
            changes,
        )
        + (
            "<h4>Отказы</h4>" + table(["Seed", "Метод", "Этап", "Причина"], failure_rows)
            if failures
            else "<p>Отказов в этой серии нет.</p>"
        )
        + f'<p><a href="{href}/index.html">Подробная статистика и ограничения</a> · '
        f'<a href="{href}/runs.csv">Все попытки CSV</a> · '
        f'<a href="{href}/summary.json">Сводка JSON</a></p></div>'
    )


def _write_gallery(
    output_dir: Path, entries: list[dict], series_dir: Path | None, design: dict | None = None
) -> None:
    """Собрать обзор сцен и единую таблицу четырёх представителей каждого фронта."""
    statistics_link = (
        Path(os.path.relpath(series_dir / "index.html", output_dir)).as_posix()
        if series_dir is not None
        else None
    )
    sections = []
    combined = []
    for entry in entries:
        name = entry["scene"]
        variants_file = output_dir / name / "variants.csv"
        table = ""
        if variants_file.exists():
            with variants_file.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            combined.extend(
                {
                    "scene": name,
                    "seed": entry["seed"],
                    "purpose": entry.get("purpose", "series_example"),
                    **row,
                }
                for row in rows
            )
        variants_json = output_dir / name / "variants.json"
        if variants_json.exists():
            records = json.loads(variants_json.read_text(encoding="utf-8"))["variants"]
            table = variants_html(records)
        safe = html.escape(name, quote=True)
        descriptions = [entry[key] for key in ("title", "description", "notes") if entry.get(key)]
        description = " ".join(
            "; ".join(map(str, value)) if isinstance(value, list) else str(value)
            for value in descriptions
        )
        sections.append(
            f"<section><h2>{safe} · seed {entry['seed']}</h2>"
            + (f"<p>{html.escape(description)}</p>" if description else "")
            + f'<p><a href="{safe}/view.html">Открыть сцену, анимации и параметры</a> · '
            f'<a href="{safe}/interactive.html">Вращаемая 3D-сцена</a> · '
            f'<a href="{safe}/variants.csv">Таблица CSV</a></p>'
            f'<a href="{safe}/view.html"><img src="{safe}/comparison.png" '
            f'alt="Четыре траектории, RRT* и базовый сплайн: {safe}"></a>'
            + table
            + f'<figure id="pareto-{safe}"><h3>Фронт Парето: {safe}</h3>'
            f'<a href="{safe}/pareto.png" aria-label="Увеличить фронт Парето: {safe}">'
            f'<img src="{safe}/pareto.png" loading="lazy" '
            f'alt="Фронт Парето и три попарные проекции критериев: {safe}"></a>'
            "<figcaption>Пространственный фронт и три попарные проекции: усилия, рывок и время. "
            "Четыре варианта подписаны и выделены цветами траекторий: синий — минимум времени, "
            "зелёный — минимум усилий, фиолетовый — минимум рывка, оранжевый — сбалансированный. "
            "Совпадающие решения показаны вложенными контурами без смещения точек. "
            f"Показаны допустимые решения для seed {entry['seed']}.</figcaption></figure>"
            + _statistics_html(output_dir, entry)
            + "</section>"
        )
    if combined:
        with (output_dir / "variants.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            fields = list(dict.fromkeys(key for row in combined for key in row))
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(combined)
    series_description = ""
    if design is not None:
        series_description = (
            f"Исходная статистическая серия: {len(design['scenes'])} сцен × "
            f"{len(design['seeds'])} seed = {design['pairs']} пар. "
        )
    if statistics_link is not None:
        series_description += (
            f'<a href="{html.escape(statistics_link, quote=True)}">Открыть статистику серии</a>. '
        )
    page = (
        '<!doctype html><html lang="ru"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Галерея сцен: движение манипулятора 6R</title><style>"
        "body{font:16px/1.5 system-ui,sans-serif;max-width:1200px;margin:auto;padding:24px;"
        "background:#f3f6fa;color:#203448}section{background:white;padding:24px;margin:24px 0;"
        "border-radius:12px}img{width:100%;height:auto}a{color:#225ba8}.table{overflow:auto}"
        "figure{margin:24px 0}figcaption{color:#526378;font-size:14px}"
        "table{border-collapse:collapse;font-size:13px}th,td{padding:6px 10px;text-align:right;"
        "border-bottom:1px solid #ddd}</style><h1>Галерея сцен: движение манипулятора 6R</h1>"
        "<p>Для каждой сцены показаны минимальное время, минимальные затраты усилий, "
        "минимальный рывок (максимальная плавность) и сбалансированный вариант. "
        "Это представители найденного приближённого фронта. Совпадение номеров решений "
        "означает совпадение траекторий. Анимации воспроизводят сбалансированный B-сплайн.</p>"
        f"<p>Сцен в галерее: {len(entries)}. "
        + series_description
        + f'<a href="variants.csv">CSV всех представителей: {len(combined)} строк</a>.</p>'
        + "".join(sections)
        + "</html>"
    )
    (output_dir / "index.html").write_text(page, encoding="utf-8")
