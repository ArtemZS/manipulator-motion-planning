"""Парные серии экспериментов с сохранением отказов и общей нормировкой критериев."""

import csv
import hashlib
import html
import json
import multiprocessing
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from importlib import metadata
from pathlib import Path

import numpy as np
from pymoo.indicators.hv import HV

from .pipeline import PipelineOptions, run_pipeline, write_json

PROTOCOL_VERSION = 1
OBJECTIVES = ("J_tau", "J_jerk", "T")
METHODS = ("baseline", "full")
METHOD_LABELS = {"baseline": "RRT* → B-сплайн", "full": "RRT* → B-сплайн → NSGA-II"}
BOOTSTRAP_SAMPLES = 4000


def _source_hash() -> str:
    """Учесть исходники и ресурсы модели, чтобы не возобновлять устаревший расчёт."""
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".xml", ".urdf", ".yaml", ".yml"}:
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _environment() -> dict:
    """Зафиксировать версии библиотек, влияющих на численный результат."""
    return {
        "python": platform.python_version(),
        "packages": {
            name: metadata.version(name) for name in ("numpy", "scipy", "mujoco", "pymoo")
        },
    }


def _fingerprint(
    scene: dict,
    options: PipelineOptions,
    source_hash: str,
    environment: dict,
    workers: int = 1,
) -> str:
    """Связать кеш с точной постановкой, бюджетом, кодом и окружением."""
    payload = {
        "protocol": PROTOCOL_VERSION,
        "scene": scene,
        "options": asdict(options),
        "source_hash": source_hash,
        "environment": environment,
        "workers": workers,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _valid_point(item: dict | None) -> bool:
    """Допускать только конечные критерии и подтверждённую допустимость."""
    if not isinstance(item, dict) or not item.get("validation", {}).get("feasible", False):
        return False
    point = np.asarray(item.get("objectives", []), dtype=float)
    return point.shape == (3,) and np.isfinite(point).all() and bool(np.all(point >= 0))


def _numerical_time(report: dict, method: str) -> float | None:
    """Исключить рендеринг и запись артефактов из стоимости метода."""
    timings = report.get("timings_seconds", {})
    keys = ["planning", "smoothing"]
    if method == "full":
        keys.append("optimization_and_validation")
    if any(key not in timings for key in keys):
        if report.get("status") != "failed" or report.get("failure_stage") not in {
            "planning",
            "smoothing",
            "optimization",
            "artifacts",
        }:
            return None
        available = [timings[key] for key in keys if key in timings]
        return float(sum(available)) if available else None
    return float(sum(timings[key] for key in keys))


def _failure_stage(report: dict) -> str:
    """Сохранить явный этап; для старого отчёта вывести его из причины отказа."""
    if report.get("failure_stage"):
        return str(report["failure_stage"])
    reason = report.get("reason", "")
    if "RRT" in reason:
        return "planning"
    if "сплайн" in reason.lower():
        return "smoothing"
    if _valid_point(report.get("baseline")):
        return "optimization_or_artifacts"
    return "pipeline"


def _paired_rows(scene_name: str, seed: int, report: dict, run_path: str) -> list[dict]:
    """Получить оба метода из единственного общего пути и базового сплайна."""
    baseline = report.get("baseline", {})
    solutions = report.get("solutions", [])
    selected = report.get("selected_index", 0)
    selected_solution = solutions[selected] if 0 <= selected < len(solutions) else {}
    result = []
    for method, item in (("baseline", baseline), ("full", selected_solution)):
        success = _valid_point(item) and (method == "baseline" or report.get("status") == "success")
        values = item["objectives"] if success else [None] * 3
        row = {
            "scene": scene_name,
            "seed": seed,
            "method": method,
            "success": bool(success),
            "failure_stage": "" if success else _failure_stage(report),
            "reason": "" if success else report.get("reason", "Нет допустимого результата"),
            **dict(zip(OBJECTIVES, values, strict=True)),
            "runtime_numerical_s": _numerical_time(report, method),
            "clearance_lower_bound_m": (
                item.get("validation", {}).get("collision_clearance_lower_bound")
                if success
                else None
            ),
            "front_size": (
                1 if method == "baseline" and success else len(solutions) if success else 0
            ),
            "hypervolume": None,
            "run_directory": run_path,
        }
        result.append(row)
    return result


def _front(report: dict, method: str) -> np.ndarray:
    """Учесть проверенный базовый сплайн как доступный результат полного метода."""
    items = [report.get("baseline", {})]
    if method == "full":
        items.extend(report.get("solutions", []))
    valid = [item["objectives"] for item in items if _valid_point(item)]
    return np.asarray(valid, dtype=float).reshape(-1, 3)


def _normalization_and_hypervolume(pairs: list[dict], rows: list[dict]) -> dict:
    """Использовать единую систему координат и опорную точку для каждой сцены."""
    normalization = {}
    lookup = {(row["scene"], row["seed"], row["method"]): row for row in rows}
    for scene in dict.fromkeys(pair["scene"] for pair in pairs):
        current = [pair for pair in pairs if pair["scene"] == scene]
        points = [_front(pair["report"], "full") for pair in current]
        points = [point for point in points if len(point)]
        if not points:
            normalization[scene] = None
            continue
        all_points = np.concatenate(points)
        ideal, nadir = all_points.min(axis=0), all_points.max(axis=0)
        span = nadir - ideal
        # Постоянная координата не должна приводить к делению на ноль.
        span = np.where(span > 1e-12, span, np.maximum(np.abs(ideal), 1.0))
        reference = np.full(3, 1.1)
        normalization[scene] = {
            "ideal": ideal.tolist(),
            "nadir": nadir.tolist(),
            "scale": span.tolist(),
            "reference_normalized": reference.tolist(),
            "source": "Объединение допустимых результатов всех seed и обоих методов данной серии",
        }
        indicator = HV(ref_point=reference)
        for pair in current:
            for method in METHODS:
                row = lookup[scene, pair["seed"], method]
                if row["success"]:
                    front = _front(pair["report"], method)
                    row["hypervolume"] = float(indicator((front - ideal) / span))
    return normalization


def _distribution(values: list[float]) -> dict:
    """Описать выборку, включая пустую, без подмены отсутствующего результата нулём."""
    array = np.asarray([value for value in values if value is not None], dtype=float)
    if not len(array):
        return {"n": 0, "median": None, "q25": None, "q75": None, "iqr": None}
    q25, median, q75 = np.percentile(array, [25, 50, 75])
    return {
        "n": len(array),
        "median": float(median),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
    }


def _paired_change(before: list[float], after: list[float], seed: int = 1937) -> dict:
    """Оценить парное изменение и бутстреп-интервал медианы по seed."""
    before_array, after_array = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    absolute = after_array - before_array
    nonzero = np.abs(before_array) > 1e-12
    relative = absolute[nonzero] / before_array[nonzero] * 100
    result = {
        "absolute": _distribution(absolute.tolist()),
        "percent": _distribution(relative.tolist()),
    }
    result["percent"]["undefined_zero_baseline"] = int(np.count_nonzero(~nonzero))
    # Один seed не позволяет оценить межзапусковую неопределённость.
    for label, values in (("absolute", absolute), ("percent", relative)):
        interval = None
        if len(values) >= 2:
            rng = np.random.default_rng(seed)
            samples = rng.choice(values, size=(BOOTSTRAP_SAMPLES, len(values)), replace=True)
            interval = np.percentile(np.median(samples, axis=1), [2.5, 97.5]).tolist()
        result[label]["median_bootstrap_95_ci"] = interval
    return result


def summarize_experiments(pairs: list[dict], rows: list[dict]) -> dict:
    """Посчитать успешность по всем запускам и сравнения по успешно завершённым парам."""
    normalization = _normalization_and_hypervolume(pairs, rows)
    summaries, comparisons = [], []
    for scene in dict.fromkeys(pair["scene"] for pair in pairs):
        for method in METHODS:
            current = [row for row in rows if row["scene"] == scene and row["method"] == method]
            successful = [row for row in current if row["success"]]
            summaries.append(
                {
                    "scene": scene,
                    "method": method,
                    "attempted": len(current),
                    "succeeded": len(successful),
                    "failed": len(current) - len(successful),
                    "success_rate": len(successful) / len(current) if current else None,
                    "runtime_all_attempts": _distribution(
                        [row["runtime_numerical_s"] for row in current]
                    ),
                    "metrics_successful_runs": {
                        key: _distribution([row[key] for row in successful])
                        for key in (
                            *OBJECTIVES,
                            "runtime_numerical_s",
                            "clearance_lower_bound_m",
                            "hypervolume",
                        )
                    },
                }
            )
        matched = [
            pair
            for pair in pairs
            if pair["scene"] == scene
            and all(
                row["success"]
                for row in rows
                if row["scene"] == scene and row["seed"] == pair["seed"]
            )
        ]
        for strategy, objective_index in (
            ("balanced", None),
            ("min_effort", 0),
            ("min_jerk", 1),
            ("min_time", 2),
        ):
            before, after = [], []
            for pair in matched:
                report = pair["report"]
                candidates = _front(report, "full")
                before.append(report["baseline"]["objectives"])
                after.append(
                    report["solutions"][report["selected_index"]]["objectives"]
                    if objective_index is None
                    else candidates[np.argmin(candidates[:, objective_index])]
                )
            comparisons.append(
                {
                    "scene": scene,
                    "selection": strategy,
                    "paired_successes": len(matched),
                    "changes_full_minus_baseline": {
                        objective: _paired_change(
                            [point[index] for point in before], [point[index] for point in after]
                        )
                        for index, objective in enumerate(OBJECTIVES)
                    },
                }
            )
        baseline_rows = {
            row["seed"]: row
            for row in rows
            if row["scene"] == scene and row["method"] == "baseline" and row["success"]
        }
        full_rows = {
            row["seed"]: row
            for row in rows
            if row["scene"] == scene and row["method"] == "full" and row["success"]
        }
        common = sorted(baseline_rows.keys() & full_rows.keys())
        extras = {}
        for key in ("hypervolume", "runtime_numerical_s"):
            available = [
                seed
                for seed in common
                if baseline_rows[seed][key] is not None and full_rows[seed][key] is not None
            ]
            extras[key] = _paired_change(
                [baseline_rows[seed][key] for seed in available],
                [full_rows[seed][key] for seed in available],
            )
        comparisons.append(
            {
                "scene": scene,
                "selection": "front_and_runtime",
                "paired_successes": len(common),
                "changes_full_minus_baseline": extras,
            }
        )
    return {"normalization_by_scene": normalization, "methods": summaries, "paired": comparisons}


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Записать таблицу с BOM для корректного открытия русского текста в Excel."""
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _flatten_summary(summary: dict) -> list[dict]:
    """Развернуть статистики методов в прямоугольную таблицу."""
    rows = []
    for item in summary["methods"]:
        row = {
            key: value
            for key, value in item.items()
            if key not in {"metrics_successful_runs", "runtime_all_attempts"}
        }
        row.update(
            {
                f"runtime_all_attempts_{key}": value
                for key, value in item["runtime_all_attempts"].items()
            }
        )
        for metric, values in item["metrics_successful_runs"].items():
            row.update({f"{metric}_{key}": value for key, value in values.items()})
        rows.append(row)
    return rows


def _plot_statistics(rows: list[dict], output_dir: Path) -> None:
    """Показать все результаты, медианы и отказы без зависимости от интерактивного окна."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenes = list(dict.fromkeys(row["scene"] for row in rows))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    metrics = [
        ("J_tau", "Усилия Jτ, (Н·м)²·с"),
        ("J_jerk", "Рывок, рад²/с⁵"),
        ("T", "Длительность движения, с"),
        ("runtime_numerical_s", "Расчёт успешных запусков, с"),
        ("hypervolume", "Гиперобъём (общая нормировка сцены)"),
    ]
    colors = ("#3975b4", "#e58b2b")
    for axis, (metric, label) in zip(axes.flat, metrics, strict=False):
        for scene_index, scene in enumerate(scenes):
            for method_index, method in enumerate(METHODS):
                values = [
                    row[metric]
                    for row in rows
                    if row["scene"] == scene
                    and row["method"] == method
                    and row["success"]
                    and row[metric] is not None
                ]
                if values:
                    x = scene_index + (-0.18 if method_index == 0 else 0.18)
                    axis.boxplot(
                        values,
                        positions=[x],
                        widths=0.27,
                        showfliers=False,
                        patch_artist=True,
                        boxprops={"facecolor": colors[method_index], "alpha": 0.5},
                        medianprops={"color": "black"},
                    )
                    offsets = np.linspace(-0.07, 0.07, len(values))
                    axis.scatter(x + offsets, values, s=13, color=colors[method_index], zorder=3)
        axis.set_xticks(range(len(scenes)), scenes, rotation=15, ha="right")
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.25)
        if metric in {"J_tau", "J_jerk", "runtime_numerical_s"}:
            axis.set_yscale("symlog", linthresh=1e-4)
    axis = axes.flat[-1]
    for method_index, method in enumerate(METHODS):
        rates = []
        for scene in scenes:
            current = [row for row in rows if row["scene"] == scene and row["method"] == method]
            rates.append(sum(row["success"] for row in current) / len(current) * 100)
        axis.bar(
            np.arange(len(scenes)) + (-0.18 if method_index == 0 else 0.18),
            rates,
            width=0.34,
            color=colors[method_index],
            label=METHOD_LABELS[method],
        )
    axis.set_xticks(range(len(scenes)), scenes, rotation=15, ha="right")
    axis.set_ylim(0, 110)
    axis.set_title("Успешность, % всех запланированных запусков")
    axis.legend(fontsize=8, loc="upper right")
    fig.suptitle("Парное сравнение: общий путь RRT* и общий базовый B-сплайн", fontsize=14)
    fig.savefig(output_dir / "statistics.png", dpi=160)
    plt.close(fig)


def _number(value: float | None, digits: int = 3) -> str:
    """Отобразить отсутствующую оценку явно."""
    return "—" if value is None else f"{value:.{digits}g}"


def _create_report(summary: dict, rows: list[dict], output_dir: Path) -> None:
    """Сформировать русскоязычный отчёт с областью применимости результатов."""
    _plot_statistics(rows, output_dir)
    lines = [
        "# Парные эксперименты пространственного манипулятора 6R",
        "",
        "Оба метода используют один и тот же путь RRT* и один базовый B-сплайн для каждой "
        "пары (сцена, seed). NSGA-II добавляется после получения допустимой базы. "
        "Время расчёта включает планирование, сглаживание и, для полного метода, "
        "оптимизацию с проверкой. Рендеринг и запись файлов исключены.",
        "",
        "| Сцена | Метод | Успех | Jτ, медиана | Рывок, медиана | T, с | Расчёт, с | HV |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    table_rows = []
    for item in summary["methods"]:
        metrics = item["metrics_successful_runs"]
        cells = [
            item["scene"],
            METHOD_LABELS[item["method"]],
            f"{item['succeeded']}/{item['attempted']}",
            *[
                _number(metrics[key]["median"])
                for key in (*OBJECTIVES, "runtime_numerical_s", "hypervolume")
            ],
        ]
        table_rows.append(cells)
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Парные изменения выбранного компромисса",
            "",
            "Отрицательное изменение критерия означает уменьшение. Интервал — "
            "95% процентильный бутстреп-интервал медианы по seed. Все оценки ниже "
            "условны на успешном завершении обоих методов.",
            "",
            "| Сцена | Пар | ΔJτ, % [95% ДИ] | Δрывка, % [95% ДИ] | ΔT, % [95% ДИ] |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    paired_rows = []
    for comparison in summary["paired"]:
        if comparison["selection"] != "balanced":
            continue
        cells = [comparison["scene"], str(comparison["paired_successes"])]
        for objective in OBJECTIVES:
            stats = comparison["changes_full_minus_baseline"][objective]["percent"]
            interval = stats["median_bootstrap_95_ci"]
            cells.append(
                _number(stats["median"])
                + (
                    f" [{_number(interval[0])}; {_number(interval[1])}]"
                    if interval
                    else " [нет оценки]"
                )
            )
        paired_rows.append(cells)
        lines.append("| " + " | ".join(cells) + " |")
    caveats = [
        "Серия использует конечные бюджеты и выбранный набор seed. Это приближённый фронт, "
        "а не доказательство глобальной Парето-оптимальности или превосходства на всех задачах.",
        "Доли успеха включают все запланированные запуски; причины отказов сохранены в runs.csv "
        "и отдельных отчётах. Медианы и парные изменения считаются только для доступных "
        "допустимых решений.",
        "Компромисс выбирается по нормированному расстоянию до идеала внутри фронта "
        "одного запуска; "
        "он может улучшить одни критерии и ухудшить другие. В summary.json отдельно приведены "
        "экстремальные решения по каждому из трёх критериев.",
        "Гиперобъём использует общие для всей сцены ideal/nadir и опорную точку (1.1, 1.1, 1.1). "
        "Эти границы зависят от состава серии; HV разных сцен и независимо построенных серий "
        "не сравниваются напрямую. В доступное множество полного метода включён "
        "проверенный базовый сплайн.",
        "Доверительный интервал не оценивается для единственного seed. При малой выборке он "
        "неустойчив. Нулевой базовый критерий исключает относительное изменение; "
        "абсолютное сохраняется.",
        "Моменты проверяются на временной сетке; это не непрерывная гарантия "
        "динамической допустимости. "
        "Модель учебная; Jτ характеризует квадраты моментов и не является электрической энергией.",
        "Время расчёта в таблице и на графике условно на успехе метода. В summary.json "
        "и summary.csv отдельно приведено runtime_all_attempts с затратами всех попыток, "
        "включая отказы с доступными измерениями.",
        "Возобновление использует только совпадающие хеши сцены, параметров, исходников, ресурсов "
        "версий библиотек и числа процессов. Уже зарегистрированные отказы также сохраняются; "
        "для повтора нужен resume=False.",
        "При параллельном запуске время измеряется под нагрузкой нескольких процессов. "
        "Стоимость подготовки пары общая; это не независимый замер ускорения двух "
        "программ на свободной машине.",
    ]
    lines.extend(
        [
            "",
            "## Интерпретация и ограничения",
            "",
            *[f"- {text}" for text in caveats],
            "",
            "![Статистика](statistics.png)",
            "",
            "Подробные данные: [runs.csv](runs.csv), [summary.csv](summary.csv), "
            "[summary.json](summary.json).",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    headers = ["Сцена", "Метод", "Успех", "Jτ", "Рывок", "T, с", "Расчёт, с", "HV"]

    def table(labels, values):
        return (
            "<table><thead><tr>"
            + "".join(f"<th>{html.escape(x)}</th>" for x in labels)
            + (
                "</tr></thead><tbody>"
                + "".join(
                    "<tr>" + "".join(f"<td>{html.escape(x)}</td>" for x in row) + "</tr>"
                    for row in values
                )
                + "</tbody></table>"
            )
        )

    page = (
        '<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" '
        'content="width=device-width,initial-scale=1"><title>Парные эксперименты 6R</title>'
        "<style>body{font:16px/1.5 system-ui;max-width:1250px;margin:30px auto;padding:0 20px;"
        "color:#182a3c}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px;"
        "border:1px solid #ccd7df;text-align:left}th{background:#eaf0f4}img{max-width:100%}"
        "li{margin:9px 0}a{color:#216ca8}</style><h1>Парные эксперименты 6R</h1><p>"
        + html.escape(lines[2])
        + "</p>"
        + table(headers, table_rows)
        + '<img src="statistics.png" alt="Статистика по всем сценам"><h2>Изменения компромисса</h2>'
        + table(
            ["Сцена", "Пар", "ΔJτ, % [95% ДИ]", "Δрывка, % [95% ДИ]", "ΔT, % [95% ДИ]"], paired_rows
        )
        + "<h2>Интерпретация и ограничения</h2><ul>"
        + "".join(f"<li>{html.escape(text)}</li>" for text in caveats)
        + '</ul><p><a href="runs.csv">Все запуски CSV</a> · <a href="summary.csv">Сводка CSV</a> · '
        '<a href="summary.json">Полная статистика JSON</a> · '
        '<a href="REPORT.md">Отчёт Markdown</a></p></html>'
    )
    (output_dir / "index.html").write_text(page, encoding="utf-8")


def _run_pair(request: tuple) -> dict:
    """Рассчитать одну независимую пару в отдельном процессе без передачи модели MuJoCo."""
    name, scene, seed, run_dir, run_options, fingerprint, resume = request
    run_dir.mkdir(parents=True, exist_ok=True)
    pair_file = run_dir / "pair.json"
    if resume and pair_file.exists():
        try:
            cached = json.loads(pair_file.read_text(encoding="utf-8"))
            if (
                isinstance(cached, dict)
                and cached.get("fingerprint") == fingerprint
                and cached.get("completed")
                and isinstance(cached.get("report"), dict)
            ):
                return cached
        except OSError, ValueError:
            pass
    print(f"Эксперимент: {name}, seed={seed}", flush=True)
    # Старый частичный результат не должен пережить повтор с другими параметрами.
    write_json(run_dir / "report.json", {"status": "running", "options": asdict(run_options)})
    try:
        report = run_pipeline(scene, run_dir, run_options)
    except (RuntimeError, ValueError, OSError, FloatingPointError) as exc:
        report = {"status": "failed", "reason": str(exc), "options": asdict(run_options)}
        report_file = run_dir / "report.json"
        if report_file.exists():
            try:
                partial = json.loads(report_file.read_text(encoding="utf-8"))
                # Неполный отчёт текущего запуска может сохранить допустимую базу.
                if isinstance(partial, dict) and partial.get("status") in {"failed", "running"}:
                    report.update(partial)
            except OSError, ValueError:
                pass
        report.update(status="failed", reason=str(exc))
        report.setdefault("failure_stage", _failure_stage(report))
        write_json(report_file, report)
    pair = {
        "protocol_version": PROTOCOL_VERSION,
        "completed": True,
        "fingerprint": fingerprint,
        "scene": name,
        "seed": seed,
        "report": report,
    }
    write_json(pair_file, pair)
    return pair


def run_experiments(
    scene_names: list[str],
    seeds: list[int],
    output_dir: Path,
    options: PipelineOptions,
    resume: bool = True,
    workers: int = 1,
) -> dict:
    """Выполнить парную серию, сохранить каждый отказ и собрать воспроизводимую статистику."""
    from .scenes import load_named_scene

    options.validate()
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("Число рабочих процессов должно быть положительным целым")
    if not scene_names or not seeds:
        raise ValueError("Нужны хотя бы одна сцена и один seed")
    if len(set(scene_names)) != len(scene_names) or len(set(seeds)) != len(seeds):
        raise ValueError("Повторяющиеся сцены и seed запрещены: они искажают размер выборки")
    if any(not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("Seed должен быть неотрицательным целым числом")
    scenes = {name: load_named_scene(name) for name in scene_names}
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_hash, environment = _source_hash(), _environment()
    requests = []
    for name, scene in scenes.items():
        for seed in seeds:
            run_options = replace(options, seed=seed, plots=False, method="full")
            fingerprint = _fingerprint(scene, run_options, source_hash, environment, workers)
            run_dir = output_dir / "runs" / name / f"seed_{seed}"
            requests.append((name, scene, seed, run_dir, run_options, fingerprint, resume))
    if workers == 1:
        pairs = [_run_pair(request) for request in requests]
    else:
        ordered = {}
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as executor:
            futures = {
                executor.submit(_run_pair, request): index for index, request in enumerate(requests)
            }
            for future in as_completed(futures):
                pair = future.result()
                ordered[futures[future]] = pair
                print(
                    f"Завершено {len(ordered)}/{len(requests)}: "
                    f"{pair['scene']}, seed={pair['seed']}",
                    flush=True,
                )
        pairs = [ordered[index] for index in range(len(requests))]
    rows = []
    for pair in pairs:
        relative = f"runs/{pair['scene']}/seed_{pair['seed']}"
        rows.extend(_paired_rows(pair["scene"], pair["seed"], pair["report"], relative))
    summary = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "scenes": scene_names,
        "seeds": seeds,
        "options": asdict(replace(options, plots=False, method="full")),
        "source_hash": source_hash,
        "environment": environment,
        "protocol": {
            "pairing": "Один запуск RRT* и сглаживания на пару; база извлекается до NSGA-II",
            "runtime": "planning + smoothing (+ optimization_and_validation); без артефактов",
            "workers": workers,
            "process_start_method": "spawn" if workers > 1 else "sequential",
            "runtime_interpretation": (
                "Время измерено при указанном числе одновременных процессов. "
                "База использует общую стоимость подготовки пары. Это не независимый "
                "замер ускорения двух программ на свободной машине."
            ),
            "balanced_selection": "Индекс компромисса, сохранённый полным пайплайном",
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": 1937,
            "failure_policy": "Все отказы входят в знаменатель доли успеха и сохраняются",
        },
        **summarize_experiments(pairs, rows),
        "artifacts": {
            "runs": "runs.csv",
            "summary": "summary.csv",
            "report": "index.html",
            "markdown": "REPORT.md",
            "statistics": "statistics.png",
        },
    }
    write_json(output_dir / "summary.json", summary)
    _write_csv(output_dir / "runs.csv", rows)
    _write_csv(output_dir / "summary.csv", _flatten_summary(summary))
    _create_report(summary, rows, output_dir)
    return summary
