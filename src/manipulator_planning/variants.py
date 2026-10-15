"""Четыре представителя Парето и общие данные для наглядного сравнения."""

from __future__ import annotations

import csv
import json
from html import escape
from pathlib import Path
from typing import Any

import numpy as np

VARIANT_STYLES = (
    ("min_time", "Минимальное время", "#2563eb"),
    ("min_effort", "Минимальные усилия", "#15803d"),
    ("min_jerk", "Максимальная плавность", "#9333ea"),
    ("balanced", "Сбалансированное решение", "#e47714"),
)


def obstacle_label(obstacle: dict[str, Any], index: int) -> str:
    """Сохранить явную подпись, заменив служебные имена короткими обозначениями."""
    if obstacle.get("label"):
        return str(obstacle["label"])
    name = str(obstacle.get("name", ""))
    if name and "_" not in name:
        return name
    return f"{'Сфера' if obstacle['kind'] == 'sphere' else 'Блок'} {index + 1}"


def select_variants(objectives: np.ndarray, selected_index: int) -> list[dict[str, Any]]:
    """Выбрать представителей, сохранив повторения настоящих номеров решений.

    Максимальная плавность соответствует минимуму интеграла квадрата рывка.
    Сбалансированное решение уже выбрано пайплайном; здесь оно не заменяется.
    """
    values = np.asarray(objectives, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Для выбора представителей нужен конечный непустой массив N×3")
    if not 0 <= selected_index < len(values):
        raise ValueError("Номер сбалансированного решения выходит за границы массива")
    indices = [int(np.argmin(values[:, column])) for column in (2, 0, 1)] + [int(selected_index)]
    records = []
    for (key, label, color), index in zip(VARIANT_STYLES, indices, strict=True):
        records.append(
            {
                "key": key,
                "label": label,
                "color": color,
                "solution_index": index,
                "J_tau": float(values[index, 0]),
                "J_jerk": float(values[index, 1]),
                "T": float(values[index, 2]),
                "same_solution_as": [
                    other_key
                    for (other_key, _, _), other_index in zip(VARIANT_STYLES, indices, strict=True)
                    if other_index == index and other_key != key
                ],
            }
        )
    return records


def interpolate_rrt_path(path: Any, max_joint_step: float = 0.02) -> np.ndarray:
    """Густо интерполировать каждое ребро в пространстве суставных координат.

    После прямой кинематики получается искривлённый пространственный путь
    инструмента, соответствующий движению вдоль ребра RRT*, а не хорда в 3D.
    """
    points = np.asarray(path, dtype=float)
    if points.ndim != 2 or len(points) < 2 or not points.shape[1] or not np.isfinite(points).all():
        raise ValueError("RRT-путь должен содержать не меньше двух конечных конфигураций")
    if not np.isfinite(max_joint_step) or max_joint_step <= 0:
        raise ValueError("Шаг суставной интерполяции должен быть положительным")
    interpolated = [points[:1]]
    for a, b in zip(points[:-1], points[1:], strict=True):
        steps = max(1, int(np.ceil(np.max(np.abs(b - a)) / max_joint_step)))
        parameter = np.linspace(0, 1, steps + 1)[1:]
        interpolated.append(a + parameter[:, None] * (b - a))
    return np.concatenate(interpolated)


def rrt_tool_path(robot: Any, path: Any) -> np.ndarray:
    """Прямая кинематика густой суставной ломаной RRT*."""
    return np.asarray([robot.tool_position(q) for q in interpolate_rrt_path(path)])


def variants_html(records: list[dict[str, Any]]) -> str:
    """Таблица четырёх вариантов с явными совпадениями и источником анимации."""
    labels = {record["key"]: record["label"] for record in records}
    rows = []
    for record in records:
        duplicates = "; ".join(labels[key] for key in record["same_solution_as"]) or "—"
        rows.append(
            '<tr><td style="text-align:left">'
            f'<span style="color:{record["color"]};font-size:20px">●</span> '
            f"{escape(record['label'])}</td><td>{record['solution_index']}</td>"
            f"<td>{record['T']:.4g}</td><td>{record['J_tau']:.6g}</td>"
            f'<td>{record["J_jerk"]:.6g}</td><td style="text-align:left">'
            f"{escape(duplicates)}</td></tr>"
        )
    balanced = next(record for record in records if record["key"] == "balanced")
    return (
        '<section style="margin:24px 0"><h2>Четыре представителя фронта Парето</h2>'
        "<p>Максимальная плавность означает минимальное значение J<sub>jerk</sub>. "
        "Номера соответствуют сохранённому массиву решений и начинаются с нуля. "
        "При совпадении номеров показана одна и та же траектория.</p>"
        "<style>.variants-table th,.variants-table td{padding:8px 12px;text-align:right;"
        "border-bottom:1px solid #dce3eb}.variants-table th{white-space:nowrap}</style>"
        '<div style="overflow:auto"><table class="variants-table" '
        'style="border-collapse:collapse;width:100%;'
        'background:white"><thead><tr><th>Представитель</th><th>№ решения</th><th>T [с]</th>'
        "<th>Jτ [(Н·м)²·с]</th><th>Jjerk [рад²/с⁵]</th>"
        "<th>Совпадает с</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
        "<p><strong>Источник анимации: сбалансированный B-сплайн, "
        f"решение № {balanced['solution_index']}.</strong> RRT* показан отдельной "
        "пунктирной линией после густой линейной интерполяции суставных углов. "
        "Исходный B-сплайн без NSGA-II также показан отдельно.</p></section>"
    )


def save_variants(directory: Path, records: list[dict[str, Any]]) -> dict[str, Path]:
    """Сохранить четыре строки с критериями и совпадениями в CSV и JSON."""
    paths = {
        "variants_csv": directory / "variants.csv",
        "variants_json": directory / "variants.json",
    }
    paths["variants_json"].write_text(
        json.dumps(
            {
                "schema_version": 1,
                "animation_source": "balanced_bspline",
                "variants": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    fields = ["key", "label", "solution_index", "T", "J_tau", "J_jerk", "same_solution_as"]
    with paths["variants_csv"].open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {key: record[key] for key in fields}
            row["same_solution_as"] = ";".join(row["same_solution_as"])
            writer.writerow(row)
    return paths
