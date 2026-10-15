"""Воспроизводимые графики и автономный просмотр сохранённых траекторий.

Каждый график и кадр анимации вычисляется непосредственно по переданному
B-сплайну. Повторная интерполяция и сглаживание не выполняются. Графики
и подписи используют единицы СИ.
"""

from __future__ import annotations

import base64
import shutil
from html import escape
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .variants import obstacle_label, rrt_tool_path, save_variants, select_variants

if TYPE_CHECKING:
    from .trajectory import JointSplineTrajectory


_LINK_COLORS = ["#246a9a", "#3d8dc0", "#50a3a4", "#e1a73b", "#d97935", "#b85450"]
_STYLE = {"font.size": 10, "axes.labelsize": 10, "axes.titlesize": 12}
_VIEW_ANGLES = {
    "isometric": (24, -55, "Общий вид"),
    "front": (12, -90, "Спереди"),
    "side": (12, 0, "Сбоку"),
    "top": (90, -90, "Сверху"),
}


def _figure(figsize: tuple[float, float]) -> Figure:
    figure = Figure(figsize=figsize, layout="constrained")
    FigureCanvasAgg(figure)
    return figure


def _capsule_faces(a: np.ndarray, b: np.ndarray, radius: float) -> list[np.ndarray]:
    """Замкнутая сетка капсулы по фактическим координатам концов звена."""
    axis = np.asarray(b, dtype=float) - a
    length = np.linalg.norm(axis)
    if length < 1e-10:
        axis = np.array([0.0, 0.0, 1.0])
    else:
        axis /= length
    helper = np.array([1.0, 0.0, 0.0])
    if abs(axis @ helper) > 0.85:
        helper = np.array([0.0, 1.0, 0.0])
    first = np.cross(axis, helper)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    angles = np.linspace(0.0, 2.0 * np.pi, 13)
    directions = np.outer(np.cos(angles), first) + np.outer(np.sin(angles), second)
    rings = []
    # Скруглённые концы и прямое тело звена с радиусом из модели.
    for phi in (-np.pi / 2, -np.pi / 4, 0.0):
        rings.append(a + radius * np.sin(phi) * axis + radius * np.cos(phi) * directions)
    for phi in (0.0, np.pi / 4, np.pi / 2):
        rings.append(b + radius * np.sin(phi) * axis + radius * np.cos(phi) * directions)
    return [
        np.array([lower[j], lower[j + 1], upper[j + 1], upper[j]])
        for lower, upper in pairwise(rings)
        for j in range(len(angles) - 1)
    ]


def _link_radii(robot: Any, count: int) -> np.ndarray:
    radii = np.asarray(getattr(robot, "link_radii", getattr(robot, "link_radius", 0.035)))
    return np.broadcast_to(radii, (count,)).astype(float)


def _draw_robot(
    ax: Any, robot: Any, q: np.ndarray, *, alpha: float = 1.0, ghost: bool = False
) -> list[Poly3DCollection]:
    positions = np.asarray(robot.joint_positions(q))
    radii = _link_radii(robot, len(positions) - 1)
    artists = []
    for i, (a, b) in enumerate(pairwise(positions)):
        artist = Poly3DCollection(
            _capsule_faces(a, b, radii[i]),
            facecolors="#778a9b" if ghost else _LINK_COLORS[i % 6],
            edgecolors="none",
            alpha=alpha,
            zsort="average",
        )
        ax.add_collection3d(artist)
        artists.append(artist)
    return artists


def _box_faces(position: np.ndarray, half_size: np.ndarray) -> list[np.ndarray]:
    vertices = (
        position
        + np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * half_size
    )
    return [
        vertices[index]
        for index in (
            [0, 1, 3, 2],
            [4, 6, 7, 5],
            [0, 4, 5, 1],
            [2, 3, 7, 6],
            [0, 2, 6, 4],
            [1, 5, 7, 3],
        )
    ]


def _draw_obstacles(ax: Any, scene: dict[str, Any]) -> None:
    for index, obstacle in enumerate(scene.get("obstacles", [])):
        position = np.asarray(obstacle["position"], dtype=float)
        kind = obstacle["kind"]
        if kind == "box":
            ax.add_collection3d(
                Poly3DCollection(
                    _box_faces(position, np.asarray(obstacle["half_size"])),
                    facecolors="#b75b63",
                    edgecolors="#7b3d45",
                    linewidths=0.5,
                    alpha=0.32,
                )
            )
        elif kind == "sphere":
            azimuth = np.linspace(0.0, 2.0 * np.pi, 22)
            polar = np.linspace(0.0, np.pi, 13)
            radius = float(obstacle["radius"])
            ax.plot_surface(
                position[0] + radius * np.outer(np.cos(azimuth), np.sin(polar)),
                position[1] + radius * np.outer(np.sin(azimuth), np.sin(polar)),
                position[2] + radius * np.outer(np.ones_like(azimuth), np.cos(polar)),
                color="#b75b63",
                alpha=0.32,
                linewidth=0,
                shade=True,
            )
        else:
            raise ValueError(f"Неизвестная форма препятствия: {kind!r}")
        label_height = (
            float(obstacle["radius"]) if kind == "sphere" else float(obstacle["half_size"][2])
        )
        anchor = position + [0, 0, label_height]
        # Выноски разносят подписи близких препятствий в текущей проекции сцены.
        side = 1 if index % 2 == 0 else -1
        label_position = anchor + [side * 0.16, 0, 0.06 + (index % 2) * 0.04]
        ax.plot(*np.array([anchor, label_position]).T, color="#77323b", lw=0.6, alpha=0.7)
        ax.text(
            *label_position,
            obstacle_label(obstacle, index),
            fontsize=8,
            color="#77323b",
            ha="left" if side > 0 else "right",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 1.0},
        )


def _draw_base(ax: Any, robot: Any) -> None:
    """Отрисовка неподвижного основания по геометрии модели."""
    if not hasattr(robot, "model"):
        return
    base_id = robot.model.geom("base").id
    position = np.asarray(robot.model.geom_pos[base_id])
    radius, half_height = robot.model.geom_size[base_id, :2]
    angle = np.linspace(0, 2 * np.pi, 25)
    circle = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), np.zeros_like(angle)))
    lower = circle + position + [0, 0, -half_height]
    upper = circle + position + [0, 0, half_height]
    faces = [
        np.array([lower[i], lower[i + 1], upper[i + 1], upper[i]]) for i in range(len(angle) - 1)
    ] + [lower, upper]
    ax.add_collection3d(Poly3DCollection(faces, facecolors="#607285", edgecolors="none"))


def _positions(robot: Any, q: np.ndarray) -> np.ndarray:
    return np.array([robot.joint_positions(config) for config in q], dtype=float)


def _scene_bounds(positions: np.ndarray, scene: dict[str, Any]) -> tuple[np.ndarray, float]:
    """Общая кубическая область обзора включает движение, пол и препятствия."""
    points = [positions.reshape(-1, 3), np.array([[0.0, 0.0, 0.0]])]
    for obstacle in scene.get("obstacles", []):
        position = np.asarray(obstacle["position"], dtype=float)
        half_size = np.asarray(obstacle.get("half_size", [obstacle.get("radius", 0.0)] * 3))
        points.append(np.array([position - half_size, position + half_size]))
    points = np.concatenate(points)
    minimum, maximum = points.min(axis=0), points.max(axis=0)
    center = (minimum + maximum) / 2.0
    radius = max(float(np.max(maximum - minimum)) * 0.57, 0.3)
    return center, radius


def _configure_scene_axes(
    ax: Any, positions: np.ndarray, scene: dict[str, Any], view: str = "isometric"
) -> None:
    center, radius = _scene_bounds(positions, scene)
    ax.set(
        xlim=(center[0] - radius, center[0] + radius),
        ylim=(center[1] - radius, center[1] + radius),
        zlim=(center[2] - radius, center[2] + radius),
        xlabel="x [м]",
        ylabel="y [м]",
        zlabel="z [м]",
    )
    ax.set_box_aspect((1, 1, 1))
    elevation, azimuth, _ = _VIEW_ANGLES[view]
    ax.view_init(elev=elevation, azim=azimuth)
    ax.tick_params(labelsize=8)
    floor = np.array(
        [
            [center[0] - radius, center[1] - radius, 0.0],
            [center[0] + radius, center[1] - radius, 0.0],
            [center[0] + radius, center[1] + radius, 0.0],
            [center[0] - radius, center[1] + radius, 0.0],
        ]
    )
    ax.add_collection3d(Poly3DCollection([floor], facecolors="#dce3e9", alpha=0.2))


def _goal_position(
    robot: Any, scene: dict[str, Any], trajectory: JointSplineTrajectory
) -> np.ndarray:
    """Фиксированная цель вычисляется из задания, а не из текущего кадра."""
    return np.asarray(robot.tool_position(scene.get("goal", trajectory.evaluate(1.0))))


def _draw_target(ax: Any, goal: np.ndarray, start: np.ndarray) -> None:
    """Яркие неподвижные ориентиры и подпись поверх проекции сцены."""
    ax.plot(
        *start[:, None],
        markersize=8,
        marker="o",
        color="#16834b",
        markeredgecolor="white",
        markeredgewidth=0.8,
        linestyle="none",
        label="СТАРТ",
        zorder=90,
    )
    ax.plot(
        *goal[:, None],
        markersize=16,
        marker="*",
        color="#ed168f",
        markeredgecolor="#621044",
        markeredgewidth=0.9,
        linestyle="none",
        label="ЦЕЛЬ",
        zorder=100,
    )
    ax.plot(
        [goal[0], goal[0]],
        [goal[1], goal[1]],
        [0, goal[2]],
        color="#d20d7f",
        lw=1.1,
        ls=":",
        alpha=0.8,
    )
    ax.text(
        *(goal + [0, 0, 0.075]),
        "ЦЕЛЬ",
        color="#b5066d",
        fontsize=11,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "#ed168f", "alpha": 0.9},
    )
    ax.text2D(
        0.02,
        0.02,
        f"★ ЦЕЛЬ: x = {goal[0]:.3f}; y = {goal[1]:.3f}; z = {goal[2]:.3f} м",
        transform=ax.transAxes,
        fontsize=9,
        color="#b5066d",
        fontweight="bold",
        bbox={"facecolor": "#fff0f8", "edgecolor": "#ed168f", "alpha": 0.95, "pad": 4},
    )


def _scene_figure(
    robot: Any,
    scene: dict[str, Any],
    trajectory: JointSplineTrajectory,
    baseline: JointSplineTrajectory | None,
    view: str = "isometric",
) -> Figure:
    figure = _figure((10, 8))
    figure.get_layout_engine().set(rect=(0, 0.05, 1, 0.95))
    ax = figure.add_subplot(projection="3d")
    data = trajectory.sample(241)
    positions = _positions(robot, data["q"])
    goal = _goal_position(robot, scene, trajectory)
    _configure_scene_axes(ax, np.concatenate((positions.reshape(-1, 3), [goal])), scene, view)
    _draw_obstacles(ax, scene)
    _draw_base(ax, robot)
    tool = positions[:, -1]
    ax.plot(*tool.T, color="#165f9c", lw=2.3, label="Выбранный путь инструмента")
    if baseline is not None:
        baseline_tool = _positions(robot, baseline.sample(181)["q"])[:, -1]
        ax.plot(*baseline_tool.T, color="#a57724", lw=1.5, ls="--", label="Исходный сплайн")
    _draw_robot(ax, robot, data["q"][0], alpha=0.20, ghost=True)
    _draw_robot(ax, robot, scene.get("goal", data["q"][-1]), alpha=0.20, ghost=True)
    _draw_target(ax, goal, tool[0])
    _draw_robot(ax, robot, data["q"][len(data["q"]) // 2], alpha=0.95)
    ax.plot(
        *tool[len(tool) // 2, :, None],
        markersize=8,
        marker="o",
        color="#ffd23f",
        markeredgecolor="#27374c",
        markeredgewidth=1.1,
        linestyle="none",
        label="Инструмент в показанной позе",
        zorder=95,
    )
    ax.set_title(
        "Учебный манипулятор 6R: выбранная траектория\n"
        f"{_VIEW_ANGLES[view][2]}; T = {trajectory.duration:.3f} с"
    )
    ax.legend(loc="upper left", fontsize=9)
    figure.text(
        0.5,
        0.015,
        "Цвет: средняя поза; полупрозрачные звенья: старт и цель. Масштаб осей одинаковый.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    return figure


def _place_pareto_labels(ax: Any, annotations: list, points: np.ndarray, renderer: Any) -> None:
    """Разместить выноски внутри панели, сохранив видимость четырёх маркеров."""
    markers = [Bbox.from_bounds(x - 12, y - 12, 24, 24) for x, y in points]
    occupied = []
    for annotation, anchor in zip(annotations, points, strict=True):
        horizontal = 1 if anchor[0] < ax.bbox.x0 + ax.bbox.width / 2 else -1
        vertical = 1 if anchor[1] < ax.bbox.y0 + ax.bbox.height / 2 else -1
        best = None
        for sign_x in (horizontal, -horizontal):
            annotation.set_ha("left" if sign_x > 0 else "right")
            for sign_y in (vertical, -vertical):
                for distance in (18, 42, 66, 90):
                    position = (sign_x * 15, sign_y * distance)
                    annotation.set_position(position)
                    annotation.update_positions(renderer)
                    bounds = Text.get_window_extent(annotation, renderer).padded(3)
                    outside = not (
                        ax.bbox.contains(bounds.x0, bounds.y0)
                        and ax.bbox.contains(bounds.x1, bounds.y1)
                    )
                    collisions = sum(bounds.overlaps(marker) for marker in markers)
                    collisions += sum(bounds.overlaps(other) for other in occupied)
                    score = 1000 * outside + 100 * collisions + distance
                    if best is None or score < best[0]:
                        best = (score, position, annotation.get_ha(), bounds)
        _, position, alignment, bounds = best
        annotation.set_position(position)
        annotation.set_ha(alignment)
        occupied.append(bounds)


def _pareto_figure(objectives: np.ndarray, selected_index: int) -> Figure:
    """Выделить четыре настоящих представителя цветами соответствующих путей."""
    objectives = np.asarray(objectives, dtype=float)
    records = select_variants(objectives, selected_index)
    names = {
        "min_time": "Минимум времени",
        "min_effort": "Минимум усилий",
        "min_jerk": "Минимум рывка",
        "balanced": "Сбалансированный",
    }
    # Вложенные контуры показывают несколько критериев в одной точке без её сдвига.
    sizes = (320, 220, 140, 90)
    figure = _figure((14, 10))
    figure.get_layout_engine().set(rect=(0, 0.045, 1, 0.86), wspace=0.10, hspace=0.12)
    labels = (
        r"$J_\tau$ [(N m)$^2$ s]",
        r"$J_{\mathrm{jerk}}$ [rad$^2$/s$^5$]",
        "T [s]",
    )
    ax = figure.add_subplot(2, 2, 1, projection="3d", computed_zorder=False)
    ax.scatter(*objectives.T, color="#94a3b8", s=22, alpha=0.65, depthshade=False)
    for record, size in zip(records, sizes, strict=True):
        balanced = record["key"] == "balanced"
        ax.scatter(
            *objectives[record["solution_index"]],
            marker="*" if balanced else "o",
            facecolors=record["color"] if balanced else "none",
            edgecolors=record["color"],
            linewidths=2.0,
            s=size,
            label=names[record["key"]],
            depthshade=False,
            zorder=5,
        )
    ax.set(
        xlabel=r"$J_\tau$",
        ylabel=r"$J_{\mathrm{jerk}}$",
        zlabel="T [s]",
        title="Минимизация трёх критериев",
    )
    ax.tick_params(labelsize=8)
    ax.view_init(elev=23, azim=-55)
    projections = []
    for slot, (i, j) in enumerate(((0, 1), (0, 2), (1, 2)), start=2):
        ax = figure.add_subplot(2, 2, slot)
        ax.scatter(
            objectives[:, i],
            objectives[:, j],
            color="#94a3b8",
            s=22,
            alpha=0.65,
        )
        annotations, anchors = [], []
        for record, size in zip(records, sizes, strict=True):
            point = objectives[record["solution_index"], [i, j]]
            balanced = record["key"] == "balanced"
            ax.scatter(
                *point,
                marker="*" if balanced else "o",
                facecolors=record["color"] if balanced else "none",
                edgecolors=record["color"],
                linewidths=2.0,
                s=size,
                label=names[record["key"]],
                zorder=5,
            )
            # Близкие точки сохраняют координаты, разносятся только их выноски.
            annotation = ax.annotate(
                names[record["key"]],
                xy=point,
                xytext=(15, 18),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=9,
                color=record["color"],
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.5},
                arrowprops={"arrowstyle": "-", "color": record["color"], "lw": 0.8},
                zorder=6,
            )
            annotation.set_in_layout(False)
            annotations.append(annotation)
            anchors.append(point)
        ax.set(xlabel=labels[i], ylabel=labels[j])
        ax.grid(alpha=0.22)
        ax.ticklabel_format(axis="both", style="sci", scilimits=(-3, 4), useMathText=True)
        projections.append((ax, annotations, anchors))
    figure.suptitle(
        f"Приближённый фронт Парето: допустимых решений — {len(objectives)}",
        fontsize=14,
    )
    figure.legend(
        handles=[
            Line2D(
                [],
                [],
                linestyle="none",
                marker="*" if record["key"] == "balanced" else "o",
                markerfacecolor=record["color"] if record["key"] == "balanced" else "none",
                markeredgecolor=record["color"],
                markeredgewidth=2,
                markersize=10,
                label=f"{names[record['key']]} · № {record['solution_index']}",
            )
            for record in records
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=2,
        frameon=False,
        fontsize=11,
    )
    figure.text(
        0.5,
        0.012,
        "Цвета совпадают с траекториями. Серые точки — допустимые решения фронта.\n"
        "Совпадающие номера: одно решение; вложенные контуры сохраняют цвета вариантов.",
        ha="center",
        fontsize=10,
        color="#526378",
    )
    figure.canvas.draw()
    for ax, annotations, anchors in projections:
        _place_pareto_labels(
            ax, annotations, ax.transData.transform(anchors), figure.canvas.get_renderer()
        )
    return figure


def _comparison_figure(
    robot: Any,
    scene: dict[str, Any],
    trajectories: list[JointSplineTrajectory],
    records: list[dict[str, Any]],
    baseline: JointSplineTrajectory | None,
    rrt_path: Any = None,
    view: str = "isometric",
) -> Figure:
    """Общее сравнение настоящих путей инструмента четырёх представителей."""
    figure = _figure((11, 9))
    figure.get_layout_engine().set(rect=(0, 0.03, 1, 0.94))
    ax = figure.add_subplot(projection="3d")
    sampled = [
        _positions(robot, trajectories[record["solution_index"]].sample(181)["q"])
        for record in records
    ]
    selected = trajectories[
        next(record["solution_index"] for record in records if record["key"] == "balanced")
    ]
    goal = _goal_position(robot, scene, selected)
    all_points = [points.reshape(-1, 3) for points in sampled] + [goal[None, :]]
    baseline_tool = None
    if baseline is not None:
        baseline_positions = _positions(robot, baseline.sample(181)["q"])
        all_points.append(baseline_positions.reshape(-1, 3))
        baseline_tool = baseline_positions[:, -1]
    rrt_tool = None
    if rrt_path is not None:
        rrt_tool = rrt_tool_path(robot, rrt_path)
        all_points.append(rrt_tool)
    _configure_scene_axes(ax, np.concatenate(all_points), scene, view)
    _draw_obstacles(ax, scene)
    _draw_base(ax, robot)
    if rrt_tool is not None:
        ax.plot(
            *rrt_tool.T,
            color="#334155",
            lw=1.6,
            ls=":",
            label="RRT*: линейные рёбра в углах суставов",
            zorder=5,
        )
    if baseline_tool is not None:
        ax.plot(*baseline_tool.T, color="#737373", lw=2, ls="--", label="B-сплайн без NSGA-II")
    for record, positions in zip(records, sampled, strict=True):
        ax.plot(
            *positions[:, -1].T,
            color=record["color"],
            lw=2.4,
            label=f"{record['label']} · № {record['solution_index']}",
        )
    _draw_target(ax, goal, sampled[0][0, -1])
    ax.set_title("Четыре представителя Парето, исходный сплайн и RRT*\n" + _VIEW_ANGLES[view][2])
    ax.legend(loc="upper left", fontsize=8)
    duplicates = len({record["solution_index"] for record in records}) < 4
    figure.text(
        0.5,
        0.012,
        "Совпадающие номера означают один и тот же путь; линии могут накладываться."
        if duplicates
        else "Цвета соответствуют четырём строкам таблицы представителей.",
        ha="center",
        fontsize=9,
        color="#526378",
    )
    return figure


def _joint_figure(robot: Any, trajectory: JointSplineTrajectory) -> Figure:
    data = trajectory.sample(301)
    groups = [
        ("q", "Углы [rad]"),
        ("qd", "Скорость [rad/s]"),
        ("qdd", r"Ускорение [rad/s$^2$]"),
        ("qddd", r"Рывок [rad/s$^3$]"),
    ]
    if hasattr(robot, "inverse_dynamics"):
        data["tau"] = np.array(
            [
                robot.inverse_dynamics(q, qd, qdd)
                for q, qd, qdd in zip(data["q"], data["qd"], data["qdd"], strict=True)
            ]
        )
        groups.append(("tau", "Момент [N m]"))
    usage = []
    for key, attribute, label, color in (
        ("qd", "velocity_limits", "Скорость", "#246a9a"),
        ("qdd", "acceleration_limits", "Ускорение", "#d97935"),
        ("tau", "torque_limits", "Момент", "#b85450"),
    ):
        if key in data and hasattr(robot, attribute):
            ratio = np.max(np.abs(data[key]) / np.asarray(getattr(robot, attribute)), axis=1)
            usage.append((label, ratio, color))
    panel_count = len(groups) + bool(usage)
    figure = _figure((12, 2.4 * len(groups) + (1.9 if usage else 0)))
    grid = figure.add_gridspec(
        panel_count, 1, height_ratios=[1] * len(groups) + ([0.8] if usage else [])
    )
    previous = None
    for i, (key, label) in enumerate(groups, start=1):
        ax = figure.add_subplot(grid[i - 1], sharex=previous)
        previous = ax
        for joint in range(data[key].shape[1]):
            ax.plot(
                data["time"],
                data[key][:, joint],
                lw=1.4,
                color=_LINK_COLORS[joint % 6],
                label=f"Сустав {joint + 1}",
            )
        ax.set_ylabel(label)
        ax.grid(alpha=0.23)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useMathText=True)
        if i == 1:
            ax.legend(loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.37), frameon=False)
        if i == panel_count:
            ax.set_xlabel("Время [s]")
        else:
            ax.tick_params(labelbottom=False)
    if usage:
        ax = figure.add_subplot(grid[-1], sharex=previous)
        # Для каждого момента времени показан максимум отношения по всем суставам.
        for label, ratio, color in usage:
            ax.plot(data["time"], ratio, label=label, color=color, lw=1.5)
        ax.axhline(1.0, color="#77323b", ls="--", lw=1.3, label="Предел 1,0")
        ax.set(
            xlabel="Время [s]",
            ylabel="Доля предела",
            ylim=(0, max(1.08, max(float(np.max(ratio)) for _, ratio, _ in usage) * 1.08)),
        )
        ax.legend(loc="upper left", ncol=4, fontsize=9, frameon=False)
        ax.grid(alpha=0.23)
    figure.suptitle("Выбранная B-сплайновая траектория и обратная динамика", fontsize=14)
    return figure


def _save_animation(
    path: Path,
    robot: Any,
    scene: dict[str, Any],
    trajectory: JointSplineTrajectory,
    view: str = "isometric",
    frame_count: int = 55,
) -> None:
    figure = _figure((7.2, 6.2))
    figure.get_layout_engine().set(rect=(0, 0.055, 1, 0.945))
    ax = figure.add_subplot(projection="3d")
    data = trajectory.sample(frame_count)
    positions = _positions(robot, data["q"])
    goal = _goal_position(robot, scene, trajectory)
    _configure_scene_axes(ax, np.concatenate((positions.reshape(-1, 3), [goal])), scene, view)
    _draw_obstacles(ax, scene)
    _draw_base(ax, robot)
    _draw_target(ax, goal, positions[0, -1])
    ax.plot(*positions[:, -1].T, color="#7f9eb7", lw=1, ls="--", alpha=0.7)
    artists = _draw_robot(ax, robot, data["q"][0])
    radii = _link_radii(robot, len(artists))
    (trail,) = ax.plot([], [], [], color="#165f9c", lw=2)
    (tool_marker,) = ax.plot(
        [],
        [],
        [],
        marker="o",
        markersize=8,
        color="#ffd23f",
        markeredgecolor="#27374c",
        linestyle="none",
        label="Текущий инструмент",
        zorder=95,
    )
    title = ax.set_title("")
    figure.text(
        0.5,
        0.015,
        "Кадры: q(t) B-сплайна → прямая кинематика. Прямые звенья — жёсткие тела.\n"
        "Сбалансированное решение. Розовая звезда: ЦЕЛЬ; жёлтый: инструмент.",
        ha="center",
        fontsize=8,
    )

    def update(frame: int) -> tuple[Any, ...]:
        for i, artist in enumerate(artists):
            artist.set_verts(_capsule_faces(positions[frame, i], positions[frame, i + 1], radii[i]))
        tool = positions[: frame + 1, -1]
        trail.set_data_3d(*tool.T)
        tool_marker.set_data_3d(*tool[-1:, :].T)
        title.set_text(
            f"6R | {_VIEW_ANGLES[view][2]} | "
            f"t = {data['time'][frame]:.2f} / {trajectory.duration:.2f} с"
        )
        return (*artists, trail, tool_marker, title)

    fps = max(1, round(frame_count / 5.5))
    animation = FuncAnimation(
        figure, update, frames=len(positions), interval=1000 / fps, blit=False
    )
    animation.save(path, writer=PillowWriter(fps=fps), dpi=100)
    figure.clear()


def _data_uri(path: Path) -> str:
    mime = "image/gif" if path.suffix == ".gif" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _write_viewer(
    paths: dict[str, Path],
    objectives: np.ndarray,
    selected_index: int,
    interactive_html: str = "",
) -> None:
    values = objectives[selected_index]
    rows = "".join(
        f"<tr{' class=selected' if i == selected_index else ''}><td>{i}</td>"
        f"<td>{row[0]:.6g}</td><td>{row[1]:.6g}</td><td>{row[2]:.6g}</td></tr>"
        for i, row in enumerate(objectives)
    )
    images = "".join(
        f'<figure><img src="{_data_uri(paths[key])}" '
        f'alt="{escape(caption)}"><figcaption>{escape(caption)}</figcaption></figure>'
        for key, caption in (
            ("animation", "Выбранное решение: движение всех шести звеньев"),
            ("scene", "Сбалансированное решение: путь инструмента и позы манипулятора"),
            ("comparison", "Четыре представителя Парето, исходный B-сплайн и суставный RRT*"),
            ("pareto", "Критерии проверенных допустимых решений"),
            ("joints", "Суставная траектория и обратная динамика"),
        )
    )
    named_animations = [(key, path) for key, path in paths.items() if key.startswith("animation_")]
    # Основной ракурс уже показан выше; дополнительные GIF включаются по одному разу.
    extra_views = "".join(
        f'<figure><img src="{_data_uri(path)}" '
        f'alt="{_VIEW_ANGLES[key.removeprefix("animation_")][2]}">'
        f"<figcaption>{_VIEW_ANGLES[key.removeprefix('animation_')][2]}</figcaption></figure>"
        for key, path in named_animations[1:]
    )
    paths["viewer"].write_text(
        '<!doctype html><html lang="ru"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Планирование траектории манипулятора 6R</title><style>"
        "body{font:16px/1.5 system-ui,sans-serif;color:#233447;background:#f4f6f9;"
        "max-width:1200px;margin:auto;padding:32px}h1{line-height:1.2}"
        "figure{margin:24px 0;padding:16px;background:white;border-radius:10px}"
        "img{max-width:100%;height:auto;display:block;margin:auto}"
        "figcaption{text-align:center;color:#526378}table{border-collapse:collapse;"
        "background:white;width:100%}th,td{text-align:right;padding:8px 16px;"
        "border-bottom:1px solid #dce3eb}.selected{background:#e2f0fa}"
        ".table{max-height:360px;overflow:auto}</style><h1>Планирование траектории 6R</h1>"
        "<p>RRT* → B-сплайны → NSGA-II. Геометрия задана в метрах, суставные координаты — "
        "в радианах. Графики вычислены непосредственно по сохранённому B-сплайну.</p>"
        f"<p>Выбранное решение: <strong>{selected_index}</strong>; "
        f"J<sub>τ</sub> = {values[0]:.6g} (N m)² s; "
        f"J<sub>jerk</sub> = {values[1]:.6g} rad²/s⁵; T = {values[2]:.6g} s.</p>"
        + interactive_html
        + images
        + extra_views
        + '<h2>Сохранённые допустимые решения</h2><div class="table"><table>'
        "<thead><tr><th>Номер</th><th>Jτ [(N m)² s]</th><th>Jjerk [rad²/s⁵]</th><th>T [s]</th>"
        "</tr></thead><tbody>" + rows + "</tbody></table></div>"
        "<p>Фронт приближённый: он получен за конечное число итераций. Критерий усилий "
        "определён интегралом квадратов моментов; его единица измерения "
        "отличается от джоуля.</p></html>",
        encoding="utf-8",
    )


def create_report(
    output_dir: Path,
    robot: Any,
    scene: dict[str, Any],
    trajectories: list[JointSplineTrajectory],
    objectives: np.ndarray,
    selected_index: int,
    baseline: JointSplineTrajectory | None = None,
    *,
    views: tuple[str, ...] = ("isometric",),
    frame_count: int = 55,
    rrt_path: Any = None,
) -> dict[str, str]:
    """Сохранить PNG, GIF с выбранных ракурсов и автономную вращаемую 3D-сцену.

    Строки ``objectives`` соответствуют переданным проверенным траекториям;
    порядок критериев: ``[integral(torque**2), integral(jerk**2), T]``.
    Проверка допустимости и выбор решения выполняются вызывающей стороной.
    Возвращаются абсолютные пути, пригодные для сохранения в ``report.json``.
    """
    objectives = np.asarray(objectives, dtype=float)
    if objectives.shape != (len(trajectories), 3) or not np.isfinite(objectives).all():
        raise ValueError("objectives must be a finite (number of trajectories, 3) array")
    if not trajectories or not 0 <= selected_index < len(trajectories):
        raise ValueError("selected_index must identify an existing trajectory")
    if (
        not views
        or any(view not in _VIEW_ANGLES for view in views)
        or len(set(views)) != len(views)
    ):
        raise ValueError("views: нужны неповторяющиеся isometric, front, side или top")
    if not isinstance(frame_count, int) or frame_count < 2:
        raise ValueError("frame_count должен быть целым числом не меньше двух")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        key: output_dir / name
        for key, name in (
            ("scene", "scene.png"),
            ("pareto", "pareto.png"),
            ("joints", "joints.png"),
            ("animation", "animation.gif"),
            ("viewer", "view.html"),
            ("interactive", "interactive.html"),
            ("comparison", "comparison.png"),
        )
    }
    selected = trajectories[selected_index]
    records = select_variants(objectives, selected_index)
    paths.update(save_variants(output_dir, records))
    # Локальный стиль сохраняет настройки интерактивного ноутбука пользователя.
    import matplotlib as mpl

    with mpl.rc_context(_STYLE):
        figures = [
            ("scene", _scene_figure(robot, scene, selected, baseline, views[0])),
            ("pareto", _pareto_figure(objectives, selected_index)),
            ("joints", _joint_figure(robot, selected)),
            (
                "comparison",
                _comparison_figure(
                    robot,
                    scene,
                    trajectories,
                    records,
                    baseline,
                    rrt_path,
                    views[0],
                ),
            ),
        ]
        for key, figure in figures:
            figure.savefig(paths[key], dpi=160, facecolor="white")
            figure.clear()
        _save_animation(paths["animation"], robot, scene, selected, views[0], frame_count)
        for view in views:
            paths[f"animation_{view}"] = output_dir / f"animation_{view}.gif"
            paths[f"scene_{view}"] = output_dir / f"scene_{view}.png"
            if view == views[0]:
                shutil.copyfile(paths["animation"], paths[f"animation_{view}"])
                shutil.copyfile(paths["scene"], paths[f"scene_{view}"])
            else:
                figure = _scene_figure(robot, scene, selected, baseline, view)
                figure.savefig(paths[f"scene_{view}"], dpi=160, facecolor="white")
                figure.clear()
                _save_animation(
                    paths[f"animation_{view}"], robot, scene, selected, view, frame_count
                )
    from .interactive import interactive_fragment, write_interactive_viewer

    interactive_html = interactive_fragment(
        robot,
        scene,
        selected,
        baseline,
        frame_count,
        views[0],
        trajectories=trajectories,
        records=records,
        rrt_path=rrt_path,
    )
    write_interactive_viewer(paths["interactive"], interactive_html)
    _write_viewer(paths, objectives, selected_index, interactive_html)
    return {key: str(path) for key, path in paths.items()}
