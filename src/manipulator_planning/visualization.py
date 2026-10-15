"""Воспроизводимые графики и автономный просмотр сохранённых траекторий.

Каждый график и кадр анимации вычисляется непосредственно по переданному
B-сплайну. Повторная интерполяция и сглаживание не выполняются. Графики
и подписи используют единицы СИ.
"""

from __future__ import annotations

import base64
from html import escape
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

if TYPE_CHECKING:
    from .trajectory import JointSplineTrajectory


_LINK_COLORS = ["#246a9a", "#3d8dc0", "#50a3a4", "#e1a73b", "#d97935", "#b85450"]
_STYLE = {"font.size": 10, "axes.labelsize": 10, "axes.titlesize": 12}


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
            raise ValueError(f"Unsupported obstacle for visualization: {kind!r}")
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
            str(obstacle.get("name", "obstacle")),
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


def _configure_scene_axes(ax: Any, positions: np.ndarray, scene: dict[str, Any]) -> None:
    points = [positions.reshape(-1, 3), np.array([[0.0, 0.0, 0.0]])]
    for obstacle in scene.get("obstacles", []):
        position = np.asarray(obstacle["position"], dtype=float)
        half_size = np.asarray(obstacle.get("half_size", [obstacle.get("radius", 0.0)] * 3))
        points.append(np.array([position - half_size, position + half_size]))
    points = np.concatenate(points)
    minimum, maximum = points.min(axis=0), points.max(axis=0)
    center = (minimum + maximum) / 2.0
    radius = max(float(np.max(maximum - minimum)) * 0.57, 0.3)
    ax.set(
        xlim=(center[0] - radius, center[0] + radius),
        ylim=(center[1] - radius, center[1] + radius),
        zlim=(center[2] - radius, center[2] + radius),
        xlabel="x [m]",
        ylabel="y [m]",
        zlabel="z [m]",
    )
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=24, azim=-55)
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


def _scene_figure(
    robot: Any,
    scene: dict[str, Any],
    trajectory: JointSplineTrajectory,
    baseline: JointSplineTrajectory | None,
) -> Figure:
    figure = _figure((10, 8))
    figure.get_layout_engine().set(rect=(0, 0.05, 1, 0.95))
    ax = figure.add_subplot(projection="3d")
    data = trajectory.sample(241)
    positions = _positions(robot, data["q"])
    _configure_scene_axes(ax, positions, scene)
    _draw_obstacles(ax, scene)
    _draw_base(ax, robot)
    tool = positions[:, -1]
    ax.plot(*tool.T, color="#165f9c", lw=2.3, label="Выбранный путь инструмента")
    if baseline is not None:
        baseline_tool = _positions(robot, baseline.sample(181)["q"])[:, -1]
        ax.plot(*baseline_tool.T, color="#a57724", lw=1.5, ls="--", label="Исходный сплайн")
    for index, label, color in ((0, "Старт", "#237a44"), (-1, "Цель", "#93444b")):
        _draw_robot(ax, robot, data["q"][index], alpha=0.20, ghost=True)
        ax.scatter(*tool[index], s=40, color=color, label=label, depthshade=False)
    _draw_robot(ax, robot, data["q"][len(data["q"]) // 2], alpha=0.95)
    ax.set_title(
        "Учебный манипулятор 6R: выбранная траектория\n"
        f"T = {trajectory.duration:.3f} s; препятствия в натуральном масштабе"
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


def _pareto_figure(objectives: np.ndarray, selected_index: int) -> Figure:
    figure = _figure((12, 9))
    figure.get_layout_engine().set(wspace=0.10, hspace=0.07)
    colors = objectives[:, 2]
    normalizer = None
    labels = (
        r"$J_\tau$ [(N m)$^2$ s]",
        r"$J_{\mathrm{jerk}}$ [rad$^2$/s$^5$]",
        "T [s]",
    )
    ax = figure.add_subplot(2, 2, 1, projection="3d")
    points = ax.scatter(*objectives.T, c=colors, cmap="viridis", s=28, alpha=0.8)
    normalizer = points.norm
    ax.scatter(
        *objectives[selected_index],
        marker="*",
        color="#dc5b36",
        s=180,
        edgecolors="#633021",
        label="Выбранное решение",
        depthshade=False,
    )
    ax.set(
        xlabel=r"$J_\tau$",
        ylabel=r"$J_{\mathrm{jerk}}$",
        zlabel="T [s]",
        title="Минимизация трёх критериев",
    )
    ax.tick_params(labelsize=8)
    ax.view_init(elev=23, azim=-55)
    ax.legend(fontsize=9)
    for slot, (i, j) in enumerate(((0, 1), (0, 2), (1, 2)), start=2):
        ax = figure.add_subplot(2, 2, slot)
        ax.scatter(
            objectives[:, i],
            objectives[:, j],
            c=colors,
            cmap="viridis",
            norm=normalizer,
            s=28,
            alpha=0.85,
        )
        ax.scatter(
            objectives[selected_index, i],
            objectives[selected_index, j],
            marker="*",
            color="#dc5b36",
            edgecolors="#633021",
            s=160,
        )
        ax.set(xlabel=labels[i], ylabel=labels[j])
        ax.grid(alpha=0.22)
        ax.ticklabel_format(axis="both", style="sci", scilimits=(-3, 4), useMathText=True)
    figure.suptitle(
        f"Приближённый фронт Парето: допустимых решений — {len(objectives)}",
        fontsize=14,
    )
    figure.colorbar(points, ax=figure.axes, label="Длительность T [s]", shrink=0.65, pad=0.04)
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
) -> None:
    figure = _figure((7.2, 6.2))
    figure.get_layout_engine().set(rect=(0, 0.055, 1, 0.945))
    ax = figure.add_subplot(projection="3d")
    data = trajectory.sample(55)
    positions = _positions(robot, data["q"])
    _configure_scene_axes(ax, positions, scene)
    _draw_obstacles(ax, scene)
    _draw_base(ax, robot)
    ax.plot(*positions[:, -1].T, color="#7f9eb7", lw=1, ls="--", alpha=0.7)
    artists = _draw_robot(ax, robot, data["q"][0])
    radii = _link_radii(robot, len(artists))
    (trail,) = ax.plot([], [], [], color="#165f9c", lw=2)
    title = ax.set_title("")
    figure.text(
        0.5,
        0.015,
        "Показ за 5,5 s; физическое время движения указано в заголовке.",
        ha="center",
        fontsize=9,
    )

    def update(frame: int) -> tuple[Any, ...]:
        for i, artist in enumerate(artists):
            artist.set_verts(_capsule_faces(positions[frame, i], positions[frame, i + 1], radii[i]))
        tool = positions[: frame + 1, -1]
        trail.set_data_3d(*tool.T)
        title.set_text(
            f"Учебный манипулятор 6R | t = {data['time'][frame]:.2f} / {trajectory.duration:.2f} s"
        )
        return (*artists, trail, title)

    animation = FuncAnimation(figure, update, frames=len(positions), interval=100, blit=False)
    animation.save(path, writer=PillowWriter(fps=10), dpi=100)
    figure.clear()


def _data_uri(path: Path) -> str:
    mime = "image/gif" if path.suffix == ".gif" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _write_viewer(paths: dict[str, Path], objectives: np.ndarray, selected_index: int) -> None:
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
            ("scene", "Трёхмерная сцена и путь инструмента по исходному сплайну"),
            ("pareto", "Критерии проверенных допустимых решений"),
            ("joints", "Суставная траектория и обратная динамика"),
        )
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
        + images
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
) -> dict[str, str]:
    """Сохранить PNG-графики, GIF-анимацию и автономный HTML-отчёт.

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
        )
    }
    selected = trajectories[selected_index]
    # Локальный стиль сохраняет настройки интерактивного ноутбука пользователя.
    import matplotlib as mpl

    with mpl.rc_context(_STYLE):
        figures = [
            ("scene", _scene_figure(robot, scene, selected, baseline)),
            ("pareto", _pareto_figure(objectives, selected_index)),
            ("joints", _joint_figure(robot, selected)),
        ]
        for key, figure in figures:
            figure.savefig(paths[key], dpi=160, facecolor="white")
            figure.clear()
        _save_animation(paths["animation"], robot, scene, selected)
    _write_viewer(paths, objectives, selected_index)
    return {key: str(path) for key, path in paths.items()}
