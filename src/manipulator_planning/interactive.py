"""Автономная трёхмерная сцена с вращением камеры и воспроизведением сплайна.

Геометрия звеньев берётся из той же модели, что и проверка столкновений.
JavaScript Plotly включается в HTML целиком: сервер и доступ к сети не нужны.
"""

from __future__ import annotations

from html import escape
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import plotly.graph_objects as go

from .variants import obstacle_label, rrt_tool_path, variants_html

if TYPE_CHECKING:
    from .trajectory import JointSplineTrajectory


CAMERA_PRESETS = {
    "isometric": {"eye": {"x": 1.36, "y": -1.87, "z": 1.15}, "up": {"x": 0, "y": 0, "z": 1}},
    "front": {"eye": {"x": 0, "y": -2.8, "z": 0.4}, "up": {"x": 0, "y": 0, "z": 1}},
    "side": {"eye": {"x": 2.8, "y": 0, "z": 0.4}, "up": {"x": 0, "y": 0, "z": 1}},
    "top": {"eye": {"x": 0, "y": 0, "z": 3.0}, "up": {"x": 0, "y": 1, "z": 0}},
}
_VIEW_LABELS = {"isometric": "Общий вид", "front": "Спереди", "side": "Сбоку", "top": "Сверху"}


def _mesh_arrays(faces: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Триангуляция многоугольников без изменения физических размеров."""
    vertices = np.concatenate(faces)
    triangles = []
    offset = 0
    for face in faces:
        triangles.extend((offset, offset + i, offset + i + 1) for i in range(1, len(face) - 1))
        offset += len(face)
    indices = np.asarray(triangles, dtype=int)
    return dict(zip(("x", "y", "z", "i", "j", "k"), (*vertices.T, *indices.T), strict=True))


def _sphere_faces(position: np.ndarray, radius: float) -> list[np.ndarray]:
    azimuth = np.linspace(0, 2 * np.pi, 21)
    polar = np.linspace(0, np.pi, 13)
    points = (
        np.stack(
            (
                np.outer(np.cos(azimuth), np.sin(polar)),
                np.outer(np.sin(azimuth), np.sin(polar)),
                np.outer(np.ones_like(azimuth), np.cos(polar)),
            ),
            axis=-1,
        )
        * radius
        + position
    )
    return [
        points[[i, i + 1, i + 1, i], [j, j, j + 1, j + 1]]
        for i in range(len(azimuth) - 1)
        for j in range(len(polar) - 1)
    ]


def _base_faces(robot: Any) -> list[np.ndarray]:
    """Цилиндр неподвижного основания с размерами из модели MuJoCo."""
    if not hasattr(robot, "model"):
        return []
    base_id = robot.model.geom("base").id
    position = np.asarray(robot.model.geom_pos[base_id])
    radius, half_height = robot.model.geom_size[base_id, :2]
    angle = np.linspace(0, 2 * np.pi, 25)
    circle = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), np.zeros_like(angle)))
    lower = circle + position + [0, 0, -half_height]
    upper = circle + position + [0, 0, half_height]
    return [
        np.array([a, b, d, c])
        for (a, b), (c, d) in zip(pairwise(lower), pairwise(upper), strict=True)
    ] + [
        lower,
        upper,
    ]


def build_interactive_figure(
    robot: Any,
    scene: dict[str, Any],
    trajectory: JointSplineTrajectory,
    baseline: JointSplineTrajectory | None = None,
    frame_count: int = 55,
    view: str = "isometric",
    *,
    trajectories: list[JointSplineTrajectory] | None = None,
    records: list[dict[str, Any]] | None = None,
    rrt_path: Any = None,
) -> go.Figure:
    """Настоящие объёмные звенья, фиксированная цель и кадры исходного B-сплайна.

    Перетаскивание вращает камеру, колесо меняет масштаб. Положение камеры
    сохраняется при воспроизведении; ползунок задаёт физическое время движения.
    """
    from .visualization import (
        _LINK_COLORS,
        _box_faces,
        _capsule_faces,
        _goal_position,
        _link_radii,
        _positions,
        _scene_bounds,
    )

    if view not in CAMERA_PRESETS:
        raise ValueError(f"Неизвестный ракурс: {view}")
    if not isinstance(frame_count, int) or frame_count < 2:
        raise ValueError("Число кадров должно быть целым и не меньше двух")
    if records is not None:
        if trajectories is None:
            raise ValueError("Для представителей необходим массив сохранённых траекторий")
        balanced_index = next(
            record["solution_index"] for record in records if record["key"] == "balanced"
        )
        trajectory = trajectories[balanced_index]
    data = trajectory.sample(frame_count)
    positions = _positions(robot, data["q"])
    goal = _goal_position(robot, scene, trajectory)
    tool = positions[:, -1]
    view_points = [positions.reshape(-1, 3), goal[None, :]]
    variant_tools = []
    for record in records or []:
        variant_tool = _positions(robot, trajectories[record["solution_index"]].sample(181)["q"])[
            :, -1
        ]
        variant_tools.append(variant_tool)
        view_points.append(variant_tool)
    baseline_tool = None
    if baseline is not None:
        baseline_tool = _positions(robot, baseline.sample(121)["q"])[:, -1]
        view_points.append(baseline_tool)
    rrt_tool = None
    if rrt_path is not None:
        rrt_tool = rrt_tool_path(robot, rrt_path)
        view_points.append(rrt_tool)
    center, radius = _scene_bounds(np.concatenate(view_points), scene)
    radii = _link_radii(robot, positions.shape[1] - 1)
    figure = go.Figure()

    for index, obstacle in enumerate(scene.get("obstacles", [])):
        position = np.asarray(obstacle["position"], dtype=float)
        if obstacle["kind"] == "box":
            faces = _box_faces(position, np.asarray(obstacle["half_size"]))
        elif obstacle["kind"] == "sphere":
            faces = _sphere_faces(position, float(obstacle["radius"]))
        else:
            raise ValueError(f"Неизвестная форма препятствия: {obstacle['kind']}")
        figure.add_trace(
            go.Mesh3d(
                **_mesh_arrays(faces),
                color="#bd6670",
                opacity=0.4,
                name=obstacle_label(obstacle, index),
                showlegend=index == 0,
                legendgroup="препятствия",
                hoverinfo="name",
                flatshading=True,
            )
        )

    floor = np.array(
        [
            [center[0] - radius, center[1] - radius, 0],
            [center[0] + radius, center[1] - radius, 0],
            [center[0] + radius, center[1] + radius, 0],
            [center[0] - radius, center[1] + radius, 0],
        ]
    )
    figure.add_trace(
        go.Mesh3d(
            **_mesh_arrays([floor]),
            color="#cbd5e1",
            opacity=0.18,
            name="Рабочая поверхность",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    base = _base_faces(robot)
    if base:
        figure.add_trace(
            go.Mesh3d(
                **_mesh_arrays(base),
                color="#607285",
                name="Основание",
                showlegend=False,
                hoverinfo="skip",
                flatshading=True,
            )
        )

    if records is not None:
        for record, variant_tool in zip(records, variant_tools, strict=True):
            figure.add_trace(
                go.Scatter3d(
                    x=variant_tool[:, 0],
                    y=variant_tool[:, 1],
                    z=variant_tool[:, 2],
                    mode="lines",
                    line={"color": record["color"], "width": 4},
                    name=f"{record['label']} · № {record['solution_index']}",
                    hoverinfo="skip",
                )
            )
    else:
        figure.add_trace(
            go.Scatter3d(
                x=tool[:, 0],
                y=tool[:, 1],
                z=tool[:, 2],
                mode="lines",
                line={"color": "#3d83b5", "width": 3, "dash": "dot"},
                name="Путь сбалансированного B-сплайна",
                hoverinfo="skip",
            )
        )
    if baseline_tool is not None:
        figure.add_trace(
            go.Scatter3d(
                x=baseline_tool[:, 0],
                y=baseline_tool[:, 1],
                z=baseline_tool[:, 2],
                mode="lines",
                line={"color": "#737373", "width": 4, "dash": "dash"},
                name="B-сплайн без NSGA-II",
                hoverinfo="skip",
            )
        )
    if rrt_tool is not None:
        figure.add_trace(
            go.Scatter3d(
                x=rrt_tool[:, 0],
                y=rrt_tool[:, 1],
                z=rrt_tool[:, 2],
                mode="lines",
                line={"color": "#334155", "width": 3, "dash": "dot"},
                name="RRT*: линейные рёбра в углах суставов",
                hoverinfo="skip",
            )
        )
    figure.add_trace(
        go.Scatter3d(
            x=[tool[0, 0]],
            y=[tool[0, 1]],
            z=[tool[0, 2]],
            mode="markers+text",
            marker={"size": 7, "color": "#16834b", "line": {"color": "white", "width": 2}},
            text=["СТАРТ"],
            textposition="bottom center",
            textfont={"color": "#16834b"},
            name="СТАРТ",
            hovertemplate="Старт<br>x=%{x:.3f}; y=%{y:.3f}; z=%{z:.3f} м<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=[goal[0], goal[0]],
            y=[goal[1], goal[1]],
            z=[0, goal[2]],
            mode="lines",
            line={"color": "#ed168f", "width": 3, "dash": "dot"},
            name="Проекция цели",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=[goal[0]],
            y=[goal[1]],
            z=[goal[2]],
            mode="markers",
            marker={
                "size": 13,
                "symbol": "diamond",
                "color": "#ed168f",
                "line": {"color": "#621044", "width": 3},
            },
            name="ЦЕЛЬ — фиксированная точка",
            hovertemplate="ЦЕЛЬ<br>x=%{x:.3f}; y=%{y:.3f}; z=%{z:.3f} м<extra></extra>",
        )
    )

    robot_traces = []
    for index, (a, b) in enumerate(pairwise(positions[0])):
        robot_traces.append(len(figure.data))
        figure.add_trace(
            go.Mesh3d(
                **_mesh_arrays(_capsule_faces(a, b, radii[index])),
                color=_LINK_COLORS[index % 6],
                name="Манипулятор 6R",
                showlegend=index == 0,
                legendgroup="манипулятор",
                hoverinfo="skip",
                flatshading=False,
                lighting={"ambient": 0.6, "diffuse": 0.8, "specular": 0.3},
            )
        )
    trail_trace = len(figure.data)
    figure.add_trace(
        go.Scatter3d(
            x=tool[:1, 0],
            y=tool[:1, 1],
            z=tool[:1, 2],
            mode="lines",
            line={"color": "#125d9b", "width": 7},
            name="Пройденный путь",
            hoverinfo="skip",
        )
    )
    tool_trace = len(figure.data)
    figure.add_trace(
        go.Scatter3d(
            x=[tool[0, 0]],
            y=[tool[0, 1]],
            z=[tool[0, 2]],
            mode="markers",
            marker={"size": 6, "color": "#ffd23f", "line": {"color": "#27374c", "width": 2}},
            name="Текущий инструмент",
            hovertemplate="Инструмент<br>x=%{x:.3f}; y=%{y:.3f}; z=%{z:.3f} м<extra></extra>",
        )
    )

    frames = []
    for frame, points in enumerate(positions):
        meshes = []
        for index, (a, b) in enumerate(pairwise(points)):
            mesh = _mesh_arrays(_capsule_faces(a, b, radii[index]))
            meshes.append(go.Mesh3d(x=mesh["x"], y=mesh["y"], z=mesh["z"]))
        frames.append(
            go.Frame(
                name=str(frame),
                data=[
                    *meshes,
                    go.Scatter3d(
                        x=tool[: frame + 1, 0], y=tool[: frame + 1, 1], z=tool[: frame + 1, 2]
                    ),
                    go.Scatter3d(
                        x=tool[frame : frame + 1, 0],
                        y=tool[frame : frame + 1, 1],
                        z=tool[frame : frame + 1, 2],
                    ),
                ],
                traces=[*robot_traces, trail_trace, tool_trace],
            )
        )
    figure.frames = frames
    frame_duration = round(5500 / (frame_count - 1))
    animation_settings = {
        "frame": {"duration": frame_duration, "redraw": True},
        "transition": {"duration": 0},
        "fromcurrent": True,
        "mode": "immediate",
    }
    figure.update_layout(
        title={
            "text": escape(
                str(scene.get("label", scene.get("title", scene.get("name", "Учебная сцена 6R"))))
            ),
            "x": 0.03,
            "y": 0.99,
            "yanchor": "top",
            "font": {"size": 16},
        },
        font={"family": "Arial, sans-serif", "size": 13, "color": "#233447"},
        height=740,
        margin={"l": 0, "r": 0, "t": 175, "b": 85},
        paper_bgcolor="white",
        uirevision="камера-6R",
        showlegend=True,
        legend={"orientation": "h", "y": -0.2, "x": 0},
        scene={
            "xaxis": {"title": "x [м]", "range": [center[0] - radius, center[0] + radius]},
            "yaxis": {"title": "y [м]", "range": [center[1] - radius, center[1] + radius]},
            "zaxis": {"title": "z [м]", "range": [center[2] - radius, center[2] + radius]},
            "aspectmode": "cube",
            "dragmode": "orbit",
            "camera": CAMERA_PRESETS[view],
            "annotations": [
                {
                    "x": goal[0],
                    "y": goal[1],
                    "z": goal[2],
                    "text": "<b>ЦЕЛЬ</b>",
                    "showarrow": True,
                    "ax": 35,
                    "ay": -45,
                    "arrowcolor": "#ed168f",
                    "arrowhead": 2,
                    "font": {"size": 16, "color": "#b5066d"},
                    "bgcolor": "#fff0f8",
                    "bordercolor": "#ed168f",
                    "borderpad": 4,
                }
            ],
        },
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0,
                "y": 1.22,
                "xanchor": "left",
                "yanchor": "top",
                "showactive": False,
                "buttons": [
                    {"label": "▶ Пуск", "method": "animate", "args": [None, animation_settings]},
                    {
                        "label": "❚❚ Пауза",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            },
            {
                "type": "dropdown",
                "direction": "down",
                "x": 0,
                "y": 1.10,
                "xanchor": "left",
                "yanchor": "top",
                "showactive": True,
                "active": list(CAMERA_PRESETS).index(view),
                "buttons": [
                    {
                        "label": _VIEW_LABELS[key],
                        "method": "relayout",
                        "args": [{"scene.camera": camera}],
                    }
                    for key, camera in CAMERA_PRESETS.items()
                ],
            },
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.03,
                "len": 0.94,
                "y": -0.04,
                "currentvalue": {"prefix": "Время: ", "suffix": " с", "visible": True},
                "pad": {"t": 18, "b": 0},
                "steps": [
                    {
                        "label": f"{data['time'][i]:.2f}",
                        "method": "animate",
                        "args": [
                            [str(i)],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                    }
                    for i in range(frame_count)
                ],
            }
        ],
        meta={
            "goal": goal.tolist(),
            "duration": trajectory.duration,
            "sample_times": data["time"].tolist(),
            "views": list(CAMERA_PRESETS),
            "animation_source": "balanced_bspline",
            "animation_solution_index": next(
                (
                    record["solution_index"]
                    for record in records or []
                    if record["key"] == "balanced"
                ),
                None,
            ),
        },
    )
    return figure


def interactive_fragment(
    robot: Any,
    scene: dict[str, Any],
    trajectory: JointSplineTrajectory,
    baseline: JointSplineTrajectory | None = None,
    frame_count: int = 55,
    view: str = "isometric",
    *,
    trajectories: list[JointSplineTrajectory] | None = None,
    records: list[dict[str, Any]] | None = None,
    rrt_path: Any = None,
) -> str:
    """Самодостаточный фрагмент отчёта вместе с локально встроенной библиотекой."""
    figure = build_interactive_figure(
        robot,
        scene,
        trajectory,
        baseline,
        frame_count,
        view,
        trajectories=trajectories,
        records=records,
        rrt_path=rrt_path,
    )
    goal = figure.layout.meta["goal"]
    description = (
        '<section style="padding:20px;background:white;border-radius:12px;margin:24px 0">'
        "<h2>Вращаемая 3D-сцена</h2>"
        "<p>Вращение: перетаскивайте сцену левой кнопкой. Масштаб: колесо мыши. "
        "Сдвиг: правая кнопка. Ракурс можно менять во время воспроизведения. "
        "Ползунок показывает физическое время, полный показ длится около 5,5 с.</p>"
        "<p><strong>Анимация воспроизводит сбалансированный B-сплайн.</strong> "
        "Каждый кадр: суставные углы q(t) из B-сплайна → прямая кинематика. "
        "Прямые звенья изображают жёсткие тела; кривые линии показывают путь инструмента. "
        "Линии остальных вариантов позволяют сравнить геометрию путей.</p>"
        '<p style="color:#b5066d;background:#fff0f8;padding:10px;border:1px solid #ed168f">'
        "<strong>◆ ЦЕЛЬ — фиксированное положение инструмента:</strong> "
        f"x = {goal[0]:.3f}; y = {goal[1]:.3f}; z = {goal[2]:.3f} м. "
        "Жёлтый маркер — текущий инструмент; зелёный — старт.</p>"
    )
    return (
        (variants_html(records) if records is not None else "")
        + description
        + figure.to_html(
            full_html=False,
            include_plotlyjs=True,
            auto_play=False,
            config={"scrollZoom": True, "displayModeBar": False, "responsive": True},
            div_id="manipulator-3d",
        )
        + "</section>"
    )


def write_interactive_viewer(path: Path, fragment: str) -> None:
    """Сохранить отдельную сцену, открываемую двойным щелчком без сервера."""
    path.write_text(
        '<!doctype html><html lang="ru"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Интерактивная сцена манипулятора 6R</title>"
        "<style>body{font:16px/1.5 system-ui,sans-serif;margin:auto;padding:16px;"
        "max-width:1400px;background:#f4f6f9;color:#233447}h2{margin-top:0}</style>"
        "</head><body>" + fragment + "</body></html>",
        encoding="utf-8",
    )
