"""Воспроизводимая учебная модель с шестью вращательными суставами, в СИ.

Локальные оси суставов: z, y, y, x, y, x. Массы капсул задают положительные
инерции твёрдых тел; MuJoCo вычисляет прямую кинематику и динамику RNE.
Это численная учебная модель без калибровки по реальному оборудованию.
Данные MuJoCo изменяемы: экземпляр нельзя одновременно использовать в разных потоках.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from importlib.resources import files

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray


class RobotModel:
    """Неподвижное основание, 6R-манипулятор, сферические и прямоугольные препятствия."""

    nq = 6
    velocity_limits = np.array([1.8, 1.6, 1.8, 2.5, 2.5, 3.0])
    acceleration_limits = np.array([4.0, 4.0, 5.0, 6.0, 6.0, 8.0])
    torque_limits = np.array([60.0, 60.0, 40.0, 12.0, 12.0, 8.0])
    # Суммы длин последующих звеньев и наибольшего радиуса капсулы.
    # Они ограничивают расстояния до всех затрагиваемых точек при любой позе.
    _sweep_radii = np.array([1.335, 1.235, 0.815, 0.475, 0.315, 0.175])

    def __init__(self, scene: dict | None = None) -> None:
        self.scene = scene or {}
        source = files("manipulator_planning").joinpath("assets/educational_6r.xml")
        root = ET.fromstring(source.read_text(encoding="utf-8"))
        world = root.find("worldbody")
        assert world is not None
        self.obstacles: list[dict] = []
        used_names = {"floor", "base", *(f"link{i}" for i in range(1, 7))}
        for i, obstacle in enumerate(self.scene.get("obstacles", [])):
            obstacle = dict(obstacle)
            name = str(obstacle.get("name", f"obstacle_{i}"))
            if name in used_names:
                raise ValueError(f"Duplicate or reserved geometry name: {name}")
            used_names.add(name)
            kind = obstacle["kind"]
            position = self._vector(obstacle["position"], 3, "obstacle position")
            attrs = {"name": name, "pos": self._xml_vector(position), "rgba": "0.87 0.32 0.25 0.85"}
            if kind == "sphere":
                radius = float(obstacle["radius"])
                if not np.isfinite(radius) or radius <= 0:
                    raise ValueError("Obstacle radius must be finite and positive")
                attrs.update(type="sphere", size=str(radius))
            elif kind == "box":
                size = self._vector(obstacle["half_size"], 3, "box half_size")
                if np.any(size <= 0):
                    raise ValueError("Box half_size must be positive")
                attrs.update(type="box", size=self._xml_vector(size))
            else:
                raise ValueError(f"Unsupported obstacle kind: {kind}")
            ET.SubElement(world, "geom", attrs)
            obstacle["name"] = name
            self.obstacles.append(obstacle)
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.data = mujoco.MjData(self.model)
        self.joint_lower = self.model.jnt_range[:, 0].copy()
        self.joint_upper = self.model.jnt_range[:, 1].copy()
        self.link_geom_ids = tuple(self.model.geom(f"link{i}").id for i in range(1, 7))
        self.link_radii = self.model.geom_size[list(self.link_geom_ids), 0].copy()
        self.world_geom_ids = tuple(
            self.model.geom(name).id
            for name in ("floor", "base", *(o["name"] for o in self.obstacles))
        )
        self.tool_site_id = self.model.site("tool").id
        self.set_configuration(np.zeros(self.nq))

    @staticmethod
    def _xml_vector(value: NDArray[np.float64]) -> str:
        return " ".join(map(str, value))

    @staticmethod
    def _vector(value: ArrayLike, size: int, label: str) -> NDArray[np.float64]:
        vector = np.asarray(value, dtype=float)
        if vector.shape != (size,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"{label} must be a finite vector of shape ({size},)")
        return vector

    def set_configuration(self, q: ArrayLike) -> None:
        self.data.qpos[:] = self._vector(q, self.nq, "q")
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def joint_positions(self, q: ArrayLike) -> NDArray[np.float64]:
        """Координаты шести суставов и следующей за ними конечной точки, в метрах."""
        self.set_configuration(q)
        return np.vstack((self.data.xanchor, self.data.site_xpos[self.tool_site_id])).copy()

    def tool_position(self, q: ArrayLike) -> NDArray[np.float64]:
        self.set_configuration(q)
        return self.data.site_xpos[self.tool_site_id].copy()

    def inverse_dynamics(self, q: ArrayLike, qd: ArrayLike, qdd: ArrayLike) -> NDArray[np.float64]:
        """Моменты твёрдых тел M(q)qdd + C(q, qd) с гравитацией, в Н·м.

        Явный RNE исключает контактные силы, реакции ограничителей, приводы
        и трение. Ограничения суставов проверяются отдельно в пайплайне.
        """
        self.data.qpos[:] = self._vector(q, self.nq, "q")
        self.data.qvel[:] = self._vector(qd, self.nq, "qd")
        mujoco.mj_forward(self.model, self.data)
        self.data.qacc[:] = self._vector(qdd, self.nq, "qdd")
        moments = np.empty(self.nq)
        mujoco.mj_rne(self.model, self.data, 1, moments)
        return moments

    def mass_matrix(self, q: ArrayLike) -> NDArray[np.float64]:
        self.set_configuration(q)
        matrix = np.empty((self.nq, self.nq))
        mujoco.mj_fullM(self.model, self.data, matrix)
        return matrix

    def motion_bound(self, delta_q: ArrayLike) -> float:
        """Консервативная граница смещения любой материальной точки, в метрах.

        Поворот на d радиан смещает последующую точку не более чем на r*abs(d).
        Сумма оценок действительна и при одновременном вращении суставов.
        """
        delta = self._vector(delta_q, self.nq, "delta_q")
        return float(self._sweep_radii @ np.abs(delta))
