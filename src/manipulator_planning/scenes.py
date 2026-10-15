"""Каталог воспроизводимых пространственных сцен для учебного манипулятора."""

from importlib.resources import files

import yaml

_DESCRIPTIONS = {
    "simple": "Два препятствия: центральная сфера и низкий блок; исходный пример.",
    "scattered": "Семь рассеянных сфер и блоков разного размера.",
    "gate": "Рама с окном, нижней перемычкой и дополнительными препятствиями.",
    "shelves": "Стеллаж с двумя полками, стенками и предметом между полками.",
    "cluttered": "Восемь объектов на разных высотах, включая верхнюю балку.",
}


def load_named_scene(name: str) -> dict:
    """Прочитать независимую копию сцены из ресурсов установленного пакета.

    Диагностический обход хранится только для геометрических тестов. Планировщик
    должен получать начальную и конечную конфигурации, препятствия и ограничения;
    готовый обход не служит подсказкой или начальной популяцией поиска.
    """
    if name not in _DESCRIPTIONS:
        available = ", ".join(_DESCRIPTIONS)
        raise ValueError(f"Неизвестная сцена {name!r}. Доступны: {available}")
    relative = "assets/simple.yaml" if name == "simple" else f"assets/scenes/{name}.yaml"
    source = files("manipulator_planning").joinpath(relative)
    scene = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(scene, dict):
        raise ValueError(f"Описание сцены {name!r} должно быть словарём YAML")
    return scene


def list_scenes() -> list[dict]:
    """Вернуть имена, русские описания и число препятствий доступных сцен."""
    return [
        {
            "name": name,
            "description": description,
            "obstacle_count": len(load_named_scene(name).get("obstacles", [])),
        }
        for name, description in _DESCRIPTIONS.items()
    ]
