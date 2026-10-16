"""Атомарная запись отчётов с ограниченным повтором временных блокировок Windows."""

import json
import os
import tempfile
from pathlib import Path
from time import sleep

_RETRY_DELAYS = (0.02, 0.05, 0.10, 0.20)


def write_json(path: Path, value: dict) -> None:
    """Заменить JSON целиком; повторить только PermissionError при атомарном переименовании.

    Уникальное имя в том же каталоге исключает конфликт временных файлов разных
    писателей. Ошибка сериализации или иной ввода-вывода сразу передаётся вызывающему
    коду. Старый файл остаётся целым при исчерпании пяти попыток переименования.
    """
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
        for attempt in range(len(_RETRY_DELAYS) + 1):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt == len(_RETRY_DELAYS):
                    raise
                sleep(_RETRY_DELAYS[attempt])
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except PermissionError:
            # Запертый посторонним процессом временный файл не отменяет исходную ошибку.
            pass
