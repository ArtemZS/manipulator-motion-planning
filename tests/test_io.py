"""Временная блокировка файла не должна терять рассчитанную траекторию."""

import json
from pathlib import Path

import pytest

from manipulator_planning import io


def test_atomic_json_retries_only_temporary_permission_denial(tmp_path, monkeypatch):
    target = tmp_path / "trajectories.json"
    target.write_text('{"old":true}', encoding="utf-8")
    replace = Path.replace
    attempts, delays = [], []

    def sometimes_locked(source, destination):
        attempts.append(source)
        assert json.loads(target.read_text(encoding="utf-8")) == {"old": True}
        if len(attempts) < 3:
            raise PermissionError("Временная блокировка защитником Windows")
        return replace(source, destination)

    monkeypatch.setattr(Path, "replace", sometimes_locked)
    monkeypatch.setattr(io, "sleep", delays.append)
    io.write_json(target, {"результат": [1, 2, 3]})
    assert len(attempts) == 3 and delays == [0.02, 0.05]
    assert json.loads(target.read_text(encoding="utf-8")) == {"результат": [1, 2, 3]}
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("error", [OSError("Ошибка диска"), PermissionError("Постоянный запрет")])
def test_atomic_json_preserves_previous_file_on_error_and_bounds_retries(
    tmp_path, monkeypatch, error
):
    target = tmp_path / "report.json"
    target.write_text('{"old":true}', encoding="utf-8")
    attempts, delays = [], []

    def locked(source, destination):
        attempts.append(source)
        raise error

    monkeypatch.setattr(Path, "replace", locked)
    monkeypatch.setattr(io, "sleep", delays.append)
    with pytest.raises(type(error)):
        io.write_json(target, {"new": True})
    expected_attempts = 5 if isinstance(error, PermissionError) else 1
    assert len(attempts) == expected_attempts
    assert len(delays) == expected_attempts - 1
    assert json.loads(target.read_text(encoding="utf-8")) == {"old": True}
    assert list(tmp_path.iterdir()) == [target]


def test_invalid_json_is_rejected_before_creating_temporary_file(tmp_path):
    with pytest.raises(ValueError):
        io.write_json(tmp_path / "report.json", {"x": float("nan")})
    assert not list(tmp_path.iterdir())
