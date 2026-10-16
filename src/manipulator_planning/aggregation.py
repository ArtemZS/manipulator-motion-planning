"""Явное объединение завершённых серий с сохранением исходного происхождения пар."""

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import yaml

from .experiments import (
    CALCULATION_FILES,
    PROTOCOL_VERSION,
    _create_report,
    _flatten_summary,
    _paired_rows,
    _write_csv,
    calculation_hashes,
    pipeline_calculation_hash,
    summarize_experiments,
)
from .io import write_json
from .scenes import load_named_scene


def _read_json(path: Path) -> dict:
    """Требовать объект JSON, чтобы повреждённый источник не стал тихо пустой серией."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Ожидался объект JSON: {path}")
    return value


def _canonical_hash(value: dict) -> str:
    """Сравнивать содержание YAML независимо от порядка ключей и форматирования."""
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _numeric_options(options: dict) -> dict:
    """Исключить seed и оформление из сравнения одинаковых вычислительных бюджетов."""
    ignored = {"seed", "plots", "views", "frame_count"}
    return {key: value for key, value in options.items() if key not in ignored}


def _reference_hashes(revision: str) -> dict[str, str]:
    """Проверить неизменность численных файлов относительно явно заданного коммита."""
    if not re.fullmatch(r"[0-9a-fA-F]{7,40}", revision):
        raise ValueError("Контрольная ревизия должна быть хешем коммита Git")
    workspace = Path(__file__).resolve().parents[2]
    result = {}
    for name in CALCULATION_FILES:
        relative = f"src/manipulator_planning/{name}"
        completed = subprocess.run(
            ["git", "show", f"{revision}:{relative}"],
            cwd=workspace,
            capture_output=True,
            check=True,
        )
        result[name] = hashlib.sha256(completed.stdout.replace(b"\r\n", b"\n")).hexdigest()
    pipeline_source = subprocess.run(
        ["git", "show", f"{revision}:src/manipulator_planning/pipeline.py"],
        cwd=workspace,
        capture_output=True,
        check=True,
    )
    result["pipeline_numeric_prefix"] = pipeline_calculation_hash(pipeline_source.stdout)
    return result


def combine_experiments(
    scene_sources: dict[str, Path],
    output_dir: Path,
    *,
    replacements: dict[tuple[str, int], Path] | None = None,
    seeds: list[int] | None = None,
    reference_revision: str | None = None,
) -> dict:
    """Объединить выбранные сцены и явные повторы, не меняя кеш и исходные серии.

    У каждой сцены ровно один основной источник. Замена пары задаётся отдельно и
    сохраняет сведения о первоначальной попытке. Проверяются бюджеты, seed, версии
    библиотек, точное актуальное содержание сцен и численные исходники. Старые серии
    без отдельных хешей численных файлов требуют явной контрольной ревизии Git;
    её выбор фиксируется как предпосылка совместимости в provenance.json.
    """
    if not scene_sources:
        raise ValueError("Нужна хотя бы одна исходная серия")
    replacements = replacements or {}
    sources = {name: Path(path).resolve() for name, path in scene_sources.items()}
    replacement_sources = {key: Path(path).resolve() for key, path in replacements.items()}
    output_dir = output_dir.resolve()
    for source in (*sources.values(), *replacement_sources.values()):
        if output_dir.is_relative_to(source) or source.is_relative_to(output_dir):
            raise ValueError("Каталог объединения не должен пересекаться с исходными сериями")
    summaries = {
        source: _read_json(source / "summary.json")
        for source in set(sources.values()) | set(replacement_sources.values())
    }
    first = summaries[next(iter(sources.values()))]
    seeds = list(first["seeds"] if seeds is None else seeds)
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(not isinstance(s, int) or s < 0 for s in seeds)
    ):
        raise ValueError("Нужны различные неотрицательные целые seed")
    if any(scene not in sources or seed not in seeds for scene, seed in replacements):
        raise ValueError("Замена пары должна принадлежать выбранным сценам и seed")
    numeric_options = _numeric_options(first["options"])
    environment = first["environment"]
    current_hashes = calculation_hashes()
    reference = _reference_hashes(reference_revision) if reference_revision else None
    origins = {}
    for source, summary in summaries.items():
        if summary.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError(f"Несовместимый протокол: {source}")
        if _numeric_options(summary["options"]) != numeric_options:
            raise ValueError(f"Вычислительные параметры отличаются: {source}")
        if summary["environment"] != environment:
            raise ValueError(f"Версии численных библиотек отличаются: {source}")
        recorded_hashes = summary.get("calculation_sources")
        compatibility_basis = "Хеши численных файлов, записанные исходной серией"
        if recorded_hashes is None:
            if reference is None:
                raise ValueError(f"Для старой серии нужна явная контрольная ревизия: {source}")
            recorded_hashes = reference
            compatibility_basis = (
                "Явная предпосылка: численные методы исходной серии соответствуют "
                f"{reference_revision}; текущие семь численных модулей и XML модели "
                "побайтово проверены с нормализацией LF. AST настроек, сглаживания и "
                "вычислительной части пайплайна до сохранения артефактов также совпадает."
            )
        if recorded_hashes != current_hashes:
            raise ValueError(f"Изменены численные методы или модель: {source}")
        manifest = source / "environment.json"
        origins[str(source)] = {
            "directory": str(source),
            "source_hash": summary.get("source_hash"),
            "environment": summary["environment"],
            "protocol": summary["protocol"],
            "environment_manifest": _read_json(manifest) if manifest.exists() else None,
            "numerical_compatibility_basis": compatibility_basis,
            "calculation_sources": recorded_hashes,
        }
    pairs, plans, provenance_rows = [], [], []
    for name, primary in sources.items():
        primary_summary = summaries[primary]
        if name not in primary_summary["scenes"] or set(primary_summary["seeds"]) != set(seeds):
            raise ValueError(f"Основная серия не содержит полный набор seed сцены {name}")
        expected_scene_hash = _canonical_hash(load_named_scene(name))
        for seed in seeds:
            source = replacement_sources.get((name, seed), primary)
            summary = summaries[source]
            if name not in summary["scenes"] or seed not in summary["seeds"]:
                raise ValueError(f"В источнике отсутствует пара {name}, seed={seed}: {source}")
            directory = source / "runs" / name / f"seed_{seed}"
            pair_path = directory / "pair.json"
            pair = _read_json(pair_path)
            if (
                pair.get("scene") != name
                or pair.get("seed") != seed
                or not pair.get("completed")
                or not isinstance(pair.get("report"), dict)
                or not pair.get("fingerprint")
            ):
                raise ValueError(f"Неполная или неверная пара: {pair_path}")
            options = pair["report"]["options"]
            if options["seed"] != seed or _numeric_options(options) != numeric_options:
                raise ValueError(
                    f"Пара имеет другой seed или вычислительные параметры: {pair_path}"
                )
            scene = yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))
            if _canonical_hash(scene) != expected_scene_hash:
                raise ValueError(f"Содержание сцены отличается от актуального: {directory}")
            target = output_dir / "runs" / name / f"seed_{seed}"
            pair_digest = hashlib.sha256(pair_path.read_bytes()).hexdigest()
            if target.exists():
                copied = target / "pair.json"
                if (
                    not copied.exists()
                    or hashlib.sha256(copied.read_bytes()).hexdigest() != pair_digest
                ):
                    raise ValueError(f"Каталог назначения содержит другую пару: {target}")
            entry = {
                "scene": name,
                "seed": seed,
                "source_series": str(source),
                "source_pair": str(pair_path),
                "original_fingerprint": pair["fingerprint"],
                "pair_json_sha256": pair_digest,
                "scene_content_sha256": expected_scene_hash,
                "source_hash": summary.get("source_hash"),
                "execution_workers": summary["protocol"].get("workers"),
                "replacement": (name, seed) in replacements,
            }
            if entry["replacement"]:
                original_path = primary / "runs" / name / f"seed_{seed}" / "pair.json"
                original = _read_json(original_path)
                if (
                    original["report"].get("status") != "failed"
                    or original["report"].get("failure_stage") != "artifacts"
                ):
                    raise ValueError(
                        "Операционная замена разрешена только после ошибки сохранения; "
                        "вычислительный отказ или успешную попытку заменять нельзя: "
                        f"{original_path}"
                    )
                entry["superseded_attempt"] = {
                    "source_pair": str(original_path),
                    "fingerprint": original["fingerprint"],
                    "status": original["report"].get("status"),
                    "failure_stage": original["report"].get("failure_stage"),
                    "reason": original["report"].get("reason"),
                    "pair_json_sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(),
                }
                entry["execution_note"] = (
                    "Отдельный повтор после ошибки ввода-вывода; численные методы "
                    "и seed сохранены. "
                    "Нагрузка отличается от основной серии; время не является строго сопоставимым."
                )
            pairs.append(pair)
            plans.append((directory, target))
            provenance_rows.append(entry)
    # Все проверки завершены до появления новой сводки и копирования результатов.
    output_dir.mkdir(parents=True, exist_ok=True)
    for source, destination in plans:
        if not destination.exists():
            shutil.copytree(source, destination, ignore=shutil.ignore_patterns("*.tmp"))
    rows = []
    for pair, origin in zip(pairs, provenance_rows, strict=True):
        current_rows = _paired_rows(
            pair["scene"], pair["seed"], pair["report"], f"runs/{pair['scene']}/seed_{pair['seed']}"
        )
        for row in current_rows:
            row.update(
                {
                    "source_series": origin["source_series"],
                    "source_fingerprint": origin["original_fingerprint"],
                    "execution_workers": origin["execution_workers"],
                    "replacement": origin["replacement"],
                    "execution_note": origin.get("execution_note", ""),
                }
            )
        rows.extend(current_rows)
    provenance = {
        "schema_version": 1,
        "policy": "Явное объединение; pair.json и fingerprints сохранены без изменений",
        "reference_revision_assumption": reference_revision,
        "current_calculation_sources": current_hashes,
        "origins": origins,
        "pairs": provenance_rows,
    }
    write_json(output_dir / "provenance.json", provenance)
    summary = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "scenes": list(sources),
        "seeds": seeds,
        "options": first["options"],
        "source_hash": None,
        "source_hashes": sorted({summary["source_hash"] for summary in summaries.values()}),
        "calculation_sources": current_hashes,
        "environment": environment,
        "protocol": {
            **first["protocol"],
            "workers": None,
            "worker_counts": sorted({entry["execution_workers"] for entry in provenance_rows}),
            "aggregation": "Численные критерии объединены; нагрузка сохранена для каждой пары",
        },
        "aggregation": {
            "provenance": "provenance.json",
            "pairs": len(pairs),
            "replacements": sum(entry["replacement"] for entry in provenance_rows),
            "physical_attempts": len(pairs)
            + sum(entry["replacement"] for entry in provenance_rows),
            "success_rate_unit": (
                "Логическая пара (сцена, seed) после явного повтора ошибки сохранения. "
                "Первоначальные операционные отказы сохранены в provenance.json и исходной серии."
            ),
            "original_fingerprints_preserved": True,
        },
        **summarize_experiments(pairs, rows),
        "artifacts": {
            "runs": "runs.csv",
            "summary": "summary.csv",
            "report": "index.html",
            "markdown": "REPORT.md",
            "statistics": "statistics.png",
            "provenance": "provenance.json",
        },
    }
    write_json(output_dir / "summary.json", summary)
    _write_csv(output_dir / "runs.csv", rows)
    _write_csv(output_dir / "summary.csv", _flatten_summary(summary))
    _create_report(summary, rows, output_dir)
    return summary
