"""Командная строка демонстрации и повторного воспроизведения результатов."""

import argparse
import sys
from pathlib import Path

from .pipeline import PipelineOptions, load_scene, replay_results, run_pipeline


def main(argv: list[str] | None = None) -> int:
    """Запустить выбранную команду, вернув ненулевой код при неуспехе."""
    parser = argparse.ArgumentParser(description="Оптимизация движения учебного 6R-манипулятора")
    subcommands = parser.add_subparsers(dest="command", required=True)
    demo = subcommands.add_parser("demo", help="Запустить полный каскад на 3D-сцене")
    demo.add_argument("--scene", type=Path, help="Путь к YAML-описанию сцены")
    demo.add_argument("--output", type=Path, default=Path("results/demo"))
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument(
        "--quick", action="store_true", help="Уменьшенный бюджет для проверки запуска"
    )
    demo.add_argument("--iterations", type=int, help="Бюджет итераций RRT*")
    demo.add_argument("--population", type=int, help="Размер популяции NSGA-II")
    demo.add_argument("--generations", type=int, help="Число поколений NSGA-II")
    demo.add_argument("--no-plots", action="store_true", help="Сохранить числа без графиков и GIF")
    replay = subcommands.add_parser("replay", help="Проверить и показать сохранённый результат")
    replay.add_argument("directory", type=Path)
    replay.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            options = PipelineOptions(
                seed=args.seed,
                rrt_iterations=args.iterations
                if args.iterations is not None
                else (180 if args.quick else 500),
                population=args.population
                if args.population is not None
                else (12 if args.quick else 32),
                generations=args.generations
                if args.generations is not None
                else (5 if args.quick else 24),
                plots=not args.no_plots,
            )
            run_pipeline(load_scene(args.scene), args.output, options)
        else:
            result = replay_results(args.directory, regenerate=not args.no_plots)
            print(f"Повторно проверено решений: {len(result['solutions'])}")
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    return 0
