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
    demo.add_argument("--scene", help="Имя сцены из каталога или путь к YAML")
    demo.add_argument("--output", type=Path, default=Path("results/demo"))
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument(
        "--quick", action="store_true", help="Уменьшенный бюджет для проверки запуска"
    )
    demo.add_argument("--iterations", type=int, help="Бюджет итераций RRT*")
    demo.add_argument("--population", type=int, help="Размер популяции NSGA-II")
    demo.add_argument("--generations", type=int, help="Число поколений NSGA-II")
    demo.add_argument("--no-plots", action="store_true", help="Сохранить числа без графиков и GIF")
    demo.add_argument("--method", choices=("full", "baseline"), default="full")
    demo.add_argument(
        "--views",
        nargs="+",
        choices=("isometric", "front", "side", "top"),
        default=["isometric"],
        help="Ракурсы для отдельных PNG и GIF",
    )
    demo.add_argument(
        "--frames", type=int, default=55, help="Число кадров GIF и интерактивного просмотра"
    )
    replay = subcommands.add_parser("replay", help="Проверить и показать сохранённый результат")
    replay.add_argument("directory", type=Path)
    replay.add_argument("--no-plots", action="store_true")
    replay.add_argument(
        "--views",
        nargs="+",
        choices=("isometric", "front", "side", "top"),
        default=["isometric"],
    )
    replay.add_argument("--frames", type=int, default=55)
    subcommands.add_parser("scenes", help="Показать каталог доступных 3D-сцен")
    showcase = subcommands.add_parser(
        "showcase", help="Анимации и четыре варианта для каждой сцены серии"
    )
    showcase.add_argument("directory", type=Path, help="Каталог завершённой серии экспериментов")
    showcase.add_argument("--output", type=Path, default=Path("results/showcase"))
    showcase.add_argument(
        "--seed", type=int, default=0, help="Предпочтительный демонстрационный seed"
    )
    showcase.add_argument("--frames", type=int, default=121)
    showcase.add_argument(
        "--views",
        nargs="+",
        choices=("isometric", "front", "side", "top"),
        default=["isometric", "top"],
    )
    showcase_add = subcommands.add_parser(
        "showcase-add", help="Добавить отдельный готовый запуск в галерею"
    )
    showcase_add.add_argument("directory", type=Path, help="Каталог успешного запуска demo")
    showcase_add.add_argument("--output", type=Path, default=Path("results/showcase"))
    showcase_add.add_argument("--scene", help="Имя сцены в галерее")
    showcase_add.add_argument("--frames", type=int, default=121)
    showcase_add.add_argument(
        "--views",
        nargs="+",
        choices=("isometric", "front", "side", "top"),
        default=["isometric", "side"],
    )
    showcase_stats = subcommands.add_parser(
        "showcase-statistics", help="Добавить статистику отдельной сцены в галерею"
    )
    showcase_stats.add_argument("directory", type=Path, help="Каталог завершённой парной серии")
    showcase_stats.add_argument("--output", type=Path, default=Path("results/showcase"))
    experiments = subcommands.add_parser("experiments", help="Парные пакетные эксперименты")
    experiments.add_argument(
        "--scenes", nargs="+", default=["simple", "scattered", "gate", "shelves", "cluttered"]
    )
    experiments.add_argument("--seeds", nargs="+", type=int, default=list(range(20)))
    experiments.add_argument("--output", type=Path, default=Path("results/experiments"))
    experiments.add_argument("--iterations", type=int, default=800)
    experiments.add_argument("--population", type=int, default=20)
    experiments.add_argument("--generations", type=int, default=12)
    experiments.add_argument("--workers", type=int, default=1, help="Число независимых процессов")
    experiments.add_argument(
        "--no-resume", action="store_true", help="Пересчитать сохранённые пары"
    )
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
                method=args.method,
                views=tuple(args.views),
                frame_count=args.frames,
            )
            run_pipeline(load_scene(args.scene), args.output, options)
        elif args.command == "replay":
            result = replay_results(
                args.directory,
                regenerate=not args.no_plots,
                views=tuple(args.views),
                frame_count=args.frames,
            )
            print(f"Повторно проверено решений: {len(result['solutions'])}")
        elif args.command == "scenes":
            from .scenes import list_scenes

            for item in list_scenes():
                print(
                    f"{item['name']}: {item['obstacle_count']} препятствий — {item['description']}"
                )
        elif args.command == "showcase":
            from .showcase import create_showcase

            create_showcase(
                args.directory,
                args.output,
                views=tuple(args.views),
                frame_count=args.frames,
                preferred_seed=args.seed,
            )
        elif args.command == "showcase-add":
            from .showcase import add_showcase_run

            add_showcase_run(
                args.directory,
                args.output,
                args.scene,
                views=tuple(args.views),
                frame_count=args.frames,
            )
        elif args.command == "showcase-statistics":
            from .showcase import add_showcase_statistics

            result = add_showcase_statistics(args.directory, args.output)
            print(f"Статистика добавлена: {result['viewer']}")
        elif args.command == "experiments":
            from .experiments import run_experiments

            options = PipelineOptions(
                rrt_iterations=args.iterations,
                population=args.population,
                generations=args.generations,
                plots=False,
            )
            run_experiments(
                args.scenes,
                args.seeds,
                args.output,
                options,
                resume=not args.no_resume,
                workers=args.workers,
            )
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    return 0
