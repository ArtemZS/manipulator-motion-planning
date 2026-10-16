"""Экспорт сохранённой галереи в автономный статический сайт GitHub Pages."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import posixpath
import re
import shutil
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

STATISTICS_FILES = (
    "index.html",
    "REPORT.md",
    "runs.csv",
    "summary.csv",
    "summary.json",
    "statistics.png",
)
SCENE_FILES = ("view.html", "interactive.html", "comparison.png", "pareto.png", "variants.csv")


class PageLinks(HTMLParser):
    """Собирает ссылки на файлы, не разбирая содержимое JavaScript."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name in {"href", "src"} and value:
                self.links.append(value)


def validate_links(root: Path) -> int:
    """Проверяет, что локальные ссылки существуют и остаются внутри сайта."""
    root = root.resolve()
    count = 0
    for page in root.rglob("*.html"):
        parser = PageLinks()
        parser.feed(page.read_text(encoding="utf-8"))
        for link in parser.links:
            parts = urlsplit(link)
            if parts.scheme or parts.netloc or not parts.path:
                continue
            if parts.path.startswith("/"):
                raise ValueError(f"Ссылка не учитывает подпапку репозитория: {page}: {link}")
            target = (page.parent / unquote(parts.path)).resolve()
            if not target.is_relative_to(root) or not target.is_file():
                raise ValueError(f"Недоступная ссылка: {page}: {link}")
            count += 1
    return count


def export_showcase(source: Path, statistics: Path, output: Path) -> dict:
    """Копирует публичные материалы, отделяя изображения и общий код Plotly."""
    source, statistics, output = source.resolve(), statistics.resolve(), output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Выходная папка должна быть пустой: используйте новый каталог экспорта")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    files = [(source / "index.html", Path("showcase/index.html"))]
    files.append((source / "variants.csv", Path("showcase/variants.csv")))
    scenes = [entry["scene"] for entry in manifest["scenes"]]
    for scene in scenes:
        if re.fullmatch(r"[a-z0-9_]+", scene) is None:
            raise ValueError(f"Некорректное имя сцены: {scene}")
        files.extend(
            (source / scene / name, Path("showcase") / scene / name) for name in SCENE_FILES
        )
        if (source / scene / "statistics").is_dir():
            files.extend(
                (
                    source / scene / "statistics" / name,
                    Path("showcase") / scene / "statistics" / name,
                )
                for name in STATISTICS_FILES
            )
    files.extend(
        (statistics / name, Path("experiments") / name)
        for name in (*STATISTICS_FILES, "provenance.json")
    )
    for original, _ in files:
        if not original.is_file():
            raise FileNotFoundError(original)
    output.mkdir(parents=True, exist_ok=True)

    def asset(data: bytes, suffix: str, page: Path) -> str:
        """Сохраняет одинаковые ресурсы один раз и возвращает относительный URL."""
        name = Path("assets") / f"{hashlib.sha256(data).hexdigest()[:20]}.{suffix}"
        target = output / name
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            target.write_bytes(data)
        return posixpath.relpath(name.as_posix(), page.parent.as_posix())

    for original, relative in files:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if original.suffix != ".html":
            shutil.copyfile(original, target)
            continue
        html = original.read_text(encoding="utf-8")
        if relative == Path("showcase/index.html"):
            html = re.sub(
                r'href="\.\./[^"<>]+/index\.html"', 'href="../experiments/index.html"', html
            )

        def extract_image(match: re.Match, page: Path = relative) -> str:
            return asset(base64.b64decode(match[2], validate=True), match[1], page)

        html = re.sub(r"data:image/(png|gif|jpeg);base64,([A-Za-z0-9+/=]+)", extract_image, html)

        def extract_plotly(match: re.Match, page: Path = relative) -> str:
            code = match[1]
            if "* plotly.js v" not in code[:200]:
                return match[0]
            url = asset(code.encode("utf-8"), "js", page)
            return f'<script src="{url}"></script>'

        html = re.sub(r"<script[^>]*>(.*?)</script>", extract_plotly, html, flags=re.DOTALL)
        target.write_text(html, encoding="utf-8", newline="\n")
    (output / ".nojekyll").touch()
    (output / "index.html").write_text(
        '<!doctype html><html lang="ru"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta http-equiv="refresh" content="0;url=showcase/index.html">'
        "<title>Манипулятор 6R — галерея</title>"
        '<a href="showcase/index.html">Открыть галерею шести сцен</a></html>',
        encoding="utf-8",
    )
    links = validate_links(output)
    paths = [path for path in output.rglob("*") if path.is_file()]
    return {
        "scenes": scenes,
        "files": len(paths),
        "links": links,
        "size_mb": round(sum(path.stat().st_size for path in paths) / 2**20, 2),
    }


def main() -> None:
    """Разбирает параметры и выводит итог проверки статического сайта."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/showcase"))
    parser.add_argument("--statistics", type=Path, default=Path("docs/experiments"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            export_showcase(args.source, args.statistics, args.output), ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
