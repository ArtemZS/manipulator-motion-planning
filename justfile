set windows-shell := ["pwsh", "-NoLogo", "-NoProfile", "-Command"]

export UV_CACHE_DIR := justfile_directory() / ".cache/uv"
export MPLCONFIGDIR := justfile_directory() / ".cache/matplotlib"

default:
    @just --list

sync:
    uv sync --locked

test:
    uv run pytest

check:
    uv run ruff check .
    uv run ruff format --check .

format:
    uv run ruff format .

demo:
    uv run manipulator-planning demo --output results/demo

quick:
    uv run manipulator-planning demo --quick --output results/quick

replay directory="results/demo":
    uv run manipulator-planning replay "{{directory}}"

build:
    uv build
