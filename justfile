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

demo scene="simple" output="results/demo":
    uv run manipulator-planning demo --scene "{{scene}}" --output "{{output}}"

quick:
    uv run manipulator-planning demo --quick --output results/quick

replay directory="results/demo":
    uv run manipulator-planning replay "{{directory}}"

scenes:
    uv run manipulator-planning scenes

views directory="results/demo":
    uv run manipulator-planning replay "{{directory}}" --views isometric front side top

experiments:
    uv run manipulator-planning experiments --workers 2

showcase series="results/experiments_current" output="results/showcase":
    uv run manipulator-planning showcase "{{series}}" --output "{{output}}"

showcase-add directory output="results/showcase":
    uv run manipulator-planning showcase-add "{{directory}}" --output "{{output}}"

showcase-statistics directory output="results/showcase":
    uv run manipulator-planning showcase-statistics "{{directory}}" --output "{{output}}"

serve port="8766":
    uv run python -m http.server {{port}} --bind 127.0.0.1 --directory results

build:
    uv build
