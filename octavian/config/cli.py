"""Command-line runner for declarative mission files."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from . import load_mission


def main(argv: Sequence[str] | None = None) -> int:
    """Load, solve, and summarize one JSON or YAML mission file."""
    parser = argparse.ArgumentParser(
        prog="python -m octavian.config",
        description="Load and solve an Octavian JSON or YAML mission.",
    )
    parser.add_argument("path", help="Path to a .json, .yaml, or .yml mission file.")
    args = parser.parse_args(argv)

    mission = load_mission(args.path)
    solution = mission.solve()
    print(solution.summary())
    return 0 if solution.ok else 1
