"""Run the plain-Python runtime comparison from its examples location."""

import runpy
from pathlib import Path

runpy.run_path(
    str(
        Path(__file__).resolve().parents[2] / "examples/analysis/03_spherical_harmonics_runtime.py"
    ),
    run_name="__main__",
)
