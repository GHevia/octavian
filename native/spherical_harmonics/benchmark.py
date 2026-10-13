"""Run the plain-Python runtime comparison for native backend development."""

import runpy
from pathlib import Path

runpy.run_path(
    str(
        Path(__file__).resolve().parents[2] / "native/spherical_harmonics/runtime_comparison.py"
    ),
    run_name="__main__",
)
