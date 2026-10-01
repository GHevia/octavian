"""Run the shipped tutorials with real solvers and isolated output directories."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("asset_asrl", exc_type=ImportError)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = sorted((ROOT / "examples").rglob("*.py")) + sorted(
    (ROOT / "examples" / "config").glob("*.json")
)


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda path: str(path.relative_to(ROOT)))
def test_executable_example(example: Path, tmp_path: Path) -> None:
    if example.name in {"02_spherical_harmonics.py", "03_spherical_harmonics_runtime.py"} and not any(
        importlib.util.find_spec(name) is not None
        for name in ("octavian.octavian_harmonics_native", "octavian_harmonics_native")
    ):
        pytest.skip("This example selects the native backend; tested in native-wheel CI")
    command = [sys.executable]
    if example.suffix == ".json":
        command.extend(["-m", "octavian.config"])
    command.append(str(example))
    environment = {
        **os.environ,
        "MPLBACKEND": "Agg",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(ROOT),
        "PYTHONUTF8": "1",
    }
    log_path = tmp_path / "example.log"
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=tmp_path,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=300,
            check=False,
        )
    output = log_path.read_text(encoding="utf-8", errors="replace")
    assert result.returncode == 0, f"{example.relative_to(ROOT)} failed:\n{output[-16000:]}"
    assert "Octavian result: NOT CONVERGED" not in output, output[-16000:]
    assert "Octavian solution: FAILED" not in output, output[-16000:]
