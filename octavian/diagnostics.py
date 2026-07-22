"""Runtime-environment diagnostics for ASSET-backed Octavian workflows.

The solver depends on a compiled ASSET extension. A Python environment can find
``asset_asrl`` through a global user-site directory while loading its DLLs from
some unrelated environment, which is not a reproducible installation. This
module detects that arrangement before a user starts a long optimization.
"""

from __future__ import annotations

import argparse
import importlib
import site
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


@dataclass(frozen=True, slots=True)
class RuntimeCheck:
    """One environment diagnostic and its user-facing result."""

    name: str
    ok: bool
    detail: str


def path_is_within(path: str | Path, directory: str | Path) -> bool:
    """Return whether ``path`` is contained by ``directory`` after resolution."""
    try:
        Path(path).resolve().relative_to(Path(directory).resolve())
    except ValueError:
        return False
    return True


def runtime_checks() -> tuple[RuntimeCheck, ...]:
    """Inspect whether ASSET is loaded from the active isolated environment."""
    checks: list[RuntimeCheck] = [
        RuntimeCheck(
            name="user site disabled",
            ok=not site.ENABLE_USER_SITE,
            detail=(
                "PYTHONNOUSERSITE is active."
                if not site.ENABLE_USER_SITE
                else "Python can import packages from the global user site."
            ),
        )
    ]

    for module_name in ("asset", "asset_asrl"):
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - native import failures vary by platform
            checks.append(
                RuntimeCheck(
                    name=f"{module_name} import",
                    ok=False,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        module_path = Path(str(getattr(module, "__file__", ""))).resolve()
        checks.append(
            RuntimeCheck(
                name=f"{module_name} location",
                ok=path_is_within(module_path, sys.prefix),
                detail=str(module_path),
            )
        )

    return tuple(checks)


def format_runtime_report(checks: tuple[RuntimeCheck, ...] | None = None) -> str:
    """Return a readable report for the active Python runtime."""
    selected_checks = runtime_checks() if checks is None else checks
    lines = ["Octavian runtime diagnostics", f"  Python: {sys.executable}", f"  Prefix: {sys.prefix}"]
    for check in selected_checks:
        status = "PASS" if check.ok else "FAIL"
        lines.append(f"  [{status}] {check.name}: {check.detail}")
    return "\n".join(lines)


def main(argv: list[str] | None = None, *, stream: TextIO | None = None) -> int:
    """Print runtime diagnostics and return nonzero when isolation is incomplete."""
    parser = argparse.ArgumentParser(
        prog="python -m octavian.diagnostics",
        description="Verify that ASSET is loaded from the active Python environment.",
    )
    parser.parse_args(argv)

    checks = runtime_checks()
    output = sys.stdout if stream is None else stream
    print(format_runtime_report(checks), file=output)
    return 0 if all(check.ok for check in checks) else 1


if __name__ == "__main__":  # pragma: no cover - exercised through module execution
    raise SystemExit(main())
