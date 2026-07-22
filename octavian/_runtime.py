"""Small runtime adjustments required by native optional dependencies.

The ASSET wheel installs the Intel OpenMP runtime in ``<prefix>/Library/bin``
on Windows.  Conda adds that directory to ``PATH`` during activation, whereas
a standard virtual environment does not.  Python 3.8+ provides a narrower
per-process DLL search path that lets Octavian support both environments
without changing the user's system-wide ``PATH``.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any


def enable_native_runtime(
    *,
    prefix: str | Path | None = None,
    add_dll_directory: Callable[[str], Any] | None = None,
) -> tuple[Any, ...]:
    """Add the active environment's native-runtime directory on Windows.

    The returned handles must remain alive for the duration of the process.
    On platforms without :func:`os.add_dll_directory`, or when the directory
    does not exist, this is intentionally a no-op.
    """
    environment_prefix = Path(sys.prefix if prefix is None else prefix)
    runtime_directory = environment_prefix / "Library" / "bin"
    if not runtime_directory.is_dir():
        return ()

    loader = add_dll_directory or getattr(os, "add_dll_directory", None)
    if loader is None:
        return ()

    return (loader(str(runtime_directory)),)
