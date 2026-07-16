"""JSON and YAML input handling for mission configuration."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .errors import MissionConfigError


def read_config_file(path: str | Path) -> Mapping[str, Any]:
    """Read a JSON or YAML configuration file into a mapping.

    JSON support uses the Python standard library. YAML support is optional and
    uses ``yaml.safe_load`` from PyYAML; install it with ``octavian[yaml]``.

    Args:
        path: File ending in ``.json``, ``.yaml``, or ``.yml``.

    Returns:
        The parsed top-level mapping.

    Raises:
        MissionConfigError: If the extension, syntax, or top-level value is invalid.
    """
    config_path = Path(path)
    suffix = config_path.suffix.lower()
    try:
        source = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MissionConfigError(f"Could not read mission config {config_path}: {exc}") from exc

    try:
        if suffix == ".json":
            value = json.loads(source)
        elif suffix in {".yaml", ".yml"}:
            value = _read_yaml(source)
        else:
            raise MissionConfigError(
                f"Unsupported mission config extension {suffix!r}; use .json, .yaml, or .yml."
            )
    except MissionConfigError:
        raise
    except (json.JSONDecodeError, ValueError) as exc:
        raise MissionConfigError(f"Could not parse mission config {config_path}: {exc}") from exc

    if not isinstance(value, Mapping):
        raise MissionConfigError(
            f"Mission config {config_path} must contain a mapping at the top level."
        )
    return value


def _read_yaml(source: str) -> Any:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise MissionConfigError(
            "YAML mission files require PyYAML. Install Octavian with 'octavian[yaml]'."
        ) from exc

    try:
        return yaml.safe_load(source)
    except yaml.YAMLError as exc:
        raise MissionConfigError(f"Invalid YAML: {exc}") from exc
