"""Declarative JSON and YAML mission configuration.

The config layer only constructs Octavian's existing Python objects. Python
mission scripts remain the primary and most expressive interface.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..mission import Mission
from .builder import mission_from_dict
from .errors import MissionConfigError
from .io import read_config_file


def load_mission(path: str | Path) -> Mission:
    """Load a JSON or YAML file and construct its :class:`~octavian.Mission`.

    Args:
        path: Path to a ``.json``, ``.yaml``, or ``.yml`` mission file.

    Returns:
        A validated mission built from the public Python configuration objects.
    """
    return mission_from_dict(read_config_file(path))


def load_mission_mapping(value: Mapping[str, Any]) -> Mission:
    """Construct a mission from an already-parsed configuration mapping."""
    return mission_from_dict(value)


__all__ = [
    "MissionConfigError",
    "load_mission",
    "load_mission_mapping",
    "mission_from_dict",
    "read_config_file",
]
