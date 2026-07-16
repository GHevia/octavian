"""Small validation and reference helpers shared by config parsers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..phase import state
from ..specs import BoundaryState
from .errors import MissionConfigError


def mapping(value: Any, path: str) -> Mapping[str, Any]:
    """Require a mapping with string keys."""
    if not isinstance(value, Mapping):
        raise MissionConfigError(f"{path} must be a mapping.")
    if any(not isinstance(key, str) for key in value):
        raise MissionConfigError(f"{path} keys must be strings.")
    return value


def sequence(value: Any, path: str) -> list[Any]:
    """Require a JSON/YAML-style list value."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise MissionConfigError(f"{path} must be a list.")
    return list(value)


def required(config: Mapping[str, Any], key: str, path: str) -> Any:
    """Return one required key or raise with its full config path."""
    if key not in config:
        raise MissionConfigError(f"{path}.{key} is required.")
    return config[key]


def reject_unknown(config: Mapping[str, Any], allowed: set[str], path: str) -> None:
    """Reject misspelled or unsupported keys."""
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise MissionConfigError(f"{path} contains unknown key(s): {', '.join(unknown)}.")


def normalized_type(value: Any) -> str:
    """Normalize declaration type spellings to lowercase snake case."""
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def boolean(value: Any, path: str) -> bool:
    """Require an actual JSON/YAML boolean instead of truthy text or numbers."""
    if not isinstance(value, bool):
        raise MissionConfigError(f"{path} must be true or false.")
    return value


def pair(value: Any, path: str) -> tuple[Any, Any]:
    """Require a list containing exactly two values."""
    items = sequence(value, path)
    if len(items) != 2:
        raise MissionConfigError(f"{path} must contain exactly two values.")
    return items[0], items[1]


def optional_pair(value: Any, path: str) -> tuple[float, float] | None:
    """Parse an optional two-scalar tuple."""
    if value is None:
        return None
    first, second = pair(value, path)
    return float(first), float(second)


def reference(value: Any, registry: Mapping[str, Any], path: str, kind: str) -> Any:
    """Resolve a named object and include available names in failures."""
    if not isinstance(value, str):
        raise MissionConfigError(f"{path} must be a named {kind} reference.")
    try:
        return registry[value]
    except KeyError as exc:
        available = ", ".join(sorted(registry)) or "none"
        raise MissionConfigError(
            f"{path} references unknown {kind} {value!r}; available: {available}."
        ) from exc


def optional_reference(
    value: Any, registry: Mapping[str, Any], path: str, kind: str
) -> Any | None:
    """Resolve an optional named object."""
    if value is None:
        return None
    return reference(value, registry, path, kind)


def state_reference(
    value: Any, states: Mapping[str, BoundaryState], path: str
) -> BoundaryState:
    """Resolve a named state or parse an explicit inline state."""
    if isinstance(value, str):
        return reference(value, states, path, "state")
    config = mapping(value, path)
    reject_unknown(config, {"r_m", "v_mps"}, path)
    return state(required(config, "r_m", path), required(config, "v_mps", path))


def optional_state(
    value: Any, states: Mapping[str, BoundaryState], path: str
) -> BoundaryState | None:
    """Resolve an optional named or inline state."""
    if value is None:
        return None
    return state_reference(value, states, path)
