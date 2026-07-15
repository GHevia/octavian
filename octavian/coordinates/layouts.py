"""Named state and control layouts used at the compiler boundary."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StateLayout:
    """Describe state/control columns without hard-coded solver indices.

    Group indices refer to either state or control columns. Time remains the
    ASSET phase-time column immediately after the state columns and is exposed
    through :attr:`time_column`.
    """

    name: str
    state_names: tuple[str, ...]
    control_names: tuple[str, ...] = ()
    state_groups: tuple[tuple[str, tuple[int, ...]], ...] = ()
    control_groups: tuple[tuple[str, tuple[int, ...]], ...] = ()

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("StateLayout.name must not be empty")
        _validate_names("state_names", self.state_names)
        _validate_names("control_names", self.control_names, allow_empty=True)
        _validate_groups("state_groups", self.state_groups, len(self.state_names))
        _validate_groups("control_groups", self.control_groups, len(self.control_names))

    @property
    def state_dim(self) -> int:
        """Return the number of differential states."""
        return len(self.state_names)

    @property
    def control_dim(self) -> int:
        """Return the number of controls."""
        return len(self.control_names)

    @property
    def time_column(self) -> int:
        """Return the trajectory column containing phase time."""
        return self.state_dim

    def state_indices(self, group: str) -> tuple[int, ...]:
        """Return state indices for a named semantic group."""
        return _find_group(self.state_groups, group, "state")

    def control_indices(self, group: str) -> tuple[int, ...]:
        """Return control indices for a named semantic group."""
        return _find_group(self.control_groups, group, "control")

    def public_rvt_columns(self) -> tuple[int, ...]:
        """Return position, velocity, and time columns for Cartesian reporting."""
        return self.state_indices("position") + self.state_indices("velocity") + (self.time_column,)


def _validate_names(label: str, names: tuple[str, ...], *, allow_empty: bool = False) -> None:
    if not names and not allow_empty:
        raise ValueError(f"StateLayout.{label} must not be empty")
    normalized = [str(name).strip() for name in names]
    if any(not name for name in normalized):
        raise ValueError(f"StateLayout.{label} entries must not be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"StateLayout.{label} entries must be unique")


def _validate_groups(
    label: str,
    groups: tuple[tuple[str, tuple[int, ...]], ...],
    dimension: int,
) -> None:
    names = [str(name).strip().lower() for name, _ in groups]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError(f"StateLayout.{label} names must be non-empty and unique")
    for name, indices in groups:
        if not indices:
            raise ValueError(f"StateLayout group {name!r} must contain at least one index")
        if len(set(indices)) != len(indices):
            raise ValueError(f"StateLayout group {name!r} contains duplicate indices")
        if any(index < 0 or index >= dimension for index in indices):
            raise ValueError(f"StateLayout group {name!r} contains an out-of-range index")


def _find_group(
    groups: tuple[tuple[str, tuple[int, ...]], ...],
    requested: str,
    kind: str,
) -> tuple[int, ...]:
    normalized = str(requested).strip().lower()
    for name, indices in groups:
        if name.lower() == normalized:
            return indices
    raise KeyError(f"Unknown {kind} group {requested!r}")


_CARTESIAN_STATES = ("rx", "ry", "rz", "vx", "vy", "vz")
_CARTESIAN_GROUPS = (
    ("position", (0, 1, 2)),
    ("velocity", (3, 4, 5)),
)

CARTESIAN = StateLayout(
    name="cartesian",
    state_names=_CARTESIAN_STATES,
    state_groups=_CARTESIAN_GROUPS,
)

CARTESIAN_MASS = StateLayout(
    name="cartesian_mass",
    state_names=(*_CARTESIAN_STATES, "mass"),
    state_groups=(*_CARTESIAN_GROUPS, ("mass", (6,))),
)

CARTESIAN_MASS_THRUST = StateLayout(
    name="cartesian_mass_thrust",
    state_names=(*_CARTESIAN_STATES, "mass"),
    control_names=("thrust_x", "thrust_y", "thrust_z"),
    state_groups=(*_CARTESIAN_GROUPS, ("mass", (6,))),
    control_groups=(("thrust", (0, 1, 2)),),
)
