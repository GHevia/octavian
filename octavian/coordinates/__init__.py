"""Coordinate frames, state layouts, and solver scaling declarations."""

from .frames import EARTH_INERTIAL, CoordinateFrame, inertial, lvlh
from .layouts import (
    CARTESIAN,
    CARTESIAN_MASS,
    CARTESIAN_MASS_THRUST,
    RELATIVE_CARTESIAN,
    StateLayout,
)
from .scaling import SolverScaling

__all__ = [
    "CARTESIAN",
    "CARTESIAN_MASS",
    "CARTESIAN_MASS_THRUST",
    "RELATIVE_CARTESIAN",
    "EARTH_INERTIAL",
    "CoordinateFrame",
    "SolverScaling",
    "StateLayout",
    "inertial",
    "lvlh",
]
