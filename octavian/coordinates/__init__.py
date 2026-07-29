"""Coordinate frames, state layouts, and solver scaling declarations."""

from .frames import EARTH_INERTIAL, CoordinateFrame, inertial, lvlh, ric
from .layouts import (
    CARTESIAN,
    CARTESIAN_MASS,
    CARTESIAN_MASS_THRUST,
    CLASSICAL_RELATIVE_ELEMENTS,
    COUPLED_RELATIVE_ECI,
    COUPLED_RELATIVE_ECI_MASS,
    COUPLED_RELATIVE_ECI_MASS_THRUST,
    COUPLED_RELATIVE_RIC,
    DAMICO_RELATIVE_ELEMENTS,
    RELATIVE_CARTESIAN,
    StateLayout,
)
from .scaling import SolverScaling

__all__ = [
    "CARTESIAN",
    "CARTESIAN_MASS",
    "CARTESIAN_MASS_THRUST",
    "CLASSICAL_RELATIVE_ELEMENTS",
    "COUPLED_RELATIVE_ECI",
    "COUPLED_RELATIVE_ECI_MASS",
    "COUPLED_RELATIVE_ECI_MASS_THRUST",
    "COUPLED_RELATIVE_RIC",
    "DAMICO_RELATIVE_ELEMENTS",
    "RELATIVE_CARTESIAN",
    "EARTH_INERTIAL",
    "CoordinateFrame",
    "SolverScaling",
    "StateLayout",
    "inertial",
    "lvlh",
    "ric",
]
