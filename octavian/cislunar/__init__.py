"""Circular restricted three-body problem tools for cislunar design."""

from .dynamics import CR3BPODE
from .guessing import cr3bp_hermite_guess
from .model import CR3BPSystem
from .propagation import cr3bp_derivative, jacobi_constant, propagate_cr3bp
from .transforms import (
    dimensionalize_state,
    dimensionalize_time,
    inertial_to_synodic_state,
    nondimensionalize_state,
    nondimensionalize_time,
    synodic_to_inertial_state,
)

__all__ = [
    "CR3BPSystem",
    "CR3BPODE",
    "cr3bp_derivative",
    "cr3bp_hermite_guess",
    "dimensionalize_state",
    "dimensionalize_time",
    "inertial_to_synodic_state",
    "jacobi_constant",
    "nondimensionalize_state",
    "nondimensionalize_time",
    "propagate_cr3bp",
    "synodic_to_inertial_state",
]
