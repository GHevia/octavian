"""ASSET implementations of unforced and differentially perturbed CWH motion."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .._asset import oc, require_asset, vf
from ..dynamics import (
    ThirdBodyTable,
    _j2_acceleration,
    _third_body_acceleration,
)


@dataclass(frozen=True)
class RelativeReferenceTable:
    """ASSET interpolation tables for a prescribed circular chief reference."""

    chief_position_table: Any
    inertial_to_ric_table: Any


class ClohessyWiltshireODE(oc.ODEBase if oc is not None else object):
    """Six-state unforced CWH dynamics in a chief-centered LVLH frame."""

    def __init__(self, *, mean_motion_radps: float) -> None:
        require_asset("Clohessy-Wiltshire dynamics")
        self.mean_motion_radps = float(mean_motion_radps)
        if self.mean_motion_radps <= 0.0:
            raise ValueError("mean_motion_radps must be positive")

        arguments = oc.ODEArguments(6, 0)
        state = arguments.XVec()
        position = state.head(3)
        velocity = state.segment(3, 3)
        x = position[0]
        z = position[2]
        xdot = velocity[0]
        ydot = velocity[1]
        n = self.mean_motion_radps

        acceleration = vf.stack(
            [
                3.0 * n**2 * x + 2.0 * n * ydot,
                -2.0 * n * xdot,
                -(n**2) * z,
            ]
        )
        ode = vf.stack([velocity, acceleration])
        groups = {
            ("R", "Position", "RelativePosition"): position,
            ("V", "Velocity", "RelativeVelocity"): velocity,
            ("t", "time"): arguments.TVar(),
            "RV": [0, 1, 2, 3, 4, 5],
        }
        super().__init__(ode, 6, 0, Vgroups=groups)


class PerturbedClohessyWiltshireODE(oc.ODEBase if oc is not None else object):
    """CWH dynamics with differential J2 and third-body acceleration.

    The six optimized states remain relative RIC position and velocity.  A
    prescribed circular chief reference supplies inertial position and RIC
    orientation as functions of time.  Perturbing acceleration is evaluated at
    both chief and reconstructed deputy positions, differenced, and rotated
    into RIC before being added to the linear CWH acceleration.
    """

    def __init__(
        self,
        *,
        mean_motion_radps: float,
        reference_table: RelativeReferenceTable,
        j2: bool = False,
        mu_m3ps2: float,
        central_body_radius_m: float,
        j2_coefficient: float,
        third_body_tables: Sequence[ThirdBodyTable] = (),
    ) -> None:
        require_asset("perturbed Clohessy-Wiltshire dynamics")
        self.mean_motion_radps = float(mean_motion_radps)
        if self.mean_motion_radps <= 0.0:
            raise ValueError("mean_motion_radps must be positive")

        arguments = oc.ODEArguments(6, 0)
        state = arguments.XVec()
        position = state.head(3)
        velocity = state.segment(3, 3)
        time = arguments.TVar()
        x = position[0]
        z = position[2]
        xdot = velocity[0]
        ydot = velocity[1]
        n = self.mean_motion_radps
        acceleration = vf.stack(
            [
                3.0 * n**2 * x + 2.0 * n * ydot,
                -2.0 * n * xdot,
                -(n**2) * z,
            ]
        )

        chief_position = reference_table.chief_position_table(time)
        basis = reference_table.inertial_to_ric_table(time)
        deputy_position = chief_position + _ric_to_inertial(position, basis)
        differential_acceleration = position * 0.0
        if j2:
            chief_j2 = _j2_acceleration(
                chief_position,
                mu_m3ps2=float(mu_m3ps2),
                radius_m=float(central_body_radius_m),
                j2=float(j2_coefficient),
            )
            deputy_j2 = _j2_acceleration(
                deputy_position,
                mu_m3ps2=float(mu_m3ps2),
                radius_m=float(central_body_radius_m),
                j2=float(j2_coefficient),
            )
            differential_acceleration = differential_acceleration + (
                deputy_j2 - chief_j2
            )
        for body in third_body_tables:
            body_position = body.position_table(time)
            chief_third_body = _third_body_acceleration(
                chief_position,
                body_position,
                mu_m3ps2=float(body.mu_m3ps2),
            )
            deputy_third_body = _third_body_acceleration(
                deputy_position,
                body_position,
                mu_m3ps2=float(body.mu_m3ps2),
            )
            differential_acceleration = differential_acceleration + (
                deputy_third_body - chief_third_body
            )
        if j2 or third_body_tables:
            acceleration = acceleration + _inertial_to_ric(
                differential_acceleration,
                basis,
            )

        ode = vf.stack([velocity, acceleration])
        groups = {
            ("R", "Position", "RelativePosition"): position,
            ("V", "Velocity", "RelativeVelocity"): velocity,
            ("t", "time"): time,
            "RV": [0, 1, 2, 3, 4, 5],
        }
        super().__init__(ode, 6, 0, Vgroups=groups)


def _ric_to_inertial(vector, basis):
    """Apply the transpose of a flattened row-major inertial-to-RIC matrix."""
    return vf.stack(
        [
            basis[0] * vector[0] + basis[3] * vector[1] + basis[6] * vector[2],
            basis[1] * vector[0] + basis[4] * vector[1] + basis[7] * vector[2],
            basis[2] * vector[0] + basis[5] * vector[1] + basis[8] * vector[2],
        ]
    )


def _inertial_to_ric(vector, basis):
    """Apply a flattened row-major inertial-to-RIC matrix."""
    return vf.stack(
        [
            basis[0] * vector[0] + basis[1] * vector[1] + basis[2] * vector[2],
            basis[3] * vector[0] + basis[4] * vector[1] + basis[5] * vector[2],
            basis[6] * vector[0] + basis[7] * vector[1] + basis[8] * vector[2],
        ]
    )
