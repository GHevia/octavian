"""ASSET dynamics for simplified and full nonlinear relative motion."""

from __future__ import annotations

from collections.abc import Sequence

from .._asset import oc, require_asset, vf
from ..dynamics import ThirdBodyTable, _gravity_acceleration


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


class CoupledRelativeODE(oc.ODEBase if oc is not None else object):
    """Propagate chief and deputy absolute states under one full force model.

    ODE states are ``[chief_r, chief_v, deputy_r, deputy_v]`` in ECI.  RIC
    conversion belongs to compiler/reporting services, keeping the equations
    of motion free of coordinate singularities and avoiding any relative
    gravity linearization.
    """

    def __init__(
        self,
        *,
        mu_m3ps2: float,
        j2: bool = False,
        central_body_radius_m: float = 6_378_136.3,
        j2_coefficient: float = 1.08262668e-3,
        third_body_tables: Sequence[ThirdBodyTable] = (),
    ) -> None:
        require_asset("nonlinear relative dynamics")
        arguments = oc.ODEArguments(12, 0)
        state = arguments.XVec()
        chief_position = state.head(3)
        chief_velocity = state.segment(3, 3)
        deputy_position = state.segment(6, 3)
        deputy_velocity = state.segment(9, 3)
        time = arguments.TVar()
        force_options = {
            "mu_m3ps2": float(mu_m3ps2),
            "include_j2": bool(j2),
            "central_body_radius_m": float(central_body_radius_m),
            "j2_coefficient": float(j2_coefficient),
            "time_var": time,
            "third_body_tables": tuple(third_body_tables),
        }
        chief_acceleration = _gravity_acceleration(
            chief_position,
            **force_options,
        )
        deputy_acceleration = _gravity_acceleration(
            deputy_position,
            **force_options,
        )
        ode = vf.stack(
            [
                chief_velocity,
                chief_acceleration,
                deputy_velocity,
                deputy_acceleration,
            ]
        )
        groups = {
            ("ChiefR", "ChiefPosition"): chief_position,
            ("ChiefV", "ChiefVelocity"): chief_velocity,
            ("DeputyR", "DeputyPosition"): deputy_position,
            ("DeputyV", "DeputyVelocity"): deputy_velocity,
            ("t", "time"): time,
        }
        super().__init__(ode, 12, 0, Vgroups=groups)
