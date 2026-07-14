"""ASSET implementation of the Clohessy-Wiltshire equations of motion."""

from __future__ import annotations

from .._asset import oc, require_asset, vf


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
