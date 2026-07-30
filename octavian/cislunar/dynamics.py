"""ASSET equations of motion for dimensional CR3BP phases."""

from __future__ import annotations

from .._asset import oc, require_asset, vf
from .model import CR3BPSystem


class CR3BPODE(oc.ODEBase if oc is not None else object):
    """Dimensional barycentric-synodic CR3BP equations of motion.

    Args:
        system: Primary-secondary physical system.
    """

    def __init__(self, *, system: CR3BPSystem) -> None:
        require_asset("CR3BP dynamics")
        self.system = system
        arguments = oc.ODEArguments(6, 0)
        state = arguments.XVec()
        position = state.head(3)
        velocity = state.segment(3, 3)
        x = position[0]
        y = position[1]
        z = position[2]
        xdot = velocity[0]
        ydot = velocity[1]

        primary_x = float(system.primary_position_m[0])
        secondary_x = float(system.secondary_position_m[0])
        primary_displacement = vf.stack([x - primary_x, y, z])
        secondary_displacement = vf.stack([x - secondary_x, y, z])
        primary_distance = primary_displacement.norm()
        secondary_distance = secondary_displacement.norm()
        mean_motion = float(system.mean_motion_radps)

        acceleration = vf.stack(
            [
                2.0 * mean_motion * ydot + mean_motion**2 * x,
                -2.0 * mean_motion * xdot + mean_motion**2 * y,
                0.0 * z,
            ]
        )
        acceleration -= (
            float(system.primary.mu_m3ps2)  # type: ignore[union-attr]
            * primary_displacement
            / primary_distance**3
        )
        acceleration -= (
            float(system.secondary.mu_m3ps2)  # type: ignore[union-attr]
            * secondary_displacement
            / secondary_distance**3
        )
        ode = vf.stack([velocity, acceleration])
        groups = {
            ("R", "Position"): position,
            ("V", "Velocity"): velocity,
            ("t", "time"): arguments.TVar(),
            "RV": [0, 1, 2, 3, 4, 5],
        }
        super().__init__(ode, 6, 0, Vgroups=groups)
