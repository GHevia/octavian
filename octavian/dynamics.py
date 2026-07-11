"""Dynamics models.

Octavian's primary optimal-control backend is ASSET. To keep the rest of the
package importable (for utilities, studies, and result I/O), ASSET is treated as
an optional runtime dependency at import time.

If ASSET is not installed, constructing ASSET-backed dynamics will raise a
clear error, but importing this module will still succeed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ._asset import oc, require_asset, vf

SUN_MU_M3PS2 = 1.32712440018e20
MOON_MU_M3PS2 = 4.9048695e12


@dataclass(frozen=True)
class ThirdBodyTable:
    """ASSET interpolation table and gravity parameter for a third body."""

    name: str
    mu_m3ps2: float
    position_table: Any


def _require_asset() -> None:
    """Require ASSET before constructing a backend ODE."""
    require_asset("ASSET-backed dynamics")


def _point_mass_acceleration(position_vec, mu_m3ps2: float):
    """Return central-body point-mass acceleration for an ASSET position vector."""
    return (-float(mu_m3ps2)) * position_vec.normalized_power3()


def j2_acceleration_components(
    position_m,
    *,
    mu_m3ps2: float,
    radius_m: float = 6_378_136.3,
    j2: float = 1.08262668e-3,
):
    """Return the Cartesian J2 acceleration components for an ECI position.

    This follows the standard oblate-body acceleration model used by the ASSET
    vector-function implementation below, and is intentionally numpy-free so it
    can be used in lightweight validation tests.
    """
    x, y, z = [float(component) for component in position_m]
    radius_sq = x * x + y * y + z * z
    radius = radius_sq**0.5
    z_sq_over_r_sq = (z * z) / radius_sq
    scale = 1.5 * float(j2) * float(mu_m3ps2) * (float(radius_m) ** 2) / (radius**5)
    common_xy = 5.0 * z_sq_over_r_sq - 1.0
    z_term = 5.0 * z_sq_over_r_sq - 3.0
    return (
        scale * x * common_xy,
        scale * y * common_xy,
        scale * z * z_term,
    )


def _j2_acceleration(position_vec, *, mu_m3ps2: float, radius_m: float, j2: float):
    """Return J2 acceleration as an ASSET vector-function expression."""
    radius_sq = position_vec.dot(position_vec)
    radius = position_vec.norm()
    z = position_vec[2]
    z_sq_over_r_sq = (z * z) / radius_sq
    scale = 1.5 * float(j2) * float(mu_m3ps2) * (float(radius_m) ** 2) / (radius**5)
    common_xy = 5.0 * z_sq_over_r_sq - 1.0
    z_term = 5.0 * z_sq_over_r_sq - 3.0
    return vf.stack(
        [
            scale * position_vec[0] * common_xy,
            scale * position_vec[1] * common_xy,
            scale * z * z_term,
        ]
    )


def third_body_acceleration_components(
    spacecraft_position_m,
    body_position_m,
    *,
    mu_m3ps2: float,
):
    """Return third-body acceleration in an Earth-centered frame.

    The expression includes both the third body's gravity on the spacecraft
    and the same body's gravity on the Earth-centered frame origin. It mirrors
    the ASSET vector-function implementation used by coast and burn EOMs.
    """
    rx, ry, rz = [float(component) for component in spacecraft_position_m]
    bx, by, bz = [float(component) for component in body_position_m]

    rel_x = bx - rx
    rel_y = by - ry
    rel_z = bz - rz
    rel_radius = (rel_x * rel_x + rel_y * rel_y + rel_z * rel_z) ** 0.5
    body_radius = (bx * bx + by * by + bz * bz) ** 0.5

    mu = float(mu_m3ps2)
    return (
        mu * (rel_x / rel_radius**3 - bx / body_radius**3),
        mu * (rel_y / rel_radius**3 - by / body_radius**3),
        mu * (rel_z / rel_radius**3 - bz / body_radius**3),
    )


def _third_body_acceleration(position_vec, body_position_vec, *, mu_m3ps2: float):
    """Return third-body acceleration in an Earth-centered ASSET expression."""
    relative_to_spacecraft = body_position_vec - position_vec
    spacecraft_acceleration = relative_to_spacecraft.normalized_power3()
    frame_origin_acceleration = body_position_vec.normalized_power3()
    return float(mu_m3ps2) * (spacecraft_acceleration - frame_origin_acceleration)


def _gravity_acceleration(
    position_vec,
    *,
    mu_m3ps2: float,
    include_j2: bool = False,
    central_body_radius_m: float = 6_378_136.3,
    j2_coefficient: float = 1.08262668e-3,
    time_var=None,
    third_body_tables: Sequence[ThirdBodyTable] = (),
):
    """Compose central gravity and requested perturbation accelerations.

    ``time_var`` is needed only when third-body tables are supplied; it is used
    to query each body's interpolated Earth-centered position at the current
    mission-relative phase time.
    """
    acceleration = _point_mass_acceleration(position_vec, mu_m3ps2)
    if include_j2:
        acceleration = acceleration + _j2_acceleration(
            position_vec,
            mu_m3ps2=mu_m3ps2,
            radius_m=central_body_radius_m,
            j2=j2_coefficient,
        )
    if third_body_tables:
        if time_var is None:
            raise ValueError("Third-body dynamics require an ASSET time variable.")
        for body in third_body_tables:
            body_position = body.position_table(time_var)
            acceleration = acceleration + _third_body_acceleration(
                position_vec,
                body_position,
                mu_m3ps2=float(body.mu_m3ps2),
            )
    return acceleration


class TwoBodyECI(oc.ODEBase if oc is not None else object):
    """Two-body point-mass gravity in ECI.

    ODE state has 6 components: ``[r(3), v(3)]``.
    Time is the *phase time variable* (``ODEArguments.TVar()``), not a state component.

    Vgroups:
        - ``R``: position (3)
        - ``V``: velocity (3)
        - ``t``: phase time variable
    """

    def __init__(self, *, mu_m3ps2: float) -> None:
        _require_asset()
        self.mu = float(mu_m3ps2)

        XtU = oc.ODEArguments(6, 0)  # 6 states, 0 controls
        R = XtU.XVec().head(3)
        V = XtU.XVec().segment(3, 3)

        A = _gravity_acceleration(R, mu_m3ps2=self.mu)

        ode = vf.stack([V, A])

        Vgroups = {
            ("R", "Position"): R,
            ("V", "Velocity"): V,
            ("t", "time"): XtU.TVar(),
            "RV": [0, 1, 2, 3, 4, 5],
        }

        super().__init__(ode, 6, 0, Vgroups=Vgroups)


class PerturbedECI(oc.ODEBase if oc is not None else object):
    """Point-mass gravity with optional perturbation accelerations.

    Currently implemented perturbations:
        - J2 zonal harmonic
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
        _require_asset()
        self.mu = float(mu_m3ps2)

        XtU = oc.ODEArguments(6, 0)
        R = XtU.XVec().head(3)
        V = XtU.XVec().segment(3, 3)

        A = _gravity_acceleration(
            R,
            mu_m3ps2=self.mu,
            include_j2=bool(j2),
            central_body_radius_m=float(central_body_radius_m),
            j2_coefficient=float(j2_coefficient),
            time_var=XtU.TVar(),
            third_body_tables=tuple(third_body_tables),
        )
        ode = vf.stack([V, A])

        Vgroups = {
            ("R", "Position"): R,
            ("V", "Velocity"): V,
            ("t", "time"): XtU.TVar(),
            "RV": [0, 1, 2, 3, 4, 5],
        }

        super().__init__(ode, 6, 0, Vgroups=Vgroups)


class MassCoastECI(oc.ODEBase if oc is not None else object):
    """Ballistic coast dynamics carrying spacecraft mass as a constant state."""

    def __init__(
        self,
        *,
        mu_m3ps2: float,
        j2: bool = False,
        central_body_radius_m: float = 6_378_136.3,
        j2_coefficient: float = 1.08262668e-3,
        third_body_tables: Sequence[ThirdBodyTable] = (),
    ) -> None:
        _require_asset()
        self.mu = float(mu_m3ps2)

        XtU = oc.ODEArguments(7, 0)
        X = XtU.XVec()
        R = X.head(3)
        V = X.segment(3, 3)
        M = X[6]

        A = _gravity_acceleration(
            R,
            mu_m3ps2=self.mu,
            include_j2=bool(j2),
            central_body_radius_m=float(central_body_radius_m),
            j2_coefficient=float(j2_coefficient),
            time_var=XtU.TVar(),
            third_body_tables=tuple(third_body_tables),
        )
        ode = vf.stack([V, A, M * 0.0])

        Vgroups = {
            ("R", "Position"): R,
            ("V", "Velocity"): V,
            ("M", "Mass"): [6],
            ("t", "time"): XtU.TVar(),
            "RV": [0, 1, 2, 3, 4, 5],
        }

        super().__init__(ode, 7, 0, Vgroups=Vgroups)


class ChemicalBurnECI(oc.ODEBase if oc is not None else object):
    """Finite chemical burn dynamics with three thrust-direction controls.

    ODE state is ``[r(3), v(3), m]`` and controls are a dimensionless thrust
    direction/throttle vector. The compiler bounds the control norm to one.
    """

    def __init__(
        self,
        *,
        mu_m3ps2: float,
        thrust_N: float,
        isp_s: float,
        j2: bool = False,
        central_body_radius_m: float = 6_378_136.3,
        j2_coefficient: float = 1.08262668e-3,
        third_body_tables: Sequence[ThirdBodyTable] = (),
        g0_mps2: float = 9.80665,
    ) -> None:
        _require_asset()
        self.mu = float(mu_m3ps2)
        self.thrust_N = float(thrust_N)
        self.isp_s = float(isp_s)
        self.g0_mps2 = float(g0_mps2)

        if self.thrust_N <= 0.0:
            raise ValueError("ChemicalBurnECI requires thrust_N > 0.")
        if self.isp_s <= 0.0:
            raise ValueError("ChemicalBurnECI requires isp_s > 0.")

        XtU = oc.ODEArguments(7, 3)
        X = XtU.XVec()
        U = XtU.UVec()
        R = X.head(3)
        V = X.segment(3, 3)
        M = X[6]

        gravity = _gravity_acceleration(
            R,
            mu_m3ps2=self.mu,
            include_j2=bool(j2),
            central_body_radius_m=float(central_body_radius_m),
            j2_coefficient=float(j2_coefficient),
            time_var=XtU.TVar(),
            third_body_tables=tuple(third_body_tables),
        )
        thrust_acceleration = (self.thrust_N / M) * U
        mass_flow = -(self.thrust_N / (self.isp_s * self.g0_mps2)) * U.norm()
        ode = vf.stack([V, gravity + thrust_acceleration, mass_flow])

        Vgroups = {
            ("R", "Position"): R,
            ("V", "Velocity"): V,
            ("M", "Mass"): [6],
            ("U", "Control", "Throttle"): U,
            ("t", "time"): XtU.TVar(),
            "RV": [0, 1, 2, 3, 4, 5],
        }

        super().__init__(ode, 7, 3, Vgroups=Vgroups)
