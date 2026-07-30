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

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._asset import oc, require_asset, vf
from ._control_dynamics import thrust_vector_and_rate
from .bodies import MOON, SUN
from .control import ThrustControl

SUN_MU_M3PS2 = SUN.mu_m3ps2
MOON_MU_M3PS2 = MOON.mu_m3ps2


@dataclass(frozen=True)
class ThirdBodyTable:
    """Interpolated third-body ephemeris used by ASSET dynamics.

    Attributes:
        name: Normalized body name, currently ``"sun"`` or ``"moon"``.
        mu_m3ps2: Gravitational parameter for the third body in SI units.
        position_table: ASSET vector function that returns the body's
            Earth-centered position in meters at the current phase time.
        times_s: Optional numeric sample times used for result conversion.
        positions_eci_m: Optional numeric positions paired with ``times_s``.
    """

    name: str
    mu_m3ps2: float
    position_table: Any
    times_s: NDArray[np.float64] | None = None
    positions_eci_m: NDArray[np.float64] | None = None

    def position_at(self, time_s: float) -> NDArray[np.float64]:
        """Interpolate the numeric Earth-centered position at ``time_s``."""
        if self.times_s is None or self.positions_eci_m is None:
            raise ValueError(f"ThirdBodyTable {self.name!r} does not contain numeric samples")
        times = np.asarray(self.times_s, dtype=float)
        positions = np.asarray(self.positions_eci_m, dtype=float)
        if float(time_s) < times[0] or float(time_s) > times[-1]:
            raise ValueError(f"Requested {self.name} position lies outside the sampled time range")
        return np.asarray(
            [np.interp(float(time_s), times, positions[:, component]) for component in range(3)],
            dtype=float,
        )


def _require_asset() -> None:
    """Require ASSET before constructing a backend ODE."""
    require_asset("ASSET-backed dynamics")


def _point_mass_acceleration(position_vec, mu_m3ps2: float):
    """Return central-body point-mass acceleration for an ASSET position vector.

    ASSET vector objects expose ``normalized_power3()``, which evaluates
    ``r / |r|^3``. Multiplying by ``-mu`` gives the standard two-body
    acceleration toward the frame origin.
    """
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
    """Return J2 acceleration as an ASSET vector-function expression.

    This mirrors :func:`j2_acceleration_components`, but it is built from ASSET
    symbolic vector operations so it can be embedded directly in an ODE.
    """
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


def gravity_acceleration_components(
    position_m: ArrayLike,
    *,
    time_s: float = 0.0,
    mu_m3ps2: float,
    include_j2: bool = False,
    central_body_radius_m: float = 6_378_136.3,
    j2_coefficient: float = 1.08262668e-3,
    third_body_tables: Sequence[ThirdBodyTable] = (),
) -> NDArray[np.float64]:
    """Evaluate the full configured acceleration numerically.

    This is the numeric counterpart of :func:`_gravity_acceleration`.  It is
    used when converting solved coupled chief/deputy histories to the
    instantaneous RIC frame, where the chief acceleration determines the
    frame's cross-track angular rate.
    """
    position = np.asarray(position_m, dtype=float).reshape(3)
    radius = float(np.linalg.norm(position))
    if radius <= 0.0:
        raise ValueError("position_m must have non-zero norm")
    acceleration = -float(mu_m3ps2) * position / radius**3
    if include_j2:
        acceleration = acceleration + np.asarray(
            j2_acceleration_components(
                position,
                mu_m3ps2=float(mu_m3ps2),
                radius_m=float(central_body_radius_m),
                j2=float(j2_coefficient),
            ),
            dtype=float,
        )
    for body in third_body_tables:
        acceleration = acceleration + np.asarray(
            third_body_acceleration_components(
                position,
                body.position_at(float(time_s)),
                mu_m3ps2=float(body.mu_m3ps2),
            ),
            dtype=float,
        )
    return np.asarray(acceleration, dtype=float)


def _third_body_acceleration(position_vec, body_position_vec, *, mu_m3ps2: float):
    """Return third-body acceleration in an Earth-centered ASSET expression.

    The spacecraft feels gravity from the third body at ``body - spacecraft``.
    Because Octavian's frame is Earth-centered and non-inertial under the third
    body's pull, the acceleration of the Earth-centered frame origin is
    subtracted as ``body / |body|^3``.
    """
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
        - Sun and Moon third-body point-mass gravity through interpolation tables

    The state is ``[r(3), v(3)]``. Time remains ASSET's phase time variable, not
    a state, which is why third-body tables are sampled with ``XtU.TVar()``.
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
    """Ballistic coast dynamics carrying spacecraft mass as a constant state.

    This ODE is used between finite-burn phases so ASSET continuity links can
    preserve the spacecraft mass state across burn-coast-burn transfers. The
    mass derivative is identically zero; translational acceleration still uses
    the configured gravity and perturbation model. A phase using Euler thrust
    control also carries its kinematic attitude through the coast.
    """

    def __init__(
        self,
        *,
        mu_m3ps2: float,
        j2: bool = False,
        central_body_radius_m: float = 6_378_136.3,
        j2_coefficient: float = 1.08262668e-3,
        third_body_tables: Sequence[ThirdBodyTable] = (),
        thrust_control: ThrustControl | None = None,
    ) -> None:
        _require_asset()
        self.mu = float(mu_m3ps2)
        control_config = thrust_control or ThrustControl.vector()
        carries_attitude = control_config.carries_attitude

        state_dim = 10 if carries_attitude else 7
        control_dim = 3 if carries_attitude else 0
        XtU = oc.ODEArguments(state_dim, control_dim)
        X = XtU.XVec()
        R = X.head(3)
        V = X.segment(3, 3)
        M = X[6]
        attitude = X.segment(7, 3) if carries_attitude else None
        slew_control = XtU.UVec() if carries_attitude else None
        attitude_rate = (
            float(control_config.max_slew_rate_radps) * slew_control
            if slew_control is not None
            else None
        )

        A = _gravity_acceleration(
            R,
            mu_m3ps2=self.mu,
            include_j2=bool(j2),
            central_body_radius_m=float(central_body_radius_m),
            j2_coefficient=float(j2_coefficient),
            time_var=XtU.TVar(),
            third_body_tables=tuple(third_body_tables),
        )
        ode_terms = [V, A, M * 0.0]
        if attitude_rate is not None:
            ode_terms.append(attitude_rate)
        ode = vf.stack(ode_terms)

        Vgroups = {
            ("R", "Position"): R,
            ("V", "Velocity"): V,
            ("M", "Mass"): [6],
            ("t", "time"): XtU.TVar(),
            "RV": [0, 1, 2, 3, 4, 5],
        }
        if attitude is not None and attitude_rate is not None and slew_control is not None:
            Vgroups[("Attitude", "EulerAngles")] = attitude
            Vgroups[("AttitudeRate", "EulerRates")] = attitude_rate
            Vgroups[("SlewControl", "NormalizedAttitudeRate")] = slew_control
            Vgroups["Yaw"] = attitude[0]
            Vgroups["Pitch"] = attitude[1]
            Vgroups["Roll"] = attitude[2]

        super().__init__(ode, state_dim, control_dim, Vgroups=Vgroups)


class FiniteThrustECI(oc.ODEBase if oc is not None else object):
    """Finite-thrust dynamics with configurable direction representation.

    The default retains the original three-component inertial vector-throttle
    control. RIC vectors are rotated into inertial coordinates from the current
    state. Fixed direction uses one scalar throttle. Euler mode augments the
    state with yaw, pitch, and roll and uses throttle plus three angular rates.
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
        thrust_control: ThrustControl | None = None,
    ) -> None:
        _require_asset()
        self.mu = float(mu_m3ps2)
        self.thrust_N = float(thrust_N)
        self.isp_s = float(isp_s)
        self.g0_mps2 = float(g0_mps2)

        if self.thrust_N <= 0.0:
            raise ValueError("FiniteThrustECI requires thrust_N > 0.")
        if self.isp_s <= 0.0:
            raise ValueError("FiniteThrustECI requires isp_s > 0.")

        control_config = thrust_control or ThrustControl.vector()
        carries_attitude = control_config.carries_attitude
        state_dim = 10 if carries_attitude else 7
        control_dim = (
            4
            if control_config.representation == "euler"
            else 1
            if control_config.representation == "fixed"
            else 3
        )
        XtU = oc.ODEArguments(state_dim, control_dim)
        X = XtU.XVec()
        U = XtU.UVec()
        R = X.head(3)
        V = X.segment(3, 3)
        M = X[6]
        attitude = X.segment(7, 3) if carries_attitude else None

        gravity = _gravity_acceleration(
            R,
            mu_m3ps2=self.mu,
            include_j2=bool(j2),
            central_body_radius_m=float(central_body_radius_m),
            j2_coefficient=float(j2_coefficient),
            time_var=XtU.TVar(),
            third_body_tables=tuple(third_body_tables),
        )
        thrust_vector, throttle, attitude_rate = thrust_vector_and_rate(
            control_config,
            controls=U,
            position=R,
            velocity=V,
            attitude=attitude,
        )
        thrust_acceleration = (self.thrust_N / M) * thrust_vector
        mass_flow = -(self.thrust_N / (self.isp_s * self.g0_mps2)) * throttle
        ode_terms = [V, gravity + thrust_acceleration, mass_flow]
        if attitude_rate is not None:
            ode_terms.append(attitude_rate)
        ode = vf.stack(ode_terms)

        Vgroups = {
            ("R", "Position"): R,
            ("V", "Velocity"): V,
            ("M", "Mass"): [6],
            ("t", "time"): XtU.TVar(),
            "RV": [0, 1, 2, 3, 4, 5],
        }
        if control_config.representation == "vector":
            Vgroups[("U", "Control", "Thrust")] = U
        else:
            Vgroups["Throttle"] = vf.stack([U[0]])
        if attitude is not None and attitude_rate is not None:
            Vgroups[("Attitude", "EulerAngles")] = attitude
            Vgroups[("AttitudeRate", "EulerRates")] = attitude_rate
            Vgroups[("SlewControl", "NormalizedAttitudeRate")] = U.segment(1, 3)
            Vgroups["Yaw"] = attitude[0]
            Vgroups["Pitch"] = attitude[1]
            Vgroups["Roll"] = attitude[2]

        super().__init__(ode, state_dim, control_dim, Vgroups=Vgroups)


# Compatibility name for code written before the powered dynamics were made
# propulsion-neutral.
ChemicalBurnECI = FiniteThrustECI
