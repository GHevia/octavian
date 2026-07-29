"""ASSET ODEs for relative-motion propagation formulations."""

from __future__ import annotations

from collections.abc import Sequence

from .._asset import oc, require_asset, vf
from .._control_dynamics import thrust_vector_and_rate
from ..control import ThrustControl
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


class CoupledRelativeMassCoastODE(oc.ODEBase if oc is not None else object):
    """Propagate coupled chief/deputy ECI states and constant deputy mass.

    This is the coast member of a relative finite-thrust phase chain. Its
    base state is ``[chief r,v, deputy r,v, deputy mass]``. Carrying mass
    through the coast lets ordinary continuous phase links preserve propellant
    state between powered segments without changing either trajectory's
    gravity. Euler-controlled chains also carry their three attitude states.
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
        """Build a mass-carrying coupled relative coast ODE."""
        require_asset("mass-carrying coupled relative dynamics")
        control_config = thrust_control or ThrustControl.vector()
        carries_attitude = control_config.carries_attitude
        state_dim = 16 if carries_attitude else 13
        control_dim = 3 if carries_attitude else 0
        arguments = oc.ODEArguments(state_dim, control_dim)
        state = arguments.XVec()
        chief_position = state.head(3)
        chief_velocity = state.segment(3, 3)
        deputy_position = state.segment(6, 3)
        deputy_velocity = state.segment(9, 3)
        mass = state[12]
        attitude = state.segment(13, 3) if carries_attitude else None
        slew_control = arguments.UVec() if carries_attitude else None
        attitude_rate = (
            float(control_config.max_slew_rate_radps) * slew_control
            if slew_control is not None
            else None
        )
        time = arguments.TVar()
        force_options = {
            "mu_m3ps2": float(mu_m3ps2),
            "include_j2": bool(j2),
            "central_body_radius_m": float(central_body_radius_m),
            "j2_coefficient": float(j2_coefficient),
            "time_var": time,
            "third_body_tables": tuple(third_body_tables),
        }
        ode_terms = [
            chief_velocity,
            _gravity_acceleration(chief_position, **force_options),
            deputy_velocity,
            _gravity_acceleration(deputy_position, **force_options),
            0.0 * mass,
        ]
        if attitude_rate is not None:
            ode_terms.append(attitude_rate)
        ode = vf.stack(ode_terms)
        groups = {
            ("ChiefR", "ChiefPosition"): chief_position,
            ("ChiefV", "ChiefVelocity"): chief_velocity,
            ("DeputyR", "DeputyPosition"): deputy_position,
            ("DeputyV", "DeputyVelocity"): deputy_velocity,
            ("M", "Mass"): [12],
            ("t", "time"): time,
        }
        if attitude is not None and attitude_rate is not None and slew_control is not None:
            groups[("Attitude", "EulerAngles")] = attitude
            groups[("AttitudeRate", "EulerRates")] = attitude_rate
            groups[("SlewControl", "NormalizedAttitudeRate")] = slew_control
            groups["Yaw"] = attitude[0]
            groups["Pitch"] = attitude[1]
            groups["Roll"] = attitude[2]
        super().__init__(ode, state_dim, control_dim, Vgroups=groups)


class FiniteThrustRelativeODE(oc.ODEBase if oc is not None else object):
    """Propagate an unpowered chief and a finite-thrust deputy in ECI.

    The base state is ``[chief r,v, deputy r,v, deputy mass]``. Direction can
    use a free vector, prescribed direction, or Euler kinematics in inertial
    or chief RIC axes. Gravity and configured perturbations are applied
    independently to chief and deputy; thrust and mass depletion apply only
    to the deputy.
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
        """Build exact coupled relative finite-thrust dynamics."""
        require_asset("finite-thrust coupled relative dynamics")
        thrust = float(thrust_N)
        specific_impulse = float(isp_s)
        standard_gravity = float(g0_mps2)
        if thrust <= 0.0:
            raise ValueError("FiniteThrustRelativeODE requires thrust_N > 0")
        if specific_impulse <= 0.0:
            raise ValueError("FiniteThrustRelativeODE requires isp_s > 0")
        if standard_gravity <= 0.0:
            raise ValueError("FiniteThrustRelativeODE requires g0_mps2 > 0")

        control_config = thrust_control or ThrustControl.vector()
        carries_attitude = control_config.carries_attitude
        state_dim = 16 if carries_attitude else 13
        control_dim = (
            4
            if control_config.representation == "euler"
            else 1
            if control_config.representation == "fixed"
            else 3
        )
        arguments = oc.ODEArguments(state_dim, control_dim)
        state = arguments.XVec()
        control = arguments.UVec()
        chief_position = state.head(3)
        chief_velocity = state.segment(3, 3)
        deputy_position = state.segment(6, 3)
        deputy_velocity = state.segment(9, 3)
        mass = state[12]
        attitude = state.segment(13, 3) if carries_attitude else None
        time = arguments.TVar()
        force_options = {
            "mu_m3ps2": float(mu_m3ps2),
            "include_j2": bool(j2),
            "central_body_radius_m": float(central_body_radius_m),
            "j2_coefficient": float(j2_coefficient),
            "time_var": time,
            "third_body_tables": tuple(third_body_tables),
        }
        chief_acceleration = _gravity_acceleration(chief_position, **force_options)
        deputy_acceleration = _gravity_acceleration(deputy_position, **force_options)
        thrust_vector, throttle, attitude_rate = thrust_vector_and_rate(
            control_config,
            controls=control,
            position=chief_position,
            velocity=chief_velocity,
            attitude=attitude,
        )
        thrust_acceleration = (thrust / mass) * thrust_vector
        mass_flow = -(thrust / (specific_impulse * standard_gravity)) * throttle
        ode_terms = [
            chief_velocity,
            chief_acceleration,
            deputy_velocity,
            deputy_acceleration + thrust_acceleration,
            mass_flow,
        ]
        if attitude_rate is not None:
            ode_terms.append(attitude_rate)
        ode = vf.stack(ode_terms)
        groups = {
            ("ChiefR", "ChiefPosition"): chief_position,
            ("ChiefV", "ChiefVelocity"): chief_velocity,
            ("DeputyR", "DeputyPosition"): deputy_position,
            ("DeputyV", "DeputyVelocity"): deputy_velocity,
            ("M", "Mass"): [12],
            ("t", "time"): time,
        }
        if control_config.representation == "vector":
            groups[("U", "Control", "Thrust")] = control
        else:
            groups["Throttle"] = vf.stack([control[0]])
        if attitude is not None and attitude_rate is not None:
            groups[("Attitude", "EulerAngles")] = attitude
            groups[("AttitudeRate", "EulerRates")] = attitude_rate
            groups[("SlewControl", "NormalizedAttitudeRate")] = control.segment(1, 3)
            groups["Yaw"] = attitude[0]
            groups["Pitch"] = attitude[1]
            groups["Roll"] = attitude[2]
        super().__init__(ode, state_dim, control_dim, Vgroups=groups)


class CoupledRelativeRICODE(oc.ODEBase if oc is not None else object):
    """Propagate a chief ECI state stacked with the deputy's exact RIC state.

    States are ``[chief_r, chief_v, rho_RIC, rho_dot_RIC]``. The equations
    retain the nonlinear differential central-gravity acceleration and the
    complete rotating-frame terms for an elliptic two-body chief. Perturbing
    accelerations are intentionally excluded because their frame angular
    acceleration would require a consistent force-model derivative.
    """

    def __init__(self, *, mu_m3ps2: float) -> None:
        """Build the exact central-gravity coupled ECI/RIC ODE.

        Args:
            mu_m3ps2: Central-body gravitational parameter in SI units.
        """
        require_asset("coupled RIC relative dynamics")
        mu = float(mu_m3ps2)
        if mu <= 0.0:
            raise ValueError("mu_m3ps2 must be positive")

        arguments = oc.ODEArguments(12, 0)
        state = arguments.XVec()
        chief_position = state.head(3)
        chief_velocity = state.segment(3, 3)
        relative_position = state.segment(6, 3)
        relative_velocity = state.segment(9, 3)

        radius = chief_position.norm()
        radius_sq = chief_position.dot(chief_position)
        angular_momentum = chief_position.cross(chief_velocity)
        momentum = angular_momentum.norm()
        radial_axis = chief_position / radius
        cross_track_axis = angular_momentum / momentum
        in_track_axis = cross_track_axis.cross(radial_axis)

        x = relative_position[0]
        y = relative_position[1]
        z = relative_position[2]
        deputy_position = (
            chief_position + radial_axis * x + in_track_axis * y + cross_track_axis * z
        )
        chief_acceleration = _gravity_acceleration(
            chief_position,
            mu_m3ps2=mu,
        )
        deputy_acceleration = _gravity_acceleration(
            deputy_position,
            mu_m3ps2=mu,
        )
        acceleration_difference = deputy_acceleration - chief_acceleration
        gravity_difference_ric = vf.stack(
            [
                radial_axis.dot(acceleration_difference),
                in_track_axis.dot(acceleration_difference),
                cross_track_axis.dot(acceleration_difference),
            ]
        )

        frame_rate = momentum / radius_sq
        radial_rate = chief_position.dot(chief_velocity) / radius_sq
        frame_rate_derivative = -2.0 * frame_rate * radial_rate
        rotating_terms = vf.stack(
            [
                2.0 * frame_rate * relative_velocity[1]
                + frame_rate_derivative * y
                + frame_rate**2 * x,
                -2.0 * frame_rate * relative_velocity[0]
                - frame_rate_derivative * x
                + frame_rate**2 * y,
                0.0 * z,
            ]
        )
        relative_acceleration = gravity_difference_ric + rotating_terms
        ode = vf.stack(
            [
                chief_velocity,
                chief_acceleration,
                relative_velocity,
                relative_acceleration,
            ]
        )
        groups = {
            ("ChiefR", "ChiefPosition"): chief_position,
            ("ChiefV", "ChiefVelocity"): chief_velocity,
            ("R", "Position", "RelativePosition"): relative_position,
            ("V", "Velocity", "RelativeVelocity"): relative_velocity,
            ("t", "time"): arguments.TVar(),
        }
        super().__init__(ode, 12, 0, Vgroups=groups)


class NonlinearRelativeRICODE(oc.ODEBase if oc is not None else object):
    """Exact six-state circular-chief RIC dynamics before CWH linearization.

    The equations keep the full deputy radius in the two-body gravitational
    term. Linearizing these equations about zero separation produces the
    Clohessy-Wiltshire equations.
    """

    def __init__(
        self,
        *,
        mu_m3ps2: float,
        chief_orbit_radius_m: float,
    ) -> None:
        """Build the autonomous exact circular-chief RIC ODE.

        Args:
            mu_m3ps2: Central-body gravitational parameter in SI units.
            chief_orbit_radius_m: Constant circular chief radius in meters.
        """
        require_asset("nonlinear RIC relative dynamics")
        mu = float(mu_m3ps2)
        chief_radius = float(chief_orbit_radius_m)
        if mu <= 0.0 or chief_radius <= 0.0:
            raise ValueError("mu_m3ps2 and chief_orbit_radius_m must be positive")

        arguments = oc.ODEArguments(6, 0)
        state = arguments.XVec()
        position = state.head(3)
        velocity = state.segment(3, 3)
        x = position[0]
        y = position[1]
        z = position[2]
        deputy_radius = vf.sqrt((chief_radius + x) ** 2 + y**2 + z**2)
        mean_motion = (mu / chief_radius**3) ** 0.5
        gravity_rate = mu / deputy_radius**3
        acceleration = vf.stack(
            [
                2.0 * mean_motion * velocity[1]
                + (chief_radius + x) * (mean_motion**2 - gravity_rate),
                -2.0 * mean_motion * velocity[0] + y * (mean_motion**2 - gravity_rate),
                -gravity_rate * z,
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


class RelativeOrbitalElementsODE(oc.ODEBase if oc is not None else object):
    """Propagate exact two-body relative orbital-element differences.

    D'Amico quasi-nonsingular elements and classical element differences are
    constant under two-body dynamics except for relative mean longitude or
    mean anomaly. Its rate is the deputy-minus-chief Keplerian mean motion.
    """

    def __init__(
        self,
        *,
        mu_m3ps2: float,
        chief_semi_major_axis_m: float,
        representation: str = "damico",
    ) -> None:
        """Build a native six-state relative-element ODE.

        Args:
            mu_m3ps2: Central-body gravitational parameter in SI units.
            chief_semi_major_axis_m: Chief osculating semi-major axis.
            representation: ``"damico"`` or ``"classical_elements"``.
        """
        require_asset("relative orbital-element dynamics")
        mu = float(mu_m3ps2)
        chief_a = float(chief_semi_major_axis_m)
        normalized = str(representation).strip().lower().replace("-", "_")
        if mu <= 0.0 or chief_a <= 0.0:
            raise ValueError("mu_m3ps2 and chief_semi_major_axis_m must be positive")
        if normalized not in {"damico", "classical_elements"}:
            raise ValueError("representation must be 'damico' or 'classical_elements'")

        arguments = oc.ODEArguments(6, 0)
        elements = arguments.XVec()
        deputy_a = (
            chief_a * (1.0 + elements[0]) if normalized == "damico" else chief_a + elements[0]
        )
        chief_mean_motion = (mu / chief_a**3) ** 0.5
        deputy_mean_motion = (mu / deputy_a**3) ** 0.5
        longitude_rate = deputy_mean_motion - chief_mean_motion
        zero = 0.0 * elements[0]
        derivative = (
            vf.stack([zero, longitude_rate, zero, zero, zero, zero])
            if normalized == "damico"
            else vf.stack([zero, zero, zero, zero, zero, longitude_rate])
        )
        names = (
            (
                ("delta_a", "DeltaA"),
                ("delta_lambda", "DeltaLambda"),
                ("delta_ex", "DeltaEx"),
                ("delta_ey", "DeltaEy"),
                ("delta_ix", "DeltaIx"),
                ("delta_iy", "DeltaIy"),
            )
            if normalized == "damico"
            else (
                ("delta_a_m", "DeltaA"),
                ("delta_e", "DeltaE"),
                ("delta_i", "DeltaI"),
                ("delta_raan", "DeltaRAAN"),
                ("delta_argp", "DeltaArgP"),
                ("delta_mean_anomaly", "DeltaMeanAnomaly"),
            )
        )
        groups: dict[object, object] = {
            aliases: elements[index] for index, aliases in enumerate(names)
        }
        groups[("RelativeElements", "ROE")] = elements
        groups[("t", "time")] = arguments.TVar()
        super().__init__(derivative, 6, 0, Vgroups=groups)
