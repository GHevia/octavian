"""Phase classification, dynamics selection, and ASSET phase construction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..._asset import Tmodes
from ...astro.kepler import cartesian_to_classic
from ...astro.types import as_vec3
from ...control import ThrustControl
from ...coordinates import (
    CARTESIAN,
    CARTESIAN_MASS,
    CARTESIAN_MASS_EULER_COAST,
    CARTESIAN_MASS_EULER_THRUST,
    CARTESIAN_MASS_FIXED_THRUST,
    CARTESIAN_MASS_THRUST,
    CLASSICAL_RELATIVE_ELEMENTS,
    COUPLED_RELATIVE_ECI,
    COUPLED_RELATIVE_ECI_MASS,
    COUPLED_RELATIVE_ECI_MASS_EULER_COAST,
    COUPLED_RELATIVE_ECI_MASS_EULER_THRUST,
    COUPLED_RELATIVE_ECI_MASS_FIXED_THRUST,
    COUPLED_RELATIVE_ECI_MASS_THRUST,
    COUPLED_RELATIVE_RIC,
    DAMICO_RELATIVE_ELEMENTS,
    RELATIVE_CARTESIAN,
    StateLayout,
)
from ...dynamics import (
    FiniteThrustECI,
    MassCoastECI,
    PerturbedECI,
    ThirdBodyTable,
    TwoBodyECI,
)
from ...phase import Phase
from ...relative import (
    ClohessyWiltshire,
    NonlinearRelative,
    RelativePropagationMode,
    relative_state_to_classical_relative_orbital_elements,
    relative_state_to_relative_orbital_elements,
)
from ...relative.dynamics import (
    ClohessyWiltshireODE,
    CoupledRelativeMassCoastODE,
    CoupledRelativeODE,
    CoupledRelativeRICODE,
    FiniteThrustRelativeODE,
    NonlinearRelativeRICODE,
    RelativeOrbitalElementsODE,
)
from ...relative.solar import circular_chief_state
from ...relative.transforms import relative_to_inertial_state, ric_basis
from ...specs import BoundaryState
from ...variables import ImpulsiveDeltaV
from ..third_bodies import phase_perturbations, tables_for_phase


@dataclass(slots=True)
class PhaseBuild:
    """Bookkeeping shared by the composable compiler passes.

    User phases and compiled phases may diverge when the compiler adds an
    internal phase, such as a post-burn terminal shell. This record keeps the
    public phase, compiled ASSET phase, dimensions, and normalized time bounds
    together for later constraint, linking, solving, and extraction passes.
    """

    ph: Phase
    asset_phase: Any
    t_bounds: tuple[float, float]
    index: int
    compile_phase: Phase | None = None
    layout: StateLayout = CARTESIAN
    powered_kind: str | None = None
    enable_adaptive_mesh: bool = True

    @property
    def state_dim(self) -> int:
        """Return the compiled differential-state dimension."""
        return self.layout.state_dim

    @property
    def control_dim(self) -> int:
        """Return the compiled control dimension."""
        return self.layout.control_dim

    @property
    def is_powered(self) -> bool:
        """Return whether this compiled phase has finite-thrust controls."""
        return self.powered_kind is not None

    @property
    def is_chemical_burn(self) -> bool:
        """Return whether this is a legacy chemical-burn phase."""
        return self.powered_kind == "chemical_burn"


def has_impulsive_variable(phase: Phase, where: str) -> bool:
    """Return whether a phase declares an impulsive delta-v at a boundary."""
    normalized_where = (where or "").strip().lower()
    location = "Front" if normalized_where in ("front", "start", "initial", "t0") else "Back"
    for variable in getattr(phase, "variables", []) or []:
        if isinstance(variable, ImpulsiveDeltaV) and getattr(variable, "where", "") == location:
            return True
    for event in getattr(phase, "events", []) or []:
        if getattr(event, "kind", "") == "impulse" and getattr(event, "where", "") == location:
            return True
    return False


def is_chemical_burn(phase: Phase) -> bool:
    """Return whether a phase uses finite chemical-burn dynamics."""
    return powered_phase_kind(phase) == "chemical_burn"


def powered_phase_kind(phase: Phase) -> str | None:
    """Return the normalized propulsion kind for a finite-thrust phase."""
    normalized_mode = (getattr(phase, "mode", "") or "").strip().lower().replace("-", "_")
    if normalized_mode in ("burn", "chemical_burn", "finite_burn"):
        return "chemical_burn"
    if normalized_mode in ("powered", "finite_thrust"):
        return "finite_thrust"
    if normalized_mode == "low_thrust":
        return "low_thrust"
    return None


def is_powered_phase(phase: Phase) -> bool:
    """Return whether a phase uses the finite-thrust state and control model."""
    return powered_phase_kind(phase) is not None


def thrust_control_for_phase(phase: Phase) -> ThrustControl:
    """Return a phase's explicit control representation or legacy default."""
    return phase.thrust_control or ThrustControl.vector()


def is_coast_like(phase: Phase) -> bool:
    """Return whether a phase uses coast-like translational dynamics."""
    normalized_mode = (getattr(phase, "mode", "") or "").strip().lower().replace("-", "_")
    return normalized_mode in ("coast", "transfer", "rendezvous", "relative_coast", "cwh")


def cwh_model(phase: Phase) -> ClohessyWiltshire | None:
    """Return the phase's CWH model, when configured."""
    dynamics = phase.dynamics
    model = dynamics.model if dynamics is not None else None
    return model if isinstance(model, ClohessyWiltshire) else None


def nonlinear_relative_model(phase: Phase) -> NonlinearRelative | None:
    """Return the phase's full nonlinear relative model, when configured."""
    dynamics = phase.dynamics
    model = dynamics.model if dynamics is not None else None
    return model if isinstance(model, NonlinearRelative) else None


def is_relative_phase(phase: Phase) -> bool:
    """Return whether a phase uses either full relative dynamics or CWH."""
    return cwh_model(phase) is not None or nonlinear_relative_model(phase) is not None


def mass_state_phase_indices(phases: Sequence[Phase]) -> set[int]:
    """Return phases that carry mass across a continuous powered chain."""
    powered_indices = [idx for idx, phase in enumerate(phases) if is_powered_phase(phase)]
    if not powered_indices:
        return set()

    first_powered = powered_indices[0]
    last_powered = powered_indices[-1]
    return set(range(first_powered, last_powered + 1))


def validate_powered_phase_chain(phases: Sequence[Phase]) -> None:
    """Validate spacecraft and phase continuity for finite-thrust phases.

    Powered phases may appear alone or in any coast/powered sequence. Coast
    phases between the first and last powered phases carry mass so continuity
    links preserve propellant depletion across the chain.
    """
    powered_indices = [idx for idx, phase in enumerate(phases) if is_powered_phase(phase)]
    if not powered_indices:
        return

    reference_spacecraft = None
    chain_controls: list[ThrustControl] = []
    for idx in range(powered_indices[0], powered_indices[-1] + 1):
        phase = phases[idx]
        chain_controls.append(thrust_control_for_phase(phase))
        if not (is_powered_phase(phase) or is_coast_like(phase)):
            raise ValueError(
                "Phases between powered phases must use powered or coast-like dynamics; "
                f"phase {phase.name!r} has mode={phase.mode!r}."
            )
        spacecraft = getattr(phase, "spacecraft", None)
        if isinstance(spacecraft, str) or spacecraft is None:
            raise ValueError(f"Mass-carrying phase {phase.name!r} requires a Spacecraft object.")
        if reference_spacecraft is None:
            reference_spacecraft = spacecraft
        elif spacecraft != reference_spacecraft:
            raise ValueError(
                "A powered phase chain must use one Spacecraft configuration so mass continuity "
                f"is unambiguous; phase {phase.name!r} uses {spacecraft.name!r}."
            )
        if is_powered_phase(phase):
            thruster = first_thruster(phase)
            if float(thruster.thrust_N) <= 0.0 or float(thruster.isp_s) <= 0.0:
                raise ValueError(
                    f"Powered phase {phase.name!r} requires thrust_N > 0 and isp_s > 0."
                )
    attitude_controls = [control for control in chain_controls if control.carries_attitude]
    if attitude_controls:
        if len(attitude_controls) != len(chain_controls):
            raise ValueError(
                "Euler attitude continuity requires every burn and intermediate "
                "coast in the powered chain to use ThrustControl.euler(...)."
            )
        frames = {control.frame for control in attitude_controls}
        if len(frames) != 1:
            raise ValueError("Linked Euler attitude phases must use one reference frame")


def validate_chemical_burn_transfer(phases: Sequence[Phase]) -> None:
    """Compatibility alias for :func:`validate_powered_phase_chain`."""
    validate_powered_phase_chain(phases)


def first_thruster(phase: Phase):
    """Return the configured thruster for a powered phase."""
    spacecraft = getattr(phase, "spacecraft", None)
    if isinstance(spacecraft, str) or spacecraft is None:
        raise ValueError(f"Powered phase {phase.name!r} requires a Spacecraft object.")
    thruster_name = str(getattr(phase, "info", {}).get("thruster", "main"))
    thruster = spacecraft.get_thruster(thruster_name)
    if thruster is not None:
        return thruster
    if len(spacecraft.thrusters) == 1:
        return spacecraft.thrusters[0]
    raise KeyError(f"No thruster named {thruster_name!r} on spacecraft {spacecraft.name!r}")


def ode_for_phase(
    phase: Phase,
    *,
    carries_mass: bool = False,
    third_body_tables: Sequence[ThirdBodyTable] = (),
):
    """Construct the ASSET ODE selected by phase intent and environment."""
    dynamics = phase.dynamics
    if dynamics is None:
        raise ValueError(f"Phase {phase.name!r} is missing dynamics.")
    perturbations = phase_perturbations(phase)
    relative_model = cwh_model(phase)
    if relative_model is not None:
        if is_powered_phase(phase) or carries_mass:
            raise ValueError("CWH phases do not yet support finite-thrust or mass states.")
        if perturbations.j2 or third_body_tables:
            raise ValueError(
                "Dynamics.cwh is an unforced linear model and cannot include "
                "perturbations. Use Dynamics.relative(...) for exact nonlinear "
                "chief/deputy propagation with J2 or third-body gravity."
            )
        return ClohessyWiltshireODE(mean_motion_radps=relative_model.mean_motion_radps)
    nonlinear_model = nonlinear_relative_model(phase)
    if nonlinear_model is not None:
        mode = nonlinear_model.propagation_mode
        has_perturbations = any(
            (
                perturbations.j2,
                perturbations.srp,
                perturbations.drag,
                bool(third_body_tables),
            )
        )
        if mode is not RelativePropagationMode.COUPLED_ECI and has_perturbations:
            raise ValueError(
                f"Relative propagation mode {mode.value!r} is a two-body "
                "formulation and cannot include perturbations. Use "
                "propagation_mode='coupled_eci' for J2 or third-body gravity."
            )
        if (is_powered_phase(phase) or carries_mass) and (
            mode is not RelativePropagationMode.COUPLED_ECI
        ):
            raise ValueError(
                "Relative finite-thrust and mass-carrying coast phases require "
                "propagation_mode='coupled_eci'. Native RIC and relative-element "
                "thrust formulations are not implemented yet."
            )
        force_options = {
            "mu_m3ps2": float(dynamics.mu_m3ps2),
            "j2": bool(perturbations.j2),
            "central_body_radius_m": float(dynamics.central_body_radius_m),
            "j2_coefficient": float(dynamics.j2_coefficient),
            "third_body_tables": tuple(third_body_tables),
        }
        if is_powered_phase(phase):
            thruster = first_thruster(phase)
            return FiniteThrustRelativeODE(
                thrust_N=float(thruster.thrust_N),
                isp_s=float(thruster.isp_s),
                thrust_control=thrust_control_for_phase(phase),
                **force_options,
            )
        if carries_mass:
            return CoupledRelativeMassCoastODE(
                thrust_control=thrust_control_for_phase(phase),
                **force_options,
            )
        if mode is RelativePropagationMode.COUPLED_ECI:
            return CoupledRelativeODE(**force_options)
        if mode is RelativePropagationMode.COUPLED_RIC:
            return CoupledRelativeRICODE(
                mu_m3ps2=float(dynamics.mu_m3ps2),
            )
        if mode is RelativePropagationMode.NONLINEAR_RIC:
            return NonlinearRelativeRICODE(
                mu_m3ps2=float(dynamics.mu_m3ps2),
                chief_orbit_radius_m=float(
                    np.linalg.norm(nonlinear_model.chief_initial_state_eci.r_m)
                ),
            )
        chief_elements = cartesian_to_classic(
            r_m=nonlinear_model.chief_initial_state_eci.r_m,
            v_mps=nonlinear_model.chief_initial_state_eci.v_mps,
            mu_m3ps2=float(dynamics.mu_m3ps2),
        )
        return RelativeOrbitalElementsODE(
            mu_m3ps2=float(dynamics.mu_m3ps2),
            chief_semi_major_axis_m=float(chief_elements["a_m"]),
            representation=mode.value,
        )
    if is_powered_phase(phase):
        thruster = first_thruster(phase)
        return FiniteThrustECI(
            mu_m3ps2=float(dynamics.mu_m3ps2),
            thrust_N=float(thruster.thrust_N),
            isp_s=float(thruster.isp_s),
            j2=bool(perturbations.j2),
            central_body_radius_m=float(dynamics.central_body_radius_m),
            j2_coefficient=float(dynamics.j2_coefficient),
            third_body_tables=tuple(third_body_tables),
            thrust_control=thrust_control_for_phase(phase),
        )
    if carries_mass:
        return MassCoastECI(
            mu_m3ps2=float(dynamics.mu_m3ps2),
            j2=bool(perturbations.j2),
            central_body_radius_m=float(dynamics.central_body_radius_m),
            j2_coefficient=float(dynamics.j2_coefficient),
            third_body_tables=tuple(third_body_tables),
            thrust_control=thrust_control_for_phase(phase),
        )
    if perturbations.j2 or third_body_tables:
        return PerturbedECI(
            mu_m3ps2=float(dynamics.mu_m3ps2),
            j2=bool(perturbations.j2),
            central_body_radius_m=float(dynamics.central_body_radius_m),
            j2_coefficient=float(dynamics.j2_coefficient),
            third_body_tables=tuple(third_body_tables),
        )
    return TwoBodyECI(mu_m3ps2=float(dynamics.mu_m3ps2))


def phase_dimensions(phase: Phase) -> tuple[int, int, bool]:
    """Return user-visible state and control dimensions for a phase."""
    layout = layout_for_phase(phase)
    return layout.state_dim, layout.control_dim, is_powered_phase(phase)


def layout_for_phase(phase: Phase, *, carries_mass: bool = False) -> StateLayout:
    """Return the named state/control layout required by a phase."""
    if cwh_model(phase) is not None:
        return RELATIVE_CARTESIAN
    relative_model = nonlinear_relative_model(phase)
    control = thrust_control_for_phase(phase)
    if relative_model is not None:
        if relative_model.propagation_mode is RelativePropagationMode.COUPLED_ECI:
            if is_powered_phase(phase):
                if control.representation == "fixed":
                    return COUPLED_RELATIVE_ECI_MASS_FIXED_THRUST
                if control.representation == "euler":
                    return COUPLED_RELATIVE_ECI_MASS_EULER_THRUST
                return COUPLED_RELATIVE_ECI_MASS_THRUST
            if carries_mass:
                if control.representation == "euler":
                    return COUPLED_RELATIVE_ECI_MASS_EULER_COAST
                return COUPLED_RELATIVE_ECI_MASS
        layouts = {
            RelativePropagationMode.COUPLED_ECI: COUPLED_RELATIVE_ECI,
            RelativePropagationMode.COUPLED_RIC: COUPLED_RELATIVE_RIC,
            RelativePropagationMode.NONLINEAR_RIC: RELATIVE_CARTESIAN,
            RelativePropagationMode.DAMICO: DAMICO_RELATIVE_ELEMENTS,
            RelativePropagationMode.CLASSICAL_ELEMENTS: CLASSICAL_RELATIVE_ELEMENTS,
        }
        return layouts[relative_model.propagation_mode]
    if is_powered_phase(phase):
        if control.representation == "fixed":
            return CARTESIAN_MASS_FIXED_THRUST
        if control.representation == "euler":
            return CARTESIAN_MASS_EULER_THRUST
        return CARTESIAN_MASS_THRUST
    if carries_mass:
        if control.representation == "euler":
            return CARTESIAN_MASS_EULER_COAST
        return CARTESIAN_MASS
    return CARTESIAN


def compile_phase_dimensions(
    phase: Phase,
    *,
    carries_mass: bool = False,
) -> tuple[int, int, bool]:
    """Return the state/control dimensions required by the compiled ODE."""
    layout = layout_for_phase(phase, carries_mass=carries_mass)
    return layout.state_dim, layout.control_dim, is_powered_phase(phase)


def apply_control_constraints(
    asset_phase: Any,
    phase: Phase,
    layout: StateLayout,
    *,
    fix_initial_attitude: bool,
) -> None:
    """Apply throttle, Euler-angle, and kinematic slew-rate bounds."""
    control = thrust_control_for_phase(phase)
    if is_powered_phase(phase):
        if control.representation == "vector":
            asset_phase.addUpperNormBound("Path", "U", 1.0)
        else:
            asset_phase.addLUVarBound("Path", "Throttle", 0.0, 1.0)

    if not control.carries_attitude:
        return
    if fix_initial_attitude:
        asset_phase.addBoundaryValue(
            "Front",
            ["Attitude"],
            np.asarray(control.initial_angles_rad, dtype=float),
        )
    asset_phase.addUpperSquaredNormBound(
        "Path",
        "SlewControl",
        1.0,
    )
    for group, bounds in (
        ("Yaw", control.yaw_bounds_rad),
        ("Pitch", control.pitch_bounds_rad),
        ("Roll", control.roll_bounds_rad),
    ):
        asset_phase.addLUVarBound("Path", group, float(bounds[0]), float(bounds[1]))


def trajectory_rvt(raw_traj: np.ndarray, layout: StateLayout | int) -> np.ndarray:
    """Return the public ``[R, V, t]`` view of an ASSET trajectory."""
    raw = np.asarray(raw_traj, dtype=float)
    if isinstance(layout, StateLayout):
        if layout in {
            COUPLED_RELATIVE_ECI,
            DAMICO_RELATIVE_ELEMENTS,
            CLASSICAL_RELATIVE_ELEMENTS,
        }:
            raise ValueError("This relative trajectory requires phase-aware RIC conversion")
        columns = layout.public_rvt_columns()
        time_column = layout.time_column
    else:
        time_column = int(layout)
        columns = (0, 1, 2, 3, 4, 5, time_column)
    if raw.shape[1] <= time_column:
        raise ValueError("ASSET trajectory is missing the phase time column.")
    return raw[:, columns]


def augment_powered_guess(
    base_guess: Sequence[np.ndarray],
    *,
    phase: Phase,
    mass0_kg: float,
    thrust_N: float,
    isp_s: float,
) -> list[np.ndarray]:
    """Add mass and the selected thrust/attitude representation to a guess."""
    rows = [np.asarray(row, dtype=float).reshape(-1) for row in base_guess]
    if not rows:
        return []

    delta_velocity = as_vec3(rows[-1][3:6] - rows[0][3:6])
    delta_velocity_magnitude = float(np.linalg.norm(delta_velocity))
    direction = (
        delta_velocity / delta_velocity_magnitude
        if delta_velocity_magnitude > 0.0
        else np.zeros(3, dtype=float)
    )

    duration_s = max(float(rows[-1][6] - rows[0][6]), 1.0)
    mass_flow_kgps = float(thrust_N) / (float(isp_s) * 9.80665)
    acceleration_mps2 = float(thrust_N) / max(float(mass0_kg), 1.0)
    impulsive_burn_time_s = delta_velocity_magnitude / max(acceleration_mps2, 1e-12)
    throttle = min(1.0, max(0.0, impulsive_burn_time_s / duration_s))
    control = throttle * direction

    augmented: list[np.ndarray] = []
    mass_start_kg = float(getattr(phase, "info", {}).get("_mass_guess_start_kg", mass0_kg))
    control_config = thrust_control_for_phase(phase)
    attitude = _attitude_guess_start(phase, control_config)
    for index, row in enumerate(rows):
        fraction = index / max(len(rows) - 1, 1)
        mass = max(
            mass_start_kg - mass_flow_kgps * throttle * duration_s * fraction,
            1.0,
        )
        if control_config.representation == "vector":
            vector_control = control
            if control_config.frame == "ric":
                vector_control = throttle * (ric_basis(row[0:3], row[3:6]) @ direction)
            controls = vector_control
            states = np.hstack([row[0:6], mass])
        elif control_config.representation == "fixed":
            controls = np.asarray([throttle], dtype=float)
            states = np.hstack([row[0:6], mass])
        else:
            controls = np.asarray([throttle, 0.0, 0.0, 0.0], dtype=float)
            states = np.hstack([row[0:6], mass, attitude])
        augmented.append(np.hstack([states, row[6], controls]))
    return augmented


# Compatibility name retained for focused downstream tests and imports.
augment_chemical_burn_guess = augment_powered_guess


def augment_mass_coast_guess(
    base_guess: Sequence[np.ndarray],
    *,
    phase: Phase,
    mass0_kg: float,
) -> list[np.ndarray]:
    """Add constant mass and optional kinematic-attitude states."""
    mass_start_kg = float(getattr(phase, "info", {}).get("_mass_guess_start_kg", mass0_kg))
    control_config = thrust_control_for_phase(phase)
    attitude = _attitude_guess_start(phase, control_config)
    augmented: list[np.ndarray] = []
    for row in base_guess:
        rvt = np.asarray(row, dtype=float).reshape(-1)
        if control_config.carries_attitude:
            augmented.append(
                np.hstack(
                    [
                        rvt[0:6],
                        mass_start_kg,
                        attitude,
                        rvt[6],
                        np.zeros(3, dtype=float),
                    ]
                )
            )
        else:
            augmented.append(np.hstack([rvt[0:6], mass_start_kg, rvt[6]]))
    return augmented


def augment_powered_relative_guess(
    coupled_guess: Sequence[np.ndarray],
    *,
    relative_guess: Sequence[np.ndarray],
    phase: Phase,
    mass0_kg: float,
    thrust_N: float,
    isp_s: float,
) -> list[np.ndarray]:
    """Add deputy mass and configured thrust controls to relative rows.

    Input rows use ``[chief r,v, deputy r,v, t]``. Output rows use
    ``[chief r,v, deputy r,v, mass, t, control]``. The deterministic control
    seed follows the endpoint RIC-velocity change, rotated into ECI at each
    sample when inertial vector controls are selected. It is only an initial
    optimizer guess; the solved control follows the configured representation.
    """
    rows = [np.asarray(row, dtype=float).reshape(-1) for row in coupled_guess]
    if not rows:
        return []
    if any(row.size != 13 for row in rows):
        raise ValueError("Coupled relative guess rows must contain 12 states and time")
    relative_rows = [np.asarray(row, dtype=float).reshape(-1) for row in relative_guess]
    if len(relative_rows) != len(rows) or any(row.size != 7 for row in relative_rows):
        raise ValueError("Relative control seeding requires matching RIC state/time rows")

    delta_velocity = as_vec3(relative_rows[-1][3:6] - relative_rows[0][3:6])
    delta_velocity_magnitude = float(np.linalg.norm(delta_velocity))
    has_direction = delta_velocity_magnitude > 1.0e-12
    direction = (
        delta_velocity / delta_velocity_magnitude
        if has_direction
        else np.asarray([1.0, 0.0, 0.0], dtype=float)
    )
    duration_s = max(float(rows[-1][12] - rows[0][12]), 1.0)
    mass_flow_kgps = float(thrust_N) / (float(isp_s) * 9.80665)
    acceleration_mps2 = float(thrust_N) / max(float(mass0_kg), 1.0)
    impulsive_burn_time_s = delta_velocity_magnitude / max(acceleration_mps2, 1.0e-12)
    throttle = min(1.0, max(0.0, impulsive_burn_time_s / duration_s)) if has_direction else 1.0e-3
    mass_start_kg = float(getattr(phase, "info", {}).get("_mass_guess_start_kg", mass0_kg))
    control_config = thrust_control_for_phase(phase)
    attitude = _attitude_guess_start(phase, control_config)
    augmented: list[np.ndarray] = []
    for index, row in enumerate(rows):
        fraction = index / max(len(rows) - 1, 1)
        mass = max(
            mass_start_kg - mass_flow_kgps * throttle * duration_s * fraction,
            1.0,
        )
        if control_config.representation == "vector":
            control = throttle * direction
            if control_config.frame == "inertial":
                control = throttle * (ric_basis(row[0:3], row[3:6]).T @ direction)
            states = np.hstack([row[0:12], mass])
            controls = control
        elif control_config.representation == "fixed":
            states = np.hstack([row[0:12], mass])
            controls = np.asarray([throttle], dtype=float)
        else:
            states = np.hstack([row[0:12], mass, attitude])
            controls = np.asarray([throttle, 0.0, 0.0, 0.0], dtype=float)
        augmented.append(np.hstack([states, row[12], controls]))
    return augmented


def augment_relative_mass_coast_guess(
    coupled_guess: Sequence[np.ndarray],
    *,
    phase: Phase,
    mass0_kg: float,
) -> list[np.ndarray]:
    """Add constant deputy mass and optional attitude to relative coast rows."""
    mass_start_kg = float(getattr(phase, "info", {}).get("_mass_guess_start_kg", mass0_kg))
    control_config = thrust_control_for_phase(phase)
    attitude = _attitude_guess_start(phase, control_config)
    augmented: list[np.ndarray] = []
    for raw_row in coupled_guess:
        row = np.asarray(raw_row, dtype=float).reshape(-1)
        if row.size != 13:
            raise ValueError("Coupled relative guess rows must contain 12 states and time")
        if control_config.carries_attitude:
            augmented.append(
                np.hstack(
                    [
                        row[0:12],
                        mass_start_kg,
                        attitude,
                        row[12],
                        np.zeros(3, dtype=float),
                    ]
                )
            )
        else:
            augmented.append(np.hstack([row[0:12], mass_start_kg, row[12]]))
    return augmented


def _attitude_guess_start(
    phase: Phase,
    control: ThrustControl,
) -> np.ndarray:
    """Return the inherited or configured Euler-angle seed."""
    value = phase.info.get(
        "_attitude_guess_start_rad",
        control.initial_angles_rad,
    )
    return np.asarray(value, dtype=float).reshape(3)


def prepare_phase_guess(
    phase: Phase,
    guess: Sequence[np.ndarray],
    *,
    carries_mass: bool = False,
) -> tuple[list[np.ndarray], StateLayout, str | None]:
    """Shape public guess rows for the selected dynamics implementation."""
    layout = layout_for_phase(
        phase,
        carries_mass=carries_mass,
    )
    propulsion_kind = powered_phase_kind(phase)
    nonlinear_model = nonlinear_relative_model(phase)
    if nonlinear_model is not None:
        rows = [np.asarray(row, dtype=float).reshape(-1) for row in guess]
        compiled_width = layout.state_dim + 1 + layout.control_dim
        if rows and all(row.size == compiled_width for row in rows):
            return rows, layout, propulsion_kind
        coupled_rows = augment_nonlinear_relative_guess(rows, nonlinear_model)
        if propulsion_kind is None and not carries_mass:
            return coupled_rows, layout, None

        spacecraft = phase.spacecraft
        if isinstance(spacecraft, str) or spacecraft is None:
            raise ValueError(f"Mass-carrying phase {phase.name!r} requires a Spacecraft object.")
        if propulsion_kind is None:
            return (
                augment_relative_mass_coast_guess(
                    coupled_rows,
                    phase=phase,
                    mass0_kg=float(spacecraft.initial_mass_kg),
                ),
                layout,
                None,
            )
        thruster = first_thruster(phase)
        return (
            augment_powered_relative_guess(
                coupled_rows,
                relative_guess=rows,
                phase=phase,
                mass0_kg=float(spacecraft.initial_mass_kg),
                thrust_N=float(thruster.thrust_N),
                isp_s=float(thruster.isp_s),
            ),
            layout,
            propulsion_kind,
        )
    if propulsion_kind is None and not carries_mass:
        return [np.asarray(row, dtype=float) for row in guess], layout, None

    rows = [np.asarray(row, dtype=float).reshape(-1) for row in guess]
    compiled_width = layout.state_dim + 1 + layout.control_dim
    if propulsion_kind is not None and rows and all(row.size == compiled_width for row in rows):
        return rows, layout, propulsion_kind
    if (
        propulsion_kind is not None
        and rows
        and layout is not CARTESIAN_MASS_THRUST
        and all(
            row.size == CARTESIAN_MASS_THRUST.state_dim + 1 + CARTESIAN_MASS_THRUST.control_dim
            for row in rows
        )
    ):
        # Specialized low-thrust seeds use the legacy [R,V,M,t,U] layout.
        # Project them back to public [R,V,t] before adding the selected
        # fixed-direction or Euler representation.
        rows = [
            np.hstack(
                [
                    row[0:6],
                    row[CARTESIAN_MASS_THRUST.time_column],
                ]
            )
            for row in rows
        ]
        guess = rows

    spacecraft = phase.spacecraft
    if isinstance(spacecraft, str) or spacecraft is None:
        raise ValueError(f"Mass-carrying phase {phase.name!r} requires a Spacecraft object.")
    if propulsion_kind is None:
        return (
            augment_mass_coast_guess(
                guess,
                phase=phase,
                mass0_kg=float(spacecraft.initial_mass_kg),
            ),
            layout,
            None,
        )

    thruster = first_thruster(phase)
    if float(thruster.thrust_N) <= 0.0 or float(thruster.isp_s) <= 0.0:
        raise ValueError(f"Powered phase {phase.name!r} requires thrust_N > 0 and isp_s > 0.")
    return (
        augment_powered_guess(
            guess,
            phase=phase,
            mass0_kg=float(spacecraft.initial_mass_kg),
            thrust_N=float(thruster.thrust_N),
            isp_s=float(thruster.isp_s),
        ),
        layout,
        propulsion_kind,
    )


def augment_nonlinear_relative_guess(
    relative_guess: Sequence[np.ndarray],
    model: NonlinearRelative,
) -> list[np.ndarray]:
    """Convert public RIC seed rows into the selected native formulation.

    CWH supplies a fast, smooth initial guess only.  The compiled ODE and all
    constraints remain full nonlinear dynamics.
    """
    augmented: list[np.ndarray] = []
    for raw_row in relative_guess:
        row = np.asarray(raw_row, dtype=float).reshape(-1)
        if row.size != 7:
            raise ValueError(
                "A nonlinear relative initial guess must contain "
                "[R, I, C, Rdot, Idot, Cdot, t] rows"
            )
        mode = model.propagation_mode
        if mode is RelativePropagationMode.NONLINEAR_RIC:
            augmented.append(row.copy())
            continue
        chief = circular_chief_state(
            model.chief_initial_state_eci,
            float(row[6]),
            model.mean_motion_radps,
        )
        relative = BoundaryState(row[0:3], row[3:6])
        if mode is RelativePropagationMode.COUPLED_RIC:
            augmented.append(
                np.hstack(
                    [
                        chief.r_m,
                        chief.v_mps,
                        relative.r_m,
                        relative.v_mps,
                        float(row[6]),
                    ]
                )
            )
            continue
        if mode is RelativePropagationMode.DAMICO:
            elements = relative_state_to_relative_orbital_elements(
                chief,
                relative,
                mu_m3ps2=model.central_body.mu_m3ps2,
            )
            augmented.append(np.hstack([elements.as_vector(), float(row[6])]))
            continue
        if mode is RelativePropagationMode.CLASSICAL_ELEMENTS:
            elements = relative_state_to_classical_relative_orbital_elements(
                chief,
                relative,
                mu_m3ps2=model.central_body.mu_m3ps2,
            )
            augmented.append(np.hstack([elements.as_vector(), float(row[6])]))
            continue
        deputy = relative_to_inertial_state(chief, relative)
        augmented.append(
            np.hstack(
                [
                    chief.r_m,
                    chief.v_mps,
                    deputy.r_m,
                    deputy.v_mps,
                    float(row[6]),
                ]
            )
        )
    return augmented


def make_asset_phase(
    phase: Phase,
    guess: Sequence[np.ndarray],
    nsegs: int,
    *,
    carries_mass: bool = False,
    third_body_tables: dict[str, ThirdBodyTable] | None = None,
):
    """Compile one user phase into an ASSET phase plus dimensional metadata."""
    prepared_guess, layout, propulsion_kind = prepare_phase_guess(
        phase,
        guess,
        carries_mass=carries_mass,
    )
    ode = ode_for_phase(
        phase,
        carries_mass=carries_mass,
        third_body_tables=tables_for_phase(phase, third_body_tables or {}),
    )
    asset_phase = ode.phase(Tmodes.LGL3, prepared_guess, int(nsegs))
    return asset_phase, layout, propulsion_kind
