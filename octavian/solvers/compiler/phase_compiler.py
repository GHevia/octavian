"""Phase classification, dynamics selection, and ASSET phase construction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..._asset import Tmodes
from ...astro.types import as_vec3
from ...coordinates import (
    CARTESIAN,
    CARTESIAN_MASS,
    CARTESIAN_MASS_THRUST,
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
from ...relative import ClohessyWiltshire
from ...relative.dynamics import ClohessyWiltshireODE
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
    location = (
        "Front"
        if normalized_where in ("front", "start", "initial", "t0")
        else "Back"
    )
    for variable in getattr(phase, "variables", []) or []:
        if (
            isinstance(variable, ImpulsiveDeltaV)
            and getattr(variable, "where", "") == location
        ):
            return True
    for event in getattr(phase, "events", []) or []:
        if (
            getattr(event, "kind", "") == "impulse"
            and getattr(event, "where", "") == location
        ):
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


def is_coast_like(phase: Phase) -> bool:
    """Return whether a phase uses coast-like translational dynamics."""
    normalized_mode = (getattr(phase, "mode", "") or "").strip().lower().replace("-", "_")
    return normalized_mode in ("coast", "transfer", "rendezvous", "relative_coast", "cwh")


def cwh_model(phase: Phase) -> ClohessyWiltshire | None:
    """Return the phase's CWH model, when configured."""
    dynamics = phase.dynamics
    model = dynamics.model if dynamics is not None else None
    return model if isinstance(model, ClohessyWiltshire) else None


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
    for idx in range(powered_indices[0], powered_indices[-1] + 1):
        phase = phases[idx]
        if not (is_powered_phase(phase) or is_coast_like(phase)):
            raise ValueError(
                "Phases between powered phases must use powered or coast-like dynamics; "
                f"phase {phase.name!r} has mode={phase.mode!r}."
            )
        spacecraft = getattr(phase, "spacecraft", None)
        if isinstance(spacecraft, str) or spacecraft is None:
            raise ValueError(
                f"Mass-carrying phase {phase.name!r} requires a Spacecraft object."
            )
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
        if any(
            (
                perturbations.j2,
                perturbations.moon,
                perturbations.sun,
                perturbations.srp,
                perturbations.drag,
                bool(perturbations.third_bodies),
                bool(third_body_tables),
            )
        ):
            raise ValueError(
                "CWH phases currently support the unforced linear model only; "
                "relative-frame perturbations require an explicit acceleration model."
            )
        return ClohessyWiltshireODE(
            mean_motion_radps=relative_model.mean_motion_radps
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
        )
    if carries_mass:
        return MassCoastECI(
            mu_m3ps2=float(dynamics.mu_m3ps2),
            j2=bool(perturbations.j2),
            central_body_radius_m=float(dynamics.central_body_radius_m),
            j2_coefficient=float(dynamics.j2_coefficient),
            third_body_tables=tuple(third_body_tables),
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
    if is_powered_phase(phase):
        return CARTESIAN_MASS_THRUST
    if carries_mass:
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


def trajectory_rvt(raw_traj: np.ndarray, layout: StateLayout | int) -> np.ndarray:
    """Return the public ``[R, V, t]`` view of an ASSET trajectory."""
    raw = np.asarray(raw_traj, dtype=float)
    if isinstance(layout, StateLayout):
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
    """Convert public ``[R,V,t]`` rows into ``[R,V,M,t,U]`` powered rows."""
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
    for index, row in enumerate(rows):
        fraction = index / max(len(rows) - 1, 1)
        mass = max(
            mass_start_kg - mass_flow_kgps * throttle * duration_s * fraction,
            1.0,
        )
        augmented.append(np.hstack([row[0:6], mass, row[6], control]))
    return augmented


# Compatibility name retained for focused downstream tests and imports.
augment_chemical_burn_guess = augment_powered_guess


def augment_mass_coast_guess(
    base_guess: Sequence[np.ndarray],
    *,
    phase: Phase,
    mass0_kg: float,
) -> list[np.ndarray]:
    """Add a constant mass state to public coast-guess rows."""
    mass_start_kg = float(getattr(phase, "info", {}).get("_mass_guess_start_kg", mass0_kg))
    augmented: list[np.ndarray] = []
    for row in base_guess:
        rvt = np.asarray(row, dtype=float).reshape(-1)
        augmented.append(np.hstack([rvt[0:6], mass_start_kg, rvt[6]]))
    return augmented


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
    if propulsion_kind is None and not carries_mass:
        return [np.asarray(row, dtype=float) for row in guess], layout, None

    rows = [np.asarray(row, dtype=float).reshape(-1) for row in guess]
    compiled_width = layout.state_dim + 1 + layout.control_dim
    if propulsion_kind is not None and rows and all(row.size == compiled_width for row in rows):
        return rows, layout, propulsion_kind

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
        raise ValueError(
            f"Powered phase {phase.name!r} requires thrust_N > 0 and isp_s > 0."
        )
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
