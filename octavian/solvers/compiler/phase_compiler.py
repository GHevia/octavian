"""Phase classification, dynamics selection, and ASSET phase construction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..._asset import Tmodes
from ...astro.types import as_vec3
from ...dynamics import (
    ChemicalBurnECI,
    MassCoastECI,
    PerturbedECI,
    ThirdBodyTable,
    TwoBodyECI,
)
from ...phase import Phase
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
    state_dim: int = 6
    control_dim: int = 0
    is_chemical_burn: bool = False
    enable_adaptive_mesh: bool = True


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
    normalized_mode = (getattr(phase, "mode", "") or "").strip().lower().replace("-", "_")
    return normalized_mode in ("burn", "chemical_burn", "finite_burn")


def is_coast_like(phase: Phase) -> bool:
    """Return whether a phase uses coast-like translational dynamics."""
    normalized_mode = (getattr(phase, "mode", "") or "").strip().lower().replace("-", "_")
    return normalized_mode in ("coast", "transfer", "rendezvous")


def mass_state_phase_indices(phases: Sequence[Phase]) -> set[int]:
    """Return phase indices that carry mass between finite burns."""
    burn_indices = [idx for idx, phase in enumerate(phases) if is_chemical_burn(phase)]
    if not burn_indices:
        return set()

    mass_indices = set(burn_indices)
    first_burn = burn_indices[0]
    last_burn = burn_indices[-1]
    for idx in range(first_burn + 1, last_burn):
        if is_coast_like(phases[idx]):
            mass_indices.add(idx)
    return mass_indices


def validate_chemical_burn_transfer(phases: Sequence[Phase]) -> None:
    """Validate the currently supported burn-coast-burn mission shape."""
    burn_indices = [idx for idx, phase in enumerate(phases) if is_chemical_burn(phase)]
    if not burn_indices:
        return
    if len(phases) < 3 or len(burn_indices) < 2:
        raise ValueError(
            "Chemical burn transfers require at least three phases: "
            "a departure burn, a coast, and an arrival burn."
        )
    first_burn = burn_indices[0]
    last_burn = burn_indices[-1]
    if first_burn != 0 or last_burn != len(phases) - 1:
        raise ValueError(
            "Chemical burn transfers must start with a burn phase and end with a burn phase."
        )
    if not any(is_coast_like(phases[idx]) for idx in range(first_burn + 1, last_burn)):
        raise ValueError("Chemical burn transfers require a coast phase between the burns.")


def first_thruster(phase: Phase):
    """Return the configured thruster for a powered phase."""
    spacecraft = getattr(phase, "spacecraft", None)
    if isinstance(spacecraft, str) or spacecraft is None:
        raise ValueError(f"Chemical burn phase {phase.name!r} requires a Spacecraft object.")
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
    if is_chemical_burn(phase):
        thruster = first_thruster(phase)
        return ChemicalBurnECI(
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
    if is_chemical_burn(phase):
        return 7, 3, True
    return 6, 0, False


def compile_phase_dimensions(
    phase: Phase,
    *,
    carries_mass: bool = False,
) -> tuple[int, int, bool]:
    """Return the state/control dimensions required by the compiled ODE."""
    if is_chemical_burn(phase):
        return 7, 3, True
    if carries_mass:
        return 7, 0, False
    return 6, 0, False


def trajectory_rvt(raw_traj: np.ndarray, state_dim: int) -> np.ndarray:
    """Return the public ``[R, V, t]`` view of an ASSET trajectory."""
    raw = np.asarray(raw_traj, dtype=float)
    time_column = int(state_dim)
    if raw.shape[1] <= time_column:
        raise ValueError("ASSET trajectory is missing the phase time column.")
    return raw[:, [0, 1, 2, 3, 4, 5, time_column]]


def augment_chemical_burn_guess(
    base_guess: Sequence[np.ndarray],
    *,
    phase: Phase,
    mass0_kg: float,
    thrust_N: float,
    isp_s: float,
) -> list[np.ndarray]:
    """Convert public ``[R,V,t]`` rows into ``[R,V,M,t,U]`` burn rows."""
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
) -> tuple[list[np.ndarray], int, int, bool]:
    """Shape public guess rows for the selected dynamics implementation."""
    state_dim, control_dim, is_burn = compile_phase_dimensions(
        phase,
        carries_mass=carries_mass,
    )
    if not carries_mass:
        return [np.asarray(row, dtype=float) for row in guess], state_dim, control_dim, is_burn

    spacecraft = phase.spacecraft
    if isinstance(spacecraft, str) or spacecraft is None:
        raise ValueError(f"Mass-carrying phase {phase.name!r} requires a Spacecraft object.")
    if not is_burn:
        return (
            augment_mass_coast_guess(
                guess,
                phase=phase,
                mass0_kg=float(spacecraft.initial_mass_kg),
            ),
            state_dim,
            control_dim,
            is_burn,
        )

    thruster = first_thruster(phase)
    if float(thruster.thrust_N) <= 0.0 or float(thruster.isp_s) <= 0.0:
        raise ValueError(
            f"Chemical burn phase {phase.name!r} requires thrust_N > 0 and isp_s > 0."
        )
    return (
        augment_chemical_burn_guess(
            guess,
            phase=phase,
            mass0_kg=float(spacecraft.initial_mass_kg),
            thrust_N=float(thruster.thrust_N),
            isp_s=float(thruster.isp_s),
        ),
        state_dim,
        control_dim,
        is_burn,
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
    prepared_guess, state_dim, control_dim, is_burn = prepare_phase_guess(
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
    return asset_phase, state_dim, control_dim, is_burn
