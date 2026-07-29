"""Low-friction mission builders."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import numpy as np

from . import constraints as mission_constraints
from . import links, objectives, variables
from .bodies import EARTH, CelestialBody
from .conops import rendezvous_precoast_then_transfer, rendezvous_two_impulse
from .mission import Mission
from .models import Dynamics, Perturbations
from .phase import Phase
from .phase import state as _state
from .solvers import SolverOptions
from .spacecraft import Spacecraft, Thruster
from .specs import BoundaryState

TimeBounds = tuple[float, float]
TimeBoundsInput = TimeBounds | Sequence[TimeBounds]


def state(
    r_m: np.ndarray | Sequence[float],
    v_mps: np.ndarray | Sequence[float],
) -> BoundaryState:
    """Create a boundary state from Cartesian position and velocity vectors.

    Args:
        r_m: Position vector in meters.
        v_mps: Velocity vector in meters per second.

    Returns:
        A boundary-state object that can be passed to quick builders, ConOps
        helpers, or lower-level specs.
    """
    return _state(r_m, v_mps)


def two_burn_rendezvous(
    x0: BoundaryState,
    xf: BoundaryState,
    *,
    mu_m3ps2: float = 3.986004418e14,
    central_body: CelestialBody | str | None = None,
    tf_bounds_s: tuple[float, float] = (600.0, 7200.0),
    nsegs: int = 60,
    lambert_grid_size: int = 60,
    nrevs_to_try: tuple[int, ...] = (0, 1),
    w_time: float = 0.0,
    precoast: bool = False,
    t1_bounds_s: tuple[float, float] = (0.0, 1800.0),
    precoast_grid_size: int = 10,
    limit_precoast_to_one_period: bool = True,
    name: str = "Two-burn rendezvous",
    constraints: Sequence[object] | None = None,
    solver_options: SolverOptions | None = None,
) -> Mission:
    """Create a ready-to-solve impulsive rendezvous mission.

    Args:
        x0: Initial boundary state.
        xf: Final boundary state.
        mu_m3ps2: Central-body gravitational parameter in m^3/s^2.
        central_body: Optional built-in or custom body. When provided, its
            gravity, radius, J2, and inertial frame replace ``mu_m3ps2`` and
            the legacy Earth defaults.
        tf_bounds_s: Bounds on the final rendezvous time in seconds.
        nsegs: Number of mesh segments used by the transfer phase.
        lambert_grid_size: Number of Lambert time-of-flight samples to try.
        nrevs_to_try: Revolution counts to include in the Lambert seed search.
        w_time: Weight on final time in the objective.
        precoast: If ``True``, build a precoast-plus-transfer mission instead
            of a single transfer phase.
        t1_bounds_s: Bounds on precoast duration in seconds when ``precoast``
            is enabled.
        precoast_grid_size: Number of precoast candidates to test when seeding.
        limit_precoast_to_one_period: Whether to cap the precoast seed sweep to
            one orbital period when possible.
        name: Human-readable mission name.
        constraints: Additional constraints attached to the rendezvous phase.
        solver_options: Optional solver overrides attached to the mission.

    Returns:
        A configured mission object ready for ``mission.solve()``.
    """
    default_thruster = Thruster(name="main")
    spacecraft = Spacecraft(name="SC", dry_mass_kg=0.0, thrusters=[default_thruster])
    dynamics = (
        Dynamics.for_body(central_body)
        if central_body is not None
        else Dynamics(mu_m3ps2=float(mu_m3ps2))
    )
    if not precoast:
        mission = rendezvous_two_impulse(
            spacecraft=spacecraft,
            dynamics=dynamics,
            initial_state=x0,
            final_state=xf,
            tf_bounds_s=tf_bounds_s,
            nsegs=nsegs,
            lambert_grid_size=lambert_grid_size,
            nrevs_to_try=nrevs_to_try,
            w_time=w_time,
            name=name,
            constraints=constraints,
        )
    else:
        mission = rendezvous_precoast_then_transfer(
            spacecraft=spacecraft,
            dynamics=dynamics,
            initial_state=x0,
            final_state=xf,
            t1_bounds_s=t1_bounds_s,
            tf_bounds_s=tf_bounds_s,
            nsegs_precoast=max(10, int(nsegs // 2)),
            nsegs_transfer=int(nsegs),
            precoast_grid_size=precoast_grid_size,
            limit_precoast_to_one_period=limit_precoast_to_one_period,
            lambert_grid_size=lambert_grid_size,
            nrevs_to_try=nrevs_to_try,
            w_time=w_time,
            name=name,
            constraints_rendezvous=constraints,
        )

    if solver_options is not None:
        mission.solver_options = solver_options
    return mission


def relative_hop(
    initial_state_ric: BoundaryState,
    final_state_ric: BoundaryState,
    *,
    chief_initial_state_eci: BoundaryState,
    departure_coast_time_bounds_s: TimeBounds = (1.0, 600.0),
    transfer_time_bounds_s: TimeBounds = (600.0, 3_600.0),
    dynamics: Dynamics | None = None,
    central_body: CelestialBody | str = EARTH,
    propagation_mode: str = "coupled_eci",
    perturbations: Perturbations | None = None,
    initial_epoch: str | datetime | float | None = None,
    spacecraft: Spacecraft | None = None,
    reference_length_m: float = 1_000.0,
    nsegs_coast: int = 30,
    nsegs_transfer: int = 60,
    seed_grid_size: int = 60,
    w_time: float = 0.0,
    name: str = "Relative hop",
    solver_options: SolverOptions | None = None,
) -> Mission:
    """Build a relative coast–impulse–transfer–impulse mission.

    The first phase lets the deputy coast on its initial relative orbit.  An
    impulsive link starts the transfer phase, and a terminal impulse matches
    the requested final RIC velocity.  Both optimized dynamics phases remain
    chief-centered and use the same native relative formulation.

    Args:
        initial_state_ric: Initial deputy state in the chief RIC frame.
        final_state_ric: Desired post-arrival deputy state in RIC.
        chief_initial_state_eci: Absolute chief state at mission time zero.
        departure_coast_time_bounds_s: Bounds on the initial coast duration.
        transfer_time_bounds_s: Bounds on transfer duration after departure.
        dynamics: Optional preconfigured relative dynamics. When omitted,
            ``Dynamics.relative`` is built from the remaining dynamics inputs.
        central_body: Central body used when constructing default dynamics.
        propagation_mode: Relative formulation used by default dynamics.
        perturbations: Optional differential perturbation configuration.
        initial_epoch: Mission epoch required by Sun, Moon, or SRP.
        spacecraft: Optional deputy configuration.
        reference_length_m: Relative-distance scaling for default dynamics.
        nsegs_coast: Mesh segments on the initial coast.
        nsegs_transfer: Mesh segments on the transfer.
        seed_grid_size: Number of CWH seed times evaluated per target.
        w_time: Optional total-time objective weight.
        name: Human-readable mission name.
        solver_options: Optional solver configuration.

    Returns:
        A two-phase mission ready for ``mission.solve()``.
    """
    return relative_transfer_chain(
        initial_state_ric,
        [final_state_ric],
        chief_initial_state_eci=chief_initial_state_eci,
        departure_coast_time_bounds_s=departure_coast_time_bounds_s,
        transfer_time_bounds_s=transfer_time_bounds_s,
        coast_time_bounds_s=(),
        dynamics=dynamics,
        central_body=central_body,
        propagation_mode=propagation_mode,
        perturbations=perturbations,
        initial_epoch=initial_epoch,
        spacecraft=spacecraft,
        reference_length_m=reference_length_m,
        nsegs_coast=nsegs_coast,
        nsegs_transfer=nsegs_transfer,
        seed_grid_size=seed_grid_size,
        w_time=w_time,
        name=name,
        solver_options=solver_options,
    )


def relative_transfer_chain(
    initial_state_ric: BoundaryState,
    target_states_ric: Sequence[BoundaryState],
    *,
    chief_initial_state_eci: BoundaryState,
    transfer_time_bounds_s: TimeBoundsInput = (600.0, 3_600.0),
    coast_time_bounds_s: TimeBoundsInput | None = None,
    departure_coast_time_bounds_s: TimeBounds | None = None,
    dynamics: Dynamics | None = None,
    central_body: CelestialBody | str = EARTH,
    propagation_mode: str = "coupled_eci",
    perturbations: Perturbations | None = None,
    initial_epoch: str | datetime | float | None = None,
    spacecraft: Spacecraft | None = None,
    reference_length_m: float = 1_000.0,
    nsegs_coast: int = 30,
    nsegs_transfer: int = 60,
    seed_grid_size: int = 60,
    w_time: float = 0.0,
    name: str = "Relative transfer chain",
    solver_options: SolverOptions | None = None,
) -> Mission:
    """Build one or more impulsive relative transfers with coasts between.

    Each target is a post-arrival RIC state.  For nonterminal targets, an
    impulsive link inserts the arrival burn and starts a natural coast at that
    exact state.  A later impulsive link departs that coast for the next
    transfer.  This yields four burns for two ordinary two-burn transfers, or
    can be adapted into a three-burn design by editing the returned phases.

    A single time-bound pair is repeated for every applicable phase.  Pass a
    sequence of pairs to configure individual transfers or intermediate
    coasts.

    Args:
        initial_state_ric: Initial deputy state in the chief RIC frame.
        target_states_ric: Ordered post-arrival RIC targets.
        chief_initial_state_eci: Absolute chief state at mission time zero.
        transfer_time_bounds_s: One duration bound or one per target.
        coast_time_bounds_s: One duration bound or one per intermediate target.
            Defaults to ``(300, 900)`` when multiple targets are supplied.
        departure_coast_time_bounds_s: Optional coast before the first burn.
        dynamics: Optional preconfigured relative dynamics.
        central_body: Central body used when constructing default dynamics.
        propagation_mode: Relative formulation used by default dynamics.
        perturbations: Optional differential perturbation configuration.
        initial_epoch: Mission epoch required by Sun, Moon, or SRP.
        spacecraft: Optional deputy configuration.
        reference_length_m: Relative-distance scaling for default dynamics.
        nsegs_coast: Mesh segments used for coast phases.
        nsegs_transfer: Mesh segments used for transfer phases.
        seed_grid_size: Number of CWH seed times evaluated per target.
        w_time: Optional total-time objective weight.
        name: Human-readable mission name.
        solver_options: Optional solver configuration.

    Returns:
        A linked relative mission ready for ``mission.solve()``.

    Raises:
        ValueError: If no targets are supplied, bounds counts are inconsistent,
            or perturbations are supplied alongside preconfigured dynamics.
    """
    targets = list(target_states_ric)
    if not targets:
        raise ValueError("target_states_ric must contain at least one target")
    transfer_bounds = _expand_time_bounds(
        transfer_time_bounds_s,
        count=len(targets),
        name="transfer_time_bounds_s",
    )
    coast_count = max(len(targets) - 1, 0)
    intermediate_coast_bounds = _expand_time_bounds(
        (300.0, 900.0) if coast_time_bounds_s is None else coast_time_bounds_s,
        count=coast_count,
        name="coast_time_bounds_s",
    )
    if dynamics is not None and perturbations is not None:
        raise ValueError(
            "Pass perturbations through the preconfigured dynamics, or omit "
            "dynamics and let relative_transfer_chain construct it."
        )
    relative_dynamics = dynamics or Dynamics.relative(
        chief_initial_state_eci=chief_initial_state_eci,
        central_body=central_body,
        reference_length_m=reference_length_m,
        propagation_mode=propagation_mode,
        perturbations=perturbations,
    )
    if relative_dynamics.frame.kind != "relative":
        raise ValueError("relative_transfer_chain requires relative dynamics")
    deputy = spacecraft or Spacecraft(name="Deputy", dry_mass_kg=250.0)

    phases: list[Phase] = []
    previous: Phase | None = None
    if departure_coast_time_bounds_s is not None:
        departure_coast = Phase(
            name="departure_coast",
            mode="relative_coast",
            spacecraft=deputy,
            dynamics=relative_dynamics,
            initial_state=initial_state_ric,
            tof_bounds_s=_validated_time_bounds(
                departure_coast_time_bounds_s,
                name="departure_coast_time_bounds_s",
            ),
            tof_is_relative=True,
            constraints=[
                mission_constraints.state(initial_state_ric, where="Front"),
            ],
        )
        phases.append(departure_coast)
        previous = departure_coast

    for target_index, (target, bounds) in enumerate(
        zip(targets, transfer_bounds, strict=True),
        start=1,
    ):
        is_last_target = target_index == len(targets)
        transfer_constraints = []
        if previous is None:
            transfer_constraints.append(
                mission_constraints.state(initial_state_ric, where="Front")
            )
        if is_last_target:
            transfer_constraints.append(
                mission_constraints.state(target, where="Back")
            )
        transfer_variables = [variables.impulsive_delta_v(at="Front")]
        if is_last_target:
            transfer_variables.append(variables.impulsive_delta_v(at="Back"))
        transfer = Phase(
            name=f"transfer_{target_index}",
            mode="relative_coast",
            spacecraft=deputy,
            dynamics=relative_dynamics,
            previous=previous,
            link=links.impulsive() if previous is not None else None,
            initial_state=initial_state_ric if previous is None else None,
            final_state=target,
            tof_bounds_s=bounds,
            tof_is_relative=True,
            constraints=transfer_constraints,
            variables=transfer_variables,
        )
        phases.append(transfer)
        previous = transfer

        if not is_last_target:
            target_coast = Phase(
                name=f"target_{target_index}_coast",
                mode="relative_coast",
                spacecraft=deputy,
                dynamics=relative_dynamics,
                previous=transfer,
                link=links.impulsive(),
                initial_state=target,
                tof_bounds_s=intermediate_coast_bounds[target_index - 1],
                tof_is_relative=True,
                constraints=[
                    mission_constraints.state(target, where="Front"),
                ],
                variables=[variables.impulsive_delta_v(at="Front")],
            )
            phases.append(target_coast)
            previous = target_coast

    mission_objectives = [objectives.minimize_total_delta_v()]
    if float(w_time) != 0.0:
        mission_objectives.append(objectives.minimize_total_time(weight=float(w_time)))
    return Mission(
        phases=phases,
        name=name,
        initial_epoch=initial_epoch,
        objectives=mission_objectives,
        mesh_nsegs_precoast=int(nsegs_coast),
        mesh_nsegs_transfer=int(nsegs_transfer),
        lambert_grid_size=int(seed_grid_size),
        w_time=float(w_time),
        solver_options=solver_options or SolverOptions(),
    )


def _expand_time_bounds(
    value: TimeBoundsInput,
    *,
    count: int,
    name: str,
) -> list[TimeBounds]:
    """Normalize one or many phase-duration bounds."""
    if count == 0:
        if len(value) == 0:
            return []
        if _is_time_bounds(value):
            return []
        raise ValueError(f"{name} is not applicable when there are no phases")
    if _is_time_bounds(value):
        return [_validated_time_bounds(value, name=name)] * count
    bounds = list(value)
    if len(bounds) != count:
        raise ValueError(f"{name} must contain {count} bound pairs")
    return [
        _validated_time_bounds(bound, name=f"{name}[{index}]")
        for index, bound in enumerate(bounds)
    ]


def _is_time_bounds(value: object) -> bool:
    """Return whether a value is one numeric lower/upper pair."""
    try:
        items = list(value)  # type: ignore[arg-type]
    except TypeError:
        return False
    return len(items) == 2 and all(np.isscalar(item) for item in items)


def _validated_time_bounds(value: Sequence[float], *, name: str) -> TimeBounds:
    """Return a finite, strictly increasing phase-duration pair."""
    lower_s, upper_s = (float(item) for item in value)
    if not np.isfinite(lower_s) or not np.isfinite(upper_s):
        raise ValueError(f"{name} must contain finite values")
    if not (upper_s > lower_s >= 0.0):
        raise ValueError(f"{name} must satisfy 0 <= lower < upper")
    return lower_s, upper_s
