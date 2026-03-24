"""ConOps (Concept of Operations) builders.

ConOps is the "middle" layer between `quick` templates and fully custom
missions. It returns a normal `Mission` object that users can modify before
solving.

v0.x ships ConOps helpers for the built-in impulsive rendezvous solver.
"""

from __future__ import annotations

from collections.abc import Sequence

from .mission import Mission
from .models import Dynamics
from .phase import Phase
from .spacecraft import Spacecraft
from .specs import BoundaryState


def rendezvous_two_impulse(
    *,
    spacecraft: Spacecraft,
    dynamics: Dynamics,
    initial_state: BoundaryState,
    final_state: BoundaryState,
    tf_bounds_s: tuple[float, float] = (600.0, 7200.0),
    nsegs: int = 60,
    lambert_grid_size: int = 60,
    nrevs_to_try: tuple[int, ...] = (0, 1),
    w_time: float = 0.0,
    name: str = "Two-impulse rendezvous",
    constraints: Sequence[object] | None = None,
) -> Mission:
    """Build a single-phase impulsive rendezvous mission.

    Args:
        spacecraft: Spacecraft used by the mission.
        dynamics: Dynamics model applied to the phase.
        initial_state: Initial state fixed at the front boundary.
        final_state: Final state target applied at the back boundary.
        tf_bounds_s: Bounds on total transfer time in seconds.
        nsegs: Number of transfer mesh segments.
        lambert_grid_size: Number of Lambert seed samples.
        nrevs_to_try: Revolution counts included in the Lambert seed search.
        w_time: Weight on final time in the objective.
        name: Human-readable mission name.
        constraints: Additional constraints for the rendezvous phase.

    Returns:
        A mission configured for the built-in two-impulse rendezvous solver.
    """

    ph = Phase(
        name="rendezvous",
        mode="rendezvous",
        spacecraft=spacecraft,
        dynamics=dynamics,
        initial_state=initial_state,
        final_state=final_state,
        tof_bounds_s=tf_bounds_s,
        constraints=list(constraints or []),
    )

    return Mission(
        phases=[ph],
        name=name,
        mesh_nsegs_transfer=int(nsegs),
        lambert_grid_size=int(lambert_grid_size),
        nrevs_to_try=tuple(int(x) for x in nrevs_to_try),
        w_time=float(w_time),
    )


def rendezvous_precoast_then_transfer(
    *,
    spacecraft: Spacecraft,
    dynamics: Dynamics,
    initial_state: BoundaryState,
    final_state: BoundaryState,
    t1_bounds_s: tuple[float, float] = (0.0, 1800.0),
    tf_bounds_s: tuple[float, float] = (600.0, 7200.0),
    nsegs_precoast: int = 30,
    nsegs_transfer: int = 60,
    precoast_grid_size: int = 10,
    limit_precoast_to_one_period: bool = True,
    lambert_grid_size: int = 60,
    nrevs_to_try: tuple[int, ...] = (0, 1),
    w_time: float = 0.0,
    name: str = "Precoast + two-impulse rendezvous",
    constraints_precoast: Sequence[object] | None = None,
    constraints_rendezvous: Sequence[object] | None = None,
) -> Mission:
    """Build a two-phase mission with a precoast before rendezvous.

    Args:
        spacecraft: Spacecraft used by the mission.
        dynamics: Dynamics model shared by both phases.
        initial_state: Initial state fixed at the start of the precoast phase.
        final_state: Final state target applied at the end of the transfer phase.
        t1_bounds_s: Bounds on precoast duration in seconds.
        tf_bounds_s: Bounds on the final mission time in seconds.
        nsegs_precoast: Number of mesh segments for the precoast phase.
        nsegs_transfer: Number of mesh segments for the transfer phase.
        precoast_grid_size: Number of precoast seed candidates to evaluate.
        limit_precoast_to_one_period: Whether to cap the seed sweep to one orbital
            period when the initial orbit period is known.
        lambert_grid_size: Number of Lambert seed samples per candidate.
        nrevs_to_try: Revolution counts included in the Lambert seed search.
        w_time: Weight on final time in the objective.
        name: Human-readable mission name.
        constraints_precoast: Additional constraints for the precoast phase.
        constraints_rendezvous: Additional constraints for the transfer phase.

    Returns:
        A mission with a precoast phase followed by an impulsive rendezvous phase.
    """

    p0 = Phase(
        name="precoast",
        mode="coast",
        spacecraft=spacecraft,
        dynamics=dynamics,
        initial_state=initial_state,
        tof_bounds_s=t1_bounds_s,
        constraints=list(constraints_precoast or []),
    )

    p1 = Phase(
        name="rendezvous",
        mode="rendezvous",
        previous=p0,
        dynamics=dynamics,
        spacecraft=spacecraft,
        final_state=final_state,
        tof_bounds_s=tf_bounds_s,
        constraints=list(constraints_rendezvous or []),
    )

    return Mission(
        phases=[p0, p1],
        name=name,
        mesh_nsegs_precoast=int(nsegs_precoast),
        mesh_nsegs_transfer=int(nsegs_transfer),
        precoast_grid_size=int(precoast_grid_size),
        limit_precoast_to_one_period=bool(limit_precoast_to_one_period),
        lambert_grid_size=int(lambert_grid_size),
        nrevs_to_try=tuple(int(x) for x in nrevs_to_try),
        w_time=float(w_time),
    )
