"""Quick templates.

Quick templates are the lowest-friction entry point:
  - callable in ~10 lines
  - return a normal `Mission`
  - use safe defaults (plan/retry/config) without exposing complexity
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .conops import rendezvous_precoast_then_transfer, rendezvous_two_impulse
from .mission import Mission
from .models import Dynamics
from .phase import state as _state
from .solvers import SolverOptions
from .spacecraft import Spacecraft, Thruster
from .specs import BoundaryState


def state(
    r_m: np.ndarray | Sequence[float],
    v_mps: np.ndarray | Sequence[float],
) -> BoundaryState:
    """Build a boundary state from position and velocity vectors.

    Args:
        r_m: Position vector in meters.
        v_mps: Velocity vector in meters per second.

    Returns:
        A boundary-state object that can be passed to quick builders,
        ConOps helpers, or lower-level specs.
    """

    return _state(r_m, v_mps)


def two_burn_rendezvous(
    x0: BoundaryState,
    xf: BoundaryState,
    *,
    mu_m3ps2: float = 3.986004418e14,
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
    """Create a ready-to-solve two-burn rendezvous mission.

    Args:
        x0: Initial boundary state.
        xf: Final boundary state.
        mu_m3ps2: Central-body gravitational parameter in m^3/s^2.
        tf_bounds_s: Bounds on the final rendezvous time in seconds.
        nsegs: Number of mesh segments used by the transfer phase.
        lambert_grid_size: Number of Lambert time-of-flight samples to try.
        nrevs_to_try: Revolution counts to include in the Lambert seed search.
        w_time: Weight on final time in the objective.
        precoast: If ``True``, build a precoast-plus-transfer mission instead of a
            single transfer phase.
        t1_bounds_s: Bounds on precoast duration in seconds when ``precoast`` is enabled.
        precoast_grid_size: Number of precoast candidates to test when seeding.
        limit_precoast_to_one_period: Whether to cap the precoast seed sweep to one
            orbital period when possible.
        name: Human-readable mission name.
        constraints: Additional constraints attached to the rendezvous phase.
        solver_options: Optional solver overrides attached to the mission.

    Returns:
        A configured mission object ready for ``mission.solve()``.
    """

    # Minimal, readable defaults (these are metadata for v0.x solvers)
    thruster = Thruster(name="main")
    sc = Spacecraft(name="SC", dry_mass_kg=0.0, thrusters=[thruster])
    dyn = Dynamics(mu_m3ps2=float(mu_m3ps2))

    if not precoast:
        m = rendezvous_two_impulse(
            spacecraft=sc,
            dynamics=dyn,
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
        if solver_options is not None:
            m.solver_options = solver_options
        return m

    m = rendezvous_precoast_then_transfer(
        spacecraft=sc,
        dynamics=dyn,
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
        m.solver_options = solver_options
    return m
