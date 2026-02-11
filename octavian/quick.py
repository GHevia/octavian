from __future__ import annotations

"""Quick templates.

Quick templates are the lowest-friction entry point:
  - callable in ~10 lines
  - return a normal `Mission`
  - use safe defaults (plan/retry/config) without exposing complexity
"""

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from .conops import rendezvous_precoast_then_transfer, rendezvous_two_impulse
from .models import Dynamics
from .mission import Mission
from .phase import state as _state
from .spacecraft import Spacecraft, Thruster
from .specs import BoundaryState
from .solvers import SolverOptions


def state(
    r_m: Union[np.ndarray, Sequence[float]],
    v_mps: Union[np.ndarray, Sequence[float]],
) -> BoundaryState:
    """Helper to build a :class:`~octavian.specs.BoundaryState`."""

    return _state(r_m, v_mps)


def two_burn_rendezvous(
    x0: BoundaryState,
    xf: BoundaryState,
    *,
    mu_m3ps2: float = 3.986004418e14,
    tf_bounds_s: Tuple[float, float] = (600.0, 7200.0),
    nsegs: int = 60,
    lambert_grid_size: int = 60,
    nrevs_to_try: Tuple[int, ...] = (0, 1),
    w_time: float = 0.0,
    precoast: bool = False,
    t1_bounds_s: Tuple[float, float] = (0.0, 1800.0),
    precoast_grid_size: int = 10,
    limit_precoast_to_one_period: bool = True,
    name: str = "Two-burn rendezvous",
    constraints: Optional[Sequence[object]] = None,
    solver_options: Optional[SolverOptions] = None,
) -> Mission:
    """Create a ready-to-solve two-impulse rendezvous mission."""

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
