from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoundaryState:
    """Dimensional Cartesian boundary state in ECI.

    Attributes:
        r_m: Position in meters, shape (3,).
        v_mps: Velocity in meters per second, shape (3,).
    """
    r_m: np.ndarray
    v_mps: np.ndarray
    def __post_init__(self) -> None:
        object.__setattr__(self, "r_m", np.asarray(self.r_m, dtype=float).reshape(3))
        object.__setattr__(self, "v_mps", np.asarray(self.v_mps, dtype=float).reshape(3))

@dataclass(frozen=True)
class TwoImpulseFreeTimeSpec:
    """Problem specification for two-impulse rendezvous with free final time."""
    x0: BoundaryState
    xf: BoundaryState

    tf_bounds_s: tuple[float, float] = (600.0, 7200.0)
    tf_guess_s: float | None = None

    mu_m3ps2: float = 3.986004418e14
    nsegs: int = 60

    # Auto-scaling overrides
    r_unit_m: float | None = None
    v_unit_mps: float | None = None
    t_unit_s: float | None = None

    w_time: float = 0.0

    # Objective toggles
    minimize_dv: bool = True
    dv_weight: float = 1.0
    minimize_time: bool = False

    # Boundary impulse structure
    dv_front: bool = True
    dv_back: bool = True
    dv_front_max_mps: float | None = None
    dv_back_max_mps: float | None = None

    lambert_grid_size: int = 50
    nrevs_to_try: tuple[int, ...] = (0, 1)

    @property
    def initial_state(self) -> BoundaryState:
        """Alias for `x0` with a clearer name."""
        return self.x0

    @property
    def final_state(self) -> BoundaryState:
        """Alias for `xf` with a clearer name."""
        return self.xf

    @property
    def final_time_bounds_s(self) -> tuple[float, float]:
        """Alias for `tf_bounds_s` with a clearer name."""
        return self.tf_bounds_s

@dataclass(frozen=True)
class TwoImpulsePreCoastSpec:
    """Problem specification for two-impulse rendezvous with variable pre-coast."""
    x0: BoundaryState
    xf: BoundaryState

    t1_bounds_s: tuple[float, float] = (0.0, 1800.0)
    tf_bounds_s: tuple[float, float] = (600.0, 7200.0)

    mu_m3ps2: float = 3.986004418e14

    nsegs_precoast: int = 30
    nsegs_transfer: int = 60

    # Auto-scaling overrides
    r_unit_m: float | None = None
    v_unit_mps: float | None = None
    t_unit_s: float | None = None

    w_time: float = 0.0

    # Objective toggles
    minimize_dv: bool = True
    dv_weight: float = 1.0
    minimize_time: bool = False

    # Boundary impulse structure
    dv_front: bool = True
    dv_back: bool = True
    dv_front_max_mps: float | None = None
    dv_back_max_mps: float | None = None

    # Link semantics between precoast and transfer:
    # "continuous" => (R, V, t) continuous
    # "impulsive"  => (R, t) continuous, allow V jump (Δv at link)
    link_kind: str = "impulsive"
    dv_link: bool = True

    precoast_grid_size: int = 10
    limit_precoast_to_one_period: bool = True

    lambert_grid_size: int = 50
    nrevs_to_try: tuple[int, ...] = (0, 1)

    min_dt_precoast_s: float = 0.0001
    min_dt_transfer_s: float = 0.0001

    @property
    def initial_state(self) -> BoundaryState:
        """Alias for `x0` with a clearer name."""
        return self.x0

    @property
    def final_state(self) -> BoundaryState:
        """Alias for `xf` with a clearer name."""
        return self.xf

    @property
    def precoast_time_bounds_s(self) -> tuple[float, float]:
        """Alias for `t1_bounds_s` with a clearer name."""
        return self.t1_bounds_s

    @property
    def final_time_bounds_s(self) -> tuple[float, float]:
        """Alias for `tf_bounds_s` with a clearer name."""
        return self.tf_bounds_s
