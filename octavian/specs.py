from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import numpy as np

@dataclass(frozen=True)
class BoundaryState:
    """Dimensional Cartesian boundary state in ECI."""
    r_m: np.ndarray
    v_mps: np.ndarray
    def __post_init__(self) -> None:
        object.__setattr__(self, "r_m", np.asarray(self.r_m, dtype=float).reshape(3))
        object.__setattr__(self, "v_mps", np.asarray(self.v_mps, dtype=float).reshape(3))

@dataclass(frozen=True)
class TwoImpulseFreeTimeSpec:
    x0: BoundaryState
    xf: BoundaryState

    tf_bounds_s: Tuple[float, float] = (600.0, 7200.0)
    tf_guess_s: Optional[float] = None

    mu_m3ps2: float = 3.986004418e14
    nsegs: int = 60

    # Auto-scaling overrides
    r_unit_m: Optional[float] = None
    v_unit_mps: Optional[float] = None
    t_unit_s: Optional[float] = None

    w_time: float = 0.0

    lambert_grid_size: int = 50
    nrevs_to_try: Sequence[int] = (0, 1)

@dataclass(frozen=True)
class TwoImpulsePreCoastSpec:
    x0: BoundaryState
    xf: BoundaryState

    t1_bounds_s: Tuple[float, float] = (0.0, 1800.0)
    tf_bounds_s: Tuple[float, float] = (600.0, 7200.0)

    mu_m3ps2: float = 3.986004418e14

    nsegs_precoast: int = 30
    nsegs_transfer: int = 60

    # Auto-scaling overrides
    r_unit_m: Optional[float] = None
    v_unit_mps: Optional[float] = None
    t_unit_s: Optional[float] = None

    w_time: float = 0.0

    precoast_grid_size: int = 10
    limit_precoast_to_one_period: bool = True

    lambert_grid_size: int = 50
    nrevs_to_try: Sequence[int] = (0, 1)

    min_dt_precoast_s: float = 0.0001
    min_dt_transfer_s: float = 0.0001
