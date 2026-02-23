from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Maneuver:
    """Instantaneous maneuver marker.

    Attributes:
        r_m: ECI position at maneuver [m], shape (3,)
        t_s: maneuver time [s]
        dv_mps: applied delta-v [m/s], shape (3,)
        name: label used for reporting/plots
    """

    r_m: np.ndarray
    t_s: float
    dv_mps: np.ndarray
    name: str = "Maneuver"

    def __post_init__(self) -> None:
        object.__setattr__(self, "r_m", np.asarray(self.r_m, dtype=float).reshape(3))
        object.__setattr__(self, "dv_mps", np.asarray(self.dv_mps, dtype=float).reshape(3))
        object.__setattr__(self, "t_s", float(self.t_s))
