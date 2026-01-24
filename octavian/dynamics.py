from __future__ import annotations

import numpy as np

import asset_asrl as ast  # type: ignore

vf = ast.VectorFunctions
oc = ast.OptimalControl


class TwoBodyECI(oc.ODEBase):
    """Two-body point-mass gravity in ECI, using ASSET's UpdatedInterface conventions.

    ODE state has 6 components: [r(3), v(3)].
    Time is the *phase time variable* (XtU.TVar()), not a state component.

    Vgroups are provided so callers can refer to variables by semantic names:
      - "R" (position, 3)
      - "V" (velocity, 3)
      - "t"/"time" (scalar time variable)
      - "RV" (indices [0..5])
    """

    def __init__(self, *, mu_m3ps2: float) -> None:
        self.mu = float(mu_m3ps2)

        XtU = oc.ODEArguments(6, 0)  # 6 states, 0 controls
        R = XtU.XVec().head(3)
        V = XtU.XVec().segment(3, 3)

        rnorm = R.norm()
        adot = (-self.mu) * R.normalized_power3()

        ode = vf.stack([V, adot])

        Vgroups = {
            ("R", "Position"): R,
            ("V", "Velocity"): V,
            ("t", "time"): XtU.TVar(),
            "RV": [0, 1, 2, 3, 4, 5],
        }

        super().__init__(ode, 6, 0, Vgroups=Vgroups)
