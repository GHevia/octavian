from __future__ import annotations

"""Dynamics models.

Octavian's primary optimal-control backend is ASSET. To keep the rest of the
package importable (for utilities, studies, and result I/O), ASSET is treated as
an optional runtime dependency at import time.

If ASSET is not installed, constructing ASSET-backed dynamics will raise a
clear error, but importing this module will still succeed.
"""

import numpy as np

import asset_asrl as ast  # type: ignore

vf = ast.VectorFunctions
oc = ast.OptimalControl


class TwoBodyECI(oc.ODEBase):
    """Two-body point-mass gravity in ECI.

    ODE state has 6 components: ``[r(3), v(3)]``.
    Time is the *phase time variable* (``ODEArguments.TVar()``), not a state component.

    Vgroups:
        - ``R``: position (3)
        - ``V``: velocity (3)
        - ``t``: phase time variable
    """

    def __init__(self, *, mu_m3ps2: float) -> None:
        self.mu = float(mu_m3ps2)

        XtU = oc.ODEArguments(6, 0)  # 6 states, 0 controls
        R = XtU.XVec().head(3)
        V = XtU.XVec().segment(3, 3)

        A = (-self.mu) * R.normalized_power3()

        ode = vf.stack([V, A])

        Vgroups = {
            ("R", "Position"): R,
            ("V", "Velocity"): V,
            ("t", "time"): XtU.TVar(),
            "RV": [0, 1, 2, 3, 4, 5],
        }

        super().__init__(ode, 6, 0, Vgroups=Vgroups)
