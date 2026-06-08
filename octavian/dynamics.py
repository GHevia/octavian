"""Dynamics models.

Octavian's primary optimal-control backend is ASSET. To keep the rest of the
package importable (for utilities, studies, and result I/O), ASSET is treated as
an optional runtime dependency at import time.

If ASSET is not installed, constructing ASSET-backed dynamics will raise a
clear error, but importing this module will still succeed.
"""

from __future__ import annotations

from ._asset import oc, require_asset, vf


def _require_asset() -> None:
    """Require ASSET before constructing a backend ODE."""
    require_asset("ASSET-backed dynamics")


def _point_mass_acceleration(position_vec, mu_m3ps2: float):
    return (-float(mu_m3ps2)) * position_vec.normalized_power3()


def _j2_acceleration(position_vec, *, mu_m3ps2: float, radius_m: float, j2: float):
    radius_sq = position_vec.dot(position_vec)
    radius = position_vec.norm()
    z = position_vec[2]
    z_sq_over_r_sq = (z * z) / radius_sq
    scale = 1.5 * float(j2) * float(mu_m3ps2) * (float(radius_m) ** 2) / (radius**5)
    common_xy = 5.0 * z_sq_over_r_sq - 1.0
    z_term = 5.0 * z_sq_over_r_sq - 3.0
    return vf.stack(
        [
            scale * position_vec[0] * common_xy,
            scale * position_vec[1] * common_xy,
            scale * z * z_term,
        ]
    )


def _gravity_acceleration(
    position_vec,
    *,
    mu_m3ps2: float,
    include_j2: bool = False,
    central_body_radius_m: float = 6_378_136.3,
    j2_coefficient: float = 1.08262668e-3,
):
    acceleration = _point_mass_acceleration(position_vec, mu_m3ps2)
    if include_j2:
        acceleration = acceleration + _j2_acceleration(
            position_vec,
            mu_m3ps2=mu_m3ps2,
            radius_m=central_body_radius_m,
            j2=j2_coefficient,
        )
    return acceleration


class TwoBodyECI(oc.ODEBase if oc is not None else object):
    """Two-body point-mass gravity in ECI.

    ODE state has 6 components: ``[r(3), v(3)]``.
    Time is the *phase time variable* (``ODEArguments.TVar()``), not a state component.

    Vgroups:
        - ``R``: position (3)
        - ``V``: velocity (3)
        - ``t``: phase time variable
    """

    def __init__(self, *, mu_m3ps2: float) -> None:
        _require_asset()
        self.mu = float(mu_m3ps2)

        XtU = oc.ODEArguments(6, 0)  # 6 states, 0 controls
        R = XtU.XVec().head(3)
        V = XtU.XVec().segment(3, 3)

        A = _gravity_acceleration(R, mu_m3ps2=self.mu)

        ode = vf.stack([V, A])

        Vgroups = {
            ("R", "Position"): R,
            ("V", "Velocity"): V,
            ("t", "time"): XtU.TVar(),
            "RV": [0, 1, 2, 3, 4, 5],
        }

        super().__init__(ode, 6, 0, Vgroups=Vgroups)


class PerturbedECI(oc.ODEBase if oc is not None else object):
    """Point-mass gravity with optional perturbation accelerations.

    Currently implemented perturbations:
        - J2 zonal harmonic
    """

    def __init__(
        self,
        *,
        mu_m3ps2: float,
        j2: bool = False,
        central_body_radius_m: float = 6_378_136.3,
        j2_coefficient: float = 1.08262668e-3,
    ) -> None:
        _require_asset()
        self.mu = float(mu_m3ps2)

        XtU = oc.ODEArguments(6, 0)
        R = XtU.XVec().head(3)
        V = XtU.XVec().segment(3, 3)

        A = _gravity_acceleration(
            R,
            mu_m3ps2=self.mu,
            include_j2=bool(j2),
            central_body_radius_m=float(central_body_radius_m),
            j2_coefficient=float(j2_coefficient),
        )
        ode = vf.stack([V, A])

        Vgroups = {
            ("R", "Position"): R,
            ("V", "Velocity"): V,
            ("t", "time"): XtU.TVar(),
            "RV": [0, 1, 2, 3, 4, 5],
        }

        super().__init__(ode, 6, 0, Vgroups=Vgroups)


class MassCoastECI(oc.ODEBase if oc is not None else object):
    """Ballistic coast dynamics carrying spacecraft mass as a constant state."""

    def __init__(
        self,
        *,
        mu_m3ps2: float,
        j2: bool = False,
        central_body_radius_m: float = 6_378_136.3,
        j2_coefficient: float = 1.08262668e-3,
    ) -> None:
        _require_asset()
        self.mu = float(mu_m3ps2)

        XtU = oc.ODEArguments(7, 0)
        X = XtU.XVec()
        R = X.head(3)
        V = X.segment(3, 3)
        M = X[6]

        A = _gravity_acceleration(
            R,
            mu_m3ps2=self.mu,
            include_j2=bool(j2),
            central_body_radius_m=float(central_body_radius_m),
            j2_coefficient=float(j2_coefficient),
        )
        ode = vf.stack([V, A, M * 0.0])

        Vgroups = {
            ("R", "Position"): R,
            ("V", "Velocity"): V,
            ("M", "Mass"): [6],
            ("t", "time"): XtU.TVar(),
            "RV": [0, 1, 2, 3, 4, 5],
        }

        super().__init__(ode, 7, 0, Vgroups=Vgroups)


class ChemicalBurnECI(oc.ODEBase if oc is not None else object):
    """Finite chemical burn dynamics with three thrust-direction controls.

    ODE state is ``[r(3), v(3), m]`` and controls are a dimensionless thrust
    direction/throttle vector. The compiler bounds the control norm to one.
    """

    def __init__(
        self,
        *,
        mu_m3ps2: float,
        thrust_N: float,
        isp_s: float,
        j2: bool = False,
        central_body_radius_m: float = 6_378_136.3,
        j2_coefficient: float = 1.08262668e-3,
        g0_mps2: float = 9.80665,
    ) -> None:
        _require_asset()
        self.mu = float(mu_m3ps2)
        self.thrust_N = float(thrust_N)
        self.isp_s = float(isp_s)
        self.g0_mps2 = float(g0_mps2)

        if self.thrust_N <= 0.0:
            raise ValueError("ChemicalBurnECI requires thrust_N > 0.")
        if self.isp_s <= 0.0:
            raise ValueError("ChemicalBurnECI requires isp_s > 0.")

        XtU = oc.ODEArguments(7, 3)
        X = XtU.XVec()
        U = XtU.UVec()
        R = X.head(3)
        V = X.segment(3, 3)
        M = X[6]

        gravity = _gravity_acceleration(
            R,
            mu_m3ps2=self.mu,
            include_j2=bool(j2),
            central_body_radius_m=float(central_body_radius_m),
            j2_coefficient=float(j2_coefficient),
        )
        thrust_acceleration = (self.thrust_N / M) * U
        mass_flow = -(self.thrust_N / (self.isp_s * self.g0_mps2)) * U.norm()
        ode = vf.stack([V, gravity + thrust_acceleration, mass_flow])

        Vgroups = {
            ("R", "Position"): R,
            ("V", "Velocity"): V,
            ("M", "Mass"): [6],
            ("U", "Control", "Throttle"): U,
            ("t", "time"): XtU.TVar(),
            "RV": [0, 1, 2, 3, 4, 5],
        }

        super().__init__(ode, 7, 3, Vgroups=Vgroups)
