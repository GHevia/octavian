from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from octavian import Dynamics, Phase, Solution, Spacecraft, StateLayout, Thruster
from octavian.astro import default_scaling, default_units
from octavian.coordinates import (
    CARTESIAN,
    CARTESIAN_MASS,
    CARTESIAN_MASS_THRUST,
    COUPLED_RELATIVE_ECI_MASS,
    COUPLED_RELATIVE_ECI_MASS_THRUST,
    EARTH_INERTIAL,
    SolverScaling,
    inertial,
)
from octavian.solvers.compiler.phase_compiler import PhaseBuild, layout_for_phase, trajectory_rvt
from octavian.solvers.preconfigured import RendezvousResult
from octavian.specs import BoundaryState


def test_standard_state_layouts_expose_named_groups_and_time_columns() -> None:
    assert CARTESIAN.state_indices("position") == (0, 1, 2)
    assert CARTESIAN.state_indices("velocity") == (3, 4, 5)
    assert CARTESIAN.time_column == 6

    assert CARTESIAN_MASS.state_indices("mass") == (6,)
    assert CARTESIAN_MASS.time_column == 7

    assert CARTESIAN_MASS_THRUST.control_indices("thrust") == (0, 1, 2)
    assert CARTESIAN_MASS_THRUST.state_dim == 7
    assert CARTESIAN_MASS_THRUST.control_dim == 3
    assert COUPLED_RELATIVE_ECI_MASS.state_indices("mass") == (12,)
    assert COUPLED_RELATIVE_ECI_MASS.time_column == 13
    assert COUPLED_RELATIVE_ECI_MASS_THRUST.control_indices("thrust") == (0, 1, 2)
    assert COUPLED_RELATIVE_ECI_MASS_THRUST.control_dim == 3


def test_state_layout_rejects_duplicate_names_and_invalid_groups() -> None:
    with pytest.raises(ValueError, match="unique"):
        StateLayout(name="bad", state_names=("x", "x"))
    with pytest.raises(ValueError, match="out-of-range"):
        StateLayout(
            name="bad",
            state_names=("x",),
            state_groups=(("position", (1,)),),
        )


def test_phase_compiler_selects_layout_from_phase_intent() -> None:
    spacecraft = Spacecraft(
        name="vehicle",
        dry_mass_kg=100.0,
        thrusters=[Thruster(name="main", thrust_N=10.0, isp_s=300.0)],
    )
    coast = Phase(mode="coast", spacecraft=spacecraft, dynamics=Dynamics())
    burn = Phase(mode="chemical_burn", spacecraft=spacecraft, dynamics=Dynamics())

    assert layout_for_phase(coast) is CARTESIAN
    assert layout_for_phase(coast, carries_mass=True) is CARTESIAN_MASS
    assert layout_for_phase(burn) is CARTESIAN_MASS_THRUST


def test_phase_build_dimensions_are_derived_from_layout() -> None:
    phase = Phase()
    build = PhaseBuild(
        ph=phase,
        asset_phase=object(),
        t_bounds=(0.0, 1.0),
        index=0,
        layout=CARTESIAN_MASS_THRUST,
    )

    assert build.state_dim == 7
    assert build.control_dim == 3


def test_trajectory_projection_uses_named_layout_groups() -> None:
    raw = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 500.0, 10.0, 0.1, 0.2, 0.3],
            [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 499.0, 11.0, 0.1, 0.2, 0.3],
        ]
    )

    projected = trajectory_rvt(raw, CARTESIAN_MASS_THRUST)
    np.testing.assert_allclose(projected[:, -1], [10.0, 11.0])
    np.testing.assert_allclose(projected[:, 0:6], raw[:, 0:6])


def test_explicit_scaling_overrides_endpoint_defaults() -> None:
    explicit = SolverScaling(length_m=1.0e6, velocity_mps=1.0e3, time_s=500.0, mass_kg=250.0)
    spec = SimpleNamespace(
        x0=BoundaryState(np.array([7.0e6, 0.0, 0.0]), np.array([0.0, 7.5e3, 0.0])),
        xf=BoundaryState(np.array([8.0e6, 0.0, 0.0]), np.array([0.0, 7.0e3, 0.0])),
        tf_bounds_s=(100.0, 1_000.0),
        scaling=explicit,
    )

    assert default_scaling(spec) is explicit
    assert default_units(spec) == (explicit.length_m, explicit.velocity_mps, explicit.time_s)
    assert explicit.force_N == pytest.approx(500.0)


def test_dynamics_and_solution_report_coordinate_metadata() -> None:
    sun_frame = inertial("sun")
    dynamics = Dynamics(frame=sun_frame)
    assert dynamics.frame == sun_frame
    assert Dynamics().frame == EARTH_INERTIAL

    scaling = SolverScaling(7.0e6, 7.5e3, 1_000.0, 500.0)
    result = RendezvousResult(
        converged=True,
        traj=np.zeros((2, 7)),
        maneuvers=[],
        info={"frame": sun_frame.to_dict(), "scaling": scaling.to_dict()},
    )
    solution = Solution(ok=True, result=result)

    assert solution.frame == sun_frame
    assert solution.scaling == scaling
