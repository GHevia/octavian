from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("asset_asrl", exc_type=ImportError)

from octavian import (
    Dynamics,
    Mission,
    Phase,
    Spacecraft,
    Thruster,
    constraints,
    guesses,
    objectives,
    state,
)
from octavian.solvers import SolverOptions

MU = 3.986004418e14


def _low_thrust_orbit_raise_mission() -> Mission:
    """Build the solver fixture without importing the executable example."""
    initial_radius_m = 7_000_000.0
    target_radius_m = 8_000_000.0
    spacecraft = Spacecraft(
        name="Electric orbit-raising vehicle",
        dry_mass_kg=500.0,
        thrusters=[
            Thruster(
                name="main",
                thrust_N=5.0,
                isp_s=1_800.0,
                propellant_mass_kg=60.0,
            )
        ],
    )
    initial_state = state(
        [initial_radius_m, 0.0, 0.0],
        [0.0, float(np.sqrt(MU / initial_radius_m)), 0.0],
    )
    terminal_seed_anchor = state(
        [target_radius_m, 0.0, 0.0],
        [0.0, float(np.sqrt(MU / target_radius_m)), 0.0],
    )
    orbit_raise = Phase(
        name="electric_orbit_raise",
        mode="low_thrust",
        spacecraft=spacecraft,
        dynamics=Dynamics(mu_m3ps2=MU),
        initial_state=initial_state,
        final_state=terminal_seed_anchor,
        tof_bounds_s=(14.0 * 3_600.0, 24.0 * 3_600.0),
        initial_guess=guesses.low_thrust_spiral(throttle=0.85, steps_per_orbit=120),
        constraints=[
            constraints.state(initial_state, where="Front"),
            constraints.min_radius(6_900_000.0),
            constraints.semi_major_axis(target_radius_m, where="Back", tol_m=10_000.0),
            constraints.eccentricity(0.01, where="Back", tol=0.0099),
        ],
    )
    return Mission(
        name="Regression fixture: low-thrust circular orbit raise",
        phases=[orbit_raise],
        objectives=[objectives.minimize_propellant()],
        mesh_nsegs_transfer=100,
        solver_options=SolverOptions(
            print_level=0,
            max_ls_iters=5,
            enable_adaptive_mesh=False,
            asset_threads=(1, 1),
        ),
    )


def test_low_thrust_orbit_raise_converges_from_spiral_seed() -> None:
    mission = _low_thrust_orbit_raise_mission()
    solution = mission.solve()

    assert solution.ok
    assert solution.result is not None
    result = solution.result
    assert result.info["state_layouts"] == ["cartesian_mass_thrust"]
    seed = result.info["phase_guess_info"][0]
    assert seed["guess_kind"] == "low_thrust_tangential_spiral"
    assert seed["seed_direction"] == "prograde"
    assert seed["seed_final_radius_m"] == pytest.approx(8_000_000.0, abs=25_000.0)

    powered = result.info["powered_phases"]
    assert len(powered) == 1
    assert powered[0]["kind"] == "low_thrust"
    assert 10.0 < powered[0]["propellant_used_kg"] < 25.0
    assert result.last_obj == pytest.approx(
        -powered[0]["mass_final_kg"] / mission.phases[0].spacecraft.initial_mass_kg,
        rel=1e-8,
    )
    assert all(row["satisfied"] for row in result.info["constraint_report"])
