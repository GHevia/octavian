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


def test_qlaw_plane_change_converges_and_reports_physical_thrust() -> None:
    mission = _low_thrust_orbit_raise_mission()
    phase = mission.phases[0]
    angle = np.deg2rad(2.0)
    speed = np.sqrt(MU / 7_200_000.0)
    phase.final_state = state(
        [7_200_000.0, 0, 0], [0, speed * np.cos(angle), speed * np.sin(angle)]
    )
    phase.spacecraft = Spacecraft(
        dry_mass_kg=20, thrusters=[Thruster(thrust_N=0.1, isp_s=1800, propellant_mass_kg=3)]
    )
    phase.initial_guess = guesses.low_thrust_qlaw(throttle=0.85, plane_tol_deg=0.05)
    phase.tof_bounds_s = (30 * 3600.0, 30.2 * 3600.0)
    phase.constraints = [
        constraints.state(phase.initial_state, where="Front"),
        constraints.min_radius(6_900_000),
        constraints.semi_major_axis(7_200_000.0, where="Back", tol_m=10_000),
        constraints.eccentricity(0.005, where="Back", tol=0.0049),
        constraints.inclination_deg(2.0, where="Back", tol_deg=0.05),
    ]
    mission.mesh_nsegs_transfer = 160
    mission.solver_options.enable_adaptive_mesh = False
    solution = mission.solve()
    assert solution.ok and solution.result is not None
    assert solution.result.info["phase_guess_info"][0]["guess_kind"] == "low_thrust_qlaw"
    controls = solution.phase_control_trajectories[0]
    thrust = solution.phase_thrust_trajectories[0]
    np.testing.assert_allclose(thrust[:, 0], controls[:, 0])
    np.testing.assert_allclose(thrust[:, 1], np.linalg.norm(controls[:, 1:4], axis=1))
    np.testing.assert_allclose(thrust[:, 2], 0.1 * thrust[:, 1])
    report = next(
        row
        for row in solution.result.info["constraint_report"]
        if row["constraint"] == "inclination_deg"
    )
    assert report["satisfied"] and abs(report["actual"] - 2) < 0.05
