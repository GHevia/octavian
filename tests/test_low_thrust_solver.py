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


def test_qlaw_plane_change_converges_and_reports_physical_thrust(monkeypatch) -> None:
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
    phase.tof_bounds_s = (32 * 3600.0, 32.2 * 3600.0)
    phase.constraints = [
        constraints.state(phase.initial_state, where="Front"),
        constraints.min_radius(6_900_000),
        constraints.semi_major_axis(7_200_000.0, where="Back"),
        constraints.eccentricity(0.005, where="Back", tol=0.0049),
        constraints.inclination_deg(2.0, where="Back"),
    ]
    mission.mesh_nsegs_transfer = 120
    mission.solver_options.transcription_mode = "LGL5"
    mission.solver_options.enable_adaptive_mesh = False
    solution = mission.solve()
    assert solution.ok and solution.result is not None
    assert solution.result.info["phase_guess_info"][0]["guess_kind"] == "low_thrust_qlaw"
    assert solution.result.info["phase_guess_info"][0]["seed_target_reached"]
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
    assert solution.result.info["transcription_mode"] == "LGL5"
    # Independently re-integrate the quadratic LGL5 controls. Collocation
    # endpoint values alone do not establish the physical transfer accuracy.
    current = np.r_[solution.traj[0, :6], phase.spacecraft.initial_mass_kg]
    history = solution.phase_control_trajectories[0]
    thruster = phase.spacecraft.thrusters[0]

    def derivative(time, state_vector):
        index = min(
            max(((np.searchsorted(history[:, 0], time, side="right") - 1) // 2) * 2, 0),
            len(history) - 3,
        )
        fraction = (time - history[index, 0]) / (history[index + 2, 0] - history[index, 0])
        vector = (2 * fraction**2 - 3 * fraction + 1) * history[index, 1:4]
        vector += 4 * fraction * (1 - fraction) * history[index + 1, 1:4]
        vector += (2 * fraction**2 - fraction) * history[index + 2, 1:4]
        position = state_vector[:3]
        return np.r_[
            state_vector[3:6],
            -MU * position / np.linalg.norm(position) ** 3
            + thruster.thrust_N * vector / state_vector[6],
            -thruster.thrust_N * np.linalg.norm(vector) / (thruster.isp_s * 9.80665),
        ]

    final_time = history[-1, 0]
    count = int(np.ceil(final_time / 15))
    step = final_time / count
    time = 0.0
    for _ in range(count):
        k1 = derivative(time, current)
        k2 = derivative(time + step / 2, current + step * k1 / 2)
        k3 = derivative(time + step / 2, current + step * k2 / 2)
        k4 = derivative(time + step, current + step * k3)
        current += step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        time += step
    position, velocity = current[:3], current[3:6]
    momentum = np.cross(position, velocity)
    axis = 1 / (2 / np.linalg.norm(position) - velocity @ velocity / MU)
    inclination = np.rad2deg(np.arccos(momentum[2] / np.linalg.norm(momentum)))
    assert abs(axis - 7_200_000) < 100
    assert abs(inclination - 2) < 0.001

    # Continue the same flight under J2/Sun/Moon gravity. Preserve the solved
    # mass and controls, avoid unrelated Lambert seeding, and honor the new
    # mesh rather than interpreting every dense sample as a segment boundary.
    from octavian import Perturbations
    from octavian.solvers import composable

    guess = solution.trajectory_guess()
    np.testing.assert_allclose(guess.rows[:, 8:11], controls[:, 1:4])
    np.testing.assert_allclose(guess.rows[:, 7], solution.traj[:, 6])
    phase.initial_guess = guess
    phase.dynamics = Dynamics(
        mu_m3ps2=MU,
        perturbations=Perturbations(j2=True, sun=True, moon=True),
        third_body_table_step_s=600,
    )
    mission.initial_epoch = "2026-01-01T00:00:00Z"
    mission.mesh_nsegs_transfer = 160

    def unrelated_seed(*args, **kwargs):
        raise AssertionError("A full solved history should bypass automatic seeding")

    monkeypatch.setattr(composable, "_build_guess_single_phase_terminal_position", unrelated_seed)
    perturbed = mission.solve()
    assert perturbed.ok and perturbed.result is not None
    assert perturbed.result.info["phase_guess_info"][0]["guess_kind"] == "explicit_trajectory"
    assert perturbed.result.info["phase_guess_info"][0]["seed_duration"] == pytest.approx(
        controls[-1, 0] - controls[0, 0]
    )
    assert len(perturbed.traj) == 2 * 160 + 1
    assert all(row["satisfied"] for row in perturbed.result.info["constraint_report"])
