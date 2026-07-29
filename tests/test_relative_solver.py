from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("asset_asrl", exc_type=ImportError)

from octavian import (
    EARTH,
    Dynamics,
    Mission,
    Perturbations,
    Phase,
    Spacecraft,
    Thruster,
    constraints,
    objectives,
    state,
    variables,
)
from octavian.astro import classical_to_cartesian
from octavian.relative import (
    RelativeOrbitalElements,
    cwh_rendezvous_velocity,
    propagate_cwh,
    propagate_relative_numerical,
    propagate_relative_orbital_elements,
    propagate_two_body_state,
    relative_orbital_elements_to_relative_state,
)
from octavian.solvers import SolverOptions
from octavian.solvers.compiler import phase_compiler


def _relative_rendezvous_mission() -> Mission:
    initial_state = state([0.0, -1_000.0, 0.0], [0.0, 0.0, 0.0])
    final_state = state([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    phase = Phase(
        name="relative_rendezvous",
        mode="relative_coast",
        spacecraft=Spacecraft(name="Deputy", dry_mass_kg=250.0),
        dynamics=Dynamics.cwh(
            chief_orbit_radius_m=EARTH.mean_radius_m + 400_000.0,
            chief_name="Chief",
            reference_length_m=1_000.0,
        ),
        initial_state=initial_state,
        final_state=final_state,
        tof_bounds_s=(1_200.0, 2_400.0),
        constraints=[
            constraints.state(initial_state, where="Front"),
            constraints.state(final_state, where="Back"),
        ],
        variables=[
            variables.impulsive_delta_v(at="Front"),
            variables.impulsive_delta_v(at="Back"),
        ],
    )
    return Mission(
        phases=[phase],
        objectives=[objectives.minimize_total_delta_v()],
        mesh_nsegs_transfer=40,
        lambert_grid_size=60,
        solver_options=SolverOptions(
            print_level=0,
            max_ls_iters=5,
            asset_threads=(1, 1),
        ),
    )


def test_cwh_relative_rendezvous_compiles_solves_and_reports_frame() -> None:
    mission = _relative_rendezvous_mission()
    solution = mission.solve()
    assert solution.ok
    assert solution.result is not None

    result = solution.result
    model = mission.phases[0].dynamics.model
    assert model is not None
    assert result.info["dynamics_model"] == "cwh"
    assert result.info["state_layouts"] == ["relative_cartesian"]
    assert solution.frame is not None
    assert solution.frame.kind == "relative"
    assert solution.frame.origin == "chief"
    assert result.traj[0, 0:3] == pytest.approx([0.0, -1_000.0, 0.0], abs=1e-6)
    assert result.traj[-1, 0:3] == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)

    tof_s = result.tf_s()
    departure_velocity = cwh_rendezvous_velocity(
        result.traj[0, 0:3],
        result.traj[-1, 0:3],
        tof_s,
        model.mean_motion_radps,
    )
    analytic_arrival = propagate_cwh(
        np.hstack([result.traj[0, 0:3], departure_velocity]),
        tof_s,
        model.mean_motion_radps,
    )
    expected_dv_mps = float(
        np.linalg.norm(departure_velocity) + np.linalg.norm(analytic_arrival[3:6])
    )
    assert result.total_dv_mps() == pytest.approx(expected_dv_mps, rel=2e-5)
    assert result.traj[-1, 3:6] == pytest.approx(analytic_arrival[3:6], abs=1e-7)


def test_cwh_relative_geometry_constraints_compile_and_report() -> None:
    initial_state = state([0.0, -1_000.0, 0.0], [0.0, 0.0, 0.0])
    final_state = state([0.0, -100.0, 0.0], [0.0, 0.0, 0.0])
    phase = Phase(
        name="safe_approach",
        mode="relative_coast",
        spacecraft=Spacecraft(name="Deputy", dry_mass_kg=250.0),
        dynamics=Dynamics.cwh(
            chief_orbit_radius_m=EARTH.mean_radius_m + 400_000.0,
            reference_length_m=1_000.0,
        ),
        initial_state=initial_state,
        final_state=final_state,
        tof_bounds_s=(1_200.0, 2_400.0),
        constraints=[
            constraints.state(initial_state, where="Front"),
            constraints.state(final_state, where="Back"),
            constraints.keep_out_sphere(75.0),
            constraints.approach_cone([0.0, -1.0, 0.0], 30.0),
            constraints.lighting_angle(
                [1.0, 0.0, 0.0],
                min_angle_deg=85.0,
                max_angle_deg=121.0,
            ),
        ],
        variables=[
            variables.impulsive_delta_v(at="Front"),
            variables.impulsive_delta_v(at="Back"),
        ],
    )
    solution = Mission(
        phases=[phase],
        objectives=[objectives.minimize_total_delta_v()],
        mesh_nsegs_transfer=50,
        lambert_grid_size=60,
        solver_options=SolverOptions(
            print_level=0,
            max_ls_iters=5,
            asset_threads=(1, 1),
        ),
    ).solve()

    assert solution.ok
    assert solution.result is not None
    report = solution.result.info["constraint_report"]
    assert [row["constraint"] for row in report] == [
        "keep_out_sphere",
        "approach_cone",
        "lighting_min_angle_deg",
        "lighting_max_angle_deg",
    ]
    assert all(row["satisfied"] for row in report)
    assert report[0]["actual"] >= 75.0
    assert report[1]["actual"] <= 30.0 + 1e-4
    assert report[2]["actual"] >= 85.0 - 1e-4
    assert report[3]["actual"] <= 121.0 + 1e-4


def test_nonlinear_relative_j2_compiles_solves_and_reports_absolute_states() -> None:
    chief_radius_m = EARTH.mean_radius_m + 400_000.0
    chief_speed_mps = np.sqrt(EARTH.mu_m3ps2 / chief_radius_m)
    chief_state = state(
        [chief_radius_m, 0.0, 0.0],
        [0.0, chief_speed_mps, 0.0],
    )
    mission = _relative_rendezvous_mission()
    mission.mesh_nsegs_transfer = 20
    mission.lambert_grid_size = 20
    mission.phases[0].dynamics = Dynamics.relative(
        chief_initial_state_eci=chief_state,
        perturbations=Perturbations(j2=True),
        third_body_table_step_s=300.0,
    )

    solution = mission.solve()

    assert solution.ok
    assert solution.result is not None
    assert solution.result.info["dynamics_model"] == "nonlinear_relative"
    assert solution.result.info["state_layouts"] == ["coupled_relative_eci"]
    assert solution.result.info["relative_reference_model"] is None
    assert np.asarray(solution.result.info["chief_trajectory_eci"]).shape[1] == 7
    assert np.asarray(solution.result.info["deputy_trajectory_eci"]).shape[1] == 7
    assert "nonlinear_relative" in solution.result.to_json()


@pytest.mark.parametrize(
    ("propagation_mode", "expected_layout"),
    [
        ("coupled_ric", "coupled_relative_ric"),
        ("nonlinear_ric", "relative_cartesian"),
    ],
)
def test_native_exact_ric_formulations_compile_solve_and_report(
    propagation_mode: str,
    expected_layout: str,
) -> None:
    chief_radius_m = EARTH.mean_radius_m + 400_000.0
    chief_state = state(
        [chief_radius_m, 0.0, 0.0],
        [0.0, np.sqrt(EARTH.mu_m3ps2 / chief_radius_m), 0.0],
    )
    mission = _relative_rendezvous_mission()
    mission.mesh_nsegs_transfer = 20
    mission.lambert_grid_size = 20
    mission.phases[0].dynamics = Dynamics.relative(
        chief_initial_state_eci=chief_state,
        propagation_mode=propagation_mode,
    )
    mission.phases[0].constraints.append(
        constraints.ric_state("I", 0.0, where="Back")
    )

    solution = mission.solve()

    assert solution.ok
    assert solution.result is not None
    assert solution.relative_propagation_mode == propagation_mode
    assert solution.result.info["state_layouts"] == [expected_layout]
    assert solution.traj[-1, 0:3] == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
    assert solution.result.info["constraint_report"][-1]["constraint"] == "ric_I"
    assert solution.result.info["constraint_report"][-1]["satisfied"] is True
    assert solution.chief_trajectory_eci.shape[1] == 7
    assert solution.deputy_trajectory_eci.shape[1] == 7


def test_multiple_relative_coasts_link_native_state_and_stitch_results() -> None:
    radius_m = EARTH.mean_radius_m + 400_000.0
    mean_motion = np.sqrt(EARTH.mu_m3ps2 / radius_m**3)
    initial = state([100.0, -1_000.0, 20.0], [0.05, 0.1, -0.02])
    midpoint_vector = propagate_cwh(
        np.hstack([initial.r_m, initial.v_mps]),
        300.0,
        mean_motion,
    )
    final_vector = propagate_cwh(
        np.hstack([initial.r_m, initial.v_mps]),
        600.0,
        mean_motion,
    )
    midpoint = state(midpoint_vector[0:3], midpoint_vector[3:6])
    final = state(final_vector[0:3], final_vector[3:6])
    spacecraft = Spacecraft(name="Deputy", dry_mass_kg=250.0)
    dynamics = Dynamics.cwh(
        chief_orbit_radius_m=radius_m,
        chief_name="Chief",
    )
    first_coast = Phase(
        name="first_coast",
        mode="relative_coast",
        spacecraft=spacecraft,
        dynamics=dynamics,
        initial_state=initial,
        final_state=midpoint,
        tof_bounds_s=(299.0, 301.0),
        constraints=[
            constraints.state(initial, where="Front"),
            constraints.state(midpoint, where="Back"),
        ],
    )
    second_coast = Phase(
        name="second_coast",
        mode="relative_coast",
        spacecraft=spacecraft,
        dynamics=dynamics,
        previous=first_coast,
        final_state=final,
        tof_bounds_s=(299.0, 301.0),
        tof_is_relative=True,
        constraints=[constraints.state(final, where="Back")],
    )

    solution = Mission(
        phases=[first_coast, second_coast],
        objectives=[],
        mesh_nsegs_precoast=8,
        mesh_nsegs_transfer=8,
        lambert_grid_size=8,
        solver_options=SolverOptions(
            print_level=0,
            max_ls_iters=3,
            enable_adaptive_mesh=False,
            asset_threads=(1, 1),
        ),
    ).solve()

    assert solution.ok
    assert solution.result is not None
    assert solution.result.info["nphases"] == 2
    assert solution.result.info["state_layouts"] == [
        "relative_cartesian",
        "relative_cartesian",
    ]
    assert len(solution.native_relative_phase_trajectories) == 2
    assert solution.traj[-1, 0:6] == pytest.approx(final_vector, abs=1.0e-7)
    assert solution.traj.shape[0] == sum(
        trajectory.shape[0]
        for trajectory in solution.native_relative_phase_trajectories
    ) - 1


def test_relative_finite_burn_coast_chain_compiles_solves_and_carries_mass() -> None:
    radius_m = EARTH.mean_radius_m + 400_000.0
    chief = state(
        [radius_m, 0.0, 0.0],
        [0.0, np.sqrt(EARTH.mu_m3ps2 / radius_m), 0.0],
    )
    initial = state([0.0, -1_000.0, 0.0], [0.0, 0.0, 0.0])
    propagated = propagate_relative_numerical(
        chief,
        initial,
        [0.0, 420.0],
        max_step_s=5.0,
    )
    final_vector = propagated.relative_states_ric[-1]
    final_vector[0] += 5.0
    final = state(final_vector[0:3], final_vector[3:6])
    spacecraft = Spacecraft(
        name="Finite-burn deputy",
        dry_mass_kg=250.0,
        thrusters=[
            Thruster(
                name="main",
                thrust_N=10.0,
                isp_s=300.0,
                propellant_mass_kg=10.0,
            )
        ],
    )
    dynamics = Dynamics.relative(
        chief_initial_state_eci=chief,
        propagation_mode="coupled_eci",
    )
    departure_burn = Phase(
        name="departure_burn",
        mode="finite_thrust",
        spacecraft=spacecraft,
        dynamics=dynamics,
        initial_state=initial,
        tof_bounds_s=(59.0, 61.0),
        constraints=[constraints.state(initial, where="Front")],
    )
    coast = Phase(
        name="drift_coast",
        mode="relative_coast",
        spacecraft=spacecraft,
        dynamics=dynamics,
        previous=departure_burn,
        tof_bounds_s=(299.0, 301.0),
        tof_is_relative=True,
    )
    arrival_burn = Phase(
        name="arrival_burn",
        mode="finite_thrust",
        spacecraft=spacecraft,
        dynamics=dynamics,
        previous=coast,
        final_state=final,
        tof_bounds_s=(59.0, 61.0),
        tof_is_relative=True,
        constraints=[constraints.state(final, where="Back")],
    )
    solution = Mission(
        phases=[departure_burn, coast, arrival_burn],
        objectives=[objectives.minimize_propellant()],
        mesh_nsegs_precoast=8,
        mesh_nsegs_transfer=12,
        lambert_grid_size=12,
        solver_options=SolverOptions(
            print_level=0,
            max_ls_iters=5,
            enable_adaptive_mesh=False,
            asset_threads=(1, 1),
        ),
    ).solve()

    assert solution.ok
    assert solution.result is not None
    assert solution.result.info["state_layouts"] == [
        "coupled_relative_eci_mass_thrust",
        "coupled_relative_eci_mass",
        "coupled_relative_eci_mass_thrust",
    ]
    powered_phases = solution.result.info["powered_phases"]
    assert len(powered_phases) == 2
    assert sum(float(phase["propellant_used_kg"]) for phase in powered_phases) > 0.0
    assert solution.traj[-1, 0:6] == pytest.approx(
        np.hstack([final.r_m, final.v_mps]),
        abs=1.0e-5,
    )
    native_phases = solution.native_relative_phase_trajectories
    assert len(native_phases) == 3
    assert all(trajectory.shape[1] == 14 for trajectory in native_phases)
    mass_boundaries = [
        (native_phases[index][-1, 12], native_phases[index + 1][0, 12])
        for index in range(2)
    ]
    for mass_before, mass_after in mass_boundaries:
        assert mass_after == pytest.approx(mass_before, abs=1.0e-8)
    assert solution.chief_trajectory_eci.shape == solution.deputy_trajectory_eci.shape
    assert solution.chief_trajectory_eci.shape[0] == solution.traj.shape[0]


def test_cwh_rejects_perturbations_with_relative_model_guidance() -> None:
    with pytest.raises(ValueError, match=r"Dynamics\.relative"):
        Dynamics.cwh(
            chief_orbit_radius_m=EARTH.mean_radius_m + 400_000.0,
            perturbations=Perturbations(j2=True),
        )


def test_native_ric_mode_rejects_perturbations_with_coupled_eci_guidance() -> None:
    chief_radius_m = EARTH.mean_radius_m + 400_000.0
    chief_state = state(
        [chief_radius_m, 0.0, 0.0],
        [0.0, np.sqrt(EARTH.mu_m3ps2 / chief_radius_m), 0.0],
    )
    phase = _relative_rendezvous_mission().phases[0]
    phase.dynamics = Dynamics.relative(
        chief_initial_state_eci=chief_state,
        propagation_mode="coupled_ric",
        perturbations=Perturbations(j2=True),
    )

    with pytest.raises(ValueError, match="propagation_mode='coupled_eci'"):
        phase_compiler.ode_for_phase(phase)


def test_nonlinear_spice_solar_phase_angle_compiles_solves_and_reports() -> None:
    chief_radius_m = EARTH.mean_radius_m + 400_000.0
    chief_speed_mps = np.sqrt(EARTH.mu_m3ps2 / chief_radius_m)
    chief_state = state(
        [chief_radius_m, 0.0, 0.0],
        [0.0, chief_speed_mps, 0.0],
    )
    mission = _relative_rendezvous_mission()
    mission.initial_epoch = "2026-01-01T00:00:00Z"
    mission.mesh_nsegs_transfer = 20
    mission.lambert_grid_size = 20
    phase = mission.phases[0]
    stand_off_state = state([0.0, -100.0, 0.0], [0.0, 0.0, 0.0])
    phase.final_state = stand_off_state
    phase.constraints = [
        phase.constraints[0],
        constraints.state(stand_off_state, where="Back"),
    ]
    phase.dynamics = Dynamics.relative(
        chief_initial_state_eci=chief_state,
        perturbations=Perturbations(sun=True),
        third_body_table_step_s=300.0,
    )
    phase.constraints.append(
        constraints.solar_phase_angle(
            min_angle_deg=0.0,
            max_angle_deg=180.0,
        )
    )

    solution = mission.solve()

    assert solution.ok
    assert solution.result is not None
    report = solution.result.info["constraint_report"]
    solar_rows = [row for row in report if str(row["constraint"]).startswith("solar_phase_")]
    assert [row["constraint"] for row in solar_rows] == [
        "solar_phase_min_angle_deg",
        "solar_phase_max_angle_deg",
    ]
    assert all(row["satisfied"] for row in solar_rows)


def test_damico_target_selects_free_arrival_time_without_absolute_constraint() -> None:
    chief_position, chief_velocity = classical_to_cartesian(
        a_m=7_000_000.0,
        e=0.001,
        inc_deg=40.0,
        raan_deg=20.0,
        argp_deg=10.0,
        true_anomaly_deg=30.0,
        mu_m3ps2=EARTH.mu_m3ps2,
    )
    chief = state(chief_position, chief_velocity)
    initial_elements = RelativeOrbitalElements(
        delta_a=2.0e-4,
        delta_lambda_rad=-0.004,
        delta_ex=1.0e-4,
        delta_ey=-2.0e-4,
        delta_ix_rad=1.0e-4,
        delta_iy_rad=2.0e-4,
    )
    target_time_s = 1_800.0
    target_elements = propagate_relative_orbital_elements(
        initial_elements,
        [target_time_s],
        chief_initial_state_eci=chief,
        mu_m3ps2=EARTH.mu_m3ps2,
    )[0]
    initial_relative = relative_orbital_elements_to_relative_state(
        chief,
        initial_elements,
        mu_m3ps2=EARTH.mu_m3ps2,
    )
    target_chief = propagate_two_body_state(
        chief,
        target_time_s,
        EARTH.mu_m3ps2,
    )
    target_relative = relative_orbital_elements_to_relative_state(
        target_chief,
        target_elements[0:6],
        mu_m3ps2=EARTH.mu_m3ps2,
    )
    phase = Phase(
        name="damico_free_time",
        mode="relative_coast",
        spacecraft=Spacecraft(name="Deputy", dry_mass_kg=250.0),
        dynamics=Dynamics.relative(
            chief_initial_state_eci=chief,
            propagation_mode="damico",
        ),
        # These Cartesian states seed the optimizer only; native constraints
        # below are the values enforced by the NLP.
        initial_state=initial_relative,
        final_state=target_relative,
        tof_bounds_s=(1_500.0, 2_100.0),
        constraints=[
            constraints.relative_orbital_elements(
                initial_elements,
                where="Front",
            ),
            constraints.relative_orbital_element(
                "delta_lambda",
                float(target_elements[1]),
                where="Back",
            ),
        ],
    )
    solution = Mission(
        phases=[phase],
        objectives=[],
        mesh_nsegs_transfer=10,
        lambert_grid_size=10,
        solver_options=SolverOptions(
            print_level=0,
            max_ls_iters=5,
            asset_threads=(1, 1),
        ),
    ).solve()

    assert solution.ok
    assert solution.result is not None
    assert solution.result.tf_s() == pytest.approx(target_time_s, abs=1e-5)
    assert solution.result.info["state_layouts"] == ["damico_relative_elements"]
    assert solution.result.info["relative_propagation_mode"] == "damico"
    assert solution.result.info["constraint_report"][-1]["satisfied"] is True
