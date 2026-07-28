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
    constraints,
    objectives,
    state,
    variables,
)
from octavian.relative import cwh_rendezvous_velocity, propagate_cwh
from octavian.solvers import SolverOptions


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
        np.linalg.norm(departure_velocity)
        + np.linalg.norm(analytic_arrival[3:6])
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


def test_cwh_j2_differential_perturbation_compiles_and_solves() -> None:
    chief_radius_m = EARTH.mean_radius_m + 400_000.0
    chief_speed_mps = np.sqrt(EARTH.mu_m3ps2 / chief_radius_m)
    chief_state = state(
        [chief_radius_m, 0.0, 0.0],
        [0.0, chief_speed_mps, 0.0],
    )
    mission = _relative_rendezvous_mission()
    mission.mesh_nsegs_transfer = 20
    mission.lambert_grid_size = 20
    mission.phases[0].dynamics = Dynamics.cwh(
        chief_orbit_radius_m=chief_radius_m,
        chief_initial_state_eci=chief_state,
        perturbations=Perturbations(j2=True),
        third_body_table_step_s=300.0,
    )

    solution = mission.solve()

    assert solution.ok
    assert solution.result is not None
    assert solution.result.info["dynamics_model"] == "cwh_differential_perturbations"
    assert solution.result.info["relative_reference_model"] == "prescribed_circular_chief"


def test_spice_solar_phase_angle_compiles_solves_and_reports() -> None:
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
    phase.dynamics = Dynamics.cwh(
        chief_orbit_radius_m=chief_radius_m,
        chief_initial_state_eci=chief_state,
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
    solar_rows = [
        row for row in report if str(row["constraint"]).startswith("solar_phase_")
    ]
    assert [row["constraint"] for row in solar_rows] == [
        "solar_phase_min_angle_deg",
        "solar_phase_max_angle_deg",
    ]
    assert all(row["satisfied"] for row in solar_rows)
