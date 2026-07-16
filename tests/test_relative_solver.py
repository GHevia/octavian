from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("asset_asrl", exc_type=ImportError)

from octavian import (
    EARTH,
    Dynamics,
    Mission,
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
