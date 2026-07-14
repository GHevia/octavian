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
    objectives,
    state,
)
from octavian.solvers import SolverOptions

MU = 3.986004418e14
RADIUS_M = 7_000_000.0


def test_standalone_finite_thrust_phase_compiles_solves_and_reports() -> None:
    circular_speed_mps = float(np.sqrt(MU / RADIUS_M))
    period_s = float(2.0 * np.pi * np.sqrt(RADIUS_M**3 / MU))
    transfer_angle_rad = 2.0 * np.pi / 3.0
    transfer_time_s = period_s / 3.0
    initial_state = state(
        [RADIUS_M, 0.0, 0.0],
        [0.0, circular_speed_mps, 0.0],
    )
    final_state = state(
        [
            RADIUS_M * np.cos(transfer_angle_rad),
            RADIUS_M * np.sin(transfer_angle_rad),
            0.0,
        ],
        [
            -circular_speed_mps * np.sin(transfer_angle_rad),
            circular_speed_mps * np.cos(transfer_angle_rad),
            0.0,
        ],
    )
    spacecraft = Spacecraft(
        name="Powered test vehicle",
        dry_mass_kg=500.0,
        thrusters=[
            Thruster(
                name="main",
                thrust_N=20.0,
                isp_s=1_500.0,
                propellant_mass_kg=10.0,
            )
        ],
    )
    phase = Phase(
        name="powered_orbit",
        mode="finite_thrust",
        spacecraft=spacecraft,
        dynamics=Dynamics(mu_m3ps2=MU),
        initial_state=initial_state,
        final_state=final_state,
        tof_bounds_s=(transfer_time_s - 1.0, transfer_time_s + 1.0),
        constraints=[
            constraints.state(initial_state, where="Front"),
            constraints.state(final_state, where="Back"),
        ],
    )
    solution = Mission(
        phases=[phase],
        objectives=[objectives.minimize_propellant()],
        mesh_nsegs_transfer=30,
        lambert_grid_size=12,
        nrevs_to_try=(0,),
        solver_options=SolverOptions(
            print_level=0,
            max_ls_iters=3,
            enable_adaptive_mesh=False,
            asset_threads=(1, 1),
        ),
    ).solve()

    assert solution.ok
    assert solution.result is not None
    assert solution.result.info["state_layouts"] == ["cartesian_mass_thrust"]
    assert solution.result.info["chemical_burns"] == []
    powered = solution.result.info["powered_phases"]
    assert len(powered) == 1
    assert powered[0]["kind"] == "finite_thrust"
    assert 0.0 <= powered[0]["propellant_used_kg"] < 1.0
    assert powered[0]["mass_final_kg"] <= powered[0]["mass_initial_kg"]
    assert solution.result.last_obj == pytest.approx(
        -powered[0]["mass_final_kg"] / spacecraft.initial_mass_kg,
        rel=1e-8,
    )
