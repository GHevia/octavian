from __future__ import annotations

import numpy as np
import pytest

from octavian import (
    EARTH,
    Dynamics,
    Perturbations,
    relative_hop,
    relative_transfer_chain,
    state,
)
from octavian.solvers import SolverOptions


def _chief_state():
    """Return a circular inclined-test surrogate chief state."""
    radius_m = EARTH.mean_radius_m + 400_000.0
    return state(
        [radius_m, 0.0, 0.0],
        [0.0, np.sqrt(EARTH.mu_m3ps2 / radius_m), 0.0],
    )


def test_relative_hop_builds_precoast_and_two_impulses() -> None:
    initial = state([0.0, -1_000.0, 0.0], [0.0, 0.0, 0.0])
    target = state([0.0, -100.0, 0.0], [0.0, 0.0, 0.0])

    mission = relative_hop(
        initial,
        target,
        chief_initial_state_eci=_chief_state(),
        departure_coast_time_bounds_s=(100.0, 300.0),
        transfer_time_bounds_s=(900.0, 1_800.0),
        perturbations=Perturbations(j2=True),
        initial_epoch="2026-01-01T00:00:00Z",
    )

    assert [phase.name for phase in mission.phases] == [
        "departure_coast",
        "transfer_1",
    ]
    assert mission.phases[1].previous is mission.phases[0]
    assert mission.phases[1].link is not None
    assert mission.phases[1].link.is_impulsive()
    assert [variable.where for variable in mission.phases[1].variables] == [
        "Front",
        "Back",
    ]
    assert mission.phases[0].dynamics is mission.phases[1].dynamics
    assert mission.phases[0].dynamics.active_perturbations().j2 is True
    assert mission.objectives[0].kind == "delta_v"
    assert mission.initial_epoch == "2026-01-01T00:00:00Z"


def test_relative_transfer_chain_inserts_post_arrival_coast() -> None:
    initial = state([0.0, -1_000.0, 0.0], [0.0, 0.0, 0.0])
    first_target = state([100.0, -500.0, 0.0], [0.0, 0.0, 0.0])
    final_target = state([0.0, -100.0, 0.0], [0.0, 0.0, 0.0])

    mission = relative_transfer_chain(
        initial,
        [first_target, final_target],
        chief_initial_state_eci=_chief_state(),
        transfer_time_bounds_s=[(600.0, 900.0), (900.0, 1_200.0)],
        coast_time_bounds_s=(300.0, 600.0),
    )

    assert [phase.name for phase in mission.phases] == [
        "transfer_1",
        "target_1_coast",
        "transfer_2",
    ]
    assert [phase.previous for phase in mission.phases] == [
        None,
        mission.phases[0],
        mission.phases[1],
    ]
    assert [variable.where for variable in mission.phases[1].variables] == [
        "Front"
    ]
    front_constraint = mission.phases[1].constraints[0]
    assert front_constraint.where == "Front"
    np.testing.assert_allclose(front_constraint.x.r_m, first_target.r_m)
    np.testing.assert_allclose(front_constraint.x.v_mps, first_target.v_mps)
    assert [variable.where for variable in mission.phases[2].variables] == [
        "Front",
        "Back",
    ]


def test_relative_transfer_chain_rejects_inconsistent_bounds() -> None:
    initial = state([0.0, -1_000.0, 0.0], [0.0, 0.0, 0.0])
    target = state([0.0, -100.0, 0.0], [0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="must contain 2 bound pairs"):
        relative_transfer_chain(
            initial,
            [target, target],
            chief_initial_state_eci=_chief_state(),
            transfer_time_bounds_s=[(600.0, 900.0)],
        )


def test_relative_transfer_chain_accepts_preconfigured_cwh_dynamics() -> None:
    chief = _chief_state()
    initial = state([0.0, -1_000.0, 0.0], [0.0, 0.0, 0.0])
    target = state([0.0, -100.0, 0.0], [0.0, 0.0, 0.0])
    dynamics = Dynamics.cwh(
        chief_orbit_radius_m=float(np.linalg.norm(chief.r_m)),
        chief_initial_state_eci=chief,
    )

    mission = relative_transfer_chain(
        initial,
        [target],
        chief_initial_state_eci=chief,
        dynamics=dynamics,
    )

    assert mission.phases[0].dynamics is dynamics


def test_relative_transfer_chain_solves_and_honors_waypoint_coast() -> None:
    chief = _chief_state()
    radius_m = float(np.linalg.norm(chief.r_m))
    initial = state([0.0, -1_000.0, 0.0], [0.0, 0.0, 0.0])
    first_target = state([0.0, -500.0, 0.0], [0.0, 0.0, 0.0])
    final_target = state([0.0, -100.0, 0.0], [0.0, 0.0, 0.0])
    dynamics = Dynamics.cwh(
        chief_orbit_radius_m=radius_m,
        chief_initial_state_eci=chief,
    )
    mission = relative_transfer_chain(
        initial,
        [first_target, final_target],
        chief_initial_state_eci=chief,
        dynamics=dynamics,
        transfer_time_bounds_s=[(500.0, 800.0), (500.0, 800.0)],
        coast_time_bounds_s=(100.0, 200.0),
        nsegs_coast=8,
        nsegs_transfer=10,
        seed_grid_size=12,
        solver_options=SolverOptions(
            print_level=0,
            enable_adaptive_mesh=False,
            asset_threads=(1, 1),
        ),
    )

    solution = mission.solve()

    assert solution.ok
    phase_trajectories = solution.native_relative_phase_trajectories
    durations_s = [
        float(trajectory[-1, 6] - trajectory[0, 6])
        for trajectory in phase_trajectories
    ]
    assert 500.0 <= durations_s[0] <= 800.0
    assert 100.0 <= durations_s[1] <= 200.0
    assert 500.0 <= durations_s[2] <= 800.0
    np.testing.assert_allclose(
        phase_trajectories[1][0, 0:6],
        np.hstack([first_target.r_m, first_target.v_mps]),
        atol=1.0e-8,
    )
    assert solution.result is not None
    assert len(solution.result.maneuvers) == 4
