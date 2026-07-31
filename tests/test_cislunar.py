from __future__ import annotations

import numpy as np
import pytest

from octavian import Dynamics, Mission, Phase, Spacecraft, constraints, state
from octavian.cislunar import (
    CR3BPSystem,
    cr3bp_derivative,
    dimensionalize_state,
    dimensionalize_time,
    inertial_to_synodic_state,
    jacobi_constant,
    nondimensionalize_state,
    propagate_cr3bp,
    synodic_to_inertial_state,
)
from octavian.solvers import SolverOptions
from octavian.types import Maneuver
from octavian.viz.diagnostics import cr3bp_diagnostic_panels
from octavian.viz.plotly import cr3bp_trajectory_figure


def test_earth_moon_system_scaling_and_lagrange_points() -> None:
    system = CR3BPSystem.earth_moon()

    assert system.mass_parameter == pytest.approx(0.01215565, rel=1.0e-6)
    assert system.period_s / 86_400.0 == pytest.approx(27.2845, rel=1.0e-5)
    assert system.frame.kind == "rotating"
    points = system.lagrange_points(dimensional=False)
    assert points["L1"][0] == pytest.approx(0.83689, rel=1.0e-5)
    assert points["L2"][0] == pytest.approx(1.15570, rel=1.0e-5)
    np.testing.assert_allclose(
        points["L4"],
        [0.5 - system.mass_parameter, np.sqrt(3.0) / 2.0, 0.0],
    )
    for point in points.values():
        derivative = cr3bp_derivative(
            np.hstack([point, np.zeros(3)]),
            system=system,
            dimensional=False,
        )
        np.testing.assert_allclose(derivative, np.zeros(6), atol=1.0e-12)


def test_cr3bp_scaling_and_synodic_inertial_round_trips() -> None:
    system = CR3BPSystem.earth_moon()
    canonical = state([0.82, 0.04, -0.02], [0.01, -0.02, 0.005])
    dimensional = dimensionalize_state(canonical, system)

    recovered_canonical = nondimensionalize_state(dimensional, system)
    np.testing.assert_allclose(recovered_canonical.r_m, canonical.r_m)
    np.testing.assert_allclose(recovered_canonical.v_mps, canonical.v_mps)

    inertial = synodic_to_inertial_state(
        dimensional,
        time_s=123_456.0,
        system=system,
        origin="earth",
        phase_at_epoch_rad=0.3,
    )
    recovered_synodic = inertial_to_synodic_state(
        inertial,
        time_s=123_456.0,
        system=system,
        origin="primary",
        phase_at_epoch_rad=0.3,
    )
    np.testing.assert_allclose(recovered_synodic.r_m, dimensional.r_m, atol=1.0e-7)
    np.testing.assert_allclose(recovered_synodic.v_mps, dimensional.v_mps, atol=1.0e-12)


def test_cr3bp_propagation_preserves_jacobi_constant() -> None:
    system = CR3BPSystem.earth_moon()
    initial_position = system.lagrange_points(dimensional=False)["L4"].copy()
    initial_position[0] += 1.0e-3
    initial = state(initial_position, [0.0, 0.0, 0.0])
    history = propagate_cr3bp(
        initial,
        np.linspace(0.0, 2.0, 201),
        system=system,
        dimensional=False,
        max_step=1.0e-3,
    )
    constants = np.asarray(
        [
            jacobi_constant(
                row[0:6],
                system=system,
                dimensional=False,
            )
            for row in history
        ]
    )
    assert np.ptp(constants) < 2.0e-12


def test_cr3bp_plot_contains_trajectory_primaries_and_lagrange_points() -> None:
    pytest.importorskip("plotly")
    system = CR3BPSystem.earth_moon()
    trajectory = np.zeros((2, 7), dtype=float)
    trajectory[:, 0] = system.lagrange_points()["L1"][0]
    trajectory[:, 6] = [0.0, 100.0]

    figure = cr3bp_trajectory_figure(trajectory, system=system)

    assert [trace.name for trace in figure.data] == [
        "Trajectory",
        "Primaries",
        "Lagrange points",
    ]
    assert list(figure.data[2].text) == ["L1", "L2", "L3", "L4", "L5"]
    panels = cr3bp_diagnostic_panels(trajectory, system=system)
    assert [panel.title for panel in panels] == [
        "Synodic position",
        "Synodic velocity",
        "Primary geometry",
        "CR3BP invariant",
    ]


def test_cr3bp_plot_can_overlay_references_phases_and_maneuvers() -> None:
    pytest.importorskip("plotly")
    system = CR3BPSystem.earth_moon()
    trajectory = np.zeros((4, 7), dtype=float)
    trajectory[:, 0] = [0.82, 0.83, 0.84, 0.85]
    trajectory[:, 6] = [0.0, 1.0, 2.0, 3.0]
    reference = trajectory.copy()
    reference[:, 1] = 0.01
    maneuver = Maneuver(
        r_m=trajectory[1, 0:3],
        t_s=1.0,
        dv_mps=[0.0, 0.01, 0.0],
        name="departure",
    )

    figure = cr3bp_trajectory_figure(
        trajectory,
        system=system,
        dimensional=False,
        maneuvers=[maneuver],
        phase_segments=[
            {
                "name": "transfer",
                "t_start_s": 0.0,
                "t_end_s": 3.0,
                "color": "gold",
            }
        ],
        reference_trajectories=[
            {
                "name": "L1 reference",
                "traj": reference,
                "color": "green",
            }
        ],
    )

    assert [trace.name for trace in figure.data] == [
        "Trajectory",
        "L1 reference",
        "transfer",
        "M1: departure",
        "Primaries",
        "Lagrange points",
    ]


def test_jacobi_constraint_validates_units_and_tolerance() -> None:
    constraint = constraints.jacobi_constant(
        3.16,
        where="start",
        tolerance=1.0e-5,
        dimensional=False,
    )

    assert constraint.target == pytest.approx(3.16)
    assert constraint.where == "Front"
    assert constraint.tolerance == pytest.approx(1.0e-5)
    assert constraint.dimensional is False
    with pytest.raises(ValueError, match="target must be finite"):
        constraints.jacobi_constant(np.nan)
    with pytest.raises(ValueError, match="tolerance must be >= 0"):
        constraints.jacobi_constant(3.16, tolerance=-1.0)


def test_composable_cr3bp_jacobi_targeted_periodic_state_solves() -> None:
    system = CR3BPSystem.earth_moon()
    canonical_initial = state(
        [0.82, 0.0, 0.0],
        [0.0, 0.16221305707437475, 0.0],
    )
    period_tu = 2.779749966597294
    target_jacobi = 3.16
    initial = dimensionalize_state(canonical_initial, system)
    period_s = float(dimensionalize_time(period_tu, system))
    propagated = propagate_cr3bp(
        initial,
        [0.0, period_s],
        system=system,
        max_step=300.0,
    )
    terminal_seed = state(propagated[-1, 0:3], propagated[-1, 3:6])
    phase = Phase(
        name="L1_planar_periodic",
        mode="coast",
        spacecraft=Spacecraft(name="probe", dry_mass_kg=1.0),
        dynamics=Dynamics.cr3bp(),
        initial_state=initial,
        final_state=terminal_seed,
        tof_bounds_s=(0.85 * period_s, 1.15 * period_s),
        constraints=[
            constraints.periodic_state(),
            constraints.state_component("y", 0.0, where="Front"),
            constraints.jacobi_constant(
                target_jacobi,
                where="Front",
                dimensional=False,
            ),
        ],
    )

    solution = Mission(
        phases=[phase],
        mesh_nsegs_transfer=40,
        solver_options=SolverOptions(
            print_level=0,
            max_ls_iters=3,
            enable_adaptive_mesh=False,
            asset_threads=(1, 1),
        ),
    ).solve()

    assert solution.ok
    closure_canonical = (solution.traj[-1, 0:6] - solution.traj[0, 0:6]) / np.hstack(
        [
            np.full(3, system.separation_m),
            np.full(3, system.velocity_scale_mps),
        ]
    )
    assert np.linalg.norm(closure_canonical) < 1.0e-7
    solved_jacobi = jacobi_constant(
        solution.traj[0, 0:6],
        system=system,
        dimensional=True,
    ) / system.velocity_scale_mps**2
    assert solved_jacobi == pytest.approx(target_jacobi, abs=1.0e-8)
    assert solution.traj[-1, 6] / system.time_scale_s == pytest.approx(
        2.801037769641,
        rel=1.0e-4,
    )
    report = solution.result.info["constraint_report"]
    assert len(report) == 1
    assert report[0]["constraint"] == "jacobi_constant"
    assert report[0]["actual"] == pytest.approx(target_jacobi, abs=1.0e-8)
    assert report[0]["satisfied"] is True


def test_composable_cr3bp_arc_compiles_and_solves() -> None:
    system = CR3BPSystem.earth_moon()
    initial_position = system.lagrange_points()["L4"].copy()
    initial_position[0] += 100_000.0
    initial = state(initial_position, [0.0, 0.0, 0.0])
    duration_s = 43_200.0
    propagated = propagate_cr3bp(
        initial,
        [0.0, duration_s],
        system=system,
        max_step=300.0,
    )
    final = state(propagated[-1, 0:3], propagated[-1, 3:6])
    phase = Phase(
        name="earth_moon_synodic_arc",
        mode="coast",
        spacecraft=Spacecraft(name="probe", dry_mass_kg=1.0),
        dynamics=Dynamics.cr3bp(),
        initial_state=initial,
        final_state=final,
        tof_bounds_s=(duration_s - 1.0, duration_s + 1.0),
        constraints=[
            constraints.state(initial, where="Front"),
            constraints.state(final, where="Back"),
        ],
    )
    solution = Mission(
        phases=[phase],
        mesh_nsegs_transfer=16,
        solver_options=SolverOptions(
            print_level=0,
            max_ls_iters=3,
            enable_adaptive_mesh=False,
            asset_threads=(1, 1),
        ),
    ).solve()

    assert solution.ok
    assert solution.frame is not None
    assert solution.frame.kind == "rotating"
    assert solution.result is not None
    assert solution.result.info["dynamics_model"] == "cr3bp"
    assert solution.result.info["cr3bp_system"]["primary"] == "earth"
    assert solution.cr3bp_system == system
    np.testing.assert_allclose(
        solution.traj[-1, 0:6],
        np.hstack([final.r_m, final.v_mps]),
        atol=1.0e-6,
    )
