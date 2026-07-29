from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest

from octavian.solution import Solution
from octavian.solvers.preconfigured import RendezvousResult
from octavian.types import Maneuver

ROOT = Path(__file__).resolve().parents[1]


def _fake_solution() -> Solution:
    traj = np.zeros((3, 7), dtype=float)
    traj[:, 6] = [0.0, 100.0, 200.0]
    maneuvers = [Maneuver(r_m=[0.0, 0.0, 0.0], t_s=100.0, dv_mps=[2.0, 0.0, 0.0], name="dv")]
    res = RendezvousResult(
        converged=True,
        traj=traj,
        maneuvers=maneuvers,
        last_obj=2.0,
        info={
            "dynamics_model": "cwh",
            "state_layouts": ["coupled_relative_eci"],
            "mu_m3ps2": 3.986004418e14,
            "chemical_burns": [
                {
                    "phase": "burn",
                    "propellant_used_kg": 1.0,
                    "equivalent_dv_mps": 143.98885710585398,
                }
            ],
            "powered_phases": [
                {
                    "phase": "burn",
                    "kind": "low_thrust",
                    "propellant_used_kg": 1.0,
                    "equivalent_dv_mps": 143.98885710585398,
                }
            ],
            "phase_guess_info": {
                0: {
                    "seed_tof_s": 3_600.0,
                    "seed_final_radius_m": 8_000_000.0,
                }
            },
            "constraint_report": [],
            "phase_segments": [
                {"name": "burn", "t_start_s": 0.0, "t_end_s": 100.0, "color": "red"},
                {"name": "coast", "t_start_s": 100.0, "t_end_s": 200.0, "color": "blue"},
            ],
        },
    )
    return Solution(ok=True, result=res)


@pytest.mark.parametrize(
    ("script_rel", "expected_solve_calls"),
    [
        ("examples/composable/01_single_phase_terminal_dv_objective.py", 1),
        ("examples/composable/02_precoast_continuous_link.py", 1),
        ("examples/composable/03_precoast_impulsive_link.py", 1),
        ("examples/composable/04_terminal_velocity_hard_vs_objective.py", 2),
        ("examples/composable/05_plot_with_maneuvers.py", 1),
        ("examples/composable/06_precoast_impulsive_link_3burn.py", 1),
        ("examples/composable/07_terminal_orbital_elements.py", 2),
        ("examples/composable/08_chemical_burn_j2.py", 1),
        ("examples/composable/09_impulse_vs_chemical_burn.py", 1),
        ("examples/composable/10_sun_moon_perturbations.py", 1),
        ("examples/composable/11_cwh_relative_rendezvous.py", 1),
        ("examples/composable/12_cwh_safety_corridor.py", 1),
        ("examples/composable/14_nonlinear_relative_rendezvous.py", 1),
        ("examples/composable/15_perturbed_relative_solar.py", 1),
        ("examples/composable/17_damico_free_time_target.py", 1),
        ("examples/composable/18_low_thrust_orbit_raise.py", 1),
        ("examples/composable/18_safety_ellipse_transfer.py", 1),
        ("examples/composable/19_relative_finite_burn_coast.py", 1),
        ("examples/composable/20_relative_three_burn_transfer.py", 1),
        ("examples/composable/21_perturbed_relative_element_propagation.py", 0),
    ],
)
def test_composable_examples_run_as_scenarios(monkeypatch: pytest.MonkeyPatch, script_rel: str, expected_solve_calls: int) -> None:
    missions = []
    plotted = []
    plotted_trajectories = []

    def fake_solve(self):  # type: ignore[no-untyped-def]
        missions.append(self)
        return _fake_solution()

    def fake_plot(*args, **kwargs):  # type: ignore[no-untyped-def]
        if args:
            plotted_trajectories.append(np.asarray(args[0], dtype=float))
        if "out_html" in kwargs:
            plotted.append(kwargs["out_html"])
        elif len(args) >= 2:
            plotted.append(args[1])

    monkeypatch.setattr("octavian.mission.Mission.solve", fake_solve, raising=True)
    monkeypatch.setattr("octavian.viz.plotly.save_trajectory_html", fake_plot, raising=True)
    monkeypatch.setattr("octavian.viz.save_trajectory_html", fake_plot, raising=True)
    monkeypatch.setattr(
        "octavian.viz.plotly.save_relative_trajectory_html",
        fake_plot,
        raising=True,
    )
    monkeypatch.setattr(
        "octavian.viz.save_relative_trajectory_html",
        fake_plot,
        raising=True,
    )
    monkeypatch.setattr(
        "octavian.viz.plotly.save_trajectory_diagnostics_html",
        fake_plot,
        raising=True,
    )
    monkeypatch.setattr(
        "octavian.viz.save_trajectory_diagnostics_html",
        fake_plot,
        raising=True,
    )

    runpy.run_path(str(ROOT / script_rel), run_name="__main__")

    assert len(missions) == expected_solve_calls
    assert len(plotted) >= 1

    if script_rel.endswith("01_single_phase_terminal_dv_objective.py"):
        assert len(missions[0].phases) == 1
        assert missions[0].phases[0].variables
    elif script_rel.endswith("02_precoast_continuous_link.py"):
        assert len(missions[0].phases) == 2
        assert missions[0].phases[1].link.kind == "continuous"
    elif script_rel.endswith("03_precoast_impulsive_link.py"):
        assert len(missions[0].phases) == 2
        assert missions[0].phases[1].link.kind == "impulsive"
    elif script_rel.endswith("06_precoast_impulsive_link_3burn.py"):
        assert len(missions[0].phases) == 3
        assert missions[0].phases[1].link.kind == "impulsive"
        assert missions[0].phases[2].link.kind == "impulsive"
    elif script_rel.endswith("07_terminal_orbital_elements.py"):
        assert len(missions) == 2
        assert all(len(mission.phases) == 1 for mission in missions)
        assert len(missions[0].phases[0].variables) == 1
        assert len(missions[1].phases[0].variables) == 2
    elif script_rel.endswith("08_chemical_burn_j2.py"):
        assert [phase.mode for phase in missions[0].phases] == ["chemical_burn", "coast", "chemical_burn"]
        assert plotted == ["traj_composable_chemical_burn_j2.html"]
    elif script_rel.endswith("09_impulse_vs_chemical_burn.py"):
        assert [phase.mode for phase in missions[0].phases] == ["chemical_burn", "coast", "chemical_burn"]
        assert plotted == [
            "traj_composable_impulse_reference.html",
            "traj_composable_chemical_reference.html",
        ]
    elif script_rel.endswith("10_sun_moon_perturbations.py"):
        mission = missions[0]
        perturbations = mission.phases[0].dynamics.active_perturbations()
        assert mission.initial_epoch == "2026-01-01T00:00:00Z"
        assert perturbations.j2 is True
        assert perturbations.active_third_bodies() == ("moon", "sun")
        assert plotted == ["traj_composable_sun_moon_perturbations.html"]
    elif script_rel.endswith("11_cwh_relative_rendezvous.py"):
        mission = missions[0]
        assert mission.phases[0].mode == "relative_coast"
        assert mission.phases[0].dynamics.frame.kind == "relative"
        assert mission.phases[0].dynamics.model.mean_motion_radps > 0.0
        assert plotted == [
            "traj_composable_cwh_relative_rendezvous.html",
            "diagnostics_composable_cwh_relative_rendezvous.html",
        ]
    elif script_rel.endswith("12_cwh_safety_corridor.py"):
        mission = missions[0]
        kinds = [constraint.kind for constraint in mission.phases[0].constraints]
        assert "keep_out_sphere" in kinds
        assert "approach_cone" in kinds
        assert "lighting_angle" in kinds
        assert plotted == [
            "traj_composable_cwh_safety_corridor.html",
            "diagnostics_composable_cwh_safety_corridor.html",
        ]
    elif script_rel.endswith("14_nonlinear_relative_rendezvous.py"):
        phase = missions[0].phases[0]
        assert phase.dynamics.model.chief_initial_state_eci is not None
        assert phase.dynamics.active_perturbations().j2 is False
        assert plotted == [
            "traj_composable_nonlinear_relative_rendezvous.html",
            "diagnostics_composable_nonlinear_relative_rendezvous.html",
        ]
    elif script_rel.endswith("18_low_thrust_orbit_raise.py"):
        mission = missions[0]
        phase = mission.phases[0]
        assert phase.mode == "low_thrust"
        assert phase.initial_guess.throttle == pytest.approx(0.85)
        assert mission.objectives[0].kind == "propellant"
        assert plotted == [
            "traj_composable_low_thrust_orbit_raise.html",
            "diagnostics_composable_low_thrust_orbit_raise.html",
        ]
    elif script_rel.endswith("18_safety_ellipse_transfer.py"):
        mission = missions[0]
        phase = mission.phases[0]
        assert phase.dynamics.model.propagation_mode.value == "coupled_eci"
        perturbations = phase.dynamics.active_perturbations()
        assert perturbations.j2 is True
        assert perturbations.sun is True
        assert mission.initial_epoch is not None
        assert [constraint.kind for constraint in phase.constraints] == [
            "state",
            "state",
        ]
        assert [variable.where for variable in phase.variables] == [
            "Front",
            "Back",
        ]
        assert mission.objectives[0].kind == "delta_v"
        assert plotted == [
            "traj_safety_ellipse_transfer.html",
            "diagnostics_safety_ellipse_transfer.html",
        ]
        assert len(plotted_trajectories) == 2
        assert plotted_trajectories[0][0, 6] == pytest.approx(0.0)
        assert plotted_trajectories[0][-1, 6] == pytest.approx(12_200.0)
        assert plotted_trajectories[0].shape[0] > _fake_solution().traj.shape[0]
    elif script_rel.endswith("17_damico_free_time_target.py"):
        mission = missions[0]
        phase = mission.phases[0]
        assert phase.dynamics.model.propagation_mode.value == "damico"
        assert [constraint.kind for constraint in phase.constraints] == [
            "relative_orbital_elements",
            "relative_orbital_element",
        ]
        assert plotted == [
            "traj_composable_damico_free_time.html",
            "diagnostics_composable_damico_free_time.html",
        ]
        assert len(plotted_trajectories) == 2
        assert plotted_trajectories[0][0, 6] == pytest.approx(0.0)
        assert plotted_trajectories[0][-1, 6] == pytest.approx(1_400.0)
        assert plotted_trajectories[0].shape[0] > _fake_solution().traj.shape[0]
    elif script_rel.endswith("19_relative_finite_burn_coast.py"):
        mission = missions[0]
        assert [phase.mode for phase in mission.phases] == [
            "finite_thrust",
            "relative_coast",
            "finite_thrust",
        ]
        assert all(
            phase.dynamics.model.propagation_mode.value == "coupled_eci"
            for phase in mission.phases
        )
        assert mission.phases[1].previous is mission.phases[0]
        assert mission.phases[2].previous is mission.phases[1]
        assert mission.objectives[0].kind == "propellant"
        assert plotted == [
            "traj_composable_relative_finite_burn_coast.html",
            "diagnostics_composable_relative_finite_burn_coast.html",
        ]
    elif script_rel.endswith("20_relative_three_burn_transfer.py"):
        mission = missions[0]
        assert [phase.name for phase in mission.phases] == [
            "initial_coast",
            "transfer_1",
            "transfer_2",
        ]
        assert all(phase.tof_is_relative for phase in mission.phases)
        assert [len(phase.variables) for phase in mission.phases] == [0, 1, 2]
        assert plotted == [
            "traj_composable_relative_three_burn.html",
            "diagnostics_composable_relative_three_burn.html",
        ]
    elif script_rel.endswith("21_perturbed_relative_element_propagation.py"):
        assert missions == []
        assert plotted == [
            "traj_perturbed_relative_elements.html",
            "diagnostics_perturbed_relative_elements.html",
        ]
    elif script_rel.endswith("15_perturbed_relative_solar.py"):
        mission = missions[0]
        phase = mission.phases[0]
        perturbations = phase.dynamics.active_perturbations()
        assert mission.initial_epoch == "2026-01-01T00:00:00Z"
        assert perturbations.j2 is True
        assert perturbations.sun is True
        assert phase.dynamics.model.chief_initial_state_eci is not None
        assert "solar_phase_angle" in [
            constraint.kind for constraint in phase.constraints
        ]
        assert plotted == [
            "traj_composable_perturbed_relative_solar.html",
            "diagnostics_composable_perturbed_relative_solar.html",
        ]
    elif script_rel.endswith("07_terminal_orbital_elements.py"):
        assert len(missions[0].phases) == 1
        kinds = [getattr(c, "kind", "") for c in missions[0].phases[0].constraints]
        assert "semi_major_axis" in kinds
        assert "eccentricity" in kinds
        assert "inclination_deg" in kinds
        assert len(missions[0].phases[0].variables) == 2


def test_relative_representation_example_round_trips(capsys) -> None:
    runpy.run_path(
        str(ROOT / "examples/composable/13_relative_representations.py"),
        run_name="__main__",
    )
    output = capsys.readouterr().out
    assert "Recovered deputy RIC state" in output
    assert "relative orbital elements" in output
