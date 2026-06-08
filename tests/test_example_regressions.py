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
    links,
    objectives,
    state,
    variables,
)
from octavian.astro import classical_to_cartesian
from octavian.solvers import SolverOptions

MU = 3.986004418e14
DEFAULT_OPTS = SolverOptions(print_level=0)
HOHMANN_R0_M = 7_000e3
HOHMANN_RF_M = 12_000e3
LINK_R_FINAL_M = 10_000e3
LINK_TARGET_ANOMALY_RAD = np.deg2rad(140.0)


def _circular_state(radius_m: float, *, true_anomaly_rad: float = 0.0):
    c = float(np.cos(true_anomaly_rad))
    s = float(np.sin(true_anomaly_rad))
    speed = float(np.sqrt(MU / radius_m))
    return state(
        r_m=[radius_m * c, radius_m * s, 0.0],
        v_mps=[-speed * s, speed * c, 0.0],
    )


def _hohmann_reference() -> tuple[float, float]:
    transfer_a_m = 0.5 * (HOHMANN_R0_M + HOHMANN_RF_M)
    v0_mps = np.sqrt(MU / HOHMANN_R0_M)
    vf_mps = np.sqrt(MU / HOHMANN_RF_M)
    transfer_perigee_mps = np.sqrt(MU * (2.0 / HOHMANN_R0_M - 1.0 / transfer_a_m))
    transfer_apogee_mps = np.sqrt(MU * (2.0 / HOHMANN_RF_M - 1.0 / transfer_a_m))
    dv_mps = float((transfer_perigee_mps - v0_mps) + (vf_mps - transfer_apogee_mps))
    tof_s = float(np.pi * np.sqrt((transfer_a_m**3) / MU))
    return dv_mps, tof_s


def _solve_ok(mission: Mission):
    sol = mission.solve(solver_options=DEFAULT_OPTS)
    assert sol.result is not None
    assert sol.result.converged
    return sol.result


def _assert_results_close(
    lhs,
    rhs,
    *,
    tf_rtol: float = 5e-2,
    dv_rtol: float = 5e-2,
    tf_atol_s: float = 30.0,
    dv_atol_mps: float = 5.0,
    pos_atol_m: float = 5.0e3,
) -> None:
    assert np.isclose(lhs.tf_s(), rhs.tf_s(), rtol=tf_rtol, atol=tf_atol_s)
    assert np.isclose(lhs.total_dv_mps(), rhs.total_dv_mps(), rtol=dv_rtol, atol=dv_atol_mps)
    assert np.allclose(lhs.traj[-1, 0:3], rhs.traj[-1, 0:3], atol=pos_atol_m)


def _demo_spacecraft() -> Spacecraft:
    return Spacecraft(name="DemoSat", dry_mass_kg=150.0, thrusters=[Thruster(name="main")])


def _quick01_mission() -> Mission:
    x0 = _circular_state(HOHMANN_R0_M)
    xf = _circular_state(HOHMANN_RF_M, true_anomaly_rad=np.pi)
    from octavian import two_burn_rendezvous

    return two_burn_rendezvous(
        x0,
        xf,
        mu_m3ps2=MU,
        tf_bounds_s=(1_200.0, 12_000.0),
        nsegs=60,
        lambert_grid_size=60,
        nrevs_to_try=(0,),
        solver_options=DEFAULT_OPTS,
        name="Quick: Hohmann transfer between circular orbits",
    )


def _composable01_mission() -> Mission:
    spacecraft = _demo_spacecraft()
    dynamics = Dynamics(mu_m3ps2=MU)
    x0 = _circular_state(HOHMANN_R0_M)
    xf = _circular_state(HOHMANN_RF_M, true_anomaly_rad=np.pi)

    phase = Phase(
        name="transfer",
        mode="coast",
        spacecraft=spacecraft,
        dynamics=dynamics,
        tof_bounds_s=(3_000.0, 7_000.0),
        constraints=[constraints.state(x0, where="Front"), constraints.state(xf, where="Back")],
        variables=[
            variables.ImpulsiveDeltaV(where="Front"),
            variables.ImpulsiveDeltaV(where="Back"),
        ],
    )
    return Mission(
        name="Composable: Hohmann transfer between circular orbits",
        phases=[phase],
        objectives=[objectives.minimize_total_delta_v()],
        solver_options=DEFAULT_OPTS,
        lambert_grid_size=60,
        nrevs_to_try=(0,),
    )


def _quick02_mission() -> Mission:
    x0 = _circular_state(HOHMANN_R0_M)
    xf = _circular_state(HOHMANN_RF_M, true_anomaly_rad=np.pi)
    from octavian import two_burn_rendezvous

    return two_burn_rendezvous(
        x0,
        xf,
        mu_m3ps2=MU,
        precoast=True,
        t1_bounds_s=(1.0, 1_000.0),
        tf_bounds_s=(3_000.0, 7_000.0),
        nsegs=60,
        precoast_grid_size=12,
        lambert_grid_size=50,
        solver_options=DEFAULT_OPTS,
        nrevs_to_try=(0,),
        name="Quick: precoast plus circular-orbit transfer",
    )


def _composable_quick02_equivalent() -> Mission:
    spacecraft = _demo_spacecraft()
    dynamics = Dynamics(mu_m3ps2=MU)
    x0 = _circular_state(HOHMANN_R0_M)
    xf = _circular_state(HOHMANN_RF_M, true_anomaly_rad=np.pi)
    precoast = Phase(
        name="precoast",
        mode="coast",
        spacecraft=spacecraft,
        dynamics=dynamics,
        tof_bounds_s=(1.0, 1_000.0),
        constraints=[constraints.state(x0, where="Front")],
    )
    transfer = Phase(
        name="transfer",
        mode="coast",
        spacecraft=spacecraft,
        dynamics=dynamics,
        previous=precoast,
        link=links.impulsive(),
        tof_bounds_s=(3_000.0, 7_000.0),
        constraints=[constraints.state(xf, where="Back")],
        variables=[
            variables.ImpulsiveDeltaV(where="Front"),
            variables.ImpulsiveDeltaV(where="Back"),
        ],
    )
    return Mission(
        name="Composable quick02 equivalent",
        phases=[precoast, transfer],
        objectives=[objectives.minimize_total_delta_v()],
        solver_options=DEFAULT_OPTS,
        mesh_nsegs_precoast=30,
        mesh_nsegs_transfer=60,
        precoast_grid_size=12,
        lambert_grid_size=50,
        nrevs_to_try=(0, 1),
    )


def _quick03_mission(w_time: float) -> Mission:
    x0 = _circular_state(HOHMANN_R0_M)
    xf = _circular_state(LINK_R_FINAL_M, true_anomaly_rad=np.deg2rad(120.0))
    from octavian import two_burn_rendezvous

    return two_burn_rendezvous(
        x0,
        xf,
        mu_m3ps2=MU,
        tf_bounds_s=(600.0, 20_000.0),
        nsegs=60,
        lambert_grid_size=60,
        nrevs_to_try=(0, 1),
        w_time=w_time,
        solver_options=DEFAULT_OPTS,
        name=f"Quick: two-impulse (w_time={w_time})",
    )


def _composable_single_phase_mission(x0, xf, *, tf_bounds_s, w_time: float) -> Mission:
    spacecraft = _demo_spacecraft()
    dynamics = Dynamics(mu_m3ps2=MU)
    objs = [objectives.minimize_total_delta_v()]
    if w_time != 0.0:
        objs.append(objectives.minimize_total_time(weight=w_time))

    phase = Phase(
        name="transfer",
        mode="coast",
        spacecraft=spacecraft,
        dynamics=dynamics,
        tof_bounds_s=tf_bounds_s,
        constraints=[constraints.state(x0, where="Front"), constraints.state(xf, where="Back")],
        variables=[
            variables.ImpulsiveDeltaV(where="Front"),
            variables.ImpulsiveDeltaV(where="Back"),
        ],
    )
    return Mission(
        name=f"Composable single-phase (w_time={w_time})",
        phases=[phase],
        objectives=objs,
        solver_options=DEFAULT_OPTS,
        lambert_grid_size=60,
        nrevs_to_try=(0, 1),
    )


def _composable02_mission() -> Mission:
    spacecraft = _demo_spacecraft()
    dynamics = Dynamics(mu_m3ps2=MU)
    x0 = _circular_state(HOHMANN_R0_M)
    xf = _circular_state(LINK_R_FINAL_M, true_anomaly_rad=LINK_TARGET_ANOMALY_RAD)
    precoast = Phase(
        name="precoast",
        mode="coast",
        spacecraft=spacecraft,
        dynamics=dynamics,
        tof_bounds_s=(0.0, 6000.0),
        constraints=[constraints.state(x0, where="Front")],
        variables=[variables.ImpulsiveDeltaV(where="Front")],
    )
    transfer = Phase(
        name="transfer",
        mode="coast",
        spacecraft=spacecraft,
        dynamics=dynamics,
        previous=precoast,
        link=links.continuous(),
        tof_bounds_s=(400.0, 60_000.0),
        constraints=[constraints.state(xf, where="Back")],
        variables=[variables.ImpulsiveDeltaV(where="Back")],
    )
    return Mission(
        name="Composable: precoast + transfer (continuous link)",
        phases=[precoast, transfer],
        objectives=[objectives.minimize_total_delta_v()],
        solver_options=DEFAULT_OPTS,
    )


def _composable02_single_phase_equivalent() -> Mission:
    spacecraft = _demo_spacecraft()
    dynamics = Dynamics(mu_m3ps2=MU)
    x0 = _circular_state(HOHMANN_R0_M)
    xf = _circular_state(LINK_R_FINAL_M, true_anomaly_rad=LINK_TARGET_ANOMALY_RAD)
    phase = Phase(
        name="transfer",
        mode="coast",
        spacecraft=spacecraft,
        dynamics=dynamics,
        tof_bounds_s=(400.0, 60_000.0),
        constraints=[constraints.state(x0, where="Front"), constraints.state(xf, where="Back")],
        variables=[
            variables.ImpulsiveDeltaV(where="Front"),
            variables.ImpulsiveDeltaV(where="Back"),
        ],
    )
    return Mission(
        name="Composable single-phase equivalent of example 02",
        phases=[phase],
        objectives=[objectives.minimize_total_delta_v()],
        solver_options=DEFAULT_OPTS,
    )


def _composable03_mission() -> Mission:
    spacecraft = _demo_spacecraft()
    dynamics = Dynamics(mu_m3ps2=MU)
    x0 = _circular_state(HOHMANN_R0_M)
    xf = _circular_state(LINK_R_FINAL_M, true_anomaly_rad=LINK_TARGET_ANOMALY_RAD)
    precoast = Phase(
        name="precoast",
        mode="coast",
        spacecraft=spacecraft,
        dynamics=dynamics,
        tof_bounds_s=(0.0, 6000.0),
        constraints=[constraints.state(x0, where="Front")],
    )
    transfer = Phase(
        name="transfer",
        mode="coast",
        spacecraft=spacecraft,
        dynamics=dynamics,
        previous=precoast,
        link=links.impulsive(),
        tof_bounds_s=(400.0, 60_000.0),
        constraints=[constraints.state(xf, where="Back")],
        variables=[
            variables.ImpulsiveDeltaV(where="Front"),
            variables.ImpulsiveDeltaV(where="Back"),
        ],
    )
    return Mission(
        name="Composable: precoast + transfer (impulsive link)",
        phases=[precoast, transfer],
        objectives=[objectives.minimize_total_delta_v()],
        solver_options=DEFAULT_OPTS,
    )


def _composable07_mission() -> Mission:
    spacecraft = Spacecraft(
        name="DemoSat",
        dry_mass_kg=150.0,
        thrusters=[Thruster(name="main", thrust_N=0.0, isp_s=1e9)],
    )
    dynamics = Dynamics(mu_m3ps2=MU)
    x0 = _circular_state(HOHMANN_R0_M)
    xf = _circular_state(LINK_R_FINAL_M, true_anomaly_rad=LINK_TARGET_ANOMALY_RAD)
    precoast = Phase(
        name="precoast",
        mode="coast",
        spacecraft=spacecraft,
        dynamics=dynamics,
        tof_bounds_s=(0.0, 6000.0),
        constraints=[constraints.state(x0, where="Front")],
    )
    transfer = Phase(
        name="transfer",
        mode="coast",
        spacecraft=spacecraft,
        dynamics=dynamics,
        previous=precoast,
        tof_bounds_s=(400.0, 60000.0),
        link=links.impulsive(),
        constraints=[constraints.state(xf, where="Back")],
        variables=[
            variables.ImpulsiveDeltaV(where="Front"),
            variables.ImpulsiveDeltaV(where="Back"),
        ],
    )
    return Mission(
        phases=[precoast, transfer],
        name="Composable mission: precoast + impulsive link + terminal dv",
        objectives=[objectives.minimize_total_delta_v()],
        solver_options=DEFAULT_OPTS,
    )


def _composable05_mission() -> tuple[Mission, float]:
    spacecraft = _demo_spacecraft()
    dynamics = Dynamics(mu_m3ps2=MU)
    r_min_m = 6378.1363e3 + 60e3
    x0 = _circular_state(HOHMANN_R0_M)
    xf = _circular_state(HOHMANN_RF_M, true_anomaly_rad=np.pi)
    precoast = Phase(
        name="precoast",
        mode="coast",
        spacecraft=spacecraft,
        dynamics=dynamics,
        tof_bounds_s=(1.0, 600.0),
        constraints=[constraints.state(x0, where="Front")],
    )
    transfer = Phase(
        name="transfer",
        mode="coast",
        spacecraft=spacecraft,
        dynamics=dynamics,
        previous=precoast,
        link=links.impulsive(),
        tof_bounds_s=(600.0, 60_000.0),
        constraints=[constraints.state(xf, where="Back"), constraints.min_radius(r_min_m, where="Path")],
        variables=[
            variables.ImpulsiveDeltaV(where="Front"),
            variables.ImpulsiveDeltaV(where="Back"),
        ],
    )
    mission = Mission(
        name="Composable: plotting maneuvers",
        phases=[precoast, transfer],
        objectives=[objectives.minimize_total_delta_v()],
        solver_options=DEFAULT_OPTS,
    )
    return mission, r_min_m


def _composable06_mission() -> Mission:
    spacecraft = _demo_spacecraft()
    dynamics = Dynamics(mu_m3ps2=MU)
    r_min_m = 6378.1363e3 + 60e3
    x0 = _circular_state(HOHMANN_R0_M)
    xf = _circular_state(HOHMANN_RF_M, true_anomaly_rad=np.pi)
    precoast = Phase(
        name="precoast",
        mode="coast",
        spacecraft=spacecraft,
        dynamics=dynamics,
        tof_bounds_s=(0.0, 600.0),
        tof_is_relative=True,
        constraints=[constraints.state(x0, where="Front"), constraints.min_radius(r_min_m, where="Path")],
    )
    transfer1 = Phase(
        name="transfer1",
        mode="coast",
        spacecraft=spacecraft,
        dynamics=dynamics,
        previous=precoast,
        link=links.impulsive(),
        tof_bounds_s=(600.0, 30_000.0),
        tof_is_relative=True,
        variables=[variables.ImpulsiveDeltaV(where="Front")],
        constraints=[constraints.min_radius(r_min_m, where="Path")],
    )
    transfer2 = Phase(
        name="transfer2",
        mode="coast",
        spacecraft=spacecraft,
        dynamics=dynamics,
        previous=transfer1,
        link=links.impulsive(),
        tof_bounds_s=(600.0, 30_000.0),
        tof_is_relative=True,
        constraints=[constraints.state(xf, where="Back")],
        variables=[
            variables.ImpulsiveDeltaV(where="Front"),
            variables.ImpulsiveDeltaV(where="Back"),
        ],
    )
    return Mission(
        name="Composable: precoast + two transfers (impulsive link)",
        phases=[precoast, transfer1, transfer2],
        objectives=[objectives.minimize_total_delta_v()],
        solver_options=DEFAULT_OPTS,
    )


def _composable08_mission() -> Mission:
    spacecraft = Spacecraft(
        name="Deputy",
        dry_mass_kg=120.0,
        thrusters=[Thruster(name="main", thrust_N=0.0, isp_s=0.0)],
    )
    dynamics = Dynamics(mu_m3ps2=MU)
    x0 = state(
        r_m=[7000e3, 0.0, 0.0],
        v_mps=[0.0, float(np.sqrt(MU / 7000e3)), 0.0],
    )
    xf = state(
        r_m=[6100e3, 5000e3, 0.0],
        v_mps=[0.0, float(np.sqrt(MU / 7000e3)), 0.0],
    )
    precoast = Phase(
        name="precoast",
        mode="coast",
        spacecraft=spacecraft,
        dynamics=dynamics,
        initial_state=x0,
        tof_bounds_s=(0.0, 6000.0),
    )
    rendezvous = Phase(
        name="rendezvous",
        mode="rendezvous",
        previous=precoast,
        final_state=xf,
        tof_bounds_s=(400.0, 60000.0),
    )
    return Mission(
        phases=[precoast, rendezvous],
        name="Composable mission: pre-coast rendezvous",
        mesh_nsegs_precoast=40,
        mesh_nsegs_transfer=80,
        lambert_grid_size=100,
        solver_options=DEFAULT_OPTS,
    )


def _composable09_mission(*, use_terminal_burn: bool) -> Mission:
    spacecraft = _demo_spacecraft()
    dynamics = Dynamics(mu_m3ps2=MU)
    initial_state = state(
        r_m=[7000e3, 0.0, 0.0],
        v_mps=[0.0, float(np.sqrt(MU / 7000e3)), 250.0],
    )
    target_a_m = 8_400e3
    target_e = 0.18
    target_inc_deg = 28.5
    guess_r_m, guess_v_mps = classical_to_cartesian(
        a_m=target_a_m,
        e=target_e,
        inc_deg=target_inc_deg,
        raan_deg=35.0,
        argp_deg=20.0,
        true_anomaly_deg=70.0,
        mu_m3ps2=MU,
    )
    terminal_guess = state(r_m=guess_r_m, v_mps=guess_v_mps)

    phase_variables = [variables.ImpulsiveDeltaV(where="Front")]
    if use_terminal_burn:
        phase_variables.append(variables.ImpulsiveDeltaV(where="Back"))

    transfer = Phase(
        name="transfer",
        mode="coast",
        spacecraft=spacecraft,
        dynamics=dynamics,
        initial_state=initial_state,
        final_state=terminal_guess,
        tof_bounds_s=(1_200.0, 24_000.0),
        constraints=[
            constraints.state(initial_state, where="Front"),
            constraints.semi_major_axis(target_a_m, where="Back", tol_m=2.0e3),
            constraints.eccentricity(target_e, where="Back", tol=5.0e-3),
            constraints.inclination_deg(target_inc_deg, where="Back", tol_deg=0.2),
        ],
        variables=phase_variables,
    )
    return Mission(
        name="Composable: terminal orbital-element constraints",
        phases=[transfer],
        objectives=[objectives.minimize_total_delta_v()],
        solver_options=DEFAULT_OPTS,
    )


def test_example_01_quick_and_composable_match() -> None:
    quick = _solve_ok(_quick01_mission())
    composable = _solve_ok(_composable01_mission())
    _assert_results_close(quick, composable, tf_rtol=3e-2, dv_rtol=3e-2, tf_atol_s=20.0, dv_atol_mps=2.0)


def test_example_01_matches_hohmann_reference_solution() -> None:
    expected_dv_mps, expected_tof_s = _hohmann_reference()
    quick = _solve_ok(_quick01_mission())

    assert quick.tf_s() == pytest.approx(expected_tof_s, rel=2.0e-2, abs=30.0)
    assert quick.total_dv_mps() == pytest.approx(expected_dv_mps, rel=2.0e-2, abs=10.0)


def test_example_02_quick_and_equivalent_composable_match() -> None:
    quick = _solve_ok(_quick02_mission())
    composable = _solve_ok(_composable_quick02_equivalent())
    _assert_results_close(quick, composable, tf_rtol=5e-2, dv_rtol=5e-2, tf_atol_s=60.0, dv_atol_mps=5.0)


def test_example_03_time_tradeoff_matches_composable_formulation() -> None:
    x0 = _circular_state(HOHMANN_R0_M)
    xf = _circular_state(LINK_R_FINAL_M, true_anomaly_rad=np.deg2rad(120.0))
    quick_dv = _solve_ok(_quick03_mission(0.0))
    comp_dv = _solve_ok(_composable_single_phase_mission(x0, xf, tf_bounds_s=(600.0, 20_000.0), w_time=0.0))
    _assert_results_close(quick_dv, comp_dv, tf_rtol=5e-2, dv_rtol=5e-2, tf_atol_s=40.0, dv_atol_mps=5.0)

    quick_time = _solve_ok(_quick03_mission(2.0))
    comp_time = _solve_ok(_composable_single_phase_mission(x0, xf, tf_bounds_s=(600.0, 20_000.0), w_time=2.0))
    _assert_results_close(quick_time, comp_time, tf_rtol=5e-2, dv_rtol=5e-2, tf_atol_s=40.0, dv_atol_mps=5.0)
    assert quick_time.tf_s() <= quick_dv.tf_s() + 1.0


def test_example_02_continuous_split_matches_single_phase_equivalent() -> None:
    split = _solve_ok(_composable02_mission())
    single = _solve_ok(_composable02_single_phase_equivalent())
    _assert_results_close(split, single, tf_rtol=5e-2, dv_rtol=5e-2, tf_atol_s=60.0, dv_atol_mps=5.0)


def test_examples_03_and_07_match() -> None:
    ex03 = _solve_ok(_composable03_mission())
    ex07 = _solve_ok(_composable07_mission())
    _assert_results_close(ex03, ex07, tf_rtol=3e-2, dv_rtol=3e-2, tf_atol_s=30.0, dv_atol_mps=3.0)



def test_example_05_respects_path_constraint_and_maneuvers() -> None:
    mission, r_min_m = _composable05_mission()
    res = _solve_ok(mission)
    radii = np.linalg.norm(res.traj[:, 0:3], axis=1)
    assert np.min(radii) >= r_min_m - 5.0
    assert len(res.maneuvers) == 2


def test_example_06_has_three_maneuvers() -> None:
    res = _solve_ok(_composable06_mission())
    assert len(res.maneuvers) == 3


def test_example_09_two_impulse_is_no_worse_than_one_impulse() -> None:
    one_impulse = _solve_ok(_composable09_mission(use_terminal_burn=False))
    two_impulse = _solve_ok(_composable09_mission(use_terminal_burn=True))

    assert all(row["satisfied"] for row in one_impulse.info["constraint_report"])
    assert all(row["satisfied"] for row in two_impulse.info["constraint_report"])
    assert two_impulse.total_dv_mps() <= one_impulse.total_dv_mps() + 1e-3


# def test_example_08_matches_direct_rendezvous_spec_solve() -> None:
#     mission = _composable08_mission()
#     direct = solve_rendezvous(_mission_to_rendezvous_spec(mission), options=DEFAULT_OPTS)
#     via_runner = _solve_ok(mission)
#     assert direct.converged
#     _assert_results_close(via_runner, direct, tf_rtol=2e-2, dv_rtol=2e-2, tf_atol_s=20.0, dv_atol_mps=2.0)
