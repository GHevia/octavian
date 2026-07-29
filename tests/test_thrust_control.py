from __future__ import annotations

import math

import numpy as np
import pytest

from octavian import (
    EARTH,
    Dynamics,
    Mission,
    Phase,
    Spacecraft,
    ThrustControl,
    Thruster,
    constraints,
    objectives,
    state,
)
from octavian.control import euler_thrust_direction
from octavian.solvers import SolverOptions
from octavian.solvers.compiler.phase_compiler import (
    layout_for_phase,
    prepare_phase_guess,
    validate_powered_phase_chain,
)


def _spacecraft() -> Spacecraft:
    return Spacecraft(
        name="vehicle",
        dry_mass_kg=100.0,
        thrusters=[
            Thruster(
                name="main",
                thrust_N=10.0,
                isp_s=300.0,
                propellant_mass_kg=20.0,
            )
        ],
    )


def _inertial_guess() -> list[np.ndarray]:
    return [
        np.asarray([7.0e6, 0.0, 0.0, 0.0, 7_500.0, 0.0, 0.0]),
        np.asarray([7.0e6, 75_000.0, 0.0, -80.0, 7_499.0, 0.0, 10.0]),
    ]


def test_thrust_control_normalizes_frames_and_fixed_direction() -> None:
    vector = ThrustControl.vector(frame="LVLH")
    fixed = ThrustControl.fixed([10.0, 0.0, 0.0], frame="ECI")

    assert vector.frame == "ric"
    assert fixed.frame == "inertial"
    assert fixed.direction == pytest.approx((1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="non-zero"):
        ThrustControl.fixed([0.0, 0.0, 0.0])


def test_euler_direction_uses_yaw_pitch_and_preserves_roll_state() -> None:
    direction = euler_thrust_direction([0.5 * math.pi, 0.0, math.radians(45.0)])
    np.testing.assert_allclose(direction, [0.0, 1.0, 0.0], atol=1.0e-15)

    pitched = euler_thrust_direction([0.0, 0.5 * math.pi, 0.0])
    np.testing.assert_allclose(pitched, [0.0, 0.0, -1.0], atol=1.0e-15)


@pytest.mark.parametrize(
    ("control", "layout_name", "row_width"),
    [
        (ThrustControl.vector(frame="ric"), "cartesian_mass_thrust", 11),
        (
            ThrustControl.fixed([1.0, 0.0, 0.0], frame="inertial"),
            "cartesian_mass_fixed_thrust",
            9,
        ),
        (
            ThrustControl.euler(
                frame="ric",
                initial_angles_rad=(0.1, 0.2, 0.3),
                max_slew_rate_radps=0.01,
            ),
            "cartesian_mass_euler_thrust",
            15,
        ),
    ],
)
def test_powered_guess_matches_selected_control_layout(
    control: ThrustControl,
    layout_name: str,
    row_width: int,
) -> None:
    phase = Phase(
        name="burn",
        mode="finite_thrust",
        spacecraft=_spacecraft(),
        dynamics=Dynamics(),
        thrust_control=control,
    )

    prepared, layout, kind = prepare_phase_guess(phase, _inertial_guess())
    trajectory = np.asarray(prepared)

    assert layout.name == layout_name
    assert kind == "finite_thrust"
    assert trajectory.shape == (2, row_width)
    np.testing.assert_allclose(trajectory[:, layout.time_column], [0.0, 10.0])
    if control.carries_attitude:
        np.testing.assert_allclose(
            trajectory[:, list(layout.state_indices("attitude"))],
            [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]],
        )


def test_euler_attitude_is_carried_through_intermediate_coast() -> None:
    vehicle = _spacecraft()
    attitude = ThrustControl.euler(
        frame="ric",
        initial_angles_rad=(0.0, 0.0, 0.0),
        max_slew_rate_radps=0.02,
    )
    departure = Phase(
        name="departure",
        mode="finite_thrust",
        spacecraft=vehicle,
        dynamics=Dynamics(),
        thrust_control=attitude,
    )
    coast = Phase(
        name="coast",
        mode="coast",
        spacecraft=vehicle,
        dynamics=Dynamics(),
        previous=departure,
    )
    arrival = Phase(
        name="arrival",
        mode="finite_thrust",
        spacecraft=vehicle,
        dynamics=Dynamics(),
        previous=coast,
    )
    for phase in (departure, coast, arrival):
        phase.inherit_defaults()

    validate_powered_phase_chain([departure, coast, arrival])
    assert layout_for_phase(departure).name == "cartesian_mass_euler_thrust"
    assert layout_for_phase(coast, carries_mass=True).name == "cartesian_mass_euler_coast"
    assert layout_for_phase(arrival).name == "cartesian_mass_euler_thrust"

    coast_guess, coast_layout, _ = prepare_phase_guess(
        coast,
        _inertial_guess(),
        carries_mass=True,
    )
    assert np.asarray(coast_guess).shape == (2, 14)
    assert coast_layout.control_indices("attitude_rate") == (0, 1, 2)


def test_euler_chain_rejects_missing_attitude_segment() -> None:
    vehicle = _spacecraft()
    departure = Phase(
        name="departure",
        mode="finite_thrust",
        spacecraft=vehicle,
        dynamics=Dynamics(),
        thrust_control=ThrustControl.euler(max_slew_rate_radps=0.01),
    )
    coast = Phase(
        name="coast",
        mode="coast",
        spacecraft=vehicle,
        dynamics=Dynamics(),
        previous=departure,
        thrust_control=ThrustControl.vector(),
    )
    arrival = Phase(
        name="arrival",
        mode="finite_thrust",
        spacecraft=vehicle,
        dynamics=Dynamics(),
        previous=coast,
        thrust_control=ThrustControl.euler(max_slew_rate_radps=0.01),
    )

    with pytest.raises(ValueError, match="every burn and intermediate coast"):
        validate_powered_phase_chain([departure, coast, arrival])


def test_relative_fixed_and_euler_controls_use_coupled_layouts() -> None:
    radius = EARTH.mean_radius_m + 400_000.0
    chief = state(
        [radius, 0.0, 0.0],
        [0.0, np.sqrt(EARTH.mu_m3ps2 / radius), 0.0],
    )
    fixed = Phase(
        name="relative_fixed",
        mode="finite_thrust",
        spacecraft=_spacecraft(),
        dynamics=Dynamics.relative(chief_initial_state_eci=chief),
        thrust_control=ThrustControl.fixed([0.0, 1.0, 0.0], frame="ric"),
    )
    euler = Phase(
        name="relative_euler",
        mode="finite_thrust",
        spacecraft=_spacecraft(),
        dynamics=Dynamics.relative(chief_initial_state_eci=chief),
        thrust_control=ThrustControl.euler(frame="ric", max_slew_rate_radps=0.01),
    )

    assert layout_for_phase(fixed).name == "coupled_relative_eci_mass_fixed_thrust"
    assert layout_for_phase(euler).name == "coupled_relative_eci_mass_euler_thrust"


@pytest.mark.parametrize(
    ("control", "expected_layout"),
    [
        (
            ThrustControl.vector(frame="ric"),
            "cartesian_mass_thrust",
        ),
        (
            ThrustControl.fixed([0.0, 1.0, 0.0], frame="ric"),
            "cartesian_mass_fixed_thrust",
        ),
        (
            ThrustControl.euler(
                frame="ric",
                initial_angles_rad=(0.5 * math.pi, 0.0, 0.25),
                max_slew_rate_radps=0.01,
            ),
            "cartesian_mass_euler_thrust",
        ),
    ],
)
def test_control_representation_compiles_and_solves(
    control: ThrustControl,
    expected_layout: str,
) -> None:
    radius = 7_000_000.0
    speed = np.sqrt(EARTH.mu_m3ps2 / radius)
    transfer_angle = math.pi / 3.0
    transfer_time = transfer_angle / np.sqrt(EARTH.mu_m3ps2 / radius**3)
    initial = state([radius, 0.0, 0.0], [0.0, speed, 0.0])
    final = state(
        [radius * math.cos(transfer_angle), radius * math.sin(transfer_angle), 0.0],
        [
            -(speed + 0.5) * math.sin(transfer_angle),
            (speed + 0.5) * math.cos(transfer_angle),
            0.0,
        ],
    )
    phase = Phase(
        name="zero_throttle_orbit_arc",
        mode="finite_thrust",
        spacecraft=_spacecraft(),
        dynamics=Dynamics.for_body(EARTH),
        initial_state=initial,
        final_state=final,
        tof_bounds_s=(transfer_time - 1.0, transfer_time + 1.0),
        constraints=[
            constraints.state(initial, where="Front"),
            constraints.state(final, where="Back"),
        ],
        thrust_control=control,
    )
    solution = Mission(
        phases=[phase],
        objectives=[objectives.minimize_propellant()],
        mesh_nsegs_transfer=12,
        lambert_grid_size=8,
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
    assert solution.result.info["state_layouts"] == [expected_layout]
    assert solution.result.info["thrust_controls"][0]["frame"] == "ric"
    assert len(solution.phase_control_trajectories) == 1
    if control.carries_attitude:
        assert len(solution.attitude_phase_trajectories) == 1
        attitude = solution.attitude_phase_trajectories[0]
        np.testing.assert_allclose(
            attitude[0, 0:3],
            control.initial_angles_rad,
            atol=1.0e-10,
        )
        assert np.max(np.linalg.norm(attitude[:, 4:7], axis=1)) <= (
            control.max_slew_rate_radps + 1.0e-10
        )
