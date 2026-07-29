from __future__ import annotations

import numpy as np
import pytest

from octavian import (
    EARTH,
    Cannonball,
    Dynamics,
    ExponentialAtmosphere,
    Perturbations,
    Phase,
    Spacecraft,
    Thruster,
    state,
)
from octavian._asset import vf
from octavian.dynamics import (
    SUN_MU_M3PS2,
    PerturbedECI,
    ThirdBodyTable,
)
from octavian.forces import (
    ASTRONOMICAL_UNIT_M,
    SOLAR_PRESSURE_AT_1_AU_NPM2,
    cannonball_drag_acceleration,
    cannonball_srp_acceleration,
)
from octavian.relative.dynamics import (
    CoupledRelativeMassCoastODE,
    FiniteThrustRelativeODE,
)
from octavian.solvers.compiler.phase_compiler import ode_for_phase
from octavian.solvers.third_bodies import (
    phase_ephemeris_body_names,
    phase_third_body_names,
)


def _sun_table() -> ThirdBodyTable:
    """Return a small constant Sun table suitable for ASSET build tests."""
    times = np.linspace(0.0, 100.0, 5)
    positions = np.repeat(
        np.asarray([[ASTRONOMICAL_UNIT_M, 0.0, 0.0]]),
        times.size,
        axis=0,
    )
    return ThirdBodyTable(
        name="sun",
        mu_m3ps2=SUN_MU_M3PS2,
        position_table=vf.InterpTable1D(
            times,
            positions,
            axis=0,
            kind="cubic",
        ),
        times_s=times,
        positions_eci_m=positions,
    )


def test_exponential_atmosphere_matches_reference_density_and_scale_height() -> None:
    atmosphere = ExponentialAtmosphere(
        reference_density_kgpm3=1.0e-12,
        reference_altitude_m=400_000.0,
        scale_height_m=50_000.0,
        rotation_rate_radps=0.0,
    )

    assert atmosphere.density_kgpm3(400_000.0) == pytest.approx(1.0e-12)
    assert atmosphere.density_kgpm3(450_000.0) == pytest.approx(1.0e-12 / np.e)


def test_drag_uses_velocity_relative_to_the_corotating_atmosphere() -> None:
    body_radius_m = 6_000_000.0
    altitude_m = 400_000.0
    rotation_rate_radps = 1.0e-3
    atmosphere = ExponentialAtmosphere(
        reference_density_kgpm3=2.0e-12,
        reference_altitude_m=altitude_m,
        scale_height_m=60_000.0,
        rotation_rate_radps=rotation_rate_radps,
    )
    position = np.asarray([body_radius_m + altitude_m, 0.0, 0.0])
    atmospheric_speed = rotation_rate_radps * position[0]
    relative_speed = 100.0
    velocity = np.asarray([0.0, atmospheric_speed + relative_speed, 0.0])
    properties = Cannonball(drag_area_m2=4.0, drag_coefficient=2.0)

    acceleration = cannonball_drag_acceleration(
        position,
        velocity,
        mass_kg=200.0,
        central_body_radius_m=body_radius_m,
        cannonball=properties,
        atmosphere=atmosphere,
    )

    expected_y = -0.5 * 2.0e-12 * 2.0 * 4.0 / 200.0 * relative_speed**2
    np.testing.assert_allclose(acceleration, [0.0, expected_y, 0.0])


def test_srp_points_away_from_the_sun_and_scales_at_one_au() -> None:
    properties = Cannonball(
        srp_area_m2=10.0,
        reflectivity_coefficient=1.5,
    )

    acceleration = cannonball_srp_acceleration(
        [0.0, 0.0, 0.0],
        [-ASTRONOMICAL_UNIT_M, 0.0, 0.0],
        mass_kg=500.0,
        cannonball=properties,
    )

    expected = SOLAR_PRESSURE_AT_1_AU_NPM2 * 1.5 * 10.0 / 500.0
    np.testing.assert_allclose(acceleration, [expected, 0.0, 0.0])


def test_srp_ephemeris_does_not_implicitly_enable_solar_gravity() -> None:
    spacecraft = Spacecraft(
        name="probe",
        dry_mass_kg=100.0,
        cannonball=Cannonball(srp_area_m2=2.0),
    )
    phase = Phase(
        name="srp_coast",
        mode="coast",
        spacecraft=spacecraft,
        dynamics=Dynamics(perturbations=Perturbations(srp=True)),
    )

    assert phase_third_body_names(phase) == ()
    assert phase_ephemeris_body_names(phase) == ("sun",)
    assert isinstance(ode_for_phase(phase, sun_table=_sun_table()), PerturbedECI)


def test_drag_phase_requires_explicit_spacecraft_area() -> None:
    phase = Phase(
        name="drag_coast",
        mode="coast",
        spacecraft=Spacecraft(name="probe", dry_mass_kg=100.0),
        dynamics=Dynamics(perturbations=Perturbations(drag=True)),
    )

    with pytest.raises(ValueError, match=r"drag_area_m2 > 0"):
        ode_for_phase(phase)


def test_relative_finite_burn_and_mass_coast_share_cannonball_forces() -> None:
    radius_m = EARTH.mean_radius_m + 400_000.0
    chief_state = state(
        [radius_m, 0.0, 0.0],
        [0.0, np.sqrt(EARTH.mu_m3ps2 / radius_m), 0.0],
    )
    chief = Spacecraft(
        name="chief",
        dry_mass_kg=200.0,
        cannonball=Cannonball(drag_area_m2=2.0),
    )
    deputy = Spacecraft(
        name="deputy",
        dry_mass_kg=100.0,
        thrusters=[
            Thruster(
                name="main",
                thrust_N=1.0,
                isp_s=300.0,
                propellant_mass_kg=10.0,
            )
        ],
        cannonball=Cannonball(drag_area_m2=3.0),
    )
    dynamics = Dynamics.relative(
        chief_initial_state_eci=chief_state,
        chief_spacecraft=chief,
        perturbations=Perturbations(drag=True),
    )
    burn = Phase(
        name="burn",
        mode="finite_thrust",
        spacecraft=deputy,
        dynamics=dynamics,
    )
    coast = Phase(
        name="coast",
        mode="coast",
        spacecraft=deputy,
        dynamics=dynamics,
    )

    assert isinstance(ode_for_phase(burn), FiniteThrustRelativeODE)
    assert isinstance(
        ode_for_phase(coast, carries_mass=True),
        CoupledRelativeMassCoastODE,
    )


@pytest.mark.parametrize(
    "properties",
    [
        {"drag_area_m2": -1.0},
        {"drag_coefficient": 0.0},
        {"srp_area_m2": float("nan")},
        {"reflectivity_coefficient": -1.0},
    ],
)
def test_cannonball_rejects_nonphysical_properties(
    properties: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        Cannonball(**properties)
