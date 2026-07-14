from __future__ import annotations

import pytest

from octavian import EARTH, MOON, SUN, Dynamics, state, two_burn_rendezvous
from octavian.bodies import CATALOG, CelestialBody, resolve
from octavian.runner import _mission_to_rendezvous_spec


def test_catalog_resolves_names_and_aliases() -> None:
    assert resolve("Earth") is EARTH
    assert resolve("luna") is MOON
    assert resolve("SOL") is SUN
    assert CATALOG["terra"] is EARTH

    with pytest.raises(KeyError, match="Available bodies"):
        resolve("ceres")


def test_custom_body_validates_physical_constants() -> None:
    body = CelestialBody(name="Example", mu_m3ps2=1.0e12, mean_radius_m=1.0e6)
    assert body.name == "example"
    assert resolve(body) is body

    with pytest.raises(ValueError, match="positive"):
        CelestialBody(name="bad", mu_m3ps2=0.0, mean_radius_m=1.0)


def test_body_dynamics_use_consistent_constants_and_frame() -> None:
    dynamics = Dynamics.for_body(SUN, mu_m3ps2=1.0, central_body_radius_m=2.0)

    assert dynamics.central_body is SUN
    assert dynamics.mu_m3ps2 == SUN.mu_m3ps2
    assert dynamics.central_body_radius_m == SUN.mean_radius_m
    assert dynamics.frame.origin == "sun"
    assert dynamics.frame.kind == "inertial"


def test_quick_builder_accepts_central_body_name() -> None:
    x0 = state([1.0e9, 0.0, 0.0], [0.0, 1.0e3, 0.0])
    xf = state([-2.0e9, 0.0, 0.0], [0.0, -800.0, 0.0])

    mission = two_burn_rendezvous(x0, xf, central_body="sun")
    dynamics = mission.phases[0].dynamics

    assert dynamics.central_body is SUN
    assert dynamics.mu_m3ps2 == SUN.mu_m3ps2
    assert dynamics.frame == SUN.inertial_frame()

    spec = _mission_to_rendezvous_spec(mission)
    assert spec.central_body_name == "sun"
    assert spec.frame.origin == "sun"
