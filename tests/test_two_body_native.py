"""Numerical validation of the specialized C++ two-body integrator."""

import numpy as np
import pytest

from octavian import EARTH
from octavian._asset import ast
from octavian.dynamics import TwoBodyECI

pytestmark = pytest.mark.skipif(ast is None, reason="ASSET is required")


@pytest.mark.parametrize("velocity_scale", [1.0, 0.8, 1.6])
@pytest.mark.parametrize("direction", [1, -1])
def test_native_two_body_matches_expression_and_conserves_invariants(velocity_scale, direction):
    mu = EARTH.mu_m3ps2
    initial = np.array([7e6, 0, 0, 0, velocity_scale * np.sqrt(mu / 7e6), 0, 0])
    histories = {}
    for backend in ["cpp", "expression"]:
        ode = TwoBodyECI(mu_m3ps2=mu, backend=backend)
        integrator = ode.integrator("DOPRI87", 10.0)
        integrator.setAbsTol(1e-8)
        histories[backend] = np.asarray(integrator.integrate_dense(initial, direction * 1000.0, 101))
    native = histories["cpp"]
    assert native.shape == (101, 7)
    assert np.isfinite(native).all()
    np.testing.assert_allclose(native, histories["expression"], atol=1e-5, rtol=1e-10)
    np.testing.assert_allclose(native[0], initial, atol=1e-9)
    np.testing.assert_allclose(native[-1, 6], direction * 1000.0)
    energy = np.sum(native[:, 3:6] ** 2, axis=1) / 2 - mu / np.linalg.norm(native[:, :3], axis=1)
    np.testing.assert_allclose(energy, energy[0], rtol=1e-10)
    momentum = np.cross(native[:, :3], native[:, 3:6])
    np.testing.assert_allclose(momentum, np.broadcast_to(momentum[0], momentum.shape), rtol=1e-10, atol=1e-5)
    if velocity_scale == 1:
        angle = direction * 1000 * np.sqrt(mu / 7e6**3)
        np.testing.assert_allclose(native[-1, :3], 7e6 * np.array([np.cos(angle), np.sin(angle), 0]), atol=1e-5)


def test_native_two_body_is_default_and_preserves_phase_groups():
    ode = TwoBodyECI(mu_m3ps2=EARTH.mu_m3ps2)
    assert isinstance(ode.integrator(10.0), ast.Astro.Kepler.integrator)
    # Phase creation still accepts the established position and velocity groups.
    phase = ode.phase("LGL3")
    phase.addBoundaryValue("Front", "R", [7e6, 0, 0])
    phase.addBoundaryValue("Front", "V", [0, 7500, 0])


@pytest.mark.parametrize("mu", [0, -1, np.inf, np.nan])
def test_two_body_rejects_invalid_mu(mu):
    with pytest.raises(ValueError, match="finite and positive"):
        TwoBodyECI(mu_m3ps2=mu)


def test_two_body_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        TwoBodyECI(mu_m3ps2=EARTH.mu_m3ps2, backend="unknown")
