from __future__ import annotations

import numpy as np
import pytest

import octavian.relative.elements as relative_elements_module
from octavian import (
    EARTH,
    Perturbations,
    RelativeElementPropagationResult,
    propagate,
    state,
)
from octavian.astro import classical_to_cartesian
from octavian.relative import RelativeOrbitalElements, RelativePropagationResult


def _chief_state():
    """Return a nonsingular inclined chief state for propagation tests."""
    position, velocity = classical_to_cartesian(
        a_m=EARTH.mean_radius_m + 500_000.0,
        e=0.001,
        inc_deg=40.0,
        raan_deg=20.0,
        argp_deg=10.0,
        true_anomaly_deg=30.0,
        mu_m3ps2=EARTH.mu_m3ps2,
    )
    return state(position, velocity)


def test_propagate_namespace_returns_consistent_state_histories() -> None:
    chief = _chief_state()
    times = np.asarray([0.0, 30.0, 60.0])
    mean_motion = np.sqrt(EARTH.mu_m3ps2 / (EARTH.mean_radius_m + 500_000.0) ** 3)
    relative_initial = np.asarray([100.0, -200.0, 50.0, 0.0, 0.01, 0.0])

    two_body = propagate.two_body(
        chief,
        times,
        mu_m3ps2=EARTH.mu_m3ps2,
    )
    cwh = propagate.cwh(
        relative_initial,
        times,
        mean_motion_radps=mean_motion,
    )
    nonlinear = propagate.nonlinear_ric(
        relative_initial,
        times,
        mu_m3ps2=EARTH.mu_m3ps2,
        chief_orbit_radius_m=EARTH.mean_radius_m + 500_000.0,
        max_step_s=2.0,
    )

    assert two_body.shape == cwh.shape == nonlinear.shape == (3, 7)
    np.testing.assert_allclose(two_body[0, 0:3], chief.r_m, atol=1.0e-8)
    np.testing.assert_allclose(cwh[0, 0:6], relative_initial)
    np.testing.assert_allclose(nonlinear[0, 0:6], relative_initial)
    np.testing.assert_allclose(two_body[:, 6], times)


def test_propagate_namespace_couples_absolute_relative_states() -> None:
    chief = _chief_state()
    relative_initial = state([100.0, -200.0, 50.0], [0.0, 0.01, 0.0])

    result = propagate.relative(
        chief,
        relative_initial,
        [0.0, 60.0],
        max_step_s=2.0,
    )

    assert isinstance(result, RelativePropagationResult)
    assert result.relative_trajectory_ric.shape == (2, 7)
    assert result.chief_trajectory_eci.shape == (2, 7)
    assert result.deputy_trajectory_eci.shape == (2, 7)


def test_propagate_relative_elements_returns_native_and_ric_views() -> None:
    chief = _chief_state()
    initial = RelativeOrbitalElements(
        delta_a=1.0e-4,
        delta_lambda_rad=-0.002,
        delta_ex=1.0e-4,
        delta_ey=-2.0e-4,
        delta_ix_rad=3.0e-4,
        delta_iy_rad=-4.0e-4,
    )

    result = propagate.relative_elements(
        initial,
        [0.0, 300.0],
        chief_initial_state_eci=chief,
        mu_m3ps2=EARTH.mu_m3ps2,
    )

    assert isinstance(result, RelativeElementPropagationResult)
    assert result.representation == "damico"
    assert result.elements.shape == result.ric.shape == (2, 7)
    np.testing.assert_allclose(result.elements[0, 0:6], initial.as_vector())
    np.testing.assert_allclose(result.times_s, [0.0, 300.0])


def test_perturbed_relative_element_facade_propagates_only_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chief = _chief_state()
    initial = RelativeOrbitalElements(
        delta_a=1.0e-4,
        delta_lambda_rad=-0.002,
        delta_ex=1.0e-4,
        delta_ey=-2.0e-4,
        delta_ix_rad=3.0e-4,
        delta_iy_rad=-4.0e-4,
    )
    original = relative_elements_module._propagate_relative_elements_numerical
    call_count = 0

    def counted_propagation(*args, **kwargs):
        """Count coupled integrations while preserving the real calculation."""
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        relative_elements_module,
        "_propagate_relative_elements_numerical",
        counted_propagation,
    )

    result = propagate.relative_elements(
        initial,
        [0.0, 60.0],
        chief_initial_state_eci=chief,
        mu_m3ps2=EARTH.mu_m3ps2,
        perturbations=Perturbations(j2=True),
        max_step_s=2.0,
    )

    assert call_count == 1
    assert result.elements.shape == result.ric.shape == (2, 7)


def test_propagate_namespace_includes_cr3bp() -> None:
    try:
        from octavian import CR3BPSystem
    except ImportError:
        with pytest.raises(ImportError, match="cislunar"):
            propagate.cr3bp(
                state([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
                [0.0, 0.01],
                system=object(),
            )
        return

    system = CR3BPSystem.earth_moon()
    initial = state(
        system.lagrange_points(dimensional=False)["L4"],
        [0.0, 0.0, 0.0],
    )

    history = propagate.cr3bp(
        initial,
        [0.0, 0.01],
        system=system,
        dimensional=False,
        max_step=0.001,
    )

    assert history.shape == (2, 7)
    np.testing.assert_allclose(
        history[:, 0:6],
        np.tile(history[0, 0:6], (2, 1)),
        atol=1.0e-15,
    )
