"""Published data integrity, truncation, model defaults, and config routing."""

import gzip
import hashlib
import json
from importlib.resources import files

import numpy as np
import pytest

from octavian import SphericalHarmonics
from octavian.config.environment import build_perturbations


def test_published_earth_defaults_and_anchor_coefficients():
    gravity = SphericalHarmonics.earth()
    assert gravity.backend == "cpp"
    assert gravity.degree == gravity.order == 200
    assert gravity.model_name == "EGM2008"
    assert gravity.tide_system == "tide_free"
    assert gravity.reference_mu_m3ps2 == 3.986004415e14
    assert gravity.reference_radius_m == 6378136.3
    assert gravity.cosine[2][0] == -0.484165143790815e-3
    assert gravity.cosine[2][2] == 0.243938357328313e-5
    assert gravity.sine[2][2] == -0.140027370385934e-5
    assert gravity.cosine[200][200] != 0.0
    assert gravity.sine[200][200] != 0.0
    assert len(gravity.cosine) == 201


def test_bundled_data_complete_and_checksum_matches():
    resources = files("octavian").joinpath("data/gravity")
    metadata = json.loads(resources.joinpath("egm2008_200.json").read_text())
    payload = gzip.decompress(resources.joinpath("egm2008_200.gfc.gz").read_bytes())
    assert hashlib.sha256(payload).hexdigest() == metadata["uncompressed_subset_sha256"]
    assert payload.count(b"gfc ") == 201 * 202 // 2
    assert metadata["source_url"].startswith("https://earth-info.nga.mil/")


def test_truncation_and_python_remain_explicit():
    full = SphericalHarmonics.earth()
    small = SphericalHarmonics.earth(degree=4, order=2, backend="python")
    np.testing.assert_array_equal(small.cosine, np.asarray(full.cosine)[:5, :5])
    assert small.order == 2
    assert small.backend == "python"
    assert np.isfinite(small.acceleration([7e6, 1e6, 2e6], mu_m3ps2=small.reference_mu_m3ps2)).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(degree=201),
        dict(degree=1),
        dict(degree=4.5),
        dict(degree=True),
        dict(order=201),
        dict(order=-1),
        dict(order=3.5),
        dict(order=True),
    ],
)
def test_unsupported_degree_order_rejected(kwargs):
    with pytest.raises(ValueError):
        SphericalHarmonics.earth(**kwargs)


def test_earth_config_and_no_mixed_custom_coefficients():
    config = {"spherical_harmonics": {"model": "EGM2008", "degree": 4, "backend": "python"}}
    gravity = build_perturbations(config, "perturbations").spherical_harmonics
    assert gravity == SphericalHarmonics.earth(degree=4, backend="python")
    for bad in ({"model": "unknown"}, {"model": "EGM2008", "cosine": [[1.0]]}):
        with pytest.raises(ValueError):
            build_perturbations({"spherical_harmonics": bad}, "perturbations")


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_published_mu_applies_to_mission_numerical_and_asset_gravity(backend):
    from octavian import EARTH, Dynamics, Perturbations, propagate, state
    from octavian.dynamics import PerturbedECI, gravity_acceleration_components

    if backend == "cpp":
        import importlib.util

        if not any(
            importlib.util.find_spec(name)
            for name in ("octavian.octavian_harmonics_native", "octavian_harmonics_native")
        ):
            pytest.skip("native backend is not installed")
    gravity = SphericalHarmonics.earth(degree=2, order=0, backend=backend)
    forces = Perturbations(spherical_harmonics=gravity)
    dynamics = Dynamics.for_body(EARTH, perturbations=forces)
    assert dynamics.mu_m3ps2 == gravity.reference_mu_m3ps2
    assert Dynamics(perturbations=forces).mu_m3ps2 == gravity.reference_mu_m3ps2
    position = np.array([7e6, 1e6, 2e6])
    expected = -gravity.reference_mu_m3ps2 * position / np.linalg.norm(
        position
    ) ** 3 + gravity.acceleration(position)
    # Even a different central-body GM cannot split a published gravity model.
    actual = gravity_acceleration_components(position, mu_m3ps2=1e14, spherical_harmonics=gravity)
    np.testing.assert_allclose(actual, expected, rtol=2e-15)
    ode = PerturbedECI(spherical_harmonics=gravity)
    assert ode.mu == gravity.reference_mu_m3ps2
    initial = state(position, [0, 7000, 200])
    from dataclasses import replace

    times = [0.0, 10.0]
    automatic = propagate.inertial(initial, times, perturbations=forces, max_step_s=1)
    explicit = propagate.inertial(
        initial,
        times,
        perturbations=forces,
        central_body=replace(EARTH, mu_m3ps2=gravity.reference_mu_m3ps2),
        max_step_s=1,
    )
    np.testing.assert_array_equal(automatic, explicit)
    # Coupled chief/deputy propagation must resolve the same GM as inertial propagation.
    offset = state([100, 0, 0], [0, 0, 0])
    relative = propagate.relative(initial, offset, times, perturbations=forces, max_step_s=1)
    np.testing.assert_allclose(
        relative.chief_states_eci[:, :6], automatic[:, :6], rtol=0, atol=1e-8
    )

    integrator = ode.integrator(1.0)
    integrator.setAbsTol(1e-7)
    asset = np.asarray(integrator.integrate(np.r_[position, initial.v_mps, 0], 10.0))
    np.testing.assert_allclose(asset[:6], automatic[-1, :6], rtol=0, atol=1e-5)


def test_custom_field_requires_mu_unless_it_has_model_metadata():
    from dataclasses import replace

    gravity = replace(SphericalHarmonics.earth(degree=2, backend="python"), reference_mu_m3ps2=None)
    with pytest.raises(ValueError, match="requires a positive mu_m3ps2"):
        gravity.acceleration([7e6, 0, 0])
    a = gravity.acceleration([7e6, 0, 0], mu_m3ps2=1e14)
    b = gravity.acceleration([7e6, 0, 0], mu_m3ps2=2e14)
    np.testing.assert_allclose(b, 2 * a)
