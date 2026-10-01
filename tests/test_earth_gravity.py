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
