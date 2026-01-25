from __future__ import annotations

import numpy as np

from octavian.astro.kepler import estimate_orbital_period_s


def test_estimate_orbital_period_s():
    mu = 3.986004418e14
    r0 = np.array([7000e3, 0.0, 0.0])
    # circular orbit speed
    v0 = np.array([0.0, np.sqrt(mu / np.linalg.norm(r0)), 0.0])

    T = estimate_orbital_period_s(r0, v0, mu)
    assert T is not None
    # ~ 5828 seconds for 7000 km circular around Earth
    assert 5000.0 < float(T) < 7000.0
