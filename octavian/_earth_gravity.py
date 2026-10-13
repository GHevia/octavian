"""Read the packaged, checksum-verified NGA EGM2008 coefficient subset."""

import gzip
import hashlib
import io
import json
from functools import lru_cache
from importlib.resources import files

import numpy as np


@lru_cache(maxsize=1)
def _egm2008():
    resources = files("octavian").joinpath("data/gravity")
    metadata = json.loads(resources.joinpath("egm2008_200.json").read_text())
    payload = gzip.decompress(resources.joinpath("egm2008_200.gfc.gz").read_bytes())
    if hashlib.sha256(payload).hexdigest() != metadata["uncompressed_subset_sha256"]:
        raise RuntimeError("Bundled EGM2008 coefficient checksum mismatch; reinstall Octavian")
    records = payload.decode("ascii").split("end_of_head\n", 1)[1]
    values = np.loadtxt(io.StringIO(records), usecols=(1, 2, 3, 4))
    degree = metadata["maximum_degree"]
    c, s = np.zeros((degree + 1, degree + 1)), np.zeros((degree + 1, degree + 1))
    indices = values[:, :2].astype(int)
    if {tuple(row) for row in indices} != {
        (n, m) for n in range(degree + 1) for m in range(n + 1)
    } or len(values) != (degree + 1) * (degree + 2) // 2:
        raise RuntimeError("Incomplete or duplicate bundled EGM2008 coefficients")
    c[indices[:, 0], indices[:, 1]] = values[:, 2]
    s[indices[:, 0], indices[:, 1]] = values[:, 3]
    c.setflags(write=False)
    s.setflags(write=False)
    return c, s, metadata


def earth_coefficients(degree, order):
    if (
        isinstance(degree, bool)
        or not isinstance(degree, (int, np.integer))
        or not 2 <= degree <= 200
    ):
        raise ValueError("Bundled EGM2008 degree must be an integer from 2 through 200")
    order = degree if order is None else order
    if (
        isinstance(order, bool)
        or not isinstance(order, (int, np.integer))
        or not 0 <= order <= degree
    ):
        raise ValueError("order must be an integer from 0 through degree")
    c, s, metadata = _egm2008()
    return c[: degree + 1, : degree + 1], s[: degree + 1, : degree + 1], metadata
