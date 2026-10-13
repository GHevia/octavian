"""Reproduce the bundled EGM2008 subset from NGA's downloaded archive.

Download the archive from the source URL below, save it as EGM2008.zip in the
working directory, and run this script. This is a maintainer task, not an
installation step. No gravity values are fitted, normalized, or rounded here.
"""

import gzip
import hashlib
import json
import zipfile
from pathlib import Path

source_archive = Path("EGM2008.zip")
source_url = "https://earth-info.nga.mil/php/download.php?file=egm-08spherical"
maximum_degree = 200
output = Path(__file__).resolve().parents[1] / "octavian/data/gravity"
digest = hashlib.sha256()
with source_archive.open("rb") as source:
    for block in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(block)
archive_hash = digest.hexdigest()
if archive_hash != "65a9072f337f156e8cbd76ffd773f536e6fb0de18697ea6726ecdb790fac0fbd":
    raise ValueError("NGA archive checksum changed; review the source before regenerating data")
header = f"""product_type gravity_field
modelname EGM2008
earth_gravity_constant 3.986004415e14
radius 6378136.3
max_degree {maximum_degree}
errors calibrated
norm fully_normalized
tide_system tide_free
end_of_head
gfc 0 0 1.0 0.0 0.0 0.0
gfc 1 0 0.0 0.0 0.0 0.0
gfc 1 1 0.0 0.0 0.0 0.0
"""
records = []
seen = set()
with zipfile.ZipFile(source_archive) as archive, archive.open("EGM2008_to2190_TideFree") as stream:
    for raw in stream:
        fields = raw.decode("ascii").split()
        n, m = map(int, fields[:2])
        if n > maximum_degree:
            break  # NGA file is ordered by increasing degree, then order.
        if (n, m) in seen:
            raise ValueError("Duplicate coefficient")
        seen.add((n, m))
        records.append("gfc " + " ".join(fields).replace("D", "E") + "\n")
assert seen == {(n, m) for n in range(2, maximum_degree + 1) for m in range(n + 1)}
payload = (header + "".join(records)).encode("ascii")
output.mkdir(parents=True, exist_ok=True)
(output / "egm2008_200.gfc.gz").write_bytes(gzip.compress(payload, mtime=0))
metadata = {
    "model": "EGM2008",
    "maximum_degree": maximum_degree,
    "normalization": "fully_normalized",
    "tide_system": "tide_free",
    "reference_radius_m": 6378136.3,
    "mu_m3ps2": 3.986004415e14,
    "source_url": source_url,
    "source_archive_sha256": archive_hash,
    "source_member": "EGM2008_to2190_TideFree",
    "uncompressed_subset_sha256": hashlib.sha256(payload).hexdigest(),
    "citation": "Pavlis, Holmes, Kenyon & Factor (2012), The development and evaluation of the Earth Gravitational Model 2008 (EGM2008)",
    "doi": "https://doi.org/10.1029/2011JB008916",
    "processing": "Retain degrees 2 through 200 unchanged; add conventional C00=1 and zero degree-one terms; convert Fortran D exponents to E; add ICGEM header. Reference constants verified against NGA hsynth_WGS84.f and ICGEM EGM2008 header.",
}
(output / "egm2008_200.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(f"Wrote {maximum_degree}x{maximum_degree} EGM2008 to {output}")
