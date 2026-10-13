# EGM2008, truncated to degree and order 200

Numerical gravity coefficients from the U.S. National Geospatial-Intelligence
Agency's public EGM2008 distribution:
https://earth-info.nga.mil/php/download.php?file=egm-08spherical

Source member: `EGM2008_to2190_TideFree`. Values for degrees 2–200 are retained
without numerical conversion, fitting, or rounding (Fortran D exponents are
written as E). C00=1 and zero degree-one entries are explicitly added. The
ICGEM-compatible header specifies fully normalized, tide-free coefficients,
GM=3.986004415e14 m³/s² and reference radius 6378136.3 m, verified against
NGA's `hsynth_WGS84.f` and ICGEM's EGM2008 header.

Citation: Pavlis, N. K., Holmes, S. A., Kenyon, S. C., & Factor, J. K. (2012),
The development and evaluation of the Earth Gravitational Model 2008 (EGM2008).
https://doi.org/10.1029/2011JB008916

`egm2008_200.json` records the source URL, archive SHA-256, subset SHA-256,
constants and processing. `tools/prepare_egm2008.py` reproduces the subset from
the official archive. This is NGA's numerical dataset, not Octavian-generated
coefficients; Octavian's code license does not assert ownership of the data.
No network access is performed by the runtime loader.
