"""Runtime ephemeris helpers for Sun/Moon third-body perturbations."""

from __future__ import annotations

import math
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import spiceypy as spice

DATA_DIR = Path(__file__).resolve().parent
DEFAULT_EPHEMERIS_BSP = DATA_DIR / "sun_moon_scheduled.bsp"

_ECI_TOD_FRAME_KERNEL_POOL = (
    "FRAME_ECI_TOD = 1400010",
    "FRAME_1400010_NAME = 'ECI_TOD'",
    "FRAME_1400010_CLASS = 5",
    "FRAME_1400010_CLASS_ID = 1400010",
    "FRAME_1400010_CENTER = 399",
    "FRAME_1400010_RELATIVE = 'J2000'",
    "FRAME_1400010_DEF_STYLE = 'PARAMETERIZED'",
    "FRAME_1400010_FAMILY = 'TRUE_EQUATOR_AND_EQUINOX_OF_DATE'",
    "FRAME_1400010_PREC_MODEL = 'EARTH_IAU_1976'",
    "FRAME_1400010_NUT_MODEL = 'EARTH_IAU_1980'",
    "FRAME_1400010_ROTATION_STATE = 'ROTATING'",
)

UTC = timezone.utc
_J2000_UTC = datetime(2000, 1, 1, 12, 0, 0, tzinfo=UTC)
_LEAP_SECOND_EFFECTIVE_DATES = (
    datetime(1972, 1, 1, tzinfo=UTC),
    datetime(1972, 7, 1, tzinfo=UTC),
    datetime(1973, 1, 1, tzinfo=UTC),
    datetime(1974, 1, 1, tzinfo=UTC),
    datetime(1975, 1, 1, tzinfo=UTC),
    datetime(1976, 1, 1, tzinfo=UTC),
    datetime(1977, 1, 1, tzinfo=UTC),
    datetime(1978, 1, 1, tzinfo=UTC),
    datetime(1979, 1, 1, tzinfo=UTC),
    datetime(1980, 1, 1, tzinfo=UTC),
    datetime(1981, 7, 1, tzinfo=UTC),
    datetime(1982, 7, 1, tzinfo=UTC),
    datetime(1983, 7, 1, tzinfo=UTC),
    datetime(1985, 7, 1, tzinfo=UTC),
    datetime(1988, 1, 1, tzinfo=UTC),
    datetime(1990, 1, 1, tzinfo=UTC),
    datetime(1991, 1, 1, tzinfo=UTC),
    datetime(1992, 7, 1, tzinfo=UTC),
    datetime(1993, 7, 1, tzinfo=UTC),
    datetime(1994, 7, 1, tzinfo=UTC),
    datetime(1996, 1, 1, tzinfo=UTC),
    datetime(1997, 7, 1, tzinfo=UTC),
    datetime(1999, 1, 1, tzinfo=UTC),
    datetime(2006, 1, 1, tzinfo=UTC),
    datetime(2009, 1, 1, tzinfo=UTC),
    datetime(2012, 7, 1, tzinfo=UTC),
    datetime(2015, 7, 1, tzinfo=UTC),
    datetime(2017, 1, 1, tzinfo=UTC),
)


def load_reduced_ephemeris(
    bsp_path: str | Path = DEFAULT_EPHEMERIS_BSP,
) -> None:
    """Load the reduced Earth-centered Sun/Moon BSP and ECI_TOD frame."""
    _load_eci_tod_frame()
    spice.furnsh(str(Path(bsp_path).resolve()))


def get_sun_moon_states_eci_tod(et: float) -> tuple[np.ndarray, np.ndarray]:
    """Return Earth-centered geometric Sun and Moon states in ECI_TOD.

    States are returned in SPICE units: position in kilometers and velocity in
    kilometers per second.
    """
    sun_state, _ = spice.spkezr("SUN", float(et), "ECI_TOD", "NONE", "EARTH")
    moon_state, _ = spice.spkezr("MOON", float(et), "ECI_TOD", "NONE", "EARTH")
    return np.asarray(sun_state, dtype=float), np.asarray(moon_state, dtype=float)


def epoch_to_et(epoch: str | datetime | float | int) -> float:
    """Convert a user epoch to SPICE ephemeris seconds past J2000.

    Numeric inputs are treated as ET seconds. String and ``datetime`` inputs are
    interpreted as UTC. If a SPICE leap-second kernel is loaded, SpiceyPy
    performs the conversion; otherwise a built-in UTC-to-TT fallback using the
    published leap-second history is used.
    """
    if isinstance(epoch, int | float):
        return float(epoch)

    if isinstance(epoch, str):
        text = epoch.strip()
        with suppress(Exception):
            return float(spice.str2et(text))
        return _datetime_utc_to_et_approx(_parse_utc_datetime(text))

    if isinstance(epoch, datetime):
        with suppress(Exception):
            return float(spice.datetime2et(epoch))
        return _datetime_utc_to_et_approx(epoch)

    raise TypeError("epoch must be a UTC string, datetime, or ET seconds.")


def sample_sun_moon_positions_eci_tod(
    *,
    initial_epoch: str | datetime | float | int,
    duration_s: float,
    step_s: float = 3600.0,
    bsp_path: str | Path = DEFAULT_EPHEMERIS_BSP,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Sample Earth-centered Sun/Moon positions in ECI_TOD for table building.

    Returns ``(times_s, positions_m)`` where ``positions_m`` has ``"sun"`` and
    ``"moon"`` arrays with shape ``(N, 3)`` in meters.
    """
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive.")
    if step_s <= 0.0:
        raise ValueError("step_s must be positive.")

    n_intervals = max(5, int(math.ceil(float(duration_s) / float(step_s))))
    times_s = np.linspace(0.0, float(duration_s), n_intervals + 1, dtype=np.float64)
    epochs = epoch_to_et(initial_epoch) + times_s

    kernel_path = Path(bsp_path).resolve()
    if not kernel_path.is_file():
        raise FileNotFoundError(f"Sun/Moon BSP does not exist: {kernel_path}")

    try:
        _load_eci_tod_frame()
        spice.furnsh(str(kernel_path))
        sun_states, _ = spice.spkezr("SUN", epochs, "ECI_TOD", "NONE", "EARTH")
        moon_states, _ = spice.spkezr("MOON", epochs, "ECI_TOD", "NONE", "EARTH")
    finally:
        with suppress(Exception):
            spice.unload(str(kernel_path))

    return times_s, {
        "sun": np.asarray(sun_states, dtype=np.float64).reshape(-1, 6)[:, 0:3] * 1000.0,
        "moon": np.asarray(moon_states, dtype=np.float64).reshape(-1, 6)[:, 0:3] * 1000.0,
    }


def _parse_utc_datetime(text: str) -> datetime:
    """Parse ISO-like UTC strings accepted by mission scripts."""
    normalized = text.strip()
    if normalized.upper().endswith(" UTC"):
        normalized = normalized[:-4]
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(
            "Epoch strings must be parseable by SPICE or ISO-8601, "
            "for example '2026-01-01T00:00:00Z'."
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _load_eci_tod_frame() -> None:
    """Load the embedded ECI_TOD frame definition into SPICE's kernel pool."""
    spice.lmpool(list(_ECI_TOD_FRAME_KERNEL_POOL))


def _datetime_utc_to_et_approx(epoch: datetime) -> float:
    """Approximate UTC datetime to ET without requiring a leap-second kernel."""
    epoch_utc = epoch if epoch.tzinfo is not None else epoch.replace(tzinfo=timezone.utc)
    epoch_utc = epoch_utc.astimezone(UTC)
    utc_seconds_since_j2000 = (epoch_utc - _J2000_UTC).total_seconds()
    tai_minus_utc = float(_tai_minus_utc_seconds(epoch_utc))
    return utc_seconds_since_j2000 + tai_minus_utc + 32.184


def _tai_minus_utc_seconds(epoch_utc: datetime) -> int:
    """Return TAI-UTC seconds from the built-in leap-second history."""
    if epoch_utc < _LEAP_SECOND_EFFECTIVE_DATES[0]:
        raise ValueError("UTC epoch fallback supports dates on or after 1972-01-01.")
    return 10 + sum(epoch_utc >= date for date in _LEAP_SECOND_EFFECTIVE_DATES[1:])
