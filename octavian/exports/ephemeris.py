"""Validated trajectory exports for common ephemeris file formats."""

from __future__ import annotations

import csv
import os
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import spiceypy as spice
from numpy.typing import ArrayLike, NDArray

from ..data.ephemeris import epoch_to_datetime_utc, epoch_to_et

EphemerisFormat = Literal["stk", "oem", "bsp", "csv"]
Epoch = str | datetime | float | int

_EXTENSION_FORMATS: dict[str, EphemerisFormat] = {
    ".e": "stk",
    ".oem": "oem",
    ".bsp": "bsp",
    ".spk": "bsp",
    ".csv": "csv",
}


@dataclass(frozen=True, slots=True)
class Ephemeris:
    """A Cartesian state history with the metadata needed for file export.

    Octavian stores positions in meters, velocities in meters per second, and
    elapsed times in seconds. Writers perform any format-required unit
    conversion at the file boundary.

    Attributes:
        times_s: Strictly increasing elapsed seconds from ``epoch``.
        states_m_mps: Cartesian rows ``[x, y, z, vx, vy, vz]`` in SI units.
        epoch: UTC string/datetime or SPICE ET seconds at ``times_s == 0``.
        frame_name: Name of the frame already represented by the state rows.
            Export does not rotate coordinates; this label must describe the
            supplied data.
        center_name: Human-readable center of motion.
        object_name: Human-readable ephemeris object name.
        object_id: NAIF object ID used by the BSP writer.
        center_id: NAIF center ID used by the BSP writer.
    """

    times_s: NDArray[np.float64]
    states_m_mps: NDArray[np.float64]
    epoch: Epoch
    frame_name: str = "J2000"
    center_name: str = "EARTH"
    object_name: str = "SPACECRAFT"
    object_id: int = -100_000
    center_id: int = 399

    def __post_init__(self) -> None:
        times = np.asarray(self.times_s, dtype=np.float64).reshape(-1)
        states = np.asarray(self.states_m_mps, dtype=np.float64)
        if states.ndim != 2 or states.shape[1] != 6:
            raise ValueError("states_m_mps must have shape (N, 6)")
        if states.shape[0] != times.size:
            raise ValueError("times_s and states_m_mps must contain the same number of rows")
        if times.size < 2:
            raise ValueError("Ephemeris export requires at least two samples")
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(states)):
            raise ValueError("Ephemeris times and states must be finite")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("times_s must be strictly increasing")
        if not str(self.frame_name).strip():
            raise ValueError("frame_name must not be empty")
        if not str(self.center_name).strip():
            raise ValueError("center_name must not be empty")
        if not str(self.object_name).strip():
            raise ValueError("object_name must not be empty")
        # Validate the epoch at construction so file writers fail before
        # creating partial output.
        epoch_to_et(self.epoch)

        object.__setattr__(self, "times_s", times.copy())
        object.__setattr__(self, "states_m_mps", states.copy())
        object.__setattr__(self, "frame_name", str(self.frame_name).strip())
        object.__setattr__(self, "center_name", str(self.center_name).strip().upper())
        object.__setattr__(self, "object_name", str(self.object_name).strip())
        object.__setattr__(self, "object_id", int(self.object_id))
        object.__setattr__(self, "center_id", int(self.center_id))

    @classmethod
    def from_trajectory(
        cls,
        trajectory: ArrayLike,
        *,
        epoch: Epoch,
        frame_name: str = "J2000",
        center_name: str = "EARTH",
        object_name: str = "SPACECRAFT",
        object_id: int = -100_000,
        center_id: int = 399,
    ) -> Ephemeris:
        """Construct an ephemeris from Octavian ``[R, V, t]`` rows.

        Args:
            trajectory: Array with shape ``(N, 7)`` and columns
                ``[x, y, z, vx, vy, vz, elapsed_time]`` in SI units.
            epoch: Epoch corresponding to elapsed time zero.
            frame_name: Frame already represented by the trajectory.
            center_name: Human-readable center of motion.
            object_name: Human-readable ephemeris object name.
            object_id: NAIF object ID for BSP output.
            center_id: NAIF center ID for BSP output.

        Returns:
            A validated ephemeris ready for one or more exports.
        """
        rows = np.asarray(trajectory, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != 7:
            raise ValueError("trajectory must have shape (N, 7) with [R, V, t] columns")
        return cls(
            times_s=rows[:, 6],
            states_m_mps=rows[:, 0:6],
            epoch=epoch,
            frame_name=frame_name,
            center_name=center_name,
            object_name=object_name,
            object_id=object_id,
            center_id=center_id,
        )

    @property
    def epochs_et(self) -> NDArray[np.float64]:
        """Return one SPICE ET epoch per state row."""
        return np.asarray(epoch_to_et(self.epoch) + self.times_s, dtype=np.float64)

    def write(
        self,
        path: str | Path,
        *,
        format: EphemerisFormat | None = None,
        overwrite: bool = False,
        interpolation_degree: int = 7,
    ) -> Path:
        """Write this ephemeris, inferring the format from the extension.

        Supported extensions are ``.e`` (STK), ``.oem`` (CCSDS OEM), ``.bsp``
        or ``.spk`` (SPICE type-9 SPK), and ``.csv``.

        Args:
            path: Output file.
            format: Explicit format override, normally inferred from ``path``.
            overwrite: Replace an existing file when true.
            interpolation_degree: Requested Lagrange degree for OEM metadata
                and BSP interpolation. It is reduced when the history contains
                too few samples.

        Returns:
            The resolved output path.
        """
        return write_ephemeris(
            self,
            path,
            format=format,
            overwrite=overwrite,
            interpolation_degree=interpolation_degree,
        )


def write_ephemeris(
    ephemeris: Ephemeris,
    path: str | Path,
    *,
    format: EphemerisFormat | None = None,
    overwrite: bool = False,
    interpolation_degree: int = 7,
) -> Path:
    """Write an ephemeris using an explicit format or output extension."""
    output = Path(path).expanduser().resolve()
    selected = _resolve_format(output, format)
    if selected == "stk":
        return write_stk_ephemeris(
            ephemeris,
            output,
            overwrite=overwrite,
            interpolation_degree=interpolation_degree,
        )
    if selected == "oem":
        return write_oem(
            ephemeris,
            output,
            overwrite=overwrite,
            interpolation_degree=interpolation_degree,
        )
    if selected == "bsp":
        return write_spice_bsp(
            ephemeris,
            output,
            overwrite=overwrite,
            interpolation_degree=interpolation_degree,
        )
    return write_csv_ephemeris(ephemeris, output, overwrite=overwrite)


def write_stk_ephemeris(
    ephemeris: Ephemeris,
    path: str | Path,
    *,
    overwrite: bool = False,
    interpolation_degree: int = 7,
) -> Path:
    """Write an STK ASCII ephemeris (``.e``) in meters and seconds."""
    output = _prepare_output_path(path, overwrite=overwrite)
    epoch = epoch_to_datetime_utc(ephemeris.epoch)
    scenario_epoch = epoch.strftime("%d %b %Y %H:%M:%S.%f")
    interpolation_order = _interpolation_degree(
        len(ephemeris.times_s),
        interpolation_degree,
        require_odd=False,
    )
    lines = [
        "stk.v.12.0",
        "",
        "BEGIN Ephemeris",
        "",
        f"NumberOfEphemerisPoints {len(ephemeris.times_s)}",
        f"ScenarioEpoch {scenario_epoch}",
        "InterpolationMethod Lagrange",
        f"InterpolationOrder {interpolation_order}",
        f"CentralBody {ephemeris.center_name.title()}",
        f"CoordinateSystem {ephemeris.frame_name}",
        "DistanceUnit Meters",
        "",
        "EphemerisTimePosVel",
    ]
    for time_s, state in zip(
        ephemeris.times_s,
        ephemeris.states_m_mps,
        strict=True,
    ):
        values = " ".join(_format_float(value) for value in state)
        lines.append(f"{_format_float(time_s)} {values}")
    lines.extend(["", "END Ephemeris", ""])
    _write_text(output, "\n".join(lines), overwrite=overwrite)
    return output


def write_oem(
    ephemeris: Ephemeris,
    path: str | Path,
    *,
    overwrite: bool = False,
    interpolation_degree: int = 7,
    originator: str = "OCTAVIAN",
) -> Path:
    """Write a CCSDS OEM 2.0 KVN file in kilometers and UTC."""
    output = _prepare_output_path(path, overwrite=overwrite)
    epochs = _utc_epochs(ephemeris)
    degree = _interpolation_degree(
        len(ephemeris.times_s),
        interpolation_degree,
        require_odd=False,
    )
    now = datetime.now(timezone.utc)
    lines = [
        "CCSDS_OEM_VERS = 2.0",
        f"CREATION_DATE = {_format_utc_iso(now)}",
        f"ORIGINATOR = {str(originator).strip() or 'OCTAVIAN'}",
        "",
        "META_START",
        f"OBJECT_NAME = {ephemeris.object_name}",
        f"OBJECT_ID = {ephemeris.object_id}",
        f"CENTER_NAME = {ephemeris.center_name}",
        f"REF_FRAME = {ephemeris.frame_name}",
        "TIME_SYSTEM = UTC",
        f"START_TIME = {_format_utc_iso(epochs[0])}",
        f"STOP_TIME = {_format_utc_iso(epochs[-1])}",
        "INTERPOLATION = LAGRANGE",
        f"INTERPOLATION_DEGREE = {degree}",
        "META_STOP",
        "",
    ]
    states_km_kmps = ephemeris.states_m_mps / 1_000.0
    for epoch, state in zip(epochs, states_km_kmps, strict=True):
        values = " ".join(_format_float(value) for value in state)
        lines.append(f"{_format_utc_iso(epoch)} {values}")
    lines.append("")
    _write_text(output, "\n".join(lines), overwrite=overwrite)
    return output


def write_spice_bsp(
    ephemeris: Ephemeris,
    path: str | Path,
    *,
    overwrite: bool = False,
    interpolation_degree: int = 7,
) -> Path:
    """Write a SPICE type-9 SPK segment (normally named ``.bsp``).

    SPICE stores positions in kilometers, velocities in kilometers per second,
    and epochs as TDB-compatible ephemeris seconds. ``frame_name`` must be a
    frame recognized by SPICE, such as ``"J2000"``.
    """
    output = _prepare_output_path(path, overwrite=overwrite)
    degree = _interpolation_degree(
        len(ephemeris.times_s),
        interpolation_degree,
        require_odd=True,
    )
    epochs = ephemeris.epochs_et
    states = ephemeris.states_m_mps / 1_000.0
    temp_path = _temporary_sibling(output)
    handle: int | None = None
    try:
        handle = spice.spkopn(str(temp_path), "OCTAVIAN EPHEMERIS", 0)
        segment_id = f"OCTAVIAN {ephemeris.object_name}"[:40]
        spice.spkw09(
            handle,
            ephemeris.object_id,
            ephemeris.center_id,
            ephemeris.frame_name,
            float(epochs[0]),
            float(epochs[-1]),
            segment_id,
            degree,
            len(epochs),
            states,
            epochs,
        )
        spice.spkcls(handle)
        handle = None
        if output.exists() and not overwrite:
            raise FileExistsError(f"Output file already exists: {output}")
        os.replace(temp_path, output)
    finally:
        if handle is not None:
            with suppress(Exception):
                spice.spkcls(handle)
        with suppress(FileNotFoundError):
            temp_path.unlink()
    return output


def write_csv_ephemeris(
    ephemeris: Ephemeris,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a portable SI-unit CSV ephemeris with UTC and elapsed time."""
    output = _prepare_output_path(path, overwrite=overwrite)
    mode = "w" if overwrite else "x"
    epochs = _utc_epochs(ephemeris)
    with output.open(mode, newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "epoch_utc",
                "elapsed_time_s",
                "x_m",
                "y_m",
                "z_m",
                "vx_mps",
                "vy_mps",
                "vz_mps",
            ]
        )
        for epoch, time_s, state in zip(
            epochs,
            ephemeris.times_s,
            ephemeris.states_m_mps,
            strict=True,
        ):
            writer.writerow(
                [
                    _format_utc_iso(epoch),
                    _format_float(time_s),
                    *(_format_float(value) for value in state),
                ]
            )
    return output


def _resolve_format(path: Path, format: EphemerisFormat | None) -> EphemerisFormat:
    """Return a normalized writer format."""
    if format is not None:
        selected = str(format).strip().lower()
        if selected not in {"stk", "oem", "bsp", "csv"}:
            raise ValueError("format must be one of: stk, oem, bsp, csv")
        return selected  # type: ignore[return-value]
    try:
        return _EXTENSION_FORMATS[path.suffix.lower()]
    except KeyError as exc:
        supported = ", ".join(sorted(_EXTENSION_FORMATS))
        raise ValueError(
            f"Cannot infer ephemeris format from {path.suffix!r}; use one of {supported} "
            "or pass format= explicitly."
        ) from exc


def _prepare_output_path(path: str | Path, *, overwrite: bool) -> Path:
    """Resolve and validate a requested output path without creating it."""
    output = Path(path).expanduser().resolve()
    if not output.parent.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {output.parent}")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output}")
    return output


def _write_text(path: Path, content: str, *, overwrite: bool) -> None:
    """Write UTF-8 text with explicit overwrite semantics."""
    mode = "w" if overwrite else "x"
    with path.open(mode, encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def _temporary_sibling(output: Path) -> Path:
    """Reserve a unique sibling name that SPICE can create itself."""
    descriptor, name = tempfile.mkstemp(
        prefix=f".{output.stem}-",
        suffix=output.suffix,
        dir=output.parent,
    )
    os.close(descriptor)
    path = Path(name)
    path.unlink()
    return path


def _utc_epochs(ephemeris: Ephemeris) -> list[datetime]:
    """Return absolute UTC datetimes for every ephemeris row."""
    if isinstance(ephemeris.epoch, int | float):
        return [
            epoch_to_datetime_utc(float(epoch_et))
            for epoch_et in ephemeris.epochs_et
        ]
    start = epoch_to_datetime_utc(ephemeris.epoch)
    return [start + timedelta(seconds=float(time_s)) for time_s in ephemeris.times_s]


def _interpolation_degree(
    sample_count: int,
    requested: int,
    *,
    require_odd: bool,
) -> int:
    """Return a valid interpolation degree for the available samples."""
    degree = int(requested)
    if degree < 1:
        raise ValueError("interpolation_degree must be positive")
    degree = min(degree, sample_count - 1)
    if require_odd and degree % 2 == 0:
        degree -= 1
    return max(1, degree)


def _format_utc_iso(value: datetime) -> str:
    """Return an OEM-compatible UTC timestamp."""
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _format_float(value: float) -> str:
    """Return a compact, round-trippable ephemeris number."""
    return f"{float(value):.16e}"
