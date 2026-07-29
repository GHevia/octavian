"""Solution and reporting.

`Solution` is Octavian's stable output contract.
It wraps backend-specific result objects (currently `RendezvousResult`) and
adds:
  - attempt history
  - a consistent summary string
  - a small viz namespace
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .coordinates import CoordinateFrame, SolverScaling
from .exports import Ephemeris
from .exports.ephemeris import EphemerisFormat
from .solvers.preconfigured import RendezvousResult


@dataclass(slots=True)
class AttemptLog:
    """One solver attempt recorded by `MissionRunner`.

    Attributes:
        stage: Runner stage label, or ``"default"`` when no staged plan is used.
        attempt: One-based attempt number within the stage.
        status: Short status string such as ``"ok"`` or ``"fail"``.
        message: Optional failure or diagnostic message.
    """

    stage: str
    attempt: int
    status: str
    message: str = ""


@dataclass(slots=True)
class Solution:
    """Stable user-facing wrapper around a backend solve result.

    `Solution` keeps backend result objects from leaking directly into mission
    scripts. Successful solves expose the backend result through ``result`` and
    convenience accessors such as ``traj``. Failed solves still return structured
    attempt logs when ``SolveConfig.raise_on_fail`` is false.
    """

    ok: bool
    result: RendezvousResult | None = None
    attempts: list[AttemptLog] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)
    last_error: str | None = None

    def summary(self) -> str:
        """Return a human-readable solve summary."""
        if self.result is not None:
            return self.result.summary()
        lines = ["Octavian solution: FAILED"]
        if self.last_error:
            lines.append(f"  last_error: {self.last_error}")
        if self.attempts:
            lines.append("  attempts:")
            for a in self.attempts:
                lines.append(
                    f"    - stage={a.stage} attempt={a.attempt}: {a.status} {a.message}".rstrip()
                )
        return "\n".join(lines)

    @property
    def traj(self) -> np.ndarray:
        """Return the solved trajectory array, or an empty array on failure."""
        if self.result is None:
            return np.empty((0, 0), dtype=float)
        return np.asarray(self.result.traj, dtype=float)

    @property
    def frame(self) -> CoordinateFrame | None:
        """Return the reference frame declared by the solver result."""
        if self.result is None:
            return None
        value = self.result.info.get("frame")
        if isinstance(value, CoordinateFrame):
            return value
        if isinstance(value, dict):
            return CoordinateFrame.from_dict(value)
        return None

    @property
    def scaling(self) -> SolverScaling | None:
        """Return characteristic units used by the solver, when available."""
        if self.result is None:
            return None
        value = self.result.info.get("scaling")
        if isinstance(value, SolverScaling):
            return value
        if isinstance(value, dict):
            return SolverScaling(**value)
        return None

    @property
    def chief_trajectory_eci(self) -> np.ndarray:
        """Return the propagated chief absolute history for relative solves."""
        if self.result is None:
            return np.empty((0, 7), dtype=float)
        value = self.result.info.get("chief_trajectory_eci")
        if value is None:
            return np.empty((0, 7), dtype=float)
        return np.asarray(value, dtype=float)

    @property
    def deputy_trajectory_eci(self) -> np.ndarray:
        """Return the propagated deputy absolute history for relative solves."""
        if self.result is None:
            return np.empty((0, 7), dtype=float)
        value = self.result.info.get("deputy_trajectory_eci")
        if value is None:
            return np.empty((0, 7), dtype=float)
        return np.asarray(value, dtype=float)

    @property
    def native_relative_trajectory(self) -> np.ndarray:
        """Return the solver's native relative state and time history.

        For a D'Amico phase, for example, columns are
        ``[δa, δλ, δex, δey, δix, δiy, t]``. The ordinary :attr:`traj`
        accessor always remains the reconstructed ``[RIC state, t]`` view.
        """
        if self.result is None:
            return np.empty((0, 7), dtype=float)
        value = self.result.info.get("native_relative_trajectory")
        if value is None:
            return np.empty((0, 7), dtype=float)
        return np.asarray(value, dtype=float)

    @property
    def native_relative_phase_trajectories(self) -> tuple[np.ndarray, ...]:
        """Return one control-free native state/time array per relative phase.

        A stitched native trajectory is unavailable when phase layouts change,
        for example when an initial coast does not carry mass but a later
        finite-burn chain does. This accessor always preserves each phase's
        declared native representation.
        """
        if self.result is None:
            return ()
        value = self.result.info.get("native_relative_phase_trajectories")
        if value is None:
            return ()
        return tuple(np.asarray(trajectory, dtype=float) for trajectory in value)

    @property
    def relative_propagation_mode(self) -> str | None:
        """Return the selected relative formulation, or ``None`` if inertial."""
        if self.result is None:
            return None
        value = self.result.info.get("relative_propagation_mode")
        return None if value is None else str(value)

    @property
    def phase_control_trajectories(self) -> tuple[np.ndarray, ...]:
        """Return ``[time, controls...]`` arrays for every compiled phase.

        Vector-throttle components and scalar throttle are dimensionless.
        Euler-rate columns are converted from normalized optimizer variables
        to physical radians per second.
        """
        if self.result is None:
            return ()
        value = self.result.info.get("phase_control_trajectories")
        if value is None:
            return ()
        return tuple(np.asarray(trajectory, dtype=float) for trajectory in value)

    @property
    def attitude_phase_trajectories(self) -> tuple[np.ndarray, ...]:
        """Return Euler attitude histories for kinematic-attitude phases.

        Each array contains ``[yaw, pitch, roll, time, yaw_rate, pitch_rate,
        roll_rate]`` in radians, seconds, and radians per second.
        """
        if self.result is None:
            return ()
        value = self.result.info.get("attitude_phase_trajectories")
        if value is None:
            return ()
        return tuple(np.asarray(trajectory, dtype=float) for trajectory in value)

    def to_ephemeris(
        self,
        *,
        trajectory: str = "auto",
        epoch: str | datetime | float | int | None = None,
        frame_name: str | None = None,
        center_name: str | None = None,
        object_name: str | None = None,
        object_id: int = -100_000,
        center_id: int | None = None,
    ) -> Ephemeris:
        """Return a validated ephemeris selected from this solution.

        Args:
            trajectory: ``"auto"`` (the deputy for relative solutions,
                otherwise the solved trajectory), ``"solved"``, ``"chief"``,
                or ``"deputy"``. Chief and deputy histories are absolute ECI
                trajectories reconstructed by exact relative formulations.
            epoch: Epoch corresponding to trajectory time zero. When omitted,
                use the originating mission's ``initial_epoch``.
            frame_name: Frame label for the state values. This does not rotate
                coordinates. Absolute ECI histories default to ``"J2000"``.
            center_name: Center-of-motion label, normally inferred from the
                solution metadata.
            object_name: Exported object label.
            object_id: NAIF object ID used by BSP output.
            center_id: NAIF center ID used by BSP output, inferred for Earth,
                Moon, and Sun when omitted.

        Returns:
            An :class:`octavian.exports.Ephemeris` ready to write.

        Raises:
            RuntimeError: If the solution has no result.
            ValueError: If the selected trajectory or epoch is unavailable.

        Note:
            File export labels the selected data but does not perform a frame
            rotation. Select ``"chief"`` or ``"deputy"`` when exporting a
            relative solution to an inertial OEM or BSP.
        """
        if self.result is None:
            raise RuntimeError("No result to export")

        selected = str(trajectory).strip().lower()
        if selected == "auto":
            selected = (
                "deputy"
                if self.frame is not None
                and self.frame.kind == "relative"
                and self.deputy_trajectory_eci.size
                else "solved"
            )
        if selected not in {"solved", "chief", "deputy"}:
            raise ValueError("trajectory must be one of: auto, solved, chief, deputy")

        if selected == "chief":
            rows = self.chief_trajectory_eci
        elif selected == "deputy":
            rows = self.deputy_trajectory_eci
        else:
            rows = self.traj
        if rows.ndim != 2 or rows.shape[1] != 7 or len(rows) < 2:
            raise ValueError(f"The solution does not contain an exportable {selected} trajectory")

        export_epoch = epoch
        if export_epoch is None:
            export_epoch = self.info.get("initial_epoch")
        if export_epoch is None:
            export_epoch = self.result.info.get("initial_epoch")
        if export_epoch is None:
            raise ValueError("Ephemeris export requires epoch= or Mission(initial_epoch=...).")

        absolute_history = selected in {"chief", "deputy"}
        solution_frame = self.frame
        inferred_frame = "J2000"
        if not absolute_history and solution_frame is not None:
            orientation = solution_frame.orientation.strip()
            inferred_frame = (
                "J2000"
                if orientation.upper() in {"ECI", "ICRF", "J2000", "EME2000"}
                else orientation
            )
        inferred_center = str(
            self.result.info.get("central_body")
            or (solution_frame.origin if solution_frame is not None else "earth")
        )
        if not absolute_history and solution_frame is not None:
            inferred_center = solution_frame.origin

        resolved_center = str(center_name or inferred_center).strip().upper()
        resolved_object = object_name or (
            "CHIEF" if selected == "chief" else "DEPUTY" if selected == "deputy" else "SPACECRAFT"
        )
        resolved_center_id = (
            int(center_id) if center_id is not None else _common_naif_center_id(resolved_center)
        )
        return Ephemeris.from_trajectory(
            rows,
            epoch=export_epoch,
            frame_name=frame_name or inferred_frame,
            center_name=resolved_center,
            object_name=resolved_object,
            object_id=object_id,
            center_id=resolved_center_id,
        )

    def export_ephemeris(
        self,
        path: str | Path,
        *,
        trajectory: str = "auto",
        epoch: str | datetime | float | int | None = None,
        frame_name: str | None = None,
        center_name: str | None = None,
        object_name: str | None = None,
        object_id: int = -100_000,
        center_id: int | None = None,
        format: EphemerisFormat | None = None,
        overwrite: bool = False,
        interpolation_degree: int = 7,
    ) -> Path:
        """Export a solved, chief, or deputy trajectory to an ephemeris file.

        The output extension selects STK ``.e``, CCSDS ``.oem``, SPICE
        ``.bsp``/``.spk``, or SI-unit ``.csv``. See :meth:`to_ephemeris` for
        trajectory and frame semantics.

        Returns:
            The resolved path written by the selected exporter.
        """
        ephemeris = self.to_ephemeris(
            trajectory=trajectory,
            epoch=epoch,
            frame_name=frame_name,
            center_name=center_name,
            object_name=object_name,
            object_id=object_id,
            center_id=center_id,
        )
        return ephemeris.write(
            path,
            format=format,
            overwrite=overwrite,
            interpolation_degree=interpolation_degree,
        )

    def viz(self):
        """Namespace-style access to visualization helpers."""
        from .viz import plotly as _plotly

        self_outer = self

        class _Viz:
            def save_html(self, out_html: str, *, title: str = "trajectory") -> None:
                """Save a frame-aware trajectory plot.

                Relative results use RIC axes and a chief marker; inertial
                results retain the Earth-centered trajectory view.
                """
                if self_outer.result is None:
                    raise RuntimeError("No result to visualize")
                frame = self_outer.frame
                phase_segments = self_outer.result.info.get("phase_segments")
                if frame is not None and frame.kind == "relative":
                    _plotly.save_relative_trajectory_html(
                        self_outer.result.traj,
                        out_html,
                        maneuvers=self_outer.result.maneuvers,
                        phase_segments=phase_segments,
                        title=title,
                    )
                    return
                _plotly.save_trajectory_html(
                    self_outer.result.traj,
                    out_html,
                    maneuvers=self_outer.result.maneuvers,
                    phase_segments=phase_segments,
                    title=title,
                )

            def save_relative_html(
                self,
                out_html: str,
                *,
                title: str = "relative trajectory",
                chief_radius_m: float = 0.0,
            ) -> None:
                """Save the result explicitly as a chief-centered RIC plot."""
                if self_outer.result is None:
                    raise RuntimeError("No result to visualize")
                _plotly.save_relative_trajectory_html(
                    self_outer.result.traj,
                    out_html,
                    maneuvers=self_outer.result.maneuvers,
                    phase_segments=self_outer.result.info.get("phase_segments"),
                    title=title,
                    chief_radius_m=chief_radius_m,
                )

            def save_diagnostics_html(
                self,
                out_html: str,
                *,
                title: str = "trajectory diagnostics",
            ) -> None:
                """Save frame-aware state and geometry values over time."""
                if self_outer.result is None:
                    raise RuntimeError("No result to visualize")
                frame = self_outer.frame
                _plotly.save_trajectory_diagnostics_html(
                    self_outer.result.traj,
                    out_html,
                    frame_kind=(frame.kind if frame is not None else "inertial"),
                    mu_m3ps2=self_outer.result.info.get("mu_m3ps2"),
                    solar_directions_ric=self_outer.result.info.get("solar_directions_ric"),
                    title=title,
                )

        return _Viz()


def _common_naif_center_id(center_name: str) -> int:
    """Return a NAIF center code for common Octavian central bodies."""
    normalized = str(center_name).strip().upper().replace("-", "_").replace(" ", "_")
    center_ids = {
        "SOLAR_SYSTEM_BARYCENTER": 0,
        "SUN": 10,
        "EARTH": 399,
        "MOON": 301,
    }
    try:
        return center_ids[normalized]
    except KeyError as exc:
        raise ValueError(f"center_id= is required for unrecognized center {center_name!r}") from exc
