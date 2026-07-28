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
from typing import Any

import numpy as np

from .coordinates import CoordinateFrame, SolverScaling
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

        return _Viz()
