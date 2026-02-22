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

from .solvers.rendezvous import RendezvousResult


@dataclass(slots=True)
class AttemptLog:
    stage: str
    attempt: int
    status: str
    message: str = ""


@dataclass(slots=True)
class Solution:
    ok: bool
    result: RendezvousResult | None = None
    attempts: list[AttemptLog] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)
    last_error: str | None = None

    def summary(self) -> str:
        if self.result is not None:
            return self.result.summary()
        lines = ["Octavian solution: FAILED"]
        if self.last_error:
            lines.append(f"  last_error: {self.last_error}")
        if self.attempts:
            lines.append("  attempts:")
            for a in self.attempts:
                lines.append(f"    - stage={a.stage} attempt={a.attempt}: {a.status} {a.message}".rstrip())
        return "\n".join(lines)

    @property
    def traj(self) -> np.ndarray:
        if self.result is None:
            return np.empty((0, 0), dtype=float)
        return np.asarray(self.result.traj, dtype=float)

    def viz(self):
        """Namespace-style access to visualization helpers."""
        from .viz import plotly as _plotly

        self_outer = self

        class _Viz:
            def save_html(self, out_html: str, *, title: str = "trajectory") -> None:
                if self_outer.result is None:
                    raise RuntimeError("No result to visualize")
                _plotly.save_trajectory_html(self_outer.result.traj, out_html, title=title)

        return _Viz()
