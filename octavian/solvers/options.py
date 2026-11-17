"""Solver configuration objects.

Octavian's philosophy is that *Python is the GUI*:
- User-facing configuration is explicit and typed.
- Mission/solver scripts are readable and composable.
- Internals remain small and unsurprising.

This module defines small option dataclasses that keep optimizer knobs out of
solver implementations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(slots=True)
class SolverOptions:
    """Common options shared across ASSET-backed solvers.

    Attributes:
        print_level: ASSET optimizer PrintLevel. 0 is quiet.
        max_ls_iters: ASSET optimizer MaxLSIters (line-search iterations).
        max_iterations: Maximum iterations per feasibility/optimization stage.
        feasibility_tolerance: Scaled equality, inequality, and barrier tolerances.
        optimality_tolerance: Scaled KKT tolerance for objective optimization.
        transcription_mode: ``LGL3`` (default), ``LGL5``, or ``LGL7``.
            Higher orders use more collocation nodes per segment and can
            resolve long orbital arcs with fewer segments.
        qp_ordering_mode: QP ordering mode used by ASSET (string passed to
            ``optimizer.set_QPOrderingMode``).
        enable_auto_scaling: Whether to enable ASSET phase auto-scaling.
        enable_adaptive_mesh: Whether to allow adaptive mesh refinement.
        asset_threads: Optional ``(optimizer_threads, mesh_threads)`` pair passed
            to ASSET's ``ocp.setThreads``. Use ``(1, 1)`` for deterministic
            regression tests when ASSET's multi-threaded solve path is noisy.
    """

    print_level: int = 0
    max_ls_iters: int = 2
    max_iterations: int = 500
    feasibility_tolerance: float = 1e-6
    optimality_tolerance: float = 1e-6
    transcription_mode: str = "LGL3"
    qp_ordering_mode: str = "MINDEG"
    enable_auto_scaling: bool = True
    enable_adaptive_mesh: bool = True
    asset_threads: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, int)
            or self.max_iterations < 1
        ):
            raise ValueError("max_iterations must be a positive integer.")
        for name in ("feasibility_tolerance", "optimality_tolerance"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.transcription_mode not in {"LGL3", "LGL5", "LGL7"}:
            raise ValueError("transcription_mode must be 'LGL3', 'LGL5', or 'LGL7'.")
