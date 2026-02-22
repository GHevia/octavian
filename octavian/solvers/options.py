"""Solver configuration objects.

Octavian's philosophy is that *Python is the GUI*:
- User-facing configuration is explicit and typed.
- Mission/solver scripts are readable and composable.
- Internals remain small and unsurprising.

This module defines small option dataclasses that keep optimizer knobs out of
solver implementations.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SolverOptions:
    """Common options shared across ASSET-backed solvers.

    Attributes:
        print_level: ASSET optimizer PrintLevel. 0 is quiet.
        max_ls_iters: ASSET optimizer MaxLSIters (line-search iterations).
        qp_ordering_mode: QP ordering mode used by ASSET (string passed to
            ``optimizer.set_QPOrderingMode``).
        enable_auto_scaling: Whether to enable ASSET phase auto-scaling.
        enable_adaptive_mesh: Whether to allow adaptive mesh refinement.
    """

    print_level: int = 0
    max_ls_iters: int = 2
    qp_ordering_mode: str = "MINDEG"
    enable_auto_scaling: bool = True
    enable_adaptive_mesh: bool = True
