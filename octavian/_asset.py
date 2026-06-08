"""ASSET backend access and compatibility helpers."""

from __future__ import annotations

from typing import Any

try:
    import asset_asrl as ast  # type: ignore
except Exception:  # pragma: no cover
    ast = None  # type: ignore

if ast is not None:  # pragma: no cover
    vf = ast.VectorFunctions
    oc = ast.OptimalControl
    Tmodes = oc.TranscriptionModes
else:  # pragma: no cover
    vf = None  # type: ignore
    oc = None  # type: ignore
    Tmodes = None  # type: ignore


def require_asset(feature: str = "ASSET-backed solves") -> None:
    """Raise a clear error when ASSET is unavailable.

    Parameters
    ----------
    feature
        User-facing feature name that needs ASSET.

    Raises
    ------
    RuntimeError
        If ``asset_asrl`` could not be imported.
    """
    if ast is None:
        raise RuntimeError(
            f"asset_asrl is required for {feature}. "
            "Install it and its compiled dependencies before calling this API."
        )


def add_back_time_bound(asset_phase: Any, state_dim: int, lower_s: float, upper_s: float) -> None:
    """Apply an ASSET back-bound on phase time.

    Parameters
    ----------
    asset_phase
        ASSET phase object.
    state_dim
        State dimension of the phase. Kept for call-site readability and future
        backend adapters.
    lower_s
        Lower bound on back time in seconds.
    upper_s
        Upper bound on back time in seconds.
    """
    del state_dim
    asset_phase.addLUVarBound("Back", "time", float(lower_s), float(upper_s))


def fix_front_time(asset_phase: Any, time_s: float = 0.0) -> None:
    """Fix the phase front time using the semantic ASSET time variable.

    Parameters
    ----------
    asset_phase
        ASSET phase object.
    time_s
        Boundary time in seconds.
    """
    import numpy as np

    asset_phase.addBoundaryValue("Front", ["t"], np.asarray([float(time_s)], dtype=float))


def solve_with_standard_sequence(ocp: Any) -> bool:
    """Run Octavian's standard ASSET solve sequence.

    Parameters
    ----------
    ocp
        ASSET optimal-control problem.

    Returns
    -------
    bool
        ASSET convergence flag.
    """
    return bool(ocp.solve_optimize_solve())
