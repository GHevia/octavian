"""ASSET backend access, compatibility helpers, and solve safety wrappers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

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


NON_MONOTONIC_TIME_MESSAGE = "Non monotonic time coordinates in LGLInterpTable."
_NON_MONOTONIC_TIME_ALIASES = (
    "non monotonic time coordinates in lglinterptable",
    "non monotonic time coordinates",
    "duplicate time coordinates in lglinterptable",
    "duplicate time coordinates",
)


class AssetNonMonotonicTimeError(RuntimeError):
    """Raised when ASSET repeatedly fails with non-monotonic mesh time coordinates."""


@dataclass(frozen=True, slots=True)
class AssetSolveAttempt:
    """Configuration for one ASSET solve attempt inside Octavian's wrapper."""

    label: str
    adaptive_mesh: bool | None = None


@dataclass(frozen=True)
class AssetSolveReport:
    """Separate solver stages and preserve the optimization iterate's duals.

    Feasibility accepts ASSET's CONVERGED and ACCEPTABLE flags. Optimality is
    established only by CONVERGED from optimize(), before any recovery solve.
    Costates and trajectories are snapshots of that same optimization iterate.
    """

    stages: tuple[dict[str, Any], ...]
    optimization_phases: tuple[dict[str, Any], ...]
    mesh_converged: bool | None

    @property
    def feasible(self) -> bool:
        return self.stages[-1]["flag"] in {"CONVERGED", "ACCEPTABLE"}

    @property
    def optimality_converged(self) -> bool:
        return any(
            stage["stage"] == "optimize" and stage["flag"] == "CONVERGED" for stage in self.stages
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "feasible": self.feasible,
            "optimality_converged": self.optimality_converged,
            "mesh_converged": self.mesh_converged,
            "stages": list(self.stages),
        }


def _phase_snapshot(phase: Any) -> dict[str, Any]:
    snapshot = {"trajectory": np.asarray(phase.returnTraj(), dtype=float).tolist()}
    for name, method in (("costates", "returnCostateTraj"), ("defects", "returnTrajError")):
        try:
            snapshot[name] = np.asarray(getattr(phase, method)(), dtype=float).tolist()
        except (AttributeError, RuntimeError, ValueError) as exc:
            snapshot[name] = None
            snapshot[f"{name}_error"] = str(exc)
    return snapshot


def is_non_monotonic_time_error(error: BaseException | str) -> bool:
    """Return whether an ASSET error is the known non-monotonic time failure.

    ASSET can raise a hard exception when adaptive mesh refinement creates time
    coordinates that are not strictly increasing. The exact exception type is
    backend-dependent, so Octavian identifies this mode by the stable diagnostic
    text emitted by ASSET.
    """
    message = str(error).lower().replace("_", " ").replace("-", " ").replace(".", "")
    return any(alias in message for alias in _NON_MONOTONIC_TIME_ALIASES)


def solve_with_standard_sequence(
    ocp: Any,
    *,
    phases: Sequence[Any] = (),
    retry_non_monotonic_time: bool = True,
) -> AssetSolveReport:
    """Run Octavian's standard ASSET solve sequence with targeted recovery.

    Parameters
    ----------
    ocp
        ASSET optimal-control problem.
    phases
        Compiled ASSET phases that belong to ``ocp``. Passing phases lets the
        retry path disable adaptive mesh on every phase, not just the OCP.
    retry_non_monotonic_time
        Whether to catch ASSET's non-monotonic time-coordinate failure and retry
        with adaptive mesh disabled. This failure usually comes from adaptive
        mesh refinement moving collocation nodes out of time order inside a
        single ``solve_optimize_solve`` call.

    Returns
    -------
    AssetSolveReport
        Explicit feasibility, optimization, recovery, and mesh status. ASSET
        convergence enums must never be interpreted through bool(): every
        enum, including DIVERGING, is truthy in supported versions.

    Raises
    ------
    AssetNonMonotonicTimeError
        If the first solve and the adaptive-mesh-disabled retry both fail with
        the non-monotonic time-coordinate diagnostic.
    """
    attempts = [AssetSolveAttempt("standard")]
    if retry_non_monotonic_time:
        attempts.append(AssetSolveAttempt("retry_without_adaptive_mesh", adaptive_mesh=False))

    non_monotonic_messages: list[str] = []
    for attempt in attempts:
        if attempt.adaptive_mesh is not None:
            _set_adaptive_mesh(ocp, phases, attempt.adaptive_mesh)
        try:
            stages = []
            snapshots = ()
            for stage in ("solve", "optimize", "recover"):
                started = perf_counter()
                flag = getattr(ocp, "solve" if stage == "recover" else stage)()
                name = flag.name
                stages.append(
                    {
                        "stage": stage,
                        "flag": name,
                        "iterations": int(ocp.optimizer.LastIterNum),
                        "objective": float(ocp.optimizer.LastObjVal),
                        "elapsed_s": perf_counter() - started,
                        "attempt": attempt.label,
                    }
                )
                if stage == "optimize":
                    snapshots = tuple(_phase_snapshot(phase) for phase in phases)
                    if name == "CONVERGED":
                        break
            mesh_converged = (
                bool(ocp.MeshConverged) if getattr(ocp, "AdaptiveMesh", False) else None
            )
            return AssetSolveReport(tuple(stages), snapshots, mesh_converged)
        except Exception as exc:  # noqa: BLE001 - backend raises implementation-specific exceptions
            if not is_non_monotonic_time_error(exc):
                raise
            non_monotonic_messages.append(f"{attempt.label}: {exc}")

    joined_messages = "\n".join(non_monotonic_messages)
    raise AssetNonMonotonicTimeError(
        "ASSET failed because collocation mesh time coordinates became non-monotonic. "
        "Octavian retried once with adaptive mesh disabled, but the retry also failed. "
        "Try a coarser mesh, wider time bounds, or a simpler initial guess.\n"
        f"{joined_messages}"
    )


def set_ocp_threads(ocp: Any, threads: tuple[int, int] | None) -> None:
    """Apply ASSET OCP threading controls when requested.

    Parameters
    ----------
    ocp
        ASSET optimal-control problem.
    threads
        Optional ``(optimizer_threads, mesh_threads)`` pair passed to
        ``ocp.setThreads``. Use ``(1, 1)`` for deterministic test solves.
    """
    if threads is None:
        return

    if len(threads) != 2:
        raise ValueError("asset_threads must be a two-item tuple, such as (1, 1).")

    optimizer_threads, mesh_threads = (int(threads[0]), int(threads[1]))
    if optimizer_threads < 1 or mesh_threads < 1:
        raise ValueError("asset_threads values must both be positive integers.")

    setter = getattr(ocp, "setThreads", None)
    if setter is None:
        raise RuntimeError("This ASSET OCP object does not expose setThreads().")
    setter(optimizer_threads, mesh_threads)


def configure_optimizer(optimizer: Any, options: Any) -> None:
    """Apply shared iteration limits, tolerances, and output settings."""
    optimizer.PrintLevel = int(options.print_level)
    optimizer.MaxLSIters = int(options.max_ls_iters)
    optimizer.MaxIters = options.max_iterations
    optimizer.KKTtol = options.optimality_tolerance
    optimizer.EContol = options.feasibility_tolerance
    optimizer.IContol = options.feasibility_tolerance
    optimizer.Bartol = options.feasibility_tolerance
    optimizer.set_QPOrderingMode(str(options.qp_ordering_mode))


def _set_adaptive_mesh(ocp: Any, phases: Sequence[Any], enabled: bool) -> None:
    """Best-effort adaptive mesh switch for an OCP and its phases."""
    for target in (ocp, *tuple(phases)):
        setter = getattr(target, "setAdaptiveMesh", None)
        if setter is None:
            continue
        setter(bool(enabled))
