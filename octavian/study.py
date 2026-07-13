"""Study helpers for repeatable parameter sweeps."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from .solvers import SolverOptions
from .solvers.preconfigured import RendezvousResult, solve
from .specs import TwoImpulseFreeTimeSpec, TwoImpulsePreCoastSpec

Spec = TwoImpulseFreeTimeSpec | TwoImpulsePreCoastSpec


def grid(
    base_spec: Spec,
    overrides: Sequence[dict[str, Any]],
    *,
    options: SolverOptions | None = None,
    save_dir: str | Path | None = None,
    save_prefix: str = "run",
) -> list[RendezvousResult]:
    """Run a grid study by applying overrides to a base spec.

    Args:
        base_spec: Baseline problem specification.
        overrides: Override dictionaries passed to
            ``dataclasses.replace(base_spec, **override)``.
        options: Shared solver options. If omitted, defaults are used.
        save_dir: Optional directory where each result is saved as ``.npz`` and
            companion JSON.
        save_prefix: Prefix for output filenames.

    Returns:
        Results in the same order as ``overrides``.
    """
    solver_options = options or SolverOptions()
    output_dir = Path(save_dir).expanduser().resolve() if save_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    results: list[RendezvousResult] = []
    for override_index, override_values in enumerate(overrides):
        sweep_spec = replace(base_spec, **override_values)
        result = solve(sweep_spec, options=solver_options)
        result.info.setdefault("study_index", override_index)
        result.info.setdefault("overrides", dict(override_values))
        results.append(result)

        if output_dir is not None:
            stem = f"{save_prefix}_{override_index:04d}"
            result.to_npz(output_dir / f"{stem}.npz")
            (output_dir / f"{stem}.json").write_text(result.to_json(indent=2))

    return results


def best_by(
    results: Iterable[RendezvousResult],
    key: str = "total_dv_mps",
    *,
    require_converged: bool = True,
) -> RendezvousResult:
    """Select the best result in a study by a named metric.

    Args:
        results: Results to compare.
        key: Metric name. Supported values are ``"total_dv_mps"``, ``"tf_s"``,
            and ``"last_obj"``.
        require_converged: Whether to discard non-converged results before
            ranking.

    Returns:
        The best result under the requested metric.

    Raises:
        ValueError: If no results remain after filtering or the metric name is
            unsupported.
    """
    candidate_results = [
        result for result in results if (result.converged or not require_converged)
    ]
    if not candidate_results:
        raise ValueError("No results available after filtering.")
    if key == "total_dv_mps":
        return min(candidate_results, key=lambda result: result.total_dv_mps())
    if key == "tf_s":
        return min(candidate_results, key=lambda result: result.tf_s())
    if key == "last_obj":
        return min(candidate_results, key=lambda result: float(result.last_obj))
    raise ValueError(f"Unsupported key: {key!r}.")
