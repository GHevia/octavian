"""Study utilities (parameter sweeps).

Octavian encourages a workflow where a mission script is a *configuration file*
and experiments are repeatable.

This module provides small helpers to sweep specs, run solvers, and persist
results in a consistent layout.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from .solvers import SolverOptions
from .solvers.rendezvous import RendezvousResult, solve
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
        base_spec: The baseline problem specification.
        overrides: A list of dictionaries. Each dictionary is passed to
            ``dataclasses.replace(base_spec, **override)`` to create a new spec.
        options: Shared solver options. If omitted, defaults are used.
        save_dir: If provided, each result is saved as ``.npz`` and a small
            metadata JSON in this directory.
        save_prefix: Prefix for output filenames.

    Returns:
        List of results in the same order as ``overrides``.

    Notes:
        - If ``save_dir`` is provided, the directory is created if needed.
        - Results include the override index in ``result.info['study_index']``.
    """
    opts = options or SolverOptions()
    out_dir = Path(save_dir).expanduser().resolve() if save_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    results: list[RendezvousResult] = []
    for i, ov in enumerate(overrides):
        spec_i = replace(base_spec, **ov)
        res = solve(spec_i, options=opts)
        res.info.setdefault("study_index", i)
        res.info.setdefault("overrides", dict(ov))
        results.append(res)

        if out_dir is not None:
            stem = f"{save_prefix}_{i:04d}"
            npz_path = out_dir / f"{stem}.npz"
            json_path = out_dir / f"{stem}.json"
            res.to_npz(npz_path)
            json_path.write_text(res.to_json(indent=2))

    return results


def best_by(
    results: Iterable[RendezvousResult],
    key: str = "total_dv_mps",
    *,
    require_converged: bool = True,
) -> RendezvousResult:
    """Select the best result in a study by a named metric.

    Args:
        results: Iterable of results.
        key: Metric name. Supported:
            - ``"total_dv_mps"``: total delta-v magnitude sum.
            - ``"tf_s"``: final time in seconds (from trajectory).
            - ``"last_obj"``: last objective value.
        require_converged: If True, ignore non-converged results.

    Returns:
        The best result.

    Raises:
        ValueError: If no results are available after filtering.
    """
    candidates = [r for r in results if (r.converged or not require_converged)]
    if not candidates:
        raise ValueError("No results available after filtering.")
    if key == "total_dv_mps":
        return min(candidates, key=lambda r: r.total_dv_mps())
    if key == "tf_s":
        return min(candidates, key=lambda r: r.tf_s())
    if key == "last_obj":
        return min(candidates, key=lambda r: float(r.last_obj))
    raise ValueError(f"Unsupported key: {key!r}")
