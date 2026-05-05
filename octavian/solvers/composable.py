"""Composable mission solver (ASSET backend).

This module compiles a Mission made of Phase objects into a single ASSET
OptimalControlProblem.

Scope (v0.1):
  - Two-body coast dynamics (TwoBodyECI)
  - Phase boundary constraints:
      * State (R,V) at Front/Back
      * Position (R) at Front/Back
  - Links:
      * continuous: (R,V,t)
      * impulsive: (R,t)
  - Variables:
      * ImpulsiveDeltaV at Front / Back
  - Objectives:
      * Minimize total Δv (default via Mission.objectives)
      * Optional Minimize time (via Mission.objectives)

This is the foundation for a general composable layer. Specialized solvers
(e.g., Lambert-aided seed searches) can be used as *guess builders* without
changing the compilation model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import asset_asrl as ast  # type: ignore
except Exception:  # pragma: no cover
    ast = None  # type: ignore

from typing import TYPE_CHECKING

from ..constraints import Constraint
from ..phase import Phase
from ..types import Maneuver
from ..variables import ImpulsiveDeltaV
from .options import SolverOptions
from .rendezvous import RendezvousResult  # reuse stable result type

if TYPE_CHECKING:  # pragma: no cover
    from ..mission import Mission
import contextlib

from ..astro.kepler import estimate_orbital_period_s, kepler_dense_guess, propagate_cartesian_rv
from ..astro.lambert import select_best_lambert_seed
from ..astro.types import as_vec3
from ..astro.units import default_units
from ..dynamics import TwoBodyECI
from ..time import normalize_time_bounds

if ast is not None:  # pragma: no cover
    vf = ast.VectorFunctions
    oc = ast.OptimalControl
    Tmodes = oc.TranscriptionModes
else:  # pragma: no cover
    vf = None  # type: ignore
    oc = None  # type: ignore
    Tmodes = None  # type: ignore


def _require_asset() -> None:
    if ast is None:
        raise RuntimeError(
            "asset_asrl is required for composable optimization solves. "
            "Install it (and its compiled dependencies) in your environment."
        )


@dataclass
class _PhaseBuild:
    ph: Phase
    asset_phase: Any
    t_bounds: tuple[float, float]
    # for maneuver bookkeeping
    index: int


def _has_impulsive_var(phase: Phase, where: str) -> bool:
    w = (where or "").strip().lower()
    loc = "Front" if w in ("front", "start", "initial", "t0") else "Back"
    # variables
    for v in getattr(phase, "variables", []) or []:
        if isinstance(v, ImpulsiveDeltaV) and getattr(v, "where", "") == loc:
            return True
    # legacy events
    try:
        if phase.has_impulse(loc):
            return True
    except Exception:
        pass
    return False


def _get_constraint(phase: Phase, kind: str, where: str) -> Constraint | None:
    loc = "Front" if where.lower().startswith("f") else "Back"
    for c in getattr(phase, "constraints", []) or []:
        if getattr(c, "kind", "") == kind and getattr(c, "where", "") == loc:
            return c
    return None


def _get_state_constraint(phase: Phase, where: str) -> Constraint | None:
    return _get_constraint(phase, kind="state", where=where)


def _get_position_constraint(phase: Phase, where: str) -> Constraint | None:
    return _get_constraint(phase, kind="position", where=where)


def _state_boundary_value(c: Constraint | None) -> Any:
    if c is None:
        return None
    return getattr(c, "value", {}).get("x")


def _state_groups(c: Constraint | None) -> tuple[str, ...]:
    if c is None:
        return tuple()
    groups = getattr(c, "value", {}).get("groups", ("R", "V"))
    return tuple(str(g) for g in groups)


def _position_boundary_value(c: Constraint | None) -> np.ndarray | None:
    if c is None:
        return None
    return np.asarray(c.value, dtype=float).reshape(3)


def _apply_orbital_element_constraint(asset_phase: Any, c: Constraint, mu_m3ps2: float) -> None:
    args = vf.Arguments(6)
    rvec, vvec = args.tolist([(0, 3), (3, 3)])
    hvec = rvec.cross(vvec)
    r = rvec.norm()
    v = vvec.norm()
    eps = 0.5 * (v**2) - float(mu_m3ps2) / r
    h2 = hvec.dot(hvec)

    kind = getattr(c, "kind", "")
    val = getattr(c, "value", {})
    where = getattr(c, "where", "Path")

    if kind == "semi_major_axis":
        target = float(val["a_m"])
        a_expr = -0.5 * float(mu_m3ps2) / eps
        tol = None if val["tol_m"] is None else float(val["tol_m"])
        if tol is None:
            asset_phase.addEqualCon(where, vf.stack([a_expr - target]), range(0, 6))
            return
        asset_phase.addInequalCon(where, vf.stack([a_expr - (target + tol)]), range(0, 6))
        asset_phase.addInequalCon(where, vf.stack([(target - tol) - a_expr]), range(0, 6))
        return

    if kind == "eccentricity":
        target = float(val["e"])
        e2_expr = 1.0 + (2.0 * eps * h2) / (float(mu_m3ps2) ** 2)
        tol = None if val["tol"] is None else float(val["tol"])
        if tol is None:
            asset_phase.addEqualCon(where, vf.stack([e2_expr - target**2]), range(0, 6))
            return
        asset_phase.addInequalCon(where, vf.stack([e2_expr - (target + tol) ** 2]), range(0, 6))
        asset_phase.addInequalCon(where, vf.stack([(target - tol) ** 2 - e2_expr]), range(0, 6))
        return

    if kind == "inclination_deg":
        target_deg = float(val["inc_deg"])
        hz_hat = hvec.normalized()[2]
        target_cos = float(np.cos(np.deg2rad(target_deg)))
        tol_deg = None if val["tol_deg"] is None else float(val["tol_deg"])
        if tol_deg is None:
            asset_phase.addEqualCon(where, vf.stack([hz_hat - target_cos]), range(0, 6))
            return
        upper_cos = float(np.cos(np.deg2rad(target_deg - tol_deg)))
        lower_cos = float(np.cos(np.deg2rad(target_deg + tol_deg)))
        asset_phase.addInequalCon(where, vf.stack([hz_hat - upper_cos]), range(0, 6))
        asset_phase.addInequalCon(where, vf.stack([lower_cos - hz_hat]), range(0, 6))
        return

    raise ValueError(f"Unsupported orbital-element constraint kind: {kind!r}")


def _objective_weights(mission: Mission) -> tuple[bool, float, bool, float]:
    # minimize_dv, w_dv, minimize_time, w_time
    minimize_dv = True
    w_dv = 1.0
    minimize_time = False
    w_time = float(getattr(mission, "w_time", 0.0) or 0.0)
    objs = list(getattr(mission, "objectives", []) or [])
    if objs:
        minimize_dv = any(getattr(o, "kind", "") == "delta_v" for o in objs)
        for o in objs:
            if getattr(o, "kind", "") == "delta_v":
                w_dv = float(getattr(o, "weight", 1.0) or 1.0)
                break
        for o in objs:
            if getattr(o, "kind", "") == "time":
                minimize_time = True
                w_time = float(getattr(o, "weight", w_time or 1.0))
                break
    return bool(minimize_dv), float(w_dv), bool(minimize_time), float(w_time)


def _build_guess_two_phase_precoast_transfer(
    p0: Phase,
    p1: Phase,
    *,
    mu: float,
    t1_bounds: tuple[float, float],
    tf_bounds: tuple[float, float],
    nsegs0: int,
    nsegs1: int,
    precoast_grid_size: int,
    limit_precoast_to_one_period: bool,
    lambert_grid_size: int,
    nrevs_to_try: Sequence[int],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Specialized guess builder for (precoast, transfer) with terminal position.

    This is used only for guesses; the compiled problem is still generic.
    """
    if p0.initial_state is None and _get_state_constraint(p0, "Front") is None:
        raise ValueError(
            "Precoast phase requires an initial state (phase.initial_state or State constraint at Front)."
        )

    x0 = p0.initial_state or _state_boundary_value(_get_state_constraint(p0, "Front"))  # type: ignore[union-attr]
    xf_state = _get_state_constraint(p1, "Back")
    xf_pos = _get_position_constraint(p1, "Back")

    if xf_state is None and xf_pos is None and p1.final_state is None:
        raise ValueError(
            "Transfer phase requires a terminal position (State/Position constraint or phase.final_state)."
        )

    xf_state_val = _state_boundary_value(xf_state)
    xf_pos_val = _position_boundary_value(xf_pos)
    rf = as_vec3(xf_state_val.r_m if xf_state_val is not None else (p1.final_state.r_m if p1.final_state is not None else xf_pos_val))  # type: ignore[arg-type]
    vf_target = as_vec3(
        xf_state_val.v_mps
        if xf_state_val is not None
        else (p1.final_state.v_mps if p1.final_state is not None else x0.v_mps)
    )

    t1min, t1max = t1_bounds
    tfmin, tfmax = tf_bounds

    t1min = float(t1min)
    t1max = float(t1max)
    tfmin = float(tfmin)
    tfmax = float(tfmax)

    # candidate t1 times
    n_t1 = max(int(precoast_grid_size), 2)
    t1_candidates = np.linspace(t1min, t1max, n_t1)

    if bool(limit_precoast_to_one_period):
        T0 = estimate_orbital_period_s(x0.r_m, x0.v_mps, mu)
        if T0 is not None:
            span = t1max - t1min
            if span > 1.5 * T0:
                t1_candidates = np.linspace(t1min, min(t1min + T0, t1max), n_t1)

    rv0 = np.hstack([as_vec3(x0.r_m), as_vec3(x0.v_mps)])

    best = None
    for t1_try in t1_candidates:
        try:
            rv1 = propagate_cartesian_rv(rv0, float(t1_try), mu)
        except Exception:
            continue

        r1 = rv1[0:3]
        v1_minus = rv1[3:6]

        dtmin = max(1.0, tfmin - float(t1_try))
        dtmax = max(dtmin + 1.0, tfmax - float(t1_try))
        if dtmax <= dtmin:
            continue

        seed = select_best_lambert_seed(
            r0_m=as_vec3(r1),
            rf_m=rf,
            v0_mps=as_vec3(v1_minus),
            vf_mps=vf_target,
            mu_m3ps2=mu,
            tmin_s=float(dtmin),
            tmax_s=float(dtmax),
            n_tofs=int(lambert_grid_size),
            nrevs=tuple(int(n) for n in nrevs_to_try),
        )

        dv1 = as_vec3(seed.v1_mps) - as_vec3(v1_minus)
        dv2 = vf_target - as_vec3(seed.v2_mps)
        score = float(np.linalg.norm(dv1) + np.linalg.norm(dv2))
        if best is None or score < best["score"]:
            best = {"t1": float(t1_try), "rv1": rv1, "seed": seed, "score": score}

    if best is None:
        # fallback: simple midpoint propagation + straight guess
        t1_guess = 0.5 * (t1min + t1max)
        tf_guess = 0.5 * (tfmin + tfmax)
        ig0 = kepler_dense_guess(
            r0_m=as_vec3(x0.r_m),
            v0_mps=as_vec3(x0.v_mps),
            t0_s=0.0,
            tf_s=t1_guess,
            npts=nsegs0 + 1,
            mu_m3ps2=mu,
        )
        # start phase1 at end of ig0 with same velocity
        rv1 = np.hstack([ig0[-1][0:3], ig0[-1][3:6]])
        ig1 = kepler_dense_guess(
            r0_m=as_vec3(rv1[0:3]),
            v0_mps=as_vec3(rv1[3:6]),
            t0_s=t1_guess,
            tf_s=tf_guess,
            npts=nsegs1 + 1,
            mu_m3ps2=mu,
        )
        return ig0, ig1

    t1_guess = float(best["t1"])
    seed = best["seed"]
    rv1_guess = np.asarray(best["rv1"], dtype=float).reshape(6)
    r1_guess = rv1_guess[0:3]
    v1_plus_guess = as_vec3(seed.v1_mps)
    tf_guess = float(t1_guess + seed.tof_s)

    ig0 = kepler_dense_guess(
        r0_m=as_vec3(x0.r_m),
        v0_mps=as_vec3(x0.v_mps),
        t0_s=0.0,
        tf_s=t1_guess,
        npts=nsegs0 + 1,
        mu_m3ps2=mu,
    )
    ig1 = kepler_dense_guess(
        r0_m=as_vec3(r1_guess),
        v0_mps=v1_plus_guess,
        t0_s=t1_guess,
        tf_s=tf_guess,
        npts=nsegs1 + 1,
        mu_m3ps2=mu,
    )
    return ig0, ig1


def _build_guess_single_phase_terminal_position(
    phase: Phase,
    *,
    mu: float,
    tf_bounds: tuple[float, float],
    nsegs: int,
    lambert_grid_size: int,
    nrevs_to_try: Sequence[int],
) -> tuple[list[np.ndarray], dict[str, float | int | bool]]:
    """Build a Lambert-seeded guess for a single transfer phase.

    This mirrors the quick rendezvous solver setup:
    the phase starts from the desired initial velocity, uses a Lambert seed to
    reach the terminal position, and lets boundary delta-v objectives account
    for the impulsive mismatch.
    """
    x0 = phase.initial_state or _state_boundary_value(_get_state_constraint(phase, "Front"))  # type: ignore[union-attr]
    xf_state = _get_state_constraint(phase, "Back")
    xf_pos = _get_position_constraint(phase, "Back")

    if x0 is None:
        raise ValueError(
            "Single-phase Lambert seeding requires an initial state or State constraint at Front."
        )

    xf_state_val = _state_boundary_value(xf_state)
    xf_pos_val = _position_boundary_value(xf_pos)

    if xf_state_val is None and xf_pos_val is None and phase.final_state is None:
        raise ValueError(
            "Single-phase Lambert seeding requires a terminal position at Back."
        )

    rf = as_vec3(
        xf_state_val.r_m
        if xf_state_val is not None
        else (phase.final_state.r_m if phase.final_state is not None else xf_pos_val)
    )
    vf_target = as_vec3(
        xf_state_val.v_mps
        if xf_state_val is not None
        else (phase.final_state.v_mps if phase.final_state is not None else x0.v_mps)
    )

    tfmin, tfmax = map(float, tf_bounds)
    seed = select_best_lambert_seed(
        r0_m=as_vec3(x0.r_m),
        rf_m=rf,
        v0_mps=as_vec3(x0.v_mps),
        vf_mps=vf_target,
        mu_m3ps2=mu,
        tmin_s=tfmin,
        tmax_s=tfmax,
        n_tofs=int(lambert_grid_size),
        nrevs=tuple(int(n) for n in nrevs_to_try),
    )

    tf_guess = min(max(float(seed.tof_s), tfmin), tfmax)
    ig = kepler_dense_guess(
        r0_m=as_vec3(x0.r_m),
        v0_mps=as_vec3(seed.v1_mps),
        t0_s=0.0,
        tf_s=tf_guess,
        npts=nsegs + 1,
        mu_m3ps2=mu,
    )

    return ig, {
        "seed_tof_s": float(seed.tof_s),
        "seed_longway": bool(seed.longway),
        "seed_nrev": int(seed.nrev),
        "seed_rightbranch": bool(seed.rightbranch),
        "seed_total_dv_mps": float(seed.total_dv_mps),
    }


def _phase_terminal_target(phase: Phase) -> tuple[np.ndarray, np.ndarray | None] | None:
    """Return the phase's terminal position target and optional velocity target."""
    xf_state = _get_state_constraint(phase, "Back")
    xf_pos = _get_position_constraint(phase, "Back")
    xf_state_val = _state_boundary_value(xf_state)
    xf_pos_val = _position_boundary_value(xf_pos)

    if xf_state_val is not None:
        return as_vec3(xf_state_val.r_m), as_vec3(xf_state_val.v_mps)
    if phase.final_state is not None:
        return as_vec3(phase.final_state.r_m), as_vec3(phase.final_state.v_mps)
    if xf_pos_val is not None:
        return as_vec3(xf_pos_val), None
    return None


def _find_downstream_terminal_anchor(
    phases: Sequence[Phase], start_idx: int
) -> tuple[int, np.ndarray, np.ndarray | None] | None:
    """Find the nearest downstream phase with an explicit terminal position target."""
    for idx in range(start_idx, len(phases)):
        target = _phase_terminal_target(phases[idx])
        if target is not None:
            r_target, v_target = target
            return idx, r_target, v_target
    return None


def _continuous_chain_end(phases: Sequence[Phase], start_idx: int) -> int:
    """Return the end index of the continuous-link chain starting at ``start_idx``."""
    end_idx = start_idx
    for idx in range(start_idx + 1, len(phases)):
        ph = phases[idx]
        if ph.previous is None:
            break
        link_kind = (ph.link.kind if ph.link is not None else "continuous").lower()
        if link_kind != "continuous":
            break
        end_idx = idx
    return end_idx


def _find_chain_terminal_anchor(
    phases: Sequence[Phase], start_idx: int, end_idx: int
) -> tuple[int, np.ndarray, np.ndarray | None] | None:
    """Find the furthest explicit terminal target inside a continuous chain."""
    for idx in range(end_idx, start_idx - 1, -1):
        target = _phase_terminal_target(phases[idx])
        if target is not None:
            r_target, v_target = target
            return idx, r_target, v_target
    return None


def _split_continuous_chain_times(
    *,
    start_idx: int,
    end_idx: int,
    abs_bounds: Sequence[tuple[float, float] | None],
    t_start: float,
    tf_total: float,
) -> list[float] | None:
    """Split a chain end time into monotone per-phase end times."""
    nph = end_idx - start_idx + 1
    duration = float(tf_total) - float(t_start)
    if duration <= 0.0:
        return None

    ends: list[float] = []
    prev = float(t_start)
    for local_idx, phase_idx in enumerate(range(start_idx, end_idx + 1), start=1):
        raw = float(t_start) + duration * (local_idx / nph)
        if phase_idx == end_idx:
            raw = float(tf_total)

        bounds = abs_bounds[phase_idx]
        lo = max(prev + 0.1, float(bounds[0])) if bounds is not None else prev + 0.1
        hi = float(bounds[1]) if bounds is not None else float(tf_total)
        if phase_idx == end_idx:
            lo = max(lo, float(tf_total))
            hi = min(hi, float(tf_total))

        end_t = min(max(raw, lo), hi)
        if end_t < lo - 1e-9 or end_t > hi + 1e-9:
            return None

        ends.append(float(end_t))
        prev = float(end_t)

    return ends


def _build_continuous_chain_guesses(
    *,
    phases: Sequence[Phase],
    start_idx: int,
    abs_bounds: Sequence[tuple[float, float] | None],
    t_start: float,
    r_start_m: np.ndarray,
    v_start_mps: np.ndarray,
    mu_m3ps2: float,
    nsegs0: int,
    nsegs1: int,
    lambert_grid_size: int,
    nrevs_to_try: Sequence[int],
) -> tuple[dict[int, list[np.ndarray]], dict[int, dict[str, float | int | bool | str]]]:
    """Build one Lambert arc and split it across continuous coast phases."""
    end_idx = _continuous_chain_end(phases, start_idx)
    if end_idx <= start_idx:
        return {}, {}

    anchor = _find_chain_terminal_anchor(phases, start_idx, end_idx)
    if anchor is None:
        return {}, {}

    anchor_idx, r_target, v_target = anchor
    anchor_bounds = abs_bounds[anchor_idx]
    if anchor_bounds is None:
        return {}, {}

    t_anchor_min, t_anchor_max = map(float, anchor_bounds)
    remain_min = max(1.0, t_anchor_min - float(t_start))
    remain_max = max(remain_min + 1.0, t_anchor_max - float(t_start))
    v_lambert_target = as_vec3(v_target) if v_target is not None else as_vec3(v_start_mps)

    try:
        seed = select_best_lambert_seed(
            r0_m=as_vec3(r_start_m),
            rf_m=as_vec3(r_target),
            v0_mps=as_vec3(v_start_mps),
            vf_mps=v_lambert_target,
            mu_m3ps2=float(mu_m3ps2),
            tmin_s=float(remain_min),
            tmax_s=float(remain_max),
            n_tofs=int(lambert_grid_size),
            nrevs=tuple(int(n) for n in nrevs_to_try),
        )
    except Exception:
        return {}, {}

    tf_total = float(t_start + min(max(float(seed.tof_s), remain_min), remain_max))
    split_times = _split_continuous_chain_times(
        start_idx=start_idx,
        end_idx=anchor_idx,
        abs_bounds=abs_bounds,
        t_start=float(t_start),
        tf_total=float(tf_total),
    )
    if split_times is None:
        return {}, {}

    guesses: dict[int, list[np.ndarray]] = {}
    infos: dict[int, dict[str, float | int | bool | str]] = {}

    r_curr = as_vec3(r_start_m)
    v_curr = as_vec3(seed.v1_mps)
    t_curr = float(t_start)

    for phase_idx, t_end in zip(range(start_idx, anchor_idx + 1), split_times, strict=True):
        nsegs = int(nsegs0 if phase_idx == 0 else nsegs1)
        ig = kepler_dense_guess(
            r0_m=as_vec3(r_curr),
            v0_mps=as_vec3(v_curr),
            t0_s=float(t_curr),
            tf_s=float(t_end),
            npts=nsegs + 1,
            mu_m3ps2=float(mu_m3ps2),
        )
        guesses[phase_idx] = ig
        infos[phase_idx] = {
            "guess_kind": "lambert_continuous_chain",
            "guess_anchor_phase_index": int(anchor_idx),
            "guess_anchor_phase_name": phases[anchor_idx].name,
            "guess_chain_start_index": int(start_idx),
            "guess_chain_end_index": int(anchor_idx),
            "guess_phase_tf_s": float(t_end),
            "seed_tof_s": float(seed.tof_s),
            "seed_longway": bool(seed.longway),
            "seed_nrev": int(seed.nrev),
            "seed_rightbranch": bool(seed.rightbranch),
            "seed_total_dv_mps": float(seed.total_dv_mps),
        }
        last = np.asarray(ig[-1], dtype=float)
        r_curr = as_vec3(last[0:3])
        v_curr = as_vec3(last[3:6])
        t_curr = float(last[6])

    return guesses, infos


def _build_lambert_guided_phase_guess(
    *,
    phase: Phase,
    phase_idx: int,
    phases: Sequence[Phase],
    abs_bounds: Sequence[tuple[float, float] | None],
    t_start: float,
    r_start_m: np.ndarray,
    v_start_mps: np.ndarray,
    mu_m3ps2: float,
    nsegs: int,
    lambert_grid_size: int,
    nrevs_to_try: Sequence[int],
) -> tuple[list[np.ndarray], dict[str, float | int | bool]] | None:
    """Build a Lambert-guided guess toward the nearest downstream terminal target.

    The Lambert arc is computed to the nearest known downstream terminal target.
    For intermediate phases, the current phase guess follows the beginning of that
    Lambert arc for this phase's own time span.
    """
    anchor = _find_downstream_terminal_anchor(phases, phase_idx)
    if anchor is None:
        return None

    anchor_idx, r_target, v_target = anchor
    anchor_bounds = abs_bounds[anchor_idx]
    if anchor_bounds is None:
        return None

    t_anchor_min, t_anchor_max = map(float, anchor_bounds)
    remain_min = max(1.0, t_anchor_min - float(t_start))
    remain_max = max(remain_min + 1.0, t_anchor_max - float(t_start))

    v_lambert_target = as_vec3(v_target) if v_target is not None else as_vec3(v_start_mps)

    try:
        seed = select_best_lambert_seed(
            r0_m=as_vec3(r_start_m),
            rf_m=as_vec3(r_target),
            v0_mps=as_vec3(v_start_mps),
            vf_mps=v_lambert_target,
            mu_m3ps2=float(mu_m3ps2),
            tmin_s=float(remain_min),
            tmax_s=float(remain_max),
            n_tofs=int(lambert_grid_size),
            nrevs=tuple(int(n) for n in nrevs_to_try),
        )
    except Exception:
        return None

    tf_guess = _fallback_phase_tf_guess(
        phase=phase,
        bounds_abs=abs_bounds[phase_idx],
        t_start=float(t_start),
        r_start_m=as_vec3(r_start_m),
        v_start_mps=as_vec3(v_start_mps),
        mu_m3ps2=float(mu_m3ps2),
    )
    tf_guess = max(float(tf_guess), float(t_start) + 0.1)

    ig = kepler_dense_guess(
        r0_m=as_vec3(r_start_m),
        v0_mps=as_vec3(seed.v1_mps),
        t0_s=float(t_start),
        tf_s=float(tf_guess),
        npts=int(nsegs) + 1,
        mu_m3ps2=float(mu_m3ps2),
    )
    return ig, {
        "guess_kind": "lambert_downstream",
        "guess_anchor_phase_index": int(anchor_idx),
        "guess_anchor_phase_name": phases[anchor_idx].name,
        "seed_tof_s": float(seed.tof_s),
        "seed_longway": bool(seed.longway),
        "seed_nrev": int(seed.nrev),
        "seed_rightbranch": bool(seed.rightbranch),
        "seed_total_dv_mps": float(seed.total_dv_mps),
    }


def _boundary_velocity_target(phase: Phase, where: str) -> np.ndarray | None:
    """Return a boundary velocity target from State constraint or phase state."""
    st = _get_state_constraint(phase, where)
    if st is not None:
        st_val = _state_boundary_value(st)
        if st_val is not None:
            return as_vec3(st_val.v_mps)
    if where.lower().startswith("f") and phase.initial_state is not None:
        return as_vec3(phase.initial_state.v_mps)
    if where.lower().startswith("b") and phase.final_state is not None:
        return as_vec3(phase.final_state.v_mps)
    return None


def _explicit_boundary_velocity_target(phase: Phase, where: str) -> np.ndarray | None:
    """Return an explicitly constrained boundary velocity target from State constraints."""
    st = _get_state_constraint(phase, where)
    if st is None:
        return None
    if "V" not in _state_groups(st):
        return None
    st_val = _state_boundary_value(st)
    if st_val is None:
        return None
    return as_vec3(st_val.v_mps)


def _build_front_impulse_velocity_targets(phases: Sequence[Phase]) -> dict[int, np.ndarray]:
    """Seed only the first front-link impulse; later front links default to zero impulse.

    This keeps multi-link guesses close to the two-burn structure:
    - first front impulsive link: nonzero seed (midpoint in velocity space)
    - additional front impulsive links: zero seed (inherit incoming velocity)
    """
    if not phases:
        return {}

    v_start = _boundary_velocity_target(phases[0], "Front")
    v_end = _boundary_velocity_target(phases[-1], "Back")
    if v_start is None or v_end is None:
        return {}

    front_impulse_idxs: list[int] = []
    for idx, ph in enumerate(phases):
        if idx == 0 or ph.previous is None:
            continue
        link_kind = (ph.link.kind if ph.link is not None else "continuous").lower()
        if link_kind != "continuous" and _has_impulsive_var(ph, "Front"):
            front_impulse_idxs.append(idx)

    if not front_impulse_idxs:
        return {}

    out: dict[int, np.ndarray] = {}
    first_idx = int(front_impulse_idxs[0])
    out[first_idx] = 0.5 * as_vec3(v_start) + 0.5 * as_vec3(v_end)
    return out


def _fallback_phase_tf_guess(
    *,
    phase: Phase,
    bounds_abs: tuple[float, float] | None,
    t_start: float,
    r_start_m: np.ndarray,
    v_start_mps: np.ndarray,
    mu_m3ps2: float,
) -> float:
    """Choose a practical fallback Back-time guess for a phase."""
    if bounds_abs is None:
        return float(t_start + 3600.0)

    tmin_abs, tmax_abs = map(float, bounds_abs)

    if phase.tof_is_relative and phase.tof_bounds_s is not None:
        dmin, dmax = map(float, phase.tof_bounds_s)
        dmid = 0.5 * (dmin + dmax)

        # Keep relative-duration fallback physically moderate when wide bounds are provided.
        T = estimate_orbital_period_s(r_start_m, v_start_mps, mu_m3ps2)
        if T is not None:
            dcap = max(dmin, min(dmax, float(T)))
            dguess = min(dmid, dcap)
        else:
            dguess = dmid

        dguess = max(dmin, min(dguess, dmax))
        tf_guess = float(t_start + dguess)
    else:
        tf_guess = float(0.5 * (tmin_abs + tmax_abs))

    tf_guess = min(max(tf_guess, tmin_abs), tmax_abs)
    tf_guess = max(tf_guess, float(t_start) + 0.1)
    return float(tf_guess)


def solve_composable_mission(
    mission: Mission,
    *,
    options: SolverOptions | None = None,
) -> RendezvousResult:
    """Solve a composable mission via ASSET compilation."""
    _require_asset()

    phases = list(mission.phases)
    if not phases:
        raise ValueError("Mission has no phases")

    # Currently: only coast phases supported
    for ph in phases:
        if (ph.mode or "").lower() not in ("coast", "transfer", "rendezvous"):
            raise NotImplementedError(
                f"Composable solver currently supports only coast-like phases. Got mode={ph.mode!r}"
            )

    minimize_dv, w_dv, minimize_time, w_time = _objective_weights(mission)

    # Normalize time bounds (absolute Back-time bounds for each phase).
    abs_bounds = normalize_time_bounds(phases)

    # Units / scaling: use default_units() with a tiny dummy spec carrying x0/xf.
    first = phases[0]
    last = phases[-1]
    mu = float(first.dynamics.mu_m3ps2)  # type: ignore[union-attr]

    x0_for_units = (
        first.initial_state
        or _state_boundary_value(_get_state_constraint(first, "Front"))
        or _state_boundary_value(_get_state_constraint(first, "Back"))
    )
    xf_for_units = (
        last.final_state
        or _state_boundary_value(_get_state_constraint(last, "Back"))
        or _state_boundary_value(_get_state_constraint(last, "Front"))
    )

    if x0_for_units is None or xf_for_units is None:
        raise ValueError(
            "Composable mission needs boundary State information (x0 and xf) to choose scaling units."
        )

    class _UnitSpec:
        def __init__(self, x0, xf, mu_m3ps2):
            self.x0 = x0
            self.xf = xf
            self.mu_m3ps2 = mu_m3ps2
            # allow mission-level overrides in the future
            self.r_unit_m = None
            self.v_unit_mps = None
            self.t_unit_s = None
            # approximate time bounds from last phase if available
            last_bounds = abs_bounds[-1]
            self.tf_bounds_s = last_bounds if last_bounds is not None else (0.0, 10.0)

    r_unit, v_unit, t_unit = default_units(_UnitSpec(x0_for_units, xf_for_units, mu))

    ode = TwoBodyECI(mu_m3ps2=mu)

    # Build guesses: handle common 2-phase precoast+transfer case for better robustness
    nsegs0 = int(getattr(mission, "mesh_nsegs_precoast", 30))
    nsegs1 = int(getattr(mission, "mesh_nsegs_transfer", 60))

    guesses: dict[int, list[np.ndarray]] = {}
    guess_info: dict[int, dict[str, float | int | bool | str]] = {}

    if len(phases) == 1:
        p0 = phases[0]
        tf_bounds = abs_bounds[0] or (600.0, 7200.0)
        ig0, info0 = _build_guess_single_phase_terminal_position(
            p0,
            mu=mu,
            tf_bounds=tf_bounds,
            nsegs=nsegs1,
            lambert_grid_size=int(getattr(mission, "lambert_grid_size", 60)),
            nrevs_to_try=tuple(int(x) for x in getattr(mission, "nrevs_to_try", (0, 1))),
        )
        guesses[0] = ig0
        guess_info[0] = info0

    elif (
        len(phases) == 2
        and (phases[0].mode or "").lower() == "coast"
        and phases[1].previous is not None
        and (phases[1].link.kind if phases[1].link is not None else "continuous").lower()
        != "continuous"
    ):
        p0, p1 = phases
        t1_bounds = abs_bounds[0] or (0.0, 1800.0)
        tf_bounds = abs_bounds[1] or (600.0, 7200.0)
        ig0, ig1 = _build_guess_two_phase_precoast_transfer(
            p0,
            p1,
            mu=mu,
            t1_bounds=t1_bounds,
            tf_bounds=tf_bounds,
            nsegs0=nsegs0,
            nsegs1=nsegs1,
            precoast_grid_size=int(getattr(mission, "precoast_grid_size", 10)),
            limit_precoast_to_one_period=bool(
                getattr(mission, "limit_precoast_to_one_period", True)
            ),
            lambert_grid_size=int(getattr(mission, "lambert_grid_size", 60)),
            nrevs_to_try=tuple(int(x) for x in getattr(mission, "nrevs_to_try", (0, 1))),
        )
        guesses[0] = ig0
        guesses[1] = ig1

    front_impulse_v_targets = _build_front_impulse_velocity_targets(phases)

    # Compile phases
    ocp = oc.OptimalControlProblem()
    built: list[_PhaseBuild] = []

    for idx, ph in enumerate(phases):
        # determine guess
        if idx in guesses:
            ig = guesses[idx]
            # infer nsegs from guess length - 1
            nsegs = len(ig) - 1
        else:
            nsegs = nsegs0 if idx == 0 else nsegs1
            if idx == 0:
                x0_chain = ph.initial_state or _state_boundary_value(_get_state_constraint(ph, "Front"))
                if x0_chain is None:
                    raise ValueError(
                        "First phase must have an initial_state or State constraint at Front."
                    )
                chain_t_start = 0.0
                chain_r_start = as_vec3(x0_chain.r_m)
                chain_v_start = as_vec3(x0_chain.v_mps)
            else:
                prev_guess = guesses.get(idx - 1)
                if prev_guess is None:
                    prev_guess = np.asarray(built[-1].asset_phase.returnTraj(), dtype=float).tolist()  # type: ignore
                prev_last = np.asarray(prev_guess[-1], dtype=float)
                chain_t_start = float(prev_last[6])
                chain_r_start = as_vec3(prev_last[0:3])
                chain_v_start = as_vec3(prev_last[3:6])

            chain_guesses, chain_info = _build_continuous_chain_guesses(
                phases=phases,
                start_idx=idx,
                abs_bounds=abs_bounds,
                t_start=chain_t_start,
                r_start_m=chain_r_start,
                v_start_mps=chain_v_start,
                mu_m3ps2=mu,
                nsegs0=nsegs0,
                nsegs1=nsegs1,
                lambert_grid_size=int(getattr(mission, "lambert_grid_size", 60)),
                nrevs_to_try=tuple(int(x) for x in getattr(mission, "nrevs_to_try", (0, 1))),
            )
            if chain_guesses:
                guesses.update(chain_guesses)
                guess_info.update(chain_info)
                ig = guesses[idx]
                nsegs = len(ig) - 1
                asset_phase = ode.phase(Tmodes.LGL3, ig, int(nsegs))
                ocp.addPhase(asset_phase)

                built.append(
                    _PhaseBuild(
                        ph=ph,
                        asset_phase=asset_phase,
                        t_bounds=tuple(abs_bounds[idx] or (0.0, 1.0)),
                        index=idx,
                    )
                )
                continue
            # fallback midpoint guess using available initial state
            if idx == 0:
                x0 = x0_chain
                lambert_guess = _build_lambert_guided_phase_guess(
                    phase=ph,
                    phase_idx=idx,
                    phases=phases,
                    abs_bounds=abs_bounds,
                    t_start=0.0,
                    r_start_m=as_vec3(x0.r_m),
                    v_start_mps=as_vec3(x0.v_mps),
                    mu_m3ps2=mu,
                    nsegs=nsegs,
                    lambert_grid_size=int(getattr(mission, "lambert_grid_size", 60)),
                    nrevs_to_try=tuple(int(x) for x in getattr(mission, "nrevs_to_try", (0, 1))),
                )
                if lambert_guess is not None:
                    ig, info = lambert_guess
                    guess_info[idx] = info
                else:
                    bounds = abs_bounds[idx]
                    tf_guess = _fallback_phase_tf_guess(
                        phase=ph,
                        bounds_abs=bounds,
                        t_start=0.0,
                        r_start_m=as_vec3(x0.r_m),
                        v_start_mps=as_vec3(x0.v_mps),
                        mu_m3ps2=mu,
                    )
                    ig = kepler_dense_guess(
                        r0_m=as_vec3(x0.r_m),
                        v0_mps=as_vec3(x0.v_mps),
                        t0_s=0.0,
                        tf_s=tf_guess,
                        npts=nsegs + 1,
                        mu_m3ps2=mu,
                    )
            else:
                # start from previous guess end
                prev_guess = guesses.get(idx - 1)
                if prev_guess is None:
                    prev_guess = np.asarray(built[-1].asset_phase.returnTraj(), dtype=float).tolist()  # type: ignore
                rv_start = np.asarray(prev_guess[-1])[0:6]
                t_start = float(np.asarray(prev_guess[-1])[6])
                bounds = abs_bounds[idx]
                tf_guess = _fallback_phase_tf_guess(
                    phase=ph,
                    bounds_abs=bounds,
                    t_start=t_start,
                    r_start_m=as_vec3(rv_start[0:3]),
                    v_start_mps=as_vec3(rv_start[3:6]),
                    mu_m3ps2=mu,
                )
                v_start_guess = as_vec3(rv_start[3:6])
                if idx in front_impulse_v_targets:
                    v_start_guess = as_vec3(front_impulse_v_targets[idx])
                lambert_guess = _build_lambert_guided_phase_guess(
                    phase=ph,
                    phase_idx=idx,
                    phases=phases,
                    abs_bounds=abs_bounds,
                    t_start=t_start,
                    r_start_m=as_vec3(rv_start[0:3]),
                    v_start_mps=v_start_guess,
                    mu_m3ps2=mu,
                    nsegs=nsegs,
                    lambert_grid_size=int(getattr(mission, "lambert_grid_size", 60)),
                    nrevs_to_try=tuple(int(x) for x in getattr(mission, "nrevs_to_try", (0, 1))),
                )
                if lambert_guess is not None:
                    ig, info = lambert_guess
                    guess_info[idx] = info
                else:
                    ig = kepler_dense_guess(
                        r0_m=as_vec3(rv_start[0:3]),
                        v0_mps=v_start_guess,
                        t0_s=t_start,
                        tf_s=tf_guess,
                        npts=nsegs + 1,
                        mu_m3ps2=mu,
                    )

        asset_phase = ode.phase(Tmodes.LGL3, ig, int(nsegs))
        ocp.addPhase(asset_phase)

        built.append(
            _PhaseBuild(
                ph=ph,
                asset_phase=asset_phase,
                t_bounds=tuple(abs_bounds[idx] or (0.0, 1.0)),
                index=idx,
            )
        )

    # Set scaling & mesh options
    opts = options or SolverOptions()
    ocp.optimizer.PrintLevel = int(opts.print_level)
    ocp.optimizer.MaxLSIters = int(opts.max_ls_iters)
    ocp.optimizer.set_QPOrderingMode(str(opts.qp_ordering_mode))

    for b in built:
        b.asset_phase.setAutoScaling(bool(opts.enable_auto_scaling))
        b.asset_phase.setUnits(R=r_unit, V=v_unit, t=t_unit)
        b.asset_phase.setAdaptiveMesh(bool(opts.enable_adaptive_mesh))

    ocp.setAutoScaling(True, True)
    ocp.setAdaptiveMesh(True)
    ocp.PrintMeshInfo = False

    # Apply constraints and time bounds per phase
    for b in built:
        ph = b.ph
        ap = b.asset_phase

        # First phase front time fixed at 0 unless user provides otherwise
        if b.index == 0:
            with contextlib.suppress(Exception):
                ap.addBoundaryValue("Front", ["t"], np.asarray([0.0], dtype=float))

        # Apply State/Position boundary constraints with impulsive-variable override logic
        for loc in ("Front", "Back"):
            st = _get_state_constraint(ph, loc)
            pos = _get_position_constraint(ph, loc)

            # Position constraint always applied (if present)
            if pos is not None:
                ap.addBoundaryValue(
                    loc, ["R"], np.asarray(as_vec3(_position_boundary_value(pos)), dtype=float)
                )

            if st is not None:
                st_val = _state_boundary_value(st)
                groups = _state_groups(st)

                # If an impulsive Δv is declared at this boundary, we drop V from the constraint
                if _has_impulsive_var(ph, loc) and "V" in groups:
                    groups = tuple(g for g in groups if g != "V")

                # build value vector for groups
                vals = []
                use_groups = []
                for g in groups:
                    if g == "R":
                        vals.extend(list(as_vec3(st_val.r_m)))
                        use_groups.append("R")
                    elif g == "V":
                        vals.extend(list(as_vec3(st_val.v_mps)))
                        use_groups.append("V")
                    elif g in ("t", "time"):
                        # if user requested fixing time explicitly, we honor it if phase has epoch semantics (not yet)
                        # for now ignore unless first phase front (already fixed)
                        use_groups.append("t")
                        vals.append(float(0.0 if (b.index == 0 and loc == "Front") else np.nan))
                    else:
                        raise ValueError(f"Unsupported State.groups element: {g!r}")

                # Only apply if we have something concrete
                if use_groups:
                    if "t" in use_groups and not np.isfinite(vals[-1]):
                        # if time is nan we skip setting time
                        use_groups = [g for g in use_groups if g != "t"]
                        vals = vals[:-1]
                    if use_groups:
                        ap.addBoundaryValue(loc, use_groups, np.asarray(vals, dtype=float))

        # Path constraints (e.g., min radius)
        for c in getattr(ph, "constraints", []) or []:
            if getattr(c, "kind", "") == "min_radius":
                rmin = float(c.value)
                # rmin_nd = rmin / float(r_unit)
                loc = (
                    "Path" if getattr(c, "where", "Path") == "Path" else getattr(c, "where", "Path")
                )
                # "R" is in scaled units here, so use a scaled bound with AutoScale=1.
                ap.addLowerNormBound(loc, "R", rmin)
                # ap.addInequalCon(loc, rmin*rmin - vf.Arguments(3).squared_norm(), ["R"])

                # breakpoint()
            elif getattr(c, "kind", "") in {"semi_major_axis", "eccentricity", "inclination_deg"}:
                _apply_orbital_element_constraint(ap, c, mu)

        # Time bounds: normalize tof_bounds_s to absolute Back-time bounds.
        bounds = abs_bounds[b.index]
        if bounds is not None:
            tmin, tmax = map(float, bounds)
            try:
                ap.addLUVarBound("Back", "time", tmin, tmax)
            except Exception:
                ap.addLUVarBound("Back", 6, tmin, tmax)
            ap.addLowerDeltaTimeBound(0.1)

    # Apply links and link objectives
    for b in built:
        ph = b.ph
        if ph.previous is None:
            continue

        # Find previous compiled phase
        prev_idx = None
        for bb in built:
            if bb.ph is ph.previous:
                prev_idx = bb.index
                prev_ap = bb.asset_phase
                break
        if prev_idx is None:
            raise ValueError(f"Phase {ph.name!r} references previous phase not in mission.")

        ap = b.asset_phase
        link_kind = (ph.link.kind if ph.link is not None else "continuous").lower()

        if link_kind == "continuous":
            ocp.addForwardLinkEqualCon(prev_ap, ap, ["R", "V", "t"])
        else:
            ocp.addForwardLinkEqualCon(prev_ap, ap, ["R", "t"])

        # If an impulsive Δv is declared at this phase Front and link is not continuous,
        # add a link objective ||V_plus - V_minus||
        if minimize_dv and link_kind != "continuous" and _has_impulsive_var(ph, "Front"):
            a = vf.Arguments(6)
            v_minus = a.head(3)
            v_plus = a.segment(3, 3)
            dv = v_plus - v_minus
            dvmag = vf.sqrt(dv.dot(dv))
            ocp.addLinkObjective(
                float(w_dv) * dvmag,
                prev_ap,
                "Back",
                [3, 4, 5],
                [],
                [],
                ap,
                "Front",
                [3, 4, 5],
                [],
                [],
                [],
                AutoScale=1.0 / float(v_unit),
            )

    # Boundary Δv objectives (mission start and terminal)
    if minimize_dv:
        # Start: if first phase has impulsive Δv at Front, penalize relative to desired initial velocity
        first_ph = built[0].ph
        first_ap = built[0].asset_phase
        if _has_impulsive_var(first_ph, "Front"):
            st = _get_state_constraint(first_ph, "Front")
            if st is None and first_ph.initial_state is None:
                raise ValueError(
                    "ImpulsiveDeltaV at mission start requires a desired initial velocity (State constraint or initial_state)."
                )
            st_val = _state_boundary_value(st)
            v_target = as_vec3(st_val.v_mps if st_val is not None else first_ph.initial_state.v_mps)  # type: ignore[union-attr]
            a0 = vf.Arguments(3)
            dv0 = vf.sqrt((a0 - v_target).dot(a0 - v_target))
            first_ap.addStateObjective(
                "Front", float(w_dv) * dv0, [3, 4, 5], [], [], AutoScale=1.0 / float(v_unit)
            )

        # Terminal: if last phase has impulsive Δv at Back, penalize relative to desired terminal velocity
        last_ph = built[-1].ph
        last_ap = built[-1].asset_phase
        if _has_impulsive_var(last_ph, "Back"):
            v_target = _explicit_boundary_velocity_target(last_ph, "Back")
            if v_target is not None:
                b0 = vf.Arguments(3)
                dvf = vf.sqrt((v_target - b0).dot(v_target - b0))
                last_ap.addStateObjective(
                    "Back", float(w_dv) * dvf, [3, 4, 5], [], [], AutoScale=1.0 / float(v_unit)
                )

    # Time objective: minimize final time at last phase Back
    if minimize_time and float(w_time) != 0.0:
        last_ap = built[-1].asset_phase
        at = vf.Arguments(1).tolist()[0]
        last_ap.addStateObjective(
            "Back", float(w_time) * at, [6], [], [], AutoScale=1.0 / float(t_unit)
        )

    # Solve
    converged = (
        ocp.solve_optimize_solve() if hasattr(ocp, "solve_optimize_solve") else ocp.optimize_solve()
    )
    # converged = True

    # Extract trajectory (stitch)
    trajs = [np.asarray(b.asset_phase.returnTraj(), dtype=float) for b in built]
    traj = trajs[0]
    for t in trajs[1:]:
        traj = np.vstack([traj, t[1:, :]])

    # Maneuver bookkeeping: link dv and terminal dv if requested
    maneuvers: list[Maneuver] = []
    # start maneuver (mission start boundary)
    if built and _has_impulsive_var(built[0].ph, "Front"):
        st0 = _get_state_constraint(built[0].ph, "Front")
        st0_val = _state_boundary_value(st0)
        v_target0 = as_vec3(st0_val.v_mps if st0_val is not None else built[0].ph.initial_state.v_mps)  # type: ignore[union-attr]
        v_start = trajs[0][0, 3:6]
        dv0 = v_start - v_target0
        maneuvers.append(
            Maneuver(r_m=trajs[0][0, 0:3], t_s=float(trajs[0][0, 6]), dv_mps=dv0, name="Δv (start)")
        )

    # link maneuvers: for each phase with previous and impulsive var at front and link not continuous
    for i, b in enumerate(built):
        ph = b.ph
        if ph.previous is None:
            continue
        link_kind = (ph.link.kind if ph.link is not None else "continuous").lower()
        if link_kind == "continuous" or not _has_impulsive_var(ph, "Front"):
            continue
        prev_traj = trajs[i - 1]
        this_traj = trajs[i]
        dv = this_traj[0, 3:6] - prev_traj[-1, 3:6]
        maneuvers.append(
            Maneuver(
                r_m=prev_traj[-1, 0:3],
                t_s=float(prev_traj[-1, 6]),
                dv_mps=dv,
                name=f"Δv ({ph.name} front/link)",
            )
        )

    # terminal maneuver
    if built and _has_impulsive_var(built[-1].ph, "Back"):
        v_target = _explicit_boundary_velocity_target(built[-1].ph, "Back")
        if v_target is not None:
            v_end = trajs[-1][-1, 3:6]
            dv = v_target - v_end
            maneuvers.append(
                Maneuver(
                r_m=trajs[-1][-1, 0:3], t_s=float(trajs[-1][-1, 6]), dv_mps=dv, name="Δv (terminal)"
            )
            )

    return RendezvousResult(
        converged=bool(converged),
        traj=np.asarray(traj, dtype=np.float64),
        maneuvers=maneuvers,
        last_obj=float(getattr(ocp.optimizer, "LastObjVal", np.nan)),
        info={
            "backend": "asset_composable",
            "nphases": len(built),
            "r_unit_m": r_unit,
            "v_unit_mps": v_unit,
            "t_unit_s": t_unit,
            "phase_guess_info": guess_info,
        }
        | guess_info.get(0, {}),
    )
