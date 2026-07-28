"""Composable mission solver (ASSET backend).

This module compiles a Mission made of Phase objects into a single ASSET
OptimalControlProblem.

Scope (v0.1):
  - Two-body, J2-perturbed, finite-/low-thrust, and CWH relative dynamics
  - Phase boundary constraints:
      * State (R,V) at Front/Back
      * Position (R) at Front/Back
  - Links:
      * continuous: (R,V,t)
      * impulsive: (R,t)
  - Variables:
      * ImpulsiveDeltaV at Front / Back
      * Powered phase controls via ``mode="finite_thrust"`` or compatibility
        mode ``"chemical_burn"``
  - Objectives:
      * Minimize total Δv (default via Mission.objectives)
      * Minimize powered-phase propellant use
      * Optional Minimize time (via Mission.objectives)

This is the foundation for a general composable layer. Specialized solvers
(e.g., Lambert-aided seed searches) can be used as *guess builders* without
changing the compilation model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from .._asset import (
    add_back_time_bound,
    fix_front_time,
    oc,
    require_asset,
    set_ocp_threads,
    solve_with_standard_sequence,
    vf,
)
from ..astro.kepler import (
    estimate_orbital_period_s,
    kepler_dense_guess,
    propagate_cartesian_rv,
)
from ..astro.lambert import select_best_lambert_seed
from ..astro.types import as_vec3
from ..astro.units import default_scaling
from ..constraints import OrbitalElementConstraint
from ..guesses import LowThrustSpiralGuess
from ..phase import Phase
from ..relative import CWHRendezvousSeed, cwh_dense_guess, select_cwh_rendezvous_seed
from ..time import normalize_time_bounds
from ..types import Maneuver
from . import constraint_compiler
from .compiler import (
    nonlinear_relative_compiler,
    phase_compiler,
    powered_guessing,
    relative_constraint_compiler,
)
from .options import SolverOptions
from .preconfigured import RendezvousResult  # reuse stable result type
from .relative_environment import (
    build_solar_direction_tables,
)
from .third_bodies import build_third_body_tables, tables_for_phase

if TYPE_CHECKING:  # pragma: no cover
    from ..mission import Mission


# Private compatibility aliases keep existing internal imports and focused
# tests stable while implementation ownership moves into ``solvers.compiler``.
_PhaseBuild = phase_compiler.PhaseBuild
_augment_guess_for_powered_phase = phase_compiler.augment_powered_guess
_augment_guess_for_chemical_burn = phase_compiler.augment_chemical_burn_guess
_augment_guess_for_mass_coast = phase_compiler.augment_mass_coast_guess
_compile_phase_dimensions = phase_compiler.compile_phase_dimensions
_cwh_model = phase_compiler.cwh_model
_nonlinear_relative_model = phase_compiler.nonlinear_relative_model
_first_thruster = phase_compiler.first_thruster
_has_impulsive_var = phase_compiler.has_impulsive_variable
_is_coast_like = phase_compiler.is_coast_like
_make_asset_phase = phase_compiler.make_asset_phase
_mass_state_phase_indices = phase_compiler.mass_state_phase_indices
_ode_for_phase = phase_compiler.ode_for_phase
_phase_dimensions = phase_compiler.phase_dimensions
_phase_is_chemical_burn = phase_compiler.is_chemical_burn
_phase_is_powered = phase_compiler.is_powered_phase
_powered_phase_kind = phase_compiler.powered_phase_kind
_prepare_phase_guess = phase_compiler.prepare_phase_guess
_trajectory_rvt = phase_compiler.trajectory_rvt
_validate_chemical_burn_transfer = phase_compiler.validate_chemical_burn_transfer
_validate_powered_phase_chain = phase_compiler.validate_powered_phase_chain


def _require_asset() -> None:
    """Require ASSET before compiling a composable mission."""
    require_asset("composable optimization solves")


def _objective_weights(mission: Mission) -> tuple[bool, float, bool, float]:
    """Return normalized delta-v and final-time objective switches and weights.

    Older mission scripts can still use ``Mission.w_time`` directly. Newer
    scripts can provide explicit objective declarations. This normalizer
    preserves the default delta-v objective while letting explicit objectives
    decide which costs are active and how they are weighted.
    """
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
                w_dv = float(getattr(o, "weight", 1.0))
                break
        for o in objs:
            if getattr(o, "kind", "") == "time":
                minimize_time = True
                w_time = float(getattr(o, "weight", w_time or 1.0))
                break
    return bool(minimize_dv), float(w_dv), bool(minimize_time), float(w_time)


def _propellant_objective_weight(mission: Mission) -> float | None:
    """Return the propellant objective weight, or ``None`` when inactive."""
    for objective in list(getattr(mission, "objectives", []) or []):
        if getattr(objective, "kind", "") == "propellant":
            return float(getattr(objective, "weight", 1.0))
    return None


def _unit_vector_interpolator(
    sample_times_s: np.ndarray,
    sample_vectors: np.ndarray,
):
    """Return a normalized component-wise interpolator for reporting."""
    times = np.asarray(sample_times_s, dtype=float)
    vectors = np.asarray(sample_vectors, dtype=float)

    def at(query_times_s):
        query = np.asarray(query_times_s, dtype=float)
        flat_query = query.reshape(-1)
        values = np.column_stack(
            [
                np.interp(flat_query, times, vectors[:, component])
                for component in range(3)
            ]
        )
        values /= np.linalg.norm(values, axis=1)[:, None]
        return values.reshape((*query.shape, 3))

    return at


def _build_guess_single_phase_low_thrust(
    phase: Phase,
    *,
    mu: float,
    tf_bounds: tuple[float, float],
    nsegs: int,
) -> tuple[list[np.ndarray], dict[str, float | int | str]]:
    """Build a dynamics-integrated tangential spiral seed."""
    initial_state = phase.initial_state or constraint_compiler.state_boundary_value(
        constraint_compiler.get_state_constraint(phase, "Front")
    )
    if initial_state is None:
        raise ValueError(
            "Low-thrust spiral seeding requires an initial state or Front state constraint."
        )
    terminal = _phase_terminal_target(phase)
    if terminal is None:
        raise ValueError(
            "Low-thrust spiral seeding requires final_state or a terminal position anchor."
        )
    target_position, _ = terminal

    config = phase.initial_guess
    if config is None:
        config = LowThrustSpiralGuess()
    if not isinstance(config, LowThrustSpiralGuess):
        raise TypeError(
            "A low_thrust phase initial_guess must be guesses.low_thrust_spiral(...)."
        )
    spacecraft = phase.spacecraft
    if isinstance(spacecraft, str) or spacecraft is None:
        raise ValueError("Low-thrust spiral seeding requires a Spacecraft object.")
    thruster = _first_thruster(phase)
    return powered_guessing.build_low_thrust_spiral_seed(
        initial_position_m=initial_state.r_m,
        initial_velocity_mps=initial_state.v_mps,
        target_radius_m=float(np.linalg.norm(target_position)),
        mu_m3ps2=float(mu),
        initial_mass_kg=float(spacecraft.initial_mass_kg),
        dry_mass_kg=float(spacecraft.dry_mass_kg),
        thrust_N=float(thruster.thrust_N),
        isp_s=float(thruster.isp_s),
        time_bounds_s=tf_bounds,
        npts=int(nsegs) + 1,
        config=config,
    )


def _build_guess_single_phase_cwh(
    phase: Phase,
    *,
    tf_bounds: tuple[float, float],
    nsegs: int,
    samples: int,
    solar_direction_table=None,
) -> tuple[list[np.ndarray], dict[str, float | int | str]]:
    """Build an analytic position-targeted guess for one CWH phase."""
    model = _cwh_model(phase) or _nonlinear_relative_model(phase)
    if model is None:
        raise TypeError("Relative guess construction requires relative dynamics")
    initial_state = phase.initial_state or constraint_compiler.state_boundary_value(
        constraint_compiler.get_state_constraint(phase, "Front")
    )
    final_state = phase.final_state or constraint_compiler.state_boundary_value(
        constraint_compiler.get_state_constraint(phase, "Back")
    )
    if initial_state is None or final_state is None:
        raise ValueError(
            "A CWH rendezvous phase requires initial and final State values for guess generation."
        )
    geometry_constraints = relative_constraint_compiler.relative_geometry_constraints(phase)

    def geometry_is_feasible(candidate: CWHRendezvousSeed) -> bool:
        if not geometry_constraints:
            return True
        candidate_guess = cwh_dense_guess(
            initial_state.r_m,
            candidate.departure_velocity_mps,
            mean_motion_radps=model.mean_motion_radps,
            t0_s=0.0,
            tf_s=candidate.tof_s,
            npts=max(int(nsegs) + 1, 61),
        )
        candidate_traj = np.asarray(candidate_guess, dtype=float)
        return all(
            row["satisfied"]
            for constraint in geometry_constraints
            for row in relative_constraint_compiler.relative_constraint_report_rows(
                phase_name=phase.name,
                constraint=constraint,
                phase_traj=candidate_traj,
                solar_direction_at=(
                    solar_direction_table.at
                    if solar_direction_table is not None
                    else None
                ),
            )
        )

    seed = select_cwh_rendezvous_seed(
        initial_state,
        final_state,
        mean_motion_radps=model.mean_motion_radps,
        tof_bounds_s=tf_bounds,
        samples=samples,
        candidate_filter=geometry_is_feasible if geometry_constraints else None,
    )
    guess = cwh_dense_guess(
        initial_state.r_m,
        seed.departure_velocity_mps,
        mean_motion_radps=model.mean_motion_radps,
        t0_s=0.0,
        tf_s=seed.tof_s,
        npts=int(nsegs) + 1,
    )
    return guess, {
        "guess_kind": (
            "cwh_position_targeted"
            if _cwh_model(phase) is not None
            else "cwh_seed_for_nonlinear_relative"
        ),
        "seed_tof_s": seed.tof_s,
        "seed_total_dv_mps": seed.total_dv_mps,
        "seed_samples": int(samples),
        "seed_geometry_feasible": bool(geometry_is_feasible(seed)),
    }


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
    """Build a Lambert-seeded guess for a precoast plus transfer pair.

    This helper sweeps candidate precoast end times, propagates the initial
    orbit to each candidate, and scores Lambert transfers from that propagated
    state to the terminal position. The specialized work is only for seeding;
    the compiled OCP still uses the generic composable phase/link model. If no
    Lambert candidate is viable, midpoint Kepler guesses are returned so the
    solver can still attempt the problem.
    """
    if p0.initial_state is None and constraint_compiler.get_state_constraint(p0, "Front") is None:
        raise ValueError(
            "Precoast phase requires an initial state (phase.initial_state or State constraint at Front)."
        )

    x0 = p0.initial_state or constraint_compiler.state_boundary_value(constraint_compiler.get_state_constraint(p0, "Front"))  # type: ignore[union-attr]
    xf_state = constraint_compiler.get_state_constraint(p1, "Back")
    xf_pos = constraint_compiler.get_position_constraint(p1, "Back")

    if xf_state is None and xf_pos is None and p1.final_state is None:
        raise ValueError(
            "Transfer phase requires a terminal position (State/Position constraint or phase.final_state)."
        )

    xf_state_val = constraint_compiler.state_boundary_value(xf_state)
    xf_pos_val = constraint_compiler.position_boundary_value(xf_pos)
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
    x0 = phase.initial_state or constraint_compiler.state_boundary_value(constraint_compiler.get_state_constraint(phase, "Front"))  # type: ignore[union-attr]
    xf_state = constraint_compiler.get_state_constraint(phase, "Back")
    xf_pos = constraint_compiler.get_position_constraint(phase, "Back")

    if x0 is None:
        raise ValueError(
            "Single-phase Lambert seeding requires an initial state or State constraint at Front."
        )

    xf_state_val = constraint_compiler.state_boundary_value(xf_state)
    xf_pos_val = constraint_compiler.position_boundary_value(xf_pos)

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
    xf_state = constraint_compiler.get_state_constraint(phase, "Back")
    xf_pos = constraint_compiler.get_position_constraint(phase, "Back")
    xf_state_val = constraint_compiler.state_boundary_value(xf_state)
    xf_pos_val = constraint_compiler.position_boundary_value(xf_pos)

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
    """Build one Lambert arc and split it across continuous coast phases.

    A continuous chain can span several phases before the next explicit
    terminal target. Seeding each phase independently tends to create boundary
    mismatch. This helper computes one downstream Lambert arc, splits its end
    time monotonically across phase bounds, and propagates each phase from the
    previous guess endpoint.
    """
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
    return _guess_boundary_velocity_target(phase, where)


def _explicit_boundary_velocity_target(phase: Phase, where: str) -> np.ndarray | None:
    """Return an explicitly constrained boundary velocity target from State constraints."""
    st = constraint_compiler.get_state_constraint(phase, where)
    if st is None:
        return None
    if "V" not in constraint_compiler.state_groups(st):
        return None
    st_val = constraint_compiler.state_boundary_value(st)
    if st_val is None:
        return None
    return as_vec3(st_val.v_mps)


def _guess_boundary_velocity_target(phase: Phase, where: str) -> np.ndarray | None:
    """Return the best available boundary velocity anchor for guess construction."""
    explicit_target = _explicit_boundary_velocity_target(phase, where)
    if explicit_target is not None:
        return explicit_target
    if where.lower().startswith("f") and phase.initial_state is not None:
        return as_vec3(phase.initial_state.v_mps)
    if where.lower().startswith("b") and phase.final_state is not None:
        return as_vec3(phase.final_state.v_mps)
    return None


def _midpoint(bounds: tuple[float, float] | None, fallback: float) -> float:
    """Return the midpoint of finite bounds, or ``fallback`` if bounds are absent."""
    if bounds is None:
        return float(fallback)
    lo, hi = map(float, bounds)
    return 0.5 * (lo + hi)


def _linear_rvt_guess(
    *,
    r0_m: np.ndarray,
    v0_mps: np.ndarray,
    rf_m: np.ndarray,
    vf_mps: np.ndarray,
    t0_s: float,
    tf_s: float,
    npts: int,
) -> list[np.ndarray]:
    """Build a simple Cartesian interpolation guess for short burn arcs."""
    rows: list[np.ndarray] = []
    for frac in np.linspace(0.0, 1.0, max(int(npts), 2)):
        r = (1.0 - frac) * as_vec3(r0_m) + frac * as_vec3(rf_m)
        v = (1.0 - frac) * as_vec3(v0_mps) + frac * as_vec3(vf_mps)
        t = (1.0 - frac) * float(t0_s) + frac * float(tf_s)
        rows.append(np.hstack([r, v, t]))
    return rows


def _rocket_mass_after_impulse(
    *,
    mass0_kg: float,
    dv_mps: float,
    isp_s: float,
) -> float:
    """Return post-impulse mass using the ideal rocket equation.

    This is a seed-generation estimate, not a solver constraint. It turns a
    Lambert-derived impulsive delta-v into a plausible post-burn mass so
    finite-burn phases start with a reasonable mass trajectory.
    """
    if float(isp_s) <= 0.0:
        return float(mass0_kg)
    return float(mass0_kg) * float(np.exp(-max(float(dv_mps), 0.0) / (float(isp_s) * 9.80665)))


def _build_guess_three_phase_powered_transfer(
    phases: Sequence[Phase],
    *,
    mu: float,
    abs_bounds: Sequence[tuple[float, float] | None],
    nsegs_burn: int,
    nsegs_coast: int,
    lambert_grid_size: int,
    nrevs_to_try: Sequence[int],
) -> tuple[dict[int, list[np.ndarray]], dict[int, dict[str, float | int | bool | str]]]:
    """Seed powered-coast-powered from an equivalent impulsive transfer.

    The optimizer solves finite thrust controls, but the initial guess is easier
    to construct from an impulsive approximation. This helper estimates
    departure and arrival impulses, converts them into short linear burn arcs,
    propagates the coast between them, and stores per-phase mass guesses in
    ``Phase.info`` for later row augmentation.
    """
    if len(phases) != 3:
        return {}, {}

    burn0, coast, burn1 = phases
    if not (_phase_is_powered(burn0) and _is_coast_like(coast) and _phase_is_powered(burn1)):
        return {}, {}

    x0 = burn0.initial_state or constraint_compiler.state_boundary_value(
        constraint_compiler.get_state_constraint(burn0, "Front")
    )
    terminal = _phase_terminal_target(burn1)
    if x0 is None or terminal is None:
        return {}, {}

    rf, vf_target = terminal
    if vf_target is None:
        vf_target = as_vec3(x0.v_mps)

    t1 = _midpoint(abs_bounds[0], 120.0)
    t2_fallback = max(t1 + 600.0, _midpoint(abs_bounds[1], t1 + 3600.0))
    t2 = t2_fallback
    t3 = max(t2 + 120.0, _midpoint(abs_bounds[2], t2 + 120.0))

    coast_bounds = abs_bounds[1]
    if coast_bounds is not None:
        coast_min = max(1.0, float(coast_bounds[0]) - t1)
        coast_max = max(coast_min + 1.0, float(coast_bounds[1]) - t1)
    else:
        coast_min = max(1.0, t2 - t1)
        coast_max = max(coast_min + 1.0, coast_min * 1.25)

    seed_info: dict[str, float | int | bool | str]
    try:
        seed = select_best_lambert_seed(
            r0_m=as_vec3(x0.r_m),
            rf_m=as_vec3(rf),
            v0_mps=as_vec3(x0.v_mps),
            vf_mps=as_vec3(vf_target),
            mu_m3ps2=float(mu),
            tmin_s=float(coast_min),
            tmax_s=float(coast_max),
            n_tofs=int(lambert_grid_size),
            nrevs=tuple(int(n) for n in nrevs_to_try),
        )
        v_depart = as_vec3(seed.v1_mps)
        v_arrive = as_vec3(seed.v2_mps)
        t2 = float(t1 + seed.tof_s)
        seed_info = {
            "guess_kind": "powered_two_impulse_equivalent",
            "seed_tof_s": float(seed.tof_s),
            "seed_longway": bool(seed.longway),
            "seed_nrev": int(seed.nrev),
            "seed_rightbranch": bool(seed.rightbranch),
            "seed_total_dv_mps": float(seed.total_dv_mps),
        }
    except Exception:
        v_depart = as_vec3(x0.v_mps)
        v_arrive = as_vec3(vf_target)
        seed_info = {"guess_kind": "powered_linear_fallback"}

    if abs_bounds[2] is not None:
        t3_min, t3_max = map(float, abs_bounds[2])
        t3 = min(max(t3, max(t3_min, t2 + 1.0)), t3_max)

    burn0_guess = _linear_rvt_guess(
        r0_m=as_vec3(x0.r_m),
        v0_mps=as_vec3(x0.v_mps),
        rf_m=as_vec3(x0.r_m),
        vf_mps=v_depart,
        t0_s=0.0,
        tf_s=t1,
        npts=int(nsegs_burn) + 1,
    )
    coast_guess = kepler_dense_guess(
        r0_m=as_vec3(burn0_guess[-1][0:3]),
        v0_mps=v_depart,
        t0_s=t1,
        tf_s=t2,
        npts=int(nsegs_coast) + 1,
        mu_m3ps2=float(mu),
    )
    burn1_guess = _linear_rvt_guess(
        r0_m=as_vec3(coast_guess[-1][0:3]),
        v0_mps=v_arrive,
        rf_m=as_vec3(rf),
        vf_mps=as_vec3(vf_target),
        t0_s=t2,
        tf_s=t3,
        npts=int(nsegs_burn) + 1,
    )

    spacecraft = burn0.spacecraft
    if not isinstance(spacecraft, str) and spacecraft is not None:
        thruster = _first_thruster(burn0)
        m0 = float(spacecraft.initial_mass_kg)
        m1 = _rocket_mass_after_impulse(
            mass0_kg=m0,
            dv_mps=float(np.linalg.norm(v_depart - as_vec3(x0.v_mps))),
            isp_s=float(thruster.isp_s),
        )
        burn0.info["_mass_guess_start_kg"] = m0
        coast.info["_mass_guess_start_kg"] = m1
        burn1.info["_mass_guess_start_kg"] = m1

    guesses = {0: burn0_guess, 1: coast_guess, 2: burn1_guess}
    infos = {idx: dict(seed_info, guess_phase_index=idx, guess_phase_name=phases[idx].name) for idx in guesses}
    return guesses, infos


# Compatibility name for focused tests and downstream imports.
_build_guess_three_phase_chemical_transfer = _build_guess_three_phase_powered_transfer


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
    """Choose a practical fallback Back-time guess for a phase.

    When Lambert seeding is not available, the compiler still needs a strictly
    increasing end time inside the absolute phase bounds. This heuristic uses a
    moderate fraction of the local orbital period when one can be estimated,
    otherwise it falls back to the midpoint of the configured time window.
    """
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
    """Solve a composable mission via ASSET compilation.

    The compiler validates phase structure, normalizes absolute phase time
    bounds, builds initial guesses, instantiates ASSET phases, applies
    constraints/links/objectives, runs the protected ASSET solve sequence, and
    finally extracts trajectories, maneuvers, burn summaries, and constraint
    reports.
    """
    _require_asset()

    phases = list(mission.phases)
    if not phases:
        raise ValueError("Mission has no phases")

    frames = {phase.dynamics.frame for phase in phases if phase.dynamics is not None}
    if len(frames) > 1:
        raise ValueError(
            "Composable missions currently require one coordinate frame across all phases. "
            "Add an explicit frame transformation before linking phases in different frames."
        )

    _validate_powered_phase_chain(phases)
    mass_state_indices = _mass_state_phase_indices(phases)

    relative_phases = [
        phase for phase in phases if phase_compiler.is_relative_phase(phase)
    ]
    if relative_phases and len(relative_phases) != len(phases):
        raise ValueError(
            "A composable mission cannot link relative and inertial phases "
            "without an explicit frame transform."
        )
    if relative_phases and len(phases) != 1:
        raise NotImplementedError(
            "Relative-motion compilation currently supports one optimized phase per mission."
        )
    if relative_phases and any(
        isinstance(constraint, OrbitalElementConstraint)
        for phase in relative_phases
        for constraint in phase.constraints
    ):
        raise ValueError(
            "Inertial orbital-element constraints are not valid in a relative frame."
        )

    for ph in phases:
        normalized_mode = (ph.mode or "").lower().replace("-", "_")
        if normalized_mode not in (
            "coast",
            "transfer",
            "rendezvous",
            "relative_coast",
            "cwh",
            "burn",
            "chemical_burn",
            "finite_burn",
            "powered",
            "finite_thrust",
            "low_thrust",
        ):
            raise NotImplementedError(
                "Composable solver supports inertial/relative coast-like and finite-thrust phases. "
                f"Got mode={ph.mode!r}"
            )

    minimize_dv, w_dv, minimize_time, w_time = _objective_weights(mission)
    propellant_weight = _propellant_objective_weight(mission)
    if propellant_weight is not None and not mass_state_indices:
        raise ValueError("A propellant objective requires at least one powered phase.")

    # Normalize time bounds (absolute Back-time bounds for each phase).
    abs_bounds = normalize_time_bounds(phases)
    third_body_tables = build_third_body_tables(mission, phases, abs_bounds)
    solar_direction_tables = build_solar_direction_tables(mission, phases, abs_bounds)
    asset_solar_direction_tables = {
        index: vf.InterpTable1D(
            table.times_s,
            table.directions_ric,
            axis=0,
            kind="cubic",
        )
        for index, table in solar_direction_tables.items()
    }
    asset_solar_position_tables = {
        index: vf.InterpTable1D(
            table.times_s,
            table.sun_positions_eci_m,
            axis=0,
            kind="cubic",
        )
        for index, table in solar_direction_tables.items()
        if table.sun_positions_eci_m is not None
    }

    # Characteristic scaling remains dimensional at the public API boundary.
    first = phases[0]
    last = phases[-1]
    mu = float(first.dynamics.mu_m3ps2)  # type: ignore[union-attr]

    x0_for_units = (
        first.initial_state
        or constraint_compiler.state_boundary_value(
            constraint_compiler.get_state_constraint(first, "Front")
        )
        or constraint_compiler.state_boundary_value(
            constraint_compiler.get_state_constraint(first, "Back")
        )
    )
    xf_for_units = (
        last.final_state
        or constraint_compiler.state_boundary_value(
            constraint_compiler.get_state_constraint(last, "Back")
        )
        or constraint_compiler.state_boundary_value(
            constraint_compiler.get_state_constraint(last, "Front")
        )
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
            self.scaling = first.dynamics.scaling  # type: ignore[union-attr]
            spacecraft = first.spacecraft
            self.mass_unit_kg = (
                float(spacecraft.initial_mass_kg)
                if not isinstance(spacecraft, str) and spacecraft is not None
                else 1.0
            )
            # approximate time bounds from last phase if available
            last_bounds = abs_bounds[-1]
            self.tf_bounds_s = last_bounds if last_bounds is not None else (0.0, 10.0)

    solver_scaling = default_scaling(_UnitSpec(x0_for_units, xf_for_units, mu))
    r_unit = solver_scaling.length_m
    v_unit = solver_scaling.velocity_mps
    t_unit = solver_scaling.time_s

    # Build guesses: handle common 2-phase precoast+transfer case for better robustness
    nsegs0 = int(getattr(mission, "mesh_nsegs_precoast", 30))
    nsegs1 = int(getattr(mission, "mesh_nsegs_transfer", 60))

    guesses: dict[int, list[np.ndarray]] = {}
    guess_info: dict[int, dict[str, float | int | bool | str]] = {}

    if relative_phases:
        p0 = phases[0]
        tf_bounds = abs_bounds[0] or (600.0, 7_200.0)
        ig0, info0 = _build_guess_single_phase_cwh(
            p0,
            tf_bounds=tf_bounds,
            nsegs=nsegs1,
            samples=int(getattr(mission, "lambert_grid_size", 60)),
            solar_direction_table=solar_direction_tables.get(0),
        )
        guesses[0] = ig0
        guess_info[0] = info0

    elif len(phases) == 1 and _powered_phase_kind(phases[0]) == "low_thrust":
        p0 = phases[0]
        tf_bounds = abs_bounds[0] or (86_400.0, 30.0 * 86_400.0)
        ig0, info0 = _build_guess_single_phase_low_thrust(
            p0,
            mu=mu,
            tf_bounds=tf_bounds,
            nsegs=nsegs1,
        )
        guesses[0] = ig0
        guess_info[0] = info0

    elif mass_state_indices and len(phases) == 3:
        powered_guesses, powered_info = _build_guess_three_phase_powered_transfer(
            phases,
            mu=mu,
            abs_bounds=abs_bounds,
            nsegs_burn=max(4, int(nsegs0 // 2)),
            nsegs_coast=nsegs1,
            lambert_grid_size=int(getattr(mission, "lambert_grid_size", 60)),
            nrevs_to_try=tuple(int(x) for x in getattr(mission, "nrevs_to_try", (0, 1))),
        )
        guesses.update(powered_guesses)
        guess_info.update(powered_info)

    elif len(phases) == 1:
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
    terminal_shell = constraint_compiler.make_terminal_shell(phases[-1])
    last_compile_phase = terminal_shell[0] if terminal_shell is not None else None
    shell_phase = terminal_shell[1] if terminal_shell is not None else None

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
                x0_chain = ph.initial_state or constraint_compiler.state_boundary_value(
                    constraint_compiler.get_state_constraint(ph, "Front")
                )
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
                    prev_guess = _trajectory_rvt(
                        np.asarray(built[-1].asset_phase.returnTraj(), dtype=float),
                        built[-1].layout,
                    ).tolist()
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
                asset_phase, layout, powered_kind = _make_asset_phase(
                    ph,
                    ig,
                    int(nsegs),
                    carries_mass=idx in mass_state_indices,
                    third_body_tables=third_body_tables,
                )
                ocp.addPhase(asset_phase)

                built.append(
                    _PhaseBuild(
                        ph=ph,
                        asset_phase=asset_phase,
                        t_bounds=tuple(abs_bounds[idx] or (0.0, 1.0)),
                        index=idx,
                        compile_phase=(last_compile_phase if ph is phases[-1] else None),
                        layout=layout,
                        powered_kind=powered_kind,
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
                    prev_guess = _trajectory_rvt(
                        np.asarray(built[-1].asset_phase.returnTraj(), dtype=float),
                        built[-1].layout,
                    ).tolist()
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

        asset_phase, layout, powered_kind = _make_asset_phase(
            ph,
            ig,
            int(nsegs),
            carries_mass=idx in mass_state_indices,
            third_body_tables=third_body_tables,
        )
        ocp.addPhase(asset_phase)

        built.append(
            _PhaseBuild(
                ph=ph,
                asset_phase=asset_phase,
                t_bounds=tuple(abs_bounds[idx] or (0.0, 1.0)),
                index=idx,
                compile_phase=(last_compile_phase if ph is phases[-1] else None),
                layout=layout,
                powered_kind=powered_kind,
            )
        )

    if shell_phase is not None:
        last_guess = guesses.get(len(phases) - 1)
        if last_guess is None:
            last_guess = _trajectory_rvt(
                np.asarray(built[-1].asset_phase.returnTraj(), dtype=float),
                built[-1].layout,
            ).tolist()
        last_pt = np.asarray(last_guess[-1], dtype=float)
        shell_t0 = float(last_pt[6])
        shell_tf = shell_t0 + 1.0
        shell_v0_guess = _guess_boundary_velocity_target(phases[-1], "Back")
        if shell_v0_guess is None:
            shell_v0_guess = as_vec3(last_pt[3:6])
        shell_guess = kepler_dense_guess(
            r0_m=as_vec3(last_pt[0:3]),
            v0_mps=shell_v0_guess,
            t0_s=shell_t0,
            tf_s=shell_tf,
            npts=3,
            mu_m3ps2=mu,
        )
        shell_asset_phase, layout, powered_kind = _make_asset_phase(
            shell_phase,
            shell_guess,
            2,
            third_body_tables=third_body_tables,
        )
        ocp.addPhase(shell_asset_phase)
        built.append(
            _PhaseBuild(
                ph=shell_phase,
                asset_phase=shell_asset_phase,
                t_bounds=(shell_t0 + 0.1, shell_tf),
                index=len(built),
                compile_phase=shell_phase,
                layout=layout,
                powered_kind=powered_kind,
                enable_adaptive_mesh=False,
            )
        )

    # Set scaling & mesh options
    opts = options or SolverOptions()
    ocp.optimizer.PrintLevel = int(opts.print_level)
    ocp.optimizer.MaxLSIters = int(opts.max_ls_iters)
    ocp.optimizer.set_QPOrderingMode(str(opts.qp_ordering_mode))
    set_ocp_threads(ocp, opts.asset_threads)
    nonlinear_ephemeris_phase = any(
        _nonlinear_relative_model(build.ph) is not None
        and bool(tables_for_phase(build.ph, third_body_tables))
        for build in built
    )
    adaptive_mesh_enabled = bool(
        opts.enable_adaptive_mesh and not nonlinear_ephemeris_phase
    )

    for b in built:
        b.asset_phase.setAutoScaling(bool(opts.enable_auto_scaling))
        nonlinear_model = _nonlinear_relative_model(b.ph)
        if nonlinear_model is not None:
            chief = nonlinear_model.chief_initial_state_eci
            absolute_position_unit_m = float(np.linalg.norm(chief.r_m))
            absolute_velocity_unit_mps = float(np.linalg.norm(chief.v_mps))
            b.asset_phase.setUnits(
                ChiefR=absolute_position_unit_m,
                ChiefV=absolute_velocity_unit_mps,
                DeputyR=absolute_position_unit_m,
                DeputyV=absolute_velocity_unit_mps,
                t=t_unit,
            )
        elif b.state_dim == 7:
            b.asset_phase.setUnits(
                R=r_unit,
                V=v_unit,
                M=solver_scaling.mass_kg,
                t=t_unit,
            )
        else:
            b.asset_phase.setUnits(R=r_unit, V=v_unit, t=t_unit)
        b.asset_phase.setAdaptiveMesh(
            bool(adaptive_mesh_enabled and b.enable_adaptive_mesh)
        )

    ocp.setAutoScaling(True, True)
    ocp.setAdaptiveMesh(adaptive_mesh_enabled)
    ocp.PrintMeshInfo = False

    # Apply constraints and time bounds per phase
    for b in built:
        ph = b.compile_phase or b.ph
        ap = b.asset_phase
        nonlinear_model = _nonlinear_relative_model(ph)
        relative_expressions = (
            nonlinear_relative_compiler.relative_state_expressions(
                ph,
                third_body_tables,
            )
            if nonlinear_model is not None
            else None
        )

        # First phase front time fixed at 0 unless user provides otherwise
        if b.index == 0:
            fix_front_time(ap, 0.0)
        if nonlinear_model is not None:
            nonlinear_relative_compiler.fix_initial_chief(ap, nonlinear_model)

        if b.state_dim == 7:
            spacecraft = ph.spacecraft
            if isinstance(spacecraft, str) or spacecraft is None:
                raise ValueError(f"Mass-carrying phase {ph.name!r} requires a Spacecraft object.")
            has_mass_predecessor = any(
                previous_build.ph is ph.previous and previous_build.state_dim == 7
                for previous_build in built
            )
            if not has_mass_predecessor:
                ap.addBoundaryValue(
                    "Front",
                    ["M"],
                    np.asarray([float(spacecraft.initial_mass_kg)], dtype=float),
                )
            dry_mass_kg = float(spacecraft.dry_mass_kg)
            if dry_mass_kg > 0.0:
                ap.addLUVarBound("Back", "M", dry_mass_kg, float(spacecraft.initial_mass_kg))
            if b.is_powered:
                ap.addUpperNormBound("Path", "U", 1.0)

        # Apply State/Position boundary constraints with impulsive-variable override logic
        for loc in ("Front", "Back"):
            st = constraint_compiler.get_state_constraint(ph, loc)
            pos = constraint_compiler.get_position_constraint(ph, loc)

            # Position constraint always applied (if present)
            if pos is not None:
                position_value = np.asarray(
                    as_vec3(constraint_compiler.position_boundary_value(pos)),
                    dtype=float,
                )
                if relative_expressions is not None:
                    nonlinear_relative_compiler.apply_position_boundary(
                        ap,
                        loc,
                        position_value,
                        relative_expressions,
                    )
                else:
                    ap.addBoundaryValue(loc, ["R"], position_value)

            if st is not None:
                st_val = constraint_compiler.state_boundary_value(st)
                groups = constraint_compiler.state_groups(st)

                # If an impulsive Δv is declared at this boundary, we drop V from the constraint
                if _has_impulsive_var(ph, loc) and "V" in groups:
                    groups = tuple(g for g in groups if g != "V")

                if relative_expressions is not None:
                    nonlinear_relative_compiler.apply_state_boundary(
                        ap,
                        loc,
                        st_val,
                        groups,
                        relative_expressions,
                    )
                    continue

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
        for constraint in getattr(ph, "constraints", []) or []:
            if getattr(constraint, "kind", "") == "min_radius":
                minimum_radius_m = float(constraint.value)
                location = (
                    "Path"
                    if getattr(constraint, "where", "Path") == "Path"
                    else getattr(constraint, "where", "Path")
                )
                if relative_expressions is not None:
                    nonlinear_relative_compiler.apply_minimum_range(
                        ap,
                        location,
                        minimum_radius_m,
                        relative_expressions,
                    )
                else:
                    ap.addLowerNormBound(location, "R", minimum_radius_m)
            elif isinstance(constraint, OrbitalElementConstraint):
                constraint_compiler.apply_orbital_element_constraint(ap, constraint, mu)
            elif relative_constraint_compiler.is_relative_geometry_constraint(constraint):
                if relative_expressions is not None:
                    nonlinear_relative_compiler.apply_geometry_constraint(
                        ap,
                        constraint,
                        relative_expressions,
                        solar_position_table=asset_solar_position_tables.get(b.index),
                    )
                else:
                    relative_constraint_compiler.apply_relative_geometry_constraint(
                        ap,
                        constraint,
                        b.layout.state_indices("position"),
                        time_index=b.layout.time_column,
                        solar_direction_table=asset_solar_direction_tables.get(b.index),
                    )

        # Time bounds: normalize tof_bounds_s to absolute Back-time bounds.
        bounds = b.t_bounds
        if bounds is not None:
            tmin, tmax = map(float, bounds)
            add_back_time_bound(ap, b.state_dim, tmin, tmax)
            ap.addLowerDeltaTimeBound(0.1)

    # Apply links and link objectives
    for b in built:
        ph = b.ph
        if ph.previous is None:
            continue

        # Find previous compiled phase
        prev_idx = None
        prev_build = None
        for bb in built:
            if bb.ph is ph.previous:
                prev_idx = bb.index
                prev_ap = bb.asset_phase
                prev_build = bb
                break
        if prev_idx is None:
            raise ValueError(f"Phase {ph.name!r} references previous phase not in mission.")

        ap = b.asset_phase
        link_kind = (ph.link.kind if ph.link is not None else "continuous").lower()

        if link_kind == "continuous":
            link_groups = ["R", "V", "t"]
            if prev_build is not None and prev_build.state_dim == 7 and b.state_dim == 7:
                link_groups = ["R", "V", "M", "t"]
            ocp.addForwardLinkEqualCon(prev_ap, ap, link_groups)
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
                list(prev_build.layout.state_indices("velocity")),
                [],
                [],
                ap,
                "Front",
                list(b.layout.state_indices("velocity")),
                [],
                [],
                [],
                AutoScale=1.0 / float(v_unit),
            )

    # Powered propellant objective, applied once at the end of the mass chain.
    if propellant_weight is not None:
        final_powered = next(build for build in reversed(built) if build.is_powered)
        spacecraft = final_powered.ph.spacecraft
        if isinstance(spacecraft, str) or spacecraft is None:  # guarded by validation
            raise ValueError("A propellant objective requires a configured spacecraft.")
        final_powered.asset_phase.addValueObjective(
            "Back",
            final_powered.layout.state_indices("mass")[0],
            -float(propellant_weight) / max(float(spacecraft.initial_mass_kg), 1.0),
        )

    if minimize_dv:
        # Start: if first phase has impulsive Δv at Front, penalize relative to desired initial velocity
        first_ph = built[0].ph
        first_ap = built[0].asset_phase
        if _has_impulsive_var(first_ph, "Front"):
            st = constraint_compiler.get_state_constraint(first_ph, "Front")
            if st is None and first_ph.initial_state is None:
                raise ValueError(
                    "ImpulsiveDeltaV at mission start requires a desired initial velocity (State constraint or initial_state)."
                )
            st_val = constraint_compiler.state_boundary_value(st)
            v_target = as_vec3(st_val.v_mps if st_val is not None else first_ph.initial_state.v_mps)  # type: ignore[union-attr]
            nonlinear_model = _nonlinear_relative_model(first_ph)
            if nonlinear_model is not None:
                expressions = nonlinear_relative_compiler.relative_state_expressions(
                    first_ph,
                    third_body_tables,
                )
                nonlinear_relative_compiler.add_velocity_objective(
                    first_ap,
                    "Front",
                    v_target,
                    float(w_dv),
                    float(v_unit),
                    expressions,
                )
            else:
                a0 = vf.Arguments(3)
                dv0 = vf.sqrt((a0 - v_target).dot(a0 - v_target))
                first_ap.addStateObjective(
                    "Front",
                    float(w_dv) * dv0,
                    list(built[0].layout.state_indices("velocity")),
                    [],
                    [],
                    AutoScale=1.0 / float(v_unit),
                )

        # Terminal: if last phase has impulsive Δv at Back, penalize relative to desired terminal velocity
        last_ph = built[-1].ph
        last_ap = built[-1].asset_phase
        if _has_impulsive_var(last_ph, "Back"):
            v_target = _explicit_boundary_velocity_target(last_ph, "Back")
            if v_target is not None:
                nonlinear_model = _nonlinear_relative_model(last_ph)
                if nonlinear_model is not None:
                    expressions = (
                        nonlinear_relative_compiler.relative_state_expressions(
                            last_ph,
                            third_body_tables,
                        )
                    )
                    nonlinear_relative_compiler.add_velocity_objective(
                        last_ap,
                        "Back",
                        v_target,
                        float(w_dv),
                        float(v_unit),
                        expressions,
                    )
                else:
                    b0 = vf.Arguments(3)
                    dvf = vf.sqrt((v_target - b0).dot(v_target - b0))
                    last_ap.addStateObjective(
                        "Back",
                        float(w_dv) * dvf,
                        list(built[-1].layout.state_indices("velocity")),
                        [],
                        [],
                        AutoScale=1.0 / float(v_unit),
                    )

    # Time objective: minimize final time at last phase Back
    if minimize_time and float(w_time) != 0.0:
        last_build = built[-1]
        last_ap = last_build.asset_phase
        at = vf.Arguments(1).tolist()[0]
        last_ap.addStateObjective(
            "Back",
            float(w_time) * at,
            [last_build.layout.time_column],
            [],
            [],
            AutoScale=1.0 / float(t_unit),
        )

    # Solve
    converged = solve_with_standard_sequence(
        ocp,
        phases=tuple(build.asset_phase for build in built),
    )

    # Extract trajectory (stitch)
    raw_trajs = [np.asarray(b.asset_phase.returnTraj(), dtype=float) for b in built]
    trajs = [
        (
            nonlinear_relative_compiler.coupled_trajectory_rvt(
                raw_traj,
                build.ph,
                third_body_tables,
            )
            if _nonlinear_relative_model(build.ph) is not None
            else _trajectory_rvt(raw_traj, build.layout)
        )
        for raw_traj, build in zip(raw_trajs, built, strict=True)
    ]
    solved_solar_directions = {
        build.index: nonlinear_relative_compiler.solar_directions_ric(
            raw_traj,
            solar_direction_tables[build.index],
        )
        for raw_traj, build in zip(raw_trajs, built, strict=True)
        if _nonlinear_relative_model(build.ph) is not None
        and build.index in solar_direction_tables
    }
    traj = trajs[0]
    for t in trajs[1:]:
        traj = np.vstack([traj, t[1:, :]])

    # Maneuver bookkeeping: link dv and explicit terminal dv if requested
    maneuvers: list[Maneuver] = []
    # start maneuver (mission start boundary)
    if built and _has_impulsive_var(built[0].ph, "Front"):
        st0 = constraint_compiler.get_state_constraint(built[0].ph, "Front")
        st0_val = constraint_compiler.state_boundary_value(st0)
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
                    r_m=trajs[-1][-1, 0:3],
                    t_s=float(trajs[-1][-1, 6]),
                    dv_mps=dv,
                    name="Δv (terminal)",
                )
            )

    constraint_report: list[dict[str, float | str | bool]] = []
    for build, phase_traj in zip(built, trajs, strict=True):
        constraint_phase = build.compile_phase or build.ph
        for constraint in constraint_compiler.orbital_element_constraints(constraint_phase):
            if constraint.where not in {"Front", "Back"}:
                continue
            constraint_report.append(
                constraint_compiler.orbital_constraint_report_row(
                    phase_name=constraint_phase.name,
                    constraint=constraint,
                    phase_traj=phase_traj,
                    mu_m3ps2=mu,
                )
            )
        for constraint in relative_constraint_compiler.relative_geometry_constraints(
            constraint_phase
        ):
            solar_direction_at = None
            if build.index in solved_solar_directions:
                solar_direction_at = _unit_vector_interpolator(
                    phase_traj[:, 6],
                    solved_solar_directions[build.index],
                )
            elif build.index in solar_direction_tables:
                solar_direction_at = solar_direction_tables[build.index].at
            constraint_report.extend(
                relative_constraint_compiler.relative_constraint_report_rows(
                    phase_name=constraint_phase.name,
                    constraint=constraint,
                    phase_traj=phase_traj,
                    solar_direction_at=solar_direction_at,
                )
            )

    powered_phases: list[dict[str, float | str]] = []
    chemical_burns: list[dict[str, float | str]] = []
    phase_segments: list[dict[str, float | str]] = []
    for build, phase_traj in zip(built, trajs, strict=True):
        is_powered = build.is_powered
        is_relative = phase_compiler.is_relative_phase(build.ph)
        phase_segments.append(
            {
                "name": build.ph.name,
                "mode": (
                    str(build.powered_kind)
                    if is_powered
                    else "relative_coast"
                    if is_relative
                    else "coast"
                ),
                "t_start_s": float(phase_traj[0, 6]),
                "t_end_s": float(phase_traj[-1, 6]),
                "color": "red" if is_powered else "blue",
            }
        )
        if not is_powered:
            continue
        raw_traj = raw_trajs[build.index]
        mass_index = build.layout.state_indices("mass")[0]
        mass_initial_kg = float(raw_traj[0, mass_index])
        mass_final_kg = float(raw_traj[-1, mass_index])
        propellant_used_kg = max(0.0, mass_initial_kg - mass_final_kg)
        equivalent_dv_mps = 0.0
        spacecraft = build.ph.spacecraft
        if not isinstance(spacecraft, str) and spacecraft is not None and mass_final_kg > 0.0:
            thruster = _first_thruster(build.ph)
            equivalent_dv_mps = float(thruster.isp_s) * 9.80665 * float(
                np.log(max(mass_initial_kg, mass_final_kg) / mass_final_kg)
            )
        summary = {
            "phase": build.ph.name,
            "kind": str(build.powered_kind),
            "t_start_s": float(phase_traj[0, 6]),
            "t_end_s": float(phase_traj[-1, 6]),
            "mass_initial_kg": mass_initial_kg,
            "mass_final_kg": mass_final_kg,
            "propellant_used_kg": propellant_used_kg,
            "equivalent_dv_mps": equivalent_dv_mps,
        }
        powered_phases.append(summary)
        if build.is_chemical_burn:
            chemical_burns.append(dict(summary))

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
            "scaling": solver_scaling.to_dict(),
            "frame": first.dynamics.frame.to_dict(),  # type: ignore[union-attr]
            "central_body": (
                first.dynamics.central_body.name  # type: ignore[union-attr]
                if first.dynamics.central_body is not None  # type: ignore[union-attr]
                else first.dynamics.frame.origin  # type: ignore[union-attr]
            ),
            "mu_m3ps2": mu,
            "dynamics_model": (
                "nonlinear_relative"
                if any(_nonlinear_relative_model(phase) is not None for phase in phases)
                else "cwh"
                if relative_phases
                else "central_gravity"
            ),
            "relative_reference_model": None,
            "solar_directions_ric": (
                solved_solar_directions[0].tolist()
                if solved_solar_directions
                else solar_direction_tables[0].at(traj[:, 6]).tolist()
                if relative_phases and 0 in solar_direction_tables
                else None
            ),
            "chief_trajectory_eci": (
                np.column_stack(
                    [raw_trajs[0][:, 0:6], raw_trajs[0][:, 12]]
                ).tolist()
                if built and _nonlinear_relative_model(built[0].ph) is not None
                else None
            ),
            "deputy_trajectory_eci": (
                np.column_stack(
                    [raw_trajs[0][:, 6:12], raw_trajs[0][:, 12]]
                ).tolist()
                if built and _nonlinear_relative_model(built[0].ph) is not None
                else None
            ),
            "state_layouts": [build.layout.name for build in built],
            "constraint_report": constraint_report,
            "powered_phases": powered_phases,
            "chemical_burns": chemical_burns,
            "phase_segments": phase_segments,
            "phase_guess_info": guess_info,
        }
        | guess_info.get(0, {}),
    )
