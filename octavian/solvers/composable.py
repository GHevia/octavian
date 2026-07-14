"""Composable mission solver (ASSET backend).

This module compiles a Mission made of Phase objects into a single ASSET
OptimalControlProblem.

Scope (v0.1):
  - Two-body, J2-perturbed, and finite chemical-burn dynamics
  - Phase boundary constraints:
      * State (R,V) at Front/Back
      * Position (R) at Front/Back
  - Links:
      * continuous: (R,V,t)
      * impulsive: (R,t)
  - Variables:
      * ImpulsiveDeltaV at Front / Back
      * Chemical-burn phase controls via ``mode="chemical_burn"``
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
from typing import TYPE_CHECKING, Any

import numpy as np

from .._asset import (
    Tmodes,
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
from ..astro.units import default_units
from ..constraints import OrbitalElementConstraint
from ..dynamics import (
    ChemicalBurnECI,
    MassCoastECI,
    PerturbedECI,
    ThirdBodyTable,
    TwoBodyECI,
)
from ..phase import Phase
from ..time import normalize_time_bounds
from ..types import Maneuver
from ..variables import ImpulsiveDeltaV
from . import constraint_compiler
from .options import SolverOptions
from .preconfigured import RendezvousResult  # reuse stable result type
from .third_bodies import build_third_body_tables, phase_perturbations, tables_for_phase

if TYPE_CHECKING:  # pragma: no cover
    from ..mission import Mission


def _require_asset() -> None:
    """Require ASSET before compiling a composable mission."""
    require_asset("composable optimization solves")


@dataclass
class _PhaseBuild:
    """Bookkeeping for one compiled ASSET phase.

    User-facing phases and compiled phases can diverge when the compiler adds
    internal helper phases, such as the post-burn shell used for terminal
    orbital-element targets. This record keeps the ASSET phase, original phase,
    compile-time phase override, dimensions, and time bounds together for the
    later constraint, link, solve, and result-extraction passes.
    """

    ph: Phase
    asset_phase: Any
    t_bounds: tuple[float, float]
    index: int
    compile_phase: Phase | None = None
    state_dim: int = 6
    control_dim: int = 0
    is_chemical_burn: bool = False
    enable_adaptive_mesh: bool = True


def _has_impulsive_var(phase: Phase, where: str) -> bool:
    """Return whether a phase has an impulsive delta-v at a boundary."""
    w = (where or "").strip().lower()
    loc = "Front" if w in ("front", "start", "initial", "t0") else "Back"
    for variable in getattr(phase, "variables", []) or []:
        if isinstance(variable, ImpulsiveDeltaV) and getattr(variable, "where", "") == loc:
            return True
    for event in getattr(phase, "events", []) or []:
        if getattr(event, "kind", "") == "impulse" and getattr(event, "where", "") == loc:
            return True
    return False


def _phase_is_chemical_burn(phase: Phase) -> bool:
    """Return whether a phase should compile to finite chemical-burn dynamics."""
    normalized_mode = (getattr(phase, "mode", "") or "").strip().lower().replace("-", "_")
    return normalized_mode in ("burn", "chemical_burn", "finite_burn")


def _is_coast_like(phase: Phase) -> bool:
    """Return whether a phase mode uses coast-like translational dynamics."""
    normalized_mode = (getattr(phase, "mode", "") or "").strip().lower().replace("-", "_")
    return normalized_mode in ("coast", "transfer", "rendezvous")


def _mass_state_phase_indices(phases: Sequence[Phase]) -> set[int]:
    """Return phases that should carry mass for finite-burn transfers."""
    burn_indices = [idx for idx, phase in enumerate(phases) if _phase_is_chemical_burn(phase)]
    if not burn_indices:
        return set()

    out = set(burn_indices)
    first_burn = burn_indices[0]
    last_burn = burn_indices[-1]
    for idx in range(first_burn + 1, last_burn):
        if _is_coast_like(phases[idx]):
            out.add(idx)
    return out


def _validate_chemical_burn_transfer(phases: Sequence[Phase]) -> None:
    """Validate the burn-coast-burn shape required for finite chemical transfers."""
    burn_indices = [idx for idx, phase in enumerate(phases) if _phase_is_chemical_burn(phase)]
    if not burn_indices:
        return
    if len(phases) < 3 or len(burn_indices) < 2:
        raise ValueError(
            "Chemical burn transfers require at least three phases: "
            "a departure burn, a coast, and an arrival burn."
        )
    first_burn = burn_indices[0]
    last_burn = burn_indices[-1]
    if first_burn != 0 or last_burn != len(phases) - 1:
        raise ValueError(
            "Chemical burn transfers must start with a burn phase and end with a burn phase."
        )
    if not any(_is_coast_like(phases[idx]) for idx in range(first_burn + 1, last_burn)):
        raise ValueError("Chemical burn transfers require a coast phase between the burns.")


def _first_thruster(phase: Phase):
    """Return the thruster configured for a chemical burn phase."""
    spacecraft = getattr(phase, "spacecraft", None)
    if isinstance(spacecraft, str) or spacecraft is None:
        raise ValueError(f"Chemical burn phase {phase.name!r} requires a Spacecraft object.")
    thruster_name = str(getattr(phase, "info", {}).get("thruster", "main"))
    thruster = spacecraft.get_thruster(thruster_name)
    if thruster is not None:
        return thruster
    if len(spacecraft.thrusters) == 1:
        return spacecraft.thrusters[0]
    raise KeyError(f"No thruster named {thruster_name!r} on spacecraft {spacecraft.name!r}")


def _ode_for_phase(
    phase: Phase,
    *,
    carries_mass: bool = False,
    third_body_tables: Sequence[ThirdBodyTable] = (),
):
    """Build the ASSET ODE for one phase.

    Phase mode, mass bookkeeping, and perturbation flags jointly select the ODE
    class. Chemical burns need controls and mass, coasts between burns may carry
    mass without controls, and ordinary coast-like phases use either two-body or
    perturbed translational dynamics.
    """
    dynamics = phase.dynamics
    if dynamics is None:
        raise ValueError(f"Phase {phase.name!r} is missing dynamics.")
    perturbations = phase_perturbations(phase)
    if _phase_is_chemical_burn(phase):
        thruster = _first_thruster(phase)
        return ChemicalBurnECI(
            mu_m3ps2=float(dynamics.mu_m3ps2),
            thrust_N=float(thruster.thrust_N),
            isp_s=float(thruster.isp_s),
            j2=bool(perturbations.j2),
            central_body_radius_m=float(dynamics.central_body_radius_m),
            j2_coefficient=float(dynamics.j2_coefficient),
            third_body_tables=tuple(third_body_tables),
        )
    if carries_mass:
        return MassCoastECI(
            mu_m3ps2=float(dynamics.mu_m3ps2),
            j2=bool(perturbations.j2),
            central_body_radius_m=float(dynamics.central_body_radius_m),
            j2_coefficient=float(dynamics.j2_coefficient),
            third_body_tables=tuple(third_body_tables),
        )
    if perturbations.j2 or third_body_tables:
        return PerturbedECI(
            mu_m3ps2=float(dynamics.mu_m3ps2),
            j2=bool(perturbations.j2),
            central_body_radius_m=float(dynamics.central_body_radius_m),
            j2_coefficient=float(dynamics.j2_coefficient),
            third_body_tables=tuple(third_body_tables),
        )
    return TwoBodyECI(mu_m3ps2=float(dynamics.mu_m3ps2))


def _phase_dimensions(phase: Phase) -> tuple[int, int, bool]:
    """Return the user-visible state/control dimensions implied by phase mode."""
    if _phase_is_chemical_burn(phase):
        return 7, 3, True
    return 6, 0, False


def _compile_phase_dimensions(phase: Phase, *, carries_mass: bool = False) -> tuple[int, int, bool]:
    """Return state/control dimensions used for ASSET compilation.

    Coast-like phases normally use six Cartesian states. In burn-coast-burn
    missions, coast phases between burns carry mass as a seventh constant state
    so continuity links can preserve propellant bookkeeping.
    """
    if _phase_is_chemical_burn(phase):
        return 7, 3, True
    if carries_mass:
        return 7, 0, False
    return 6, 0, False


def _trajectory_rvt(raw_traj: np.ndarray, state_dim: int) -> np.ndarray:
    """Return the public ``[R, V, t]`` view of an ASSET trajectory."""
    raw = np.asarray(raw_traj, dtype=float)
    time_col = int(state_dim)
    if raw.shape[1] <= time_col:
        raise ValueError("ASSET trajectory is missing the phase time column.")
    return raw[:, [0, 1, 2, 3, 4, 5, time_col]]


def _augment_guess_for_chemical_burn(
    base_guess: Sequence[np.ndarray],
    *,
    phase: Phase,
    mass0_kg: float,
    thrust_N: float,
    isp_s: float,
) -> list[np.ndarray]:
    """Convert an impulsive-style ``[R,V,t]`` guess into burn state/control rows.

    The shared guess builders work in public trajectory rows, while chemical
    burn ASSET phases require ``[R,V,M,t,U]`` rows. This helper adds a plausible
    mass history and a constant thrust direction inferred from the velocity
    change across the base guess. The throttle estimate is capped to ``[0, 1]``.
    """
    rows = [np.asarray(row, dtype=float).reshape(-1) for row in base_guess]
    if not rows:
        return []

    dv_vec = as_vec3(rows[-1][3:6] - rows[0][3:6])
    dv_mag = float(np.linalg.norm(dv_vec))
    direction = dv_vec / dv_mag if dv_mag > 0.0 else np.zeros(3, dtype=float)

    duration_s = max(float(rows[-1][6] - rows[0][6]), 1.0)
    mass_flow_kgps = float(thrust_N) / (float(isp_s) * 9.80665)
    accel_mps2 = float(thrust_N) / max(float(mass0_kg), 1.0)
    impulsive_burn_time_s = dv_mag / max(accel_mps2, 1e-12)
    throttle = min(1.0, max(0.0, impulsive_burn_time_s / duration_s))
    control = throttle * direction

    augmented: list[np.ndarray] = []
    mass_start_kg = float(getattr(phase, "info", {}).get("_mass_guess_start_kg", mass0_kg))
    for idx, row in enumerate(rows):
        frac = idx / max(len(rows) - 1, 1)
        mass = max(mass_start_kg - mass_flow_kgps * throttle * duration_s * frac, 1.0)
        augmented.append(np.hstack([row[0:6], mass, row[6], control]))
    return augmented


def _augment_guess_for_mass_coast(
    base_guess: Sequence[np.ndarray],
    *,
    phase: Phase,
    mass0_kg: float,
) -> list[np.ndarray]:
    """Add a constant mass state to a coast guess."""
    mass_start_kg = float(getattr(phase, "info", {}).get("_mass_guess_start_kg", mass0_kg))
    augmented: list[np.ndarray] = []
    for row in base_guess:
        rvt = np.asarray(row, dtype=float).reshape(-1)
        augmented.append(np.hstack([rvt[0:6], mass_start_kg, rvt[6]]))
    return augmented


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


def _build_guess_three_phase_chemical_transfer(
    phases: Sequence[Phase],
    *,
    mu: float,
    abs_bounds: Sequence[tuple[float, float] | None],
    nsegs_burn: int,
    nsegs_coast: int,
    lambert_grid_size: int,
    nrevs_to_try: Sequence[int],
) -> tuple[dict[int, list[np.ndarray]], dict[int, dict[str, float | int | bool | str]]]:
    """Seed burn-coast-burn from the equivalent two-impulse Lambert transfer.

    The optimizer solves finite thrust controls, but the initial guess is easier
    to construct from an impulsive approximation. This helper estimates
    departure and arrival impulses, converts them into short linear burn arcs,
    propagates the coast between them, and stores per-phase mass guesses in
    ``Phase.info`` for later row augmentation.
    """
    if len(phases) != 3:
        return {}, {}

    burn0, coast, burn1 = phases
    if not (_phase_is_chemical_burn(burn0) and _is_coast_like(coast) and _phase_is_chemical_burn(burn1)):
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
            "guess_kind": "chemical_burn_two_impulse_equivalent",
            "seed_tof_s": float(seed.tof_s),
            "seed_longway": bool(seed.longway),
            "seed_nrev": int(seed.nrev),
            "seed_rightbranch": bool(seed.rightbranch),
            "seed_total_dv_mps": float(seed.total_dv_mps),
        }
    except Exception:
        v_depart = as_vec3(x0.v_mps)
        v_arrive = as_vec3(vf_target)
        seed_info = {"guess_kind": "chemical_burn_linear_fallback"}

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


def _prepare_phase_guess(
    phase: Phase,
    guess: Sequence[np.ndarray],
    *,
    carries_mass: bool = False,
) -> tuple[list[np.ndarray], int, int, bool]:
    """Return a guess with the row shape required by the phase dynamics.

    Public guess builders produce ``[R,V,t]`` rows. ASSET phase construction
    needs rows matching the selected ODE state/control dimensions. This helper
    dispatches to burn or mass-coast augmentation and returns the dimensions for
    downstream time bounds, trajectory extraction, and mass bookkeeping.
    """
    state_dim, control_dim, is_burn = _compile_phase_dimensions(phase, carries_mass=carries_mass)
    if not carries_mass:
        return [np.asarray(row, dtype=float) for row in guess], state_dim, control_dim, is_burn

    spacecraft = phase.spacecraft
    if isinstance(spacecraft, str) or spacecraft is None:
        raise ValueError(f"Mass-carrying phase {phase.name!r} requires a Spacecraft object.")
    if not is_burn:
        return (
            _augment_guess_for_mass_coast(
                guess,
                phase=phase,
                mass0_kg=float(spacecraft.initial_mass_kg),
            ),
            state_dim,
            control_dim,
            is_burn,
        )

    thruster = _first_thruster(phase)
    if float(thruster.thrust_N) <= 0.0 or float(thruster.isp_s) <= 0.0:
        raise ValueError(f"Chemical burn phase {phase.name!r} requires thrust_N > 0 and isp_s > 0.")
    return (
        _augment_guess_for_chemical_burn(
            guess,
            phase=phase,
            mass0_kg=float(spacecraft.initial_mass_kg),
            thrust_N=float(thruster.thrust_N),
            isp_s=float(thruster.isp_s),
        ),
        state_dim,
        control_dim,
        is_burn,
    )


def _make_asset_phase(
    phase: Phase,
    guess: Sequence[np.ndarray],
    nsegs: int,
    *,
    carries_mass: bool = False,
    third_body_tables: dict[str, ThirdBodyTable] | None = None,
):
    """Create an ASSET phase and return dimensional metadata.

    This is the final boundary between Octavian mission objects and ASSET phase
    objects. It chooses the concrete ODE, reshapes the guess rows, applies
    third-body tables for the phase, and returns the metadata needed by later
    compiler passes.
    """
    prepared_guess, state_dim, control_dim, is_burn = _prepare_phase_guess(
        phase,
        guess,
        carries_mass=carries_mass,
    )
    ode = _ode_for_phase(
        phase,
        carries_mass=carries_mass,
        third_body_tables=tables_for_phase(phase, third_body_tables or {}),
    )
    return ode.phase(Tmodes.LGL3, prepared_guess, int(nsegs)), state_dim, control_dim, is_burn


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

    _validate_chemical_burn_transfer(phases)
    mass_state_indices = _mass_state_phase_indices(phases)

    for ph in phases:
        normalized_mode = (ph.mode or "").lower().replace("-", "_")
        if normalized_mode not in ("coast", "transfer", "rendezvous", "burn", "chemical_burn", "finite_burn"):
            raise NotImplementedError(
                "Composable solver supports coast-like and chemical-burn phases. "
                f"Got mode={ph.mode!r}"
            )

    minimize_dv, w_dv, minimize_time, w_time = _objective_weights(mission)

    # Normalize time bounds (absolute Back-time bounds for each phase).
    abs_bounds = normalize_time_bounds(phases)
    third_body_tables = build_third_body_tables(mission, phases, abs_bounds)

    # Units / scaling: use default_units() with a tiny dummy spec carrying x0/xf.
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
            # approximate time bounds from last phase if available
            last_bounds = abs_bounds[-1]
            self.tf_bounds_s = last_bounds if last_bounds is not None else (0.0, 10.0)

    r_unit, v_unit, t_unit = default_units(_UnitSpec(x0_for_units, xf_for_units, mu))

    # Build guesses: handle common 2-phase precoast+transfer case for better robustness
    nsegs0 = int(getattr(mission, "mesh_nsegs_precoast", 30))
    nsegs1 = int(getattr(mission, "mesh_nsegs_transfer", 60))

    guesses: dict[int, list[np.ndarray]] = {}
    guess_info: dict[int, dict[str, float | int | bool | str]] = {}

    if mass_state_indices and len(phases) == 3:
        chemical_guesses, chemical_info = _build_guess_three_phase_chemical_transfer(
            phases,
            mu=mu,
            abs_bounds=abs_bounds,
            nsegs_burn=max(4, int(nsegs0 // 2)),
            nsegs_coast=nsegs1,
            lambert_grid_size=int(getattr(mission, "lambert_grid_size", 60)),
            nrevs_to_try=tuple(int(x) for x in getattr(mission, "nrevs_to_try", (0, 1))),
        )
        guesses.update(chemical_guesses)
        guess_info.update(chemical_info)

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
                        built[-1].state_dim,
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
                asset_phase, state_dim, control_dim, is_burn = _make_asset_phase(
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
                        state_dim=state_dim,
                        control_dim=control_dim,
                        is_chemical_burn=is_burn,
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
                        built[-1].state_dim,
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

        asset_phase, state_dim, control_dim, is_burn = _make_asset_phase(
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
                state_dim=state_dim,
                control_dim=control_dim,
                is_chemical_burn=is_burn,
            )
        )

    if shell_phase is not None:
        last_guess = guesses.get(len(phases) - 1)
        if last_guess is None:
            last_guess = _trajectory_rvt(
                np.asarray(built[-1].asset_phase.returnTraj(), dtype=float),
                built[-1].state_dim,
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
        shell_asset_phase, state_dim, control_dim, is_burn = _make_asset_phase(
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
                state_dim=state_dim,
                control_dim=control_dim,
                is_chemical_burn=is_burn,
                enable_adaptive_mesh=False,
            )
        )

    # Set scaling & mesh options
    opts = options or SolverOptions()
    ocp.optimizer.PrintLevel = int(opts.print_level)
    ocp.optimizer.MaxLSIters = int(opts.max_ls_iters)
    ocp.optimizer.set_QPOrderingMode(str(opts.qp_ordering_mode))
    set_ocp_threads(ocp, opts.asset_threads)

    for b in built:
        b.asset_phase.setAutoScaling(bool(opts.enable_auto_scaling))
        b.asset_phase.setUnits(R=r_unit, V=v_unit, t=t_unit)
        b.asset_phase.setAdaptiveMesh(bool(opts.enable_adaptive_mesh and b.enable_adaptive_mesh))

    ocp.setAutoScaling(True, True)
    ocp.setAdaptiveMesh(bool(opts.enable_adaptive_mesh))
    ocp.PrintMeshInfo = False

    # Apply constraints and time bounds per phase
    for b in built:
        ph = b.compile_phase or b.ph
        ap = b.asset_phase

        # First phase front time fixed at 0 unless user provides otherwise
        if b.index == 0:
            fix_front_time(ap, 0.0)

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
            if b.is_chemical_burn:
                ap.addUpperNormBound("Path", "U", 1.0)

        # Apply State/Position boundary constraints with impulsive-variable override logic
        for loc in ("Front", "Back"):
            st = constraint_compiler.get_state_constraint(ph, loc)
            pos = constraint_compiler.get_position_constraint(ph, loc)

            # Position constraint always applied (if present)
            if pos is not None:
                ap.addBoundaryValue(
                    loc,
                    ["R"],
                    np.asarray(
                        as_vec3(constraint_compiler.position_boundary_value(pos)),
                        dtype=float,
                    ),
                )

            if st is not None:
                st_val = constraint_compiler.state_boundary_value(st)
                groups = constraint_compiler.state_groups(st)

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
        for constraint in getattr(ph, "constraints", []) or []:
            if getattr(constraint, "kind", "") == "min_radius":
                minimum_radius_m = float(constraint.value)
                location = (
                    "Path"
                    if getattr(constraint, "where", "Path") == "Path"
                    else getattr(constraint, "where", "Path")
                )
                ap.addLowerNormBound(location, "R", minimum_radius_m)
            elif isinstance(constraint, OrbitalElementConstraint):
                constraint_compiler.apply_orbital_element_constraint(ap, constraint, mu)

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
        for b in built:
            if not b.is_chemical_burn:
                continue
            spacecraft = b.ph.spacecraft
            if isinstance(spacecraft, str) or spacecraft is None:
                continue
            mass_argument = vf.Arguments(1).tolist()[0]
            b.asset_phase.addStateObjective(
                "Back",
                -float(w_dv) * mass_argument,
                [6],
                [],
                [],
                AutoScale=1.0 / max(float(spacecraft.initial_mass_kg), 1.0),
            )

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
        last_build = built[-1]
        last_ap = last_build.asset_phase
        at = vf.Arguments(1).tolist()[0]
        last_ap.addStateObjective(
            "Back",
            float(w_time) * at,
            [last_build.state_dim],
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
    trajs = [_trajectory_rvt(raw_traj, b.state_dim) for raw_traj, b in zip(raw_trajs, built, strict=True)]
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

    chemical_burns: list[dict[str, float | str]] = []
    phase_segments: list[dict[str, float | str]] = []
    for build, phase_traj in zip(built, trajs, strict=True):
        is_burn = bool(build.is_chemical_burn)
        phase_segments.append(
            {
                "name": build.ph.name,
                "mode": "chemical_burn" if is_burn else "coast",
                "t_start_s": float(phase_traj[0, 6]),
                "t_end_s": float(phase_traj[-1, 6]),
                "color": "red" if is_burn else "blue",
            }
        )
        if not build.is_chemical_burn:
            continue
        raw_traj = raw_trajs[build.index]
        mass_initial_kg = float(raw_traj[0, 6])
        mass_final_kg = float(raw_traj[-1, 6])
        propellant_used_kg = max(0.0, mass_initial_kg - mass_final_kg)
        equivalent_dv_mps = 0.0
        spacecraft = build.ph.spacecraft
        if not isinstance(spacecraft, str) and spacecraft is not None and mass_final_kg > 0.0:
            thruster = _first_thruster(build.ph)
            equivalent_dv_mps = float(thruster.isp_s) * 9.80665 * float(
                np.log(max(mass_initial_kg, mass_final_kg) / mass_final_kg)
            )
        chemical_burns.append(
            {
                "phase": build.ph.name,
                "t_start_s": float(phase_traj[0, 6]),
                "t_end_s": float(phase_traj[-1, 6]),
                "mass_initial_kg": mass_initial_kg,
                "mass_final_kg": mass_final_kg,
                "propellant_used_kg": propellant_used_kg,
                "equivalent_dv_mps": equivalent_dv_mps,
            }
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
            "constraint_report": constraint_report,
            "chemical_burns": chemical_burns,
            "phase_segments": phase_segments,
            "phase_guess_info": guess_info,
        }
        | guess_info.get(0, {}),
    )
