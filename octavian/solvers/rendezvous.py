from __future__ import annotations

"""Impulsive rendezvous solvers built on ASSET's UpdatedInterface.

Key correctness notes (from ASSET source/examples):
- **Time is not a state**. Time is the phase time variable (ODEArguments.TVar()) and
  appears in the phase variable stack after X: for XVars=6, the time index is 6.
- Use **Vgroups** in the ODE ("R", "V", "t"/"time") so constraints can refer
  to semantic groups (see `examples/UpdatedInterface/Delta3Launch.py` in ASSET).

This module keeps everything dimensional at the user API, and relies on ASSET auto-scaling.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Union

import numpy as np
from numpy.typing import NDArray

import asset_asrl as ast  # type: ignore

from ..dynamics import TwoBodyECI
from ..specs import TwoImpulseFreeTimeSpec, TwoImpulsePreCoastSpec
from ..types import Maneuver
from ..astro.types import as_vec3, Vec3
from ..astro.units import default_units
from ..astro.lambert import LambertSeed, select_best_lambert_seed
from ..astro.kepler import kepler_dense_guess, estimate_orbital_period_s, propagate_cartesian_rv

vf = ast.VectorFunctions
oc = ast.OptimalControl
Tmodes = oc.TranscriptionModes

TrajArray = NDArray[np.float64]


@dataclass
class RendezvousResult:
    """Common result returned by rendezvous solvers."""
    converged: bool
    traj: TrajArray
    maneuvers: List[Maneuver] = field(default_factory=list)
    last_obj: float = float("nan")
    info: Dict[str, Any] = field(default_factory=dict)


def solve(spec: Union[TwoImpulseFreeTimeSpec, TwoImpulsePreCoastSpec]) -> RendezvousResult:
    """Dispatch to the appropriate rendezvous solver."""
    if isinstance(spec, TwoImpulseFreeTimeSpec):
        return solve_two_impulse_free_time(spec)
    if isinstance(spec, TwoImpulsePreCoastSpec):
        return solve_two_impulse_precoast(spec)
    raise TypeError(f"Unsupported spec type: {type(spec).__name__}")


def solve_two_impulse_free_time(spec: TwoImpulseFreeTimeSpec) -> RendezvousResult:
    """Two-impulse rendezvous with a single coast phase and bounded free final time."""
    tfmin, tfmax = map(float, spec.tf_bounds_s)
    if not (tfmin > 0.0 and tfmax > tfmin):
        raise ValueError("tf_bounds_s must satisfy 0 < tfmin < tfmax")

    r_unit, v_unit, t_unit = default_units(spec)

    ode = TwoBodyECI(mu_m3ps2=float(spec.mu_m3ps2))
    t0 = 0.0

    seed = select_best_lambert_seed(
        r0_m=as_vec3(spec.x0.r_m),
        rf_m=as_vec3(spec.xf.r_m),
        v0_mps=as_vec3(spec.x0.v_mps),
        vf_mps=as_vec3(spec.xf.v_mps),
        mu_m3ps2=float(spec.mu_m3ps2),
        tmin_s=tfmin,
        tmax_s=tfmax,
        n_tofs=int(spec.lambert_grid_size),
        nrevs=tuple(int(n) for n in spec.nrevs_to_try),
    )

    tf_guess = float(spec.tf_guess_s) if spec.tf_guess_s is not None else float(seed.tof_s)
    tf_guess = min(max(tf_guess, tfmin), tfmax)

    # Initial guess: Kepler propagation from Lambert departure velocity
    ig = kepler_dense_guess(
        r0_m=as_vec3(spec.x0.r_m),
        v0_mps=as_vec3(seed.v1_mps),
        t0_s=t0,
        tf_s=tf_guess,
        npts=int(spec.nsegs) + 1,
        mu_m3ps2=float(spec.mu_m3ps2),
    )

    phase = ode.phase(Tmodes.LGL3, ig, int(spec.nsegs))


    # Constraints: fix start R and time, fix end R, bound end time (time index = 6)
    phase.addBoundaryValue("Front", ["R", "t"], np.hstack([as_vec3(spec.x0.r_m), [t0]]))
    phase.addBoundaryValue("Back", ["R"], as_vec3(spec.xf.r_m))

    # Back time bound
    phase.addLUVarBound("Back", "time", tfmin, tfmax)

    # Objectives: ||dv1||^2 + ||dv2||^2 (+ w_time * tf).
    # We construct objectives in dimensional units and supply explicit AutoScale so that
    # state and link/terminal terms are scaled consistently.
    v0 = as_vec3(spec.x0.v_mps)
    vf_ = as_vec3(spec.xf.v_mps)

    a = vf.Arguments(3)
    d1 = a - v0
    dv1_sq = vf.sqrt(d1.dot(d1))

    b = vf.Arguments(3)
    d2 = vf_ - b
    dv2_sq = vf.sqrt(d2.dot(d2))

    vel_obj_scale = float(v_unit)  # (m/s)^2

    phase.addStateObjective("Front", dv1_sq, [3, 4, 5], [], [], vel_obj_scale)
    phase.addStateObjective("Back", dv2_sq, [3, 4, 5], [], [], vel_obj_scale)

    phase.addLowerDeltaTimeBound(0.1)

    if float(spec.w_time) != 0.0:
        at = vf.Arguments(1).tolist()[0]
        phase.addStateObjective("Back", float(spec.w_time) * at, [6], [], [], float(t_unit))

    ocp = oc.OptimalControlProblem()
    ocp.addPhase(phase)

    ocp.optimizer.PrintLevel = 0
    ocp.optimizer.MaxLSIters = 2
    ocp.optimizer.set_QPOrderingMode("MINDEG")

    phase.setAutoScaling(True)
    phase.setUnits(R=r_unit, V=v_unit, t=t_unit)
    phase.setAdaptiveMesh(True)
    ocp.setAutoScaling(True,True)
    ocp.setAdaptiveMesh(True)

    converged = bool(ocp.solve_optimize()) if hasattr(ocp, "solve_optimize") else bool(ocp.optimize())

    traj = np.asarray(phase.returnTraj(), dtype=np.float64)
    # traj columns for XVars=6,UVars=0 typically: [x0..x5, t]
    v_front = traj[0, 3:6]
    v_back = traj[-1, 3:6]

    dv1 = v_front - v0
    dv2 = vf_ - v_back

    maneuvers = [
        Maneuver(r_m=traj[0, 0:3], t_s=float(traj[0, 6]), dv_mps=dv1, name="Δv1 (start)"),
        Maneuver(r_m=traj[-1, 0:3], t_s=float(traj[-1, 6]), dv_mps=dv2, name="Δv2 (end)"),
    ]

    return RendezvousResult(
        converged=converged,
        traj=traj,
        maneuvers=maneuvers,
        last_obj=float(getattr(ocp.optimizer, "LastObjVal", np.nan)),
        info={
            "tf_sol_s": float(traj[-1, 6]),
            "r_unit_m": r_unit,
            "v_unit_mps": v_unit,
            "t_unit_s": t_unit,
            "seed_tof_s": seed.tof_s,
            "seed_longway": seed.longway,
            "seed_nrev": seed.nrev,
            "seed_rightbranch": seed.rightbranch,
            "seed_total_dv_mps": seed.total_dv_mps,
        },
    )


def solve_two_impulse_precoast(spec: TwoImpulsePreCoastSpec) -> RendezvousResult:
    """Two-impulse rendezvous with variable pre-coast before the first impulse."""
    t1min, t1max = map(float, spec.t1_bounds_s)
    tfmin, tfmax = map(float, spec.tf_bounds_s)

    if not (t1min >= 0.0 and t1max > t1min):
        raise ValueError("t1_bounds_s must satisfy 0 <= t1min < t1max")
    if not (tfmin > 0.0 and tfmax > tfmin):
        raise ValueError("tf_bounds_s must satisfy 0 < tfmin < tfmax")
    if tfmax <= t1min:
        raise ValueError("tf must be after t1: require tfmax > t1min")

    r_unit, v_unit, t_unit = default_units(spec)
    mu = float(spec.mu_m3ps2)
    ode = TwoBodyECI(mu_m3ps2=mu)
    t0 = 0.0

    # --- Candidate t1 sweep using Kepler propagation
    n_t1 = max(int(spec.precoast_grid_size), 2)
    t1_candidates = np.linspace(t1min, t1max, n_t1)

    if bool(spec.limit_precoast_to_one_period):
        T0 = estimate_orbital_period_s(spec.x0.r_m, spec.x0.v_mps, mu)
        if T0 is not None:
            span = t1max - t1min
            if span > 1.5 * T0:
                t1_candidates = np.linspace(t1min, min(t1min + T0, t1max), n_t1)

    rv0 = np.hstack([as_vec3(spec.x0.r_m), as_vec3(spec.x0.v_mps)])

    best = None
    for t1_try in t1_candidates:
        try:
            rv1 = propagate_cartesian_rv(rv0, float(t1_try - t0), mu)
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
            rf_m=as_vec3(spec.xf.r_m),
            v0_mps=as_vec3(v1_minus),
            vf_mps=as_vec3(spec.xf.v_mps),
            mu_m3ps2=mu,
            tmin_s=float(dtmin),
            tmax_s=float(dtmax),
            n_tofs=int(spec.lambert_grid_size),
            nrevs=tuple(int(n) for n in spec.nrevs_to_try),
        )


        dv1 = as_vec3(seed.v1_mps) - as_vec3(v1_minus)
        dv2 = as_vec3(spec.xf.v_mps) - as_vec3(seed.v2_mps)
        score = float(np.linalg.norm(dv1) + np.linalg.norm(dv2))
        if best is None or score < best["score"]:
            best = {"t1": float(t1_try), "rv1": rv1, "seed": seed, "score": score}

    if best is None:
        raise RuntimeError("Failed to find any feasible pre-coast + Lambert seed.")

    t1_guess = float(best["t1"])
    rv1_guess = np.asarray(best["rv1"], dtype=float).reshape(6)
    r1_guess = rv1_guess[0:3]
    v1_minus_guess = rv1_guess[3:6]
    seed: LambertSeed = best["seed"]

    tf_guess = float(t1_guess + seed.tof_s)
    v1_plus_guess = as_vec3(seed.v1_mps)

    # Dense guesses for both phases
    ig0 = kepler_dense_guess(
        r0_m=as_vec3(spec.x0.r_m),
        v0_mps=as_vec3(spec.x0.v_mps),
        t0_s=t0,
        tf_s=t1_guess,
        npts=int(spec.nsegs_precoast) + 1,
        mu_m3ps2=mu,
    )
    ig1 = kepler_dense_guess(
        r0_m=as_vec3(r1_guess),
        v0_mps=v1_plus_guess,
        t0_s=t1_guess,
        tf_s=tf_guess,
        npts=int(spec.nsegs_transfer) + 1,
        mu_m3ps2=mu,
    )

    phase0 = ode.phase(Tmodes.LGL3, ig0, int(spec.nsegs_precoast))
    phase1 = ode.phase(Tmodes.LGL3, ig1, int(spec.nsegs_transfer))

    

    # Phase 0 boundary: fix full state and t0, bound t1 (time index 6)
    phase0.addBoundaryValue("Front", ["R", "V", "t"], np.hstack([as_vec3(spec.x0.r_m), as_vec3(spec.x0.v_mps), [t0]]))
    try:
        phase0.addLUVarBound("Back", "time", t1min, t1max)
    except Exception:
        phase0.addLUVarBound("Back", 6, t1min, t1max)
    phase0.addLowerDeltaTimeBound(float(spec.min_dt_precoast_s))

    # Phase 1 boundary: fix final position, bound tf
    phase1.addBoundaryValue("Back", ["R"], as_vec3(spec.xf.r_m))
    phase1.addLUVarBound("Back", "time", tfmin, tfmax)
    phase1.addLowerDeltaTimeBound(float(spec.min_dt_transfer_s))

    # phase0

    ocp = oc.OptimalControlProblem()
    ocp.addPhase(phase0)
    ocp.addPhase(phase1)

    # Link constraints: continuity of position and time at burn
    ocp.addForwardLinkEqualCon(phase0, phase1, ["R", "t"])

    # Objectives:
    # Link Δv1: minimize ||v_plus - v_minus||^2 at link.
    a = vf.Arguments(6)
    v_minus = a.head(3)
    v_plus = a.segment(3, 3)
    dv = v_plus - v_minus
    dv1_sq = vf.sqrt(dv.dot(dv))

    vel_obj_scale = float(v_unit)
    ocp.addLinkObjective(
        dv1_sq,
        phase0, "Back",  [3, 4, 5], [], [],
        phase1, "Front", [3, 4, 5], [], [],
        [],
        vel_obj_scale,
    )

    # Terminal Δv2: minimize ||vf - v(tf-)||^2
    vf_ = as_vec3(spec.xf.v_mps)
    b = vf.Arguments(3)
    dv2_sq = vf.sqrt((vf_ - b).dot(vf_ - b))
    phase1.addStateObjective("Back", dv2_sq, [3, 4, 5], [], [], vel_obj_scale)

    if float(spec.w_time) != 0.0:
        at = vf.Arguments(1).tolist()[0]
        phase1.addStateObjective("Back", float(spec.w_time) * at, [6], [], [], float(t_unit))

    ocp.optimizer.PrintLevel = 0
    ocp.optimizer.MaxLSIters = 2
    ocp.optimizer.set_QPOrderingMode("MINDEG")

    # ocp.optimizer.set_EContol(tol)
    ocp.optimizer.set_AccKKTtol(1e-6)

    for ph in (phase0, phase1):
        ph.setAutoScaling(True)
        ph.setUnits(R=r_unit, V=v_unit, t=t_unit)
        ph.setAdaptiveMesh(True)
    
    ocp.setAutoScaling(True,True)
    ocp.setAdaptiveMesh(True)

    converged = ocp.solve()
    converged = ocp.optimize()

    traj0 = np.asarray(phase0.returnTraj(), dtype=np.float64)
    traj1 = np.asarray(phase1.returnTraj(), dtype=np.float64)
    traj = np.vstack([traj0, traj1[1:, :]])

    v1m = traj0[-1, 3:6]
    v1p = traj1[0, 3:6]
    dv1 = v1p - v1m

    v2m = traj1[-1, 3:6]
    dv2 = vf_ - v2m

    maneuvers = [
        Maneuver(r_m=traj0[-1, 0:3], t_s=float(traj0[-1, 6]), dv_mps=dv1, name="Δv1 (link)"),
        Maneuver(r_m=traj1[-1, 0:3], t_s=float(traj1[-1, 6]), dv_mps=dv2, name="Δv2 (end)"),
    ]

    return RendezvousResult(
        converged=converged,
        traj=traj,
        maneuvers=maneuvers,
        last_obj=float(getattr(ocp.optimizer, "LastObjVal", np.nan)),
        info={
            "t1_sol_s": float(traj0[-1, 6]),
            "tf_sol_s": float(traj1[-1, 6]),
            "r_unit_m": r_unit,
            "v_unit_mps": v_unit,
            "t_unit_s": t_unit,
            "seed_dt_s": seed.tof_s,
            "seed_longway": seed.longway,
            "seed_nrev": seed.nrev,
            "seed_rightbranch": seed.rightbranch,
            "seed_total_dv_mps": seed.total_dv_mps,
            "precoast_seed_score_mps": float(best["score"]),
        },
    )
