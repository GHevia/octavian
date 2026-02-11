from __future__ import annotations

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

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import asset_asrl as ast  # type: ignore
except Exception:  # pragma: no cover
    ast = None  # type: ignore

from .options import SolverOptions
from .rendezvous import RendezvousResult  # reuse stable result type
from ..types import Maneuver
from ..constraints import State as StateConstraint, Position as PositionConstraint
from ..variables import ImpulsiveDeltaV
from ..phase import Phase
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ..mission import Mission
from ..dynamics import TwoBodyECI
from ..astro.types import as_vec3
from ..astro.units import default_units
from ..astro.kepler import kepler_dense_guess, propagate_cartesian_rv, estimate_orbital_period_s
from ..astro.lambert import select_best_lambert_seed


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
    t_bounds: Tuple[float, float]
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


def _get_state_constraint(phase: Phase, where: str) -> Optional[StateConstraint]:
    loc = "Front" if where.lower().startswith("f") else "Back"
    for c in getattr(phase, "constraints", []) or []:
        if isinstance(c, StateConstraint) and getattr(c, "where", "") == loc:
            return c
    return None


def _get_position_constraint(phase: Phase, where: str) -> Optional[PositionConstraint]:
    loc = "Front" if where.lower().startswith("f") else "Back"
    for c in getattr(phase, "constraints", []) or []:
        if isinstance(c, PositionConstraint) and getattr(c, "where", "") == loc:
            return c
    return None


def _objective_weights(mission: Mission) -> Tuple[bool, float, bool, float]:
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
    nsegs0: int,
    nsegs1: int,
    precoast_grid_size: int,
    limit_precoast_to_one_period: bool,
    lambert_grid_size: int,
    nrevs_to_try: Sequence[int],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Specialized guess builder for (precoast, transfer) with terminal position.

    This is used only for guesses; the compiled problem is still generic.
    """
    if p0.initial_state is None and _get_state_constraint(p0, "Front") is None:
        raise ValueError("Precoast phase requires an initial state (phase.initial_state or State constraint at Front).")

    x0 = p0.initial_state or _get_state_constraint(p0, "Front").x  # type: ignore[union-attr]
    xf_state = _get_state_constraint(p1, "Back")
    xf_pos = _get_position_constraint(p1, "Back")

    if xf_state is None and xf_pos is None and p1.final_state is None:
        raise ValueError("Transfer phase requires a terminal position (State/Position constraint or phase.final_state).")

    rf = as_vec3((xf_state.x.r_m if xf_state is not None else (p1.final_state.r_m if p1.final_state is not None else xf_pos.r_m)))  # type: ignore[arg-type]
    vf_target = as_vec3((xf_state.x.v_mps if xf_state is not None else (p1.final_state.v_mps if p1.final_state is not None else x0.v_mps)))

    t1min, t1max = p0.tof_bounds_s or (0.0, 1800.0)
    tfmin, tfmax = p1.tof_bounds_s or (600.0, 7200.0)

    t1min = float(t1min); t1max = float(t1max)
    tfmin = float(tfmin); tfmax = float(tfmax)

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
        ig0 = kepler_dense_guess(r0_m=as_vec3(x0.r_m), v0_mps=as_vec3(x0.v_mps), t0_s=0.0, tf_s=t1_guess, npts=nsegs0 + 1, mu_m3ps2=mu)
        # start phase1 at end of ig0 with same velocity
        rv1 = np.hstack([ig0[-1][0:3], ig0[-1][3:6]])
        ig1 = kepler_dense_guess(r0_m=as_vec3(rv1[0:3]), v0_mps=as_vec3(rv1[3:6]), t0_s=t1_guess, tf_s=tf_guess, npts=nsegs1 + 1, mu_m3ps2=mu)
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


def solve_composable_mission(
    mission: Mission,
    *,
    options: Optional[SolverOptions] = None,
) -> RendezvousResult:
    """Solve a composable mission via ASSET compilation."""
    _require_asset()

    phases = list(mission.phases)
    if not phases:
        raise ValueError("Mission has no phases")

    # Currently: only coast phases supported
    for ph in phases:
        if (ph.mode or "").lower() not in ("coast", "transfer", "rendezvous"):
            raise NotImplementedError(f"Composable solver currently supports only coast-like phases. Got mode={ph.mode!r}")

    minimize_dv, w_dv, minimize_time, w_time = _objective_weights(mission)

    # Units / scaling: use default_units() with a tiny dummy spec carrying x0/xf.
    first = phases[0]
    last = phases[-1]
    mu = float(first.dynamics.mu_m3ps2)  # type: ignore[union-attr]

    x0_for_units = (
        first.initial_state
        or (_get_state_constraint(first, "Front").x if _get_state_constraint(first, "Front") else None)
        or (_get_state_constraint(first, "Back").x if _get_state_constraint(first, "Back") else None)
    )
    xf_for_units = (
        last.final_state
        or (_get_state_constraint(last, "Back").x if _get_state_constraint(last, "Back") else None)
        or (_get_state_constraint(last, "Front").x if _get_state_constraint(last, "Front") else None)
    )

    if x0_for_units is None or xf_for_units is None:
        raise ValueError("Composable mission needs boundary State information (x0 and xf) to choose scaling units.")

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
            self.tf_bounds_s = getattr(last, "tof_bounds_s", (0.0, 10.0))

    r_unit, v_unit, t_unit = default_units(_UnitSpec(x0_for_units, xf_for_units, mu))

    ode = TwoBodyECI(mu_m3ps2=mu)

    # Build guesses: handle common 2-phase precoast+transfer case for better robustness
    nsegs0 = int(getattr(mission, "mesh_nsegs_precoast", 30))
    nsegs1 = int(getattr(mission, "mesh_nsegs_transfer", 60))

    guesses: Dict[str, List[np.ndarray]] = {}

    if len(phases) == 2 and (phases[0].mode or "").lower() == "coast" and phases[1].previous is not None:
        p0, p1 = phases
        ig0, ig1 = _build_guess_two_phase_precoast_transfer(
            p0, p1,
            mu=mu,
            nsegs0=nsegs0,
            nsegs1=nsegs1,
            precoast_grid_size=int(getattr(mission, "precoast_grid_size", 10)),
            limit_precoast_to_one_period=bool(getattr(mission, "limit_precoast_to_one_period", True)),
            lambert_grid_size=int(getattr(mission, "lambert_grid_size", 60)),
            nrevs_to_try=tuple(int(x) for x in getattr(mission, "nrevs_to_try", (0,1))),
        )
        guesses[p0.name] = ig0
        guesses[p1.name] = ig1

    # Compile phases
    ocp = oc.OptimalControlProblem()
    built: List[_PhaseBuild] = []

    for idx, ph in enumerate(phases):
        # determine guess
        if ph.name in guesses:
            ig = guesses[ph.name]
            # infer nsegs from guess length - 1
            nsegs = len(ig) - 1
        else:
            nsegs = nsegs0 if idx == 0 else nsegs1
            # fallback midpoint guess using available initial state
            if idx == 0:
                x0 = ph.initial_state or (_get_state_constraint(ph, "Front").x if _get_state_constraint(ph, "Front") else None)
                if x0 is None:
                    raise ValueError("First phase must have an initial_state or State constraint at Front.")
                tf_guess = float(ph.tof_bounds_s[1] if ph.tof_bounds_s else 1800.0) * 0.5
                ig = kepler_dense_guess(r0_m=as_vec3(x0.r_m), v0_mps=as_vec3(x0.v_mps), t0_s=0.0, tf_s=tf_guess, npts=nsegs+1, mu_m3ps2=mu)
            else:
                # start from previous guess end
                prev = built[-1].ph
                prev_guess = guesses.get(prev.name)
                if prev_guess is None:
                    prev_guess = np.asarray(built[-1].asset_phase.returnTraj(), dtype=float).tolist()  # type: ignore
                rv_start = np.asarray(prev_guess[-1])[0:6]
                t_start = float(np.asarray(prev_guess[-1])[6])
                tf_guess = float(ph.tof_bounds_s[1] if ph.tof_bounds_s else (t_start + 3600.0))
                ig = kepler_dense_guess(r0_m=as_vec3(rv_start[0:3]), v0_mps=as_vec3(rv_start[3:6]), t0_s=t_start, tf_s=tf_guess, npts=nsegs+1, mu_m3ps2=mu)

        asset_phase = ode.phase(Tmodes.LGL3, ig, int(nsegs))
        ocp.addPhase(asset_phase)

        built.append(_PhaseBuild(ph=ph, asset_phase=asset_phase, t_bounds=tuple(ph.tof_bounds_s or (0.0, 1.0)), index=idx))

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
            try:
                ap.addBoundaryValue("Front", ["t"], np.asarray([0.0], dtype=float))
            except Exception:
                pass

        # Apply State/Position boundary constraints with impulsive-variable override logic
        for loc in ("Front", "Back"):
            st = _get_state_constraint(ph, loc)
            pos = _get_position_constraint(ph, loc)

            # Position constraint always applied (if present)
            if pos is not None:
                ap.addBoundaryValue(loc, ["R"], np.asarray(as_vec3(pos.r_m), dtype=float))

            if st is not None:
                groups = tuple(getattr(st, "groups", ("R","V")))

                # If an impulsive Δv is declared at this boundary, we drop V from the constraint
                if _has_impulsive_var(ph, loc) and "V" in groups:
                    groups = tuple(g for g in groups if g != "V")

                # build value vector for groups
                vals = []
                use_groups = []
                for g in groups:
                    if g == "R":
                        vals.extend(list(as_vec3(st.x.r_m)))
                        use_groups.append("R")
                    elif g == "V":
                        vals.extend(list(as_vec3(st.x.v_mps)))
                        use_groups.append("V")
                    elif g in ("t","time"):
                        # if user requested fixing time explicitly, we honor it if phase has epoch semantics (not yet)
                        # for now ignore unless first phase front (already fixed)
                        use_groups.append("t")
                        vals.append(float(0.0 if (b.index==0 and loc=="Front") else np.nan))
                    else:
                        raise ValueError(f"Unsupported State.groups element: {g!r}")

                # Only apply if we have something concrete
                if use_groups:
                    if "t" in use_groups:
                        # if time is nan we skip setting time
                        if not np.isfinite(vals[-1]):
                            use_groups = [g for g in use_groups if g != "t"]
                            vals = vals[:-1]
                    if use_groups:
                        ap.addBoundaryValue(loc, use_groups, np.asarray(vals, dtype=float))

        # Time bounds: interpret phase.tof_bounds_s as absolute bounds on Back time
        if ph.tof_bounds_s is not None:
            tmin, tmax = map(float, ph.tof_bounds_s)
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
            ocp.addForwardLinkEqualCon(prev_ap, ap, ["R","V","t"])
        else:
            ocp.addForwardLinkEqualCon(prev_ap, ap, ["R","t"])

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
                prev_ap, "Back",  [3,4,5], [], [],
                ap,     "Front", [3,4,5], [], [],
                [], float(v_unit),
            )

    # Boundary Δv objectives (mission start and terminal)
    if minimize_dv:
        # Start: if first phase has impulsive Δv at Front, penalize relative to desired initial velocity
        first_ph = built[0].ph
        first_ap = built[0].asset_phase
        if _has_impulsive_var(first_ph, "Front"):
            st = _get_state_constraint(first_ph, "Front")
            if st is None and first_ph.initial_state is None:
                raise ValueError("ImpulsiveDeltaV at mission start requires a desired initial velocity (State constraint or initial_state).")
            v_target = as_vec3((st.x.v_mps if st is not None else first_ph.initial_state.v_mps))  # type: ignore[union-attr]
            a0 = vf.Arguments(3)
            dv0 = vf.sqrt((a0 - v_target).dot(a0 - v_target))
            first_ap.addStateObjective("Front", float(w_dv) * dv0, [3,4,5], [], [], float(v_unit))

        # Terminal: if last phase has impulsive Δv at Back, penalize relative to desired terminal velocity
        last_ph = built[-1].ph
        last_ap = built[-1].asset_phase
        if _has_impulsive_var(last_ph, "Back"):
            st = _get_state_constraint(last_ph, "Back")
            if st is None and last_ph.final_state is None:
                raise ValueError("ImpulsiveDeltaV at mission end requires a desired terminal velocity (State constraint or final_state).")
            v_target = as_vec3((st.x.v_mps if st is not None else last_ph.final_state.v_mps))  # type: ignore[union-attr]
            b0 = vf.Arguments(3)
            dvf = vf.sqrt((v_target - b0).dot(v_target - b0))
            last_ap.addStateObjective("Back", float(w_dv) * dvf, [3,4,5], [], [], float(v_unit))

    # Time objective: minimize final time at last phase Back
    if minimize_time and float(w_time) != 0.0:
        last_ap = built[-1].asset_phase
        at = vf.Arguments(1).tolist()[0]
        last_ap.addStateObjective("Back", float(w_time) * at, [6], [], [], float(t_unit))

    # Solve
    converged = ocp.solve_optimize_solve() if hasattr(ocp, "solve_optimize_solve") else ocp.optimize_solve()

    # Extract trajectory (stitch)
    trajs = [np.asarray(b.asset_phase.returnTraj(), dtype=float) for b in built]
    traj = trajs[0]
    for t in trajs[1:]:
        traj = np.vstack([traj, t[1:,:]])

    # Maneuver bookkeeping: link dv and terminal dv if requested
    maneuvers: List[Maneuver] = []
    # link maneuvers: for each phase with previous and impulsive var at front and link not continuous
    for i, b in enumerate(built):
        ph = b.ph
        if ph.previous is None:
            continue
        link_kind = (ph.link.kind if ph.link is not None else "continuous").lower()
        if link_kind == "continuous" or not _has_impulsive_var(ph, "Front"):
            continue
        prev_traj = trajs[i-1]
        this_traj = trajs[i]
        v_minus = prev_traj[-1, 3:6]
        v_plus = this_traj[0, 3:6]
        dv = v_plus - v_minus
        maneuvers.append(Maneuver(r_m=prev_traj[-1,0:3], t_s=float(prev_traj[-1,6]), dv_mps=dv, name=f"Δv ({ph.name} front/link)"))

    # terminal maneuver
    if built and _has_impulsive_var(built[-1].ph, "Back"):
        st = _get_state_constraint(built[-1].ph, "Back")
        v_target = as_vec3((st.x.v_mps if st is not None else built[-1].ph.final_state.v_mps))  # type: ignore[union-attr]
        v_end = trajs[-1][-1, 3:6]
        dv = v_target - v_end
        maneuvers.append(Maneuver(r_m=trajs[-1][-1,0:3], t_s=float(trajs[-1][-1,6]), dv_mps=dv, name="Δv (terminal)"))

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
        },
    )
