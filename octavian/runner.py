from __future__ import annotations

"""Mission runner.

The runner owns *how* we solve:
  - pattern-map a Mission (phases) to a supported backend solve
  - apply a (possibly empty) continuation plan
  - catch common errors and retry with alternate settings/guesses

v0.x focuses on impulsive rendezvous. The runner is intentionally conservative
so it does not hide real issues, but it can automatically handle common
"transcription"/mesh mismatches and initial-guess sensitivity.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

from .models import RetryPolicy, RunPlan, SolveConfig
from .solution import AttemptLog, Solution
from .specs import TwoImpulseFreeTimeSpec, TwoImpulsePreCoastSpec
from .solvers import SolverOptions
from .solvers.rendezvous import solve as solve_rendezvous
from .solvers.composable import solve_composable_mission
from .constraints import State as StateConstraint, Position as PositionConstraint
from .variables import ImpulsiveDeltaV


class MissionBuildError(ValueError):
    """Raised when a Mission cannot be mapped to a supported solver."""


@dataclass(slots=True)
class MissionRunner:
    solve_options: SolverOptions
    solve_config: SolveConfig = field(default_factory=SolveConfig)
    plan: RunPlan = field(default_factory=RunPlan.default)
    retry: RetryPolicy = field(default_factory=RetryPolicy.default)

    def solve(self, mission: "Mission") -> Solution:
        # deferred import to avoid cycles
        from .mission import Mission

        if not isinstance(mission, Mission):
            raise TypeError("MissionRunner.solve expects a Mission")

        # Validate phases and inherit defaults
        mission.validate()

        attempts = []
        last_error: Optional[str] = None

        stages = list(self.plan.stages) or [None]
        stage_name = lambda s: getattr(s, "name", "default") if s is not None else "default"

        for sidx, stage in enumerate(stages):
            stage_label = stage_name(stage)
            spec = None
            if not _is_composable_mission(mission):
                try:
                    spec = _mission_to_rendezvous_spec(mission)
                except Exception as e:  # noqa: BLE001
                    raise MissionBuildError(str(e)) from e

            # Stage-level adjustment hook (minimal)
            if spec is not None and stage is not None and getattr(stage, "nsegs_scale", None):
                spec = _scale_mesh(spec, float(stage.nsegs_scale))

            max_attempts = max(1, int(self.solve_config.max_attempts))
            for attempt in range(1, max_attempts + 1):
                try:
                    if _is_composable_mission(mission):
                        res = solve_composable_mission(mission, options=self.solve_options)
                    else:
                        res = solve_rendezvous(spec, options=self.solve_options)
                    attempts.append(AttemptLog(stage=stage_label, attempt=attempt, status="ok"))
                    sol = Solution(ok=bool(res.converged), result=res, attempts=attempts)
                    sol.info.update({"stage": stage_label, "stage_index": sidx})
                    if res.converged or not self.solve_config.raise_on_fail:
                        return sol
                    raise RuntimeError("Solver did not converge")
                except Exception as e:  # noqa: BLE001
                    msg = str(e)
                    last_error = msg
                    attempts.append(AttemptLog(stage=stage_label, attempt=attempt, status="fail", message=msg))

                    if not self.retry.enabled or attempt >= max_attempts:
                        break
                    if spec is not None:
                        spec = _apply_simple_retry(spec, attempt, msg)

        sol = Solution(ok=False, result=None, attempts=attempts, last_error=last_error)
        if self.solve_config.raise_on_fail:
            raise RuntimeError(sol.summary())
        return sol


def _mission_to_rendezvous_spec(mission: "Mission") -> TwoImpulseFreeTimeSpec | TwoImpulsePreCoastSpec:
    """Map Mission phases into the currently supported rendezvous specs.

    Supported patterns (v0.x):
      1) Single phase (mode in {"rendezvous","transfer"}) with initial_state + final_state.
      2) Two phases: (mode="coast"), then (mode in {"rendezvous","transfer"}).

    Notes
    -----
    v0.x solvers are *impulsive* formulations:
      - boundary velocities may be free and penalized via Δv objectives
      - between-phase impulses are expressed via link continuity choices
    """

    phases = list(mission.phases)

    # --- objectives (explicit)
    minimize_dv = True
    dv_weight = 1.0
    minimize_time = False
    w_time = float(getattr(mission, "w_time", 0.0) or 0.0)

    try:
        objs = list(getattr(mission, "objectives", []) or [])
        if objs:
            minimize_dv = any(getattr(o, "kind", "") == "delta_v" for o in objs)
            for o in objs:
                if getattr(o, "kind", "") == "delta_v":
                    dv_weight = float(getattr(o, "weight", 1.0))
                    break
            for o in objs:
                if getattr(o, "kind", "") == "time":
                    minimize_time = True
                    w_time = float(getattr(o, "weight", w_time or 1.0))
                    break
    except Exception:
        pass

    if len(phases) == 1:
        ph = phases[0]
        if ph.initial_state is None or ph.final_state is None:
            raise MissionBuildError("Single-phase mission requires initial_state and final_state")
        tf_bounds = ph.tof_bounds_s or (600.0, 7200.0)

        mode = (ph.mode or "").lower()
        if ph.events:
            dv_front = ph.has_impulse("front")
            dv_back = ph.has_impulse("back")
        else:
            dv_front = mode in ("rendezvous", "transfer")
            dv_back = mode in ("rendezvous", "transfer")

        return TwoImpulseFreeTimeSpec(
            x0=ph.initial_state,
            xf=ph.final_state,
            tf_bounds_s=tf_bounds,
            mu_m3ps2=float(ph.dynamics.mu_m3ps2),
            nsegs=int(mission.mesh_nsegs_transfer),
            w_time=float(w_time),
            lambert_grid_size=int(mission.lambert_grid_size),
            nrevs_to_try=tuple(int(x) for x in mission.nrevs_to_try),
            minimize_dv=bool(minimize_dv),
            dv_weight=float(dv_weight),
            minimize_time=bool(minimize_time),
            dv_front=bool(dv_front),
            dv_back=bool(dv_back),
        )

    if len(phases) == 2:
        p0, p1 = phases
        if p0.mode.lower() != "coast":
            raise MissionBuildError("Two-phase missions must start with a 'coast' phase")
        if p0.initial_state is None:
            raise MissionBuildError("Precoast phase requires initial_state")
        if p1.final_state is None:
            raise MissionBuildError("Rendezvous/transfer phase requires final_state")

        t1_bounds = p0.tof_bounds_s or (0.0, 1800.0)
        tf_bounds = p1.tof_bounds_s or (600.0, 7200.0)

        dv_front = p0.has_impulse("front") if p0.events else False

        link_kind = (p1.link.kind if p1.link is not None else "continuous")
        dv_link = (link_kind.lower() == "impulsive")

        mode1 = (p1.mode or "").lower()
        if p1.events:
            dv_back = p1.has_impulse("back")
        else:
            dv_back = mode1 in ("rendezvous", "transfer")

        return TwoImpulsePreCoastSpec(
            x0=p0.initial_state,
            xf=p1.final_state,
            t1_bounds_s=t1_bounds,
            tf_bounds_s=tf_bounds,
            mu_m3ps2=float(p1.dynamics.mu_m3ps2),
            nsegs_precoast=int(mission.mesh_nsegs_precoast),
            nsegs_transfer=int(mission.mesh_nsegs_transfer),
            w_time=float(w_time),
            precoast_grid_size=int(mission.precoast_grid_size),
            limit_precoast_to_one_period=bool(mission.limit_precoast_to_one_period),
            lambert_grid_size=int(mission.lambert_grid_size),
            nrevs_to_try=tuple(int(x) for x in mission.nrevs_to_try),
            minimize_dv=bool(minimize_dv),
            dv_weight=float(dv_weight),
            minimize_time=bool(minimize_time),
            dv_front=bool(dv_front),
            dv_link=bool(dv_link),
            dv_back=bool(dv_back),
            link_kind=str(link_kind),
        )

    raise MissionBuildError(
        "Unsupported mission structure for v0.x solver mapping. "
        "Use a single transfer/rendezvous phase or a (coast, transfer/rendezvous) pair."
    )


def _scale_mesh(spec: TwoImpulseFreeTimeSpec | TwoImpulsePreCoastSpec, scale: float):
    if scale <= 0:
        return spec
    if isinstance(spec, TwoImpulseFreeTimeSpec):
        return TwoImpulseFreeTimeSpec(**{**spec.__dict__, "nsegs": max(10, int(spec.nsegs * scale))})
    return TwoImpulsePreCoastSpec(
        **{
            **spec.__dict__,
            "nsegs_transfer": max(10, int(spec.nsegs_transfer * scale)),
            "nsegs_precoast": max(10, int(spec.nsegs_precoast * scale)),
        }
    )


def _apply_simple_retry(
    spec: TwoImpulseFreeTimeSpec | TwoImpulsePreCoastSpec,
    attempt: int,
    message: str,
) -> TwoImpulseFreeTimeSpec | TwoImpulsePreCoastSpec:
    """Conservative retry strategy.

    Order:
      1) If a mesh/transcription mismatch is detected: reduce mesh density.
      2) If convergence issues: tweak tf guess (free-time only).
      3) Otherwise: increase Lambert grid density (seed search).
    """

    msg = (message or "").lower()

    if "mesh" in msg and "inconsistent" in msg:
        if isinstance(spec, TwoImpulseFreeTimeSpec):
            return TwoImpulseFreeTimeSpec(**{**spec.__dict__, "nsegs": max(20, spec.nsegs // 2)})
        return TwoImpulsePreCoastSpec(
            **{
                **spec.__dict__,
                "nsegs_transfer": max(20, spec.nsegs_transfer // 2),
                "nsegs_precoast": max(10, spec.nsegs_precoast // 2),
            }
        )

    if "did not converge" in msg or "converge" in msg:
        if isinstance(spec, TwoImpulseFreeTimeSpec):
            tfmin, tfmax = spec.tf_bounds_s
            mid = 0.5 * (float(tfmin) + float(tfmax))
            guess = mid if (attempt % 2 == 1) else (0.75 * float(tfmax) + 0.25 * float(tfmin))
            return TwoImpulseFreeTimeSpec(**{**spec.__dict__, "tf_guess_s": guess})

    if isinstance(spec, TwoImpulseFreeTimeSpec):
        return TwoImpulseFreeTimeSpec(**{**spec.__dict__, "lambert_grid_size": int(spec.lambert_grid_size) + 20})
    return TwoImpulsePreCoastSpec(**{**spec.__dict__, "lambert_grid_size": int(spec.lambert_grid_size) + 20})


def _is_composable_mission(mission: "Mission") -> bool:
    """Detect whether a mission should use the composable compiler backend.

    Heuristic (v0.1):
      - any Phase.variables is non-empty, OR
      - any Phase.constraints contains a composable State/Position constraint.
    """
    for ph in getattr(mission, "phases", []) or []:
        if getattr(ph, "variables", None):
            if len(list(getattr(ph, "variables") or [])) > 0:
                return True
        for c in getattr(ph, "constraints", []) or []:
            if isinstance(c, (StateConstraint, PositionConstraint)):
                return True
    return False
