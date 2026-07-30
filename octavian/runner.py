"""Mission runner and backend mapping logic."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from .models import RetryPolicy, RunPlan, SolveConfig
from .solution import AttemptLog, Solution
from .solvers import SolverOptions
from .solvers.composable import solve_composable_mission
from .solvers.preconfigured import solve as solve_preconfigured
from .specs import TwoImpulseFreeTimeSpec, TwoImpulsePreCoastSpec

if TYPE_CHECKING:
    from .mission import Mission


class MissionBuildError(ValueError):
    """Raised when a mission cannot be mapped to a supported solver."""


@dataclass(slots=True)
class MissionRunner:
    """Coordinate mission validation, backend selection, and retry behavior."""

    solve_options: SolverOptions
    solve_config: SolveConfig = field(default_factory=SolveConfig)
    plan: RunPlan = field(default_factory=RunPlan.default)
    retry: RetryPolicy = field(default_factory=RetryPolicy.default)

    def solve(self, mission: Mission) -> Solution:
        """Solve a mission through validation, staging, backend dispatch, and retry.

        Parameters
        ----------
        mission
            Mission to validate, compile, and solve.

        Returns
        -------
        Solution
            Solver result plus an attempt log describing each stage and retry.
        """
        self._validate_mission(mission)
        backend = _select_backend(mission)
        attempt_logs: list[AttemptLog] = []
        last_error: str | None = None

        for stage_index, stage in enumerate(_runner_stages(self.plan)):
            stage_label = _stage_label(stage)
            stage_problem = _build_stage_problem(mission, backend, stage)

            for attempt_index in range(1, self._max_attempts() + 1):
                result = None
                try:
                    result = _solve_stage_problem(stage_problem, self.solve_options)
                except Exception as exc:  # noqa: BLE001
                    last_error = str(exc)
                    attempt_logs.append(
                        AttemptLog(
                            stage=stage_label,
                            attempt=attempt_index,
                            status="fail",
                            message=last_error,
                        )
                    )
                    if not self._should_retry(attempt_index):
                        break
                    stage_problem = _retry_stage_problem(
                        stage_problem,
                        attempt_index,
                        last_error,
                    )
                    continue

                attempt_logs.append(AttemptLog(stage=stage_label, attempt=attempt_index, status="ok"))
                solution = Solution(ok=bool(result.converged), result=result, attempts=attempt_logs)
                solution.info.update(
                    {
                        "stage": stage_label,
                        "stage_index": stage_index,
                        "mission_name": mission.name,
                        "initial_epoch": mission.initial_epoch,
                    }
                )
                if result.converged or not self.solve_config.raise_on_fail:
                    return solution
                last_error = "Solver did not converge"
                attempt_logs.append(
                    AttemptLog(
                        stage=stage_label,
                        attempt=attempt_index,
                        status="fail",
                        message=last_error,
                    )
                )
                if not self._should_retry(attempt_index):
                    break
                stage_problem = _retry_stage_problem(stage_problem, attempt_index, last_error)

        solution = Solution(
            ok=False,
            result=None,
            attempts=attempt_logs,
            last_error=last_error,
        )
        if self.solve_config.raise_on_fail:
            raise RuntimeError(solution.summary())
        return solution

    def _validate_mission(self, mission: Mission) -> None:
        """Validate the mission object before backend selection.

        Parameters
        ----------
        mission
            Candidate mission object.

        Raises
        ------
        TypeError
            If ``mission`` is not an Octavian ``Mission``.
        ValueError
            If mission-level validation fails.
        """
        from .mission import Mission

        if not isinstance(mission, Mission):
            raise TypeError("MissionRunner.solve expects a Mission.")
        mission.validate()

    def _max_attempts(self) -> int:
        """Return the configured maximum attempt count.

        Returns
        -------
        int
            At least one attempt.
        """
        configured_attempts = max(1, int(self.solve_config.max_attempts))
        if not self.retry.enabled:
            return configured_attempts
        retry_limited_attempts = max(1, 1 + int(self.retry.max_retries))
        return min(configured_attempts, retry_limited_attempts)

    def _should_retry(self, attempt_index: int) -> bool:
        """Return whether another attempt should be made.

        Parameters
        ----------
        attempt_index
            One-based attempt index that just finished.

        Returns
        -------
        bool
            Whether retry policy and solve config allow another attempt.
        """
        return bool(self.retry.enabled) and attempt_index < self._max_attempts()


@dataclass(frozen=True, slots=True)
class _StageProblem:
    """Backend input for one runner stage."""

    backend: str
    mission: Mission
    rendezvous_spec: TwoImpulseFreeTimeSpec | TwoImpulsePreCoastSpec | None = None


def _select_backend(mission: Mission) -> str:
    """Choose the solver backend for a mission.

    Parameters
    ----------
    mission
        Validated mission.

    Returns
    -------
    str
        ``"composable"`` or ``"rendezvous"``.
    """
    return "composable" if _is_composable_mission(mission) else "rendezvous"


def _runner_stages(plan: RunPlan):
    """Yield configured stages or a single default stage.

    Parameters
    ----------
    plan
        Runner plan.

    Returns
    -------
    list
        Stage objects, or ``[None]`` when no stages are configured.
    """
    return list(plan.stages) or [None]


def _stage_label(stage) -> str:
    """Return a human-readable stage label.

    Parameters
    ----------
    stage
        Stage object or ``None``.

    Returns
    -------
    str
        Stage name for attempt logs.
    """
    return getattr(stage, "name", "default") if stage is not None else "default"


def _build_stage_problem(mission: Mission, backend: str, stage) -> _StageProblem:
    """Build the solver input for one stage.

    Parameters
    ----------
    mission
        Validated mission.
    backend
        Selected backend name.
    stage
        Stage configuration or ``None``.

    Returns
    -------
    _StageProblem
        Backend input for a single stage.
    """
    if backend == "composable":
        return _StageProblem(backend=backend, mission=mission)

    rendezvous_spec = _mission_to_rendezvous_spec(mission)
    if stage is not None and getattr(stage, "nsegs_scale", None):
        rendezvous_spec = _scale_mesh(rendezvous_spec, float(stage.nsegs_scale))
    return _StageProblem(backend=backend, mission=mission, rendezvous_spec=rendezvous_spec)


def _solve_stage_problem(problem: _StageProblem, options: SolverOptions):
    """Run the selected backend for one stage.

    Parameters
    ----------
    problem
        Stage-specific backend input.
    options
        Solver options.

    Returns
    -------
    RendezvousResult
        Backend result object.
    """
    if problem.backend == "composable":
        return solve_composable_mission(problem.mission, options=options)
    if problem.rendezvous_spec is None:
        raise MissionBuildError("Preconfigured backend requires a transfer specification.")
    return solve_preconfigured(problem.rendezvous_spec, options=options)


def _retry_stage_problem(problem: _StageProblem, attempt_index: int, message: str) -> _StageProblem:
    """Return adjusted stage input for the next retry.

    Parameters
    ----------
    problem
        Previous stage problem.
    attempt_index
        One-based failed attempt index.
    message
        Failure message from the previous attempt.

    Returns
    -------
    _StageProblem
        Updated problem for the next attempt.
    """
    if problem.rendezvous_spec is None:
        return problem
    return replace(
        problem,
        rendezvous_spec=_apply_simple_retry(problem.rendezvous_spec, attempt_index, message),
    )


def _mission_to_rendezvous_spec(
    mission: Mission,
) -> TwoImpulseFreeTimeSpec | TwoImpulsePreCoastSpec:
    """Map a mission into one of the built-in preconfigured transfer specs.

    This is the adapter between the mission-script API and the older fixed-shape
    two-impulse solvers. Only simple one-phase and two-phase mission structures
    can be represented here. Anything with explicit composable variables,
    finite burns, perturbations, or richer constraints should be routed to the
    composable backend instead.
    """
    phases = list(mission.phases)

    minimize_delta_v = True
    delta_v_weight = 1.0
    minimize_time = False
    time_weight = float(getattr(mission, "w_time", 0.0) or 0.0)

    mission_objectives = list(getattr(mission, "objectives", []) or [])
    if mission_objectives:
        minimize_delta_v = any(
            getattr(objective, "kind", "") == "delta_v"
            for objective in mission_objectives
        )
        for objective in mission_objectives:
            if getattr(objective, "kind", "") == "delta_v":
                delta_v_weight = float(getattr(objective, "weight", 1.0))
                break
        for objective in mission_objectives:
            if getattr(objective, "kind", "") == "time":
                minimize_time = True
                time_weight = float(getattr(objective, "weight", time_weight or 1.0))
                break

    if len(phases) == 1:
        phase = phases[0]
        if phase.initial_state is None or phase.final_state is None:
            raise MissionBuildError(
                "Single-phase mission requires initial_state and final_state."
            )
        final_time_bounds_s = phase.tof_bounds_s or (600.0, 7200.0)

        normalized_mode = (phase.mode or "").lower()
        if phase.events:
            has_front_impulse = phase.has_impulse("front")
            has_back_impulse = phase.has_impulse("back")
        else:
            has_front_impulse = normalized_mode in ("rendezvous", "transfer")
            has_back_impulse = normalized_mode in ("rendezvous", "transfer")

        return TwoImpulseFreeTimeSpec(
            x0=phase.initial_state,
            xf=phase.final_state,
            tf_bounds_s=final_time_bounds_s,
            mu_m3ps2=float(phase.dynamics.mu_m3ps2),
            central_body_name=(
                phase.dynamics.central_body.name
                if phase.dynamics.central_body is not None
                else phase.dynamics.frame.origin
            ),
            frame=phase.dynamics.frame,
            scaling=phase.dynamics.scaling,
            nsegs=int(mission.mesh_nsegs_transfer),
            w_time=float(time_weight),
            lambert_grid_size=int(mission.lambert_grid_size),
            nrevs_to_try=tuple(int(revolution_count) for revolution_count in mission.nrevs_to_try),
            minimize_dv=bool(minimize_delta_v),
            dv_weight=float(delta_v_weight),
            minimize_time=bool(minimize_time),
            dv_front=bool(has_front_impulse),
            dv_back=bool(has_back_impulse),
        )

    if len(phases) == 2:
        precoast_phase, transfer_phase = phases
        if precoast_phase.mode.lower() != "coast":
            raise MissionBuildError("Two-phase missions must start with a 'coast' phase.")
        if precoast_phase.initial_state is None:
            raise MissionBuildError("Precoast phase requires initial_state.")
        if transfer_phase.final_state is None:
            raise MissionBuildError("Rendezvous/transfer phase requires final_state.")

        precoast_bounds_s = precoast_phase.tof_bounds_s or (0.0, 1800.0)
        transfer_bounds_s = transfer_phase.tof_bounds_s or (600.0, 7200.0)
        has_precoast_front_impulse = (
            precoast_phase.has_impulse("front") if precoast_phase.events else False
        )

        link_kind = (
            transfer_phase.link.kind if transfer_phase.link is not None else "continuous"
        )
        has_impulsive_link = link_kind.lower() == "impulsive"

        normalized_transfer_mode = (transfer_phase.mode or "").lower()
        has_terminal_impulse = (
            transfer_phase.has_impulse("back")
            if transfer_phase.events
            else normalized_transfer_mode in ("rendezvous", "transfer")
        )

        return TwoImpulsePreCoastSpec(
            x0=precoast_phase.initial_state,
            xf=transfer_phase.final_state,
            t1_bounds_s=precoast_bounds_s,
            tf_bounds_s=transfer_bounds_s,
            mu_m3ps2=float(transfer_phase.dynamics.mu_m3ps2),
            central_body_name=(
                transfer_phase.dynamics.central_body.name
                if transfer_phase.dynamics.central_body is not None
                else transfer_phase.dynamics.frame.origin
            ),
            frame=transfer_phase.dynamics.frame,
            scaling=transfer_phase.dynamics.scaling,
            nsegs_precoast=int(mission.mesh_nsegs_precoast),
            nsegs_transfer=int(mission.mesh_nsegs_transfer),
            w_time=float(time_weight),
            precoast_grid_size=int(mission.precoast_grid_size),
            limit_precoast_to_one_period=bool(mission.limit_precoast_to_one_period),
            lambert_grid_size=int(mission.lambert_grid_size),
            nrevs_to_try=tuple(int(revolution_count) for revolution_count in mission.nrevs_to_try),
            minimize_dv=bool(minimize_delta_v),
            dv_weight=float(delta_v_weight),
            minimize_time=bool(minimize_time),
            dv_front=bool(has_precoast_front_impulse),
            dv_link=bool(has_impulsive_link),
            dv_back=bool(has_terminal_impulse),
            link_kind=str(link_kind),
        )

    raise MissionBuildError(
        "Unsupported mission structure for v0.x solver mapping. "
        "Use a single transfer/rendezvous phase or a (coast, transfer/rendezvous) pair."
    )


def _scale_mesh(
    spec: TwoImpulseFreeTimeSpec | TwoImpulsePreCoastSpec,
    scale: float,
) -> TwoImpulseFreeTimeSpec | TwoImpulsePreCoastSpec:
    """Scale mesh counts for a rendezvous specification."""
    if scale <= 0:
        return spec
    if isinstance(spec, TwoImpulseFreeTimeSpec):
        return replace(spec, nsegs=max(10, int(spec.nsegs * scale)))
    return replace(
        spec,
        nsegs_transfer=max(10, int(spec.nsegs_transfer * scale)),
        nsegs_precoast=max(10, int(spec.nsegs_precoast * scale)),
    )


def _apply_simple_retry(
    spec: TwoImpulseFreeTimeSpec | TwoImpulsePreCoastSpec,
    attempt: int,
    message: str,
) -> TwoImpulseFreeTimeSpec | TwoImpulsePreCoastSpec:
    """Apply a conservative retry update to a preconfigured transfer spec.

    The runner retry layer works above ASSET compilation, so it can only adjust
    problem specifications. Mesh inconsistency messages reduce mesh density,
    convergence messages perturb free-time guesses, and generic failures expand
    the Lambert seed search grid. ASSET-internal mesh-time failures are handled
    lower down in `_asset.solve_with_standard_sequence`.
    """
    error_message = (message or "").lower()

    if "mesh" in error_message and "inconsistent" in error_message:
        if isinstance(spec, TwoImpulseFreeTimeSpec):
            return replace(spec, nsegs=max(20, spec.nsegs // 2))
        return replace(
            spec,
            nsegs_transfer=max(20, spec.nsegs_transfer // 2),
            nsegs_precoast=max(10, spec.nsegs_precoast // 2),
        )

    if ("did not converge" in error_message or "converge" in error_message) and isinstance(
        spec, TwoImpulseFreeTimeSpec
    ):
        tf_min_s, tf_max_s = spec.tf_bounds_s
        midpoint_time_s = 0.5 * (float(tf_min_s) + float(tf_max_s))
        tf_guess_s = (
            midpoint_time_s
            if (attempt % 2 == 1)
            else (0.75 * float(tf_max_s) + 0.25 * float(tf_min_s))
        )
        return replace(spec, tf_guess_s=tf_guess_s)

    if isinstance(spec, TwoImpulseFreeTimeSpec):
        return replace(spec, lambert_grid_size=int(spec.lambert_grid_size) + 20)
    return replace(spec, lambert_grid_size=int(spec.lambert_grid_size) + 20)


def _is_composable_mission(mission: Mission) -> bool:
    """Return whether a mission needs the general composable backend.

    The preconfigured backend is kept for simple quick-start rendezvous shapes.
    This detector upgrades to the composable compiler when the mission uses
    features that require explicit phase compilation: finite burns, J2,
    user-declared variables, or direct boundary constraint objects.
    """
    for phase in getattr(mission, "phases", []) or []:
        normalized_mode = (getattr(phase, "mode", "") or "").lower().replace("-", "_")
        if normalized_mode in (
            "burn",
            "chemical_burn",
            "finite_burn",
            "powered",
            "finite_thrust",
            "low_thrust",
        ):
            return True
        dynamics = getattr(phase, "dynamics", None)
        if dynamics is not None and dynamics.model is not None:
            return True
        if dynamics is not None and dynamics.active_perturbations().j2:
            return True
        if getattr(phase, "variables", None) and len(list(phase.variables or [])) > 0:
            return True
        for constraint in getattr(phase, "constraints", []) or []:
            if getattr(constraint, "kind", "") in ("state", "position"):
                return True
    return False
