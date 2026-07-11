"""Mission composition primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .models import RetryPolicy, RunPlan, SolveConfig
from .objectives import Objective, minimize_total_delta_v
from .phase import Phase
from .runner import MissionRunner
from .solution import Solution
from .solvers import SolverOptions
from .spacecraft import Spacecraft


@dataclass(slots=True)
class Mission:
    """Top-level user-facing mission container."""

    phases: list[Phase] = field(default_factory=list)
    spacecraft: dict[str, Spacecraft] = field(default_factory=dict)
    name: str = "Mission"
    initial_epoch: str | datetime | float | None = None

    objectives: list[Objective] = field(default_factory=lambda: [minimize_total_delta_v()])

    plan: RunPlan = field(default_factory=RunPlan.default)
    retry: RetryPolicy = field(default_factory=RetryPolicy.default)
    solve_config: SolveConfig = field(default_factory=SolveConfig)
    solver_options: SolverOptions = field(default_factory=SolverOptions)

    mesh_nsegs_transfer: int = 60
    mesh_nsegs_precoast: int = 30
    lambert_grid_size: int = 60
    nrevs_to_try: tuple[int, ...] = (0, 1)
    precoast_grid_size: int = 10
    limit_precoast_to_one_period: bool = True
    w_time: float = 0.0

    def __post_init__(self) -> None:
        self.phases = list(self.phases)

    def add_phase(self, phase: Phase) -> None:
        """Append a phase to the mission.

        Args:
            phase: Phase to append.
        """
        self.phases.append(phase)

    def validate(self) -> None:
        """Validate mission structure before solving.

        Raises:
            ValueError: If the mission has no phases or a phase is invalid.
        """
        if not self.phases:
            raise ValueError("Mission has no phases.")
        for phase in self.phases:
            phase.validate()

    def solve(
        self,
        *,
        plan: RunPlan | None = None,
        retry: RetryPolicy | None = None,
        solve_config: SolveConfig | None = None,
        solver_options: SolverOptions | None = None,
    ) -> Solution:
        """Solve the mission using the configured runner.

        Args:
            plan: Optional continuation or staging plan override.
            retry: Optional retry-policy override.
            solve_config: Optional solve-configuration override.
            solver_options: Optional backend solver-options override.

        Returns:
            The mission solution wrapper.
        """
        mission_runner = MissionRunner(
            solve_options=solver_options or self.solver_options,
            solve_config=solve_config or self.solve_config,
            plan=plan or self.plan,
            retry=retry or self.retry,
        )
        return mission_runner.solve(self)
