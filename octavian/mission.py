from __future__ import annotations

"""Mission composition.

The `Mission` object is Octavian's main user-facing API. It is intentionally
"config-like": you compose phases, spacecraft, and dynamics in plain Python.

Advanced behavior (continuation plans, retries, solve options) is available via
defaults so simple scripts stay simple.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .models import RetryPolicy, RunPlan, SolveConfig
from .objectives import Objective, minimize_total_delta_v
from .phase import Phase
from .runner import MissionRunner
from .solution import Solution
from .solvers import SolverOptions
from .spacecraft import Spacecraft


@dataclass(slots=True)
class Mission:
    phases: List[Phase] = field(default_factory=list)
    spacecraft: Dict[str, Spacecraft] = field(default_factory=dict)
    name: str = "Mission"

    objectives: List[Objective] = field(default_factory=lambda: [minimize_total_delta_v()])

    # solving defaults
    plan: RunPlan = field(default_factory=RunPlan.default)
    retry: RetryPolicy = field(default_factory=RetryPolicy.default)
    solve_config: SolveConfig = field(default_factory=SolveConfig)
    solver_options: SolverOptions = field(default_factory=SolverOptions)

    # v0.x rendezvous mapping defaults
    mesh_nsegs_transfer: int = 60
    mesh_nsegs_precoast: int = 30
    lambert_grid_size: int = 60
    nrevs_to_try: Tuple[int, ...] = (0, 1)
    precoast_grid_size: int = 10
    limit_precoast_to_one_period: bool = True
    w_time: float = 0.0

    def __post_init__(self) -> None:
        self.phases = list(self.phases)

    def add_phase(self, phase: Phase) -> None:
        self.phases.append(phase)

    def validate(self) -> None:
        if not self.phases:
            raise ValueError("Mission has no phases")
        for ph in self.phases:
            ph.validate()

    def solve(
        self,
        *,
        plan: Optional[RunPlan] = None,
        retry: Optional[RetryPolicy] = None,
        solve_config: Optional[SolveConfig] = None,
        solver_options: Optional[SolverOptions] = None,
    ) -> Solution:
        runner = MissionRunner(
            solve_options=solver_options or self.solver_options,
            solve_config=solve_config or self.solve_config,
            plan=plan or self.plan,
            retry=retry or self.retry,
        )
        return runner.solve(self)
