"""Generate reproducible, physically reasonable Earth-orbit transfer cases."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from octavian import (
    Dynamics,
    Mission,
    Phase,
    Spacecraft,
    Thruster,
    constraints,
    links,
    objectives,
    state,
    two_burn_rendezvous,
    variables,
)
from octavian.astro import classical_to_cartesian
from octavian.solvers import SolverOptions

EARTH_MU_M3PS2 = 3.986004418e14
EARTH_RADIUS_M = 6_378_136.3
MIN_PERIGEE_ALTITUDE_M = 300_000.0
DEFAULT_CAMPAIGN_SEED = 20260714

Backend = Literal["quick", "composable"]
LinkKind = Literal["direct", "continuous", "impulsive"]


@dataclass(frozen=True, slots=True)
class OrbitElements:
    """Classical elements used to describe one generated endpoint orbit."""

    a_m: float
    e: float
    inc_deg: float
    raan_deg: float
    argp_deg: float
    true_anomaly_deg: float

    @property
    def perigee_radius_m(self) -> float:
        """Return the orbit's perigee radius in meters."""
        return float(self.a_m * (1.0 - self.e))


@dataclass(frozen=True, slots=True)
class TransferScenario:
    """A reproducible transfer problem plus selected public API knobs."""

    case_index: int
    case_seed: int
    initial_orbit: OrbitElements
    final_orbit: OrbitElements
    backend: Backend
    link_kind: LinkKind
    tof_bounds_s: tuple[float, float]
    precoast_bounds_s: tuple[float, float] | None
    tof_is_relative: bool
    nsegs: int
    lambert_grid_size: int
    nrevs_to_try: tuple[int, ...]
    time_weight: float

    @property
    def name(self) -> str:
        """Return the stable case name used in reports and pytest IDs."""
        return f"orbit-transfer-{self.case_index:03d}-seed-{self.case_seed}"

    def boundary_states(self):
        """Return dimensional Cartesian initial and final boundary states."""
        r0_m, v0_mps = classical_to_cartesian(
            **asdict(self.initial_orbit),
            mu_m3ps2=EARTH_MU_M3PS2,
        )
        rf_m, vf_mps = classical_to_cartesian(
            **asdict(self.final_orbit),
            mu_m3ps2=EARTH_MU_M3PS2,
        )
        return state(r0_m, v0_mps), state(rf_m, vf_mps)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable record for failure reproduction."""
        record = asdict(self)
        record["name"] = self.name
        return record


def generate_transfer_scenarios(
    count: int,
    *,
    seed: int = DEFAULT_CAMPAIGN_SEED,
) -> list[TransferScenario]:
    """Generate deterministic bound-orbit transfers with broad knob coverage.

    Each case uses its own stored seed so a failure can be reproduced without
    regenerating all preceding cases. Endpoint orbits remain elliptic, stay at
    least 300 km above the Earth at perigee, and limit plane and shape changes
    to ranges appropriate for an impulsive transfer robustness campaign.
    """
    if count < 1:
        raise ValueError("count must be at least one")

    master = np.random.default_rng(int(seed))
    case_seeds = master.integers(0, np.iinfo(np.uint32).max, size=int(count), dtype=np.uint32)
    return [
        _generate_transfer_scenario(index, int(case_seed))
        for index, case_seed in enumerate(case_seeds)
    ]


def _generate_transfer_scenario(case_index: int, case_seed: int) -> TransferScenario:
    rng = np.random.default_rng(case_seed)
    minimum_perigee_m = EARTH_RADIUS_M + MIN_PERIGEE_ALTITUDE_M

    initial_e = float(rng.uniform(0.0, 0.22))
    initial_perigee_m = float(rng.uniform(minimum_perigee_m, 13_000e3))
    initial_a_m = initial_perigee_m / (1.0 - initial_e)

    final_e = float(np.clip(initial_e + rng.uniform(-0.10, 0.10), 0.0, 0.30))
    final_perigee_m = float(
        np.clip(initial_perigee_m * rng.uniform(0.88, 1.45), minimum_perigee_m, 18_000e3)
    )
    final_a_m = final_perigee_m / (1.0 - final_e)

    initial_inc_deg = float(rng.uniform(0.0, 55.0))
    final_inc_deg = float(np.clip(initial_inc_deg + rng.uniform(-12.0, 12.0), 0.0, 70.0))
    initial_raan_deg = float(rng.uniform(0.0, 360.0))
    final_raan_deg = float((initial_raan_deg + rng.uniform(-20.0, 20.0)) % 360.0)
    initial_argp_deg = float(rng.uniform(0.0, 360.0))
    final_argp_deg = float((initial_argp_deg + rng.uniform(-45.0, 45.0)) % 360.0)
    initial_anomaly_deg = float(rng.uniform(0.0, 360.0))
    final_anomaly_deg = float((initial_anomaly_deg + rng.uniform(50.0, 310.0)) % 360.0)

    initial_orbit = OrbitElements(
        a_m=initial_a_m,
        e=initial_e,
        inc_deg=initial_inc_deg,
        raan_deg=initial_raan_deg,
        argp_deg=initial_argp_deg,
        true_anomaly_deg=initial_anomaly_deg,
    )
    final_orbit = OrbitElements(
        a_m=final_a_m,
        e=final_e,
        inc_deg=final_inc_deg,
        raan_deg=final_raan_deg,
        argp_deg=final_argp_deg,
        true_anomaly_deg=final_anomaly_deg,
    )

    reference_period_s = max(
        _orbital_period_s(initial_a_m),
        _orbital_period_s(final_a_m),
    )
    uses_precoast = case_index % 3 != 0
    precoast_bounds_s = (
        (1.0, max(120.0, 0.12 * reference_period_s)) if uses_precoast else None
    )
    tof_lower_s = max(600.0, 0.22 * reference_period_s)
    tof_upper_s = max(tof_lower_s + 1_800.0, 1.35 * reference_period_s)

    backend: Backend = "quick" if case_index % 2 == 0 else "composable"
    if not uses_precoast:
        link_kind: LinkKind = "direct"
    elif case_index % 4 in (0, 1):
        link_kind = "impulsive"
    else:
        link_kind = "continuous"

    return TransferScenario(
        case_index=int(case_index),
        case_seed=int(case_seed),
        initial_orbit=initial_orbit,
        final_orbit=final_orbit,
        backend=backend,
        link_kind=link_kind,
        tof_bounds_s=(float(tof_lower_s), float(tof_upper_s)),
        precoast_bounds_s=precoast_bounds_s,
        tof_is_relative=bool(case_index % 2),
        nsegs=(40, 60, 80)[case_index % 3],
        lambert_grid_size=(40, 60, 80, 100)[case_index % 4],
        nrevs_to_try=(0,) if case_index % 3 == 0 else (0, 1),
        time_weight=0.0 if case_index % 4 else 0.05,
    )


def build_transfer_mission(
    scenario: TransferScenario,
    *,
    solver_options: SolverOptions | None = None,
) -> Mission:
    """Build a quick or composable mission for one generated scenario."""
    initial_state, final_state = scenario.boundary_states()
    options = solver_options or SolverOptions(print_level=0, max_ls_iters=5, asset_threads=(1, 1))

    if scenario.backend == "quick":
        return two_burn_rendezvous(
            initial_state,
            final_state,
            mu_m3ps2=EARTH_MU_M3PS2,
            tf_bounds_s=scenario.tof_bounds_s,
            nsegs=scenario.nsegs,
            lambert_grid_size=scenario.lambert_grid_size,
            nrevs_to_try=scenario.nrevs_to_try,
            w_time=scenario.time_weight,
            precoast=scenario.precoast_bounds_s is not None,
            t1_bounds_s=scenario.precoast_bounds_s or (0.0, 1.0),
            precoast_grid_size=12,
            name=scenario.name,
            solver_options=options,
        )

    spacecraft = Spacecraft(
        name="RobustnessSat",
        dry_mass_kg=200.0,
        thrusters=[Thruster(name="main")],
    )
    dynamics = Dynamics(mu_m3ps2=EARTH_MU_M3PS2)
    objective_list = [objectives.minimize_total_delta_v()]
    if scenario.time_weight:
        objective_list.append(objectives.minimize_total_time(weight=scenario.time_weight))

    if scenario.precoast_bounds_s is None:
        transfer = Phase(
            name="transfer",
            mode="coast",
            spacecraft=spacecraft,
            dynamics=dynamics,
            tof_bounds_s=scenario.tof_bounds_s,
            constraints=[
                constraints.state(initial_state, where="Front"),
                constraints.state(final_state, where="Back"),
            ],
            variables=[
                variables.ImpulsiveDeltaV(where="Front"),
                variables.ImpulsiveDeltaV(where="Back"),
            ],
        )
        phases = [transfer]
    else:
        precoast_variables = (
            [variables.ImpulsiveDeltaV(where="Front")]
            if scenario.link_kind == "continuous"
            else []
        )
        precoast = Phase(
            name="precoast",
            mode="coast",
            spacecraft=spacecraft,
            dynamics=dynamics,
            tof_bounds_s=scenario.precoast_bounds_s,
            tof_is_relative=scenario.tof_is_relative,
            constraints=[constraints.state(initial_state, where="Front")],
            variables=precoast_variables,
        )
        transfer_variables = [variables.ImpulsiveDeltaV(where="Back")]
        if scenario.link_kind == "impulsive":
            transfer_variables.insert(0, variables.ImpulsiveDeltaV(where="Front"))
        transfer = Phase(
            name="transfer",
            mode="coast",
            spacecraft=spacecraft,
            dynamics=dynamics,
            previous=precoast,
            link=(
                links.impulsive()
                if scenario.link_kind == "impulsive"
                else links.continuous()
            ),
            tof_bounds_s=scenario.tof_bounds_s,
            tof_is_relative=scenario.tof_is_relative,
            constraints=[constraints.state(final_state, where="Back")],
            variables=transfer_variables,
        )
        phases = [precoast, transfer]

    return Mission(
        name=scenario.name,
        phases=phases,
        objectives=objective_list,
        solver_options=options,
        mesh_nsegs_precoast=max(20, scenario.nsegs // 2),
        mesh_nsegs_transfer=scenario.nsegs,
        lambert_grid_size=scenario.lambert_grid_size,
        nrevs_to_try=scenario.nrevs_to_try,
        precoast_grid_size=12,
    )


def solution_checks(scenario: TransferScenario, solution) -> dict[str, float | int | bool]:
    """Validate solver-independent physical and boundary-result invariants."""
    if not solution.ok or solution.result is None or not solution.result.converged:
        raise AssertionError("mission did not converge")

    trajectory = np.asarray(solution.traj, dtype=float)
    if trajectory.ndim != 2 or trajectory.shape[1] < 7 or len(trajectory) < 2:
        raise AssertionError("solution trajectory has an invalid shape")
    if not np.all(np.isfinite(trajectory)):
        raise AssertionError("solution trajectory contains non-finite values")
    if np.any(np.diff(trajectory[:, 6]) < -1.0e-8):
        raise AssertionError("solution time is not monotonic")

    _, final_state = scenario.boundary_states()
    terminal_position_error_m = float(
        np.linalg.norm(trajectory[-1, 0:3] - np.asarray(final_state.r_m, dtype=float))
    )
    if terminal_position_error_m > 100.0:
        raise AssertionError(
            f"terminal position error {terminal_position_error_m:.3f} m exceeds 100 m"
        )

    total_dv_mps = float(solution.result.total_dv_mps())
    if not np.isfinite(total_dv_mps) or total_dv_mps < 0.0:
        raise AssertionError("total delta-v is not finite and non-negative")

    return {
        "converged": True,
        "trajectory_points": int(len(trajectory)),
        "terminal_position_error_m": terminal_position_error_m,
        "total_dv_mps": total_dv_mps,
        "final_time_s": float(trajectory[-1, 6]),
        "attempt_count": int(len(solution.attempts)),
    }


def _orbital_period_s(a_m: float) -> float:
    return float(2.0 * np.pi * np.sqrt(float(a_m) ** 3 / EARTH_MU_M3PS2))
