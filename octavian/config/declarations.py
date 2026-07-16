"""Parsers for phase declarations and mission execution options."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .. import constraints, events, guesses, links, objectives, variables
from ..models import RetryPolicy, RunPlan, SolveConfig, Stage
from ..solvers import SolverOptions
from ..specs import BoundaryState
from .errors import MissionConfigError
from .schema import (
    boolean,
    mapping,
    normalized_type,
    pair,
    reject_unknown,
    required,
    sequence,
    state_reference,
)


def build_constraint(
    value: Any, path: str, states: Mapping[str, BoundaryState]
) -> Any:
    """Build one supported path or boundary constraint."""
    config = mapping(value, path)
    constraint_type = normalized_type(required(config, "type", path))
    where = str(config.get("where", "Path"))

    if constraint_type == "state":
        reject_unknown(config, {"type", "state", "where", "groups"}, path)
        groups = tuple(
            str(item) for item in sequence(config.get("groups", ["R", "V"]), f"{path}.groups")
        )
        return constraints.state(
            state_reference(required(config, "state", path), states, f"{path}.state"),
            where=where,
            groups=groups,
        )
    if constraint_type == "position":
        reject_unknown(config, {"type", "r_m", "where"}, path)
        return constraints.position(required(config, "r_m", path), where=where)
    if constraint_type == "min_radius":
        reject_unknown(config, {"type", "r_min_m", "where"}, path)
        return constraints.min_radius(float(required(config, "r_min_m", path)), where=where)
    if constraint_type == "semi_major_axis":
        reject_unknown(config, {"type", "a_m", "where", "tol_m"}, path)
        tolerance = config.get("tol_m")
        return constraints.semi_major_axis(
            float(required(config, "a_m", path)),
            where=where,
            tol_m=None if tolerance is None else float(tolerance),
        )
    if constraint_type == "eccentricity":
        reject_unknown(config, {"type", "e", "where", "tol"}, path)
        tolerance = config.get("tol")
        return constraints.eccentricity(
            float(required(config, "e", path)),
            where=where,
            tol=None if tolerance is None else float(tolerance),
        )
    if constraint_type == "inclination_deg":
        reject_unknown(config, {"type", "inc_deg", "where", "tol_deg"}, path)
        tolerance = config.get("tol_deg")
        return constraints.inclination_deg(
            float(required(config, "inc_deg", path)),
            where=where,
            tol_deg=None if tolerance is None else float(tolerance),
        )
    if constraint_type == "keep_out_sphere":
        reject_unknown(config, {"type", "radius_m", "center_m", "where"}, path)
        return constraints.keep_out_sphere(
            float(required(config, "radius_m", path)),
            center_m=config.get("center_m", [0.0, 0.0, 0.0]),
            where=where,
        )
    if constraint_type == "approach_cone":
        reject_unknown(
            config,
            {"type", "axis", "half_angle_deg", "vertex_m", "where"},
            path,
        )
        return constraints.approach_cone(
            required(config, "axis", path),
            float(required(config, "half_angle_deg", path)),
            vertex_m=config.get("vertex_m", [0.0, 0.0, 0.0]),
            where=where,
        )
    if constraint_type == "lighting_angle":
        reject_unknown(
            config,
            {
                "type",
                "sun_direction",
                "min_angle_deg",
                "max_angle_deg",
                "origin_m",
                "where",
            },
            path,
        )
        return constraints.lighting_angle(
            required(config, "sun_direction", path),
            min_angle_deg=float(config.get("min_angle_deg", 0.0)),
            max_angle_deg=float(config.get("max_angle_deg", 180.0)),
            origin_m=config.get("origin_m", [0.0, 0.0, 0.0]),
            where=where,
        )
    raise MissionConfigError(f"{path}.type has unsupported value {constraint_type!r}.")


def build_variable(value: Any, path: str) -> Any:
    """Build one decision-variable declaration."""
    config = mapping(value, path)
    variable_type = normalized_type(required(config, "type", path))
    if variable_type != "impulsive_delta_v":
        raise MissionConfigError(f"{path}.type has unsupported value {variable_type!r}.")
    reject_unknown(config, {"type", "where"}, path)
    return variables.impulsive_delta_v(at=str(config.get("where", "Front")))


def build_event(value: Any, path: str) -> Any:
    """Build one boundary-event declaration."""
    config = mapping(value, path)
    event_type = normalized_type(required(config, "type", path))
    if event_type != "impulse":
        raise MissionConfigError(f"{path}.type has unsupported value {event_type!r}.")
    reject_unknown(config, {"type", "where", "dv_max_mps"}, path)
    maximum_delta_v = config.get("dv_max_mps")
    return events.impulse(
        str(config.get("where", "Front")),
        None if maximum_delta_v is None else float(maximum_delta_v),
    )


def build_link(value: Any, path: str) -> Any:
    """Build a continuous or impulsive phase link."""
    config: Mapping[str, Any] = (
        {"type": value} if isinstance(value, str) else mapping(value, path)
    )
    link_type = normalized_type(required(config, "type", path))
    if link_type == "continuous":
        reject_unknown(config, {"type", "name"}, path)
        return links.continuous(name=str(config.get("name", "continuous")))
    if link_type == "impulsive":
        reject_unknown(config, {"type", "name", "dv_max_mps"}, path)
        maximum_delta_v = config.get("dv_max_mps")
        return links.impulsive(
            name=str(config.get("name", "impulsive")),
            dv_max_mps=None if maximum_delta_v is None else float(maximum_delta_v),
        )
    raise MissionConfigError(f"{path}.type has unsupported value {link_type!r}.")


def build_initial_guess(value: Any, path: str) -> Any:
    """Build a supported phase initial-guess declaration."""
    config = mapping(value, path)
    guess_type = normalized_type(required(config, "type", path))
    if guess_type != "low_thrust_spiral":
        raise MissionConfigError(f"{path}.type has unsupported value {guess_type!r}.")
    reject_unknown(
        config,
        {"type", "throttle", "direction", "steps_per_orbit", "time_scale"},
        path,
    )
    return guesses.low_thrust_spiral(
        throttle=float(config.get("throttle", 0.8)),
        direction=str(config.get("direction", "auto")),
        steps_per_orbit=int(config.get("steps_per_orbit", 120)),
        time_scale=float(config.get("time_scale", 1.0)),
    )


def build_objectives(value: Any) -> list[Any]:
    """Build the mission objective list."""
    result = []
    factories = {
        "delta_v": objectives.minimize_total_delta_v,
        "time": objectives.minimize_total_time,
        "propellant": objectives.minimize_propellant,
    }
    for index, raw_objective in enumerate(sequence(value, "config.mission.objectives")):
        path = f"config.mission.objectives[{index}]"
        config = mapping(raw_objective, path)
        reject_unknown(config, {"type", "weight"}, path)
        objective_type = normalized_type(required(config, "type", path))
        try:
            factory = factories[objective_type]
        except KeyError as exc:
            raise MissionConfigError(
                f"{path}.type has unsupported value {objective_type!r}."
            ) from exc
        result.append(factory(weight=float(config.get("weight", 1.0))))
    return result


def build_solver_options(value: Any) -> SolverOptions:
    """Build ASSET solver options."""
    path = "config.mission.solver_options"
    config = mapping(value, path)
    allowed = {
        "print_level",
        "max_ls_iters",
        "qp_ordering_mode",
        "enable_auto_scaling",
        "enable_adaptive_mesh",
        "asset_threads",
    }
    reject_unknown(config, allowed, path)
    thread_pair = None
    if config.get("asset_threads") is not None:
        first, second = pair(config["asset_threads"], f"{path}.asset_threads")
        thread_pair = (int(first), int(second))
    return SolverOptions(
        print_level=int(config.get("print_level", 0)),
        max_ls_iters=int(config.get("max_ls_iters", 2)),
        qp_ordering_mode=str(config.get("qp_ordering_mode", "MINDEG")),
        enable_auto_scaling=boolean(
            config.get("enable_auto_scaling", True), f"{path}.enable_auto_scaling"
        ),
        enable_adaptive_mesh=boolean(
            config.get("enable_adaptive_mesh", True), f"{path}.enable_adaptive_mesh"
        ),
        asset_threads=thread_pair,
    )


def build_solve_config(value: Any) -> SolveConfig:
    """Build runner behavior settings."""
    path = "config.mission.solve_config"
    config = mapping(value, path)
    reject_unknown(config, {"max_attempts", "raise_on_fail", "verbose"}, path)
    return SolveConfig(
        max_attempts=int(config.get("max_attempts", 3)),
        raise_on_fail=boolean(config.get("raise_on_fail", True), f"{path}.raise_on_fail"),
        verbose=boolean(config.get("verbose", True), f"{path}.verbose"),
    )


def build_retry_policy(value: Any) -> RetryPolicy:
    """Build solve retry behavior."""
    path = "config.mission.retry"
    config = mapping(value, path)
    reject_unknown(config, {"enabled", "max_retries"}, path)
    return RetryPolicy(
        enabled=boolean(config.get("enabled", True), f"{path}.enabled"),
        max_retries=int(config.get("max_retries", 2)),
    )


def build_run_plan(value: Any) -> RunPlan:
    """Build the optional continuation-stage plan."""
    path = "config.mission.plan"
    config = mapping(value, path)
    reject_unknown(config, {"stages"}, path)
    stages = []
    for index, raw_stage in enumerate(sequence(config.get("stages", []), f"{path}.stages")):
        stage_path = f"{path}.stages[{index}]"
        stage = mapping(raw_stage, stage_path)
        reject_unknown(stage, {"name", "nsegs_scale", "tighten_bounds"}, stage_path)
        scale = stage.get("nsegs_scale")
        stages.append(
            Stage(
                name=str(required(stage, "name", stage_path)),
                nsegs_scale=None if scale is None else float(scale),
                tighten_bounds=boolean(
                    stage.get("tighten_bounds", False), f"{stage_path}.tighten_bounds"
                ),
            )
        )
    return RunPlan(stages=tuple(stages))
