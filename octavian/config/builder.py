"""Assemble a mission from named declarative configuration sections."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..forces import Cannonball
from ..mission import Mission
from ..phase import Phase, state
from ..spacecraft import Spacecraft, Thruster
from ..specs import BoundaryState
from .declarations import (
    build_constraint,
    build_event,
    build_initial_guess,
    build_link,
    build_objectives,
    build_retry_policy,
    build_run_plan,
    build_solve_config,
    build_solver_options,
    build_variable,
)
from .environment import build_dynamics
from .errors import MissionConfigError
from .schema import (
    boolean,
    mapping,
    optional_pair,
    optional_reference,
    optional_state,
    reference,
    reject_unknown,
    required,
    sequence,
)

SCHEMA_VERSION = 1

_ROOT_KEYS = {"schema_version", "states", "spacecraft", "dynamics", "mission"}
_MISSION_KEYS = {
    "name",
    "initial_epoch",
    "phases",
    "objectives",
    "solver_options",
    "solve_config",
    "retry",
    "plan",
    "mesh_nsegs_transfer",
    "mesh_nsegs_precoast",
    "lambert_grid_size",
    "nrevs_to_try",
    "precoast_grid_size",
    "limit_precoast_to_one_period",
    "w_time",
}


def mission_from_dict(value: Mapping[str, Any]) -> Mission:
    """Construct a validated mission from a schema-versioned mapping.

    The mapping names states, spacecraft, and dynamics, then references those
    objects from ordered phase declarations. Unknown keys are rejected so
    misspellings cannot silently change a mission.

    Args:
        value: Parsed JSON/YAML mapping using schema version 1.

    Returns:
        A mission composed from Octavian's existing public Python objects.

    Raises:
        MissionConfigError: If the mapping is malformed or references are invalid.
    """
    try:
        return _build_mission(value)
    except MissionConfigError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise MissionConfigError(str(exc)) from exc


def _build_mission(value: Mapping[str, Any]) -> Mission:
    root = mapping(value, "config")
    reject_unknown(root, _ROOT_KEYS, "config")
    schema_version = root.get("schema_version")
    schema_is_supported = (
        isinstance(schema_version, int)
        and not isinstance(schema_version, bool)
        and schema_version == SCHEMA_VERSION
    )
    if not schema_is_supported:
        raise MissionConfigError(
            f"config.schema_version must be {SCHEMA_VERSION}; got {schema_version!r}."
        )

    named_states = _build_states(root.get("states", {}))
    named_spacecraft = _build_spacecraft(root.get("spacecraft", {}))
    named_dynamics = build_dynamics(root.get("dynamics", {}))

    mission_config = mapping(required(root, "mission", "config"), "config.mission")
    reject_unknown(mission_config, _MISSION_KEYS, "config.mission")
    phases = _build_phases(
        required(mission_config, "phases", "config.mission"),
        states=named_states,
        spacecraft=named_spacecraft,
        dynamics=named_dynamics,
    )

    mission_kwargs: dict[str, Any] = {
        "phases": phases,
        "spacecraft": named_spacecraft,
        "name": mission_config.get("name", "Mission"),
        "initial_epoch": mission_config.get("initial_epoch"),
    }
    _add_execution_options(mission_kwargs, mission_config)
    _add_numerical_options(mission_kwargs, mission_config)

    mission = Mission(**mission_kwargs)
    mission.validate()
    return mission


def _add_execution_options(
    mission_kwargs: dict[str, Any], mission_config: Mapping[str, Any]
) -> None:
    """Translate objectives and runner configuration."""
    if "objectives" in mission_config:
        mission_kwargs["objectives"] = build_objectives(mission_config["objectives"])
    if "solver_options" in mission_config:
        mission_kwargs["solver_options"] = build_solver_options(mission_config["solver_options"])
    if "solve_config" in mission_config:
        mission_kwargs["solve_config"] = build_solve_config(mission_config["solve_config"])
    if "retry" in mission_config:
        mission_kwargs["retry"] = build_retry_policy(mission_config["retry"])
    if "plan" in mission_config:
        mission_kwargs["plan"] = build_run_plan(mission_config["plan"])


def _add_numerical_options(
    mission_kwargs: dict[str, Any], mission_config: Mapping[str, Any]
) -> None:
    """Translate mesh, Lambert, and time-weight settings."""
    integer_fields = (
        "mesh_nsegs_transfer",
        "mesh_nsegs_precoast",
        "lambert_grid_size",
        "precoast_grid_size",
    )
    for field_name in integer_fields:
        if field_name in mission_config:
            mission_kwargs[field_name] = int(mission_config[field_name])
    if "nrevs_to_try" in mission_config:
        mission_kwargs["nrevs_to_try"] = tuple(
            int(item)
            for item in sequence(mission_config["nrevs_to_try"], "config.mission.nrevs_to_try")
        )
    if "limit_precoast_to_one_period" in mission_config:
        mission_kwargs["limit_precoast_to_one_period"] = boolean(
            mission_config["limit_precoast_to_one_period"],
            "config.mission.limit_precoast_to_one_period",
        )
    if "w_time" in mission_config:
        mission_kwargs["w_time"] = float(mission_config["w_time"])


def _build_states(value: Any) -> dict[str, BoundaryState]:
    configs = mapping(value, "config.states")
    result: dict[str, BoundaryState] = {}
    for name, raw_config in configs.items():
        path = f"config.states.{name}"
        config = mapping(raw_config, path)
        reject_unknown(config, {"r_m", "v_mps"}, path)
        result[str(name)] = state(
            required(config, "r_m", path),
            required(config, "v_mps", path),
        )
    return result


def _build_spacecraft(value: Any) -> dict[str, Spacecraft]:
    configs = mapping(value, "config.spacecraft")
    result: dict[str, Spacecraft] = {}
    for reference_name, raw_config in configs.items():
        path = f"config.spacecraft.{reference_name}"
        config = mapping(raw_config, path)
        reject_unknown(
            config,
            {"name", "dry_mass_kg", "thrusters", "cannonball", "info"},
            path,
        )
        thrusters = [
            _build_thruster(item, f"{path}.thrusters[{index}]")
            for index, item in enumerate(sequence(config.get("thrusters", []), f"{path}.thrusters"))
        ]
        result[str(reference_name)] = Spacecraft(
            name=str(config.get("name", reference_name)),
            dry_mass_kg=float(config.get("dry_mass_kg", 0.0)),
            thrusters=thrusters,
            cannonball=_build_cannonball(
                config.get("cannonball", {}),
                f"{path}.cannonball",
            ),
            info=dict(mapping(config.get("info", {}), f"{path}.info")),
        )
    return result


def _build_cannonball(value: Any, path: str) -> Cannonball:
    """Build constant-area drag and SRP spacecraft properties."""
    config = mapping(value, path)
    reject_unknown(
        config,
        {
            "drag_area_m2",
            "drag_coefficient",
            "srp_area_m2",
            "reflectivity_coefficient",
        },
        path,
    )
    return Cannonball(
        drag_area_m2=float(config.get("drag_area_m2", 0.0)),
        drag_coefficient=float(config.get("drag_coefficient", 2.2)),
        srp_area_m2=float(config.get("srp_area_m2", 0.0)),
        reflectivity_coefficient=float(config.get("reflectivity_coefficient", 1.3)),
    )


def _build_thruster(value: Any, path: str) -> Thruster:
    config = mapping(value, path)
    reject_unknown(
        config,
        {"name", "thrust_N", "isp_s", "propellant_mass_kg", "info"},
        path,
    )
    propellant = config.get("propellant_mass_kg")
    return Thruster(
        name=str(config.get("name", "main")),
        thrust_N=float(config.get("thrust_N", 0.0)),
        isp_s=float(config.get("isp_s", 0.0)),
        propellant_mass_kg=None if propellant is None else float(propellant),
        info=dict(mapping(config.get("info", {}), f"{path}.info")),
    )


def _build_phases(
    value: Any,
    *,
    states: Mapping[str, BoundaryState],
    spacecraft: Mapping[str, Spacecraft],
    dynamics: Mapping[str, Any],
) -> list[Phase]:
    raw_phases = sequence(value, "config.mission.phases")
    phases: list[Phase] = []
    phases_by_name: dict[str, Phase] = {}
    for index, raw_phase in enumerate(raw_phases):
        path = f"config.mission.phases[{index}]"
        phase = _build_one_phase(
            raw_phase,
            path,
            phases_by_name=phases_by_name,
            states=states,
            spacecraft=spacecraft,
            dynamics=dynamics,
        )
        if phase.name in phases_by_name:
            raise MissionConfigError(f"{path}.name duplicates phase {phase.name!r}.")
        phases.append(phase)
        phases_by_name[phase.name] = phase
    return phases


def _build_one_phase(
    value: Any,
    path: str,
    *,
    phases_by_name: Mapping[str, Phase],
    states: Mapping[str, BoundaryState],
    spacecraft: Mapping[str, Spacecraft],
    dynamics: Mapping[str, Any],
) -> Phase:
    config = mapping(value, path)
    allowed = {
        "name",
        "mode",
        "spacecraft",
        "dynamics",
        "initial_state",
        "final_state",
        "epoch",
        "constraints",
        "events",
        "variables",
        "previous",
        "link",
        "tof_bounds_s",
        "tof_is_relative",
        "info",
        "initial_guess",
    }
    reject_unknown(config, allowed, path)
    name = str(required(config, "name", path)).strip()
    if not name:
        raise MissionConfigError(f"{path}.name must not be empty.")
    previous = None
    if "previous" in config:
        previous = reference(config["previous"], phases_by_name, f"{path}.previous", "phase")

    return Phase(
        name=name,
        mode=str(config.get("mode", "coast")),
        spacecraft=optional_reference(
            config.get("spacecraft"), spacecraft, f"{path}.spacecraft", "spacecraft"
        ),
        dynamics=optional_reference(
            config.get("dynamics"), dynamics, f"{path}.dynamics", "dynamics"
        ),
        initial_state=optional_state(config.get("initial_state"), states, f"{path}.initial_state"),
        final_state=optional_state(config.get("final_state"), states, f"{path}.final_state"),
        epoch=config.get("epoch"),
        constraints=[
            build_constraint(item, f"{path}.constraints[{index}]", states)
            for index, item in enumerate(
                sequence(config.get("constraints", []), f"{path}.constraints")
            )
        ],
        events=[
            build_event(item, f"{path}.events[{index}]")
            for index, item in enumerate(sequence(config.get("events", []), f"{path}.events"))
        ],
        variables=[
            build_variable(item, f"{path}.variables[{index}]")
            for index, item in enumerate(sequence(config.get("variables", []), f"{path}.variables"))
        ],
        previous=previous,
        link=build_link(config["link"], f"{path}.link") if "link" in config else None,
        tof_bounds_s=optional_pair(config.get("tof_bounds_s"), f"{path}.tof_bounds_s"),
        tof_is_relative=boolean(config.get("tof_is_relative", False), f"{path}.tof_is_relative"),
        info=dict(mapping(config.get("info", {}), f"{path}.info")),
        initial_guess=(
            build_initial_guess(config["initial_guess"], f"{path}.initial_guess")
            if "initial_guess" in config
            else None
        ),
    )
