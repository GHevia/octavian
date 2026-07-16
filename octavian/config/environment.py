"""Parsers for dynamics, perturbations, frames, and solver scaling."""

from __future__ import annotations

from typing import Any

from ..coordinates import CoordinateFrame, SolverScaling
from ..models import Dynamics, Perturbations
from .errors import MissionConfigError
from .schema import boolean, mapping, normalized_type, reject_unknown, required, sequence


def build_dynamics(value: Any) -> dict[str, Dynamics]:
    """Build the named dynamics registry."""
    configs = mapping(value, "config.dynamics")
    return {
        str(name): _build_one_dynamics(raw_config, f"config.dynamics.{name}")
        for name, raw_config in configs.items()
    }


def _build_one_dynamics(value: Any, path: str) -> Dynamics:
    config = mapping(value, path)
    allowed = {
        "model",
        "central_body",
        "mu_m3ps2",
        "central_body_radius_m",
        "j2_coefficient",
        "third_body_table_step_s",
        "third_body_table_margin_s",
        "perturbations",
        "frame",
        "scaling",
        "chief_orbit_radius_m",
        "chief_name",
        "reference_length_m",
        "info",
    }
    reject_unknown(config, allowed, path)
    model = normalized_type(config.get("model", "two_body"))

    if model == "cwh":
        return _build_cwh_dynamics(config, path)
    if model not in {"two_body", "cartesian"}:
        raise MissionConfigError(f"{path}.model has unsupported value {model!r}.")
    return _build_cartesian_dynamics(config, path, model)


def _build_cwh_dynamics(config: Any, path: str) -> Dynamics:
    invalid = set(config) & {
        "mu_m3ps2",
        "central_body_radius_m",
        "j2_coefficient",
        "perturbations",
        "frame",
    }
    if invalid:
        names = ", ".join(sorted(invalid))
        raise MissionConfigError(f"{path} cannot combine model='cwh' with: {names}.")
    scaling = build_scaling(config["scaling"], f"{path}.scaling") if "scaling" in config else None
    return Dynamics.cwh(
        chief_orbit_radius_m=float(required(config, "chief_orbit_radius_m", path)),
        central_body=config.get("central_body", "earth"),
        chief_name=str(config.get("chief_name", "chief")),
        reference_length_m=float(config.get("reference_length_m", 1_000.0)),
        third_body_table_step_s=float(config.get("third_body_table_step_s", 3_600.0)),
        third_body_table_margin_s=float(config.get("third_body_table_margin_s", 86_400.0)),
        scaling=scaling,
        info=dict(mapping(config.get("info", {}), f"{path}.info")),
    )


def _build_cartesian_dynamics(config: Any, path: str, model: str) -> Dynamics:
    if set(config) & {"chief_orbit_radius_m", "chief_name", "reference_length_m"}:
        raise MissionConfigError(f"{path} contains CWH-only fields but model is {model!r}.")
    if "central_body" in config and set(config) & {
        "mu_m3ps2",
        "central_body_radius_m",
        "j2_coefficient",
    }:
        raise MissionConfigError(
            f"{path} must use either central_body or raw gravity/radius/J2 constants, not both."
        )

    kwargs: dict[str, Any] = {
        "third_body_table_step_s": float(config.get("third_body_table_step_s", 3_600.0)),
        "third_body_table_margin_s": float(config.get("third_body_table_margin_s", 86_400.0)),
        "info": dict(mapping(config.get("info", {}), f"{path}.info")),
    }
    for key in ("mu_m3ps2", "central_body_radius_m", "j2_coefficient", "central_body"):
        if key in config:
            kwargs[key] = config[key]
    if "perturbations" in config:
        kwargs["perturbations"] = build_perturbations(
            config["perturbations"], f"{path}.perturbations"
        )
    if "frame" in config:
        kwargs["frame"] = build_frame(config["frame"], f"{path}.frame")
    if "scaling" in config:
        kwargs["scaling"] = build_scaling(config["scaling"], f"{path}.scaling")
    return Dynamics(**kwargs)


def build_perturbations(value: Any, path: str) -> Perturbations:
    """Build a perturbation declaration."""
    config = mapping(value, path)
    allowed = {"j2", "moon", "sun", "srp", "drag", "third_bodies"}
    reject_unknown(config, allowed, path)
    third_bodies = tuple(
        str(item) for item in sequence(config.get("third_bodies", []), f"{path}.third_bodies")
    )
    return Perturbations(
        j2=boolean(config.get("j2", False), f"{path}.j2"),
        moon=boolean(config.get("moon", False), f"{path}.moon"),
        sun=boolean(config.get("sun", False), f"{path}.sun"),
        srp=boolean(config.get("srp", False), f"{path}.srp"),
        drag=boolean(config.get("drag", False), f"{path}.drag"),
        third_bodies=third_bodies,
    )


def build_frame(value: Any, path: str) -> CoordinateFrame:
    """Build coordinate-frame metadata."""
    config = mapping(value, path)
    reject_unknown(config, {"name", "origin", "orientation", "kind"}, path)
    return CoordinateFrame(
        name=str(required(config, "name", path)),
        origin=str(required(config, "origin", path)),
        orientation=str(required(config, "orientation", path)),
        kind=str(config.get("kind", "inertial")),  # type: ignore[arg-type]
    )


def build_scaling(value: Any, path: str) -> SolverScaling:
    """Build explicit solver characteristic units."""
    config = mapping(value, path)
    reject_unknown(config, {"length_m", "velocity_mps", "time_s", "mass_kg"}, path)
    return SolverScaling(
        length_m=float(required(config, "length_m", path)),
        velocity_mps=float(required(config, "velocity_mps", path)),
        time_s=float(required(config, "time_s", path)),
        mass_kg=float(config.get("mass_kg", 1.0)),
    )
