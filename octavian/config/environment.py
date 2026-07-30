"""Parsers for dynamics, perturbations, frames, and solver scaling."""

from __future__ import annotations

from typing import Any

from ..coordinates import CoordinateFrame, SolverScaling
from ..forces import ExponentialAtmosphere
from ..models import Dynamics, Perturbations
from ..phase import state
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
        "chief_initial_state_eci",
        "propagation_mode",
        "info",
    }
    reject_unknown(config, allowed, path)
    model = normalized_type(config.get("model", "two_body"))

    if model == "cwh":
        return _build_cwh_dynamics(config, path)
    if model in {"relative", "nonlinear_relative"}:
        return _build_relative_dynamics(config, path)
    if model not in {"two_body", "cartesian"}:
        raise MissionConfigError(f"{path}.model has unsupported value {model!r}.")
    return _build_cartesian_dynamics(config, path, model)


def _build_cwh_dynamics(config: Any, path: str) -> Dynamics:
    invalid = set(config) & {
        "mu_m3ps2",
        "central_body_radius_m",
        "j2_coefficient",
        "frame",
        "propagation_mode",
    }
    if invalid:
        names = ", ".join(sorted(invalid))
        raise MissionConfigError(f"{path} cannot combine model='cwh' with: {names}.")
    scaling = build_scaling(config["scaling"], f"{path}.scaling") if "scaling" in config else None
    chief_initial_state = _build_chief_state(config, path, required_value=False)
    return Dynamics.cwh(
        chief_orbit_radius_m=float(required(config, "chief_orbit_radius_m", path)),
        central_body=config.get("central_body", "earth"),
        chief_name=str(config.get("chief_name", "chief")),
        reference_length_m=float(config.get("reference_length_m", 1_000.0)),
        chief_initial_state_eci=chief_initial_state,
        perturbations=(
            build_perturbations(config["perturbations"], f"{path}.perturbations")
            if "perturbations" in config
            else None
        ),
        third_body_table_step_s=float(config.get("third_body_table_step_s", 3_600.0)),
        third_body_table_margin_s=float(config.get("third_body_table_margin_s", 86_400.0)),
        scaling=scaling,
        info=dict(mapping(config.get("info", {}), f"{path}.info")),
    )


def _build_relative_dynamics(config: Any, path: str) -> Dynamics:
    """Build one nonlinear or relative-element dynamics formulation."""
    invalid = set(config) & {
        "mu_m3ps2",
        "central_body_radius_m",
        "j2_coefficient",
        "frame",
        "chief_orbit_radius_m",
    }
    if invalid:
        names = ", ".join(sorted(invalid))
        raise MissionConfigError(f"{path} cannot combine model='relative' with: {names}.")
    scaling = build_scaling(config["scaling"], f"{path}.scaling") if "scaling" in config else None
    chief_initial_state = _build_chief_state(config, path, required_value=True)
    return Dynamics.relative(
        chief_initial_state_eci=chief_initial_state,
        central_body=config.get("central_body", "earth"),
        chief_name=str(config.get("chief_name", "chief")),
        reference_length_m=float(config.get("reference_length_m", 1_000.0)),
        propagation_mode=str(config.get("propagation_mode", "coupled_eci")),
        perturbations=(
            build_perturbations(config["perturbations"], f"{path}.perturbations")
            if "perturbations" in config
            else None
        ),
        third_body_table_step_s=float(config.get("third_body_table_step_s", 3_600.0)),
        third_body_table_margin_s=float(config.get("third_body_table_margin_s", 86_400.0)),
        scaling=scaling,
        info=dict(mapping(config.get("info", {}), f"{path}.info")),
    )


def _build_chief_state(
    config: Any,
    path: str,
    *,
    required_value: bool,
):
    if "chief_initial_state_eci" not in config:
        if required_value:
            raise MissionConfigError(
                f"{path}.chief_initial_state_eci is required for model='relative'."
            )
        return None
    chief_path = f"{path}.chief_initial_state_eci"
    chief_config = mapping(config["chief_initial_state_eci"], chief_path)
    reject_unknown(chief_config, {"r_m", "v_mps"}, chief_path)
    return state(
        required(chief_config, "r_m", chief_path),
        required(chief_config, "v_mps", chief_path),
    )


def _build_cartesian_dynamics(config: Any, path: str, model: str) -> Dynamics:
    if set(config) & {
        "chief_orbit_radius_m",
        "chief_name",
        "reference_length_m",
        "chief_initial_state_eci",
        "propagation_mode",
    }:
        raise MissionConfigError(f"{path} contains relative-model fields but model is {model!r}.")
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
    allowed = {
        "j2",
        "moon",
        "sun",
        "srp",
        "drag",
        "third_bodies",
        "atmosphere",
        "solar_pressure_at_1au_Npm2",
    }
    reject_unknown(config, allowed, path)
    third_bodies = tuple(
        str(item) for item in sequence(config.get("third_bodies", []), f"{path}.third_bodies")
    )
    atmosphere = (
        _build_atmosphere(config["atmosphere"], f"{path}.atmosphere")
        if "atmosphere" in config
        else None
    )
    return Perturbations(
        j2=boolean(config.get("j2", False), f"{path}.j2"),
        moon=boolean(config.get("moon", False), f"{path}.moon"),
        sun=boolean(config.get("sun", False), f"{path}.sun"),
        srp=boolean(config.get("srp", False), f"{path}.srp"),
        drag=boolean(config.get("drag", False), f"{path}.drag"),
        third_bodies=third_bodies,
        atmosphere=atmosphere,
        solar_pressure_at_1au_Npm2=float(config.get("solar_pressure_at_1au_Npm2", 4.56e-6)),
    )


def _build_atmosphere(value: Any, path: str) -> ExponentialAtmosphere:
    """Build a constant-scale-height atmosphere configuration."""
    config = mapping(value, path)
    fields = {
        "reference_density_kgpm3",
        "reference_altitude_m",
        "scale_height_m",
        "rotation_rate_radps",
    }
    reject_unknown(config, fields, path)
    return ExponentialAtmosphere(
        reference_density_kgpm3=float(required(config, "reference_density_kgpm3", path)),
        reference_altitude_m=float(required(config, "reference_altitude_m", path)),
        scale_height_m=float(required(config, "scale_height_m", path)),
        rotation_rate_radps=float(required(config, "rotation_rate_radps", path)),
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
