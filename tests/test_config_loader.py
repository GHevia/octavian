from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from octavian import MissionConfigError, load_mission, load_mission_mapping

ROOT = Path(__file__).resolve().parents[1]


def _basic_config() -> dict:
    return {
        "schema_version": 1,
        "states": {
            "initial": {"r_m": [7_000_000.0, 0.0, 0.0], "v_mps": [0.0, 7_500.0, 0.0]},
            "target": {"r_m": [-8_000_000.0, 0.0, 0.0], "v_mps": [0.0, -7_000.0, 0.0]},
        },
        "spacecraft": {
            "vehicle": {
                "dry_mass_kg": 500.0,
                "thrusters": [
                    {
                        "name": "main",
                        "thrust_N": 2_000.0,
                        "isp_s": 320.0,
                        "propellant_mass_kg": 50.0,
                    }
                ],
            }
        },
        "dynamics": {"earth": {"central_body": "earth"}},
        "mission": {
            "name": "Config fixture",
            "phases": [
                {
                    "name": "transfer",
                    "mode": "coast",
                    "spacecraft": "vehicle",
                    "dynamics": "earth",
                    "initial_state": "initial",
                    "final_state": "target",
                    "tof_bounds_s": [600.0, 7_200.0],
                    "constraints": [
                        {"type": "state", "state": "initial", "where": "Front"},
                        {"type": "state", "state": "target", "where": "Back"},
                        {"type": "min_radius", "r_min_m": 6_500_000.0},
                    ],
                    "variables": [
                        {"type": "impulsive_delta_v", "where": "Front"},
                        {"type": "impulsive_delta_v", "where": "Back"},
                    ],
                }
            ],
            "objectives": [{"type": "delta_v"}, {"type": "time", "weight": 0.1}],
            "solver_options": {"print_level": 0, "asset_threads": [1, 1]},
            "nrevs_to_try": [0],
        },
    }


def test_mapping_builds_the_public_mission_objects() -> None:
    mission = load_mission_mapping(_basic_config())

    assert mission.name == "Config fixture"
    assert mission.spacecraft["vehicle"].initial_mass_kg == pytest.approx(550.0)
    assert mission.phases[0].spacecraft is mission.spacecraft["vehicle"]
    assert mission.phases[0].dynamics.central_body.name == "earth"
    assert [constraint.kind for constraint in mission.phases[0].constraints] == [
        "state",
        "state",
        "min_radius",
    ]
    assert [objective.kind for objective in mission.objectives] == ["delta_v", "time"]
    assert mission.solver_options.asset_threads == (1, 1)


def test_config_builds_cannonball_and_exponential_atmosphere() -> None:
    config = _basic_config()
    config["spacecraft"]["vehicle"]["cannonball"] = {
        "drag_area_m2": 3.0,
        "drag_coefficient": 2.1,
        "srp_area_m2": 4.0,
        "reflectivity_coefficient": 1.4,
    }
    config["dynamics"]["earth"]["perturbations"] = {
        "drag": True,
        "srp": True,
        "solar_pressure_at_1au_Npm2": 4.5e-6,
        "atmosphere": {
            "reference_density_kgpm3": 1.0e-12,
            "reference_altitude_m": 400_000.0,
            "scale_height_m": 50_000.0,
            "rotation_rate_radps": 7.2921159e-5,
        },
    }

    mission = load_mission_mapping(config)
    spacecraft = mission.spacecraft["vehicle"]
    perturbations = mission.phases[0].dynamics.active_perturbations()

    assert spacecraft.cannonball.drag_area_m2 == pytest.approx(3.0)
    assert spacecraft.cannonball.srp_area_m2 == pytest.approx(4.0)
    assert perturbations.drag is True
    assert perturbations.srp is True
    assert perturbations.atmosphere is not None
    assert perturbations.atmosphere.scale_height_m == pytest.approx(50_000.0)


def test_json_example_loads_without_special_case_code() -> None:
    mission = load_mission(ROOT / "examples/config/01_two_impulse_transfer.json")

    assert mission.name.startswith("Config: Hohmann")
    assert mission.phases[0].mode == "coast"
    assert mission.phases[0].initial_state.r_m[0] == pytest.approx(7_000_000.0)
    assert mission.phases[0].variables[1].where == "Back"


def test_yaml_uses_the_same_schema(tmp_path: Path) -> None:
    yaml = pytest.importorskip("yaml")
    config_path = tmp_path / "mission.yaml"
    config_path.write_text(yaml.safe_dump(_basic_config(), sort_keys=False), encoding="utf-8")

    mission = load_mission(config_path)

    assert mission.name == "Config fixture"
    assert mission.phases[0].dynamics.central_body.name == "earth"


def test_yaml_loader_rejects_python_object_tags(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    config_path = tmp_path / "unsafe.yaml"
    config_path.write_text(
        "!!python/object/apply:builtins.str ['not allowed']\n",
        encoding="utf-8",
    )

    with pytest.raises(MissionConfigError, match="Invalid YAML"):
        load_mission(config_path)


def test_unknown_keys_fail_instead_of_becoming_hidden_defaults() -> None:
    config = _basic_config()
    config["mission"]["lambert_grid_szie"] = 60

    with pytest.raises(MissionConfigError, match="lambert_grid_szie"):
        load_mission_mapping(config)


def test_boolean_fields_reject_truthy_strings() -> None:
    config = _basic_config()
    config["mission"]["phases"][0]["tof_is_relative"] = "false"

    with pytest.raises(MissionConfigError, match="tof_is_relative must be true or false"):
        load_mission_mapping(config)


def test_unresolved_named_references_report_the_config_path() -> None:
    config = _basic_config()
    config["mission"]["phases"][0]["dynamics"] = "mars"

    with pytest.raises(MissionConfigError, match=r"phases\[0\]\.dynamics.*mars"):
        load_mission_mapping(config)


def test_cwh_geometry_and_phase_links_translate_to_existing_objects() -> None:
    config = _basic_config()
    config["states"] = {
        "far": {"r_m": [0.0, -1_000.0, 0.0], "v_mps": [0.0, 0.0, 0.0]},
        "near": {"r_m": [0.0, -100.0, 0.0], "v_mps": [0.0, 0.0, 0.0]},
    }
    config["dynamics"] = {
        "relative": {
            "model": "cwh",
            "central_body": "earth",
            "chief_orbit_radius_m": 6_778_136.3,
            "chief_name": "Chief",
            "reference_length_m": 1_000.0,
        }
    }
    config["mission"]["phases"] = [
        {
            "name": "hold",
            "mode": "relative_coast",
            "spacecraft": "vehicle",
            "dynamics": "relative",
            "initial_state": "far",
            "tof_bounds_s": [10.0, 20.0],
        },
        {
            "name": "approach",
            "mode": "relative_coast",
            "previous": "hold",
            "link": "continuous",
            "final_state": "near",
            "tof_bounds_s": [1_200.0, 2_400.0],
            "tof_is_relative": True,
            "constraints": [
                {"type": "keep_out_sphere", "radius_m": 75.0},
                {
                    "type": "approach_cone",
                    "axis": [0.0, -1.0, 0.0],
                    "half_angle_deg": 30.0,
                },
                {
                    "type": "lighting_angle",
                    "sun_direction": [1.0, 0.0, 0.0],
                    "min_angle_deg": 85.0,
                    "max_angle_deg": 121.0,
                },
            ],
        },
    ]

    mission = load_mission_mapping(config)

    approach = mission.phases[1]
    assert approach.previous is mission.phases[0]
    assert approach.spacecraft is mission.phases[0].spacecraft
    assert approach.dynamics is mission.phases[0].dynamics
    assert approach.link.kind == "continuous"
    assert approach.dynamics.frame.kind == "relative"
    assert [constraint.kind for constraint in approach.constraints] == [
        "keep_out_sphere",
        "approach_cone",
        "lighting_angle",
    ]


def test_relative_config_accepts_chief_state_perturbations_and_solar_phase() -> None:
    config = _basic_config()
    radius_m = 6_778_136.3
    speed_mps = np.sqrt(3.986004418e14 / radius_m)
    config["dynamics"] = {
        "relative": {
            "model": "relative",
            "central_body": "earth",
            "chief_initial_state_eci": {
                "r_m": [radius_m, 0.0, 0.0],
                "v_mps": [0.0, speed_mps, 0.0],
            },
            "perturbations": {"j2": True, "sun": True},
        }
    }
    config["mission"]["initial_epoch"] = "2026-01-01T00:00:00Z"
    config["mission"]["phases"][0]["dynamics"] = "relative"
    config["mission"]["phases"][0]["mode"] = "relative_coast"
    config["mission"]["phases"][0]["constraints"].append(
        {
            "type": "solar_phase_angle",
            "min_angle_deg": 20.0,
            "max_angle_deg": 150.0,
        }
    )

    mission = load_mission_mapping(config)
    dynamics = mission.phases[0].dynamics

    assert dynamics.model.chief_initial_state_eci.r_m == pytest.approx([radius_m, 0.0, 0.0])
    assert dynamics.active_perturbations().j2 is True
    assert dynamics.active_perturbations().sun is True
    assert mission.phases[0].constraints[-1].kind == "solar_phase_angle"


def test_relative_element_mode_and_native_constraints_are_declarative() -> None:
    config = _basic_config()
    radius_m = 6_778_136.3
    speed_mps = np.sqrt(3.986004418e14 / radius_m)
    config["dynamics"] = {
        "relative": {
            "model": "relative",
            "propagation_mode": "damico",
            "chief_initial_state_eci": {
                "r_m": [radius_m, 0.0, 0.0],
                "v_mps": [0.0, speed_mps, 0.0],
            },
        }
    }
    phase = config["mission"]["phases"][0]
    phase["dynamics"] = "relative"
    phase["mode"] = "relative_coast"
    phase["constraints"] = [
        {
            "type": "relative_orbital_elements",
            "elements": [1e-4, -0.01, 0.0, 0.0, 0.0, 0.0],
            "where": "Front",
        },
        {
            "type": "relative_orbital_element",
            "element": "delta_lambda",
            "target": -0.02,
            "where": "Back",
        },
    ]

    mission = load_mission_mapping(config)

    assert mission.phases[0].dynamics.model.propagation_mode.value == "damico"
    assert [constraint.kind for constraint in mission.phases[0].constraints] == [
        "relative_orbital_elements",
        "relative_orbital_element",
    ]


def test_low_thrust_guess_and_orbital_constraints_are_declarative() -> None:
    config = _basic_config()
    phase = config["mission"]["phases"][0]
    phase["mode"] = "low_thrust"
    phase["initial_guess"] = {
        "type": "low_thrust_spiral",
        "throttle": 0.85,
        "steps_per_orbit": 120,
    }
    phase["constraints"] = [
        {"type": "state", "state": "initial", "where": "Front"},
        {"type": "semi_major_axis", "a_m": 8_000_000.0, "where": "Back", "tol_m": 10_000.0},
        {"type": "eccentricity", "e": 0.01, "where": "Back", "tol": 0.0099},
        {"type": "inclination_deg", "inc_deg": 5.0, "where": "Back", "tol_deg": 0.2},
    ]
    phase["variables"] = []
    config["mission"]["objectives"] = [{"type": "propellant"}]

    mission = load_mission_mapping(config)

    assert mission.phases[0].initial_guess.throttle == pytest.approx(0.85)
    assert [constraint.kind for constraint in mission.phases[0].constraints[1:]] == [
        "semi_major_axis",
        "eccentricity",
        "inclination_deg",
    ]
    assert mission.objectives[0].kind == "propellant"


def test_json_round_trip_input_remains_plain_data() -> None:
    config = _basic_config()
    mission = load_mission_mapping(json.loads(json.dumps(config)))

    assert mission.name == "Config fixture"
