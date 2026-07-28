from __future__ import annotations

import numpy as np
import pytest

from octavian import constraints
from octavian.solvers.compiler.relative_constraint_compiler import (
    relative_constraint_report_rows,
)


def test_relative_constraint_factories_normalize_geometry() -> None:
    keep_out = constraints.keep_out_sphere(25.0, center_m=[1.0, 2.0, 3.0])
    cone = constraints.approach_cone([0.0, -4.0, 0.0], 20.0)
    lighting = constraints.lighting_angle(
        [2.0, 0.0, 0.0],
        min_angle_deg=30.0,
        max_angle_deg=120.0,
    )

    assert keep_out.where == "Path"
    assert keep_out.center_m == pytest.approx([1.0, 2.0, 3.0])
    assert cone.axis == pytest.approx([0.0, -1.0, 0.0])
    assert lighting.sun_direction == pytest.approx([1.0, 0.0, 0.0])


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: constraints.keep_out_sphere(0.0), "radius_m"),
        (lambda: constraints.approach_cone([0.0, 0.0, 0.0], 20.0), "axis"),
        (lambda: constraints.approach_cone([1.0, 0.0, 0.0], 90.0), "half_angle"),
        (
            lambda: constraints.lighting_angle(
                [1.0, 0.0, 0.0], min_angle_deg=80.0, max_angle_deg=20.0
            ),
            "lighting angles",
        ),
    ],
)
def test_relative_constraints_reject_invalid_geometry(factory, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_relative_constraint_reports_path_extrema() -> None:
    trajectory = np.asarray(
        [
            [0.0, -200.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-20.0, -100.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    declared = [
        constraints.keep_out_sphere(90.0),
        constraints.approach_cone([0.0, -1.0, 0.0], 15.0),
        constraints.lighting_angle(
            [1.0, 0.0, 0.0],
            min_angle_deg=80.0,
            max_angle_deg=110.0,
        ),
    ]
    rows = [
        row
        for constraint in declared
        for row in relative_constraint_report_rows(
            phase_name="approach",
            constraint=constraint,
            phase_traj=trajectory,
        )
    ]

    assert [row["constraint"] for row in rows] == [
        "keep_out_sphere",
        "approach_cone",
        "lighting_min_angle_deg",
        "lighting_max_angle_deg",
    ]
    assert rows[0]["actual"] == pytest.approx(np.sqrt(20.0**2 + 100.0**2))
    assert rows[1]["actual"] == pytest.approx(np.rad2deg(np.arctan2(20.0, 100.0)))
    assert all(row["satisfied"] for row in rows)


def test_solar_phase_angle_uses_time_varying_ric_directions() -> None:
    trajectory = np.asarray(
        [
            [100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 10.0],
        ]
    )
    constraint = constraints.solar_phase_angle(
        min_angle_deg=0.0,
        max_angle_deg=1.0,
    )

    def sun_direction_at(times: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [
                np.cos(np.pi * times / 20.0),
                np.sin(np.pi * times / 20.0),
                np.zeros_like(times),
            ]
        )

    rows = relative_constraint_report_rows(
        phase_name="approach",
        constraint=constraint,
        phase_traj=trajectory,
        solar_direction_at=sun_direction_at,
    )

    assert [row["constraint"] for row in rows] == [
        "solar_phase_min_angle_deg",
        "solar_phase_max_angle_deg",
    ]
    assert rows[0]["actual"] == pytest.approx(0.0)
    assert rows[1]["actual"] == pytest.approx(0.0)
    assert all(row["satisfied"] for row in rows)
