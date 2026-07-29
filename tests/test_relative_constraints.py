from __future__ import annotations

import numpy as np
import pytest

from octavian import constraints
from octavian.coordinates import (
    COUPLED_RELATIVE_ECI,
    COUPLED_RELATIVE_RIC,
    DAMICO_RELATIVE_ELEMENTS,
)
from octavian.relative import ClassicalRelativeOrbitalElements
from octavian.solvers.compiler.relative_constraint_compiler import (
    relative_constraint_report_rows,
)
from octavian.solvers.compiler.relative_state_constraint_compiler import (
    apply_native_relative_constraint,
    relative_state_constraint_report_rows,
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


def test_native_relative_constraint_factories_normalize_names() -> None:
    ric_target = constraints.ric_state(
        "in-track",
        -25.0,
        where="end",
        tolerance=0.5,
    )
    damico_target = constraints.relative_orbital_element(
        "dlambda",
        0.01,
        where="end",
    )
    classical_target = constraints.relative_orbital_element(
        "dm",
        0.02,
        representation="classical",
    )

    assert ric_target.component == "I"
    assert ric_target.where == "Back"
    assert damico_target.element == "delta_lambda"
    assert damico_target.representation == "damico"
    assert classical_target.element == "delta_mean_anomaly"
    assert classical_target.representation == "classical_elements"


def test_relative_element_vector_constraint_rejects_mislabeled_dataclass() -> None:
    classical = ClassicalRelativeOrbitalElements(
        delta_a_m=100.0,
        delta_e=0.0,
        delta_i_rad=0.0,
        delta_raan_rad=0.0,
        delta_argp_rad=0.0,
        delta_mean_anomaly_rad=0.0,
    )
    inferred = constraints.relative_orbital_elements(classical)

    assert inferred.representation == "classical_elements"
    with pytest.raises(ValueError, match="does not match"):
        constraints.relative_orbital_elements(
            classical,
            representation="damico",
        )


class _NativeConstraintPhase:
    def __init__(self) -> None:
        self.boundaries: list[tuple[str, tuple[int, ...], np.ndarray]] = []
        self.equalities: list[tuple[str, object, tuple[int, ...]]] = []

    def addBoundaryValue(self, where, indices, values):  # type: ignore[no-untyped-def]
        self.boundaries.append(
            (
                str(where),
                tuple(int(index) for index in indices),
                np.asarray(values, dtype=float),
            )
        )

    def addEqualCon(self, where, expression, indices):  # type: ignore[no-untyped-def]
        self.equalities.append((str(where), expression, tuple(int(index) for index in indices)))


def test_native_relative_vector_constraint_uses_direct_roe_indices() -> None:
    phase = _NativeConstraintPhase()
    elements = np.asarray([1e-4, -0.01, 2e-4, 3e-4, -4e-4, 5e-4])
    constraint = constraints.relative_orbital_elements(elements, where="Front")

    apply_native_relative_constraint(
        phase,
        constraint,
        DAMICO_RELATIVE_ELEMENTS,
    )

    assert phase.boundaries[0][0:2] == ("Front", (0, 1, 2, 3, 4, 5))
    np.testing.assert_allclose(phase.boundaries[0][2], elements)


def test_ric_constraint_rejects_coupled_eci_conversion() -> None:
    with pytest.raises(ValueError, match="RIC component constraints"):
        apply_native_relative_constraint(
            _NativeConstraintPhase(),
            constraints.ric_state("R", 0.0),
            COUPLED_RELATIVE_ECI,
        )


def test_native_relative_constraint_report_uses_selected_state_index() -> None:
    trajectory = np.zeros((2, 13), dtype=float)
    trajectory[:, 12] = [0.0, 100.0]
    trajectory[-1, 7] = -25.0
    declared = constraints.ric_state("I", -25.0, where="Back")
    rows = relative_state_constraint_report_rows(
        phase_name="native_ric",
        constraint=declared,
        native_trajectory=trajectory,
        layout=COUPLED_RELATIVE_RIC,
    )

    assert rows[0]["constraint"] == "ric_I"
    assert rows[0]["actual"] == pytest.approx(-25.0)
    assert rows[0]["satisfied"] is True
