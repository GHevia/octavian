from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from octavian import constraints, state
from octavian.solvers import constraint_compiler


def test_every_builtin_constraint_owns_its_apply_method() -> None:
    boundary = state([7.0e6, 0.0, 0.0], [0.0, 7.5e3, 0.0])
    declarations = (
        constraints.semi_major_axis(7.0e6),
        constraints.eccentricity(0.1),
        constraints.inclination_deg(28.5),
        constraints.min_radius(6.5e6),
        constraints.keep_out_sphere(10.0),
        constraints.approach_cone([1.0, 0.0, 0.0], 20.0),
        constraints.lighting_angle([1.0, 0.0, 0.0]),
        constraints.solar_phase_angle(),
        constraints.state(boundary),
        constraints.position(boundary.r_m),
        constraints.state_component("x", 1.0),
        constraints.periodic_state(),
        constraints.jacobi_constant(3.1),
        constraints.ric_state("R", 10.0),
        constraints.relative_orbital_element("delta_a", 0.0),
        constraints.relative_orbital_elements([0.0] * 6),
    )

    assert all(
        type(declaration).apply is not constraints.Constraint.apply
        for declaration in declarations
    )


@dataclass(frozen=True, slots=True)
class _DeveloperConstraint(constraints.Constraint):
    """Extension used to prove that compilation needs no type registration."""

    kind: ClassVar[str] = "developer_constraint"
    family: ClassVar[str] = "test"
    where: str = "Path"

    @property
    def value(self) -> float:
        return 42.0

    def apply(
        self,
        phase: list[tuple[float, Any]],
        context: constraints.ConstraintApplicationContext,
    ) -> None:
        phase.append((self.value, context.layout))


def test_custom_constraint_compiles_without_central_type_dispatch() -> None:
    compiled: list[tuple[float, Any]] = []
    layout = object()
    context = constraint_compiler.ConstraintContext(layout=layout)

    constraint_compiler.apply_constraints(
        compiled,
        [_DeveloperConstraint()],
        context,
    )

    assert compiled == [(42.0, layout)]
