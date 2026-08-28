"""User-facing constraint declarations for composable missions.

The constraint objects in this module are intentionally small, explicit, and
Pythonic. They describe what the user wants and keep the corresponding solver
formula close by in ``apply``; compiler-owned contexts provide phase-specific
backend data without being retained by the declarations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Protocol

import numpy as np

from .specs import BoundaryState

Where = Literal["Front", "Back", "Path"]
RelativeElementRepresentation = Literal["damico", "classical_elements"]


class ConstraintApplicationContext(Protocol):
    """Compiler-owned data available while a constraint is applied."""

    vector_functions: Any
    layout: Any
    declared_phase: Any
    phase_index: int
    mu_m3ps2: float
    cr3bp_system: Any | None
    cr3bp_dimensional: bool
    is_relative_phase: bool
    relative_expressions: Any | None
    third_body_tables: dict[str, Any]
    solar_direction_table: Any | None
    solar_position_table: Any | None


class ConstraintReportContext(Protocol):
    """Solved phase data available to constraint reporting methods."""

    phase_name: str
    phase_trajectory: np.ndarray
    native_trajectory: np.ndarray
    relative_trajectory: np.ndarray | None
    layout: Any
    mu_m3ps2: float
    cr3bp_system: Any | None
    cr3bp_dimensional: bool
    solar_direction_at: Callable[[np.ndarray], np.ndarray] | None

_CARTESIAN_COMPONENT_ALIASES = {
    "x": "x",
    "rx": "x",
    "r_x": "x",
    "y": "y",
    "ry": "y",
    "r_y": "y",
    "z": "z",
    "rz": "z",
    "r_z": "z",
    "vx": "vx",
    "v_x": "vx",
    "xdot": "vx",
    "x_dot": "vx",
    "vy": "vy",
    "v_y": "vy",
    "ydot": "vy",
    "y_dot": "vy",
    "vz": "vz",
    "v_z": "vz",
    "zdot": "vz",
    "z_dot": "vz",
}


def _normalize_where(where: str) -> Where:
    """Normalize user spelling variants for constraint locations."""
    normalized = (where or "").strip().lower()
    if normalized in ("front", "start", "initial", "t0"):
        return "Front"
    if normalized in ("back", "end", "final", "tf"):
        return "Back"
    if normalized in ("path", "all", "trajectory"):
        return "Path"
    raise ValueError(f"Unknown where={where!r}. Use 'front', 'back', or 'path'.")


def _finite_float(name: str, value: float) -> float:
    """Validate and return a finite scalar float."""
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite.")
    return scalar


def _optional_nonnegative_float(name: str, value: float | None) -> float | None:
    """Validate an optional non-negative scalar float."""
    if value is None:
        return None
    scalar = _finite_float(name, value)
    if scalar < 0.0:
        raise ValueError(f"{name} must be >= 0.")
    return scalar


def _normalize_cartesian_component(component: str) -> str:
    """Return a canonical Cartesian position or velocity component name."""
    normalized = str(component).strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return _CARTESIAN_COMPONENT_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError("component must be one of x, y, z, vx, vy, or vz") from exc


def _cartesian_component_index(layout: Any, component: str) -> int:
    """Return one direct Cartesian component index for a state layout."""
    try:
        position_indices = layout.state_indices("position")
        velocity_indices = layout.state_indices("velocity")
    except KeyError as exc:
        raise ValueError(
            f"State layout {layout.name!r} does not expose a direct Cartesian "
            "position/velocity state"
        ) from exc
    component_indices = {
        "x": position_indices[0],
        "y": position_indices[1],
        "z": position_indices[2],
        "vx": velocity_indices[0],
        "vy": velocity_indices[1],
        "vz": velocity_indices[2],
    }
    return int(component_indices[component])


def _apply_expression_target(
    phase: Any,
    context: ConstraintApplicationContext,
    *,
    where: str,
    residual: Any,
    argument_indices: Sequence[int],
    tolerance: float | None,
) -> None:
    """Apply an equality or symmetric tolerance band to a scalar residual."""
    indices = tuple(int(index) for index in argument_indices)
    if tolerance is None:
        phase.addEqualCon(where, residual, indices)
        return
    vf = context.vector_functions
    phase.addInequalCon(where, vf.stack([residual - float(tolerance)]), indices)
    phase.addInequalCon(where, vf.stack([-residual - float(tolerance)]), indices)


def _apply_scalar_target(
    phase: Any,
    context: ConstraintApplicationContext,
    *,
    where: str,
    state_index: int,
    target: float,
    tolerance: float | None,
) -> None:
    """Apply an equality or tolerance band to one native state variable."""
    variable = context.vector_functions.Arguments(1).tolist()[0]
    _apply_expression_target(
        phase,
        context,
        where=where,
        residual=variable - float(target),
        argument_indices=(int(state_index),),
        tolerance=tolerance,
    )


def _validate_inertial_orbital_context(context: ConstraintApplicationContext) -> None:
    """Reject two-body osculating elements in relative and CR3BP frames."""
    if context.is_relative_phase:
        raise ValueError("Inertial orbital-element constraints are not valid in a relative frame.")
    if context.cr3bp_system is not None:
        raise ValueError(
            "Osculating two-body element constraints are not valid in a CR3BP frame."
        )


class Constraint(ABC):
    """Abstract base class for all mission constraints.

    Concrete constraints own their ASSET compilation formula in :meth:`apply`.
    The compiler supplies phase-specific data through a transient context.
    """

    kind: ClassVar[str]
    family: ClassVar[str]
    where: Where

    @property
    @abstractmethod
    def value(self) -> Any:
        """Return the canonical constraint payload."""

    @abstractmethod
    def apply(self, phase: Any, context: ConstraintApplicationContext) -> None:
        """Apply this declaration to a compiled solver phase."""

    def report(self, context: ConstraintReportContext) -> list[dict[str, Any]]:
        """Return solved-value report rows when this constraint supports them."""
        del context
        return []


class OrbitalElementConstraint(Constraint):
    """Marker base class for orbital-element constraints."""

    family: ClassVar[str] = "orbital_element"

    @property
    @abstractmethod
    def element_name(self) -> str:
        """Human-readable orbital element name."""


@dataclass(frozen=True, slots=True)
class SemiMajorAxis(OrbitalElementConstraint):
    """Constrain semi-major axis in meters."""

    kind: ClassVar[str] = "semi_major_axis"

    a_m: float = 0.0
    where: Where = "Path"
    tol_m: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "where", _normalize_where(self.where))
        semi_major_axis_m = _finite_float("a_m", self.a_m)
        if semi_major_axis_m <= 0.0:
            raise ValueError("a_m must be > 0.")
        object.__setattr__(self, "a_m", semi_major_axis_m)
        tolerance_m = _optional_nonnegative_float("tol_m", self.tol_m)
        if tolerance_m is not None and tolerance_m >= semi_major_axis_m:
            raise ValueError("tol_m must be smaller than a_m.")
        object.__setattr__(self, "tol_m", tolerance_m)

    @property
    def element_name(self) -> str:
        """Return the compiler-facing element identifier."""
        return "semi_major_axis"

    @property
    def value(self) -> dict[str, float | None]:
        """Return the target and optional tolerance in meters."""
        return {"a_m": float(self.a_m), "tol_m": self.tol_m}

    def apply(self, phase: Any, context: ConstraintApplicationContext) -> None:
        """Apply the two-body specific-energy definition of semi-major axis."""
        _validate_inertial_orbital_context(context)
        vf = context.vector_functions
        mu_m3ps2 = float(context.mu_m3ps2)
        arguments = vf.Arguments(6)
        position, velocity = arguments.tolist([(0, 3), (3, 3)])
        radius = position.norm()
        speed = velocity.norm()
        specific_energy = 0.5 * speed**2 - mu_m3ps2 / radius
        semi_major_axis_m = -0.5 * mu_m3ps2 / specific_energy
        if self.tol_m is None:
            phase.addEqualCon(
                self.where,
                vf.stack([semi_major_axis_m - self.a_m]),
                range(0, 6),
            )
            return
        phase.addInequalCon(
            self.where,
            vf.stack([semi_major_axis_m - (self.a_m + self.tol_m)]),
            range(0, 6),
        )
        phase.addInequalCon(
            self.where,
            vf.stack([(self.a_m - self.tol_m) - semi_major_axis_m]),
            range(0, 6),
        )

    def report(self, context: ConstraintReportContext) -> list[dict[str, Any]]:
        """Report solved semi-major axis at the selected boundary."""
        if self.where not in {"Front", "Back"}:
            return []
        return [_orbital_report_row(self, context, self.a_m, self.tol_m, "a_m")]


@dataclass(frozen=True, slots=True)
class Eccentricity(OrbitalElementConstraint):
    """Constrain orbital eccentricity.

    Circular targets are intentionally rejected for now because they need a
    non-singular formulation.
    """

    kind: ClassVar[str] = "eccentricity"

    e: float = 0.0
    where: Where = "Path"
    tol: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "where", _normalize_where(self.where))
        eccentricity_value = _finite_float("e", self.e)
        if not (0.0 < eccentricity_value < 1.0):
            raise ValueError(
                "Eccentricity constraint currently requires 0 < e < 1. "
                "Circular and non-elliptic handling is deferred."
            )
        object.__setattr__(self, "e", eccentricity_value)
        tolerance = _optional_nonnegative_float("tol", self.tol)
        if tolerance is not None and tolerance >= eccentricity_value:
            raise ValueError("tol must be smaller than e for eccentricity constraints.")
        object.__setattr__(self, "tol", tolerance)

    @property
    def element_name(self) -> str:
        """Return the compiler-facing element identifier."""
        return "eccentricity"

    @property
    def value(self) -> dict[str, float | None]:
        """Return the dimensionless target and optional tolerance."""
        return {"e": float(self.e), "tol": self.tol}

    def apply(self, phase: Any, context: ConstraintApplicationContext) -> None:
        """Apply eccentricity through its non-negative squared expression."""
        _validate_inertial_orbital_context(context)
        vf = context.vector_functions
        mu_m3ps2 = float(context.mu_m3ps2)
        arguments = vf.Arguments(6)
        position, velocity = arguments.tolist([(0, 3), (3, 3)])
        angular_momentum = position.cross(velocity)
        radius = position.norm()
        speed = velocity.norm()
        specific_energy = 0.5 * speed**2 - mu_m3ps2 / radius
        angular_momentum_sq = angular_momentum.dot(angular_momentum)
        eccentricity_sq = 1.0 + (
            2.0 * specific_energy * angular_momentum_sq
        ) / mu_m3ps2**2
        if self.tol is None:
            phase.addEqualCon(
                self.where,
                vf.stack([eccentricity_sq - self.e**2]),
                range(0, 6),
            )
            return
        phase.addInequalCon(
            self.where,
            vf.stack([eccentricity_sq - (self.e + self.tol) ** 2]),
            range(0, 6),
        )
        phase.addInequalCon(
            self.where,
            vf.stack([(self.e - self.tol) ** 2 - eccentricity_sq]),
            range(0, 6),
        )

    def report(self, context: ConstraintReportContext) -> list[dict[str, Any]]:
        """Report solved eccentricity at the selected boundary."""
        if self.where not in {"Front", "Back"}:
            return []
        return [_orbital_report_row(self, context, self.e, self.tol, "e")]


@dataclass(frozen=True, slots=True)
class InclinationDeg(OrbitalElementConstraint):
    """Constrain orbital inclination in degrees.

    Equatorial targets are intentionally rejected for now because they need a
    non-singular formulation.
    """

    kind: ClassVar[str] = "inclination_deg"

    inc_deg: float = 0.0
    where: Where = "Path"
    tol_deg: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "where", _normalize_where(self.where))
        inclination_deg = _finite_float("inc_deg", self.inc_deg)
        if not (0.0 < inclination_deg < 180.0):
            raise ValueError(
                "Inclination constraint currently requires 0 < inc_deg < 180. "
                "Equatorial handling is deferred."
            )
        object.__setattr__(self, "inc_deg", inclination_deg)
        tolerance_deg = _optional_nonnegative_float("tol_deg", self.tol_deg)
        if tolerance_deg is not None and tolerance_deg >= min(
            inclination_deg, 180.0 - inclination_deg
        ):
            raise ValueError("tol_deg is too large for the requested inclination target.")
        object.__setattr__(self, "tol_deg", tolerance_deg)

    @property
    def element_name(self) -> str:
        """Return the compiler-facing element identifier."""
        return "inclination_deg"

    @property
    def value(self) -> dict[str, float | None]:
        """Return the target and optional tolerance in degrees."""
        return {"inc_deg": float(self.inc_deg), "tol_deg": self.tol_deg}

    def apply(self, phase: Any, context: ConstraintApplicationContext) -> None:
        """Apply inclination using the angular-momentum direction cosine."""
        _validate_inertial_orbital_context(context)
        vf = context.vector_functions
        arguments = vf.Arguments(6)
        position, velocity = arguments.tolist([(0, 3), (3, 3)])
        angular_momentum = position.cross(velocity)
        inclination_cosine = angular_momentum.normalized()[2]
        target_cosine = float(np.cos(np.deg2rad(self.inc_deg)))
        if self.tol_deg is None:
            phase.addEqualCon(
                self.where,
                vf.stack([inclination_cosine - target_cosine]),
                range(0, 6),
            )
            return
        upper_cosine = float(np.cos(np.deg2rad(self.inc_deg - self.tol_deg)))
        lower_cosine = float(np.cos(np.deg2rad(self.inc_deg + self.tol_deg)))
        phase.addInequalCon(
            self.where,
            vf.stack([inclination_cosine - upper_cosine]),
            range(0, 6),
        )
        phase.addInequalCon(
            self.where,
            vf.stack([lower_cosine - inclination_cosine]),
            range(0, 6),
        )

    def report(self, context: ConstraintReportContext) -> list[dict[str, Any]]:
        """Report solved inclination at the selected boundary."""
        if self.where not in {"Front", "Back"}:
            return []
        return [
            _orbital_report_row(
                self,
                context,
                self.inc_deg,
                self.tol_deg,
                "inc_deg",
            )
        ]


@dataclass(frozen=True, slots=True)
class MinRadius(Constraint):
    """Constrain the trajectory radius norm from below."""

    kind: ClassVar[str] = "min_radius"
    family: ClassVar[str] = "path_geometry"

    r_min_m: float = 0.0
    where: Where = "Path"

    def __post_init__(self) -> None:
        object.__setattr__(self, "where", _normalize_where(self.where))

    @property
    def value(self) -> float:
        """Return the minimum radius in meters."""
        return float(self.r_min_m)

    def apply(self, phase: Any, context: ConstraintApplicationContext) -> None:
        """Apply a lower bound to the Cartesian position norm."""
        if context.layout.name.endswith("relative_elements"):
            raise ValueError(
                "Minimum Cartesian range is not a native relative-element constraint; "
                "select a RIC formulation."
            )
        if context.relative_expressions is None:
            phase.addLowerNormBound(self.where, "R", self.r_min_m)
            return
        position = context.relative_expressions.position
        violation = self.r_min_m**2 - position.dot(position)
        phase.addInequalCon(
            self.where,
            context.vector_functions.stack([violation]),
            context.relative_expressions.argument_indices,
        )


def _unit_vector(name: str, value: Sequence[float]) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(3)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain finite values")
    magnitude = float(np.linalg.norm(vector))
    if magnitude <= 0.0:
        raise ValueError(f"{name} must have non-zero norm")
    return vector / magnitude


def _finite_vec3(name: str, value: Sequence[float]) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(3)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain finite values")
    return vector


def _relative_position_expression(
    context: ConstraintApplicationContext,
) -> tuple[Any, tuple[int, ...]]:
    """Return the public relative position expression and phase indices."""
    if context.relative_expressions is not None:
        return (
            context.relative_expressions.position,
            tuple(context.relative_expressions.argument_indices),
        )
    if context.layout.name.endswith("relative_elements"):
        raise ValueError(
            "Cartesian relative-geometry constraints require a native RIC or "
            "coupled-ECI propagation mode; they are not defined directly on relative elements."
        )
    return (
        context.vector_functions.Arguments(3),
        tuple(context.layout.state_indices("position")),
    )


def _apply_angle_bounds(
    phase: Any,
    vf: Any,
    where: str,
    offset: Any,
    direction: Any,
    min_angle_deg: float,
    max_angle_deg: float,
    argument_indices: Sequence[int],
    *,
    scale_direction_norm: bool = True,
) -> None:
    """Apply angle bounds without inverse-trigonometric vector functions."""
    distance = offset.norm()
    direction_norm = direction.norm() if scale_direction_norm and hasattr(direction, "norm") else 1.0
    projection = offset.dot(direction)
    min_cosine = float(np.cos(np.deg2rad(min_angle_deg)))
    max_cosine = float(np.cos(np.deg2rad(max_angle_deg)))
    scale = distance * direction_norm
    indices = tuple(int(index) for index in argument_indices)
    phase.addInequalCon(
        where,
        vf.stack([max_cosine * scale - projection]),
        indices,
    )
    phase.addInequalCon(
        where,
        vf.stack([projection - min_cosine * scale]),
        indices,
    )


@dataclass(frozen=True, slots=True)
class KeepOutSphere(Constraint):
    """Keep Cartesian position outside a sphere centered in the phase frame."""

    kind: ClassVar[str] = "keep_out_sphere"
    family: ClassVar[str] = "relative_geometry"

    radius_m: float = 0.0
    center_m: Sequence[float] = (0.0, 0.0, 0.0)
    where: Where = "Path"

    def __post_init__(self) -> None:
        object.__setattr__(self, "where", _normalize_where(self.where))
        radius = _finite_float("radius_m", self.radius_m)
        if radius <= 0.0:
            raise ValueError("radius_m must be > 0")
        object.__setattr__(self, "radius_m", radius)
        object.__setattr__(self, "center_m", _finite_vec3("center_m", self.center_m))

    @property
    def value(self) -> dict[str, Any]:
        """Return the keep-out radius and center."""
        return {"radius_m": self.radius_m, "center_m": self.center_m}

    def apply(self, phase: Any, context: ConstraintApplicationContext) -> None:
        """Apply ``radius² - ||position - center||² <= 0``."""
        position, indices = _relative_position_expression(context)
        offset = position - np.asarray(self.center_m, dtype=float)
        violation = self.radius_m**2 - offset.dot(offset)
        phase.addInequalCon(
            self.where,
            context.vector_functions.stack([violation]),
            indices,
        )

    def report(self, context: ConstraintReportContext) -> list[dict[str, Any]]:
        """Report the minimum solved separation from the keep-out center."""
        positions = _positions_at_location(context.phase_trajectory, self.where)
        actual = float(np.min(np.linalg.norm(positions - self.center_m, axis=1)))
        tolerance = max(1.0e-6, 1.0e-7 * self.radius_m)
        return [
            _report_row(
                self,
                context,
                name="keep_out_sphere",
                target=self.radius_m,
                actual=actual,
                error=actual - self.radius_m,
                tolerance=tolerance,
                satisfied=actual >= self.radius_m - tolerance,
            )
        ]


@dataclass(frozen=True, slots=True)
class ApproachCone(Constraint):
    """Keep position inside a one-sided cone extending from a vertex."""

    kind: ClassVar[str] = "approach_cone"
    family: ClassVar[str] = "relative_geometry"

    axis: Sequence[float] = (1.0, 0.0, 0.0)
    half_angle_deg: float = 30.0
    vertex_m: Sequence[float] = (0.0, 0.0, 0.0)
    where: Where = "Path"

    def __post_init__(self) -> None:
        object.__setattr__(self, "where", _normalize_where(self.where))
        half_angle = _finite_float("half_angle_deg", self.half_angle_deg)
        if not (0.0 < half_angle < 90.0):
            raise ValueError("half_angle_deg must satisfy 0 < angle < 90")
        object.__setattr__(self, "half_angle_deg", half_angle)
        object.__setattr__(self, "axis", _unit_vector("axis", self.axis))
        object.__setattr__(self, "vertex_m", _finite_vec3("vertex_m", self.vertex_m))

    @property
    def value(self) -> dict[str, Any]:
        """Return the normalized cone declaration."""
        return {
            "axis": self.axis,
            "half_angle_deg": self.half_angle_deg,
            "vertex_m": self.vertex_m,
        }

    def apply(self, phase: Any, context: ConstraintApplicationContext) -> None:
        """Apply the cone-angle inequality and retain its forward half."""
        position, indices = _relative_position_expression(context)
        offset = position - np.asarray(self.vertex_m, dtype=float)
        axial_distance = offset.dot(np.asarray(self.axis, dtype=float))
        cosine_sq = float(np.cos(np.deg2rad(self.half_angle_deg)) ** 2)
        cone_violation = cosine_sq * offset.dot(offset) - axial_distance**2
        vf = context.vector_functions
        phase.addInequalCon(self.where, vf.stack([cone_violation]), indices)
        phase.addInequalCon(self.where, vf.stack([-axial_distance]), indices)

    def report(self, context: ConstraintReportContext) -> list[dict[str, Any]]:
        """Report the maximum solved angle from the cone axis."""
        positions = _positions_at_location(context.phase_trajectory, self.where)
        angles = _angles_deg(positions - self.vertex_m, np.asarray(self.axis, dtype=float))
        actual = float(np.max(angles)) if angles.size else float("nan")
        tolerance = 1.0e-4
        return [
            _report_row(
                self,
                context,
                name="approach_cone",
                target=self.half_angle_deg,
                actual=actual,
                error=actual - self.half_angle_deg,
                tolerance=tolerance,
                satisfied=bool(
                    np.isfinite(actual) and actual <= self.half_angle_deg + tolerance
                ),
            )
        ]


@dataclass(frozen=True, slots=True)
class LightingAngle(Constraint):
    """Bound the angle from position to a fixed illumination direction.

    ``sun_direction`` is expressed in the same frame as the phase position and
    points from the constraint origin toward the light source.
    """

    kind: ClassVar[str] = "lighting_angle"
    family: ClassVar[str] = "relative_geometry"

    sun_direction: Sequence[float] = (1.0, 0.0, 0.0)
    min_angle_deg: float = 0.0
    max_angle_deg: float = 180.0
    origin_m: Sequence[float] = (0.0, 0.0, 0.0)
    where: Where = "Path"

    def __post_init__(self) -> None:
        object.__setattr__(self, "where", _normalize_where(self.where))
        minimum = _finite_float("min_angle_deg", self.min_angle_deg)
        maximum = _finite_float("max_angle_deg", self.max_angle_deg)
        if not (0.0 <= minimum < maximum <= 180.0):
            raise ValueError("lighting angles must satisfy 0 <= min < max <= 180")
        object.__setattr__(self, "min_angle_deg", minimum)
        object.__setattr__(self, "max_angle_deg", maximum)
        object.__setattr__(
            self,
            "sun_direction",
            _unit_vector("sun_direction", self.sun_direction),
        )
        object.__setattr__(self, "origin_m", _finite_vec3("origin_m", self.origin_m))

    @property
    def value(self) -> dict[str, Any]:
        """Return the fixed-direction lighting declaration."""
        return {
            "sun_direction": self.sun_direction,
            "min_angle_deg": self.min_angle_deg,
            "max_angle_deg": self.max_angle_deg,
            "origin_m": self.origin_m,
        }

    def apply(self, phase: Any, context: ConstraintApplicationContext) -> None:
        """Apply lower and upper angle bounds to a fixed light direction."""
        position, indices = _relative_position_expression(context)
        offset = position - np.asarray(self.origin_m, dtype=float)
        _apply_angle_bounds(
            phase,
            context.vector_functions,
            self.where,
            offset,
            np.asarray(self.sun_direction, dtype=float),
            self.min_angle_deg,
            self.max_angle_deg,
            indices,
        )

    def report(self, context: ConstraintReportContext) -> list[dict[str, Any]]:
        """Report solved extrema of the fixed-direction lighting angle."""
        positions = _positions_at_location(context.phase_trajectory, self.where)
        angles = _angles_deg(
            positions - self.origin_m,
            np.asarray(self.sun_direction, dtype=float),
        )
        return _angle_report_rows(self, context, angles, "lighting")


@dataclass(frozen=True, slots=True)
class SolarPhaseAngle(Constraint):
    """Bound the relative position angle to the ephemeris Sun direction.

    Unlike :class:`LightingAngle`, the direction is not stored as a fixed
    vector.  The composable relative-motion compiler samples the bundled SPICE
    BSP at ``Mission.initial_epoch`` and rotates the Sun line into the chief's
    RIC frame throughout the phase.  CWH dynamics must therefore provide a
    chief initial inertial state.
    """

    kind: ClassVar[str] = "solar_phase_angle"
    family: ClassVar[str] = "relative_geometry"

    min_angle_deg: float = 0.0
    max_angle_deg: float = 180.0
    origin_m: Sequence[float] = (0.0, 0.0, 0.0)
    where: Where = "Path"

    def __post_init__(self) -> None:
        object.__setattr__(self, "where", _normalize_where(self.where))
        minimum = _finite_float("min_angle_deg", self.min_angle_deg)
        maximum = _finite_float("max_angle_deg", self.max_angle_deg)
        if not (0.0 <= minimum < maximum <= 180.0):
            raise ValueError("solar phase angles must satisfy 0 <= min < max <= 180")
        object.__setattr__(self, "min_angle_deg", minimum)
        object.__setattr__(self, "max_angle_deg", maximum)
        object.__setattr__(self, "origin_m", _finite_vec3("origin_m", self.origin_m))

    @property
    def value(self) -> dict[str, Any]:
        """Return solar-phase angle bounds and origin."""
        return {
            "min_angle_deg": self.min_angle_deg,
            "max_angle_deg": self.max_angle_deg,
            "origin_m": self.origin_m,
        }

    def apply(self, phase: Any, context: ConstraintApplicationContext) -> None:
        """Apply angle bounds to the time-varying SPICE Sun line in RIC."""
        vf = context.vector_functions
        expressions = context.relative_expressions
        if expressions is not None:
            if context.solar_position_table is None:
                raise ValueError(
                    "SolarPhaseAngle requires a SPICE-derived ECI Sun position table"
                )
            offset_ric = expressions.position - np.asarray(self.origin_m, dtype=float)
            sun_line_eci = (
                context.solar_position_table(expressions.time) - expressions.chief_position
            )
            sun_line_ric = vf.stack(
                [
                    expressions.radial_axis.dot(sun_line_eci),
                    expressions.in_track_axis.dot(sun_line_eci),
                    expressions.cross_track_axis.dot(sun_line_eci),
                ]
            )
            _apply_angle_bounds(
                phase,
                vf,
                self.where,
                offset_ric,
                sun_line_ric,
                self.min_angle_deg,
                self.max_angle_deg,
                expressions.argument_indices,
            )
            return

        if context.layout.name.endswith("relative_elements"):
            raise ValueError(
                "Cartesian relative-geometry constraints require a native RIC or "
                "coupled-ECI propagation mode; they are not defined directly on "
                "relative elements."
            )
        if context.solar_direction_table is None:
            raise ValueError("SolarPhaseAngle requires a SPICE-derived RIC direction table")
        position_time = vf.Arguments(4)
        offset = position_time.head(3) - np.asarray(self.origin_m, dtype=float)
        sun_direction = context.solar_direction_table(position_time[3])
        _apply_angle_bounds(
            phase,
            vf,
            self.where,
            offset,
            sun_direction,
            self.min_angle_deg,
            self.max_angle_deg,
            (*context.layout.state_indices("position"), context.layout.time_column),
            scale_direction_norm=False,
        )

    def report(self, context: ConstraintReportContext) -> list[dict[str, Any]]:
        """Report solved extrema of the time-varying solar phase angle."""
        if context.solar_direction_at is None:
            raise ValueError("SolarPhaseAngle reporting requires SPICE-derived RIC directions")
        trajectory = np.asarray(context.phase_trajectory, dtype=float)
        positions = _positions_at_location(trajectory, self.where)
        times = _values_at_location(trajectory[:, 6], self.where)
        directions = np.asarray(context.solar_direction_at(times), dtype=float).reshape(-1, 3)
        angles = _angles_between_deg(positions - self.origin_m, directions)
        return _angle_report_rows(self, context, angles, "solar_phase")


@dataclass(frozen=True, slots=True)
class State(Constraint):
    """Constrain a Cartesian boundary state."""

    kind: ClassVar[str] = "state"
    family: ClassVar[str] = "boundary_state"

    x: BoundaryState = BoundaryState(np.zeros(3), np.zeros(3))
    where: Where = "Front"
    groups: tuple[str, ...] = ("R", "V")

    def __post_init__(self) -> None:
        object.__setattr__(self, "where", _normalize_where(self.where))

    @property
    def value(self) -> dict[str, Any]:
        """Return the Cartesian state and selected groups."""
        return {"x": self.x, "groups": tuple(str(group) for group in self.groups)}

    def apply(self, phase: Any, context: ConstraintApplicationContext) -> None:
        """Fix the selected position, velocity, or time boundary groups."""
        if context.layout.name.endswith("relative_elements"):
            raise ValueError(
                "Cartesian State constraints are not native to a relative-element phase. "
                "Use constraints.relative_orbital_elements(...) or "
                "constraints.relative_orbital_element(...)."
            )

        groups = tuple(str(group) for group in self.groups)
        declared_phase = context.declared_phase
        previous = getattr(declared_phase, "previous", None)
        link = getattr(declared_phase, "link", None)
        linked_front_impulse = (
            self.where == "Front"
            and previous is not None
            and link is not None
            and not link.is_continuous()
        )
        has_impulse = any(
            getattr(variable, "kind", "") == "impulsive_delta_v"
            and getattr(variable, "where", "") == self.where
            for variable in getattr(declared_phase, "variables", ()) or ()
        ) or any(
            getattr(event, "kind", "") == "impulse"
            and getattr(event, "where", "") == self.where
            for event in getattr(declared_phase, "events", ()) or ()
        )
        if has_impulse and "V" in groups and not linked_front_impulse:
            groups = tuple(group for group in groups if group != "V")

        expressions = context.relative_expressions
        if expressions is not None:
            for group in groups:
                if group == "R":
                    residual = expressions.position - np.asarray(self.x.r_m, dtype=float)
                elif group == "V":
                    residual = expressions.velocity - np.asarray(self.x.v_mps, dtype=float)
                elif group in {"t", "time"}:
                    continue
                else:
                    raise ValueError(f"Unsupported State.groups element: {group!r}")
                phase.addEqualCon(self.where, residual, expressions.argument_indices)
            return

        values: list[float] = []
        applied_groups: list[str] = []
        for group in groups:
            if group == "R":
                values.extend(np.asarray(self.x.r_m, dtype=float).reshape(3))
                applied_groups.append("R")
            elif group == "V":
                values.extend(np.asarray(self.x.v_mps, dtype=float).reshape(3))
                applied_groups.append("V")
            elif group in {"t", "time"}:
                if context.phase_index == 0 and self.where == "Front":
                    applied_groups.append("t")
                    values.append(0.0)
            else:
                raise ValueError(f"Unsupported State.groups element: {group!r}")
        if applied_groups:
            phase.addBoundaryValue(
                self.where,
                applied_groups,
                np.asarray(values, dtype=float),
            )


@dataclass(frozen=True, slots=True)
class Position(Constraint):
    """Constrain a Cartesian boundary position."""

    kind: ClassVar[str] = "position"
    family: ClassVar[str] = "boundary_state"

    r_m: Sequence[float] = (0.0, 0.0, 0.0)
    where: Where = "Front"

    def __post_init__(self) -> None:
        object.__setattr__(self, "where", _normalize_where(self.where))

    @property
    def value(self) -> np.ndarray:
        """Return the constrained position as a three-vector."""
        return np.asarray(self.r_m, dtype=float).reshape(3)

    def apply(self, phase: Any, context: ConstraintApplicationContext) -> None:
        """Fix Cartesian position at the selected phase location."""
        if context.layout.name.endswith("relative_elements"):
            raise ValueError(
                "Cartesian Position constraints are not native to a relative-element phase. "
                "Use constraints.relative_orbital_element(...) instead."
            )
        position = np.asarray(self.r_m, dtype=float).reshape(3)
        if context.relative_expressions is None:
            phase.addBoundaryValue(self.where, ["R"], position)
            return
        phase.addEqualCon(
            self.where,
            context.relative_expressions.position - position,
            context.relative_expressions.argument_indices,
        )


@dataclass(frozen=True, slots=True)
class StateComponent(Constraint):
    """Constrain one Cartesian state component in the phase's native frame.

    Position components are ``x``, ``y``, and ``z``; velocity components are
    ``vx``, ``vy``, and ``vz``. Position targets and tolerances use meters,
    while velocity targets and tolerances use meters per second. The
    constraint is applied directly to the phase state without changing frames.
    """

    kind: ClassVar[str] = "state_component"
    family: ClassVar[str] = "boundary_state"

    component: str = "x"
    target: float = 0.0
    where: Where = "Front"
    tolerance: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "component",
            _normalize_cartesian_component(self.component),
        )
        object.__setattr__(self, "target", _finite_float("target", self.target))
        object.__setattr__(self, "where", _normalize_where(self.where))
        object.__setattr__(
            self,
            "tolerance",
            _optional_nonnegative_float("tolerance", self.tolerance),
        )

    @property
    def value(self) -> dict[str, float | str | None]:
        """Return the normalized component target payload."""
        return {
            "component": self.component,
            "target": self.target,
            "tolerance": self.tolerance,
        }

    def apply(self, phase: Any, context: ConstraintApplicationContext) -> None:
        """Apply one native Cartesian state component target."""
        index = _cartesian_component_index(context.layout, self.component)
        if self.tolerance is None:
            if self.where == "Path":
                argument = context.vector_functions.Arguments(1)
                phase.addEqualCon("Path", argument - self.target, [index])
                return
            phase.addBoundaryValue(
                self.where,
                [index],
                np.asarray([self.target], dtype=float),
            )
            return
        phase.addLUVarBound(
            self.where,
            index,
            self.target - self.tolerance,
            self.target + self.tolerance,
        )


@dataclass(frozen=True, slots=True)
class PeriodicState(Constraint):
    """Require selected Cartesian components to match at both phase ends.

    The equality is imposed directly between the phase's front and back state
    in its declared frame. Time is intentionally excluded: periodicity closes
    the physical state while allowing a positive orbit period. At least one
    independent front-boundary phase condition is normally needed to remove
    the arbitrary time shift of an autonomous periodic orbit.

    Args:
        components: Cartesian components to close. The default closes the full
            position and velocity state.
    """

    kind: ClassVar[str] = "periodic_state"
    family: ClassVar[str] = "boundary_state"
    where: ClassVar[str] = "FrontAndBack"

    components: tuple[str, ...] = ("x", "y", "z", "vx", "vy", "vz")

    def __post_init__(self) -> None:
        normalized = tuple(
            _normalize_cartesian_component(component) for component in self.components
        )
        if not normalized:
            raise ValueError("PeriodicState.components must not be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("PeriodicState.components must be unique")
        object.__setattr__(self, "components", normalized)

    @property
    def value(self) -> dict[str, tuple[str, ...]]:
        """Return the components equated between the phase boundaries."""
        return {"components": self.components}

    def apply(self, phase: Any, context: ConstraintApplicationContext) -> None:
        """Equate selected native Cartesian components at both boundaries."""
        indices = [
            _cartesian_component_index(context.layout, component)
            for component in self.components
        ]
        arguments = context.vector_functions.Arguments(2 * len(indices))
        front = arguments.head(len(indices))
        back = arguments.tail(len(indices))
        phase.addEqualCon("FrontandBack", front - back, indices)


@dataclass(frozen=True, slots=True)
class JacobiConstant(Constraint):
    """Constrain the Jacobi integral of a CR3BP phase.

    The compiler evaluates the invariant directly from the phase's synodic
    Cartesian state. Canonical targets are convenient for comparison with
    CR3BP literature; dimensional targets use square meters per square second.

    Args:
        target: Desired Jacobi constant.
        where: ``"Front"``, ``"Back"``, or ``"Path"``.
        tolerance: Optional symmetric non-negative tolerance in the selected
            unit system. Omitting it creates an equality.
        dimensional: Interpret ``target`` and ``tolerance`` as ``m²/s²`` when
            true or canonical CR3BP units when false.
    """

    kind: ClassVar[str] = "jacobi_constant"
    family: ClassVar[str] = "cr3bp_invariant"

    target: float = 0.0
    where: Where = "Front"
    tolerance: float | None = None
    dimensional: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "target", _finite_float("target", self.target))
        object.__setattr__(self, "where", _normalize_where(self.where))
        object.__setattr__(
            self,
            "tolerance",
            _optional_nonnegative_float("tolerance", self.tolerance),
        )
        object.__setattr__(self, "dimensional", bool(self.dimensional))

    @property
    def value(self) -> dict[str, float | bool | None]:
        """Return the target, tolerance, and selected unit system."""
        return {
            "target": self.target,
            "tolerance": self.tolerance,
            "dimensional": self.dimensional,
        }

    def apply(self, phase: Any, context: ConstraintApplicationContext) -> None:
        """Apply the CR3BP Jacobi integral in canonical solver units."""
        system = context.cr3bp_system
        if system is None:
            raise ValueError("Jacobi-constant constraints require Dynamics.cr3bp().")
        vf = context.vector_functions
        arguments = vf.Arguments(6)
        phase_position, phase_velocity = arguments.tolist([(0, 3), (3, 3)])
        if context.cr3bp_dimensional:
            position = phase_position / float(system.separation_m)
            velocity = phase_velocity / float(system.velocity_scale_mps)
        else:
            position = phase_position
            velocity = phase_velocity
        mass_parameter = float(system.mass_parameter)
        x = position[0]
        y = position[1]
        z = position[2]
        primary_displacement = vf.stack([x + mass_parameter, y, z])
        secondary_displacement = vf.stack([x - 1.0 + mass_parameter, y, z])
        potential_twice = (
            x**2
            + y**2
            + 2.0 * (1.0 - mass_parameter) / primary_displacement.norm()
            + 2.0 * mass_parameter / secondary_displacement.norm()
        )
        jacobi = potential_twice - velocity.dot(velocity)
        unit_scale = float(system.velocity_scale_mps**2) if self.dimensional else 1.0
        target = self.target / unit_scale
        tolerance = self.tolerance / unit_scale if self.tolerance is not None else None
        if tolerance is None:
            phase.addEqualCon(
                self.where,
                vf.stack([jacobi - target]),
                range(0, 6),
            )
            return
        phase.addInequalCon(
            self.where,
            vf.stack([jacobi - (target + tolerance)]),
            range(0, 6),
        )
        phase.addInequalCon(
            self.where,
            vf.stack([(target - tolerance) - jacobi]),
            range(0, 6),
        )

    def report(self, context: ConstraintReportContext) -> list[dict[str, Any]]:
        """Report the worst solved Jacobi target error at the selected location."""
        system = context.cr3bp_system
        if system is None:
            return []
        from .cislunar import jacobi_constant as evaluate_jacobi_constant

        values = np.asarray(
            [
                evaluate_jacobi_constant(
                    row[0:6],
                    system=system,
                    dimensional=context.cr3bp_dimensional,
                )
                for row in np.asarray(context.phase_trajectory, dtype=float)
            ],
            dtype=float,
        )
        if self.dimensional and not context.cr3bp_dimensional:
            values *= float(system.velocity_scale_mps**2)
        elif not self.dimensional and context.cr3bp_dimensional:
            values /= float(system.velocity_scale_mps**2)
        selected = _values_at_location(values, self.where)
        errors = np.abs(selected - self.target)
        actual = float(selected[int(np.argmax(errors))])
        error = actual - self.target
        tolerance = float(self.tolerance or 0.0)
        numerical_tolerance = tolerance + max(
            1.0e-10,
            1.0e-7 * max(1.0, abs(self.target)),
        )
        return [
            _report_row(
                self,
                context,
                name=self.kind,
                target=self.target,
                actual=actual,
                error=error,
                tolerance=tolerance,
                satisfied=abs(error) <= numerical_tolerance,
            )
        ]


@dataclass(frozen=True, slots=True)
class RelativeStateComponent(Constraint):
    """Constrain one native RIC position or velocity component.

    Components are ``R``, ``I``, ``C``, ``Rdot``, ``Idot``, and ``Cdot``.
    The target and optional tolerance use meters for position and meters per
    second for velocity. The compiler applies this constraint directly to a
    native RIC formulation; it never reconstructs an absolute deputy state.
    """

    kind: ClassVar[str] = "relative_state_component"
    family: ClassVar[str] = "relative_state"

    component: str = "R"
    target: float = 0.0
    where: Where = "Back"
    tolerance: float | None = None

    def __post_init__(self) -> None:
        aliases = {
            "r": "R",
            "radial": "R",
            "x": "R",
            "i": "I",
            "in_track": "I",
            "intrack": "I",
            "y": "I",
            "c": "C",
            "cross_track": "C",
            "crosstrack": "C",
            "z": "C",
            "rdot": "Rdot",
            "r_dot": "Rdot",
            "xdot": "Rdot",
            "idot": "Idot",
            "i_dot": "Idot",
            "ydot": "Idot",
            "cdot": "Cdot",
            "c_dot": "Cdot",
            "zdot": "Cdot",
        }
        normalized = str(self.component).strip().lower().replace("-", "_").replace(" ", "_")
        if normalized not in aliases:
            raise ValueError("component must be one of R, I, C, Rdot, Idot, or Cdot")
        object.__setattr__(self, "component", aliases[normalized])
        object.__setattr__(self, "target", _finite_float("target", self.target))
        object.__setattr__(self, "where", _normalize_where(self.where))
        object.__setattr__(
            self,
            "tolerance",
            _optional_nonnegative_float("tolerance", self.tolerance),
        )

    @property
    def value(self) -> dict[str, float | str | None]:
        """Return the normalized component target payload."""
        return {
            "component": self.component,
            "target": self.target,
            "tolerance": self.tolerance,
        }

    def apply(self, phase: Any, context: ConstraintApplicationContext) -> None:
        """Apply a scalar target to one native RIC component."""
        groups = {
            "R": ("position", 0),
            "I": ("position", 1),
            "C": ("position", 2),
            "Rdot": ("velocity", 0),
            "Idot": ("velocity", 1),
            "Cdot": ("velocity", 2),
        }
        group, offset = groups[self.component]
        try:
            state_index = context.layout.state_indices(group)[offset]
        except (KeyError, IndexError) as exc:
            raise ValueError(
                "RIC component constraints require CWH, 'nonlinear_ric', or "
                "'coupled_ric' propagation; the selected layout is "
                f"{context.layout.name!r}"
            ) from exc
        _apply_scalar_target(
            phase,
            context,
            where=self.where,
            state_index=state_index,
            target=self.target,
            tolerance=self.tolerance,
        )

    def report(self, context: ConstraintReportContext) -> list[dict[str, Any]]:
        """Report the worst solved error for one RIC component."""
        groups = {
            "R": ("position", 0),
            "I": ("position", 1),
            "C": ("position", 2),
            "Rdot": ("velocity", 0),
            "Idot": ("velocity", 1),
            "Cdot": ("velocity", 2),
        }
        group, offset = groups[self.component]
        index = context.layout.state_indices(group)[offset]
        trajectory = np.asarray(context.native_trajectory, dtype=float)
        return [
            _scalar_report_row(
                self,
                context,
                name=f"ric_{self.component}",
                target=self.target,
                tolerance=self.tolerance,
                values=_values_at_location(trajectory[:, index], self.where),
            )
        ]


@dataclass(frozen=True, slots=True)
class RelativeOrbitalElementConstraint(Constraint):
    """Constrain one native relative orbital element.

    D'Amico names are ``delta_a``, ``delta_lambda``, ``delta_ex``,
    ``delta_ey``, ``delta_ix``, and ``delta_iy``. Classical-difference names
    are ``delta_a_m``, ``delta_e``, ``delta_i``, ``delta_raan``,
    ``delta_argp``, and ``delta_mean_anomaly``. Angles are radians.
    """

    kind: ClassVar[str] = "relative_orbital_element"
    family: ClassVar[str] = "relative_orbital_element"

    element: str = "delta_a"
    target: float = 0.0
    representation: RelativeElementRepresentation | str = "damico"
    where: Where = "Back"
    tolerance: float | None = None

    def __post_init__(self) -> None:
        representation = _normalize_relative_element_representation(self.representation)
        element = _normalize_relative_element_name(
            self.element,
            representation,
        )
        object.__setattr__(self, "representation", representation)
        object.__setattr__(self, "element", element)
        object.__setattr__(self, "target", _finite_float("target", self.target))
        object.__setattr__(self, "where", _normalize_where(self.where))
        object.__setattr__(
            self,
            "tolerance",
            _optional_nonnegative_float("tolerance", self.tolerance),
        )

    @property
    def value(self) -> dict[str, float | str | None]:
        """Return the normalized relative-element target payload."""
        return {
            "representation": self.representation,
            "element": self.element,
            "target": self.target,
            "tolerance": self.tolerance,
        }

    def apply(self, phase: Any, context: ConstraintApplicationContext) -> None:
        """Apply one scalar native relative-orbital-element target."""
        expected_layout = (
            "damico_relative_elements"
            if self.representation == "damico"
            else "classical_relative_elements"
        )
        if context.layout.name != expected_layout:
            raise ValueError(
                f"{self.representation!r} constraints require a matching native "
                "relative-element propagation mode; selected layout is "
                f"{context.layout.name!r}"
            )
        _apply_scalar_target(
            phase,
            context,
            where=self.where,
            state_index=context.layout.state_indices(self.element)[0],
            target=self.target,
            tolerance=self.tolerance,
        )

    def report(self, context: ConstraintReportContext) -> list[dict[str, Any]]:
        """Report the worst solved error for one relative orbital element."""
        trajectory = np.asarray(context.native_trajectory, dtype=float)
        index = context.layout.state_indices(self.element)[0]
        return [
            _scalar_report_row(
                self,
                context,
                name=f"{self.representation}_{self.element}",
                target=self.target,
                tolerance=self.tolerance,
                values=_values_at_location(trajectory[:, index], self.where),
            )
        ]


@dataclass(frozen=True, slots=True)
class RelativeOrbitalElementsConstraint(Constraint):
    """Constrain all six native relative orbital elements at one location."""

    kind: ClassVar[str] = "relative_orbital_elements"
    family: ClassVar[str] = "relative_orbital_element"

    elements: Any = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    representation: RelativeElementRepresentation | str | None = None
    where: Where = "Front"

    def __post_init__(self) -> None:
        inferred_representation = (
            "classical_elements"
            if hasattr(self.elements, "delta_a_m")
            else "damico"
            if hasattr(self.elements, "delta_lambda_rad")
            else None
        )
        representation = _normalize_relative_element_representation(
            self.representation or inferred_representation or "damico"
        )
        if (
            inferred_representation is not None
            and inferred_representation != representation
        ):
            raise ValueError(
                "elements object does not match the requested "
                f"{representation!r} representation"
            )
        source = (
            self.elements.as_vector()  # type: ignore[attr-defined]
            if hasattr(self.elements, "as_vector")
            else self.elements
        )
        elements = np.asarray(source, dtype=float).reshape(6)
        if not np.all(np.isfinite(elements)):
            raise ValueError("elements must contain six finite values")
        object.__setattr__(self, "representation", representation)
        object.__setattr__(self, "elements", elements)
        object.__setattr__(self, "where", _normalize_where(self.where))

    @property
    def value(self) -> dict[str, Any]:
        """Return the normalized six-element target payload."""
        return {
            "representation": self.representation,
            "elements": np.asarray(self.elements, dtype=float),
        }

    def apply(self, phase: Any, context: ConstraintApplicationContext) -> None:
        """Fix all six native relative orbital elements at one location."""
        expected_layout = (
            "damico_relative_elements"
            if self.representation == "damico"
            else "classical_relative_elements"
        )
        if context.layout.name != expected_layout:
            raise ValueError(
                f"{self.representation!r} constraints require a matching native "
                "relative-element propagation mode; selected layout is "
                f"{context.layout.name!r}"
            )
        phase.addBoundaryValue(
            self.where,
            list(context.layout.state_indices("relative_elements")),
            np.asarray(self.elements, dtype=float),
        )

    def report(self, context: ConstraintReportContext) -> list[dict[str, Any]]:
        """Report the worst solved error for each relative orbital element."""
        trajectory = np.asarray(context.native_trajectory, dtype=float)
        indices = context.layout.state_indices("relative_elements")
        names = context.layout.state_names
        return [
            _scalar_report_row(
                self,
                context,
                name=f"{self.representation}_{names[index]}",
                target=float(target),
                tolerance=None,
                values=_values_at_location(trajectory[:, index], self.where),
            )
            for index, target in zip(indices, self.elements, strict=True)
        ]


def _values_at_location(values: np.ndarray, where: str) -> np.ndarray:
    """Select front, back, or all path values."""
    if where == "Front":
        return values[0:1]
    if where == "Back":
        return values[-1:]
    return values


def _positions_at_location(trajectory: np.ndarray, where: str) -> np.ndarray:
    """Select Cartesian positions at a constraint location."""
    positions = np.asarray(trajectory, dtype=float)[:, 0:3]
    return _values_at_location(positions, where)


def _angles_deg(vectors: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Return angles from non-zero vectors to one unit direction."""
    norms = np.linalg.norm(vectors, axis=1)
    valid = norms > 1.0e-12
    if not np.any(valid):
        return np.empty(0, dtype=float)
    cosines = (vectors[valid] @ direction) / norms[valid]
    return np.rad2deg(np.arccos(np.clip(cosines, -1.0, 1.0)))


def _angles_between_deg(vectors: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Return row-wise angles between pairs of non-zero vectors."""
    vector_norms = np.linalg.norm(vectors, axis=1)
    direction_norms = np.linalg.norm(directions, axis=1)
    valid = (vector_norms > 1.0e-12) & (direction_norms > 1.0e-12)
    if not np.any(valid):
        return np.empty(0, dtype=float)
    cosines = np.sum(vectors[valid] * directions[valid], axis=1) / (
        vector_norms[valid] * direction_norms[valid]
    )
    return np.rad2deg(np.arccos(np.clip(cosines, -1.0, 1.0)))


def _report_row(
    constraint: Constraint,
    context: ConstraintReportContext,
    *,
    name: str,
    target: float,
    actual: float,
    error: float,
    tolerance: float,
    satisfied: bool,
) -> dict[str, float | str | bool]:
    """Build one standard solved-constraint report row."""
    return {
        "phase": context.phase_name,
        "where": constraint.where,
        "family": constraint.family,
        "constraint": name,
        "target": float(target),
        "tolerance": float(tolerance),
        "actual": float(actual),
        "error": float(error),
        "satisfied": bool(satisfied),
    }


def _angle_report_rows(
    constraint: LightingAngle | SolarPhaseAngle,
    context: ConstraintReportContext,
    angles: np.ndarray,
    name_prefix: str,
) -> list[dict[str, Any]]:
    """Report minimum and maximum values for an angle-band constraint."""
    if angles.size:
        minimum_actual = float(np.min(angles))
        maximum_actual = float(np.max(angles))
    else:
        minimum_actual = maximum_actual = float("nan")
    tolerance = 1.0e-4
    return [
        _report_row(
            constraint,
            context,
            name=f"{name_prefix}_min_angle_deg",
            target=constraint.min_angle_deg,
            actual=minimum_actual,
            error=minimum_actual - constraint.min_angle_deg,
            tolerance=tolerance,
            satisfied=bool(
                np.isfinite(minimum_actual)
                and minimum_actual >= constraint.min_angle_deg - tolerance
            ),
        ),
        _report_row(
            constraint,
            context,
            name=f"{name_prefix}_max_angle_deg",
            target=constraint.max_angle_deg,
            actual=maximum_actual,
            error=maximum_actual - constraint.max_angle_deg,
            tolerance=tolerance,
            satisfied=bool(
                np.isfinite(maximum_actual)
                and maximum_actual <= constraint.max_angle_deg + tolerance
            ),
        ),
    ]


def _orbital_report_row(
    constraint: OrbitalElementConstraint,
    context: ConstraintReportContext,
    target: float,
    tolerance: float | None,
    element_key: str,
) -> dict[str, float | str | bool]:
    """Evaluate one osculating orbital element at a solved boundary."""
    from .astro.kepler import cartesian_to_classic

    trajectory = np.asarray(context.phase_trajectory, dtype=float)
    row = trajectory[0] if constraint.where == "Front" else trajectory[-1]
    actual_elements = cartesian_to_classic(
        r_m=row[0:3],
        v_mps=row[3:6],
        mu_m3ps2=context.mu_m3ps2,
    )
    actual = float(actual_elements[element_key])
    error = actual - float(target)
    declared_tolerance = float(tolerance) if tolerance is not None else 1.0e-6
    numerical_tolerance = declared_tolerance + max(
        1.0e-9,
        1.0e-7 * max(1.0, abs(float(target))),
    )
    return _report_row(
        constraint,
        context,
        name=constraint.kind,
        target=target,
        actual=actual,
        error=error,
        tolerance=declared_tolerance,
        satisfied=abs(error) <= numerical_tolerance,
    )


def _scalar_report_row(
    constraint: Constraint,
    context: ConstraintReportContext,
    *,
    name: str,
    target: float,
    tolerance: float | None,
    values: np.ndarray,
) -> dict[str, float | str | bool]:
    """Report the worst solved error for one scalar target."""
    declared_tolerance = 0.0 if tolerance is None else float(tolerance)
    errors = np.asarray(values, dtype=float) - float(target)
    worst_index = int(np.argmax(np.abs(errors)))
    actual = float(np.asarray(values, dtype=float)[worst_index])
    error = float(errors[worst_index])
    numerical_tolerance = max(
        declared_tolerance,
        1.0e-9 * max(1.0, abs(float(target))),
    )
    return _report_row(
        constraint,
        context,
        name=name,
        target=target,
        actual=actual,
        error=error,
        tolerance=declared_tolerance,
        satisfied=abs(error) <= numerical_tolerance,
    )


def _normalize_relative_element_representation(
    representation: RelativeElementRepresentation | str,
) -> RelativeElementRepresentation:
    """Normalize a relative-element representation name."""
    normalized = str(representation).strip().lower().replace("-", "_")
    aliases = {
        "damico": "damico",
        "d_amico": "damico",
        "roe": "damico",
        "classical": "classical_elements",
        "classical_elements": "classical_elements",
    }
    if normalized not in aliases:
        raise ValueError("representation must be 'damico' or 'classical_elements'")
    return aliases[normalized]  # type: ignore[return-value]


def _normalize_relative_element_name(
    element: str,
    representation: RelativeElementRepresentation,
) -> str:
    """Normalize one relative-element name for its representation."""
    normalized = str(element).strip().lower().replace("δ", "delta_").replace("-", "_")
    aliases = {
        "da": "delta_a" if representation == "damico" else "delta_a_m",
        "dlambda": "delta_lambda",
        "dex": "delta_ex",
        "dey": "delta_ey",
        "dix": "delta_ix",
        "diy": "delta_iy",
        "de": "delta_e",
        "di": "delta_i",
        "draan": "delta_raan",
        "dargp": "delta_argp",
        "dm": "delta_mean_anomaly",
    }
    normalized = aliases.get(normalized, normalized)
    valid = (
        {
            "delta_a",
            "delta_lambda",
            "delta_ex",
            "delta_ey",
            "delta_ix",
            "delta_iy",
        }
        if representation == "damico"
        else {
            "delta_a_m",
            "delta_e",
            "delta_i",
            "delta_raan",
            "delta_argp",
            "delta_mean_anomaly",
        }
    )
    if normalized not in valid:
        choices = ", ".join(sorted(valid))
        raise ValueError(f"Unknown {representation} element {element!r}; choose one of {choices}")
    return normalized


def state(x: BoundaryState, where: str = "Front", groups: Sequence[str] = ("R", "V")) -> State:
    """Create a boundary state constraint."""
    return State(x=x, where=where, groups=tuple(str(group) for group in groups))


def position(r_m: Sequence[float], where: str = "Front") -> Position:
    """Create a boundary position constraint."""
    return Position(r_m=r_m, where=where)


def state_component(
    component: str,
    target: float,
    *,
    where: str = "Front",
    tolerance: float | None = None,
) -> StateComponent:
    """Create a native-frame Cartesian component constraint.

    Args:
        component: Position component ``x``, ``y``, or ``z``; or velocity
            component ``vx``, ``vy``, or ``vz``. Common spellings such as
            ``xdot`` are normalized.
        target: Component target in meters or meters per second.
        where: ``"Front"``, ``"Back"``, or ``"Path"``.
        tolerance: Optional symmetric non-negative tolerance in the
            component's units. Omitting it creates an equality.

    Returns:
        A declarative constraint applied without a coordinate conversion.
    """
    return StateComponent(
        component=component,
        target=target,
        where=where,
        tolerance=tolerance,
    )


def periodic_state(
    components: Sequence[str] = ("x", "y", "z", "vx", "vy", "vz"),
) -> PeriodicState:
    """Create a direct front-to-back Cartesian periodicity constraint.

    Args:
        components: Cartesian components to equate between the phase
            boundaries. The default closes all six position/velocity
            components while leaving time unconstrained.

    Returns:
        A periodic-state declaration for the composable ASSET compiler.

    Notes:
        Autonomous periodic-orbit problems normally need an additional phase
        condition, such as a fixed front position component, to remove the
        arbitrary time shift around the orbit.
    """
    return PeriodicState(components=tuple(components))


def jacobi_constant(
    target: float,
    *,
    where: str = "Front",
    tolerance: float | None = None,
    dimensional: bool = True,
) -> JacobiConstant:
    """Create a direct Jacobi-constant constraint for a CR3BP phase.

    Args:
        target: Desired Jacobi constant.
        where: ``"Front"``, ``"Back"``, or ``"Path"``.
        tolerance: Optional symmetric non-negative tolerance.
        dimensional: Interpret the target and tolerance as ``m²/s²`` when
            true. Set false for the canonical values normally published with
            nondimensional CR3BP states.

    Returns:
        A declarative CR3BP invariant constraint.
    """
    return JacobiConstant(
        target=target,
        where=where,
        tolerance=tolerance,
        dimensional=dimensional,
    )


def ric_state(
    component: str,
    target: float,
    *,
    where: str = "Back",
    tolerance: float | None = None,
) -> RelativeStateComponent:
    """Create a direct constraint on one native RIC state component.

    Use a native RIC propagation mode (CWH, ``"nonlinear_ric"``, or
    ``"coupled_ric"``) so the optimizer applies the target without converting
    through absolute coordinates.
    """
    return RelativeStateComponent(
        component=component,
        target=target,
        where=where,
        tolerance=tolerance,
    )


def relative_orbital_element(
    element: str,
    target: float,
    *,
    representation: RelativeElementRepresentation | str = "damico",
    where: str = "Back",
    tolerance: float | None = None,
) -> RelativeOrbitalElementConstraint:
    """Create a direct scalar relative-orbital-element constraint."""
    return RelativeOrbitalElementConstraint(
        element=element,
        target=target,
        representation=representation,
        where=where,
        tolerance=tolerance,
    )


def relative_orbital_elements(
    elements: Any,
    *,
    representation: RelativeElementRepresentation | str | None = None,
    where: str = "Front",
) -> RelativeOrbitalElementsConstraint:
    """Create a direct six-element relative-orbit boundary constraint.

    ``elements`` may be a six-value sequence or one of the relative-element
    dataclasses from :mod:`octavian.relative`.
    """
    return RelativeOrbitalElementsConstraint(
        elements=elements,
        representation=representation,
        where=where,
    )


def semi_major_axis(a_m: float, where: str = "Path", tol_m: float | None = None) -> SemiMajorAxis:
    """Create a semi-major-axis constraint."""
    return SemiMajorAxis(a_m=a_m, where=where, tol_m=tol_m)


def eccentricity(e: float, where: str = "Path", tol: float | None = None) -> Eccentricity:
    """Create an eccentricity constraint."""
    return Eccentricity(e=e, where=where, tol=tol)


def inclination_deg(
    inc_deg: float, where: str = "Path", tol_deg: float | None = None
) -> InclinationDeg:
    """Create an inclination constraint."""
    return InclinationDeg(inc_deg=inc_deg, where=where, tol_deg=tol_deg)


def min_radius(r_min_m: float, where: str = "Path") -> MinRadius:
    """Create a minimum-radius constraint."""
    return MinRadius(r_min_m=r_min_m, where=where)


def keep_out_sphere(
    radius_m: float,
    *,
    center_m: Sequence[float] = (0.0, 0.0, 0.0),
    where: str = "Path",
) -> KeepOutSphere:
    """Create an offset spherical keep-out-zone constraint."""
    return KeepOutSphere(radius_m=radius_m, center_m=center_m, where=where)


def approach_cone(
    axis: Sequence[float],
    half_angle_deg: float,
    *,
    vertex_m: Sequence[float] = (0.0, 0.0, 0.0),
    where: str = "Path",
) -> ApproachCone:
    """Create a one-sided conical approach-corridor constraint."""
    return ApproachCone(
        axis=axis,
        half_angle_deg=half_angle_deg,
        vertex_m=vertex_m,
        where=where,
    )


def lighting_angle(
    sun_direction: Sequence[float],
    *,
    min_angle_deg: float = 0.0,
    max_angle_deg: float = 180.0,
    origin_m: Sequence[float] = (0.0, 0.0, 0.0),
    where: str = "Path",
) -> LightingAngle:
    """Create a fixed-direction lighting-angle constraint."""
    return LightingAngle(
        sun_direction=sun_direction,
        min_angle_deg=min_angle_deg,
        max_angle_deg=max_angle_deg,
        origin_m=origin_m,
        where=where,
    )


def solar_phase_angle(
    *,
    min_angle_deg: float = 0.0,
    max_angle_deg: float = 180.0,
    origin_m: Sequence[float] = (0.0, 0.0, 0.0),
    where: str = "Path",
) -> SolarPhaseAngle:
    """Create an ephemeris-driven RIC solar-phase-angle constraint."""
    return SolarPhaseAngle(
        min_angle_deg=min_angle_deg,
        max_angle_deg=max_angle_deg,
        origin_m=origin_m,
        where=where,
    )
