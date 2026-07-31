"""User-facing constraint declarations for composable missions.

The constraint objects in this module are intentionally small, explicit, and
Pythonic. They describe *what* the user wants the mission to satisfy; solver
backends decide *how* to compile them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np

from .specs import BoundaryState

Where = Literal["Front", "Back", "Path"]
RelativeElementRepresentation = Literal["damico", "classical_elements"]

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


class Constraint(ABC):
    """Abstract base class for all mission constraints."""

    kind: ClassVar[str]
    family: ClassVar[str]
    where: Where

    @property
    @abstractmethod
    def value(self) -> Any:
        """Return the canonical constraint payload."""


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
