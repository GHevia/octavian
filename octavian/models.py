"""High-level configuration models.

These classes exist to keep user scripts readable while still allowing
advanced solving behavior (continuation / retries) via sensible defaults.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .bodies import CelestialBody
from .bodies import resolve as resolve_body
from .coordinates import EARTH_INERTIAL, CoordinateFrame, SolverScaling
from .specs import BoundaryState


class TranslationalModel(Protocol):
    """Configuration contract for non-default translational dynamics models."""

    @property
    def frame(self) -> CoordinateFrame:
        """Return the coordinate frame used by the model."""
        ...

    @property
    def scaling(self) -> SolverScaling:
        """Return the model's recommended solver scaling."""
        ...


@dataclass(frozen=True, slots=True)
class Perturbations:
    """Perturbation flags for translational dynamics.

    The composable ASSET backend supports the core Earth-orbit perturbations:
    J2, lunar third-body gravity, and solar third-body gravity. ``moon`` and
    ``sun`` are convenience flags; ``third_bodies=("moon", "sun")`` remains
    accepted for scripts that prefer a body list.
    """

    j2: bool = False
    moon: bool = False
    sun: bool = False
    srp: bool = False
    drag: bool = False
    third_bodies: tuple[str, ...] = ()

    def active_third_bodies(self) -> tuple[str, ...]:
        """Return normalized third-body names requested by this config."""
        names: list[str] = []
        if self.moon:
            names.append("moon")
        if self.sun:
            names.append("sun")
        for body in self.third_bodies:
            normalized = str(body).strip().lower().replace("-", "_")
            if normalized in {"luna", "earth_moon"}:
                normalized = "moon"
            if normalized not in names:
                names.append(normalized)
        return tuple(names)


@dataclass(slots=True)
class Dynamics:
    """Environment and dynamics configuration.

    ``model`` selects an alternate translational model such as CWH. ``frame``
    records the meaning of Cartesian states throughout compilation and
    reporting. ``scaling`` optionally overrides the solver's automatic
    characteristic units while preserving SI inputs and outputs.
    """

    mu_m3ps2: float = 3.986004418e14
    central_body_radius_m: float = 6_378_136.3
    j2_coefficient: float = 1.08262668e-3
    third_body_table_step_s: float = 3600.0
    third_body_table_margin_s: float = 86400.0
    third_bodies: tuple[str, ...] = ()
    j2: bool = False
    moon: bool = False
    sun: bool = False
    srp: bool = False
    drag: bool = False
    perturbations: Perturbations | None = None
    central_body: CelestialBody | str | None = None
    model: TranslationalModel | None = None
    frame: CoordinateFrame = EARTH_INERTIAL
    scaling: SolverScaling | None = None
    info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.model is not None:
            self.frame = self.model.frame
            if self.scaling is None:
                self.scaling = self.model.scaling
        if self.central_body is None:
            return
        body = resolve_body(self.central_body)
        self.central_body = body
        self.mu_m3ps2 = body.mu_m3ps2
        self.central_body_radius_m = body.mean_radius_m
        self.j2_coefficient = body.j2_coefficient
        if self.model is None and self.frame.origin != body.name:
            self.frame = body.inertial_frame()

    @classmethod
    def for_body(cls, body: CelestialBody | str, **kwargs: Any) -> Dynamics:
        """Create dynamics from a built-in or custom central body.

        Body constants intentionally override raw ``mu_m3ps2``, radius, and J2
        keyword values so a named body cannot silently become inconsistent.
        """
        resolved = resolve_body(body)
        return cls(central_body=resolved, **kwargs)

    @classmethod
    def cwh(
        cls,
        *,
        chief_orbit_radius_m: float,
        central_body: CelestialBody | str = "earth",
        chief_name: str = "chief",
        reference_length_m: float = 1_000.0,
        chief_initial_state_eci: BoundaryState | None = None,
        **kwargs: Any,
    ) -> Dynamics:
        """Create circular-chief Clohessy-Wiltshire relative dynamics."""
        from .relative import ClohessyWiltshire

        body = resolve_body(central_body)
        model = ClohessyWiltshire.from_circular_orbit(
            chief_orbit_radius_m,
            body=body,
            chief_name=chief_name,
            reference_length_m=reference_length_m,
            chief_initial_state_eci=chief_initial_state_eci,
        )
        dynamics = cls(central_body=body, model=model, **kwargs)
        perturbations = dynamics.active_perturbations()
        if any(
            (
                perturbations.j2,
                perturbations.srp,
                perturbations.drag,
                bool(perturbations.active_third_bodies()),
            )
        ):
            raise ValueError(
                "Dynamics.cwh is an unforced linear model. Use "
                "Dynamics.relative(...) for exact nonlinear relative motion "
                "with perturbations."
            )
        return dynamics

    @classmethod
    def relative(
        cls,
        *,
        chief_initial_state_eci: BoundaryState,
        central_body: CelestialBody | str = "earth",
        chief_name: str = "chief",
        reference_length_m: float = 1_000.0,
        propagation_mode: str = "coupled_eci",
        **kwargs: Any,
    ) -> Dynamics:
        """Create nonlinear or relative-element dynamics.

        Args:
            chief_initial_state_eci: Absolute chief state defining the RIC frame.
            central_body: Central body name or object.
            chief_name: Name used for the returned chief-centered frame.
            reference_length_m: Characteristic relative distance for scaling.
            propagation_mode: One of ``"coupled_eci"`` (default),
                ``"coupled_ric"``, ``"nonlinear_ric"``, ``"damico"``, or
                ``"classical_elements"``. Select ``"coupled_eci"`` for J2,
                third-body perturbations, or finite-thrust relative phases.
            **kwargs: Additional :class:`Dynamics` configuration.

        Returns:
            A dynamics configuration whose public reporting frame is RIC.
        """
        from .relative import NonlinearRelative

        body = resolve_body(central_body)
        model = NonlinearRelative(
            chief_initial_state_eci=chief_initial_state_eci,
            central_body=body,
            chief_name=chief_name,
            reference_length_m=reference_length_m,
            propagation_mode=propagation_mode,
        )
        return cls(central_body=body, model=model, **kwargs)

    def active_perturbations(self) -> Perturbations:
        """Return normalized perturbation flags for solver compilation."""
        if self.perturbations is not None:
            return self.perturbations
        return Perturbations(
            j2=bool(self.j2),
            moon=bool(self.moon),
            sun=bool(self.sun),
            srp=bool(self.srp),
            drag=bool(self.drag),
            third_bodies=tuple(str(body) for body in self.third_bodies),
        )


@dataclass(slots=True)
class SolveConfig:
    """Runner behavior settings."""

    max_attempts: int = 3
    raise_on_fail: bool = True
    verbose: bool = True


@dataclass(slots=True)
class Stage:
    """A single continuation stage (minimal in v0.x)."""

    name: str
    nsegs_scale: float | None = None
    tighten_bounds: bool = False


@dataclass(slots=True)
class RunPlan:
    """Continuation / crawl-walk-run plan."""

    stages: Sequence[Stage] = ()

    @staticmethod
    def default() -> RunPlan:
        """Return an empty continuation plan."""
        return RunPlan(stages=())


@dataclass(slots=True)
class RetryPolicy:
    """Retry behavior when a solve fails."""

    enabled: bool = True
    max_retries: int = 2

    @staticmethod
    def default() -> RetryPolicy:
        """Return the default enabled retry policy."""
        return RetryPolicy(enabled=True, max_retries=2)
