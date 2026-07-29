"""Relative orbital-element types, conversions, and two-body propagation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from inspect import signature
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..astro.kepler import cartesian_to_classic, classical_to_cartesian
from ..bodies import EARTH, CelestialBody
from ..bodies import resolve as resolve_body
from ..data.ephemeris import DEFAULT_EPHEMERIS_BSP
from ..models import Perturbations
from ..specs import BoundaryState
from .transforms import inertial_to_relative_state, relative_to_inertial_state

if TYPE_CHECKING:
    from ..spacecraft import Spacecraft


@dataclass(frozen=True, slots=True)
class RelativeOrbitalElements:
    """D'Amico-style quasi-nonsingular relative orbital elements.

    The dimensionless/angular vector is ``[δa, δλ, δex, δey, δix, δiy]``:

    - ``δa = (a_d - a_c) / a_c``;
    - ``δλ = (u_d - u_c) + (Ω_d - Ω_c) cos(i_c)``;
    - ``δex, δey`` are differences of eccentricity-vector components;
    - ``δix = i_d - i_c``;
    - ``δiy = (Ω_d - Ω_c) sin(i_c)``.

    Angles are radians and wrapped to ``[-π, π)`` when constructed from
    Cartesian states.
    """

    delta_a: float
    delta_lambda_rad: float
    delta_ex: float
    delta_ey: float
    delta_ix_rad: float
    delta_iy_rad: float

    def __post_init__(self) -> None:
        values = np.asarray(self.as_vector(), dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("Relative orbital elements must be finite")

    def as_vector(self) -> NDArray[np.float64]:
        """Return ``[δa, δλ, δex, δey, δix, δiy]``."""
        return np.asarray(
            [
                self.delta_a,
                self.delta_lambda_rad,
                self.delta_ex,
                self.delta_ey,
                self.delta_ix_rad,
                self.delta_iy_rad,
            ],
            dtype=float,
        )

    @classmethod
    def from_vector(cls, values: ArrayLike) -> RelativeOrbitalElements:
        """Construct relative orbital elements from a six-value vector."""
        return cls(*np.asarray(values, dtype=float).reshape(6))


@dataclass(frozen=True, slots=True)
class ClassicalRelativeOrbitalElements:
    """Differences between deputy and chief classical orbital elements.

    The vector is ``[Δa, Δe, Δi, ΔΩ, Δω, ΔM]``. ``Δa`` is meters and all
    angular differences are radians. This representation is intuitive but
    inherits the circular/equatorial singularities of classical elements; use
    :class:`RelativeOrbitalElements` for D'Amico's quasi-nonsingular form.
    """

    delta_a_m: float
    delta_e: float
    delta_i_rad: float
    delta_raan_rad: float
    delta_argp_rad: float
    delta_mean_anomaly_rad: float

    def __post_init__(self) -> None:
        values = np.asarray(self.as_vector(), dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("Classical relative orbital elements must be finite")

    def as_vector(self) -> NDArray[np.float64]:
        """Return ``[Δa, Δe, Δi, ΔΩ, Δω, ΔM]`` in meters/radians."""
        return np.asarray(
            [
                self.delta_a_m,
                self.delta_e,
                self.delta_i_rad,
                self.delta_raan_rad,
                self.delta_argp_rad,
                self.delta_mean_anomaly_rad,
            ],
            dtype=float,
        )

    @classmethod
    def from_vector(cls, values: ArrayLike) -> ClassicalRelativeOrbitalElements:
        """Construct classical relative elements from a six-value vector."""
        return cls(*np.asarray(values, dtype=float).reshape(6))


RelativeElements = RelativeOrbitalElements | ClassicalRelativeOrbitalElements


@dataclass(frozen=True, slots=True)
class RelativeElementPropagationResult:
    """Paired osculating-element and RIC histories from one propagation.

    Attributes:
        elements: ``(N, 7)`` native relative-element and elapsed-time rows.
        ric: ``(N, 7)`` equivalent RIC Cartesian state and elapsed-time rows.
        representation: ``"damico"`` or ``"classical_elements"``.
    """

    elements: NDArray[np.float64]
    ric: NDArray[np.float64]
    representation: str

    def __post_init__(self) -> None:
        elements = np.asarray(self.elements, dtype=float)
        ric = np.asarray(self.ric, dtype=float)
        representation = _normalize_relative_representation(self.representation)
        if elements.ndim != 2 or elements.shape[1] != 7:
            raise ValueError("elements must have shape (N, 7)")
        if ric.shape != elements.shape:
            raise ValueError("ric must have the same (N, 7) shape as elements")
        if not np.all(np.isfinite(elements)) or not np.all(np.isfinite(ric)):
            raise ValueError("Propagation histories must contain finite values")
        if not np.allclose(elements[:, 6], ric[:, 6], rtol=0.0, atol=1.0e-12):
            raise ValueError("Element and RIC histories must use the same times")
        object.__setattr__(self, "elements", elements)
        object.__setattr__(self, "ric", ric)
        object.__setattr__(self, "representation", representation)

    @property
    def times_s(self) -> NDArray[np.float64]:
        """Return elapsed output times without copying the history."""
        return self.elements[:, 6]


def absolute_to_relative_orbital_elements(
    chief: BoundaryState,
    deputy: BoundaryState,
    *,
    mu_m3ps2: float,
) -> RelativeOrbitalElements:
    """Convert chief/deputy absolute Cartesian states to relative elements."""
    chief_elements = _cartesian_elements(chief, mu_m3ps2)
    deputy_elements = _cartesian_elements(deputy, mu_m3ps2)

    chief_eccentricity = chief_elements["e"]
    deputy_eccentricity = deputy_elements["e"]
    chief_argument = chief_elements["argp_rad"]
    deputy_argument = deputy_elements["argp_rad"]
    delta_raan = _wrap_angle(deputy_elements["raan_rad"] - chief_elements["raan_rad"])
    chief_mean_argument_latitude = _wrap_angle(chief_elements["mean_anomaly_rad"] + chief_argument)
    deputy_mean_argument_latitude = _wrap_angle(
        deputy_elements["mean_anomaly_rad"] + deputy_argument
    )

    return RelativeOrbitalElements(
        delta_a=(deputy_elements["a_m"] - chief_elements["a_m"]) / chief_elements["a_m"],
        delta_lambda_rad=_wrap_angle(
            deputy_mean_argument_latitude
            - chief_mean_argument_latitude
            + delta_raan * np.cos(chief_elements["inc_rad"])
        ),
        delta_ex=deputy_eccentricity * np.cos(deputy_argument)
        - chief_eccentricity * np.cos(chief_argument),
        delta_ey=deputy_eccentricity * np.sin(deputy_argument)
        - chief_eccentricity * np.sin(chief_argument),
        delta_ix_rad=_wrap_angle(deputy_elements["inc_rad"] - chief_elements["inc_rad"]),
        delta_iy_rad=delta_raan * np.sin(chief_elements["inc_rad"]),
    )


def absolute_to_classical_relative_orbital_elements(
    chief: BoundaryState,
    deputy: BoundaryState,
    *,
    mu_m3ps2: float,
) -> ClassicalRelativeOrbitalElements:
    """Convert two absolute Cartesian states to classical element differences."""
    chief_elements = _cartesian_elements(chief, mu_m3ps2)
    deputy_elements = _cartesian_elements(deputy, mu_m3ps2)
    return ClassicalRelativeOrbitalElements(
        delta_a_m=deputy_elements["a_m"] - chief_elements["a_m"],
        delta_e=deputy_elements["e"] - chief_elements["e"],
        delta_i_rad=_wrap_angle(deputy_elements["inc_rad"] - chief_elements["inc_rad"]),
        delta_raan_rad=_wrap_angle(deputy_elements["raan_rad"] - chief_elements["raan_rad"]),
        delta_argp_rad=_wrap_angle(deputy_elements["argp_rad"] - chief_elements["argp_rad"]),
        delta_mean_anomaly_rad=_wrap_angle(
            deputy_elements["mean_anomaly_rad"] - chief_elements["mean_anomaly_rad"]
        ),
    )


def relative_orbital_elements_to_absolute_state(
    chief: BoundaryState,
    relative_elements: RelativeOrbitalElements | ArrayLike,
    *,
    mu_m3ps2: float,
) -> BoundaryState:
    """Reconstruct a deputy absolute state from chief state and ROE.

    Equatorial chief orbits cannot encode right-ascension separation through
    ``δiy``.  The conversion accepts them only when ``δiy`` is effectively
    zero; use Cartesian RIC states when the missing node geometry matters.
    """
    roe = (
        relative_elements
        if isinstance(relative_elements, RelativeOrbitalElements)
        else RelativeOrbitalElements.from_vector(relative_elements)
    )
    chief_elements = _cartesian_elements(chief, mu_m3ps2)
    chief_inclination = chief_elements["inc_rad"]
    sine_inclination = float(np.sin(chief_inclination))
    if abs(sine_inclination) <= 1.0e-10:
        if abs(roe.delta_iy_rad) > 1.0e-10:
            raise ValueError("delta_iy_rad is singular for an equatorial chief orbit")
        delta_raan = 0.0
    else:
        delta_raan = roe.delta_iy_rad / sine_inclination

    deputy_semi_major_axis = chief_elements["a_m"] * (1.0 + roe.delta_a)
    if deputy_semi_major_axis <= 0.0:
        raise ValueError("Relative elements produce a non-positive deputy semi-major axis")
    deputy_ex = chief_elements["e"] * np.cos(chief_elements["argp_rad"]) + roe.delta_ex
    deputy_ey = chief_elements["e"] * np.sin(chief_elements["argp_rad"]) + roe.delta_ey
    deputy_eccentricity = float(np.hypot(deputy_ex, deputy_ey))
    if not (0.0 <= deputy_eccentricity < 1.0):
        raise ValueError("Relative elements currently require an elliptic deputy orbit")
    deputy_argument = (
        float(np.arctan2(deputy_ey, deputy_ex)) if deputy_eccentricity > 1.0e-14 else 0.0
    )
    deputy_inclination = chief_inclination + roe.delta_ix_rad
    if not (0.0 <= deputy_inclination <= np.pi):
        raise ValueError("Relative elements produce deputy inclination outside [0, pi]")
    deputy_raan = _wrap_angle(chief_elements["raan_rad"] + delta_raan)
    chief_mean_argument_latitude = _wrap_angle(
        chief_elements["mean_anomaly_rad"] + chief_elements["argp_rad"]
    )
    deputy_mean_argument_latitude = (
        chief_mean_argument_latitude + roe.delta_lambda_rad - delta_raan * np.cos(chief_inclination)
    )
    deputy_mean_anomaly = _wrap_angle(deputy_mean_argument_latitude - deputy_argument)
    deputy_true_anomaly = _mean_to_true_anomaly(
        deputy_mean_anomaly,
        deputy_eccentricity,
    )
    position, velocity = classical_to_cartesian(
        a_m=deputy_semi_major_axis,
        e=deputy_eccentricity,
        inc_deg=float(np.rad2deg(deputy_inclination)),
        raan_deg=float(np.rad2deg(deputy_raan)),
        argp_deg=float(np.rad2deg(deputy_argument)),
        true_anomaly_deg=float(np.rad2deg(deputy_true_anomaly)),
        mu_m3ps2=float(mu_m3ps2),
    )
    return BoundaryState(position, velocity)


def classical_relative_orbital_elements_to_absolute_state(
    chief: BoundaryState,
    relative_elements: ClassicalRelativeOrbitalElements | ArrayLike,
    *,
    mu_m3ps2: float,
) -> BoundaryState:
    """Reconstruct the deputy from chief state and classical differences."""
    elements = (
        relative_elements
        if isinstance(relative_elements, ClassicalRelativeOrbitalElements)
        else ClassicalRelativeOrbitalElements.from_vector(relative_elements)
    )
    chief_elements = _cartesian_elements(chief, mu_m3ps2)
    deputy_a_m = chief_elements["a_m"] + elements.delta_a_m
    deputy_e = chief_elements["e"] + elements.delta_e
    deputy_i = chief_elements["inc_rad"] + elements.delta_i_rad
    if deputy_a_m <= 0.0:
        raise ValueError("Classical relative elements produce a non-positive semi-major axis")
    if not (0.0 <= deputy_e < 1.0):
        raise ValueError("Classical relative elements currently require an elliptic deputy orbit")
    if not (0.0 <= deputy_i <= np.pi):
        raise ValueError("Classical relative elements produce inclination outside [0, pi]")
    deputy_mean_anomaly = chief_elements["mean_anomaly_rad"] + elements.delta_mean_anomaly_rad
    deputy_true_anomaly = _mean_to_true_anomaly(
        deputy_mean_anomaly,
        deputy_e,
    )
    position, velocity = classical_to_cartesian(
        a_m=deputy_a_m,
        e=deputy_e,
        inc_deg=float(np.rad2deg(deputy_i)),
        raan_deg=float(
            np.rad2deg(_wrap_angle(chief_elements["raan_rad"] + elements.delta_raan_rad))
        ),
        argp_deg=float(
            np.rad2deg(_wrap_angle(chief_elements["argp_rad"] + elements.delta_argp_rad))
        ),
        true_anomaly_deg=float(np.rad2deg(deputy_true_anomaly)),
        mu_m3ps2=float(mu_m3ps2),
    )
    return BoundaryState(position, velocity)


def relative_state_to_relative_orbital_elements(
    chief: BoundaryState,
    relative_state_ric: BoundaryState,
    *,
    mu_m3ps2: float,
) -> RelativeOrbitalElements:
    """Convert a deputy RIC state directly to D'Amico relative elements."""
    deputy = relative_to_inertial_state(chief, relative_state_ric)
    return absolute_to_relative_orbital_elements(
        chief,
        deputy,
        mu_m3ps2=mu_m3ps2,
    )


def relative_orbital_elements_to_relative_state(
    chief: BoundaryState,
    relative_elements: RelativeOrbitalElements | ArrayLike,
    *,
    mu_m3ps2: float,
) -> BoundaryState:
    """Convert D'Amico relative elements directly to a deputy RIC state."""
    deputy = relative_orbital_elements_to_absolute_state(
        chief,
        relative_elements,
        mu_m3ps2=mu_m3ps2,
    )
    return inertial_to_relative_state(chief, deputy)


def relative_state_to_classical_relative_orbital_elements(
    chief: BoundaryState,
    relative_state_ric: BoundaryState,
    *,
    mu_m3ps2: float,
) -> ClassicalRelativeOrbitalElements:
    """Convert a deputy RIC state to classical orbital-element differences."""
    deputy = relative_to_inertial_state(chief, relative_state_ric)
    return absolute_to_classical_relative_orbital_elements(
        chief,
        deputy,
        mu_m3ps2=mu_m3ps2,
    )


def classical_relative_orbital_elements_to_relative_state(
    chief: BoundaryState,
    relative_elements: ClassicalRelativeOrbitalElements | ArrayLike,
    *,
    mu_m3ps2: float,
) -> BoundaryState:
    """Convert classical element differences directly to a deputy RIC state."""
    deputy = classical_relative_orbital_elements_to_absolute_state(
        chief,
        relative_elements,
        mu_m3ps2=mu_m3ps2,
    )
    return inertial_to_relative_state(chief, deputy)


def damico_to_classical_relative_orbital_elements(
    chief: BoundaryState,
    relative_elements: RelativeOrbitalElements | ArrayLike,
    *,
    mu_m3ps2: float,
) -> ClassicalRelativeOrbitalElements:
    """Convert D'Amico elements to classical differences at one epoch."""
    deputy = relative_orbital_elements_to_absolute_state(
        chief,
        relative_elements,
        mu_m3ps2=mu_m3ps2,
    )
    return absolute_to_classical_relative_orbital_elements(
        chief,
        deputy,
        mu_m3ps2=mu_m3ps2,
    )


def classical_to_damico_relative_orbital_elements(
    chief: BoundaryState,
    relative_elements: ClassicalRelativeOrbitalElements | ArrayLike,
    *,
    mu_m3ps2: float,
) -> RelativeOrbitalElements:
    """Convert classical element differences to D'Amico elements at one epoch."""
    deputy = classical_relative_orbital_elements_to_absolute_state(
        chief,
        relative_elements,
        mu_m3ps2=mu_m3ps2,
    )
    return absolute_to_relative_orbital_elements(
        chief,
        deputy,
        mu_m3ps2=mu_m3ps2,
    )


def propagate_relative_element_history(
    initial_elements: RelativeElements | ArrayLike,
    times_s: ArrayLike,
    *,
    chief_initial_state_eci: BoundaryState,
    mu_m3ps2: float,
    representation: str = "damico",
    central_body: CelestialBody | str = EARTH,
    perturbations: Perturbations | None = None,
    initial_epoch: str | datetime | float | int | None = None,
    max_step_s: float = 10.0,
    ephemeris_step_s: float = 600.0,
    bsp_path: str | Path = DEFAULT_EPHEMERIS_BSP,
    chief_spacecraft: Spacecraft | None = None,
    deputy_spacecraft: Spacecraft | None = None,
) -> RelativeElementPropagationResult:
    """Propagate relative elements and return both native and RIC histories.

    This is the consolidated analysis entry point for relative orbital
    elements. Perturbed propagation advances the coupled chief/deputy states
    only once, then derives both output representations from that history.
    The older representation-specific functions remain available for scripts
    that need only one array.

    Args:
        initial_elements: Initial D'Amico or classical relative elements.
        times_s: Requested elapsed output times.
        chief_initial_state_eci: Chief Cartesian state at elapsed time zero.
        mu_m3ps2: Central-body gravitational parameter.
        representation: ``"damico"`` or ``"classical_elements"``.
        central_body: Body constants used by numerical perturbations.
        perturbations: Optional differential force model.
        initial_epoch: UTC or SPICE ET at time zero. Required for Sun, Moon,
            or SRP.
        max_step_s: Maximum internal RK4 step for numerical propagation.
        ephemeris_step_s: Sun/Moon interpolation spacing.
        bsp_path: SPICE BSP containing Earth-centered Sun/Moon states.
        chief_spacecraft: Optional chief mass and cannonball properties.
        deputy_spacecraft: Deputy mass and cannonball properties, required for
            drag or SRP.

    Returns:
        Native relative elements and equivalent RIC states sharing one time
        vector.
    """
    normalized = _normalize_relative_representation(representation)
    if perturbations is None:
        elements = propagate_relative_orbital_elements(
            initial_elements,
            times_s,
            chief_initial_state_eci=chief_initial_state_eci,
            mu_m3ps2=mu_m3ps2,
            representation=normalized,
        )
        ric = propagate_relative_elements_to_ric(
            initial_elements,
            times_s,
            chief_initial_state_eci=chief_initial_state_eci,
            mu_m3ps2=mu_m3ps2,
            representation=normalized,
        )
        return RelativeElementPropagationResult(
            elements=elements,
            ric=ric,
            representation=normalized,
        )

    optional_kwargs: dict[str, object] = {}
    if chief_spacecraft is not None:
        optional_kwargs["chief_spacecraft"] = chief_spacecraft
    if deputy_spacecraft is not None:
        optional_kwargs["deputy_spacecraft"] = deputy_spacecraft
    unsupported = optional_kwargs.keys() - signature(
        _propagate_relative_elements_numerical
    ).parameters.keys()
    if unsupported:
        raise NotImplementedError(
            "This Octavian build does not provide cannonball spacecraft "
            "properties for relative-element propagation"
        )

    numerical = _propagate_relative_elements_numerical(
        initial_elements,
        times_s,
        chief_initial_state_eci=chief_initial_state_eci,
        mu_m3ps2=mu_m3ps2,
        representation=normalized,
        central_body=central_body,
        perturbations=perturbations,
        initial_epoch=initial_epoch,
        max_step_s=max_step_s,
        ephemeris_step_s=ephemeris_step_s,
        bsp_path=bsp_path,
        **optional_kwargs,
    )
    return RelativeElementPropagationResult(
        elements=_relative_elements_from_coupled_history(
            numerical,
            representation=normalized,
            mu_m3ps2=mu_m3ps2,
        ),
        ric=numerical.relative_trajectory_ric,
        representation=normalized,
    )


def propagate_relative_orbital_elements(
    initial_elements: RelativeElements | ArrayLike,
    times_s: ArrayLike,
    *,
    chief_initial_state_eci: BoundaryState,
    mu_m3ps2: float,
    representation: str = "damico",
    central_body: CelestialBody | str = EARTH,
    perturbations: Perturbations | None = None,
    initial_epoch: str | datetime | float | int | None = None,
    max_step_s: float = 10.0,
    ephemeris_step_s: float = 600.0,
    bsp_path: str | Path = DEFAULT_EPHEMERIS_BSP,
) -> NDArray[np.float64]:
    """Propagate relative elements at requested elapsed times.

    The returned rows contain six relative elements followed by elapsed time.
    Without ``perturbations``, the fast analytical two-body model keeps every
    element constant except D'Amico ``δλ`` or classical ``ΔM``. With a
    perturbation configuration, the chief and deputy are instead propagated
    as coupled absolute states and converted back to osculating relative
    elements at every requested output time.

    Args:
        initial_elements: Initial D'Amico or classical relative elements.
        times_s: Finite elapsed output times. Perturbed propagation requires
            strictly monotonic values with zero at one endpoint.
        chief_initial_state_eci: Chief Cartesian state at elapsed time zero.
        mu_m3ps2: Central-body gravitational parameter.
        representation: ``"damico"`` or ``"classical_elements"``.
        central_body: Body constants used by numerical central gravity and J2.
        perturbations: Optional differential J2, Moon, or Sun configuration.
            SRP and drag are not yet supported by the numerical propagator.
        initial_epoch: UTC or SPICE ET at time zero. Required for Sun or Moon.
        max_step_s: Maximum internal RK4 step for numerical propagation.
        ephemeris_step_s: Sun/Moon interpolation spacing.
        bsp_path: SPICE BSP containing Earth-centered Sun/Moon states.

    Returns:
        An ``(N, 7)`` array of native element states and time.
    """
    if perturbations is not None:
        numerical = _propagate_relative_elements_numerical(
            initial_elements,
            times_s,
            chief_initial_state_eci=chief_initial_state_eci,
            mu_m3ps2=mu_m3ps2,
            representation=representation,
            central_body=central_body,
            perturbations=perturbations,
            initial_epoch=initial_epoch,
            max_step_s=max_step_s,
            ephemeris_step_s=ephemeris_step_s,
            bsp_path=bsp_path,
        )
        return _relative_elements_from_coupled_history(
            numerical,
            representation=representation,
            mu_m3ps2=mu_m3ps2,
        )

    times = np.asarray(times_s, dtype=float).reshape(-1)
    if times.size == 0 or not np.all(np.isfinite(times)):
        raise ValueError("times_s must contain at least one finite value")
    normalized = _normalize_relative_representation(representation)
    if isinstance(initial_elements, RelativeOrbitalElements):
        vector = initial_elements.as_vector()
        inferred = "damico"
    elif isinstance(initial_elements, ClassicalRelativeOrbitalElements):
        vector = initial_elements.as_vector()
        inferred = "classical_elements"
    else:
        vector = np.asarray(initial_elements, dtype=float).reshape(6)
        inferred = normalized
    if inferred != normalized:
        raise ValueError(f"initial_elements uses {inferred!r}, not requested {normalized!r}")
    if not np.all(np.isfinite(vector)):
        raise ValueError("initial_elements must contain finite values")

    chief_a_m = _cartesian_elements(
        chief_initial_state_eci,
        mu_m3ps2,
    )["a_m"]
    deputy_a_m = chief_a_m * (1.0 + vector[0]) if normalized == "damico" else chief_a_m + vector[0]
    if deputy_a_m <= 0.0:
        raise ValueError("Relative elements produce a non-positive deputy orbit")
    relative_rate = np.sqrt(float(mu_m3ps2) / deputy_a_m**3) - np.sqrt(
        float(mu_m3ps2) / chief_a_m**3
    )
    result = np.repeat(vector[None, :], times.size, axis=0)
    evolving_index = 1 if normalized == "damico" else 5
    result[:, evolving_index] = vector[evolving_index] + relative_rate * times
    return np.column_stack([result, times])


def propagate_relative_elements_to_ric(
    initial_elements: RelativeElements | ArrayLike,
    times_s: ArrayLike,
    *,
    chief_initial_state_eci: BoundaryState,
    mu_m3ps2: float,
    representation: str = "damico",
    central_body: CelestialBody | str = EARTH,
    perturbations: Perturbations | None = None,
    initial_epoch: str | datetime | float | int | None = None,
    max_step_s: float = 10.0,
    ephemeris_step_s: float = 600.0,
    bsp_path: str | Path = DEFAULT_EPHEMERIS_BSP,
) -> NDArray[np.float64]:
    """Propagate relative elements and return equivalent RIC state history.

    With ``perturbations=None`` this uses analytical two-body element drift and
    supports arbitrary finite output times, including negative analysis times.
    Supplying :class:`octavian.Perturbations` selects coupled numerical
    chief/deputy propagation; those output times must be strictly monotonic
    with zero at one endpoint.

    Args:
        initial_elements: Initial D'Amico or classical relative elements.
        times_s: Requested elapsed output times.
        chief_initial_state_eci: Chief Cartesian state at elapsed time zero.
        mu_m3ps2: Central-body gravitational parameter.
        representation: ``"damico"`` or ``"classical_elements"``.
        central_body: Body constants used by numerical central gravity and J2.
        perturbations: Optional differential J2, Moon, or Sun configuration.
        initial_epoch: UTC or SPICE ET at time zero. Required for Sun or Moon.
        max_step_s: Maximum internal RK4 step for numerical propagation.
        ephemeris_step_s: Sun/Moon interpolation spacing.
        bsp_path: SPICE BSP containing Earth-centered Sun/Moon states.

    Returns:
        An ``(N, 7)`` RIC state-and-time history ready for plotting.
    """
    if perturbations is not None:
        numerical = _propagate_relative_elements_numerical(
            initial_elements,
            times_s,
            chief_initial_state_eci=chief_initial_state_eci,
            mu_m3ps2=mu_m3ps2,
            representation=representation,
            central_body=central_body,
            perturbations=perturbations,
            initial_epoch=initial_epoch,
            max_step_s=max_step_s,
            ephemeris_step_s=ephemeris_step_s,
            bsp_path=bsp_path,
        )
        return numerical.relative_trajectory_ric

    element_history = propagate_relative_orbital_elements(
        initial_elements,
        times_s,
        chief_initial_state_eci=chief_initial_state_eci,
        mu_m3ps2=mu_m3ps2,
        representation=representation,
    )
    output = np.empty((element_history.shape[0], 7), dtype=float)
    for index, row in enumerate(element_history):
        chief = propagate_two_body_state(
            chief_initial_state_eci,
            float(row[6]),
            mu_m3ps2,
        )
        relative = (
            relative_orbital_elements_to_relative_state(
                chief,
                row[0:6],
                mu_m3ps2=mu_m3ps2,
            )
            if str(representation).strip().lower().replace("-", "_") == "damico"
            else classical_relative_orbital_elements_to_relative_state(
                chief,
                row[0:6],
                mu_m3ps2=mu_m3ps2,
            )
        )
        output[index] = np.hstack([relative.r_m, relative.v_mps, row[6]])
    return output


def _propagate_relative_elements_numerical(
    initial_elements: RelativeElements | ArrayLike,
    times_s: ArrayLike,
    *,
    chief_initial_state_eci: BoundaryState,
    mu_m3ps2: float,
    representation: str,
    central_body: CelestialBody | str,
    perturbations: Perturbations,
    initial_epoch: str | datetime | float | int | None,
    max_step_s: float,
    ephemeris_step_s: float,
    bsp_path: str | Path,
):
    """Convert element initial conditions and run coupled propagation."""
    from .propagation import propagate_relative_numerical

    normalized = _normalize_relative_representation(representation)
    body = resolve_body(central_body)
    if not np.isclose(
        float(mu_m3ps2),
        float(body.mu_m3ps2),
        rtol=1.0e-12,
        atol=0.0,
    ):
        raise ValueError(
            "mu_m3ps2 must match central_body.mu_m3ps2 for perturbed "
            "relative-element propagation"
        )
    initial_deputy = (
        relative_orbital_elements_to_absolute_state(
            chief_initial_state_eci,
            initial_elements,
            mu_m3ps2=mu_m3ps2,
        )
        if normalized == "damico"
        else classical_relative_orbital_elements_to_absolute_state(
            chief_initial_state_eci,
            initial_elements,
            mu_m3ps2=mu_m3ps2,
        )
    )
    return propagate_relative_numerical(
        chief_initial_state_eci,
        None,
        times_s,
        deputy_initial_eci=initial_deputy,
        central_body=body,
        perturbations=perturbations,
        initial_epoch=initial_epoch,
        max_step_s=max_step_s,
        ephemeris_step_s=ephemeris_step_s,
        bsp_path=bsp_path,
    )


def _relative_elements_from_coupled_history(
    numerical,
    *,
    representation: str,
    mu_m3ps2: float,
) -> NDArray[np.float64]:
    """Convert one coupled absolute history to osculating relative elements."""
    normalized = _normalize_relative_representation(representation)
    output = np.empty((numerical.times_s.size, 7), dtype=float)
    for index, time_s in enumerate(numerical.times_s):
        chief = BoundaryState(
            numerical.chief_states_eci[index, 0:3],
            numerical.chief_states_eci[index, 3:6],
        )
        deputy = BoundaryState(
            numerical.deputy_states_eci[index, 0:3],
            numerical.deputy_states_eci[index, 3:6],
        )
        elements = (
            absolute_to_relative_orbital_elements(
                chief,
                deputy,
                mu_m3ps2=mu_m3ps2,
            )
            if normalized == "damico"
            else absolute_to_classical_relative_orbital_elements(
                chief,
                deputy,
                mu_m3ps2=mu_m3ps2,
            )
        )
        output[index] = np.hstack([elements.as_vector(), time_s])
    return output


def _normalize_relative_representation(representation: str) -> str:
    """Normalize and validate a relative-element representation name."""
    normalized = str(representation).strip().lower().replace("-", "_")
    if normalized not in {"damico", "classical_elements"}:
        raise ValueError("representation must be 'damico' or 'classical_elements'")
    return normalized


def propagate_two_body_state(
    initial_state: BoundaryState,
    elapsed_time_s: float,
    mu_m3ps2: float,
) -> BoundaryState:
    """Propagate an elliptic Cartesian state analytically under two-body gravity."""
    elements = _cartesian_elements(initial_state, mu_m3ps2)
    mean_motion = np.sqrt(float(mu_m3ps2) / elements["a_m"] ** 3)
    true_anomaly = _mean_to_true_anomaly(
        elements["mean_anomaly_rad"] + mean_motion * float(elapsed_time_s),
        elements["e"],
    )
    position, velocity = classical_to_cartesian(
        a_m=elements["a_m"],
        e=elements["e"],
        inc_deg=float(np.rad2deg(elements["inc_rad"])),
        raan_deg=float(np.rad2deg(elements["raan_rad"])),
        argp_deg=float(np.rad2deg(elements["argp_rad"])),
        true_anomaly_deg=float(np.rad2deg(true_anomaly)),
        mu_m3ps2=float(mu_m3ps2),
    )
    return BoundaryState(position, velocity)


def _cartesian_elements(
    state: BoundaryState,
    mu_m3ps2: float,
) -> dict[str, float]:
    elements = cartesian_to_classic(
        r_m=state.r_m,
        v_mps=state.v_mps,
        mu_m3ps2=float(mu_m3ps2),
    )
    eccentricity = float(elements["e"])
    semi_major_axis = float(elements["a_m"])
    if semi_major_axis <= 0.0 or not (0.0 <= eccentricity < 1.0):
        raise ValueError("Relative orbital elements currently require elliptic orbits")
    true_anomaly = float(np.deg2rad(elements["true_anomaly_deg"]))
    return {
        "a_m": semi_major_axis,
        "e": eccentricity,
        "inc_rad": float(np.deg2rad(elements["inc_deg"])),
        "raan_rad": float(np.deg2rad(elements["raan_deg"])),
        "argp_rad": float(np.deg2rad(elements["argp_deg"])),
        "mean_anomaly_rad": _true_to_mean_anomaly(true_anomaly, eccentricity),
    }


def _true_to_mean_anomaly(true_anomaly_rad: float, eccentricity: float) -> float:
    eccentric_anomaly = 2.0 * np.arctan2(
        np.sqrt(1.0 - eccentricity) * np.sin(0.5 * true_anomaly_rad),
        np.sqrt(1.0 + eccentricity) * np.cos(0.5 * true_anomaly_rad),
    )
    return _wrap_angle(eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly))


def _mean_to_true_anomaly(mean_anomaly_rad: float, eccentricity: float) -> float:
    eccentric_anomaly = float(mean_anomaly_rad)
    for _ in range(30):
        residual = eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly) - mean_anomaly_rad
        correction = residual / (1.0 - eccentricity * np.cos(eccentric_anomaly))
        eccentric_anomaly -= correction
        if abs(correction) <= 1.0e-14:
            break
    return _wrap_angle(
        2.0
        * np.arctan2(
            np.sqrt(1.0 + eccentricity) * np.sin(0.5 * eccentric_anomaly),
            np.sqrt(1.0 - eccentricity) * np.cos(0.5 * eccentric_anomaly),
        )
    )


def _wrap_angle(angle_rad: float) -> float:
    return float((float(angle_rad) + np.pi) % (2.0 * np.pi) - np.pi)
