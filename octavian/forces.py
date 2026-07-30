"""Simple non-gravitational force models.

The models in this module are deliberately small and explicit.  They provide
useful first-order drag and solar-radiation-pressure behavior without implying
high-fidelity atmosphere, geometry, shadowing, or attitude modeling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

ASTRONOMICAL_UNIT_M = 149_597_870_700.0
SOLAR_PRESSURE_AT_1_AU_NPM2 = 4.56e-6


def _finite_nonnegative(value: float, name: str) -> float:
    """Return ``value`` as a validated finite, non-negative float."""
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _finite_positive(value: float, name: str) -> float:
    """Return ``value`` as a validated finite, positive float."""
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


@dataclass(frozen=True, slots=True)
class Cannonball:
    """Spacecraft properties for simple drag and SRP acceleration.

    Areas are projected constant areas.  They do not vary with attitude.  A
    zero area produces no corresponding acceleration. Composable phases
    require a positive area on their primary spacecraft when that force is
    enabled, while an optional relative chief may intentionally keep zero area.

    Args:
        drag_area_m2: Constant projected area used by atmospheric drag.
        drag_coefficient: Dimensionless drag coefficient.
        srp_area_m2: Constant projected area used by solar radiation pressure.
        reflectivity_coefficient: Dimensionless SRP coefficient, commonly
            called ``Cr``.
    """

    drag_area_m2: float = 0.0
    drag_coefficient: float = 2.2
    srp_area_m2: float = 0.0
    reflectivity_coefficient: float = 1.3

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "drag_area_m2",
            _finite_nonnegative(self.drag_area_m2, "drag_area_m2"),
        )
        object.__setattr__(
            self,
            "drag_coefficient",
            _finite_positive(self.drag_coefficient, "drag_coefficient"),
        )
        object.__setattr__(
            self,
            "srp_area_m2",
            _finite_nonnegative(self.srp_area_m2, "srp_area_m2"),
        )
        object.__setattr__(
            self,
            "reflectivity_coefficient",
            _finite_positive(
                self.reflectivity_coefficient,
                "reflectivity_coefficient",
            ),
        )


@dataclass(frozen=True, slots=True)
class ExponentialAtmosphere:
    """Single-scale-height co-rotating atmosphere.

    Density follows

    ``rho(h) = reference_density * exp(-(h - reference_altitude) / scale_height)``.

    The rotation rate defines the atmosphere velocity as ``omega × r`` in the
    central-body inertial frame.  This is a screening and preliminary-design
    model, not a substitute for a space-weather-driven atmosphere.

    Args:
        reference_density_kgpm3: Density at ``reference_altitude_m``.
        reference_altitude_m: Reference altitude above the central body.
        scale_height_m: Constant density scale height.
        rotation_rate_radps: Central-body atmosphere rotation rate about +Z.
    """

    reference_density_kgpm3: float
    reference_altitude_m: float
    scale_height_m: float
    rotation_rate_radps: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reference_density_kgpm3",
            _finite_positive(
                self.reference_density_kgpm3,
                "reference_density_kgpm3",
            ),
        )
        object.__setattr__(
            self,
            "reference_altitude_m",
            _finite_nonnegative(self.reference_altitude_m, "reference_altitude_m"),
        )
        object.__setattr__(
            self,
            "scale_height_m",
            _finite_positive(self.scale_height_m, "scale_height_m"),
        )
        rotation_rate = float(self.rotation_rate_radps)
        if not np.isfinite(rotation_rate):
            raise ValueError("rotation_rate_radps must be finite")
        object.__setattr__(self, "rotation_rate_radps", rotation_rate)

    def density_kgpm3(self, altitude_m: float) -> float:
        """Return atmospheric density at an altitude in meters.

        The exponent is clipped only to keep exploratory numerical propagation
        finite far outside the model's useful altitude range.
        """
        altitude = float(altitude_m)
        if not np.isfinite(altitude):
            raise ValueError("altitude_m must be finite")
        exponent = -(altitude - self.reference_altitude_m) / self.scale_height_m
        return float(self.reference_density_kgpm3 * np.exp(np.clip(exponent, -700.0, 700.0)))


EARTH_EXPONENTIAL_ATMOSPHERE = ExponentialAtmosphere(
    reference_density_kgpm3=3.614e-13,
    reference_altitude_m=700_000.0,
    scale_height_m=88_667.0,
    rotation_rate_radps=7.2921159e-5,
)


def cannonball_drag_acceleration(
    position_m: ArrayLike,
    velocity_mps: ArrayLike,
    *,
    mass_kg: float,
    central_body_radius_m: float,
    cannonball: Cannonball,
    atmosphere: ExponentialAtmosphere = EARTH_EXPONENTIAL_ATMOSPHERE,
) -> NDArray[np.float64]:
    """Return simple cannonball drag acceleration in an inertial frame.

    Args:
        position_m: Central-body inertial position.
        velocity_mps: Central-body inertial velocity.
        mass_kg: Instantaneous spacecraft mass.
        central_body_radius_m: Radius used to convert position to altitude.
        cannonball: Spacecraft projected area and drag coefficient.
        atmosphere: Exponential, co-rotating atmosphere model.

    Returns:
        Three-component inertial acceleration in m/s².
    """
    position = np.asarray(position_m, dtype=float).reshape(3)
    velocity = np.asarray(velocity_mps, dtype=float).reshape(3)
    mass = _finite_positive(mass_kg, "mass_kg")
    body_radius = _finite_positive(central_body_radius_m, "central_body_radius_m")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
        raise ValueError("position_m and velocity_mps must contain finite values")
    radius = float(np.linalg.norm(position))
    if radius <= 0.0:
        raise ValueError("position_m must have non-zero norm")
    if cannonball.drag_area_m2 == 0.0:
        return np.zeros(3, dtype=float)
    atmospheric_velocity = np.cross(
        np.asarray([0.0, 0.0, atmosphere.rotation_rate_radps]),
        position,
    )
    relative_velocity = velocity - atmospheric_velocity
    relative_speed = float(np.linalg.norm(relative_velocity))
    density = atmosphere.density_kgpm3(radius - body_radius)
    scale = -0.5 * density * cannonball.drag_coefficient * cannonball.drag_area_m2 / mass
    return np.asarray(scale * relative_speed * relative_velocity, dtype=float)


def cannonball_srp_acceleration(
    position_m: ArrayLike,
    sun_position_m: ArrayLike,
    *,
    mass_kg: float,
    cannonball: Cannonball,
    solar_pressure_at_1au_Npm2: float = SOLAR_PRESSURE_AT_1_AU_NPM2,
) -> NDArray[np.float64]:
    """Return cannonball solar-radiation-pressure acceleration.

    The acceleration points away from the Sun and varies with inverse-square
    distance.  This first-order model intentionally omits eclipses and
    penumbra transitions.

    Args:
        position_m: Spacecraft position in a central-body inertial frame.
        sun_position_m: Sun position in the same frame.
        mass_kg: Instantaneous spacecraft mass.
        cannonball: Spacecraft projected area and reflectivity coefficient.
        solar_pressure_at_1au_Npm2: Radiation pressure at one astronomical unit.

    Returns:
        Three-component inertial acceleration in m/s².
    """
    position = np.asarray(position_m, dtype=float).reshape(3)
    sun_position = np.asarray(sun_position_m, dtype=float).reshape(3)
    mass = _finite_positive(mass_kg, "mass_kg")
    pressure = _finite_positive(
        solar_pressure_at_1au_Npm2,
        "solar_pressure_at_1au_Npm2",
    )
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(sun_position)):
        raise ValueError("position_m and sun_position_m must contain finite values")
    if cannonball.srp_area_m2 == 0.0:
        return np.zeros(3, dtype=float)
    sun_to_spacecraft = position - sun_position
    distance = float(np.linalg.norm(sun_to_spacecraft))
    if distance <= 0.0:
        raise ValueError("Spacecraft and Sun positions must not coincide")
    magnitude = (
        pressure
        * cannonball.reflectivity_coefficient
        * cannonball.srp_area_m2
        / mass
        * (ASTRONOMICAL_UNIT_M / distance) ** 2
    )
    return np.asarray(magnitude * sun_to_spacecraft / distance, dtype=float)
