"""Spacecraft and propulsion models.

These objects are intentionally "config-like": they are meant to read well in
Python scripts and to remain stable even as solver backends evolve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .forces import Cannonball


@dataclass(slots=True)
class Thruster:
    """Simple thruster model (data-only)."""

    name: str = "main"
    thrust_N: float = 0.0
    isp_s: float = 0.0
    propellant_mass_kg: float | None = None
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Spacecraft:
    """Spacecraft mass, propulsion, and simple force-model properties.

    Args:
        name: Human-readable spacecraft name.
        dry_mass_kg: Dry mass in kilograms.
        thrusters: Available propulsion models.
        info: Free-form user metadata.
        cannonball: Constant projected areas and coefficients used when drag
            or solar radiation pressure is enabled.
    """

    name: str = "SC"
    dry_mass_kg: float = 0.0
    thrusters: list[Thruster] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)
    cannonball: Cannonball = field(default_factory=Cannonball)

    @property
    def initial_mass_kg(self) -> float:
        """Return dry mass plus configured propellant masses.

        Returns
        -------
        float
            Initial spacecraft mass in kilograms.
        """
        prop = 0.0
        for t in self.thrusters:
            if t.propellant_mass_kg is not None:
                prop += float(t.propellant_mass_kg)
        return float(self.dry_mass_kg) + prop

    def get_thruster(self, name: str) -> Thruster | None:
        """Return a named thruster if it exists.

        Parameters
        ----------
        name
            Thruster name to look up.

        Returns
        -------
        Thruster or None
            Matching thruster, or ``None`` when no thruster has that name.
        """
        for t in self.thrusters:
            if t.name == name:
                return t
        return None

    def thruster(self, name: str) -> Thruster:
        """Return a named thruster.

        Parameters
        ----------
        name
            Thruster name to look up.

        Returns
        -------
        Thruster
            Matching thruster.

        Raises
        ------
        KeyError
            If no thruster has the requested name.
        """
        for t in self.thrusters:
            if t.name == name:
                return t
        raise KeyError(f"No thruster named {name!r} on spacecraft {self.name!r}")
