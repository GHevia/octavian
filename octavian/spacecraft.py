from __future__ import annotations

"""Spacecraft and propulsion models.

These objects are intentionally "config-like": they are meant to read well in
Python scripts and to remain stable even as solver backends evolve.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class Thruster:
    """Simple thruster model (data-only)."""

    name: str = "main"
    thrust_N: float = 0.0
    isp_s: float = 0.0
    propellant_mass_kg: Optional[float] = None
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Spacecraft:
    """Spacecraft container.

    Future: geometry, inertia, power, thermal, etc.
    """

    name: str = "SC"
    dry_mass_kg: float = 0.0
    thrusters: List[Thruster] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def initial_mass_kg(self) -> float:
        prop = 0.0
        for t in self.thrusters:
            if t.propellant_mass_kg is not None:
                prop += float(t.propellant_mass_kg)
        return float(self.dry_mass_kg) + prop

    def thruster(self, name: str) -> Thruster:
        for t in self.thrusters:
            if t.name == name:
                return t
        raise KeyError(f"No thruster named {name!r} on spacecraft {self.name!r}")
