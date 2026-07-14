"""Characteristic scaling metadata for dimensional solver problems."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class SolverScaling:
    """Characteristic units used to condition one compiled problem."""

    length_m: float
    velocity_mps: float
    time_s: float
    mass_kg: float = 1.0

    def __post_init__(self) -> None:
        for field_name in ("length_m", "velocity_mps", "time_s", "mass_kg"):
            value = float(getattr(self, field_name))
            if value <= 0.0:
                raise ValueError(f"SolverScaling.{field_name} must be positive")
            object.__setattr__(self, field_name, value)

    @property
    def acceleration_mps2(self) -> float:
        """Return the implied characteristic acceleration."""
        return self.velocity_mps / self.time_s

    @property
    def force_N(self) -> float:
        """Return the implied characteristic force."""
        return self.mass_kg * self.acceleration_mps2

    def to_dict(self) -> dict[str, float]:
        """Return JSON-serializable scaling metadata."""
        return asdict(self)
