"""Reference-frame metadata carried by dynamics and solved trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

FrameKind = Literal["inertial", "rotating", "relative"]


@dataclass(frozen=True, slots=True)
class CoordinateFrame:
    """Describe the origin and orientation used by a trajectory.

    This object is metadata, not a coordinate transformation by itself. Frame
    transformation services are added alongside dynamics that need them, while
    every solver result can already state what its coordinates mean.
    """

    name: str
    origin: str
    orientation: str
    kind: FrameKind = "inertial"

    def __post_init__(self) -> None:
        for field_name in ("name", "origin", "orientation"):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"CoordinateFrame.{field_name} must not be empty")
            object.__setattr__(self, field_name, value)
        if self.kind not in ("inertial", "rotating", "relative"):
            raise ValueError("CoordinateFrame.kind must be inertial, rotating, or relative")

    def to_dict(self) -> dict[str, str]:
        """Return JSON-serializable frame metadata."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, str]) -> CoordinateFrame:
        """Reconstruct frame metadata stored in a result archive."""
        return cls(
            name=value["name"],
            origin=value["origin"],
            orientation=value["orientation"],
            kind=value.get("kind", "inertial"),  # type: ignore[arg-type]
        )


def inertial(origin: str, *, orientation: str = "ICRF", name: str | None = None) -> CoordinateFrame:
    """Create an inertial Cartesian frame centered on ``origin``."""
    normalized_origin = str(origin).strip().lower().replace(" ", "_")
    return CoordinateFrame(
        name=name or f"{normalized_origin}_inertial",
        origin=normalized_origin,
        orientation=orientation,
        kind="inertial",
    )


def lvlh(chief: str = "chief", *, name: str | None = None) -> CoordinateFrame:
    """Create a chief-centered local-vertical/local-horizontal frame.

    The orientation follows the RTN convention used by Octavian's relative
    dynamics: x is radial, y is along track, and z is orbit normal.
    """
    normalized_chief = str(chief).strip().lower().replace(" ", "_")
    return CoordinateFrame(
        name=name or f"{normalized_chief}_lvlh",
        origin=normalized_chief,
        orientation="LVLH/RTN",
        kind="relative",
    )


def ric(chief: str = "chief", *, name: str | None = None) -> CoordinateFrame:
    """Create a chief-centered radial/in-track/cross-track frame.

    RIC, RTN, and the LVLH convention used by Octavian describe the same
    right-handed axes: radial ``R`` points away from the central body,
    in-track ``I`` follows the chief's motion, and cross-track ``C`` follows
    the chief's angular momentum.  This named constructor lets mission scripts
    use the terminology most common in relative-motion analysis.
    """
    normalized_chief = str(chief).strip().lower().replace(" ", "_")
    return CoordinateFrame(
        name=name or f"{normalized_chief}_ric",
        origin=normalized_chief,
        orientation="RIC/RTN/LVLH",
        kind="relative",
    )


def synodic(
    primary: str,
    secondary: str,
    *,
    name: str | None = None,
) -> CoordinateFrame:
    """Create a barycentric primary-secondary rotating frame.

    The +X axis points from primary to secondary and +Z follows the circular
    system angular momentum.
    """
    primary_name = str(primary).strip().lower().replace(" ", "_")
    secondary_name = str(secondary).strip().lower().replace(" ", "_")
    if not primary_name or not secondary_name:
        raise ValueError("Synodic primary and secondary names must not be empty")
    return CoordinateFrame(
        name=name or f"{primary_name}_{secondary_name}_synodic",
        origin=f"{primary_name}_{secondary_name}_barycenter",
        orientation=f"{primary_name}_to_{secondary_name}_rotating",
        kind="rotating",
    )


EARTH_INERTIAL = inertial("earth", orientation="ECI", name="earth_inertial")
