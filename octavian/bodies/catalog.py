"""Small, explicit catalog of central-body physical constants."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ..coordinates import CoordinateFrame, inertial


def _normalize_name(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


@dataclass(frozen=True, slots=True)
class CelestialBody:
    """Physical constants and naming metadata for a central body."""

    name: str
    mu_m3ps2: float
    mean_radius_m: float
    j2_coefficient: float = 0.0
    aliases: tuple[str, ...] = ()
    inertial_orientation: str = "ICRF"

    def __post_init__(self) -> None:
        normalized_name = _normalize_name(self.name)
        if not normalized_name:
            raise ValueError("CelestialBody.name must not be empty")
        object.__setattr__(self, "name", normalized_name)
        if float(self.mu_m3ps2) <= 0.0:
            raise ValueError("CelestialBody.mu_m3ps2 must be positive")
        if float(self.mean_radius_m) <= 0.0:
            raise ValueError("CelestialBody.mean_radius_m must be positive")
        if float(self.j2_coefficient) < 0.0:
            raise ValueError("CelestialBody.j2_coefficient must be non-negative")
        object.__setattr__(self, "mu_m3ps2", float(self.mu_m3ps2))
        object.__setattr__(self, "mean_radius_m", float(self.mean_radius_m))
        object.__setattr__(self, "j2_coefficient", float(self.j2_coefficient))
        object.__setattr__(
            self,
            "aliases",
            tuple(_normalize_name(alias) for alias in self.aliases),
        )

    def inertial_frame(self, *, orientation: str | None = None) -> CoordinateFrame:
        """Return an inertial frame centered on this body."""
        return inertial(
            self.name,
            orientation=orientation or self.inertial_orientation,
            name=f"{self.name}_inertial",
        )


EARTH = CelestialBody(
    name="earth",
    mu_m3ps2=3.986004418e14,
    mean_radius_m=6_378_136.3,
    j2_coefficient=1.08262668e-3,
    aliases=("terra",),
    inertial_orientation="ECI",
)

MOON = CelestialBody(
    name="moon",
    mu_m3ps2=4.9048695e12,
    mean_radius_m=1_737_400.0,
    j2_coefficient=2.03263e-4,
    aliases=("luna",),
)

SUN = CelestialBody(
    name="sun",
    mu_m3ps2=1.32712440018e20,
    mean_radius_m=695_700_000.0,
    aliases=("sol",),
)


def _catalog() -> Mapping[str, CelestialBody]:
    entries: dict[str, CelestialBody] = {}
    for body in (EARTH, MOON, SUN):
        for name in (body.name, *body.aliases):
            if name in entries:
                raise RuntimeError(f"Duplicate celestial-body catalog name {name!r}")
            entries[name] = body
    return MappingProxyType(entries)


CATALOG = _catalog()


def resolve(body: CelestialBody | str) -> CelestialBody:
    """Resolve a body object or case-insensitive catalog name."""
    if isinstance(body, CelestialBody):
        return body
    normalized = _normalize_name(body)
    try:
        return CATALOG[normalized]
    except KeyError as exc:
        available = ", ".join(sorted({item.name for item in CATALOG.values()}))
        raise KeyError(f"Unknown celestial body {body!r}. Available bodies: {available}") from exc
