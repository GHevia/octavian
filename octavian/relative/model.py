"""Configuration models for exact and reduced-order relative motion."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..bodies import EARTH, CelestialBody
from ..coordinates import CoordinateFrame, SolverScaling, ric
from ..specs import BoundaryState


class RelativePropagationMode(str, Enum):
    """State representation and equations used by a relative phase.

    ``COUPLED_ECI`` propagates chief and deputy absolute states and is the only
    formulation that currently supports perturbations. ``COUPLED_RIC`` carries
    the chief ECI state together with an exact deputy RIC state. ``NONLINEAR_RIC``
    is the exact, pre-linearization circular-chief model from which CWH is
    obtained. The two element modes propagate native two-body relative orbital
    elements.
    """

    COUPLED_ECI = "coupled_eci"
    COUPLED_RIC = "coupled_ric"
    NONLINEAR_RIC = "nonlinear_ric"
    DAMICO = "damico"
    CLASSICAL_ELEMENTS = "classical_elements"

    @classmethod
    def parse(cls, value: RelativePropagationMode | str) -> RelativePropagationMode:
        """Normalize a public mode value and common spelling aliases."""
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower().replace("-", "_")
        aliases = {
            "inertial": cls.COUPLED_ECI,
            "converted_to_inertial": cls.COUPLED_ECI,
            "stacked_eci": cls.COUPLED_ECI,
            "stacked_ric": cls.COUPLED_RIC,
            "direct_ric": cls.NONLINEAR_RIC,
            "exact_ric": cls.NONLINEAR_RIC,
            "relative_orbital_elements": cls.DAMICO,
            "roe": cls.DAMICO,
            "damico_roe": cls.DAMICO,
            "classical": cls.CLASSICAL_ELEMENTS,
        }
        if normalized in aliases:
            return aliases[normalized]
        try:
            return cls(normalized)
        except ValueError as exc:
            choices = ", ".join(mode.value for mode in cls)
            raise ValueError(
                f"Unknown relative propagation mode {value!r}; choose one of {choices}"
            ) from exc


@dataclass(frozen=True, slots=True)
class NonlinearRelative:
    """Configure a nonlinear or relative-element propagation formulation.

    The default ``coupled_eci`` mode preserves Octavian's original nonlinear
    implementation: chief and deputy Cartesian states are integrated under the
    same absolute force model, then reported in RIC. Other modes expose native
    RIC or relative-element solver states for direct constraints.

    Args:
        chief_initial_state_eci: Absolute chief state defining the RIC frame.
        central_body: Body whose gravity defines the reference orbit.
        chief_name: Human-readable name used by the returned RIC frame.
        reference_length_m: Characteristic separation used for solver scaling.
        propagation_mode: Native state representation and relative equations.
    """

    chief_initial_state_eci: BoundaryState
    central_body: CelestialBody = EARTH
    chief_name: str = "chief"
    reference_length_m: float = 1_000.0
    propagation_mode: RelativePropagationMode | str = RelativePropagationMode.COUPLED_ECI

    def __post_init__(self) -> None:
        radius_m = float(np.linalg.norm(self.chief_initial_state_eci.r_m))
        angular_momentum = float(
            np.linalg.norm(
                np.cross(
                    self.chief_initial_state_eci.r_m,
                    self.chief_initial_state_eci.v_mps,
                )
            )
        )
        if radius_m <= 0.0 or angular_momentum <= 0.0:
            raise ValueError("chief_initial_state_eci must define a non-degenerate orbit")
        if not str(self.chief_name).strip():
            raise ValueError("NonlinearRelative.chief_name must not be empty")
        if float(self.reference_length_m) <= 0.0:
            raise ValueError("NonlinearRelative.reference_length_m must be positive")
        object.__setattr__(self, "reference_length_m", float(self.reference_length_m))
        mode = RelativePropagationMode.parse(self.propagation_mode)
        object.__setattr__(self, "propagation_mode", mode)
        if mode is RelativePropagationMode.NONLINEAR_RIC:
            radial_speed_mps = float(
                np.dot(
                    self.chief_initial_state_eci.r_m,
                    self.chief_initial_state_eci.v_mps,
                )
                / radius_m
            )
            circular_speed_mps = float(np.sqrt(self.central_body.mu_m3ps2 / radius_m))
            actual_speed_mps = float(np.linalg.norm(self.chief_initial_state_eci.v_mps))
            if abs(radial_speed_mps) > 1.0e-8 * circular_speed_mps or not np.isclose(
                actual_speed_mps,
                circular_speed_mps,
                rtol=1.0e-8,
                atol=1.0e-8,
            ):
                raise ValueError(
                    "propagation_mode='nonlinear_ric' requires a circular chief "
                    "state; use 'coupled_ric' for an eccentric chief"
                )

    @property
    def frame(self) -> CoordinateFrame:
        """Return the chief-centered RIC reporting frame."""
        return ric(self.chief_name)

    @property
    def mean_motion_radps(self) -> float:
        """Return the chief mean motion used by seeds and reduced models."""
        radius_m = float(np.linalg.norm(self.chief_initial_state_eci.r_m))
        return float(np.sqrt(self.central_body.mu_m3ps2 / radius_m**3))

    @property
    def state_representation(self) -> str:
        """Return the semantic representation of the propagated state."""
        representations = {
            RelativePropagationMode.COUPLED_ECI: "chief_deputy_eci",
            RelativePropagationMode.COUPLED_RIC: "chief_eci_deputy_ric",
            RelativePropagationMode.NONLINEAR_RIC: "relative_ric",
            RelativePropagationMode.DAMICO: "damico",
            RelativePropagationMode.CLASSICAL_ELEMENTS: "classical_relative_elements",
        }
        return representations[self.propagation_mode]

    @property
    def scaling(self) -> SolverScaling:
        """Return relative-state characteristic units for solver conditioning."""
        return SolverScaling(
            length_m=self.reference_length_m,
            velocity_mps=self.mean_motion_radps * self.reference_length_m,
            time_s=1.0 / self.mean_motion_radps,
        )
