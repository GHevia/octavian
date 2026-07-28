"""Configuration model for nonlinear chief/deputy relative motion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..bodies import EARTH, CelestialBody
from ..coordinates import CoordinateFrame, SolverScaling, ric
from ..specs import BoundaryState


@dataclass(frozen=True, slots=True)
class NonlinearRelative:
    """Full nonlinear relative dynamics represented by two absolute states.

    The solver propagates chief and deputy Cartesian states under the same
    absolute force model.  Public inputs, constraints, results, and plots are
    converted to the chief's instantaneous RIC frame at the compiler/reporting
    boundary.  This avoids linearizing central gravity or adding perturbations
    to an already-linearized CWH model.
    """

    chief_initial_state_eci: BoundaryState
    central_body: CelestialBody = EARTH
    chief_name: str = "chief"
    reference_length_m: float = 1_000.0

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
            raise ValueError(
                "chief_initial_state_eci must define a non-degenerate orbit"
            )
        if not str(self.chief_name).strip():
            raise ValueError("NonlinearRelative.chief_name must not be empty")
        if float(self.reference_length_m) <= 0.0:
            raise ValueError("NonlinearRelative.reference_length_m must be positive")
        object.__setattr__(self, "reference_length_m", float(self.reference_length_m))

    @property
    def frame(self) -> CoordinateFrame:
        """Return the chief-centered RIC reporting frame."""
        return ric(self.chief_name)

    @property
    def mean_motion_radps(self) -> float:
        """Return a circularized mean motion used only for CWH seed generation."""
        radius_m = float(np.linalg.norm(self.chief_initial_state_eci.r_m))
        return float(np.sqrt(self.central_body.mu_m3ps2 / radius_m**3))

    @property
    def scaling(self) -> SolverScaling:
        """Return relative-state characteristic units for solver conditioning."""
        return SolverScaling(
            length_m=self.reference_length_m,
            velocity_mps=self.mean_motion_radps * self.reference_length_m,
            time_s=1.0 / self.mean_motion_radps,
        )
