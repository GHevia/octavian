"""Output example 01: export one solution to common ephemeris formats.

The output extension selects the writer. Octavian preserves one validated
SI-unit ephemeris in memory, then writes the units and metadata required by
STK, CCSDS OEM, SPICE, or CSV.

Run:
  python examples/outputs/01_ephemeris_files.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from octavian import (
    EARTH,
    Dynamics,
    Mission,
    Phase,
    Spacecraft,
    constraints,
    objectives,
    state,
    variables,
)
from octavian.solvers import SolverOptions

INITIAL_RADIUS_M = EARTH.mean_radius_m + 400_000.0
TARGET_RADIUS_M = EARTH.mean_radius_m + 800_000.0

initial_state = state(
    [INITIAL_RADIUS_M, 0.0, 0.0],
    [0.0, np.sqrt(EARTH.mu_m3ps2 / INITIAL_RADIUS_M), 0.0],
)
target_state = state(
    [-TARGET_RADIUS_M, 0.0, 0.0],
    [0.0, -np.sqrt(EARTH.mu_m3ps2 / TARGET_RADIUS_M), 0.0],
)

transfer = Phase(
    name="transfer",
    mode="coast",
    spacecraft=Spacecraft(name="DemoSat", dry_mass_kg=250.0),
    dynamics=Dynamics.for_body(EARTH),
    initial_state=initial_state,
    final_state=target_state,
    tof_bounds_s=(2_000.0, 5_000.0),
    constraints=[
        constraints.state(initial_state, where="Front"),
        constraints.state(target_state, where="Back"),
    ],
    variables=[
        variables.impulsive_delta_v(at="Front"),
        variables.impulsive_delta_v(at="Back"),
    ],
)
mission = Mission(
    name="Ephemeris export demonstration",
    phases=[transfer],
    initial_epoch="2026-01-01T00:00:00Z",
    objectives=[objectives.minimize_total_delta_v()],
    solver_options=SolverOptions(print_level=0),
    nrevs_to_try=(0,),
)

solution = mission.solve()
print(solution.summary())

output_directory = Path("ephemeris_output")
output_directory.mkdir(exist_ok=True)
for filename in (
    "transfer.e",
    "transfer.oem",
    "transfer.bsp",
    "transfer.csv",
):
    output = solution.export_ephemeris(
        output_directory / filename,
        object_name="OCTAVIAN DEMO SAT",
        object_id=-100_001,
        overwrite=True,
    )
    print(f"Wrote: {output}")
