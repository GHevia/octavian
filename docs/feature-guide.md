# Feature Guide

This page is the shortest route from a mission-design need to an executable
example. Public features should appear here when they are intended for normal
user scripts.

| Need | Primary API | Tutorial or guide | Executable example |
| --- | --- | --- | --- |
| Standard free-time transfer | `two_burn_rendezvous(...)` | Getting Started | `examples/quick/01_two_impulse_free_time.py` |
| Relative coast–burn–coast hop | `relative_hop(...)` | Mission Patterns | `examples/quick/06_relative_hop.py` |
| Multiple relative transfers | `relative_transfer_chain(...)` | Mission Patterns | `examples/quick/07_relative_transfer_chain.py` |
| Arbitrary linked phases | `Mission`, `Phase`, `Link` | Concepts | `examples/composable/earth_centered/02_precoast_continuous_link.py` |
| Finite burns and mass continuity | `mode="finite_thrust"` | Mission Patterns | `examples/composable/relative/22_relative_finite_burn_coast.py` |
| RIC/fixed/Euler thrust direction | `ThrustControl` | Thrust Direction And Attitude | `examples/composable/earth_centered/12_thrust_frames_and_attitude.py` |
| CWH and exact relative motion | `Dynamics.cwh`, `Dynamics.relative` | Concepts | `examples/composable/relative/19_exact_ric_formulations.py` |
| D'Amico/classical ROE targeting | relative-element constraints | Mission Patterns | `examples/composable/relative/20_damico_free_time_target.py` |
| Three-burn relative topology | composable relative phases | Mission Patterns | `examples/composable/relative/23_relative_three_burn_transfer.py` |
| Coordinate and representation conversion | `octavian.relative` transforms | Concepts | `examples/composable/relative/16_relative_representations.py` |
| Analysis-only propagation | `octavian.propagate` | Analysis Propagation | `examples/analysis/01_propagation_namespace.py` |
| J2 and Sun/Moon gravity | `Perturbations` | Mission Patterns | `examples/composable/earth_centered/10_sun_moon_perturbations.py` |
| Cannonball drag and SRP | `Cannonball`, `Perturbations` | Cannonball Drag And SRP | `examples/composable/relative/25_cannonball_drag_srp.py` |
| Solar-phase constraints | `constraints.solar_phase_angle` | Mission Patterns | `examples/composable/relative/18_perturbed_relative_solar.py` |
| RIC trajectory and time histories | `solution.viz()` | Concepts | `examples/composable/relative/17_nonlinear_relative_rendezvous.py` |
| Earth–Moon/general CR3BP | `CR3BPSystem`, `Dynamics.cr3bp` | Cislunar CR3BP | `examples/composable/cislunar/27_earth_moon_cr3bp.py` |
| Periodic orbits at a Jacobi value | `constraints.periodic_state`, `constraints.jacobi_constant` | Cislunar Design Guide | `examples/composable/cislunar/31_jacobi_targeted_periodic_orbit.py` |
| STK/OEM/BSP/CSV output | `solution.export_ephemeris` | Output Files | `examples/outputs/01_ephemeris_files.py` |
| JSON/YAML missions | `load_mission(...)` | JSON And YAML Missions | `examples/config/01_two_impulse_transfer.json` |

Lower-level helpers and complete signatures are listed in the API Reference.
The examples stay intentionally direct: copy one, change the declarations, and
run it as a normal Python script.
