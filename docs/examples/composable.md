# Composable API Examples

The composable examples expose the lower-level mission pieces: `Mission`,
`Phase`, dynamics models, constraints, variables, links, and objectives. Use
these scripts as templates when the quick helper does not describe the mission
shape directly.

Each file is a flat Python-as-configuration tutorial: declarations appear in
solver order and the solve/report statements stay visible at the bottom. Tests
exercise these scripts as scenarios, while numerical regression fixtures live
under `tests/` so test import mechanics do not shape the user examples.

## Shared Composable API Terms

| Setting | What it controls |
| --- | --- |
| `Mission(phases=[...])` | The complete optimization problem. |
| `Phase(...)` | One segment of the trajectory. |
| `mode="coast"` | Ballistic two-body or perturbed coast dynamics. |
| `mode="finite_thrust"` | Finite-thrust dynamics with mass and vector-throttle controls; in relative missions only the deputy is powered. |
| `mode="chemical_burn"` | Compatibility spelling for a chemical finite-thrust phase. |
| `mode="low_thrust"` | Finite-thrust dynamics with an integrated low-thrust seed workflow. |
| `mode="relative_coast"` | Coast dynamics in the selected chief-centered relative formulation. |
| `previous=...` | Connects a phase to the phase before it. |
| `link=links.continuous()` | Enforces continuous position, velocity, and time at the boundary. |
| `link=links.impulsive()` | Enforces continuous position and time while allowing velocity to jump. |
| `constraints.state(..., where="Front" or "Back")` | Fixes boundary state information. |
| `constraints.min_radius(..., where="Path")` | Keeps the trajectory above a radius floor along the phase. |
| `variables.ImpulsiveDeltaV(...)` | Exposes a boundary velocity jump as a decision variable and maneuver. |
| `objectives.minimize_total_delta_v()` | Minimizes the sum of declared impulsive delta-v terms. |
| `objectives.minimize_propellant()` | Maximizes final mass after the last powered phase. |

## 01: Single-Phase Terminal Delta-v Objective

Path: `examples/composable/01_single_phase_terminal_dv_objective.py`

Run:

```bash
python examples/composable/01_single_phase_terminal_dv_objective.py
```

Capability showcased:

- One coast phase.
- Front and back impulsive delta-v variables.
- Terminal velocity relaxed into a delta-v objective while terminal position remains fixed.
- Hohmann-style circular transfer used by the regression tests.

Important choices:

| Code | Purpose |
| --- | --- |
| `constraints.state(initial_state, where="Front")` | Fixes the departure Cartesian state. |
| `constraints.state(target_state, where="Back")` | Fixes the target position and supplies target velocity. |
| `ImpulsiveDeltaV(where="Front")` | Allows a departure burn. |
| `ImpulsiveDeltaV(where="Back")` | Relaxes terminal velocity and charges the arrival burn. |
| `lambert_grid_size=60` | Seeds the coast arc with a Lambert search. |
| `nrevs_to_try=(0,)` | Keeps the solve on the direct transfer family. |

Expected output: `traj_composable_hohmann_terminal_dv_objective.html`.

Screenshot placeholder: `docs/assets/screenshots/composable-01-terminal-dv.png`.

## 02: Precoast With Continuous Link

Path: `examples/composable/02_precoast_continuous_link.py`

Run:

```bash
python examples/composable/02_precoast_continuous_link.py
```

Capability showcased:

- Explicit precoast and transfer phases.
- Continuous phase boundary.
- No link maneuver.

Important choices:

| Code | Purpose |
| --- | --- |
| `precoast` phase | Propagates from the initial state before transfer. |
| `variables.ImpulsiveDeltaV(where="Front")` on precoast | Allows an initial departure adjustment. |
| `previous=precoast` | Orders the transfer after the precoast. |
| `link=links.continuous()` | Forces the transfer to start exactly where and how the precoast ends. |
| `ImpulsiveDeltaV(where="Back")` | Allows only a terminal arrival burn on the transfer phase. |

Expected output: `traj_composable_precoast_continuous_link.html`.

Screenshot placeholder: `docs/assets/screenshots/composable-02-continuous-link.png`.

## 03: Precoast With Impulsive Link

Path: `examples/composable/03_precoast_impulsive_link.py`

Run:

```bash
python examples/composable/03_precoast_impulsive_link.py
```

Capability showcased:

- Explicit precoast and transfer phases.
- Impulsive phase boundary.
- Link maneuver marker and objective contribution.

Important choices:

| Code | Purpose |
| --- | --- |
| `link=links.impulsive()` | Allows velocity to jump between precoast and transfer. |
| `ImpulsiveDeltaV(where="Front")` on transfer | Declares the link velocity jump as a burn. |
| `ImpulsiveDeltaV(where="Back")` on transfer | Declares the terminal arrival burn. |
| no impulse variable on precoast | Keeps the precoast ballistic from the fixed initial state. |

Expected output: `traj_composable_precoast_impulsive_link.html`.

Screenshot placeholder: `docs/assets/screenshots/composable-03-impulsive-link.png`.

## 04: Hard Terminal Velocity Versus Objective

Path: `examples/composable/04_terminal_velocity_hard_vs_objective.py`

Run:

```bash
python examples/composable/04_terminal_velocity_hard_vs_objective.py
```

Capability showcased:

- Same geometry solved two ways.
- Hard terminal state constraint.
- Terminal delta-v objective formulation.
- Clear distinction between a required terminal velocity and a velocity target that can be bought with delta-v.

Important choices:

| Code | Purpose |
| --- | --- |
| `terminal_is_objective=False` | Keeps terminal velocity as a hard constraint. |
| `terminal_is_objective=True` | Adds `ImpulsiveDeltaV(where="Back")`, relaxing terminal velocity into a cost. |
| `tof_bounds = (0.5 * tof_guess, 1.5 * tof_guess)` | Centers the solve around a known circular coast time. |
| printed velocity error | Shows whether terminal velocity was exactly enforced or reached through an objective. |

Expected output:

- `traj_composable_terminal_velocity_hard.html`.
- `traj_composable_terminal_velocity_objective.html`.

Screenshot placeholders:

- `docs/assets/screenshots/composable-04-hard-terminal-velocity.png`.
- `docs/assets/screenshots/composable-04-objective-terminal-velocity.png`.

## 05: Plotting Maneuver Markers

Path: `examples/composable/05_plot_with_maneuvers.py`

Run:

```bash
python examples/composable/05_plot_with_maneuvers.py
```

Capability showcased:

- Plotly HTML output with maneuver markers.
- Raw marker placement versus snapped-to-trajectory marker placement.
- Minimum-radius path constraint included in the transfer.

Important choices:

| Code | Purpose |
| --- | --- |
| `constraints.min_radius(r_min_m, where="Path")` | Keeps the path above the configured altitude floor. |
| `snap_maneuvers_to_traj(...)` | Moves maneuver markers to the nearest plotted trajectory sample for cleaner visuals. |
| `save_trajectory_html(..., maneuvers=...)` | Writes an inspectable plot with burn markers. |

Expected output:

- `traj_plot_maneuvers_raw.html`.
- `traj_plot_maneuvers_snapped.html`.

Screenshot placeholders:

- `docs/assets/screenshots/composable-05-maneuvers-raw.png`.
- `docs/assets/screenshots/composable-05-maneuvers-snapped.png`.

## 06: Precoast Plus Two Transfers With Three Burns

Path: `examples/composable/06_precoast_impulsive_link_3burn.py`

Run:

```bash
python examples/composable/06_precoast_impulsive_link_3burn.py
```

Capability showcased:

- Three phases.
- Two impulsive links.
- Terminal impulse.
- Minimum-altitude path constraint.

Important choices:

| Code | Purpose |
| --- | --- |
| `tof_is_relative=True` | Interprets each phase time bound as a duration rather than an absolute mission time. |
| `transfer1` and `transfer2` | Split the transfer into two optimized coast arcs. |
| impulsive links on both transfer phases | Create two interior maneuver opportunities. |
| `ImpulsiveDeltaV(where="Back")` on `transfer2` | Adds the terminal burn, bringing the total to three burns. |

Expected output: `traj_composable_precoast_impulsive_link_3burn.html`.

Screenshot placeholder: `docs/assets/screenshots/composable-06-three-burn.png`.

## 07: Terminal Orbital-Element Constraints

Path: `examples/composable/07_terminal_orbital_elements.py`

Run:

```bash
python examples/composable/07_terminal_orbital_elements.py
```

Capability showcased:

- Target semi-major axis, eccentricity, and inclination directly.
- Cartesian terminal state used as a guess anchor.
- Comparison between one terminal impulse and two terminal impulse variables.

Important choices:

| Code | Purpose |
| --- | --- |
| `classical_to_cartesian(...)` | Builds a Cartesian seed from the target orbital elements. |
| `constraints.semi_major_axis(...)` | Targets orbit size at the end of the phase. |
| `constraints.eccentricity(...)` | Targets orbit shape at the end of the phase. |
| `constraints.inclination_deg(...)` | Targets orbital plane tilt at the end of the phase. |
| `use_terminal_burn` | Reports one- and two-impulse local formulations; use it to compare feasible solutions, not to prove global objective ordering. |

Expected output: `traj_composable_terminal_orbital_elements.html`.

Screenshot placeholder: `docs/assets/screenshots/composable-07-orbital-elements.png`.

## 08: Chemical Burn With J2

Path: `examples/composable/08_chemical_burn_j2.py`

Run:

```bash
python examples/composable/08_chemical_burn_j2.py
```

Capability showcased:

- Burn-coast-burn structure.
- Finite chemical burn phases.
- Mass depletion state.
- Three thrust-direction controls.
- J2 perturbation enabled in every phase.
- Explicit propellant objective and powered-phase mass reporting.

The compiler does not require this exact three-phase shape. A finite-thrust
phase can stand alone, and longer powered/coast chains carry mass continuously
from the first powered phase through the last.

The same coast and burn EOM path supports `Perturbations(moon=True, sun=True)`
when the mission sets `initial_epoch`; Octavian samples the bundled reduced
DE440 Sun/Moon ephemeris in `ECI_TOD` into ASSET interpolation tables.

Expected output: `traj_composable_chemical_burn_j2.html`.

Screenshot placeholder: `docs/assets/screenshots/composable-08-chemical-j2.png`.

## 09: Impulsive Reference Versus Chemical Burn

Path: `examples/composable/09_impulse_vs_chemical_burn.py`

Run:

```bash
python examples/composable/09_impulse_vs_chemical_burn.py
```

Capability showcased:

- Impulsive transfer reference.
- Finite-burn transfer over the same coast-time window.
- Side-by-side comparison of idealized and chemical-burn workflows.

Important choices:

| Code | Purpose |
| --- | --- |
| `COAST_BOUNDS_S` | Uses the same coast-time window for the impulsive reference and finite-burn transfer. |
| `select_best_lambert_seed(...)` | Finds the best impulsive reference over that window. |
| `kepler_dense_guess(...)` | Builds a plot trajectory for the impulsive reference. |
| `chemical_mission.solve()` | Solves the finite burn-coast-burn mission. |
| `_chemical_equivalent_delta_v(...)` | Converts mass depletion summaries into equivalent delta-v for comparison. |
| `relative_difference > 0.20` guard | Fails loudly if the finite-burn result diverges too far from the impulsive reference. |

Expected output:

- `traj_composable_impulse_reference.html`.
- `traj_composable_chemical_reference.html`.

Screenshot placeholders:

- `docs/assets/screenshots/composable-09-impulse-reference.png`.
- `docs/assets/screenshots/composable-09-chemical-reference.png`.

## 10: J2, Moon, And Sun Perturbations

Path: `examples/composable/10_sun_moon_perturbations.py`

Run:

```bash
python examples/composable/10_sun_moon_perturbations.py
```

Capability showcased:

- Coast EOM with J2, Moon, and Sun perturbations enabled together.
- `Mission.initial_epoch` driving Sun/Moon ephemeris table generation.
- The bundled reduced DE440 Earth-centered Sun/Moon BSP in the `ECI_TOD` frame.

Important choices:

| Code | Purpose |
| --- | --- |
| `Perturbations(j2=True, moon=True, sun=True)` | Enables the three core Earth-orbit perturbations. |
| `Mission(initial_epoch=...)` | Defines the start date for SPICE ephemeris sampling. |
| `third_body_table_step_s=3600.0` | Sets the Sun/Moon interpolation sample spacing. |
| `ImpulsiveDeltaV(where="Front"/"Back")` | Lets the optimizer satisfy the endpoint geometry while reporting the required burns. |

Expected output: `traj_composable_sun_moon_perturbations.html`.

Screenshot placeholder: `docs/assets/screenshots/composable-10-sun-moon-perturbations.png`.

## 11: CWH Relative Rendezvous

Path: `examples/composable/11_cwh_relative_rendezvous.py`

Run:

```bash
conda run -n octavian-dev python examples/composable/11_cwh_relative_rendezvous.py
```

Capability showcased:

- Chief-centered LVLH/RTN state and result metadata.
- CWH dynamics derived from a 400 km circular Earth orbit.
- Analytic CWH position-targeted initial guess.
- Optimized departure and arrival impulses for a one-kilometer rendezvous.

Important choices:

| Code | Purpose |
| --- | --- |
| `Dynamics.cwh(...)` | Couples mean motion, LVLH frame, and relative scaling. |
| `mode="relative_coast"` | Selects the relative coast phase semantics. |
| relative `constraints.state(...)` | Fixes deputy state values in meters and meters per second. |
| front/back `impulsive_delta_v` | Frees boundary velocities and reports both maneuvers. |

The example also writes `traj_composable_cwh_relative_rendezvous.html` with
explicit radial, in-track, and cross-track axes, a chief marker at the origin,
and the optimized impulse markers.

## 12: CWH Safety Corridor

Path: `examples/composable/12_cwh_safety_corridor.py`

Run:

```bash
conda run -n octavian-dev python examples/composable/12_cwh_safety_corridor.py
```

Capability showcased:

- A 75 m spherical keep-out zone around the chief.
- A one-sided 30° approach cone along the negative LVLH y axis.
- An 85°–121° angle bound to a fixed illumination direction.
- Geometry-aware CWH seed selection and post-solve satisfaction reporting.

The unconstrained minimum-delta-v arc falls outside the 30° corridor. The
constraint therefore moves the optimized transfer to the cone boundary,
demonstrating that the geometry changes the solution rather than only checking
it afterward.

## 13: Relative Representations

Path: `examples/composable/13_relative_representations.py`

This optimizer-free example builds a chief ECI state and deputy RIC state,
converts RIC to absolute ECI and back, and moves between D'Amico and classical
relative orbital elements. It is the representation layer used by both CWH and
nonlinear missions.

## 14: Nonlinear Relative Rendezvous

Path: `examples/composable/14_nonlinear_relative_rendezvous.py`

`Dynamics.relative(..., propagation_mode="coupled_eci")` propagates chief and deputy as two exact
central-gravity states. CWH supplies only the initial guess. Public boundary
states, keep-out geometry, maneuvers, the trajectory, and diagnostics remain
in RIC.

Expected outputs:

- `traj_composable_nonlinear_relative_rendezvous.html`
- `diagnostics_composable_nonlinear_relative_rendezvous.html`

## 15: Perturbed Relative Solar Geometry

Path: `examples/composable/15_perturbed_relative_solar.py`

This extends example 14 with differential J2 and solar gravity. The
solar-phase constraint uses the SPICE BSP at `Mission.initial_epoch`; it is
evaluated from the propagated chief/deputy state and the actual chief-to-Sun
line. The diagnostics file includes solar phase angle over time.

| Code | Purpose |
| --- | --- |
| `Dynamics.relative(chief_initial_state_eci=...)` | Selects full coupled nonlinear propagation. |
| `Perturbations(j2=True, sun=True)` | Applies the same force model independently to chief and deputy. |
| `constraints.solar_phase_angle(...)` | Bounds the changing relative-position/Sun angle. |

## 16: Exact RIC Formulations

Path: `examples/composable/16_exact_ric_formulations.py`

This optimizer-free comparison propagates the exact six-state circular-chief
RIC equations and an independent coupled chief/deputy two-body model. It also
shows the explicit `"nonlinear_ric"` and `"coupled_ric"` dynamics declarations.
The first is the nonlinear equation from which CWH is linearized; the second
stacks a propagated chief ECI state with the deputy's RIC state and also
supports eccentric chiefs.

## 17: Native D'Amico Free-Time Target

Path: `examples/composable/17_damico_free_time_target.py`

This phase propagates D'Amico relative orbital elements directly. A six-element
Front constraint fixes the initial orbit, while a scalar `delta_lambda` Back
constraint selects the arrival time within `tof_bounds_s`. The Cartesian RIC
states are only seed anchors—the target never becomes an absolute-state
constraint. After the solve, analytic two-body propagation adds six-minute
coasts before and after the optimized interval. Those coasts are not optimizer
phases; they are stitched to the RIC history for the trajectory and diagnostic
plots and colored separately through `phase_segments`.

Expected outputs:

- `traj_composable_damico_free_time.html`
- `diagnostics_composable_damico_free_time.html`

## 18: Low-Thrust Orbit Raise

Path: `examples/composable/18_low_thrust_orbit_raise.py`

The low-thrust example remains part of the broader composable suite after the
relative-motion build-up. It demonstrates a dynamics-integrated spiral seed,
free terminal orbital phase, propellant optimization, and inertial diagnostics
for Cartesian state, radius, speed, and osculating elements.

## 18B: Safety-Ellipse ROE Transfer

Path: `examples/composable/18_safety_ellipse_transfer.py`

This example defines both boundary orbits with D'Amico relative orbital
elements, converts them to RIC states at their respective chief epochs, and
optimizes a fixed-duration, two-impulse transfer between them. The transfer
uses exact `coupled_ric` dynamics because native `damico` propagation is an
unforced drift model and cannot represent an impulsive jump between arbitrary
ROE sets. Both conversions remain relative; no absolute deputy constraint is
introduced.

Analysis-only propagation extends the departure and arrival ROE sets into
safety ellipses before and after the optimized transfer.

Expected outputs:

- `traj_safety_ellipse_transfer.html`
- `diagnostics_safety_ellipse_transfer.html`

## 19: Relative Finite Burn–Coast–Burn

Path: `examples/composable/19_relative_finite_burn_coast.py`

This example composes two exact deputy finite burns around a five-minute
relative coast. `Dynamics.relative(..., propagation_mode="coupled_eci")`
propagates both absolute spacecraft states, applies thrust and mass depletion
only to the deputy, and reports the complete solution in RIC. The coast carries
deputy mass continuously between the powered phases.

Expected outputs:

- `traj_composable_relative_finite_burn_coast.html`
- `diagnostics_composable_relative_finite_burn_coast.html`
