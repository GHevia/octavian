# Composable API Examples

The composable examples expose the lower-level mission pieces: `Mission`,
`Phase`, dynamics models, constraints, variables, links, and objectives. Use
these scripts as templates when the quick helper does not describe the mission
shape directly.

## Shared Composable API Terms

| Setting | What it controls |
| --- | --- |
| `Mission(phases=[...])` | The complete optimization problem. |
| `Phase(...)` | One segment of the trajectory. |
| `mode="coast"` | Ballistic two-body or perturbed coast dynamics. |
| `mode="finite_thrust"` | Propulsion-neutral finite-thrust dynamics with mass and vector-throttle controls. |
| `mode="chemical_burn"` | Compatibility spelling for a chemical finite-thrust phase. |
| `mode="low_thrust"` | Finite-thrust dynamics with an integrated low-thrust seed workflow. |
| `mode="relative_coast"` | CWH coast dynamics in a chief-centered LVLH frame. |
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
| `constraints.state(x0, where="Front")` | Fixes the departure Cartesian state. |
| `constraints.state(xf, where="Back")` | Fixes the target position and supplies target velocity. |
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
| `use_terminal_burn` | Compares one-impulse and two-impulse formulations. |

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
conda run -n asset_env python examples/composable/11_cwh_relative_rendezvous.py
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

The example prints the solution, frame, and selected dynamics model without
creating a plot. Existing inertial trajectory plots are not labeled for LVLH
geometry yet.

## 12: CWH Safety Corridor

Path: `examples/composable/12_cwh_safety_corridor.py`

Run:

```bash
conda run -n asset_env python examples/composable/12_cwh_safety_corridor.py
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

## 13: Low-Thrust Orbit Raise

Path: `examples/composable/13_low_thrust_orbit_raise.py`

Run:

```bash
conda run -n asset_env python examples/composable/13_low_thrust_orbit_raise.py
```

Capability showcased:

- A single `mode="low_thrust"` phase using the common mass/throttle ODE.
- A typed, dynamics-integrated prograde spiral initial guess.
- Free terminal orbital phase through semi-major-axis and near-circular
  eccentricity constraints.
- Explicit final-mass optimization and powered-phase reporting.

Important choices:

| Code | Purpose |
| --- | --- |
| `guesses.low_thrust_spiral(throttle=0.85)` | Integrates a prograde seed and initializes the control history. |
| `final_state=terminal_seed_anchor` | Supplies target radius and scaling without fixing terminal longitude. |
| `objectives.minimize_propellant()` | Maximizes final spacecraft mass. |
| `tof_bounds_s=(14 h, 24 h)` | Brackets the seed's 17.59-hour burn estimate. |

The reference solve raises a 560 kg spacecraft from a 7,000 km to an 8,000 km
near-circular orbit in about 17.72 hours using about 15.05 kg of propellant.

Expected output: `traj_composable_low_thrust_orbit_raise.html`.
