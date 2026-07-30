# Example Capability Index

Octavian's examples are executable mission-design guides, not isolated API
snippets. Start with the row that matches the task you are trying to perform,
copy that script, and change its mission declarations.

All public mission states use SI units unless an argument explicitly names
another unit. The cislunar periodic-orbit examples are the deliberate
exception at the design boundary: they show canonical CR3BP inputs, convert
them to SI for the composable solver, and convert solved trajectories back for
comparison with the literature.

## Choose An API

| Need | Start here |
| --- | --- |
| A standard impulsive transfer or relative hop | The quick API examples. |
| Explicit coasts, burns, links, constraints, or force models | The composable examples. |
| Portable ephemeris files after a solve | The output example. |
| A versioned JSON or YAML mission artifact | The config example and config tutorial. |

## Quick Missions

| Task | Example | Capabilities |
| --- | --- | --- |
| Minimum complete transfer | `examples/quick/01_two_impulse_free_time.py` | Free time, two impulses, trajectory and diagnostics plots. |
| Add a departure coast | `examples/quick/02_two_impulse_precoast_impulsive_link.py` | Precoast, impulsive phase link, optimized departure time. |
| Compare objectives | `examples/quick/03_time_tradeoff.py` | Delta-v/time trade studies. |
| Sweep targets | `examples/quick/04_batch_targets.py` | Ordinary-Python batch execution. |
| Change central body | `examples/quick/05_sun_centered_transfer.py` | Sun-centered scaling and dynamics. |
| Build a relative hop | `examples/quick/06_relative_hop.py` | Coast-burn-coast-burn relative sequence. |
| String relative transfers together | `examples/quick/07_relative_transfer_chain.py` | Multiple transfers and intermediate coasts. |

## Earth-Centered Composable Missions

| Task | Example | Capabilities |
| --- | --- | --- |
| Build a phase explicitly | `examples/composable/earth_centered/01_single_phase_terminal_dv_objective.py` | Mission, phase, state constraints, impulse variables, objective. |
| Link a continuous precoast | `examples/composable/earth_centered/02_precoast_continuous_link.py` | Multiple phases and continuous state/time. |
| Insert an impulsive boundary | `examples/composable/earth_centered/03_precoast_impulsive_link.py` | Link maneuver and delta-v accounting. |
| Compare hard and soft arrival velocity | `examples/composable/earth_centered/04_terminal_velocity_hard_vs_objective.py` | Boundary constraints versus objective variables. |
| Plot maneuvers | `examples/composable/earth_centered/05_plot_with_maneuvers.py` | Interactive maneuver markers. |
| Form a three-burn transfer | `examples/composable/earth_centered/06_precoast_impulsive_link_3burn.py` | Three phases and three impulses. |
| Target orbital elements | `examples/composable/earth_centered/07_terminal_orbital_elements.py` | Semi-major axis, eccentricity, and inclination constraints. |
| Use finite chemical burns and J2 | `examples/composable/earth_centered/08_chemical_burn_j2.py` | Mass depletion, finite thrust, J2. |
| Compare impulses with finite burns | `examples/composable/earth_centered/09_impulse_vs_chemical_burn.py` | Model-fidelity comparison. |
| Add ephemeris third bodies | `examples/composable/earth_centered/10_sun_moon_perturbations.py` | BSP-driven Sun/Moon gravity and epoch handling. |
| Raise an orbit with low thrust | `examples/composable/earth_centered/18_low_thrust_orbit_raise.py` | Low-thrust seed, throttle, propellant objective. |
| Choose thrust frames and attitude coordinates | `examples/composable/earth_centered/19_thrust_frames_and_attitude.py` | Free vector, fixed direction, Euler angles, slew limits. |
| Add cannonball drag and SRP | `examples/composable/earth_centered/20_cannonball_drag_srp.py` | J2, exponential atmosphere, cannonball properties, BSP Sun. |

## Relative-Motion Composable Missions

| Task | Example | Capabilities |
| --- | --- | --- |
| Solve and independently propagate CWH | `examples/composable/relative/11_cwh_relative_rendezvous.py` | CWH seed/propagation, RIC optimization, relative plots. |
| Enforce operational geometry | `examples/composable/relative/12_cwh_safety_corridor.py` | RIC component, keep-out, approach cone, solar lighting constraints. |
| Convert absolute and relative representations | `examples/composable/relative/13_relative_representations.py` | ECI/RIC histories, D'Amico and classical relative elements. |
| Use exact nonlinear relative dynamics | `examples/composable/relative/14_nonlinear_relative_rendezvous.py` | Unlinearized relative equations. |
| Add relative perturbations and solar geometry | `examples/composable/relative/15_perturbed_relative_solar.py` | Coupled absolute propagation, J2, Sun/Moon, phase angle. |
| Compare exact RIC formulations | `examples/composable/relative/16_exact_ric_formulations.py` | Converted-to-inertial, nonlinear RIC, and coupled RIC. |
| Target D'Amico elements with free time | `examples/composable/relative/17_damico_free_time_target.py` | Native quasi-nonsingular ROE constraints. |
| Transfer between safety ellipses | `examples/composable/relative/18_safety_ellipse_transfer.py` | Solved-time pre/post coasts and relative phase links. |
| Insert finite burns | `examples/composable/relative/19_relative_finite_burn_coast.py` | Deputy finite-burn/coast chain and continuous mass. |
| Optimize a three-burn topology | `examples/composable/relative/20_relative_three_burn_transfer.py` | Independent relative phase times and an intermediate waypoint. |
| Propagate perturbed relative elements | `examples/composable/relative/21_perturbed_relative_element_propagation.py` | D'Amico propagation with J2/Sun and backward time. |
| Use differential cannonball properties | `examples/composable/relative/23_cannonball_drag_srp.py` | Chief/deputy drag and SRP differences. |
| Target classical relative elements | `examples/composable/relative/24_classical_relative_elements.py` | Native `[Δa, Δe, Δi, ΔΩ, Δω, ΔM]` propagation and constraints. |

## Cislunar Composable Missions

| Task | Example | Capabilities |
| --- | --- | --- |
| Learn the dimensional CR3BP API | `examples/composable/cislunar/22_earth_moon_cr3bp.py` | Lagrange points, propagation, Jacobi constant, synodic solve and plotting. |
| Solve a nondimensional periodic orbit | `examples/composable/cislunar/24_canonical_periodic_orbit.py` | Canonical L1 seed, ASSET front/back periodicity, phase condition, canonical diagnostics. |
| Transfer between periodic-orbit families | `examples/composable/cislunar/25_periodic_orbit_transfer.py` | L1 coast, departure impulse, free-time CR3BP transfer, L2 insertion and coast. |
| Re-target in a perturbed model | `examples/composable/cislunar/26_high_fidelity_recapture.py` | BSP Moon alignment, synodic/inertial conversion, J2, Sun/Moon gravity, SRP. |

Read [Designing In The Cislunar Regime](cislunar.md) before adapting the last
three examples. It explains where canonical and SI units meet, what
``periodic_state`` constrains, and why the perturbed example is a model
handoff rather than a perturbed CR3BP periodic orbit.

## Configuration And Output

| Task | Example | Capabilities |
| --- | --- | --- |
| Load a declarative mission | `examples/config/01_two_impulse_transfer.json` | Versioned config schema and the same public mission objects. |
| Export a solved trajectory | `examples/outputs/01_ephemeris_files.py` | STK `.e`, CCSDS OEM, SPICE BSP/SPK, and CSV. |

The focused [Quick API](quick.md) and [Composable API](composable.md) pages
explain the declarations used by each family. The
[API Reference](../api.md) is the definitive inventory of public call
signatures and docstrings.
