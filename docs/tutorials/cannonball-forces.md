# Cannonball Drag And Solar Radiation Pressure

Octavian provides first-order drag and solar-radiation-pressure (SRP) models
for preliminary mission design. The same force implementation is used by
inertial coasts, finite burns, exact coupled relative motion, and numerical
relative-element propagation.

## Configure The Spacecraft And Environment

Spacecraft geometry belongs on `Spacecraft`; force selection belongs on
`Perturbations`:

```python
from octavian import Cannonball, Perturbations, Spacecraft

vehicle = Spacecraft(
    name="Inspector",
    dry_mass_kg=150.0,
    cannonball=Cannonball(
        drag_area_m2=4.0,
        drag_coefficient=2.2,
        srp_area_m2=5.0,
        reflectivity_coefficient=1.4,
    ),
)
forces = Perturbations(j2=True, drag=True, srp=True)
```

Both areas are constant projected areas. A positive mass and the corresponding
positive area are required when a phase enables drag or SRP. Coast phases use
the spacecraft's initial mass. Mass-carrying coasts and finite burns use the
instantaneous propagated mass, so drag and SRP naturally increase as
propellant is consumed.

## Inertial Coast Or Burn

Attach the force model to ordinary Earth-centered dynamics:

```python
from octavian import Dynamics, Phase

dynamics = Dynamics.for_body("earth", perturbations=forces)
coast = Phase(
    name="perturbed_coast",
    mode="coast",
    spacecraft=vehicle,
    dynamics=dynamics,
    tof_bounds_s=(1800.0, 3600.0),
)
```

Use the same `dynamics` and spacecraft in a `finite_thrust` phase. Thrust is
added to the environmental acceleration, while spacecraft mass is depleted by
the selected thruster.

SRP reads the Earth-centered Sun state from Octavian's bundled SPICE BSP. Set
`Mission.initial_epoch` whenever SRP is active. Enabling `srp=True` samples the
Sun but does not silently enable solar third-body gravity; add `sun=True` when
both effects are desired.

## Differential Relative Motion

Exact relative dynamics can use distinct chief and deputy properties:

```python
from octavian import Dynamics

relative_dynamics = Dynamics.relative(
    chief_initial_state_eci=chief_eci,
    chief_spacecraft=chief_vehicle,
    propagation_mode="coupled_eci",
    perturbations=forces,
)

relative_phase = Phase(
    name="inspection_coast",
    mode="coast",
    spacecraft=vehicle,  # the deputy
    dynamics=relative_dynamics,
)
```

The solver propagates both absolute states, applies each spacecraft's
ballistic properties, and converts the result to RIC. Omitting
`chief_spacecraft` intentionally leaves the chief gravity-only while the
deputy receives drag or SRP. CWH and native two-body RIC/ROE solver modes remain
reduced models; select `coupled_eci` for perturbations.

For analysis without an optimization solve, the same properties work with
relative orbital elements:

```python
from octavian import EARTH
from octavian.relative import propagate_relative_orbital_elements

history = propagate_relative_orbital_elements(
    initial_roe,
    times_s,
    chief_initial_state_eci=chief_eci,
    mu_m3ps2=EARTH.mu_m3ps2,
    perturbations=forces,
    initial_epoch="2026-01-01T00:00:00Z",
    chief_spacecraft=chief_vehicle,
    deputy_spacecraft=vehicle,
)
```

See
`examples/composable/relative/25_cannonball_drag_srp.py` for osculating D'Amico
elements, RIC reconstruction, and plots.

## Atmosphere Model

The default Earth model is a co-rotating, single-scale-height exponential
atmosphere:

```text
rho(h) = rho_ref exp(-(h - h_ref) / H)
a_drag = -0.5 rho Cd A / m |v_rel| v_rel
```

Supply an explicit model to change the assumptions:

```python
from octavian import ExponentialAtmosphere, Perturbations

atmosphere = ExponentialAtmosphere(
    reference_density_kgpm3=1.0e-12,
    reference_altitude_m=400_000.0,
    scale_height_m=50_000.0,
    rotation_rate_radps=7.2921159e-5,
)
forces = Perturbations(drag=True, atmosphere=atmosphere)
```

Octavian will not apply the default Earth atmosphere to another central body.
Provide `atmosphere=` explicitly for non-Earth drag studies.

The JSON/YAML schema uses the same separation:

```yaml
spacecraft:
  inspector:
    dry_mass_kg: 150.0
    cannonball:
      drag_area_m2: 4.0
      drag_coefficient: 2.2
      srp_area_m2: 5.0
      reflectivity_coefficient: 1.4

dynamics:
  earth_perturbed:
    central_body: earth
    perturbations:
      j2: true
      drag: true
      srp: true
```

Set `mission.initial_epoch` in the same config whenever SRP is enabled.

## Fidelity Boundary

These models intentionally omit:

- density tables, space weather, winds, and oblateness in the atmosphere;
- attitude-dependent projected area;
- self-shadowing, Earth eclipse, and penumbra transitions for SRP;
- detailed optical surface properties.

SRP points directly away from the Sun and follows inverse-square distance from
the configured pressure at one astronomical unit. Use an external high-fidelity
propagator when those omitted effects drive the design.
