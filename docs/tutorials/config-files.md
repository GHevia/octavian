# JSON And YAML Mission Files

Python mission scripts remain Octavian's primary interface. They support loops,
functions, generated states, external data, and any other ordinary Python
workflow. The config reader is an optional declarative front end for users who
prefer a literal file that can be generated, reviewed, or exchanged without
embedding Python code.

Both interfaces construct the same `Mission`, `Phase`, `Spacecraft`,
`Dynamics`, constraint, variable, and objective objects. There is no separate
config solver.

## Run The Shipped JSON Mission

```bash
python -m octavian.config examples/config/01_two_impulse_transfer.json
```

The command loads schema version 1, validates all named references, solves the
mission through the normal runner, and prints the standard solution summary.

The equivalent Python entry point is:

```python
from octavian import load_mission

mission = load_mission("examples/config/01_two_impulse_transfer.json")
solution = mission.solve()
print(solution.summary())
```

Use `load_mission_mapping(data)` or `mission_from_dict(data)` when another
program has already produced a Python mapping.

## Install YAML Support

JSON support uses only the Python standard library. YAML uses PyYAML's safe
loader and is available through an optional extra:

```bash
pip install "octavian[yaml]"
```

YAML and JSON use the same schema. For example:

```yaml
schema_version: 1

states:
  departure:
    r_m: [7000000.0, 0.0, 0.0]
    v_mps: [0.0, 7546.053290107542, 0.0]
  arrival:
    r_m: [-12000000.0, 0.0, 0.0]
    v_mps: [0.0, -5763.393400014367, 0.0]

spacecraft:
  vehicle:
    name: ConfigSat
    dry_mass_kg: 150.0
    thrusters:
      - name: main

dynamics:
  earth_two_body:
    central_body: earth

mission:
  name: Config Hohmann transfer
  phases:
    - name: transfer
      mode: coast
      spacecraft: vehicle
      dynamics: earth_two_body
      initial_state: departure
      final_state: arrival
      tof_bounds_s: [3000.0, 7000.0]
      constraints:
        - {type: state, state: departure, where: Front}
        - {type: state, state: arrival, where: Back}
      variables:
        - {type: impulsive_delta_v, where: Front}
        - {type: impulsive_delta_v, where: Back}
  objectives:
    - {type: delta_v}
  nrevs_to_try: [0]
```

## How The Schema Is Organized

The top level contains five explicit keys:

| Key | Purpose |
| --- | --- |
| `schema_version` | Required integer version; currently `1`. |
| `states` | Named Cartesian states in meters and meters per second. |
| `spacecraft` | Named vehicles and their thrusters. |
| `dynamics` | Named two-body or CWH environments. |
| `mission` | Ordered phases, objectives, solver options, and mission-level settings. |

Phases refer to named objects instead of repeating them. A phase connection is
also explicit:

```yaml
- name: coast
  spacecraft: vehicle
  dynamics: earth_two_body
  tof_bounds_s: [100.0, 600.0]

- name: transfer
  previous: coast
  link: impulsive
  tof_bounds_s: [600.0, 7000.0]
  tof_is_relative: true
```

`previous` may only refer to a phase declared earlier in the list. A connected
phase inherits omitted spacecraft and dynamics through the same `Phase`
behavior used by Python scripts. Omitting `previous` creates an independent
phase; list order alone does not add a hidden connection.

## Supported Declarations

Schema version 1 covers the current public mission building blocks:

- two-body dynamics with `central_body`, optional perturbations, frame, and scaling;
- unforced linear CWH dynamics with `model: cwh`, chief orbit radius, optional
  inline `chief_initial_state_eci`, and relative scaling;
- nonlinear and relative-element dynamics with `model: relative`, a required
  inline `chief_initial_state_eci`, and `propagation_mode` set to
  `coupled_eci`, `coupled_ric`, `nonlinear_ric`, `damico`, or
  `classical_elements`; J2/Moon/Sun perturbations require `coupled_eci`;
- coast, rendezvous, relative-coast, finite-thrust, chemical-burn, and low-thrust phase modes;
- continuous and impulsive links;
- boundary state, position, minimum-radius, absolute orbital-element, native
  `ric_state`, native relative-orbital-element, keep-out, approach-cone, fixed
  lighting-angle, and ephemeris solar-phase-angle constraints;
- impulsive delta-v variables and boundary impulse events;
- delta-v, time, and propellant objectives;
- low-thrust spiral initial guesses;
- solver, retry, solve, continuation-plan, mesh, and Lambert options.

Type names use lowercase snake case, such as `semi_major_axis`,
`keep_out_sphere`, and `low_thrust_spiral`. Mission inputs use SI units unless
the field name explicitly names another unit.

## Validation And Safety

The loader rejects unknown keys, duplicate phase names, unsupported types, and
unresolved references with a path such as
`config.mission.phases[1].dynamics`. This prevents a misspelling from becoming
an unnoticed default.

YAML is parsed with `yaml.safe_load`; Python object tags and executable YAML
constructors are not accepted. The config format never evaluates Python code
or imports user-selected classes.

## When To Use Python Instead

Use a Python mission script when the configuration needs computed ephemeris
states, parameter sweeps, custom bodies, reusable functions, external services,
or a new model that is not yet represented by schema version 1. Config files
are a convenience surface, not a restriction on Octavian's Python-first API.
