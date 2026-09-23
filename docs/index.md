<p align="center">
  <img src="assets/octavian_logo.PNG" alt="Octavian logo" width="220">
</p>

# Octavian

Octavian is a Python-first astrodynamics and trajectory optimization toolkit
built around readable mission scripts.

The project is organized around a mission-script workflow: write the trajectory
intent in Python, compile it into solver structures, solve with ASSET-backed
optimization, then inspect summaries and Plotly trajectory views.

## What Octavian Covers Today

- Two-impulse rendezvous and transfer design.
- Optional precoast phases before impulsive transfers.
- Composable coast phases with continuous or impulsive links.
- Terminal state, terminal delta-v, minimum-radius, absolute/relative
  orbital-element, RIC-component, safety, solar-geometry, and periodic-state
  constraints.
- Finite chemical and low-thrust phases with mass depletion, frame-aware
  thrust directions, and kinematic Euler-angle controls.
- J2, Sun/Moon gravity, exponential drag, and cannonball SRP in inertial and
  exact relative dynamics.
- CWH, nonlinear RIC, coupled chief/deputy, relative-element, and CR3BP
  propagation through one analysis namespace.
- Canonical/dimensional CR3BP propagation, periodic-orbit correction,
  impulsive inter-orbit transfers, and inertial perturbed-model handoff.
- Plotly HTML trajectory and frame-aware time-series diagnostics with maneuver
  markers.
- STK, CCSDS OEM, SPICE BSP/SPK, and CSV trajectory output.

## Documentation Map

- [Getting Started](tutorials/getting-started.md) covers installation, ASSET, and the basic mission workflow.
- [Feature Guide](feature-guide.md) maps common needs to APIs, tutorials, and executable examples.
- [Analysis Propagation](tutorials/propagation.md) collects two-body, relative, ROE, and CR3BP propagators.
- [Development Environment](tutorials/development-environment.md) gives the supported isolated ASSET setup.
- [JSON And YAML Missions](tutorials/config-files.md) documents the optional declarative config interface.
- [Mission Patterns](tutorials/mission-patterns.md) shows how to combine the current options into real scripts.
- [Concepts](concepts.md) explains the mission, phase, constraint, variable, link, and solution model.
- [Developer Architecture](developer-architecture.md) maps the codebase, object model, and backend flow for contributors.
- [Project Principles](project-principles.md) explains the design philosophy behind the API.
- [Quick API Examples](examples/quick.md) documents the high-level helper scripts.
- [Composable API Examples](examples/composable.md) documents each lower-level mission-building example.
- [Example Capability Index](examples/index.md) maps design tasks across every regime to executable scripts.
- [Cislunar Design Guide](examples/cislunar.md) builds from CR3BP propagation through periodic orbits, transfers, and perturbed-model recapture.
- [GitHub Pages](publishing.md) explains how this site is deployed.
- [Output And Visualization](tutorials/output-files.md) covers ephemeris files,
  Matplotlib images and desktop windows, and frame-aware plotting.
- [API Reference](api.md) is generated from numpy-style docstrings with mkdocstrings.

The files under `examples/` are executable mission and analysis scripts. They
are designed to be copied and edited from top to bottom, without application
entry-point boilerplate around the declarations.

## Local Docs

```bash
pip install -e ".[dev]"
python -m mkdocs serve
python -m mkdocs build
```
