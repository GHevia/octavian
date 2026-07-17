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
- Terminal state, terminal delta-v, minimum-radius, and orbital-element constraints.
- Finite chemical burn phases with mass depletion and thrust-direction controls.
- J2 perturbation support in the composable backend.
- Plotly HTML trajectory visualization with maneuver markers.

## Documentation Map

- [Getting Started](tutorials/getting-started.md) covers installation, ASSET, and the basic mission workflow.
- [Development Environment](tutorials/development-environment.md) gives the supported isolated ASSET setup.
- [JSON And YAML Missions](tutorials/config-files.md) documents the optional declarative config interface.
- [Mission Patterns](tutorials/mission-patterns.md) shows how to combine the current options into real scripts.
- [Concepts](concepts.md) explains the mission, phase, constraint, variable, link, and solution model.
- [Developer Architecture](developer-architecture.md) maps the codebase, object model, and backend flow for contributors.
- [Project Principles](project-principles.md) explains the design philosophy behind the API.
- [Quick API Examples](examples/quick.md) documents the high-level helper scripts.
- [Composable API Examples](examples/composable.md) documents each lower-level mission-building example.
- [GitHub Pages](publishing.md) explains how this site is deployed.
- [API Reference](api.md) is generated from numpy-style docstrings with mkdocstrings.

The files under `examples/` are intentionally flat, executable mission
configurations. They are designed to be copied and edited from top to bottom,
without application entry-point boilerplate around the mission declaration.

## Local Docs

```bash
pip install -e ".[dev]"
python -m mkdocs serve
python -m mkdocs build
```
