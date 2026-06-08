# Getting Started

This tutorial sets up the current Octavian workflow: install the package, make
sure ASSET is available for solver-backed runs, execute a quick transfer, and
inspect the generated trajectory.

## Installation

For package use:

```bash
pip install octavian
```

For local development:

```bash
pip install -e ".[dev]"
```

The development extra installs pytest, ruff, MkDocs, and mkdocstrings. It does
not install `asset_asrl`.

## ASSET Requirement

Octavian treats ASSET (`asset_asrl`) as an optional runtime dependency so that
documentation, configuration objects, tests, and non-solver utilities can import
without ASSET installed.

Solver-backed workflows need ASSET in the same Python environment that runs the
examples. In this repository's local Windows setup, ASSET-backed commands should
run through the conda environment:

```bash
conda run -n asset_env python -m pytest tests/test_example_regressions.py -q
conda run -n asset_env python examples/quick/01_two_impulse_free_time.py
```

Use `conda run -n asset_env ...` for automation instead of relying on
`conda activate`, because activation does not persist across separate shell
commands.

## Run a First Transfer

The quickest path is the high-level two-burn rendezvous helper:

```bash
python examples/quick/01_two_impulse_free_time.py
```

That script builds a Hohmann-like transfer between circular orbits, solves it,
prints a short summary, and writes:

```text
traj_quick_hohmann_transfer.html
```

Open the HTML file in a browser to inspect the trajectory and maneuver markers.

Screenshot placeholder: add a trajectory image at
`docs/assets/screenshots/quick-01-hohmann-transfer.png`.

## Mission-Script Shape

Most scripts follow this structure:

```python
from octavian import state, two_burn_rendezvous
from octavian.solvers import SolverOptions

x0 = state(r_m=[7000e3, 0.0, 0.0], v_mps=[0.0, 7546.0, 0.0])
xf = state(r_m=[-12000e3, 0.0, 0.0], v_mps=[0.0, -5763.0, 0.0])

mission = two_burn_rendezvous(
    x0,
    xf,
    tf_bounds_s=(3000.0, 7000.0),
    solver_options=SolverOptions(print_level=3),
)

solution = mission.solve()
print(solution.summary())
```

Use the quick API when the standard rendezvous shape is enough. Use the
composable API when you need explicit phases, links, custom constraints, finite
burns, or perturbations.

## Validate the Install

For a lightweight local check:

```bash
python -m pytest -q
python -m ruff check .
python -m mkdocs build
```

For ASSET-backed trajectory regression checks:

```bash
conda run -n asset_env python -m pytest tests/test_example_regressions.py -q
```
