# Getting Started

This tutorial takes a new user from installation to a solved trajectory and an
inspectable Plotly output file.

## Create an Environment

For solver-backed workflows, use Octavian's supported conda environment. ASSET
is a compiled extension, so the environment keeps Python, ASSET, and its native
runtime libraries together.

```powershell
conda env create --file environment.yml
conda run --name octavian-dev python -m pip install --no-user -e ".[dev]"
conda run --name octavian-dev python -m octavian.diagnostics
```

The diagnostic must report that both `asset` and `asset_asrl` are loaded from
the `octavian-dev` prefix. See [Development Environment](development-environment.md)
for updates, daily commands, and the dependency tiers.

For a standard package install on macOS or Linux where ASSET already imports
correctly, a normal virtual environment is also supported:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "octavian[viz]"
```

Octavian declares `asset_asrl` as a runtime dependency because the optimization
backends use ASSET. If your platform cannot install ASSET from PyPI, install
ASSET using its platform-specific instructions, then install Octavian in that
same environment.

Check that ASSET imports before running solver-backed examples:

```bash
python -c "import asset_asrl; print(asset_asrl.__file__)"
```

If that command reports a missing shared library or DLL, Python found the
package but the native ASSET runtime is not loadable in the active environment.
Fix the ASSET installation first, then rerun the Octavian example.

The `dev` extra adds pytest, ruff, MkDocs, mkdocstrings, YAML, and visualization
dependencies.

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

## What the Script Does

The example starts by defining the central-body gravitational parameter and two
circular boundary states:

```python
MU = 3.986004418e14
R_INITIAL_M = 7_000e3
R_FINAL_M = 12_000e3

initial_state = state(
    r_m=[R_INITIAL_M, 0.0, 0.0],
    v_mps=[0.0, float(np.sqrt(MU / R_INITIAL_M)), 0.0],
)
target_state = state(
    r_m=[-R_FINAL_M, 0.0, 0.0],
    v_mps=[0.0, -float(np.sqrt(MU / R_FINAL_M)), 0.0],
)
```

Then it asks Octavian to solve a two-impulse transfer:

```python
mission = two_burn_rendezvous(
    initial_state,
    target_state,
    mu_m3ps2=MU,
    tf_bounds_s=(3_000.0, 7_000.0),
    nsegs=60,
    lambert_grid_size=60,
    nrevs_to_try=(0,),
    solver_options=SolverOptions(print_level=3),
    name="Quick: Hohmann transfer between circular orbits",
)
```

The important flags are:

| Flag | Meaning |
| --- | --- |
| `mu_m3ps2` | Gravitational parameter used by dynamics and Lambert seeding. |
| `tf_bounds_s` | Lower and upper bounds on final transfer time. |
| `nsegs` | Mesh resolution for the transfer phase. Higher values give the optimizer more transcription points. |
| `lambert_grid_size` | Number of Lambert time-of-flight guesses used to seed the solve. |
| `nrevs_to_try` | Revolution counts included in the Lambert seed search. `(0,)` keeps this first example single-revolution. |
| `solver_options` | Backend solver settings such as print level and line-search behavior. |
| `name` | Human-readable label used in summaries and plots. |

Finally, it solves and writes an HTML trajectory:

```python
sol = mission.solve()
print(sol.summary())

save_trajectory_html(
    sol.result.traj,
    "traj_quick_hohmann_transfer.html",
    maneuvers=sol.result.maneuvers,
    title=mission.name,
)
```

## Prefer A Literal Config File?

The same mission is available as a schema-versioned JSON example:

```bash
python -m octavian.config examples/config/01_two_impulse_transfer.json
```

See [JSON And YAML Mission Files](config-files.md) for the full schema,
optional YAML installation, strict validation behavior, and guidance on when a
Python mission script is the better fit.

## Validate a Development Checkout

For local development:

```bash
python -m pytest -q
python -m ruff check .
python -m mkdocs build --strict
```

The example regression tests solve ASSET-backed trajectories. They are slower
than import-only tests and should be run before publishing changes that affect
solver behavior:

```bash
python -m pytest tests/test_example_regressions.py -q
```
