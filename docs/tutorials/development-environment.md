# Development Environment

Octavian's solver depends on ASSET, a compiled extension with native runtime
dependencies. Use one dedicated conda environment for this repository rather
than combining a conda environment with a nested `.venv` or installing ASSET
with `pip --user`.

The recommended environment is named `octavian-dev`. It keeps Python, ASSET,
the OpenMP runtime, Octavian, and development tools in one isolated prefix.

## Create The Environment

From the repository root:

```powershell
conda env create --file environment.yml
conda run --name octavian-dev python -m pip install --no-user -e ".[dev]"
conda run --name octavian-dev pre-commit install
conda run --name octavian-dev python -m octavian.diagnostics
```

The first command installs the minimum native solver layer:

| Layer | Package source | Purpose |
| --- | --- | --- |
| Python runtime | conda-forge | Isolated CPython 3.12 and Windows runtime libraries. |
| ASSET | pinned PyPI wheel | Native optimal-control extension and its OpenMP runtime. |
| Octavian core | this checkout | Mission, astrodynamics, solver, and ephemeris code. |
| Development tools | `.[dev]` | Tests, docs, linting, build, YAML, and visualization support. |

`environment.yml` sets `PYTHONNOUSERSITE=1`. The diagnostic command verifies
that both `asset` and `asset_asrl` load from the active conda prefix, not from a
global `%APPDATA%` package directory. It exits nonzero if the environment is
not self-contained.

## Daily Use

For one command, use `conda run`:

```powershell
conda run --name octavian-dev python -m pytest tests -q
conda run --name octavian-dev python examples/composable/08_chemical_burn_j2.py
conda run --name octavian-dev python -m mkdocs build --strict
```

For an interactive shell, activate the environment once:

```powershell
conda activate octavian-dev
python -m octavian.diagnostics
python -m pytest tests -q
```

Update an existing environment after changing `environment.yml` or the project
dependencies:

```powershell
conda env update --name octavian-dev --file environment.yml --prune
conda run --name octavian-dev python -m pip install --no-user -e ".[dev]"
```

## Dependency Tiers

The package keeps optional user interfaces separate from solver core:

```powershell
# Solver and astrodynamics only.
python -m pip install octavian

# Add Plotly HTML visualization.
python -m pip install "octavian[viz]"

# Add JSON/YAML mission-file support.
python -m pip install "octavian[yaml]"
```

ASSET currently brings several scientific packages transitively. Octavian still
declares NumPy and SpiceyPy directly because its public astrodynamics and
ephemeris modules import them. Plotly and Pillow are optional because only the
visualization helper needs them.

## Environment Rule Of Thumb

Use one environment per repository or project:

- Use a standard `.venv` for pure-Python projects.
- Use a named conda environment when a project needs native libraries, compiled
  extensions, GPU tooling, or platform DLLs.
- Do not install project dependencies with `--user`.
- Do not nest a `.venv` inside conda.
- Keep the base conda environment for conda itself, not project dependencies.

If `python -m octavian.diagnostics` reports a missing DLL, do not copy DLLs into
the repository or modify system-wide `PATH`. Recreate `octavian-dev` from
`environment.yml` and verify the diagnostic report before solving a mission.
