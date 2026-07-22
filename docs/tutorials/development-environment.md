# Development Environment

Octavian's solver depends on ASSET, a compiled extension with native runtime
dependencies. Choose one isolated environment per checkout: a dedicated conda
environment is the recommended Windows development route, while a standard
Python virtual environment is also supported.

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

## Pip And Virtual Environment

Use this option when you already manage Python with the standard library and
want each checkout to keep its environment in `.venv`:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --no-user -e ".[dev]"
python -m octavian.diagnostics
```

After activation, use normal commands such as
`python examples/quick/01_two_impulse_free_time.py`. Do not combine this
virtual environment with conda, and do not use `pip install --user`.

On Windows, ASSET's pip wheel installs its OpenMP DLL under
`.venv\Library\bin`. When Octavian starts, it adds that directory to its own
DLL search path when present. This is a process-local adjustment: it does not
modify the system `PATH` or affect unrelated Python projects. The diagnostic
command verifies this route before solving a mission.

## Which Environment Should I Choose?

Use **conda** for Octavian development on Windows. It is the most robust route
for ASSET and future native scientific dependencies because conda manages the
Python runtime and DLL search path together. Use **pip + venv** when your team
already standardizes on Python virtual environments, wants a repository-local
environment, or does not otherwise need conda. Both paths are supported and
must pass `python -m octavian.diagnostics` before solver-backed work.
