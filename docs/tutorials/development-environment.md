# Development Environment

Octavian's solver depends on ASSET, a compiled extension with native runtime
dependencies. Use one isolated environment per checkout. Linux is the
recommended daily-development platform and uses a standard Python virtual
environment; Windows remains a fully supported validation and development
platform through Conda or a virtual environment.

## Supported Platforms

The pinned ASSET 0.5.1 wheel supports CPython 3.10 through 3.12 on:

- Linux x86_64 with glibc 2.31 or newer (Ubuntu 20.04+ is suitable).
- Windows x86_64.

macOS and ARM Linux need an ASSET wheel or a supported source-build workflow
before they can run solver-backed Octavian missions. The pure-Python portions
of Octavian are not the limiting factor.

## Linux: Recommended Daily Development

From the repository root, create one local virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --no-user -e ".[dev]"
python -m octavian.diagnostics
```

Once activated, use normal commands:

```bash
python examples/quick/01_two_impulse_free_time.py
python -m pytest tests -q
python -m mkdocs build --strict
```

Do not use `pip install --user`; it defeats the isolation the virtual
environment provides. To update the checkout after changing dependencies:

```bash
python -m pip install --no-user -e ".[dev]"
```

## Windows: Recommended Conda Development

The recommended Windows environment is named `octavian-dev`. It keeps Python,
ASSET, the OpenMP runtime, Octavian, and development tools in one isolated
prefix.

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
| Python runtime | conda-forge | Isolated CPython 3.12 and platform runtime libraries. |
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

## Windows: Pip And Virtual Environment

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

Use **pip + venv on Linux** for ordinary Octavian development: it is simple,
repository-local, and directly matches the Linux CI test path. Use **Conda on
Windows** when developing locally there; it is the most robust route for ASSET
and future native scientific dependencies because conda manages the Python
runtime and DLL search path together. Use **Windows pip + venv** when your team
already standardizes on virtual environments. Both platforms must pass
`python -m octavian.diagnostics` before solver-backed work.

## Cross-Platform Validation

GitHub Actions runs the full test suite on Linux for Python 3.10, 3.11, and
3.12. It also creates a Windows Python 3.12 virtual environment, verifies the
native ASSET import, and runs the full suite there. A separate Windows Conda
job verifies the project-owned `octavian-dev` environment. This keeps Linux
development and Windows compatibility continuously tested without requiring
every contributor to maintain both machines.
