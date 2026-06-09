# AGENTS.md

## Project Identity

Octavian is a Python-first astrodynamics and trajectory optimization toolkit built around
mission-script APIs. The package should feel fast to iterate on, readable to aerospace
engineers, and modular enough to grow from simple transfers to broader mission design.

Tagline: Trajectory optimization and astrodynamics in Python. Fast, adaptable, optimal.

Name: Optimal Control & Trajectory Analysis for Vehicles, Astrodynamics, and Navigation.

## Core Philosophy

### Python is the interface

- Python is the primary user experience, configuration layer, and scripting surface.
- Mission scripts should read like clear configuration with real code semantics.
- Avoid hidden behavior that obscures what the user asked the solver to do.

### Readability beats cleverness

- Prefer explicit, well-named variables and straightforward control flow.
- Favor user-facing clarity over compact internals.
- If a design saves lines but makes mission logic harder to understand, reject it.

### Pragmatism beats ideology

- Choose the simplest maintainable solution that solves the real problem.
- Reuse proven tools instead of rebuilding them.
- Do not introduce abstractions before the codebase clearly needs them.

### Modular by default

- Prefer small composable pieces over monolithic logic.
- New capabilities should fit naturally into existing abstractions.
- Avoid special-case branches in core systems when a general model is possible.

### The code is the documentation

- Docstrings, examples, and readable APIs are part of the feature, not follow-up work.
- A capability that is not explained or demonstrated is incomplete.

## Architectural Guardrails

Keep these concerns separate:

1. Configuration: user intent and problem setup.
2. Compilation: translation into solver-compatible structures.
3. Solving: optimization execution and numerical workflow.
4. Reporting: results, plots, exports, and summaries.

Do not blur these layers without a strong reason.

## What Good Changes Look Like

- Preserve the Python-first mission-script mental model.
- Keep user APIs simple and explicit.
- Improve clarity, composability, or useful capability without adding unnecessary surface area.
- Treat examples and visualization as product features, not secondary extras.
- Keep the repo approachable to a new engineer reading it for the first time.

## What To Avoid

- Hidden side effects or surprising defaults.
- Clever but opaque code.
- Large refactors that are not required for the task.
- Tight coupling between unrelated components.
- New features without tests or example coverage when applicable.
- Changes that make the public API harder to reason about.

## Repo Workflow Expectations

- Assume the default branch is `dev`.
- Treat each Codex request as its own branch and pull request unless the user explicitly asks to work on an existing branch.
- Keep changes focused on the task at hand.
- Do not revert unrelated user changes.
- Prefer small, reviewable patches over broad rewrites.
- If a change touches user-facing behavior, update docs, examples, or both.
- Every new user-facing capability must include a docs update in the same Codex request or PR. Prefer updating
  the relevant tutorial, example guide, and API docstrings together so users can discover and apply the feature.

## Testing And Validation

Use the lightest validation that proves the change, then expand as needed.

Typical commands:

```bash
pip install -e ".[dev]"
pytest
python -m build
```

ASSET-backed solver commands should run inside the local conda environment:

```bash
conda run -n asset_env python -m pytest tests/test_example_regressions.py -q
conda run -n asset_env python examples/composable/08_chemical_burn_j2.py
```

Notes:

- `asset_asrl` is installed at
  `C:\Users\19145\AppData\Roaming\Python\Python312\site-packages\asset_asrl\__init__.py`.
- Plain `python` may find that package but fail to import ASSET with a missing DLL.
- Prefer `conda run -n asset_env ...` in non-interactive automation instead of `conda activate asset_env`,
  because activation does not persist across separate shell tool calls.
- Run `conda run -n asset_env ...` commands sequentially. Parallel conda runs can collide on temporary
  activation files on Windows.

Guidance:

- Run targeted tests first when the change is narrow.
- Run `pytest` before finishing when behavior changes or new code is added.
- If solver-backed execution depends on external packages or environment setup, state clearly
  what could not be verified.
- Do not claim success without saying what was actually run.

## Examples And Documentation

- Examples are first-class artifacts.
- New capabilities should usually include either a new example or an update to an existing one.
- Examples should be readable, reasonably fast, and representative of real use.
- Favor docstrings and examples that explain intent, not just parameters.

## Current Product Direction

Near-term priorities:

- Strengthen the mission API shape before expanding breadth.
- Continue improving impulsive, finite-burn, and low-thrust workflows.
- Expand orbital targeting and composable mission design patterns.
- Make outputs easy to understand through structured results and visualization.
- Grow breadth without turning the core into a pile of special cases.

Use this question as a design test:

Does this change make Octavian feel more like one coherent system and less like a set of
stitched-together features?

## PR Expectations

When preparing a PR, include:

- What changed.
- Why it matters.
- How it was validated.
- Any limitations, follow-ups, or environment constraints.

Prefer draft PRs for work that still needs confirmation or broader review.
