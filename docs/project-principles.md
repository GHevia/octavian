# Project Principles

Octavian is a Python-first astrodynamics and trajectory optimization toolkit
built around mission-script APIs. It should feel fast to iterate on, readable
to aerospace engineers, and modular enough to grow from simple transfers to
broader mission design.

Tagline: Trajectory optimization and astrodynamics in Python. Fast, adaptable,
optimal.

Name: Optimal Control & Trajectory Analysis for Vehicles, Astrodynamics, and
Navigation.

## Python Is The Interface

Python is the primary user experience, configuration layer, and scripting
surface. Mission scripts should read like clear configuration with real code
semantics.

Good Octavian scripts make the user intent visible. They avoid hidden behavior
that obscures what the user asked the solver to do.

## Readability Beats Cleverness

Mission design code should be easy for aerospace engineers to scan and modify.
Prefer explicit, well-named variables and straightforward control flow over
compact internals that make the mission logic harder to understand.

## Pragmatism Beats Ideology

Octavian should choose the simplest maintainable solution that solves the real
problem. It should reuse proven tools instead of rebuilding them and should not
introduce abstractions before the codebase clearly needs them.

## Modular By Default

New capabilities should fit naturally into small composable pieces:

- configuration for user intent,
- compilation into solver structures,
- solving and numerical workflow,
- reporting, plots, exports, and summaries.

Keeping those concerns separate makes the system easier to extend without
turning the core into a set of special cases.

## Examples Are Product Features

Examples are not secondary extras. A feature is incomplete if users cannot see
how to apply it. When a new user-facing capability is added, the relevant docs,
example guide, and tutorial should be updated in the same change.

## Design Test

Use this question when extending Octavian:

Does this change make Octavian feel more like one coherent system and less like
a set of stitched-together features?
