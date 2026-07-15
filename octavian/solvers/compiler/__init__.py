"""Composable mission compiler building blocks.

The public solver entry point remains :mod:`octavian.solvers.composable`.
Modules in this package own one compilation responsibility and deliberately do
not expose ASSET objects through Octavian's user-facing configuration layer.
"""

from .phase_compiler import PhaseBuild

__all__ = ["PhaseBuild"]
