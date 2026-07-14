"""Compatibility shim for the renamed preconfigured solver backend.

The ASSET backend for built-in two-impulse problems now lives in
`octavian.solvers.preconfigured`. This module remains so existing imports of
`octavian.solvers.rendezvous` continue to work during the transition.
"""

from .preconfigured import (
    RendezvousResult,
    solve,
    solve_two_impulse_free_time,
    solve_two_impulse_precoast,
)

__all__ = [
    "RendezvousResult",
    "solve",
    "solve_two_impulse_free_time",
    "solve_two_impulse_precoast",
]
