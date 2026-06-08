"""Quick example 04: small batch run (sweep target geometry).

Run:
  python examples/quick/04_batch_targets.py

This is intentionally simple: it just loops, solves, prints a small summary,
and writes one HTML for the best case.
"""

from __future__ import annotations

import numpy as np

from octavian import state, two_burn_rendezvous
from octavian.solvers import SolverOptions
from octavian.viz.plotly import save_trajectory_html

MU = 3.986004418e14
R_INITIAL_M = 7_000e3



x0 = state(
    r_m=[R_INITIAL_M, 0.0, 0.0],
    v_mps=[0.0, float(np.sqrt(MU / R_INITIAL_M)), 0.0],
)

# Sweep final circular-orbit radii while keeping the target opposite the start.
target_radii_m = np.linspace(8_000e3, 14_000e3, 7)

results = []
for i, radius_m in enumerate(target_radii_m):
    target_speed_mps = float(np.sqrt(MU / radius_m))
    xf = state(
        r_m=[-float(radius_m), 0.0, 0.0],
        v_mps=[0.0, -target_speed_mps, 0.0],
    )

    mission = two_burn_rendezvous(
        x0,
        xf,
        mu_m3ps2=MU,
        tf_bounds_s=(600.0, 10_000.0),
        nsegs=50,
        precoast=True,
        lambert_grid_size=45,
        solver_options=SolverOptions(print_level=3),
        nrevs_to_try=(0,),
        name=f"Quick sweep case {i} (rf={radius_m/1e3:.0f} km)",
    )

    sol = mission.solve()
    dv = sol.result.total_dv_mps()
    tf = sol.result.tf_s()
    results.append((dv, tf, sol, radius_m))
    print(
        f"case {i}: rf={radius_m/1e3:6.0f} km | converged={sol.result.converged} | tf={tf:8.1f} s | dv={dv:10.3f} m/s"
    )

# pick best by dv among converged
converged = [r for r in results if r[2].result.converged]
best = min(converged, key=lambda x: x[0]) if converged else min(results, key=lambda x: x[0])
dv_best, tf_best, sol_best, radius_best_m = best

print("\nBEST")
print(f"rf={radius_best_m/1e3:.0f} km | tf={tf_best:.1f} s | dv={dv_best:.3f} m/s")
print(sol_best.summary())

out_html = "traj_quick_batch_best.html"
save_trajectory_html(
    sol_best.result.traj,
    out_html,
    maneuvers=sol_best.result.maneuvers,
    title=f"Quick sweep BEST (rf={radius_best_m/1e3:.0f} km)",
)
print(f"Wrote: {out_html}")

