import numpy as np
from octavian.specs import BoundaryState, TwoImpulseFreeTimeSpec
from octavian.solvers import SolverOptions
from octavian.solvers.rendezvous import solve_two_impulse_free_time
from octavian.viz import save_trajectory_html

MU = 3.986004418e14

r0 = np.array([7000e3, 0.0, 0.0])
v0 = np.array([0.0, np.sqrt(MU / 7000e3), 0.0])

rf = np.array([6100e3, 5000e3, 0.0])
vf = np.array([0.0, np.sqrt(MU / 7000e3), 0.0])

spec = TwoImpulseFreeTimeSpec(
    x0=BoundaryState(r0, v0),
    xf=BoundaryState(rf, vf),
    tf_bounds_s=(400.0, 40000.0),
    nsegs=60,
    lambert_grid_size=80,
    nrevs_to_try=(0, 1),
)

opts = SolverOptions(print_level=0)
res = solve_two_impulse_free_time(spec, options=opts)
print(res.summary())

# Persist result (trajectory + maneuvers + metadata)
res.to_npz("two_impulse_free_time.npz")

save_trajectory_html(
    traj=res.traj,
    out_html="traj_free_time.html",
    x0_r_m=spec.x0.r_m,
    x0_v_mps=spec.x0.v_mps,
    xf_r_m=spec.xf.r_m,
    xf_v_mps=spec.xf.v_mps,
    maneuvers=res.maneuvers,
    title="Octavian: two-impulse rendezvous (free time)",
)
print("Wrote traj_free_time.html")
