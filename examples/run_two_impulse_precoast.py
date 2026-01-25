import numpy as np
from octavian.specs import BoundaryState, TwoImpulsePreCoastSpec
from octavian.solvers import SolverOptions
from octavian.solvers.rendezvous import solve_two_impulse_precoast
from octavian.viz import save_trajectory_html

MU = 3.986004418e14

r0 = np.array([7000e3, 0.0, 0.0])
v0 = np.array([0.0, np.sqrt(MU / 7000e3), 0.0])

rf = np.array([6100e3, 5000e3, 0.0])
vf = np.array([0.0, np.sqrt(MU / 7000e3), 0.0])

spec = TwoImpulsePreCoastSpec(
    x0=BoundaryState(r0, v0),
    xf=BoundaryState(rf, vf),
    t1_bounds_s=(0.0, 8000.0),
    tf_bounds_s=(400.0, 40000.0),
    nsegs_precoast=30,
    nsegs_transfer=60,
    precoast_grid_size=30,
    lambert_grid_size=80,
    nrevs_to_try=(0, 0),
)

opts = SolverOptions(print_level=0)
res = solve_two_impulse_precoast(spec, options=opts)
print(res.summary())

res.to_npz("two_impulse_precoast.npz")

save_trajectory_html(
    traj=res.traj,
    out_html="traj_precoast.html",
    x0_r_m=spec.x0.r_m,
    x0_v_mps=spec.x0.v_mps,
    xf_r_m=spec.xf.r_m,
    xf_v_mps=spec.xf.v_mps,
    maneuvers=res.maneuvers,
    title="Octavian: two-impulse rendezvous (variable pre-coast)",
)
print("Wrote traj_precoast.html")
