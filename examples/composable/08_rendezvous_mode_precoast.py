import numpy as np

from octavian import Dynamics, Mission, Phase, Spacecraft, Thruster
from octavian.quick import state
from octavian.solvers import SolverOptions
from octavian.viz import save_trajectory_html

MU = 3.986004418e14

# "More complicated" usage:
#   - you can attach metadata objects via composition (Scenario, Spacecraft)
#   - you can switch to the pre-coast formulation (extra decision variable)

spacecraft = Spacecraft(
    name="Deputy",
    dry_mass_kg=120.0,
    thrusters=[Thruster(name="main", thrust_N=0.0, isp_s=0.0)],
)
dynamics = Dynamics(mu_m3ps2=MU)

x0 = state(
    r_m=[7000e3, 0.0, 0.0],
    v_mps=[0.0, float(np.sqrt(MU / 7000e3)), 0.0],
)

xf = state(
    r_m=[6100e3, 5000e3, 0.0],
    v_mps=[0.0, float(np.sqrt(MU / 7000e3)), 0.0],
)

precoast = Phase(
    name="precoast",
    mode="coast",
    spacecraft=spacecraft,
    dynamics=dynamics,
    initial_state=x0,
    tof_bounds_s=(0.0, 6000.0),
)

rendezvous = Phase(
    name="rendezvous",
    mode="rendezvous",
    previous=precoast,
    final_state=xf,
    tof_bounds_s=(400.0, 60000.0),
)

mission = Mission(
    phases=[precoast, rendezvous],
    name="Composable mission: pre-coast rendezvous",
    mesh_nsegs_precoast=40,
    mesh_nsegs_transfer=80,
    lambert_grid_size=100,
    solver_options=SolverOptions(print_level=3),
)

sol = mission.solve()
print(sol.summary())
res = sol.result
assert res is not None

save_trajectory_html(
    traj=res.traj,
    out_html="traj_composable_mission.html",
    x0_r_m=x0.r_m,
    x0_v_mps=x0.v_mps,
    xf_r_m=xf.r_m,
    xf_v_mps=xf.v_mps,
    maneuvers=res.maneuvers,
    title="Octavian: composable mission (pre-coast)",
)

print("Wrote traj_composable_mission.html")
