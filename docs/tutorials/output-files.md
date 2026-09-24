# Output Files And Static Plots

## Matplotlib Images And Windows

Install the visualization extra to use either the Matplotlib or Plotly
backend:

```bash
python -m pip install "octavian[viz]"
```

Import the plotting function at the top of your script and pass the solved
trajectory, just as with the HTML visualizers. Choose `save_trajectory_image`
for Earth-centered ECI, `save_relative_trajectory_image` for RIC, or
`save_cr3bp_trajectory_image` for rotating CR3BP coordinates.
Trajectory views are 3D by default; set
`projection` to `"xy"`, `"xz"`, or `"yz"` for a 2D plane. The output suffix
chooses PNG or JPEG:

```python
from octavian.viz.matplotlib import (
    save_trajectory_image,
    save_trajectory_diagnostics_image,
)

save_trajectory_image(
    solution.traj,
    "trajectory-xy.png",
    maneuvers=solution.result.maneuvers,
    phase_segments=solution.result.info.get("phase_segments"),
    projection="xy",
)
save_trajectory_diagnostics_image(
    solution.traj,
    "diagnostics.jpg",
    frame_kind="inertial",
    mu_m3ps2=solution.result.info["mu_m3ps2"],
    dpi=180,
)
```

For CR3BP plots, also pass `system=solution.cr3bp_system` and the appropriate
`dimensional` flag. Diagnostics accept `frame_kind="relative"` or
`frame_kind="rotating"` with the corresponding solar/CR3BP metadata.

The axis labels follow the selected plotting function. For example, `projection="xy"` means ECI
X/Y for an inertial result, radial/in-track for a relative result, and
synodic X/Y for a CR3BP result. All spatial projections use equal axis scales.
Trajectory legends sit below the axes so phase names and maneuver labels stay
clear of the plotted geometry. CR3BP body labels are clipped to the axes when
you zoom with custom limits.

For a desktop pop-up using the active Matplotlib GUI backend:

```python
from octavian.viz.matplotlib import show_trajectory, show_trajectory_diagnostics

show_trajectory(solution.traj, projection="xz", maneuvers=solution.result.maneuvers)
show_trajectory_diagnostics(
    solution.traj,
    frame_kind="inertial",
    mu_m3ps2=solution.result.info["mu_m3ps2"],
)
```

Figure builders return ordinary Matplotlib figures for customization before
display or export:

```python
from octavian.viz.matplotlib import trajectory_figure

figure = trajectory_figure(
    solution.traj,
    projection="xy",
    title="Transfer trajectory",
)
figure.savefig("custom-transfer-xy.png", dpi=200)

figure_3d = trajectory_figure(solution.traj, title="Transfer trajectory")
figure_3d.axes[0].view_init(elev=25, azim=35)
figure_3d.savefig("custom-transfer-3d.png", dpi=200)
```

Frame-specific functions follow the Plotly naming pattern:
`trajectory_figure`, `relative_trajectory_figure`,
`cr3bp_trajectory_figure`, and `trajectory_diagnostics_figure`, with matching
`save_*_image` and `show_*` helpers. The complete standalone workflow is
`examples/outputs/02_matplotlib_plots.py`. Planar image output also appears in
the Hohmann transfer, CWH rendezvous, canonical L1 orbit, and L1 orbit-family
composable examples.

### Reference Trajectories

The standalone ECI, RIC, and CR3BP trajectory visualizers accept `reference_trajectories`.
Pass a list of dictionaries with `traj` and optional `name` and `color` keys.
Each history needs at least three position columns; full `[r, v, t]` histories
also work. References use dashed lines and participate in automatic axis limits.

```python
import numpy as np
from octavian import EARTH, propagate
from octavian.viz.matplotlib import save_trajectory_image
from octavian.viz.plotly import save_trajectory_html

# initial_state is the departure state before the transfer burn.
mu = EARTH.mu_m3ps2
semi_major_axis_m = 1.0 / (
    2.0 / np.linalg.norm(initial_state.r_m)
    - np.dot(initial_state.v_mps, initial_state.v_mps) / mu
)
period_s = 2.0 * np.pi * np.sqrt(semi_major_axis_m**3 / mu)
references = [{
    "name": "Departure orbit",
    "traj": propagate.inertial(initial_state, np.linspace(0.0, period_s, 361)),
    "color": "#F59E0B",
}]
save_trajectory_image(
    solution.traj, "transfer.png", reference_trajectories=references, projection="xy",
)
save_trajectory_html(solution.traj, "transfer.html", reference_trajectories=references)
```

Supply reference positions in the **same frame and input units** as the primary
trajectory: ECI or RIC meters, or synodic meters/DU according to the CR3BP
`dimensional` flag. The visualizers apply their normal display-unit conversion.
They draw spatial curves without aligning reference timestamps or transforming
frames. These overlays do not change the mission, seed, objective, or constraints.

Quick examples 01–04 and composable examples 01, 02, 03, 05, 06, 08, and 11
propagate full departure/target reference histories. Example 08 includes J2;
the nominal two-body period sets its plotting span without imposing closure.
Pass `perturbations=Perturbations(j2=True)` to `propagate.inertial` for the
same workflow, or see the [propagation guide](propagation.md) for other forces.

Other composable comparisons include
a one-impulse solution for comparison with an inclined two-impulse transfer (07),
a Lambert reference for the finite chemical burns (09),
a nominal 1800-second analytical CWH rendezvous (14), and full L1/L2 Lyapunov
orbits (29). Where an example writes both HTML and PNG, it supplies the same
reference histories to both renderers.

### Matplotlib Example Gallery

These mission scripts save a PNG in addition to their existing interactive
HTML output. Run them from the repository root after installing `octavian[viz]`.
Files go to the current working directory; use `MPLBACKEND=Agg` on a machine
without a display. Saving an image closes its figure and does not open a GUI.

| Example | Static trajectory output | View and purpose |
| --- | --- | --- |
| `examples/quick/01_two_impulse_free_time.py` | `traj_quick_hohmann_transfer.png` | ECI XY: equatorial Hohmann transfer with departure/arrival burns. |
| `examples/composable/earth_centered/01_single_phase_terminal_dv_objective.py` | `traj_composable_hohmann_terminal_dv_objective.png` | ECI XY: Hohmann transfer between full circular reference orbits. |
| `examples/composable/earth_centered/09_impulse_vs_chemical_burn.py` | `traj_composable_chemical_reference.png` | ECI XY: finite chemical burns and the corresponding impulsive Lambert transfer. |
| `examples/composable/relative/14_cwh_relative_rendezvous.py` | `traj_composable_cwh_relative_rendezvous.png` | RIC XY: optimized rendezvous and a fixed-time analytical reference. |
| `examples/composable/earth_centered/07_terminal_orbital_elements.py` | `traj_composable_terminal_orbital_elements.png` | 3D ECI: inclined two-impulse transfer with the one-impulse solution as a reference. |
| `examples/composable/relative/21_safety_ellipse_transfer.py` | `traj_safety_ellipse_transfer.png` | 3D RIC: cross-track separation, surrounding coasts, phase colors, and burns. |
| `examples/composable/cislunar/29_periodic_orbit_transfer.py` | `traj_L1_to_L2_periodic_orbits.png` | Synodic XY in km: planar transfer with both reference Lyapunov orbits and burns. |
| `examples/composable/cislunar/31_jacobi_targeted_periodic_orbit.py` | `traj_jacobi_targeted_L1_periodic_orbit.png` | Synodic XY in DU: the Jacobi-selected planar Lyapunov orbit. |

Choose the projection from the physical geometry: XY preserves a planar orbit's
shape without an artificial viewing angle, while 3D reveals inclination and
cross-track motion. Example 29 uses the same reference orbits and phase colors
for both backends. Example 21 also preserves the pre/post-transfer coasts and
the shifted maneuver times in both views.

Examples 29 and 31 customize the returned figure's axis limits from the
trajectory bounds, with padding, to resolve the small orbits near L1/L2.
They use `axes.set_aspect("equal", adjustable="box")` to preserve both the
selected limits and equal spatial scales. Their HTML files retain the full
Earth–Moon context. Modify the ordinary Matplotlib axes before calling
`save_figure_image(figure, "orbit.png")` to choose a different view.

## Ephemeris Output Files

Octavian keeps optimization and file formatting separate. A successful
`Solution` can first produce one validated, SI-unit `Ephemeris`, then write
that same history to multiple external formats.

```python
ephemeris = solution.to_ephemeris(
    epoch="2026-01-01T00:00:00Z",
    object_name="DEMO SAT",
    object_id=-100001,
)

ephemeris.write("trajectory.e")
ephemeris.write("trajectory.oem")
ephemeris.write("trajectory.bsp")
ephemeris.write("trajectory.csv")
```

`Mission(initial_epoch=...)` is retained on the returned solution, so ordinary
mission scripts can omit the repeated `epoch=` argument:

```python
mission = Mission(
    phases=[transfer],
    initial_epoch="2026-01-01T00:00:00Z",
)
solution = mission.solve()
solution.export_ephemeris("trajectory.oem")
```

## Formats

| Extension | Format | Output units | Typical use |
|---|---|---|---|
| `.e` | STK ASCII ephemeris | m, m/s, elapsed s | Import into Ansys STK |
| `.oem` | CCSDS OEM 2.0 KVN | km, km/s, UTC | Standards-based exchange |
| `.bsp`, `.spk` | SPICE type-9 SPK | km, km/s, ET | SPICE geometry and analysis |
| `.csv` | Octavian tabular ephemeris | m, m/s, UTC and elapsed s | Inspection and custom tooling |

Files are not overwritten by default. Pass `overwrite=True` when replacement
is intentional.

## Relative Solutions

A relative solution may contain three useful histories:

- `trajectory="solved"` — the public RIC trajectory;
- `trajectory="chief"` — the reconstructed absolute chief ECI history;
- `trajectory="deputy"` — the reconstructed absolute deputy ECI history.

`trajectory="auto"` is the default. It selects the deputy absolute history for
relative solutions and the solved history for inertial solutions:

```python
solution.export_ephemeris(
    "deputy.oem",
    trajectory="deputy",
    object_name="DEPUTY",
)
solution.export_ephemeris(
    "chief.bsp",
    trajectory="chief",
    object_name="CHIEF",
    object_id=-100002,
)
```

This prevents a chief-centered RIC trajectory from being silently labeled as
an Earth-centered inertial ephemeris.

## Frame And NAIF Metadata

Exporters do not rotate state values. `frame_name` describes the frame already
represented by the selected trajectory. Absolute Octavian histories default
to `J2000`; override the label only after transforming the data into that
frame.

SPICE files also require numeric object and center identifiers:

```python
solution.export_ephemeris(
    "vehicle.bsp",
    object_id=-100001,
    center_id=399,
    frame_name="J2000",
)
```

Earth (`399`), Moon (`301`), and Sun (`10`) center IDs are inferred from normal
Octavian solution metadata. Custom centers require `center_id=`.

The complete executable workflow is
`examples/outputs/01_ephemeris_files.py` in the repository.
