# Output Files And Static Plots

## Matplotlib Images And Windows

Install the visualization extra to use either the Matplotlib or Plotly
backend:

```bash
python -m pip install "octavian[viz]"
```

A solved mission selects the correct inertial, relative RIC, or rotating
CR3BP view automatically. Trajectory views are 3D by default; set
`projection` to `"xy"`, `"xz"`, or `"yz"` for a 2D plane. The output suffix
chooses PNG or JPEG:

```python
solution.viz().save_image("trajectory.png")
solution.viz().save_image("trajectory-xy.png", projection="xy")
solution.viz().save_diagnostics_image("diagnostics.jpg", dpi=180)
```

The axis labels remain frame-aware. For example, `projection="xy"` means ECI
X/Y for an inertial result, radial/in-track for a relative result, and
synodic X/Y for a CR3BP result. All spatial projections use equal axis scales.
Trajectory legends sit below the axes so phase names and maneuver labels stay
clear of the plotted geometry. CR3BP body labels are clipped to the axes when
you zoom with custom limits.

For a desktop pop-up using the active Matplotlib GUI backend:

```python
solution.viz().show()
solution.viz().show(projection="xz")
solution.viz().show_diagnostics()
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

### Matplotlib Example Gallery

These mission scripts save a PNG in addition to their existing interactive
HTML output. Run them from the repository root after installing `octavian[viz]`.
Files go to the current working directory; use `MPLBACKEND=Agg` on a machine
without a display. Saving an image closes its figure and does not open a GUI.

| Example | Static trajectory output | View and purpose |
| --- | --- | --- |
| `examples/quick/01_two_impulse_free_time.py` | `traj_quick_hohmann_transfer.png` | ECI XY: equatorial Hohmann transfer with departure/arrival burns. |
| `examples/composable/earth_centered/07_terminal_orbital_elements.py` | `traj_composable_terminal_orbital_elements.png` | 3D ECI: transfer to an inclined target orbit. |
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
