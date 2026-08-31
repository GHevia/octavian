# Output Files And Static Plots

## Matplotlib Images And Windows

Install the visualization extra to use either the Matplotlib or Plotly
backend:

```bash
python -m pip install "octavian[viz]"
```

A solved mission selects the correct inertial, relative RIC, or rotating
CR3BP view automatically. The output suffix chooses PNG or JPEG:

```python
solution.viz().save_image("trajectory.png")
solution.viz().save_diagnostics_image("diagnostics.jpg", dpi=180)
```

For a desktop pop-up using the active Matplotlib GUI backend:

```python
solution.viz().show()
solution.viz().show_diagnostics()
```

Figure builders return ordinary Matplotlib figures for customization before
display or export:

```python
from octavian.viz.matplotlib import trajectory_figure

figure = trajectory_figure(solution.traj, title="Transfer trajectory")
figure.axes[0].view_init(elev=25, azim=35)
figure.savefig("custom-transfer.png", dpi=200)
```

Frame-specific functions follow the Plotly naming pattern:
`trajectory_figure`, `relative_trajectory_figure`,
`cr3bp_trajectory_figure`, and `trajectory_diagnostics_figure`, with matching
`save_*_image` and `show_*` helpers. The complete standalone workflow is
`examples/outputs/02_matplotlib_plots.py`.

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
