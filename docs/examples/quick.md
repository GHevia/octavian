# Quick API Examples

The quick examples use `octavian.two_burn_rendezvous` for common impulsive
transfer and rendezvous workflows. They are the best starting point when you
want a compact mission script and do not need to manage every phase yourself.

## 01: Hohmann Transfer Between Circular Orbits

Path: `examples/quick/01_two_impulse_free_time.py`

Run:

```bash
python examples/quick/01_two_impulse_free_time.py
```

Feature focus:

- Single transfer phase.
- Two impulsive maneuvers.
- Free final time within bounds.
- Circular-orbit target with an analytical Hohmann reference used in tests.

Expected output:

- Printed solution summary.
- `traj_quick_hohmann_transfer.html`.

Screenshot placeholder: `docs/assets/screenshots/quick-01-hohmann-transfer.png`.

## 02: Precoast Plus Circular-Orbit Transfer

Path: `examples/quick/02_two_impulse_precoast_impulsive_link.py`

Run:

```bash
python examples/quick/02_two_impulse_precoast_impulsive_link.py
```

Feature focus:

- Bounded precoast before the transfer.
- Impulsive link between precoast and rendezvous phases.
- Same circular transfer family as example 01, with added departure-timing freedom.

Expected output:

- Printed solution summary.
- `traj_quick_precoast_circular_transfer.html`.

Screenshot placeholder: `docs/assets/screenshots/quick-02-precoast-circular-transfer.png`.

## 03: Delta-v Versus Time Trade

Path: `examples/quick/03_time_tradeoff.py`

Run:

```bash
python examples/quick/03_time_tradeoff.py
```

Feature focus:

- Same transfer geometry solved with two objective settings.
- `w_time=0.0` minimizes delta-v only.
- `w_time>0.0` biases the optimizer toward shorter transfer time.

Expected output:

- Two printed solution summaries.
- `traj_quick_time_tradeoff_dv_only.html`.
- `traj_quick_time_tradeoff_dv_plus_time.html`.

Screenshot placeholders:

- `docs/assets/screenshots/quick-03-dv-only.png`.
- `docs/assets/screenshots/quick-03-dv-plus-time.png`.

## 04: Batch Target-Radius Sweep

Path: `examples/quick/04_batch_targets.py`

Run:

```bash
python examples/quick/04_batch_targets.py
```

Feature focus:

- Small parameter sweep over circular target radii.
- One mission per target.
- Best converged case selected by total delta-v.

Expected output:

- Per-case printed summary lines.
- Best-case summary.
- `traj_quick_batch_best.html`.

Screenshot placeholder: `docs/assets/screenshots/quick-04-batch-best.png`.

## When to Leave the Quick API

Use the composable API when the mission needs explicit phase boundaries,
continuous links, finite burns, perturbations, path constraints, or terminal
orbital-element targeting.
