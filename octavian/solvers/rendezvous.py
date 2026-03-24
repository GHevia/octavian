"""Impulsive rendezvous solvers built on ASSET's UpdatedInterface.

Key correctness notes (from ASSET source/examples):
- **Time is not a state**. Time is the phase time variable (ODEArguments.TVar()) and
  appears in the phase variable stack after X: for XVars=6, the time index is 6.
- Use **Vgroups** in the ODE ("R", "V", "t"/"time") so constraints can refer
  to semantic groups (see `examples/UpdatedInterface/Delta3Launch.py` in ASSET).

This module keeps everything dimensional at the user API, and relies on ASSET auto-scaling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    import asset_asrl as ast  # type: ignore
except Exception:  # pragma: no cover
    ast = None  # type: ignore

from ..astro.kepler import estimate_orbital_period_s, kepler_dense_guess, propagate_cartesian_rv
from ..astro.lambert import LambertSeed, select_best_lambert_seed
from ..astro.types import as_vec3
from ..astro.units import default_units
from ..dynamics import TwoBodyECI
from ..specs import TwoImpulseFreeTimeSpec, TwoImpulsePreCoastSpec
from ..types import Maneuver
from .options import SolverOptions

if ast is not None:  # pragma: no cover
    vf = ast.VectorFunctions
    oc = ast.OptimalControl
    Tmodes = oc.TranscriptionModes
else:  # pragma: no cover
    vf = None  # type: ignore
    oc = None  # type: ignore
    Tmodes = None  # type: ignore

TrajArray = NDArray[np.float64]


def _require_asset() -> None:
    """Raise a clear error if ASSET is not installed."""
    if ast is None:
        raise RuntimeError(
            "asset_asrl is required for optimization solves. Install it (and its compiled dependencies) "
            "in your environment before calling octavian.solvers.*"
        )


@dataclass
class RendezvousResult:
    """Result returned by rendezvous solvers.

    Attributes:
        converged: Whether ASSET reported convergence.
        traj: Dense trajectory array. For the current impulsive problems this is
            typically ``[x0..x5, t]`` where ``t`` is the phase time variable.
        maneuvers: Maneuver markers derived from the solution.
        last_obj: Last objective value returned by ASSET.
        info: Free-form metadata (seed selection, bounds used, etc.).

    The result object is designed to be the *currency* of studies:
    it supports human-readable summaries and simple persistence.
    """

    converged: bool
    traj: TrajArray
    maneuvers: list[Maneuver] = field(default_factory=list)
    last_obj: float = float("nan")
    info: dict[str, Any] = field(default_factory=dict)

    def tf_s(self) -> float:
        """Return the final time of flight in seconds.

        Returns:
            The last trajectory time sample in seconds, or ``nan`` if the
            trajectory is empty.
        """
        if self.traj.size == 0:
            return float("nan")
        return float(self.traj[-1, -1])

    def total_dv_mps(self) -> float:
        """Return the total maneuver delta-v magnitude in meters per second.

        Returns:
            The sum of maneuver magnitudes. Returns ``0.0`` when no maneuvers
            are stored on the result.
        """
        if not self.maneuvers:
            return 0.0
        return float(sum(np.linalg.norm(m.dv_mps) for m in self.maneuvers))

    def summary(self) -> str:
        """Build a compact human-readable summary of the result.

        Returns:
            A multiline summary string containing convergence status, timing,
            delta-v, and selected metadata.
        """
        lines: list[str] = []
        status = "CONVERGED" if self.converged else "NOT CONVERGED"
        lines.append(f"Octavian result: {status}")
        lines.append(f"  tf: {self.tf_s():.3f} s")
        lines.append(f"  total dv: {self.total_dv_mps():.6f} m/s")
        if np.isfinite(self.last_obj):
            lines.append(f"  last objective: {self.last_obj:.6g}")
        if self.maneuvers:
            lines.append("  maneuvers:")
            for m in self.maneuvers:
                dv = float(np.linalg.norm(m.dv_mps))
                lines.append(f"    - {m.name}: t={m.t_s:.3f} s | |dv|={dv:.6f} m/s")
        # show a couple of useful info keys, without dumping everything
        for k in ("seed", "nrev", "precoast_t1_s", "study_index"):
            if k in self.info:
                lines.append(f"  {k}: {self.info[k]!r}")
        return "\n".join(lines)

    def to_npz(self, path: str | Path) -> None:
        """Save this result to a compressed ``.npz`` file.

        Args:
            path: Output file path.

        Notes:
            The archive stores the trajectory, convergence flag, objective
            value, maneuver fields, and JSON-serialized ``info`` metadata.
        """
        import json as _json
        from pathlib import Path as _Path

        p = _Path(path)
        r = (
            np.asarray([m.r_m for m in self.maneuvers], dtype=float)
            if self.maneuvers
            else np.empty((0, 3), dtype=float)
        )
        dv = (
            np.asarray([m.dv_mps for m in self.maneuvers], dtype=float)
            if self.maneuvers
            else np.empty((0, 3), dtype=float)
        )
        t = (
            np.asarray([m.t_s for m in self.maneuvers], dtype=float)
            if self.maneuvers
            else np.empty((0,), dtype=float)
        )
        names = (
            np.asarray([m.name for m in self.maneuvers], dtype=object)
            if self.maneuvers
            else np.empty((0,), dtype=object)
        )
        info_json = _json.dumps(self.info, ensure_ascii=False)
        np.savez_compressed(
            p,
            traj=np.asarray(self.traj, dtype=float),
            converged=np.asarray(int(bool(self.converged)), dtype=np.int8),
            last_obj=np.asarray(float(self.last_obj), dtype=float),
            maneuver_r_m=r,
            maneuver_dv_mps=dv,
            maneuver_t_s=t,
            maneuver_name=names,
            info_json=np.asarray(info_json, dtype=object),
        )

    @classmethod
    def from_npz(cls, path: str | Path) -> RendezvousResult:
        """Load a result from a compressed ``.npz`` file.

        Args:
            path: Path previously written by :meth:`to_npz`.

        Returns:
            The reconstructed result object.
        """
        import json as _json
        from pathlib import Path as _Path

        p = _Path(path)
        data = np.load(p, allow_pickle=True)
        traj = np.asarray(data["traj"], dtype=float)
        converged = bool(int(data["converged"]))
        last_obj = float(data["last_obj"])
        info_json = str(data["info_json"].item())
        info = _json.loads(info_json) if info_json else {}
        r = np.asarray(data["maneuver_r_m"], dtype=float)
        dv = np.asarray(data["maneuver_dv_mps"], dtype=float)
        t = np.asarray(data["maneuver_t_s"], dtype=float)
        names = np.asarray(data["maneuver_name"], dtype=object)
        maneuvers = [
            Maneuver(r_m=r[i], t_s=float(t[i]), dv_mps=dv[i], name=str(names[i]))
            for i in range(len(t))
        ]
        return cls(
            converged=converged, traj=traj, maneuvers=maneuvers, last_obj=last_obj, info=info
        )

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize summary metadata to JSON.

        Args:
            indent: Optional indentation level passed to ``json.dumps``.

        Returns:
            A JSON string containing summary fields, maneuvers, and metadata.
        """
        import json as _json

        obj = {
            "converged": bool(self.converged),
            "last_obj": float(self.last_obj),
            "tf_s": self.tf_s(),
            "total_dv_mps": self.total_dv_mps(),
            "maneuvers": [
                {
                    "name": m.name,
                    "t_s": float(m.t_s),
                    "r_m": [float(x) for x in m.r_m],
                    "dv_mps": [float(x) for x in m.dv_mps],
                }
                for m in self.maneuvers
            ],
            "info": self.info,
        }
        return _json.dumps(obj, indent=indent, ensure_ascii=False)


def solve(
    spec: TwoImpulseFreeTimeSpec | TwoImpulsePreCoastSpec,
    *,
    options: SolverOptions | None = None,
) -> RendezvousResult:
    """Solve a rendezvous specification with the matching built-in solver.

    Args:
        spec: Rendezvous problem specification.
        options: Optional solver configuration overrides.

    Returns:
        The solver result for the provided specification.

    Raises:
        TypeError: If ``spec`` is not a supported rendezvous spec type.
    """
    if isinstance(spec, TwoImpulseFreeTimeSpec):
        return solve_two_impulse_free_time(spec, options=options)
    if isinstance(spec, TwoImpulsePreCoastSpec):
        return solve_two_impulse_precoast(spec, options=options)
    raise TypeError(f"Unsupported spec type: {type(spec).__name__}")


def solve_two_impulse_free_time(
    spec: TwoImpulseFreeTimeSpec,
    *,
    options: SolverOptions | None = None,
) -> RendezvousResult:
    """Solve a two-impulse rendezvous with bounded free final time.

    Args:
        spec: Single-phase rendezvous specification.
        options: Optional solver configuration overrides.

    Returns:
        The optimized rendezvous result.

    Raises:
        RuntimeError: If ASSET is not installed.
        ValueError: If the time bounds are invalid.
    """
    _require_asset()
    tfmin, tfmax = map(float, spec.tf_bounds_s)
    if not (tfmin > 0.0 and tfmax > tfmin):
        raise ValueError("tf_bounds_s must satisfy 0 < tfmin < tfmax")

    r_unit, v_unit, t_unit = default_units(spec)

    ode = TwoBodyECI(mu_m3ps2=float(spec.mu_m3ps2))
    t0 = 0.0

    seed = select_best_lambert_seed(
        r0_m=as_vec3(spec.x0.r_m),
        rf_m=as_vec3(spec.xf.r_m),
        v0_mps=as_vec3(spec.x0.v_mps),
        vf_mps=as_vec3(spec.xf.v_mps),
        mu_m3ps2=float(spec.mu_m3ps2),
        tmin_s=tfmin,
        tmax_s=tfmax,
        n_tofs=int(spec.lambert_grid_size),
        nrevs=tuple(int(n) for n in spec.nrevs_to_try),
    )

    tf_guess = float(spec.tf_guess_s) if spec.tf_guess_s is not None else float(seed.tof_s)
    tf_guess = min(max(tf_guess, tfmin), tfmax)

    # Initial guess: Kepler propagation from Lambert departure velocity
    ig = kepler_dense_guess(
        r0_m=as_vec3(spec.x0.r_m),
        v0_mps=as_vec3(seed.v1_mps),
        t0_s=t0,
        tf_s=tf_guess,
        npts=int(spec.nsegs) + 1,
        mu_m3ps2=float(spec.mu_m3ps2),
    )

    phase = ode.phase(Tmodes.LGL3, ig, int(spec.nsegs))

    # Constraints: fix start R and time, fix end R, bound end time (time index = 6)
    phase.addBoundaryValue("Front", ["R", "t"], np.hstack([as_vec3(spec.x0.r_m), [t0]]))
    phase.addBoundaryValue("Back", ["R"], as_vec3(spec.xf.r_m))

    # Back time bound
    phase.addLUVarBound("Back", "time", tfmin, tfmax)

    # Optional: if boundary impulses are disabled, fix boundary velocity to the provided boundary state.
    if not bool(getattr(spec, "dv_front", True)):
        phase.addBoundaryValue("Front", ["V"], as_vec3(spec.x0.v_mps))
    if not bool(getattr(spec, "dv_back", True)):
        phase.addBoundaryValue("Back", ["V"], as_vec3(spec.xf.v_mps))

    # Objectives (explicit):
    #  - If minimize_dv: include Δv penalties at enabled impulsive boundaries.
    #  - If w_time != 0: include w_time * tf at Back.
    v0 = as_vec3(spec.x0.v_mps)
    vf_ = as_vec3(spec.xf.v_mps)

    vel_obj_scale = 1.0 / float(v_unit)

    if bool(getattr(spec, "minimize_dv", True)):
        w_dv = float(getattr(spec, "dv_weight", 1.0) or 1.0)

        if bool(getattr(spec, "dv_front", True)):
            a = vf.Arguments(3)
            dv1 = vf.sqrt((a - v0).dot(a - v0))
            phase.addStateObjective("Front", w_dv * dv1, [3, 4, 5], [], [], AutoScale=vel_obj_scale)

        if bool(getattr(spec, "dv_back", True)):
            b = vf.Arguments(3)
            dv2 = vf.sqrt((vf_ - b).dot(vf_ - b))
            phase.addStateObjective("Back", w_dv * dv2, [3, 4, 5], [], [], AutoScale=vel_obj_scale)

    phase.addLowerDeltaTimeBound(0.1)

    if float(spec.w_time) != 0.0:
        at = vf.Arguments(1).tolist()[0]
        phase.addStateObjective(
            "Back", float(spec.w_time) * at, [6], [], [], AutoScale=1.0 / float(t_unit)
        )

    ocp = oc.OptimalControlProblem()
    ocp.addPhase(phase)

    opts = options or SolverOptions()
    ocp.optimizer.PrintLevel = int(opts.print_level)
    ocp.optimizer.MaxLSIters = int(opts.max_ls_iters)
    ocp.optimizer.set_QPOrderingMode(str(opts.qp_ordering_mode))

    phase.setAutoScaling(bool(opts.enable_auto_scaling))
    phase.setUnits(R=r_unit, V=v_unit, t=t_unit)
    phase.setAdaptiveMesh(bool(opts.enable_adaptive_mesh))
    ocp.setAutoScaling(True, True)
    ocp.setAdaptiveMesh(True)
    ocp.PrintMeshInfo = False

    converged = ocp.solve_optimize_solve()

    traj = np.asarray(phase.returnTraj(), dtype=np.float64)
    # traj columns for XVars=6,UVars=0 typically: [x0..x5, t]
    v_front = traj[0, 3:6]
    v_back = traj[-1, 3:6]

    dv1 = v_front - v0
    dv2 = vf_ - v_back

    maneuvers = [
        Maneuver(r_m=traj[0, 0:3], t_s=float(traj[0, 6]), dv_mps=dv1, name="Δv1 (start)"),
        Maneuver(r_m=traj[-1, 0:3], t_s=float(traj[-1, 6]), dv_mps=dv2, name="Δv2 (end)"),
    ]

    return RendezvousResult(
        converged=converged,
        traj=traj,
        maneuvers=maneuvers,
        last_obj=float(getattr(ocp.optimizer, "LastObjVal", np.nan)),
        info={
            "tf_sol_s": float(traj[-1, 6]),
            "r_unit_m": r_unit,
            "v_unit_mps": v_unit,
            "t_unit_s": t_unit,
            "seed_tof_s": seed.tof_s,
            "seed_longway": seed.longway,
            "seed_nrev": seed.nrev,
            "seed_rightbranch": seed.rightbranch,
            "seed_total_dv_mps": seed.total_dv_mps,
        },
    )


def solve_two_impulse_precoast(
    spec: TwoImpulsePreCoastSpec,
    *,
    options: SolverOptions | None = None,
) -> RendezvousResult:
    """Solve a rendezvous with a variable precoast before the transfer.

    Args:
        spec: Precoast rendezvous specification.
        options: Optional solver configuration overrides.

    Returns:
        The optimized rendezvous result.

    Raises:
        RuntimeError: If ASSET is not installed or no feasible seed is found.
        ValueError: If the precoast or transfer time bounds are invalid.
    """
    _require_asset()
    t1min, t1max = map(float, spec.t1_bounds_s)
    tfmin, tfmax = map(float, spec.tf_bounds_s)

    if not (t1min >= 0.0 and t1max > t1min):
        raise ValueError("t1_bounds_s must satisfy 0 <= t1min < t1max")
    if not (tfmin > 0.0 and tfmax > tfmin):
        raise ValueError("tf_bounds_s must satisfy 0 < tfmin < tfmax")
    if tfmax <= t1min:
        raise ValueError("tf must be after t1: require tfmax > t1min")

    r_unit, v_unit, t_unit = default_units(spec)
    _require_asset()

    mu = float(spec.mu_m3ps2)
    ode = TwoBodyECI(mu_m3ps2=mu)
    t0 = 0.0

    # --- Candidate t1 sweep using Kepler propagation
    n_t1 = max(int(spec.precoast_grid_size), 2)
    t1_candidates = np.linspace(t1min, t1max, n_t1)

    if bool(spec.limit_precoast_to_one_period):
        T0 = estimate_orbital_period_s(spec.x0.r_m, spec.x0.v_mps, mu)
        if T0 is not None:
            span = t1max - t1min
            if span > 1.5 * T0:
                t1_candidates = np.linspace(t1min, min(t1min + T0, t1max), n_t1)

    rv0 = np.hstack([as_vec3(spec.x0.r_m), as_vec3(spec.x0.v_mps)])

    best = None
    for t1_try in t1_candidates:
        try:
            rv1 = propagate_cartesian_rv(rv0, float(t1_try - t0), mu)
        except Exception:
            continue
        r1 = rv1[0:3]
        v1_minus = rv1[3:6]

        dtmin = max(1.0, tfmin - float(t1_try))
        dtmax = max(dtmin + 1.0, tfmax - float(t1_try))
        if dtmax <= dtmin:
            continue

        seed = select_best_lambert_seed(
            r0_m=as_vec3(r1),
            rf_m=as_vec3(spec.xf.r_m),
            v0_mps=as_vec3(v1_minus),
            vf_mps=as_vec3(spec.xf.v_mps),
            mu_m3ps2=mu,
            tmin_s=float(dtmin),
            tmax_s=float(dtmax),
            n_tofs=int(spec.lambert_grid_size),
            nrevs=tuple(int(n) for n in spec.nrevs_to_try),
        )

        dv1 = as_vec3(seed.v1_mps) - as_vec3(v1_minus)
        dv2 = as_vec3(spec.xf.v_mps) - as_vec3(seed.v2_mps)
        score = float(np.linalg.norm(dv1) + np.linalg.norm(dv2))
        if best is None or score < best["score"]:
            best = {"t1": float(t1_try), "rv1": rv1, "seed": seed, "score": score}

    if best is None:
        raise RuntimeError("Failed to find any feasible pre-coast + Lambert seed.")

    t1_guess = float(best["t1"])
    rv1_guess = np.asarray(best["rv1"], dtype=float).reshape(6)
    r1_guess = rv1_guess[0:3]
    rv1_guess[3:6]
    seed: LambertSeed = best["seed"]

    tf_guess = float(t1_guess + seed.tof_s)
    v1_plus_guess = as_vec3(seed.v1_mps)

    # Dense guesses for both phases
    ig0 = kepler_dense_guess(
        r0_m=as_vec3(spec.x0.r_m),
        v0_mps=as_vec3(spec.x0.v_mps),
        t0_s=t0,
        tf_s=t1_guess,
        npts=int(spec.nsegs_precoast) + 1,
        mu_m3ps2=mu,
    )
    ig1 = kepler_dense_guess(
        r0_m=as_vec3(r1_guess),
        v0_mps=v1_plus_guess,
        t0_s=t1_guess,
        tf_s=tf_guess,
        npts=int(spec.nsegs_transfer) + 1,
        mu_m3ps2=mu,
    )

    phase0 = ode.phase(Tmodes.LGL3, ig0, int(spec.nsegs_precoast))
    phase1 = ode.phase(Tmodes.LGL3, ig1, int(spec.nsegs_transfer))

    # Phase 0 boundary: fix full state and t0, bound t1 (time index 6)

    if bool(getattr(spec, "dv_front", False)):
        phase0.addBoundaryValue("Front", ["R", "t"], np.hstack([as_vec3(spec.x0.r_m), [t0]]))
    else:
        phase0.addBoundaryValue(
            "Front",
            ["R", "V", "t"],
            np.hstack([as_vec3(spec.x0.r_m), as_vec3(spec.x0.v_mps), [t0]]),
        )
    try:
        phase0.addLUVarBound("Back", "time", t1min, t1max)
    except Exception:
        phase0.addLUVarBound("Back", 6, t1min, t1max)
    phase0.addLowerDeltaTimeBound(float(spec.min_dt_precoast_s))

    # Phase 1 boundary: fix final position, bound tf
    phase1.addBoundaryValue("Back", ["R"], as_vec3(spec.xf.r_m))
    # Optional: if terminal impulse is disabled, fix final velocity.
    if not bool(getattr(spec, "dv_back", True)):
        phase1.addBoundaryValue("Back", ["V"], as_vec3(spec.xf.v_mps))
    phase1.addLUVarBound("Back", "time", tfmin, tfmax)
    phase1.addLowerDeltaTimeBound(float(spec.min_dt_transfer_s))

    # phase0

    ocp = oc.OptimalControlProblem()
    ocp.addPhase(phase0)
    ocp.addPhase(phase1)

    # Link constraints: choose continuity groups based on spec.link_kind
    if str(getattr(spec, "link_kind", "impulsive")).lower() == "continuous":
        ocp.addForwardLinkEqualCon(phase0, phase1, ["R", "V", "t"])
    else:
        ocp.addForwardLinkEqualCon(phase0, phase1, ["R", "t"])

    # Objectives (explicit):

    vel_obj_scale = 1.0 / float(v_unit)

    if bool(getattr(spec, "minimize_dv", True)):
        w_dv = float(getattr(spec, "dv_weight", 1.0) or 1.0)

        # Optional initial Δv0 at phase0 Front
        if bool(getattr(spec, "dv_front", False)):
            v0 = as_vec3(spec.x0.v_mps)
            a0 = vf.Arguments(3)
            dv0 = vf.sqrt((a0 - v0).dot(a0 - v0))
            phase0.addStateObjective(
                "Front", w_dv * dv0, [3, 4, 5], [], [], AutoScale=vel_obj_scale
            )

        # Link Δv1 only for impulsive links
        if (
            bool(getattr(spec, "dv_link", True))
            and str(getattr(spec, "link_kind", "impulsive")).lower() != "continuous"
        ):
            a = vf.Arguments(6)
            v_minus = a.head(3)
            v_plus = a.segment(3, 3)
            dv = v_plus - v_minus
            dv1 = vf.sqrt(dv.dot(dv))
            ocp.addLinkObjective(
                w_dv * dv1,
                phase0,
                "Back",
                [3, 4, 5],
                [],
                [],
                phase1,
                "Front",
                [3, 4, 5],
                [],
                [],
                [],
                AutoScale=vel_obj_scale,
            )

        # Terminal Δv2
        if bool(getattr(spec, "dv_back", True)):
            vf_ = as_vec3(spec.xf.v_mps)
            b = vf.Arguments(3)
            dv2 = vf.sqrt((vf_ - b).dot(vf_ - b))
            phase1.addStateObjective("Back", w_dv * dv2, [3, 4, 5], [], [], AutoScale=vel_obj_scale)

    if float(spec.w_time) != 0.0:
        at = vf.Arguments(1).tolist()[0]
        phase1.addStateObjective(
            "Back", float(spec.w_time) * at, [6], [], [], AutoScale=1.0 / float(t_unit)
        )

    # ocp.optimizer.set_EContol(tol)
    ocp.optimizer.set_AccKKTtol(1e-6)

    opts = options or SolverOptions()
    ocp.optimizer.PrintLevel = int(opts.print_level)
    ocp.optimizer.MaxLSIters = int(opts.max_ls_iters)
    ocp.optimizer.set_QPOrderingMode(str(opts.qp_ordering_mode))

    for phase in [phase1, phase0]:
        phase.setAutoScaling(bool(opts.enable_auto_scaling))
        phase.setUnits(R=r_unit, V=v_unit, t=t_unit)
        phase.setAdaptiveMesh(bool(opts.enable_adaptive_mesh))

    for ph in (phase0, phase1):
        ph.setAutoScaling(True)
        ph.setUnits(R=r_unit, V=v_unit, t=t_unit)
        ph.setAdaptiveMesh(True)

    ocp.setAutoScaling(True, True)
    ocp.setAdaptiveMesh(True)
    ocp.PrintMeshInfo = False

    converged = ocp.solve()
    converged = ocp.optimize_solve()

    traj0 = np.asarray(phase0.returnTraj(), dtype=np.float64)
    traj1 = np.asarray(phase1.returnTraj(), dtype=np.float64)
    traj = np.vstack([traj0, traj1[1:, :]])

    v1m = traj0[-1, 3:6]
    v1p = traj1[0, 3:6]
    dv1 = v1p - v1m

    vf_target = as_vec3(spec.xf.v_mps)

    v2m = traj1[-1, 3:6]
    dv2 = vf_target - v2m

    maneuvers = [
        Maneuver(r_m=traj0[-1, 0:3], t_s=float(traj0[-1, 6]), dv_mps=dv1, name="Δv1 (link)"),
    ]
    if bool(getattr(spec, "dv_back", True)):
        maneuvers.append(
            Maneuver(r_m=traj1[-1, 0:3], t_s=float(traj1[-1, 6]), dv_mps=dv2, name="Δv2 (end)")
        )

    return RendezvousResult(
        converged=converged,
        traj=traj,
        maneuvers=maneuvers,
        last_obj=float(getattr(ocp.optimizer, "LastObjVal", np.nan)),
        info={
            "t1_sol_s": float(traj0[-1, 6]),
            "tf_sol_s": float(traj1[-1, 6]),
            "r_unit_m": r_unit,
            "v_unit_mps": v_unit,
            "t_unit_s": t_unit,
            "seed_dt_s": seed.tof_s,
            "seed_longway": seed.longway,
            "seed_nrev": seed.nrev,
            "seed_rightbranch": seed.rightbranch,
            "seed_total_dv_mps": seed.total_dv_mps,
            "precoast_seed_score_mps": float(best["score"]),
        },
    )
