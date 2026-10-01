"""Gravity normalization, pole behavior, derivatives, and ASSET EOM regressions."""

import importlib.util
import math
from dataclasses import replace

import numpy as np
import pytest
from numpy.polynomial.legendre import Legendre

from octavian import (
    Dynamics,
    Mission,
    Perturbations,
    Phase,
    Spacecraft,
    SphericalHarmonics,
    propagate,
    state,
)
from octavian._asset import ast, vf
from octavian.bodies import EARTH
from octavian.config.environment import build_perturbations
from octavian.dynamics import PerturbedECI, j2_acceleration_components
from octavian.runner import _is_composable_mission

MU = EARTH.mu_m3ps2
RADIUS = EARTH.mean_radius_m


def field(degree=3, backend="python", **kwargs):
    rng = np.random.default_rng(481)
    c, s = (np.tril(rng.normal(0, 1e-6, (degree + 1, degree + 1))) for _ in range(2))
    c[:2] = 0
    s[:2] = 0
    s[:, 0] = 0
    c[2, 0] = -EARTH.j2_coefficient / np.sqrt(5)
    return SphericalHarmonics(c, s, RADIUS, backend=backend, **kwargs)


def j2_field(backend="python"):
    c = np.zeros((3, 3))
    c[2, 0] = -EARTH.j2_coefficient / np.sqrt(5)
    return SphericalHarmonics(c, np.zeros_like(c), RADIUS, backend=backend)


def require_native():
    # A broken binary must fail the test; only an absent optional backend skips.
    if not any(
        importlib.util.find_spec(name) is not None
        for name in ("octavian.octavian_harmonics_native", "octavian_harmonics_native")
    ):
        pytest.skip("native backend is not installed")


@pytest.fixture(params=["python", "cpp"])
def backend(request):
    if request.param == "cpp":
        require_native()
    return request.param


@pytest.mark.parametrize("position", [[7e6, 1e6, 2e6], [0, 0, 7e6], [1e-8, 0, -7e6], [7e6, 0, 0]])
def test_j2_equivalence(position, backend):
    actual = j2_field(backend).acceleration(position, mu_m3ps2=MU)
    expected = j2_acceleration_components(
        position, mu_m3ps2=MU, radius_m=RADIUS, j2=EARTH.j2_coefficient
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-17)


def independent_potential(position, model):
    """Latitude/longitude Legendre definition, independent of solid harmonics."""
    radius = np.linalg.norm(position)
    latitude_sine = position[2] / radius
    longitude = np.arctan2(position[1], position[0])
    result = 0.0
    for n in range(2, model.degree + 1):
        for m in range(min(n, model.order) + 1):
            normalization = np.sqrt(
                (1 if m == 0 else 2) * (2 * n + 1) * math.factorial(n - m) / math.factorial(n + m)
            )
            associated = (1 - latitude_sine**2) ** (m / 2) * Legendre.basis(n).deriv(m)(
                latitude_sine
            )
            result += (
                (RADIUS / radius) ** n
                * normalization
                * associated
                * (
                    model.cosine[n][m] * np.cos(m * longitude)
                    + model.sine[n][m] * np.sin(m * longitude)
                )
            )
    return MU / radius * result


@pytest.mark.parametrize("degree,order", [(3, 0), (3, 3), (12, 7), (20, 20)])
def test_independent_legendre_reference(degree, order, backend):
    model = field(degree, backend, order=order)
    position = np.array([6.4e6, -2.2e6, 3.1e6])
    # Five-point gradient of the independent potential.
    steps = np.eye(3) * 20.0
    expected = np.array(
        [
            (
                independent_potential(position - 2 * d, model)
                - 8 * independent_potential(position - d, model)
                + 8 * independent_potential(position + d, model)
                - independent_potential(position + 2 * d, model)
            )
            / 240
            for d in steps
        ]
    )
    np.testing.assert_allclose(
        model.acceleration(position, mu_m3ps2=MU), expected, atol=2e-12, rtol=2e-9
    )


def test_rotation_and_reference_time(backend):
    model = field(
        5, backend, rotation_rate_radps=0.01, reference_angle_rad=0.4, reference_time_s=15
    )
    time = 37.0
    theta = 0.4 + 0.01 * (time - 15)
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    fixed = replace(model, rotation_rate_radps=0, reference_angle_rad=0)
    position = np.array([7e6, 2e6, -1e6])
    expected = rotation @ fixed.acceleration(rotation.T @ position, mu_m3ps2=MU)
    np.testing.assert_allclose(
        model.acceleration(position, time_s=time, mu_m3ps2=MU), expected, atol=1e-16
    )


def test_model_validation_and_config():
    model = field()
    with pytest.raises(ValueError, match="already includes"):
        Perturbations(j2=True, spherical_harmonics=model)
    for kwargs in [
        {"order": 4},
        {"degree": 1},
        {"degree": 2.5},
        {"backend": "auto"},
        {"reference_radius_m": -1},
        {"rotation_rate_radps": np.nan},
    ]:
        with pytest.raises(ValueError):
            replace(model, **kwargs)
    config = {
        "spherical_harmonics": {
            "cosine": model.cosine,
            "sine": model.sine,
            "reference_radius_m": RADIUS,
        }
    }
    assert build_perturbations(config, "perturbations").spherical_harmonics == replace(
        model, backend="cpp"
    )
    for constructor in [
        lambda: Dynamics.cwh(
            chief_orbit_radius_m=7e6, perturbations=Perturbations(spherical_harmonics=model)
        ),
        lambda: Dynamics.cr3bp(perturbations=Perturbations(spherical_harmonics=model)),
    ]:
        with pytest.raises(ValueError, match="perturbations"):
            constructor()


def test_asset_force_derivatives_and_time(backend):
    if ast is None:
        pytest.skip("ASSET unavailable")
    model = field(3, backend, rotation_rate_radps=0.003)
    args = vf.Arguments(4)
    # Dimensionless coordinates and time keep finite-difference checks conditioned.
    function = model.asset_acceleration(args.head(3) * RADIUS, args[3] * 100, mu_m3ps2=MU)
    point = np.array([1.1, -0.3, 0.4, 0.7])
    adjoint = np.array([0.3, -0.5, 0.9])
    expected = model.acceleration(point[:3] * RADIUS, time_s=point[3] * 100, mu_m3ps2=MU)
    np.testing.assert_allclose(function.compute(point), expected, atol=1e-16)
    steps = np.eye(4) * 1e-5
    jacobian = np.column_stack(
        [(function.compute(point + d) - function.compute(point - d)) / 2e-5 for d in steps]
    )
    hessian = np.column_stack(
        [
            (function.jacobian(point + d).T @ adjoint - function.jacobian(point - d).T @ adjoint)
            / 2e-5
            for d in steps
        ]
    )
    np.testing.assert_allclose(function.jacobian(point), jacobian, rtol=2e-7, atol=2e-11)
    np.testing.assert_allclose(
        function.adjointhessian(point, adjoint), hessian, rtol=2e-7, atol=2e-10
    )


def test_asset_integration_matches_numerical_propagation(backend):
    if ast is None:
        pytest.skip("ASSET unavailable")
    model = field(3, backend, rotation_rate_radps=7.292115e-5)
    initial = state([7e6, 0, 1e5], [0, 7400, 800])
    perturbations = Perturbations(spherical_harmonics=model)
    ode = PerturbedECI(mu_m3ps2=MU, spherical_harmonics=model)
    integrator = ode.integrator(10.0)
    integrator.setAbsTol(1e-7)  # SI states: avoid chasing sub-roundoff position errors.
    result = integrator.integrate(np.r_[initial.r_m, initial.v_mps, 0.0], 300)
    reference = propagate.inertial(initial, [0, 300], perturbations=perturbations, max_step_s=1)
    np.testing.assert_allclose(result[:6], reference[-1, :6], rtol=0, atol=1e-3)
    phase = Phase(
        name="harmonics",
        mode="coast",
        spacecraft=Spacecraft(dry_mass_kg=100),
        dynamics=Dynamics(perturbations=perturbations),
        initial_state=initial,
        tof_bounds_s=(300, 300),
    )
    assert _is_composable_mission(Mission(phases=[phase]))


@pytest.mark.parametrize("degree", [50, 100])
def test_native_high_degree_derivatives_and_poles(degree):
    require_native()
    model = field(degree, "cpp")
    for position in [[0, 0, 7e6], [0, 0, -7e6], [7e6, 1e6, -2e6]]:
        np.testing.assert_allclose(
            model.acceleration(position, mu_m3ps2=MU),
            replace(model, backend="python").acceleration(position, mu_m3ps2=MU),
            rtol=2e-12,
            atol=2e-16,
        )
        point = np.asarray(position) / RADIUS
        fun = model._native_function
        adj = np.array([1.0, 2.0, 3.0])
        delta = np.eye(3) * 1e-5
        jac = np.column_stack(
            [(fun.compute(point + d) - fun.compute(point - d)) / 2e-5 for d in delta]
        )
        hess = np.column_stack(
            [
                (fun.jacobian(point + d).T @ adj - fun.jacobian(point - d).T @ adj) / 2e-5
                for d in delta
            ]
        )
        np.testing.assert_allclose(fun.jacobian(point), jac, rtol=2e-6, atol=2e-10)
        np.testing.assert_allclose(fun.adjointhessian(point, adj), hess, rtol=2e-6, atol=2e-9)


def test_harmonic_mission_optimization(backend):
    from octavian import constraints, objectives, two_burn_rendezvous, variables
    from octavian.solvers import SolverOptions

    if ast is None:
        pytest.skip("ASSET unavailable")
    model = field(12 if backend == "cpp" else 3, backend, rotation_rate_radps=7.292115e-5)
    initial = state([7e6, 0, 0], [0, np.sqrt(MU / 7e6), 0])
    target = state([0, 8e6, 0], [-np.sqrt(MU / 8e6), 0, 0])
    mission = two_burn_rendezvous(
        initial,
        target,
        tf_bounds_s=(1000, 2500),
        nsegs=24,
        lambert_grid_size=12,
        nrevs_to_try=(0,),
        solver_options=SolverOptions(
            print_level=0, enable_adaptive_mesh=False, asset_threads=(2, 2)
        ),
    )
    mission.phases[0].dynamics = Dynamics(perturbations=Perturbations(spherical_harmonics=model))
    mission.phases[0].constraints = [
        constraints.state(initial, where="Front"),
        constraints.state(target, where="Back"),
    ]
    mission.phases[0].variables = [
        variables.impulsive_delta_v(at="Front"),
        variables.impulsive_delta_v(at="Back"),
    ]
    mission.objectives = [objectives.minimize_total_delta_v()]
    solution = mission.solve(solver_options=mission.solver_options)
    assert solution.result is not None
    assert solution.result.converged
    trajectory = solution.result.traj
    np.testing.assert_allclose(trajectory[-1, :3], target.r_m, atol=0.01, rtol=0)
    # Independently reintegrate the optimized post-departure state.
    ode = PerturbedECI(mu_m3ps2=MU, spherical_harmonics=model)
    integrator = ode.integrator(10.0)
    integrator.setAbsTol(1e-7)
    end = integrator.integrate(trajectory[0, :7], trajectory[-1, 6])
    np.testing.assert_allclose(end[:3], trajectory[-1, :3], atol=1, rtol=0)


@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("powered,carries_mass", [(False, False), (False, True), (True, False)])
def test_phase_compilation_applies_harmonics_to_all_cartesian_eoms(
    backend, relative, powered, carries_mass
):
    from octavian import Thruster
    from octavian.solvers.compiler.phase_compiler import ode_for_phase

    if ast is None:
        pytest.skip("ASSET unavailable")
    chief = state([7e6, 0, 1e5], [0, 7400, 800])
    spacecraft = Spacecraft(
        dry_mass_kg=100, thrusters=[Thruster(thrust_N=1, isp_s=300, propellant_mass_kg=10)]
    )
    states = np.r_[chief.r_m, chief.v_mps]
    if relative:
        states = np.r_[states, states + np.array([100, 200, 0, 0, 0, 0])]
    if powered or carries_mass:
        states = np.r_[states, 110.0]
    row = np.r_[states, 100.0]
    if powered:
        row = np.r_[row, 0.1, 0.0, 0.0]
    outputs = []
    for perturbations in [
        Perturbations(j2=True),
        Perturbations(spherical_harmonics=j2_field(backend)),
    ]:
        dynamics = (
            Dynamics.relative(chief_initial_state_eci=chief, perturbations=perturbations)
            if relative
            else Dynamics(perturbations=perturbations)
        )
        phase = Phase(
            name="force_check",
            mode="finite_burn" if powered else "coast",
            spacecraft=spacecraft,
            dynamics=dynamics,
        )
        ode = ode_for_phase(phase, carries_mass=carries_mass)
        outputs.append(ode.vf().compute(row))
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=2e-14, atol=2e-14)


def test_zero_field_and_monopole_are_not_double_counted(backend):
    c = np.zeros((3, 3))
    c[0, 0] = 1
    model = SphericalHarmonics(c, np.zeros_like(c), RADIUS, backend=backend)
    np.testing.assert_array_equal(model.acceleration([7e6, 0, 0], mu_m3ps2=MU), np.zeros(3))
    if ast is not None:
        args = vf.Arguments(3)
        function = model.asset_acceleration(args, None, mu_m3ps2=MU)
        np.testing.assert_array_equal(function.compute([7e6, 0, 0]), np.zeros(3))


def test_missing_native_extension_has_actionable_error(monkeypatch):
    import sys

    if ast is None:
        pytest.skip("ASSET unavailable")
    monkeypatch.setitem(sys.modules, "octavian.octavian_harmonics_native", None)
    monkeypatch.setitem(sys.modules, "octavian_harmonics_native", None)
    with pytest.raises(RuntimeError, match="optional octavian-harmonics-native"):
        field(3, "cpp").acceleration([7e6, 0, 0], mu_m3ps2=MU)


def test_cpp_is_default_and_python_remains_explicit():
    model = j2_field()
    default = SphericalHarmonics(model.cosine, model.sine, RADIUS)
    assert default.backend == "cpp"
    assert model.backend == "python"


@pytest.mark.parametrize("degree", [20, 50, 100])
def test_native_sectoral_closed_form_and_laplace_equation(degree):
    """Independent x-axis formula for a single Cnn/Snn term at high degree."""
    require_native()
    c = np.zeros((degree + 1, degree + 1))
    s = np.zeros_like(c)
    c[degree, degree], s[degree, degree] = 1e-6, -2e-6
    model = SphericalHarmonics(c, s, RADIUS)
    radius = RADIUS * 1.01
    # Pbar_nn(0), from the factorial definition (no harmonic recurrence).
    pnn = math.exp(
        0.5 * (math.log(2 * (2 * degree + 1)) + math.lgamma(2 * degree + 1))
        - degree * math.log(2)
        - math.lgamma(degree + 1)
    )
    scale = MU / radius**2 * (RADIUS / radius) ** degree * pnn
    expected = scale * np.array([-(degree + 1) * c[-1, -1], degree * s[-1, -1], 0])
    np.testing.assert_allclose(
        model.acceleration([radius, 0, 0], mu_m3ps2=MU), expected, rtol=3e-12, atol=1e-16
    )
    # Exterior gravity is conservative and harmonic: symmetric gradient,
    # zero divergence. Check away from the special axis as well.
    jac = model._native_function.jacobian(np.array([1.01, 0.03, 0.02]))
    np.testing.assert_allclose(jac, jac.T, rtol=2e-12, atol=1e-15)
    assert abs(np.trace(jac)) < 2e-12 * np.linalg.norm(jac)
