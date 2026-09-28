"""Standard NUTS/window references, Gibbs dependencies and posterior stationarity."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import blackjax
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from blackjax.adaptation.staged_adaptation import build_schedule
from blackjax.mcmc import nuts
from scipy.integrate import quad, cumulative_trapezoid
from scipy.stats import norm

from bayesiancalibration.adaptation import nuts_window_adapter
from bayesiancalibration.gibbs import (
    update_discrepancy, update_branch_noise, update_spatial_mean,
    update_spatial_covariance, update_coefficient_variances, refresh_field_coefficients,
    sample_projected_coefficients,
)
from bayesiancalibration.gp import LibraryGP
from bayesiancalibration.mcmc import (
    nuts_gibbs_sweep, initialize_nuts_warmup, initialize_nuts_chain,
    run_nuts_warmup, run_fixed_nuts, validate_nuts_chain,
)
from bayesiancalibration.run import save_checkpoint, load_checkpoint
from bayesiancalibration.samplers.nuts import nuts_sweep
from bayesiancalibration.state import CalibrationState, SpatialPrior, ThetaStandardization
from bayesiancalibration.targets import CalibrationTarget
from bayesiancalibration.transforms import SiteCoordinates
import test_mcmc as reference


def scalar_posterior_start(bounded, count=12000):
    """Independent dense quadrature and posterior starts for theta/c_f.

    Freeze hyperparameters and vary theta jointly with refreshed coefficients.
    CDF interpolation error is bounded by grid refinement, not burn-in.
    This checks stationary posterior invariance, not convergence or mixing.
    """

    library = np.array([[-2.0], [0.0], [2.0]])
    standardization = ThetaStandardization.from_library(library)
    xs = np.asarray(standardization.to_standardized(library))[:, 0]
    F_s = np.array([[-0.55], [0.2], [1.2]])
    gp = LibraryGP.from_data(xs[:, None], F_s, [0.65])
    coordinates = SiteCoordinates.from_physical_bounds(
        standardization, **({"l": [-1.5], "u": [1.6]} if bounded else {})
    )
    prior = SpatialPrior.from_standardization(
        standardization, physical_center=[0.0], V_theta_0=[[1.0]],
        nu_theta_0=3.0, S_theta_0=[[1.0]],
    )
    target = CalibrationTarget.from_data(gp, coordinates, prior, [0.45], [[1.0]], [[0, 0]], (1,))
    C_ss = np.exp(-0.5 * ((xs[:, None] - xs[None, :]) / .65)**2)
    weights = np.linalg.solve(C_ss, F_s)[:, 0]

    def moments(x):
        cross = np.exp(-0.5 * ((np.asarray(x)[..., None] - xs) / .65)**2)
        m = cross @ weights
        solved = np.linalg.solve(C_ss, cross.reshape(-1, 3).T).T.reshape(cross.shape)
        V = .35 * (1 - np.sum(cross * solved, axis=-1))
        Vy = V + .06
        mean = m + V / Vy * (.45 - m - .04)
        variance = V * .06 / Vy
        density = norm.pdf(.45, m + .04, np.sqrt(Vy)) * norm.pdf(x, .12, np.sqrt(.45))
        return density, mean, variance

    limits = (-.75, .8) if bounded else (-np.inf, np.inf)
    Z = quad(lambda x: moments(x)[0], *limits, epsabs=1e-11)[0]
    def quantities(x):
        _, m, v = moments(x)
        return np.array([x, x*x, m, m*m+v, x*m, float(x <= .1)])
    expected = np.array([
        quad(lambda x: moments(x)[0]*quantities(x)[j], *limits,
             epsabs=1e-10, points=[.1] if bounded else None)[0]/Z
        for j in range(6)
    ])
    rng = np.random.default_rng(1210 + int(bounded))
    uniform = rng.uniform(size=count)
    starts = []
    for grid_size in (20001, 40001):
        grid = np.linspace(*((-.75, .8) if bounded else (-7, 7)), grid_size)
        cdf = cumulative_trapezoid(moments(grid)[0], grid, initial=0)
        starts.append(np.interp(uniform, cdf/cdf[-1], grid))
    np.testing.assert_allclose(starts[0], starts[1], rtol=0, atol=4e-6)
    theta = starts[1]
    _, m, v = moments(theta)
    c_f = rng.normal(m, np.sqrt(v))
    broadcast = lambda x: jnp.broadcast_to(jnp.asarray(x), (count, *np.shape(x)))
    state = CalibrationState(
        coordinates.theta_tilde_to_eta(jnp.asarray(theta).reshape(count, 1, 1)),
        jnp.asarray(c_f[:, None]), broadcast([.04]), broadcast([.06]),
        broadcast([.12]), broadcast([[.45]]), broadcast([.35]),
    )
    return target, state, expected


def posterior_summaries(target, state):
    theta = np.asarray(target.coordinates.eta_to_theta_tilde(state.eta)).reshape(-1)
    c = np.asarray(state.c_f).reshape(-1)
    return np.column_stack((theta, theta**2, c, c**2, theta*c, theta <= .1))


class NUTSTest(unittest.TestCase):
    assert_tree_equal = reference.GibbsSweepTest.assert_tree_equal
    collapsed = False

    def test_embedded_windows_match_standard_fixed_target_driver(self):
        mean = jnp.array([.2, -.3, .4])
        density = lambda x: -.5*jnp.sum((x-mean)**2 / jnp.array([.4, 1.2, .7]))
        initial = jnp.array([-.2, .1, .5])
        init, update, final = nuts_window_adapter()
        for length in (12, 60, 300):
            key = jax.random.key(1200 + length)
            driver = blackjax.window_adaptation(blackjax.nuts, density)
            expected, trace = jax.jit(lambda k, x: driver.run(k, x, num_steps=length))(
                key, initial
            )
            def step(carry, xs):
                state, adaptation = carry
                k, stage = xs
                s, info = nuts.build_kernel()(
                    k, state, density,
                    adaptation.step_size, adaptation.inverse_mass_matrix,
                )
                a = update(adaptation, stage, s.position, s.logdensity_grad, info.acceptance_rate)
                return (s, a), (s, info.acceptance_rate, a)
            (_, last), (states, rates, adaptations) = jax.jit(lambda k, x: jax.lax.scan(
                step, (nuts.init(x, density), init(x, 1.0)),
                (jax.random.split(k, length), build_schedule(length)),
            ))(key, initial)
            self.assert_tree_equal(states, trace.state)
            self.assert_tree_equal(adaptations, trace.adaptation_state)
            self.assert_tree_equal(rates, trace.info.acceptance_rate)
            step_size, mass = final(last)
            np.testing.assert_allclose(step_size, expected.parameters['step_size'], rtol=1e-12)
            np.testing.assert_allclose(mass, expected.parameters['inverse_mass_matrix'], rtol=1e-12)
            if length >= 20:
                self.assertFalse(np.allclose(mass, 1.0))

    def test_block_transition_matches_standard_kernel_with_current_conditioning(self):
        for bounded in (False, True):
            for loading in (False, True):
                target, state, _ = reference.make_fixture(bounded, loading)
                key = jax.random.key(1201)
                mass = jnp.array([.7, 1.3, .9, 1.5])
                def density(x):
                    if self.collapsed:
                        return target.theta_only_collapsed(
                            x.reshape(2, 2), state.delta, state.sigma_y2,
                            state.mu_theta, state.Sigma_theta, state.sigma_c2,
                        )
                    return target.theta_only_uncollapsed(
                        x.reshape(2, 2), state.c_f, state.mu_theta,
                        state.Sigma_theta, state.sigma_c2,
                    )
                expected, info = jax.jit(lambda: nuts.build_kernel()(
                    key, nuts.init(state.eta.reshape(-1), density), density, .14, mass, 5
                ))()
                actual, got = jax.jit(lambda: nuts_sweep(
                    key, target, state, .14, mass, max_num_doublings=5,
                    collapsed=self.collapsed,
                ))()
                self.assert_tree_equal(actual.reshape(-1), expected.position, True)
                self.assert_tree_equal(got.acceptance_rate, info.acceptance_rate, True)
                self.assertEqual(int(got.num_integration_steps), int(info.num_integration_steps))
                self.assertEqual(bool(got.reached_max_doublings),
                                 int(info.num_trajectory_expansions) >= 5)
                changed = state._replace(c_f=state.c_f + .15, sigma_c2=state.sigma_c2*.8)
                _, changed_info = nuts_sweep(
                    key, target, changed, .14, mass, max_num_doublings=5,
                    collapsed=self.collapsed,
                )
                self.assertNotAlmostEqual(float(changed_info.logdensity), float(got.logdensity))

    def test_outer_schedule_refresh_and_current_conditioning(self):
        for bounded, loading in ((False, False), (True, True)):
            target, s, _ = reference.make_fixture(bounded, loading)
            key = jax.random.key(1202)
            next_key, kd, ky, km, kS, kc, kt, kf = jax.random.split(key, 8)
            delta = update_discrepancy(kd, target, s.c_f, s.sigma_y2)
            noise = update_branch_noise(ky, target, s.c_f, delta)
            mean = update_spatial_mean(km, target, s.eta, s.Sigma_theta)
            covariance = update_spatial_covariance(kS, target, s.eta, mean)
            variances = update_coefficient_variances(kc, target, s.eta, s.c_f)
            conditioned = s._replace(delta=delta, sigma_y2=noise, mu_theta=mean,
                                     Sigma_theta=covariance, sigma_c2=variances)
            eta, info = nuts_sweep(
                kt, target, conditioned, .12, jnp.ones(4), max_num_doublings=4,
                collapsed=self.collapsed,
            )
            c_f = refresh_field_coefficients(kf, target, eta, delta, noise, variances)
            actual, actual_key, got = jax.jit(lambda: nuts_gibbs_sweep(
                key, target, s, .12, jnp.ones(4), max_num_doublings=4,
                collapsed=self.collapsed,
            ))()
            self.assert_tree_equal(actual, conditioned._replace(eta=eta, c_f=c_f))
            self.assert_tree_equal(got.theta, info)
            self.assert_tree_equal(jax.random.key_data(actual_key),
                                   jax.random.key_data(next_key), True)

    def test_window_restart_and_frozen_production(self):
        for bounded, loading, key in ((False, False, jax.random.key(1203)),
                                     (True, True, jax.random.PRNGKey(1204))):
            target, s, _ = reference.make_fixture(bounded, loading)
            initial = initialize_nuts_warmup(
                target, s, key, num_warmup=24, max_num_doublings=4,
                collapsed=self.collapsed,
            )
            self.assertEqual(initial.step_size, 1.0)
            self.assertEqual(initial.adaptation.target_accept, .8)
            all_chain, all_draws, all_info = run_nuts_warmup(target, initial)
            production, draws, diagnostics = run_fixed_nuts(target, all_chain, 2)
            part, first_draws, first_info = run_nuts_warmup(target, initial, 9)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'nuts.npz'
                for boundary in (initial, part, all_chain):
                    save_checkpoint(path, target, boundary)
                    restored, metadata = load_checkpoint(path, target)
                    self.assertEqual(metadata['sampler'],
                                     'collapsed_nuts' if self.collapsed else 'uncollapsed_nuts')
                    self.assertEqual(restored.collapsed, self.collapsed)
                    self.assert_tree_equal(restored.adaptation.state,
                                           boundary.adaptation.state, True)
                    self.assertEqual(restored.key.dtype, boundary.key.dtype)
                    if restored.phase == 'warmup':
                        finished, rest_draws, rest_info = run_nuts_warmup(target, restored)
                        offset = boundary.iteration
                        self.assert_tree_equal(
                            rest_draws, jax.tree.map(lambda x: x[offset:], all_draws), True
                        )
                        self.assert_tree_equal(
                            rest_info, jax.tree.map(lambda x: x[offset:], all_info), True
                        )
                    else:
                        finished = restored
                    final_chain, more, more_info = run_fixed_nuts(target, finished, 2)
                    self.assert_tree_equal(more, draws, True)
                    self.assert_tree_equal(more_info, diagnostics, True)
                    self.assert_tree_equal(final_chain.model_state, production.model_state, True)
                    self.assert_tree_equal(final_chain.adaptation.state,
                                           all_chain.adaptation.state, True)
                    np.testing.assert_array_equal(final_chain.inverse_mass_matrix,
                                                  all_chain.inverse_mass_matrix)
                    self.assertEqual(final_chain.step_size, all_chain.step_size)

    def test_invalid_configuration_and_checkpoint_payload(self):
        target, s, _ = reference.make_fixture()
        for kwargs in ({'num_warmup': 0}, {'num_warmup': True}, {'target_accept': 1.0},
                       {'initial_step_size': 0}, {'max_num_doublings': 0}):
            with self.assertRaises(ValueError):
                initialize_nuts_warmup(
                    target, s, jax.random.key(1205), collapsed=self.collapsed, **kwargs
                )
        chain = initialize_nuts_warmup(
            target, s, jax.random.key(1205), num_warmup=3, collapsed=self.collapsed
        )
        for changed in (replace(chain, phase='sampling'), replace(chain, iteration=1),
                        replace(chain, step_size=.5),
                        replace(chain, inverse_mass_matrix=-jnp.ones(4))):
            with self.assertRaises(ValueError):
                validate_nuts_chain(target, changed)
        with self.assertRaises(ValueError):
            run_fixed_nuts(target, chain, 1)
        with self.assertRaises(ValueError):
            run_nuts_warmup(target, chain, 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'nuts.npz'
            save_checkpoint(path, target, chain)
            with np.load(path, allow_pickle=False) as a:
                payload = {name: a[name] for name in a.files}
            payload['sampler.step_size'] = np.asarray(.5)
            np.savez_compressed(path, **payload)
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_checkpoint(path, target)

    def test_posterior_joint_moments_preserved_against_independent_quadrature(self):
        for bounded in (False, True):
            target, state, expected = scalar_posterior_start(bounded)
            def step(key, s):
                kt, kf = jax.random.split(key)
                eta, _ = nuts_sweep(
                    kt, target, s, .22, jnp.ones(1), max_num_doublings=4,
                    collapsed=self.collapsed,
                )
                m, _, V = target.gp.conditional_moments(
                    target.coordinates.eta_to_theta_tilde(eta), s.sigma_c2
                )
                d, noise = target.observation_arrays(s.delta, s.sigma_y2)
                c = sample_projected_coefficients(kf, target.y_tilde, m, V, target.R, d, noise)
                return s._replace(eta=eta, c_f=c)
            kernel = jax.jit(jax.vmap(step))
            for sweep in range(3):
                keys = jax.random.split(jax.random.key(1220 + sweep), state.eta.shape[0])
                state = kernel(keys, state)
                values = posterior_summaries(target, state)
                se = values.std(axis=0, ddof=1)/np.sqrt(len(values))
                np.testing.assert_array_less(np.abs(values.mean(axis=0)-expected), 6*se)


if __name__ == '__main__':
    unittest.main()
