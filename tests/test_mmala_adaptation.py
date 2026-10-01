"""MMALA Gibbs order, standard epsilon DA, frozen production and batches."""

import unittest
from dataclasses import replace

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp
import numpy as np
from blackjax.adaptation.step_size import dual_averaging_adaptation

from bayesiancalibration.gibbs import (
    update_discrepancy, update_branch_noise, update_spatial_mean,
    update_spatial_covariance, update_coefficient_variances, refresh_field_coefficients,
)
from bayesiancalibration.mcmc import (
    mmala_gibbs_sweep, initialize_mmala_warmup, initialize_mmala_chain,
    run_mmala_warmup, run_fixed_mmala, validate_mmala_chain,
)
from bayesiancalibration.samplers.mmala import update_collapsed_mmala, collapsed_mmala_metric
import test_mcmc as reference


class MMALAIntegrationTest(unittest.TestCase):
    assert_tree_equal = reference.GibbsSweepTest.assert_tree_equal

    def test_current_input_outer_schedule_and_coefficient_refresh(self):
        for bounded in (False, True):
            for loading in (False, True):
                target, s, _ = reference.make_fixture(bounded, loading)
                key = jax.random.key(1320)
                next_key, kd, ky, km, kS, kc, kt, kf = jax.random.split(key, 8)
                delta = update_discrepancy(kd, target, s.c_f, s.sigma_y2)
                noise = update_branch_noise(ky, target, s.c_f, delta)
                mean = update_spatial_mean(km, target, s.eta, s.Sigma_theta)
                covariance = update_spatial_covariance(kS, target, s.eta, mean)
                variances = update_coefficient_variances(kc, target, s.eta, s.c_f)
                eta, info = update_collapsed_mmala(
                    kt, target, s.eta, delta, noise, mean, covariance, variances, .35, .07
                )
                c = refresh_field_coefficients(kf, target, eta, delta, noise, variances)
                expected = s._replace(eta=eta, c_f=c, delta=delta, sigma_y2=noise,
                                      mu_theta=mean, Sigma_theta=covariance, sigma_c2=variances)
                actual, got_key, diagnostics = jax.jit(lambda: mmala_gibbs_sweep(
                    key, target, s, .35, .07
                ))()
                self.assert_tree_equal(actual, expected)
                self.assert_tree_equal(diagnostics.theta, info)
                np.testing.assert_array_equal(jax.random.key_data(got_key),
                                              jax.random.key_data(next_key))
                np.testing.assert_allclose(diagnostics.full_joint_logdensity,
                                           target.full_joint_uncollapsed(*expected))
                repeated, _, rejected = jax.jit(lambda: mmala_gibbs_sweep(
                    key, target, s, 1e150, .07
                ))()
                np.testing.assert_array_equal(repeated.eta, s.eta)
                self.assertFalse(np.array_equal(repeated.c_f, s.c_f))
                np.testing.assert_array_equal(rejected.theta.is_accepted, [False, False])

    def test_dynamic_epsilon_matches_standard_da_and_batches(self):
        for bounded, loading, key in ((False, False, jax.random.key(1321)),
                                     (True, True, jax.random.PRNGKey(1322))):
            target, s, _ = reference.make_fixture(bounded, loading)
            initial = initialize_mmala_warmup(
                target, s, key, num_warmup=5, epsilon=.35, epsilon_G=.07,
            )
            self.assertEqual(initial.adaptation.step_size.target_accept, .574)
            full, all_samples, all_info = run_mmala_warmup(target, initial)
            da_init, update, final = dual_averaging_adaptation(.574)
            update = jax.jit(update)
            da, epsilon, state, next_key = da_init(.35), .35, s, key
            pure = jax.jit(lambda k, s, e: mmala_gibbs_sweep(k, target, s, e, .07))
            for i in range(5):
                state, next_key, diagnostics = pure(next_key, state, epsilon)
                self.assert_tree_equal(state, jax.tree.map(lambda x: x[i], all_samples))
                self.assert_tree_equal(diagnostics, jax.tree.map(lambda x: x[i], all_info))
                da = update(da, jnp.mean(diagnostics.theta.acceptance_rate))
                epsilon = float(final(da) if i == 4 else jnp.exp(da.log_step_size))
            self.assertEqual(full.epsilon, epsilon)
            self.assert_tree_equal(full.adaptation.step_size.state, da)
            production, expected, info = run_fixed_mmala(target, full, 2)
            self.assertEqual(production.epsilon, full.epsilon)
            self.assertEqual(production.epsilon_G, full.epsilon_G)
            self.assert_tree_equal(production.adaptation.step_size.state,
                                   full.adaptation.step_size.state, True)
            # Fixed production tuning still evaluates a position-dependent G.
            first_metric = collapsed_mmala_metric(
                target, full.model_state.eta, 0, full.model_state.delta,
                full.model_state.sigma_y2, full.model_state.Sigma_theta,
                full.model_state.sigma_c2, full.epsilon_G,
            )
            last_metric = collapsed_mmala_metric(
                target, production.model_state.eta, 0, production.model_state.delta,
                production.model_state.sigma_y2, production.model_state.Sigma_theta,
                production.model_state.sigma_c2, production.epsilon_G,
            )
            self.assertFalse(np.allclose(first_metric, last_metric))
            part, _, _ = run_mmala_warmup(target, initial, 2)
            for boundary in (initial, part, full):
                self.assertEqual(boundary.key.dtype, initial.key.dtype)
                self.assertEqual(boundary.epsilon_G, .07)
                if boundary.phase == 'warmup':
                    finished, samples, diagnostics = run_mmala_warmup(target, boundary)
                    offset = boundary.iteration
                    self.assert_tree_equal(
                        samples, jax.tree.map(lambda x: x[offset:], all_samples), True
                    )
                    self.assert_tree_equal(
                        diagnostics, jax.tree.map(lambda x: x[offset:], all_info), True
                    )
                else:
                    finished = boundary
                completed, more, more_info = run_fixed_mmala(target, finished, 2)
                self.assert_tree_equal(more, expected, True)
                self.assert_tree_equal(more_info, info, True)
                self.assert_tree_equal(completed.model_state, production.model_state, True)
                np.testing.assert_array_equal(jax.random.key_data(completed.key),
                                              jax.random.key_data(production.key))
                self.assertEqual(completed.iteration, 7)

    def test_short_warmup_boundary_and_invalid_configuration(self):
        target, s, _ = reference.make_fixture()
        chain = initialize_mmala_warmup(
            target, s, jax.random.key(1323), num_warmup=1, epsilon=.4,
            epsilon_G=.08, target_accept=.7,
        )
        complete, samples, info = run_mmala_warmup(target, chain)
        init, update, final = dual_averaging_adaptation(.7)
        da = update(init(.4), jnp.mean(info.theta.acceptance_rate[0]))
        np.testing.assert_allclose(complete.epsilon, final(da), rtol=1e-14)
        self.assertEqual(complete.phase, 'sampling')
        self.assertEqual(samples.eta.shape[0], 1)
        np.testing.assert_array_equal(jax.random.key_data(complete.key),
                                      jax.random.key_data(jax.random.split(chain.key, 8)[0]))
        for kwargs in ({'num_warmup': 0}, {'num_warmup': True}, {'target_accept': 0},
                       {'target_accept': 1}, {'epsilon_G': 0}):
            settings = dict(num_warmup=4, epsilon=.4, epsilon_G=.08)
            settings.update(kwargs)
            with self.assertRaises(ValueError):
                initialize_mmala_warmup(target, s, chain.key, **settings)
        for changed in (replace(chain, epsilon=.5), replace(chain, iteration=1),
                        replace(chain, phase='sampling'), replace(chain, epsilon_G=-1)):
            with self.assertRaises(ValueError):
                validate_mmala_chain(target, changed)
        for run, args in ((run_fixed_mmala, (chain, 1)), (run_mmala_warmup, (chain, 2)),
                          (run_mmala_warmup, (complete,)), (run_fixed_mmala, (complete, 0))):
            with self.assertRaises(ValueError):
                run(target, *args)

    def test_fixed_tuning_initialization(self):
        target, s, _ = reference.make_fixture()
        fixed = initialize_mmala_chain(
            target, s, jax.random.key(1324), epsilon=.4, epsilon_G=.08
        )
        self.assertIsNone(fixed.adaptation)
        self.assertEqual(fixed.epsilon, .4)
        self.assertEqual(fixed.epsilon_G, .08)


if __name__ == '__main__':
    unittest.main()
