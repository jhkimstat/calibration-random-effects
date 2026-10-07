"""Standard DA reference, dynamic epsilon, phase boundaries, and batching."""

import unittest
from dataclasses import replace

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from blackjax.adaptation.step_size import dual_averaging_adaptation

from bayesiancalibration.mcmc import (
    collapsed_gibbs_sweep,
    initialize_mala_chain,
    initialize_mala_warmup,
    run_fixed_mala,
    run_mala_warmup,
    validate_mala_chain,
)
import test_mcmc as reference


class MALADualAveragingTest(unittest.TestCase):
    assert_tree_equal = reference.GibbsSweepTest.assert_tree_equal

    def test_dynamic_sweeps_match_standard_da_and_probability_feedback(self):
        target, state, P = reference.make_fixture(True)
        initial_epsilon, target_accept = 0.35, 0.574
        chain = initialize_mala_warmup(
            target, state, jax.random.key(1110), num_warmup=6, num_initial=6,
            epsilon=initial_epsilon, V_prop=P,
        )
        init, update, final = dual_averaging_adaptation(target_accept)
        expected_da = init(initial_epsilon)
        expected_epsilon = initial_epsilon
        expected_state, expected_key = state, chain.key
        kernel = jax.jit(lambda key, state, epsilon: collapsed_gibbs_sweep(
            key, target, state, P, epsilon=epsilon
        ))
        changed_kernel, nonbinary_feedback = False, False
        for completed in range(1, 7):
            expected_state, expected_key, info = kernel(
                expected_key, expected_state, expected_epsilon
            )
            probability = jnp.mean(info.theta.acceptance_rate)
            nonbinary_feedback |= not np.isclose(
                probability, jnp.mean(info.theta.is_accepted), rtol=0, atol=1e-8
            )
            expected_da = update(expected_da, probability)
            expected_epsilon = float(
                final(expected_da) if completed == 6
                else jnp.exp(expected_da.log_step_size)
            )
            previous = chain
            chain, samples, actual_info = run_mala_warmup(target, chain, 1)
            self.assert_tree_equal(chain.model_state, expected_state)
            self.assert_tree_equal(
                jax.tree.map(lambda x: x[0], samples), expected_state
            )
            self.assert_tree_equal(jax.tree.map(lambda x: x[0], actual_info), info)
            self.assert_tree_equal(chain.step_size_adaptation.state, expected_da)
            np.testing.assert_allclose(chain.epsilon, expected_epsilon, rtol=1e-13)
            np.testing.assert_array_equal(jax.random.key_data(chain.key),
                                          jax.random.key_data(expected_key))
            np.testing.assert_array_equal(chain.V_prop, P)
            self.assertEqual(int(chain.step_size_adaptation.state.step), completed + 1)
            if completed > 1:
                _, _, fixed_info = kernel(
                    previous.key, previous.model_state, initial_epsilon
                )
                changed_kernel |= not np.allclose(
                    actual_info.theta.acceptance_rate[0],
                    fixed_info.theta.acceptance_rate,
                )
        self.assertTrue(nonbinary_feedback)
        self.assertTrue(changed_kernel)
        self.assertEqual(chain.phase, "sampling")
        # The final averaged scale is distinct from the instantaneous scale.
        self.assertNotAlmostEqual(
            chain.epsilon,
            float(jnp.exp(chain.step_size_adaptation.state.log_step_size)),
        )

    def test_short_warmup_final_average_and_frozen_production(self):
        target, state, P = reference.make_fixture()
        chain = initialize_mala_warmup(
            target, state, jax.random.key(1111), num_warmup=2, num_initial=2,
            epsilon=0.01, V_prop=P, target_accept=0.7,
        )
        initial, update, final = dual_averaging_adaptation(0.7)
        frozen, samples, info = run_mala_warmup(target, chain)
        expected = initial(0.01)
        for rate in info.theta.acceptance_rate:
            expected = update(expected, jnp.mean(rate))
        # Independent compiled/eager tuning need only agree to negligible
        # relative error; exact production freeze is checked below.
        np.testing.assert_allclose(frozen.epsilon, float(final(expected)), rtol=1e-10, atol=0)
        self.assert_tree_equal(frozen.step_size_adaptation.state, expected)
        self.assertEqual(samples.eta.shape[0], 2)
        self.assertEqual(frozen.iteration, 2)
        np.testing.assert_array_equal(
            jax.random.key_data(frozen.key),
            jax.random.key_data(jax.random.split(jax.random.split(chain.key, 8)[0], 8)[0]),
        )
        production, draws, _ = run_fixed_mala(target, frozen, 2)
        self.assertEqual(production.epsilon, frozen.epsilon)
        self.assert_tree_equal(production.step_size_adaptation.state,
                               frozen.step_size_adaptation.state, True)
        self.assert_tree_equal(production.adaptation.moments,
                               frozen.adaptation.moments, True)
        np.testing.assert_array_equal(production.V_prop, frozen.V_prop)
        self.assertEqual(production.iteration, 4)
        self.assertEqual(draws.eta.shape[0], 2)
        with self.assertRaises(ValueError):
            run_mala_warmup(target, production)

    def test_invalid_configuration_statistics_and_phase_consistency(self):
        target, state, P = reference.make_fixture()
        for target_accept in (0, 1, -0.1, 1.1, np.nan, np.inf, True, [0.65]):
            with self.assertRaises(ValueError):
                initialize_mala_warmup(
                    target, state, jax.random.key(1112), num_warmup=4, num_initial=2,
                    epsilon=0.4, V_prop=P, target_accept=target_accept,
                )
        chain = initialize_mala_warmup(
            target, state, jax.random.key(1112), num_warmup=4, num_initial=2,
            epsilon=0.4, V_prop=P, target_accept=0.65,
        )
        adaptation = chain.step_size_adaptation
        da = adaptation.state
        for changed in (
            replace(adaptation, target_accept=1.0),
            replace(adaptation, initial_epsilon=0.5),
            replace(adaptation, state=da._replace(step=jnp.array(2))),
            replace(adaptation, state=da._replace(step=jnp.array(1.0))),
            replace(adaptation, state=da._replace(mu=da.mu + 0.1)),
            replace(adaptation, state=da._replace(avg_error=jnp.array(1.1))),
            replace(adaptation, state=da._replace(log_step_size=jnp.array(jnp.nan))),
            replace(adaptation, state=da._replace(
                log_step_size_avg=jnp.array(0, dtype=jnp.float32)
            )),
        ):
            with self.assertRaises(ValueError):
                validate_mala_chain(
                    target, replace(chain, step_size_adaptation=changed)
                )
        with self.assertRaisesRegex(ValueError, "epsilon does not match"):
            validate_mala_chain(target, replace(chain, epsilon=0.5))
        fixed = initialize_mala_chain(target, state, chain.key, P, 0.4)
        with self.assertRaisesRegex(ValueError, "warmup schedule"):
            validate_mala_chain(target, replace(fixed, step_size_adaptation=adaptation))

    def test_coupled_covariance_da_batches_match_one_run(self):
        for bounded, loading_only, key in (
            (False, False, jax.random.key(1113)),
            (True, True, jax.random.PRNGKey(1114)),
        ):
            target, state, P = reference.make_fixture(bounded, loading_only)
            chain = initialize_mala_warmup(
                target, state, key, num_warmup=8, num_initial=2,
                epsilon=0.35, V_prop=P, target_accept=0.65,
            )
            full, full_samples, full_info = run_mala_warmup(target, chain)
            expected, expected_draws, expected_info = run_fixed_mala(target, full, 2)
            first, early, early_info = run_mala_warmup(target, chain, 3)
            self.assertEqual(first.step_size_adaptation.target_accept, 0.65)
            late, late_samples, late_info = run_mala_warmup(target, first)
            self.assert_tree_equal(jax.tree.map(
                lambda a, b: jnp.concatenate((a, b)), early, late_samples
            ), full_samples, True)
            self.assert_tree_equal(jax.tree.map(
                lambda a, b: jnp.concatenate((a, b)), early_info, late_info
            ), full_info, True)
            self.assertEqual(late.epsilon, full.epsilon)
            self.assert_tree_equal(late.step_size_adaptation.state,
                                   full.step_size_adaptation.state, True)
            np.testing.assert_array_equal(late.V_prop, full.V_prop)
            final, draws, info = run_fixed_mala(target, late, 2)
            self.assert_tree_equal(draws, expected_draws, True)
            self.assert_tree_equal(info, expected_info, True)
            self.assert_tree_equal(final.model_state, expected.model_state, True)
            np.testing.assert_array_equal(jax.random.key_data(final.key),
                                          jax.random.key_data(expected.key))
            self.assertEqual(final.iteration, 10)


    def test_unusable_adapted_epsilon_stops_without_clipping_or_partial_chain(self):
        target, state, P = reference.make_fixture()
        chain = initialize_mala_warmup(
            target, state, jax.random.key(1116), num_warmup=2, num_initial=2,
            epsilon=9e153, V_prop=P, target_accept=0.01,
        )
        initial_key = np.asarray(jax.random.key_data(chain.key)).copy()
        with self.assertRaisesRegex(FloatingPointError, "epsilon/diffusion"):
            run_mala_warmup(target, chain, 1)
        self.assertEqual(chain.iteration, 0)
        self.assertEqual(chain.epsilon, 9e153)
        self.assertEqual(int(chain.step_size_adaptation.state.step), 1)
        np.testing.assert_array_equal(jax.random.key_data(chain.key), initial_key)
        self.assert_tree_equal(chain.model_state, state, True)


if __name__ == "__main__":
    unittest.main()
