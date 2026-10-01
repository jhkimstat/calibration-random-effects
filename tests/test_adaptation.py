"""Independent covariance references, warmup boundaries, and numerical batches."""

import unittest
from dataclasses import replace

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from bayesiancalibration.adaptation import (
    initialize_random_walk_adaptation,
    update_random_walk_adaptation,
    validate_random_walk_adaptation,
)
from bayesiancalibration.mcmc import (
    initialize_random_walk_warmup,
    WarmupTuningError,
    run_fixed_random_walk,
    run_random_walk_warmup,
    validate_random_walk_chain,
)
import test_mcmc as reference


class CovarianceAdaptationTest(unittest.TestCase):
    def test_online_moments_and_note_regularizer_match_numpy(self):
        rng = np.random.default_rng(950)
        history = rng.normal(size=(20, 2, 3)) + np.array([1e6, -2e6, 3e6])
        # Explicit repeats contribute to the sample covariance.
        history[5:9] = history[4]
        previous = jnp.tile(jnp.diag(jnp.array([0.4, 0.6, 0.8])), (2, 1, 1))
        adaptation = initialize_random_walk_adaptation(20, 1, previous)
        kernel = jax.jit(update_random_walk_adaptation)
        moments = adaptation.moments
        for count, eta in enumerate(history, 1):
            moments, proposal = kernel(moments, jnp.asarray(eta), previous, True)
            np.testing.assert_array_equal(moments.sample_size, [count, count])
            np.testing.assert_allclose(moments.mean, history[:count].mean(axis=0),
                                       rtol=0, atol=1e-9)
            if count == 1:
                np.testing.assert_array_equal(proposal, previous)
            else:
                S_hat = np.stack([np.cov(history[:count, i], rowvar=False, ddof=1)
                                  for i in range(2)])
                np.testing.assert_allclose(moments.m2, S_hat*(count-1),
                                           rtol=1e-7, atol=1e-8)
                trace = np.trace(S_hat, axis1=-2, axis2=-1)
                expected = (2.38**2 / 3) * (
                    S_hat + trace[:, None, None] * np.eye(3) / 3000
                )
                np.testing.assert_allclose(proposal, expected, rtol=1e-7, atol=1e-8)
            previous = proposal
        self.assertEqual(proposal.dtype, jnp.float64)
        validate_random_walk_adaptation(replace(adaptation, moments=moments), 2, 3)

    def test_zero_variance_fallback_and_rank_deficient_estimates(self):
        previous = jnp.tile(jnp.diag(jnp.array([0.4, 0.6, 0.8])), (2, 1, 1))
        adaptation = initialize_random_walk_adaptation(4, 1, previous)
        moments = adaptation.moments
        eta = jnp.array([[0.2, -0.1, 0.5], [0.6, 0.2, -0.1]])
        kernel = jax.jit(update_random_walk_adaptation)
        with jax.debug_nans(True):
            for _ in range(3):
                moments, proposal = kernel(moments, eta, previous, True)
                np.testing.assert_array_equal(proposal, previous)
        changed = eta.at[0].add(jnp.array([1.0, 2.0, 3.0]))
        moments, proposal = kernel(moments, changed, previous, True)
        np.testing.assert_array_equal(proposal[1], previous[1])
        S_hat = np.cov(np.vstack((np.tile(eta[0], (3, 1)), changed[0])),
                       rowvar=False, ddof=1)
        self.assertEqual(np.linalg.matrix_rank(S_hat), 1)
        expected = (2.38**2 / 3) * (S_hat + np.trace(S_hat)*np.eye(3)/3000)
        np.testing.assert_allclose(proposal[0], expected, rtol=1e-13, atol=1e-14)
        self.assertTrue(np.all(np.linalg.eigvalsh(proposal[0]) > 0))
        updated, held = kernel(moments, changed, previous, False)
        np.testing.assert_array_equal(held, previous)
        np.testing.assert_array_equal(updated.sample_size, [5, 5])

    def test_invalid_schedule_covariance_and_statistics(self):
        initial = jnp.tile(jnp.eye(2), (2, 1, 1))
        for total, fixed in ((0, 1), (4, 0), (2, 3), (True, 1), (4, 1.5)):
            with self.assertRaises(ValueError):
                initialize_random_walk_adaptation(total, fixed, initial)
        for proposal in (-initial, initial[0], initial.at[0, 0, 1].set(0.1)):
            with self.assertRaises(ValueError):
                initialize_random_walk_adaptation(4, 2, proposal)
        adaptation = initialize_random_walk_adaptation(4, 2, initial)
        for moments in (
            adaptation.moments._replace(sample_size=jnp.array([0, 1])),
            adaptation.moments._replace(sample_size=jnp.array([1.0, 1.0])),
            adaptation.moments._replace(mean=jnp.ones((2, 2))),
            adaptation.moments._replace(m2=-jnp.tile(jnp.eye(2), (2, 1, 1)),
                                       sample_size=jnp.array([2, 2])),
            adaptation.moments._replace(mean=jnp.full((2, 2), jnp.nan)),
        ):
            with self.assertRaises(ValueError):
                validate_random_walk_adaptation(
                    replace(adaptation, moments=moments), 2, 2
                )


class WarmupIntegrationTest(unittest.TestCase):
    assert_tree_equal = reference.GibbsSweepTest.assert_tree_equal

    def test_initial_period_adaptation_boundary_and_frozen_production(self):
        target, state, initial = reference.make_fixture(True)
        key = jax.random.key(960)
        chain = initialize_random_walk_warmup(
            target, state, key, num_warmup=4, num_initial=2, V_prop=initial
        )
        with self.assertRaisesRegex(ValueError, "Complete warmup"):
            run_fixed_random_walk(target, chain, 1)
        first, history1, _ = run_random_walk_warmup(target, chain, 1)
        self.assertEqual(first.phase, "warmup")
        np.testing.assert_array_equal(first.V_prop, initial)
        np.testing.assert_array_equal(first.adaptation.moments.sample_size, [1, 1])
        second, history2, _ = run_random_walk_warmup(target, first, 1)
        history = np.concatenate((history1.eta, history2.eta))
        S_hat = np.stack([np.cov(history[:, i], rowvar=False, ddof=1)
                          for i in range(2)])
        expected = (2.38**2 / 2) * (
            S_hat + np.trace(S_hat, axis1=-2, axis2=-1)[:, None, None]*np.eye(2)/2000
        )
        # Each site may have zero empirical movement; its initial covariance
        # must survive until that site's estimate is usable.
        for i in range(2):
            np.testing.assert_allclose(
                second.V_prop[i],
                expected[i] if np.trace(S_hat[i]) > 0 else initial[i],
                rtol=1e-12, atol=1e-14,
            )
        finished, history3, _ = run_random_walk_warmup(target, second)
        self.assertEqual(finished.phase, "sampling")
        self.assertEqual(finished.iteration, 4)
        history = np.concatenate((history, history3.eta))
        np.testing.assert_allclose(
            finished.adaptation.moments.mean, history.mean(axis=0)
        )
        for i in range(2):
            np.testing.assert_allclose(
                finished.adaptation.moments.m2[i],
                3 * np.cov(history[:, i], rowvar=False, ddof=1), atol=1e-14,
            )
        production, samples, _ = run_fixed_random_walk(target, finished, 2)
        np.testing.assert_array_equal(production.V_prop, finished.V_prop)
        self.assert_tree_equal(production.adaptation.moments,
                               finished.adaptation.moments, True)
        self.assertEqual(production.iteration, 6)
        self.assertEqual(production.adaptation.completed, 4)
        self.assertEqual(samples.eta.shape[0], 2)
        with self.assertRaisesRegex(ValueError, "phase='warmup'"):
            run_random_walk_warmup(target, production, 1)

    def test_small_variance_short_warmup_and_repeated_rejections(self):
        for total, initial_count in ((1, 1), (3, 3)):
            target, state, initial = reference.make_fixture()
            chain = initialize_random_walk_warmup(
                target, state, jax.random.key(961),
                num_warmup=total, num_initial=initial_count,
            )
            np.testing.assert_array_equal(chain.V_prop, 1e-6 * np.tile(np.eye(2), (2, 1, 1)))
            if total == 1:
                # A single completed position cannot estimate covariance.
                with self.assertRaises(WarmupTuningError) as failure:
                    run_random_walk_warmup(target, chain)
                self.assertEqual(failure.exception.diagnostics["completed"], total)
            else:
                finished, _, _ = run_random_walk_warmup(target, chain)
                self.assertEqual(finished.phase, "sampling")
            self.assertEqual(chain.phase, "warmup")
        target, state, initial = reference.make_fixture(True)
        chain = initialize_random_walk_warmup(
            target, state, jax.random.key(962), num_warmup=3, num_initial=1,
            V_prop=initial*1e100,
        )
        finished, samples, info = run_random_walk_warmup(target, chain, 2)
        np.testing.assert_array_equal(info.theta.is_accepted, np.zeros((2, 2)))
        np.testing.assert_array_equal(samples.eta, np.tile(state.eta, (2, 1, 1)))
        np.testing.assert_array_equal(finished.adaptation.moments.sample_size, [2, 2])
        with self.assertRaises(WarmupTuningError) as failure:
            run_random_walk_warmup(target, finished, 1)
        self.assertEqual(failure.exception.diagnostics["movement_count"], [0, 0])
        self.assertEqual(failure.exception.diagnostics["acceptance_count"], [0, 0])
        self.assertEqual(failure.exception.diagnostics["zero_covariance_count"], [2, 2])
        np.testing.assert_array_equal(finished.adaptation.moments.mean, state.eta)
        np.testing.assert_array_equal(
            finished.adaptation.moments.m2, np.zeros((2, 2, 2))
        )
        np.testing.assert_array_equal(finished.V_prop, chain.V_prop)
        self.assertFalse(np.array_equal(samples.c_f[0], samples.c_f[-1]))

    def test_invalid_phase_or_counter_cannot_cross_warmup_boundary(self):
        target, state, initial = reference.make_fixture()
        chain = initialize_random_walk_warmup(
            target, state, jax.random.key(963), num_warmup=3, num_initial=2,
            V_prop=initial,
        )
        for count in (0, -1, 4, True, 1.5):
            with self.assertRaises(ValueError):
                run_random_walk_warmup(target, chain, count)
        for invalid in (
            replace(chain, phase="sampling"), replace(chain, iteration=1),
            replace(chain, V_prop=initial*2), replace(chain, adaptation=None),
        ):
            with self.assertRaises(ValueError):
                validate_random_walk_chain(target, invalid)
        self.assertEqual(chain.iteration, 0)
        self.assertEqual(chain.adaptation.completed, 0)

    def test_warmup_batches_match_one_run_through_production(self):
        for bounded, loading_only, key in (
            (False, False, jax.random.key(964)),
            (True, True, jax.random.PRNGKey(965)),
        ):
            target, state, initial = reference.make_fixture(bounded, loading_only)
            chain = initialize_random_walk_warmup(
                target, state, key, num_warmup=5, num_initial=2, V_prop=initial
            )
            complete, warmup_samples, warmup_info = run_random_walk_warmup(target, chain)
            all_final, all_samples, all_info = run_fixed_random_walk(target, complete, 2)
            part, early_samples, early_info = run_random_walk_warmup(target, chain, 3)
            self.assertEqual(part.phase, "warmup")
            self.assertEqual(part.adaptation.completed, 3)
            late, late_samples, late_info = run_random_walk_warmup(target, part)
            self.assert_tree_equal(jax.tree.map(
                lambda a, b: jnp.concatenate((a, b)), early_samples, late_samples
            ), warmup_samples, True)
            self.assert_tree_equal(jax.tree.map(
                lambda a, b: jnp.concatenate((a, b)), early_info, late_info
            ), warmup_info, True)
            np.testing.assert_array_equal(late.V_prop, complete.V_prop)
            self.assert_tree_equal(late.adaptation.moments, complete.adaptation.moments, True)
            final, samples, info = run_fixed_random_walk(target, late, 2)
            self.assert_tree_equal(samples, all_samples, True)
            self.assert_tree_equal(info, all_info, True)
            self.assert_tree_equal(final.model_state, all_final.model_state, True)
            np.testing.assert_array_equal(jax.random.key_data(final.key),
                                          jax.random.key_data(all_final.key))
            self.assertEqual(final.iteration, 7)



if __name__ == "__main__":
    unittest.main()
