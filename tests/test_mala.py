"""Dense MALA proposal references, posterior quadrature, and Gibbs integration."""

import unittest
from dataclasses import replace
from unittest.mock import patch

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from blackjax.mcmc import mala
from scipy.integrate import quad
from scipy.stats import multivariate_normal, norm

from bayesiancalibration.gibbs import (
    refresh_field_coefficients,
    update_branch_noise,
    update_coefficient_variances,
    update_discrepancy,
    update_spatial_covariance,
    update_spatial_mean,
)
from bayesiancalibration.gp import LibraryGP
from bayesiancalibration.mcmc import (
    collapsed_gibbs_sweep,
    initialize_mala_warmup,
    run_fixed_mala,
    run_fixed_random_walk,
    run_mala_warmup,
    WarmupTuningError,
    validate_mala_chain,
)
from bayesiancalibration.samplers.metropolis import (
    collapsed_mala_sweep,
    update_collapsed_mala,
)
from bayesiancalibration.state import SpatialPrior, ThetaStandardization
from bayesiancalibration.targets import CalibrationTarget
from bayesiancalibration.transforms import SiteCoordinates
import test_mcmc as outer_reference
import test_random_walk as theta_reference


class CollapsedMALATest(unittest.TestCase):
    def test_dense_sequential_drift_and_asymmetric_acceptance(self):
        fixture = theta_reference.CollapsedRandomWalkTest()
        decisions, corrections = [], []
        epsilon = 0.7
        for bounded in (False, True):
            for loading_only in (False, True):
                target, inputs = fixture.target_and_state(bounded, loading_only)
                kernel = jax.jit(lambda key: collapsed_mala_sweep(
                    key, target, *inputs, epsilon
                ))

                def density(eta):
                    return fixture.reference_logdensity(target, eta, inputs)

                def gradient(eta, i):
                    # Five-point finite differences of the independent dense
                    # NumPy/SciPy density, without calling JAX autodiff.
                    h = 1e-4
                    result = []
                    for j in range(2):
                        shift = np.zeros_like(eta)
                        shift[i, j] = h
                        result.append((
                            density(eta - 2*shift) - 8*density(eta - shift)
                            + 8*density(eta + shift) - density(eta + 2*shift)
                        ) / (12*h))
                    return np.array(result)

                for seed in range(8):
                    key = jax.random.key(1010 + seed)
                    eta = np.asarray(inputs[0]).copy()
                    probabilities, accepted = [], []
                    for i, site_key in enumerate(jax.random.split(key, 2)):
                        proposal_key, accept_key = jax.random.split(site_key)
                        P = np.asarray(inputs[-1][i])
                        current = density(eta)
                        forward = eta[i] + epsilon**2 * P @ gradient(eta, i) / 2
                        candidate = eta.copy()
                        candidate[i] = forward + epsilon * np.linalg.cholesky(P) @ (
                            np.asarray(jax.random.normal(proposal_key, (2,),
                                                         dtype=jnp.float64))
                        )
                        proposed = density(candidate)
                        reverse = (
                            candidate[i] + epsilon**2 * P @ gradient(candidate, i) / 2
                        )
                        correction = multivariate_normal.logpdf(
                            eta[i], reverse, epsilon**2 * P
                        ) - multivariate_normal.logpdf(
                            candidate[i], forward, epsilon**2 * P
                        )
                        probability = np.exp(min(0.0, proposed - current + correction))
                        do_accept = bool(jax.random.bernoulli(accept_key, probability))
                        if do_accept:
                            eta = candidate
                        probabilities.append(probability)
                        accepted.append(do_accept)
                        corrections.append(correction)
                    actual, info = kernel(key)
                    np.testing.assert_allclose(actual, eta, rtol=0, atol=2e-9)
                    np.testing.assert_allclose(info.acceptance_rate, probabilities,
                                               rtol=2e-8, atol=2e-9)
                    np.testing.assert_array_equal(info.is_accepted, accepted)
                    np.testing.assert_allclose(info.logdensity, density(eta),
                                               rtol=2e-9, atol=2e-9)
                    decisions.extend(accepted)
        self.assertIn(True, decisions)
        self.assertIn(False, decisions)
        self.assertGreater(max(abs(np.asarray(corrections))), 0.1)

    def test_identity_preconditioner_matches_standard_blackjax_mala(self):
        target, inputs = theta_reference.CollapsedRandomWalkTest().target_and_state()
        eta = inputs[0]
        epsilon = 0.45
        key = jax.random.key(1020)
        rates, accepted = [], []
        for i, site_key in enumerate(jax.random.split(key, 2)):
            def density(site_position):
                return target.theta_only_collapsed(
                    eta.at[i].set(site_position), *inputs[1:6]
                )
            state = mala.init(eta[i], density)
            state, info = mala.build_kernel()(site_key, state, density, epsilon**2/2)
            eta = eta.at[i].set(state.position)
            rates.append(info.acceptance_rate)
            accepted.append(info.is_accepted)
        actual, info = collapsed_mala_sweep(
            key, target, *inputs[:-1], jnp.tile(jnp.eye(2), (2, 1, 1)), epsilon
        )
        np.testing.assert_allclose(actual, eta, rtol=0, atol=1e-15)
        np.testing.assert_allclose(info.acceptance_rate, rates, rtol=1e-12)
        np.testing.assert_array_equal(info.is_accepted, accepted)

    def test_keys_jit_vmap_and_current_conditioning(self):
        target, inputs = theta_reference.CollapsedRandomWalkTest().target_and_state(
            True
        )
        kernel = jax.jit(lambda key, *args: collapsed_mala_sweep(key, target, *args))
        key = jax.random.key(1021)
        actual, info = update_collapsed_mala(key, target, *inputs, 0.4)
        repeated, repeated_info = kernel(key, *inputs, 0.4)
        np.testing.assert_allclose(actual, repeated, rtol=0, atol=1e-15)
        np.testing.assert_array_equal(info.is_accepted, repeated_info.is_accepted)
        keys = jax.random.split(key, 3)
        draws, infos = jax.jit(jax.vmap(lambda key: kernel(key, *inputs, 0.4)))(keys)
        self.assertEqual(draws.dtype, jnp.float64)
        self.assertEqual(draws.shape, (3, 2, 2))
        self.assertEqual(infos.is_accepted.shape, (3, 2))
        self.assertFalse(np.array_equal(draws[0], draws[1]))
        for index in range(6):
            changed = list(inputs)
            changed[index] = changed[index] * 1.2 + (0.1 if index != 4 else 0)
            result, changed_info = kernel(key, *changed, 0.4)
            expected = target.theta_only_collapsed(result, *changed[1:6])
            np.testing.assert_allclose(changed_info.logdensity, expected, rtol=1e-12)
            self.assertNotAlmostEqual(
                float(changed_info.logdensity), float(info.logdensity)
            )
        np.testing.assert_array_equal(target.gp.lambda_c, [0.8, 1.2])

    def test_invalid_tuning_state_and_gradients(self):
        target, inputs = theta_reference.CollapsedRandomWalkTest().target_and_state()
        key = jax.random.key(1022)
        for epsilon in (0, -0.1, np.nan, np.inf, 1e200, 1e-200, [0.4], True):
            with self.assertRaises(ValueError):
                update_collapsed_mala(key, target, *inputs, epsilon)
        for index, value in (
            (0, inputs[0][:1]), (2, -inputs[2]), (4, -inputs[4]),
            (6, -inputs[6]), (6, inputs[6].at[0, 0, 1].set(0)),
        ):
            changed = list(inputs)
            changed[index] = value
            with self.assertRaises(ValueError):
                update_collapsed_mala(key, target, *changed, 0.4)
        for jitter in (0.0, 1e-6):
            gp = LibraryGP.from_data(
                target.gp.theta_s_tilde, target.gp.F_s, target.gp.lambda_c,
                jitter=jitter,
                kernel="se",
            )
            changed = list(inputs)
            changed[0] = inputs[0].at[0].set(target.gp.theta_s_tilde[0])
            with self.assertRaisesRegex(ValueError, "coincides"):
                update_collapsed_mala(key, replace(target, gp=gp), *changed, 0.4)

        @jax.custom_jvp
        def bad_gradient(eta):
            return -jnp.sum(eta**2)

        @bad_gradient.defjvp
        def bad_gradient_jvp(primals, tangents):
            return bad_gradient(primals[0]), jnp.sum(tangents[0] * jnp.nan)

        with patch.object(CalibrationTarget, "theta_only_collapsed",
                          lambda self, eta, *args: bad_gradient(eta)):
            with self.assertRaisesRegex(ValueError, "gradient"):
                update_collapsed_mala(key, target, *inputs, 0.4)

    def test_extreme_proposals_reject_without_clipping_or_gradient_repair(self):
        for bounded in (False, True):
            target, inputs = theta_reference.CollapsedRandomWalkTest().target_and_state(
                bounded
            )
            actual, info = collapsed_mala_sweep(
                jax.random.key(1023), target, *inputs, 1e150
            )
            np.testing.assert_array_equal(actual, inputs[0])
            np.testing.assert_array_equal(info.is_accepted, [False, False])
            np.testing.assert_array_equal(info.acceptance_rate, [0.0, 0.0])
            self.assertTrue(np.isfinite(float(info.logdensity)))


class MALAGibbsTest(unittest.TestCase):
    assert_tree_equal = outer_reference.GibbsSweepTest.assert_tree_equal

    def test_full_schedule_and_coefficient_refresh_use_current_inputs(self):
        for bounded in (False, True):
            for loading_only in (False, True):
                target, state, P = outer_reference.make_fixture(bounded, loading_only)
                key = jax.random.key(1030)
                next_key, kd, ky, km, kS, kc, kt, kf = jax.random.split(key, 8)
                delta = update_discrepancy(kd, target, state.c_f, state.sigma_y2)
                noise = update_branch_noise(ky, target, state.c_f, delta)
                mean = update_spatial_mean(km, target, state.eta, state.Sigma_theta)
                covariance = update_spatial_covariance(kS, target, state.eta, mean)
                variances = update_coefficient_variances(
                    kc, target, state.eta, state.c_f
                )
                eta, theta_info = update_collapsed_mala(
                    kt, target, state.eta, delta, noise, mean, covariance,
                    variances, P, 0.35,
                )
                c_f = refresh_field_coefficients(
                    kf, target, eta, delta, noise, variances
                )
                expected = state._replace(
                    eta=eta, c_f=c_f, delta=delta, sigma_y2=noise, mu_theta=mean,
                    Sigma_theta=covariance, sigma_c2=variances,
                )
                actual, actual_key, info = jax.jit(lambda key: collapsed_gibbs_sweep(
                    key, target, state, P, epsilon=0.35
                ))(key)
                self.assert_tree_equal(actual, expected)
                self.assert_tree_equal(info.theta, theta_info)
                np.testing.assert_array_equal(jax.random.key_data(actual_key),
                                              jax.random.key_data(next_key))
                np.testing.assert_allclose(info.full_joint_logdensity,
                                           target.full_joint_uncollapsed(*expected))

    def test_warmup_preconditioner_scale_boundary_and_frozen_production(self):
        target, state, _ = outer_reference.make_fixture(True)
        chain = initialize_mala_warmup(
            target, state, jax.random.key(1031), num_warmup=5, num_initial=2,
            epsilon=0.3, target_accept=None,
        )
        np.testing.assert_array_equal(chain.V_prop, np.tile(np.eye(2), (2, 1, 1)))
        with self.assertRaisesRegex(ValueError, "Complete warmup"):
            run_fixed_mala(target, chain, 1)
        first, sample1, _ = run_mala_warmup(target, chain, 1)
        np.testing.assert_array_equal(first.V_prop, chain.V_prop)
        finished, samples2, _ = run_mala_warmup(target, first)
        history = np.concatenate((sample1.eta, samples2.eta))
        for i in range(2):
            S = np.cov(history[:, i], rowvar=False, ddof=1)
            expected = S + np.trace(S) * np.eye(2) / 2000
            np.testing.assert_allclose(finished.V_prop[i], expected, rtol=1e-12)
        self.assertEqual(finished.phase, "sampling")
        self.assertEqual(finished.iteration, 5)
        self.assertEqual(finished.epsilon, 0.3)
        production, _, _ = run_fixed_mala(target, finished, 2)
        np.testing.assert_array_equal(production.V_prop, finished.V_prop)
        self.assert_tree_equal(production.adaptation.moments,
                               finished.adaptation.moments, True)
        self.assertEqual(production.epsilon, finished.epsilon)
        self.assertEqual(production.adaptation.completed, 5)
        for count in (0, -1, True, 1.5, 6):
            with self.assertRaises(ValueError):
                run_mala_warmup(target, chain, count)
        with self.assertRaises(ValueError):
            validate_mala_chain(target, replace(chain, epsilon=0))
        with self.assertRaises(ValueError):
            run_fixed_random_walk(target, finished, 1)
        with self.assertRaises(ValueError):
            run_mala_warmup(target, finished, 1)

    def test_rejected_states_count_and_refresh_coefficients(self):
        for bounded in (False, True):
            target, state, P = outer_reference.make_fixture(bounded)
            chain = initialize_mala_warmup(
                target, state, jax.random.key(1032), num_warmup=3, num_initial=1,
                epsilon=1e150, V_prop=P, target_accept=None,
            )
            finished, samples, info = run_mala_warmup(target, chain, 2)
            np.testing.assert_array_equal(info.theta.is_accepted, np.zeros((2, 2)))
            np.testing.assert_array_equal(samples.eta, np.tile(state.eta, (2, 1, 1)))
            with self.assertRaises(WarmupTuningError) as failure:
                run_mala_warmup(target, finished, 1)
            self.assertEqual(failure.exception.diagnostics["movement_count"], [0, 0])
            self.assertEqual(failure.exception.diagnostics["acceptance_count"], [0, 0])
            np.testing.assert_array_equal(finished.V_prop, P)
            np.testing.assert_array_equal(
                finished.adaptation.moments.sample_size, [2, 2]
            )
            np.testing.assert_array_equal(finished.adaptation.moments.m2,
                                          np.zeros((2, 2, 2)))
            self.assertFalse(np.array_equal(samples.c_f[0], samples.c_f[-1]))

    def test_warmup_batches_and_production_match_one_run(self):
        for bounded, loading_only, key in (
            (False, False, jax.random.key(1033)),
            (True, True, jax.random.PRNGKey(1034)),
        ):
            target, state, P = outer_reference.make_fixture(bounded, loading_only)
            chain = initialize_mala_warmup(
                target, state, key, num_warmup=5, num_initial=2, epsilon=0.4,
                V_prop=P, target_accept=None,
            )
            full, warm_samples, warm_info = run_mala_warmup(target, chain)
            expected, expected_samples, expected_info = run_fixed_mala(target, full, 2)
            partial, early, early_info = run_mala_warmup(target, chain, 3)
            self.assertEqual(partial.epsilon, chain.epsilon)
            late, late_samples, late_info = run_mala_warmup(target, partial)
            self.assert_tree_equal(jax.tree.map(
                lambda a, b: jnp.concatenate((a, b)), early, late_samples
            ), warm_samples, True)
            self.assert_tree_equal(jax.tree.map(
                lambda a, b: jnp.concatenate((a, b)), early_info, late_info
            ), warm_info, True)
            np.testing.assert_array_equal(late.V_prop, full.V_prop)
            self.assert_tree_equal(late.adaptation.moments, full.adaptation.moments, True)
            final, samples, info = run_fixed_mala(target, late, 2)
            self.assert_tree_equal(samples, expected_samples, True)
            self.assert_tree_equal(info, expected_info, True)
            self.assert_tree_equal(final.model_state, expected.model_state, True)
            np.testing.assert_array_equal(jax.random.key_data(final.key),
                                          jax.random.key_data(expected.key))
            self.assertEqual(final.iteration, 7)



class MALAPosteriorTest(unittest.TestCase):
    def test_scalar_calibration_posterior_matches_independent_quadrature(self):
        library = np.array([[-2.0], [0.0], [2.0]])
        standardization = ThetaStandardization.from_library(library)
        theta_s = np.asarray(standardization.to_standardized(library))
        F_s = np.array([[-0.55], [0.2], [1.2]])
        gp = LibraryGP.from_data(theta_s, F_s, [0.65], kernel="se")
        prior = SpatialPrior.from_standardization(
            standardization, physical_center=[0.0], V_theta_0=np.eye(1),
            nu_theta_0=3.0, S_theta_0=np.eye(1),
        )
        C_ss = np.exp(-0.5 * ((theta_s - theta_s.T) / 0.65)**2)
        weights = np.linalg.solve(C_ss, F_s)

        def density(theta):
            cross = np.exp(-0.5 * ((theta - theta_s[:, 0]) / 0.65)**2)
            mean = float((cross @ weights).item()) + 0.04
            variance = 0.06 + 0.35 * (1 - cross @ np.linalg.solve(C_ss, cross))
            return norm.pdf(0.45, mean, np.sqrt(variance)) * norm.pdf(
                theta, 0.12, np.sqrt(0.45)
            )

        for bounded in (False, True):
            coordinates = SiteCoordinates.from_physical_bounds(
                standardization,
                **({"l": [-1.5], "u": [1.6]} if bounded else {}),
            )
            target = CalibrationTarget.from_data(
                gp, coordinates, prior, [0.45], [[1.0]], [[0, 0]], (1,)
            )
            limits = (-0.75, 0.8) if bounded else (-np.inf, np.inf)
            Z = quad(density, *limits, epsabs=1e-11)[0]
            expected = np.array([
                quad(lambda x: x*density(x), *limits, epsabs=1e-11)[0] / Z,
                quad(lambda x: x*x*density(x), *limits, epsabs=1e-11)[0] / Z,
                quad(density, limits[0], 0.1, epsabs=1e-11)[0] / Z,
            ])
            conditioning = (
                jnp.array([0.04]), jnp.array([0.06]), jnp.array([0.12]),
                jnp.array([[0.45]]), jnp.array([0.35]), jnp.array([[[0.7]]]),
            )

            def run(key, theta_initial):
                eta = coordinates.theta_tilde_to_eta(theta_initial.reshape(1, 1))
                keys = jax.random.split(key, 26_000)

                def step(position, sweep_key):
                    position, _ = collapsed_mala_sweep(
                        sweep_key, target, position, *conditioning, 0.65
                    )
                    return position, coordinates.eta_to_theta_tilde(position)[0, 0]

                return jax.lax.scan(step, eta, keys)[1]

            chains = np.asarray(jax.jit(jax.vmap(run))(
                jax.random.split(jax.random.key(1040 + int(bounded)), 4),
                jnp.array([-0.6, -0.2, 0.3, 0.6]),
            ))[:, 2_000:]
            summaries = np.stack((chains, chains**2, chains <= 0.1), axis=-1)
            batches = summaries.reshape(4, 60, 400, 3).mean(axis=2)
            estimates = summaries.mean(axis=(0, 1))
            mcse = batches.reshape(-1, 3).std(axis=0, ddof=1) / np.sqrt(240)
            np.testing.assert_array_less(np.abs(estimates - expected), 6*mcse)
            for i in range(4):
                se = batches[i].std(axis=0, ddof=1) / np.sqrt(60)
                np.testing.assert_array_less(
                    np.abs(summaries[i].mean(axis=0) - expected), 6*se
                )
            self.assertGreater(np.mean(np.diff(chains, axis=1) == 0), 0.01)
            if bounded:
                self.assertTrue(np.all(chains > limits[0]))
                self.assertTrue(np.all(chains < limits[1]))


if __name__ == "__main__":
    unittest.main()
