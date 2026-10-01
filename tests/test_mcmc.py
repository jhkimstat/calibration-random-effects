"""Outer-sweep ordering, joint stationarity, and numerical batching."""

import unittest
from dataclasses import replace
from unittest.mock import patch

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from scipy.stats import invgamma, norm

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
    initialize_random_walk_chain,
    run_fixed_random_walk,
    validate_random_walk_chain,
)
from bayesiancalibration.samplers.metropolis import update_collapsed_random_walk
from bayesiancalibration.state import (
    CalibrationState,
    SpatialPrior,
    ThetaStandardization,
)
from bayesiancalibration.targets import CalibrationTarget
from bayesiancalibration.transforms import SiteCoordinates


def make_fixture(bounded=False, loading_only=False):
    library = np.array([[-1, -0.7], [0.9, -0.4], [0.1, 1.1], [-0.8, 0.8]])
    standardization = ThetaStandardization.from_library(library)
    k = 1 if loading_only else 2
    gp = LibraryGP.from_data(
        standardization.to_standardized(library),
        np.array([[0.2, -0.4], [0.7, 0.1], [-0.3, 0.5], [0.1, 0.3]])[:, :k],
        [0.8, 1.2],
    )
    prior = SpatialPrior.from_standardization(
        standardization, physical_center=[0.1, -0.2], V_theta_0=np.eye(2),
        nu_theta_0=7.0, S_theta_0=np.eye(2),
    )
    coordinates = SiteCoordinates.from_physical_bounds(
        standardization,
        **({"l": [-1.8, -1.2], "u": [1.7, 1.8]} if bounded else {}),
    )
    target = CalibrationTarget.from_data(
        gp, coordinates, prior, np.linspace(-0.1, 0.3, 2*k),
        np.diag(np.linspace(0.8, 1.3, 2*k)), [[0, 0], [6, 0]],
        (1,) if loading_only else (1, 1),
        alpha_y_0=np.full(k, 3.0), beta_y_0=np.full(k, 0.2),
        alpha_c_0=np.full(k, 3.5), beta_c_0=np.full(k, 0.3),
    )
    state = CalibrationState(
        jnp.array([[-0.3, 0.2], [0.5, -0.1]]),
        jnp.linspace(-0.15, 0.25, 2*k), jnp.linspace(-0.03, 0.04, k),
        jnp.linspace(0.09, 0.15, k), jnp.array([0.2, -0.1]),
        jnp.array([[0.8, 0.15], [0.15, 0.6]]), jnp.linspace(0.4, 1.2, k),
    )
    return target, state, jnp.tile(0.1 * jnp.eye(2), (2, 1, 1))


class GibbsSweepTest(unittest.TestCase):
    def assert_tree_equal(self, first, second, exact=False):
        for a, b in zip(jax.tree.leaves(first), jax.tree.leaves(second)):
            if exact:
                np.testing.assert_array_equal(a, b)
            else:
                np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-12)

    def test_full_schedule_matches_separate_checked_updates(self):
        for bounded in (False, True):
            for loading_only in (False, True):
                with self.subTest(bounded=bounded, loading_only=loading_only):
                    target, state, proposal = make_fixture(bounded, loading_only)
                    key = jax.random.key(900)
                    next_key, kd, ky, km, kS, kc, kt, kf = jax.random.split(key, 8)
                    delta = update_discrepancy(kd, target, state.c_f, state.sigma_y2)
                    noise = update_branch_noise(ky, target, state.c_f, delta)
                    mean = update_spatial_mean(km, target, state.eta, state.Sigma_theta)
                    covariance = update_spatial_covariance(kS, target, state.eta, mean)
                    variances = update_coefficient_variances(
                        kc, target, state.eta, state.c_f
                    )
                    eta, theta_info = update_collapsed_random_walk(
                        kt, target, state.eta, delta, noise, mean,
                        covariance, variances, proposal,
                    )
                    c_f = refresh_field_coefficients(
                        kf, target, eta, delta, noise, variances
                    )
                    expected = CalibrationState(
                        eta, c_f, delta, noise, mean, covariance, variances
                    )
                    actual, key_new, info = jax.jit(
                        lambda key, state: collapsed_gibbs_sweep(
                            key, target, state, proposal
                        )
                    )(key, state)
                    self.assert_tree_equal(actual, expected)
                    self.assert_tree_equal(info.theta, theta_info)
                    np.testing.assert_array_equal(
                        jax.random.key_data(key_new), jax.random.key_data(next_key)
                    )
                    np.testing.assert_allclose(
                        info.full_joint_logdensity,
                        target.full_joint_uncollapsed(*actual), rtol=1e-12,
                    )
                    self.assertFalse(np.array_equal(actual.c_f, state.c_f))

    def test_rejected_theta_still_refreshes_coefficients_and_records_state(self):
        target, state, proposal = make_fixture(True)
        chain = initialize_random_walk_chain(
            target, state, jax.random.key(901), proposal * 1e100
        )
        result, samples, infos = run_fixed_random_walk(target, chain, 2)
        self.assertEqual(result.iteration, 2)
        np.testing.assert_array_equal(infos.theta.is_accepted, np.zeros((2, 2)))
        np.testing.assert_array_equal(samples.eta, np.tile(state.eta, (2, 1, 1)))
        self.assertFalse(np.array_equal(samples.c_f[0], samples.c_f[1]))
        self.assertEqual(samples.c_f.shape, (2, 4))
        for i in range(2):
            np.testing.assert_allclose(
                infos.full_joint_logdensity[i], target.full_joint_uncollapsed(
                    *(value[i] for value in samples)
                ), rtol=1e-12,
            )
        self.assertEqual(chain.iteration, 0)
        np.testing.assert_array_equal(chain.model_state.c_f, state.c_f)

    def test_invalid_chain_or_sweep_count_cannot_record_partial_state(self):
        target, state, proposal = make_fixture()
        chain = initialize_random_walk_chain(
            target, state, jax.random.key(902), proposal
        )
        for modified in (
            replace(chain, iteration=-1), replace(chain, iteration=True),
            replace(chain, phase="warmup"),
            replace(chain, key=jax.random.split(chain.key, 2)),
            replace(chain, model_state=state._replace(c_f=state.c_f[:1])),
            replace(chain, model_state=state._replace(sigma_c2=-state.sigma_c2)),
            replace(chain, model_state=state._replace(Sigma_theta=-state.Sigma_theta)),
            replace(chain, V_prop=-proposal),
        ):
            with self.assertRaises(ValueError):
                validate_random_walk_chain(target, modified)
        for count in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                run_fixed_random_walk(target, chain, count)
        with patch("bayesiancalibration.mcmc.collapsed_gibbs_sweep",
                   side_effect=FloatingPointError("numerical failure")):
            with self.assertRaises(FloatingPointError):
                run_fixed_random_walk(target, chain, 1)
        self.assertEqual(chain.iteration, 0)


class NumericalBatchingTest(unittest.TestCase):
    assert_tree_equal = GibbsSweepTest.assert_tree_equal

    def test_batches_match_one_run_for_typed_and_legacy_keys(self):
        for bounded, loading_only, key in (
            (False, False, jax.random.key(910)),
            (True, True, jax.random.PRNGKey(911)),
        ):
            with self.subTest(bounded=bounded, loading_only=loading_only):
                target, state, proposal = make_fixture(bounded, loading_only)
                initial = initialize_random_walk_chain(target, state, key, proposal)
                complete, all_samples, all_info = run_fixed_random_walk(target, initial, 5)
                part, first_samples, first_info = run_fixed_random_walk(target, initial, 2)
                final, last_samples, last_info = run_fixed_random_walk(target, part, 3)
                self.assert_tree_equal(jax.tree.map(
                    lambda a, b: jnp.concatenate((a, b)), first_samples, last_samples
                ), all_samples, True)
                self.assert_tree_equal(jax.tree.map(
                    lambda a, b: jnp.concatenate((a, b)), first_info, last_info
                ), all_info, True)
                self.assert_tree_equal(final.model_state, complete.model_state, True)
                np.testing.assert_array_equal(
                    jax.random.key_data(final.key), jax.random.key_data(complete.key)
                )
                self.assertEqual(final.key.dtype, initial.key.dtype)
                self.assertEqual(final.iteration, complete.iteration)


class JointStationarityTest(unittest.TestCase):
    def test_complete_joint_distribution_is_preserved_in_both_bounds_modes(self):
        """Independent importance posterior checks the partial-collapse schedule.

        Draw the entire library-conditioned hierarchy, apply global site
        support, weight by the analytically integrated observation density,
        and independently draw c_f from its scalar observation conditional.
        No project sampling kernels construct the reference distribution.
        """

        rng = np.random.default_rng(920)
        library = np.array([[-2.0], [0.0], [2.0]])
        standardization = ThetaStandardization.from_library(library)
        theta_s = np.asarray(standardization.to_standardized(library))[:, 0]
        F_s = np.array([[-0.55], [0.2], [1.2]])
        gp = LibraryGP.from_data(theta_s[:, None], F_s, [0.65])
        prior = SpatialPrior.from_standardization(
            standardization, physical_center=[0.2], V_theta_0=[[0.25]],
            nu_theta_0=8.0, S_theta_0=[[1.5]],
        )
        C_ss = np.exp(-0.5 * ((theta_s[:, None] - theta_s[None, :]) / 0.65)**2)
        alpha_s = np.linalg.solve(C_ss, F_s)[:, 0]
        q_s = float(F_s[:, 0] @ alpha_s)
        N, M = 150_000, 40_000
        # Conditioning on the fixed library changes the coefficient prior.
        sigma_c2 = invgamma.rvs(3.5 + 1.5, scale=0.3 + q_s/2, size=N,
                               random_state=rng)
        sigma_y2 = invgamma.rvs(8.0, scale=1.0, size=N, random_state=rng)
        Sigma_theta = invgamma.rvs(4.0, scale=0.75, size=N, random_state=rng)
        mu_theta = rng.normal(0.1, 0.5, size=N)
        theta = rng.normal(mu_theta, np.sqrt(Sigma_theta))
        delta = rng.normal(0.03, 0.2, size=N)
        C_fs = np.exp(-0.5 * ((theta[:, None] - theta_s[None, :]) / 0.65)**2)
        m_f = C_fs @ alpha_s
        C_f = 1 - np.sum(C_fs * np.linalg.solve(C_ss, C_fs.T).T, axis=1)
        V_f = C_f * sigma_c2
        V_y = V_f + sigma_y2
        likelihood = norm.pdf(0.45, m_f + delta, np.sqrt(V_y))
        posterior_mean = m_f + V_f / V_y * (0.45 - m_f - delta)
        posterior_variance = V_f * sigma_y2 / V_y
        c_f = rng.normal(posterior_mean, np.sqrt(posterior_variance))

        def summaries(theta, c_f, delta, noise, mean, covariance, variances):
            return np.column_stack((
                theta, c_f, delta, noise, mean, covariance, variances,
                theta*c_f, theta*mean, c_f*delta, theta**2, c_f**2,
            ))

        reference_values = summaries(
            theta, c_f, delta, sigma_y2, mu_theta, Sigma_theta, sigma_c2
        )
        for bounded in (False, True):
            with self.subTest(bounded=bounded):
                coordinates = SiteCoordinates.from_physical_bounds(
                    standardization,
                    **({"l": [-1.3], "u": [1.6]} if bounded else {}),
                )
                target = CalibrationTarget.from_data(
                    gp, coordinates, prior, [0.45], [[1.0]], [[0, 0]], (1,),
                    m_delta_0=[0.03], V_delta_0=[[0.04]],
                    alpha_y_0=[8.0], beta_y_0=[1.0],
                    alpha_c_0=[3.5], beta_c_0=[0.3],
                )
                weights = likelihood.copy()
                if bounded:
                    # Reject the whole hierarchy outside site support. This
                    # is the globally restricted joint prior in the notes.
                    weights *= (theta > -0.65) & (theta < 0.8)
                weights /= weights.sum()
                self.assertGreater(1 / np.sum(weights**2), N * 0.3)
                reference = weights @ reference_values
                reference_se = np.sqrt(np.sum(
                    weights[:, None]**2 * (reference_values - reference)**2, axis=0
                ))
                indices = rng.choice(N, size=M, p=weights)
                state = CalibrationState(
                    coordinates.theta_tilde_to_eta(
                        jnp.asarray(theta[indices]).reshape(M, 1, 1)
                    ),
                    jnp.asarray(c_f[indices, None]), jnp.asarray(delta[indices, None]),
                    jnp.asarray(sigma_y2[indices, None]),
                    jnp.asarray(mu_theta[indices, None]),
                    jnp.asarray(Sigma_theta[indices, None, None]),
                    jnp.asarray(sigma_c2[indices, None]),
                )
                keys = jax.random.split(jax.random.key(921 + int(bounded)), M)
                kernel = jax.jit(jax.vmap(lambda key, state: collapsed_gibbs_sweep(
                    key, target, state, jnp.array([[[0.5]]])
                )))
                for sweep in range(4):
                    state, keys, info = kernel(keys, state)
                    self.assertTrue(np.all(np.isfinite(info.full_joint_logdensity)))
                    values = summaries(
                        np.asarray(coordinates.eta_to_theta_tilde(
                            state.eta
                        )).reshape(M),
                        *(np.asarray(value).reshape(M) for value in state[1:]),
                    )
                    se = values.std(axis=0, ddof=1) / np.sqrt(M)
                    combined_se = np.sqrt(se**2 + reference_se**2)
                    np.testing.assert_array_less(
                        np.abs(values.mean(axis=0) - reference), 6 * combined_se
                    )


if __name__ == "__main__":
    unittest.main()
