"""Independent references for the remaining Stage 6 conditionals."""

import unittest
from dataclasses import replace

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from scipy.linalg import block_diag
from scipy.stats import invgamma, invwishart, multivariate_normal

from bayesiancalibration.gibbs import (
    branch_noise_conditional_parameters,
    coefficient_variance_conditional_parameters,
    sample_branch_noise,
    sample_coefficient_variances,
    sample_inverse_gamma,
    sample_inverse_wishart,
    sample_spatial_covariance,
    sample_spatial_mean,
    spatial_covariance_conditional_parameters,
    spatial_mean_conditional_moments,
    update_branch_noise,
    update_coefficient_variances,
    update_spatial_covariance,
    update_spatial_mean,
)
from bayesiancalibration.gp import LibraryGP
from bayesiancalibration.state import SpatialPrior, ThetaStandardization
from bayesiancalibration.targets import CalibrationTarget
from bayesiancalibration.transforms import SiteCoordinates


class GibbsFixture(unittest.TestCase):
    """Shared deterministic model fixture keeps Stage 6 references comparable."""

    def setUp(self) -> None:
        self.eta = jnp.array([[-0.3, 0.2], [0.5, -0.1], [0.0, 0.7]])
        self.mu_theta = jnp.array([0.2, -0.1])
        self.Sigma_theta = jnp.array([[0.8, 0.15], [0.15, 0.6]])

    def target(self, loading_only=False, bounded=False) -> CalibrationTarget:
        """Make varying five-column QR designs and explicit nondefault priors."""

        rng = np.random.default_rng(682)
        library = np.array([
            [-1.0, -0.7], [0.9, -0.4], [0.1, 1.1], [-0.8, 0.8]
        ])
        standardization = ThetaStandardization.from_library(library)
        sizes = (5,) if loading_only else (5, 5)
        k = sum(sizes)
        gp = LibraryGP.from_data(
            standardization.to_standardized(library),
            rng.normal(scale=0.3, size=(4, k)), [0.8, 1.2],
        )
        prior = SpatialPrior.from_standardization(
            standardization, physical_center=[0.1, -0.2],
            V_theta_0=np.array([[2.0, 0.3], [0.3, 1.4]]),
            nu_theta_0=9.5, S_theta_0=np.array([[1.1, 0.2], [0.2, 0.8]]),
        )
        coordinates = SiteCoordinates.from_physical_bounds(
            standardization,
            **({"l": [-1.8, -1.2], "u": [1.7, 1.8]} if bounded else {}),
        )
        R_sites = []
        for i in range(3):
            R_sites.append(block_diag(*[
                np.triu(rng.normal(scale=0.2, size=(size, size)), 1)
                + np.diag(np.linspace(0.7, 1.3, size) + 0.1 * i)
                for size in sizes
            ]))
        return CalibrationTarget.from_data(
            gp, coordinates, prior, rng.normal(scale=0.5, size=3 * k),
            block_diag(*R_sites), [[0, 0], [6, 0], [0, 6]], sizes,
            alpha_y_0=[2.3] if loading_only else [2.3, 3.2],
            beta_y_0=[0.04] if loading_only else [0.04, 0.11],
            alpha_c_0=[2.5] if loading_only else [2.5, 3.7],
            beta_c_0=[0.2] if loading_only else [0.2, 0.6],
        )

    def state(self, target: CalibrationTarget) -> tuple:
        """Return site-major c_f, shared delta, active noise,
        and coefficient sigma_c2.
        """

        k = target.gp.F_s.shape[1]
        return (
            jnp.asarray(0.3 * np.sin(np.arange(3 * k) + 0.4)),
            jnp.linspace(-0.03, 0.04, k),
            jnp.array([0.07] if len(target.branch_sizes) == 1 else [0.07, 0.29]),
            jnp.linspace(0.4, 1.2, k),
        )

    def joint(self, target, c_f, delta, noise, sigma_c2, **changes) -> jax.Array:
        """Hold all other factors fixed for full-joint conditional ratio checks."""

        values = dict(
            eta=self.eta, c_f=c_f, delta=delta, sigma_y2=noise,
            mu_theta=self.mu_theta, Sigma_theta=self.Sigma_theta, sigma_c2=sigma_c2,
        )
        values.update(changes)
        return target.full_joint_uncollapsed(**values)

    def check_ig_draws(self, draws, shape, scale) -> None:
        """Check means and CDF probabilities, including tails, against SciPy."""

        self.assertEqual(draws.dtype, np.dtype("float64"))
        self.assertTrue(np.all(np.isfinite(draws)) and np.all(draws > 0))
        N = draws.shape[0]
        expected_mean = invgamma.mean(shape, scale=scale)
        mean_se = np.sqrt(invgamma.var(shape, scale=scale) / N)
        np.testing.assert_array_less(
            np.abs(draws.mean(axis=0) - expected_mean) / mean_se, 6.0
        )
        for p in (0.1, 0.5, 0.9):
            thresholds = invgamma.ppf(p, shape, scale=scale)
            error = np.abs((draws < thresholds).mean(axis=0) - p)
            np.testing.assert_array_less(error, 6 * np.sqrt(p * (1 - p) / N))


class BranchNoiseTest(GibbsFixture):
    def reference(self, target, c_f, delta) -> tuple:
        """Accumulate each site's/branch's residual explicitly with NumPy."""

        k = sum(target.branch_sizes)
        shape = np.asarray(target.alpha_y_0).copy()
        scale = np.asarray(target.beta_y_0).copy()
        for i in range(3):
            sl = slice(i * k, (i + 1) * k)
            residual = (
                np.asarray(target.y_tilde)[sl]
                - np.asarray(target.R)[sl, sl] @ (np.asarray(c_f)[sl] + delta)
            )
            start = 0
            for b, size in enumerate(target.branch_sizes):
                e = residual[start:start + size]
                shape[b] += 0.5 * size
                scale[b] += 0.5 * np.dot(e, e)
                start += size
        return shape, scale

    def test_parameters_joint_ratios_and_loading_only(self) -> None:
        for loading_only in (False, True):
            for bounded in (False, True):
                with self.subTest(loading_only=loading_only, bounded=bounded):
                    target = self.target(loading_only, bounded)
                    c_f, delta, noise, sigma_c2 = self.state(target)
                    shape, scale = self.reference(target, c_f, delta)
                    actual = jax.jit(
                        branch_noise_conditional_parameters,
                        static_argnames=("branch_sizes",),
                    )(
                        target.y_tilde, c_f, delta, target.R, target.branch_sizes,
                        target.alpha_y_0, target.beta_y_0,
                    )
                    np.testing.assert_allclose(actual[0], shape, rtol=1e-13)
                    np.testing.assert_allclose(actual[1], scale, rtol=1e-13)
                    for b in range(len(target.branch_sizes)):
                        candidate = noise.at[b].multiply(1.8)
                        actual_ratio = (
                            self.joint(target, c_f, delta, candidate, sigma_c2)
                            - self.joint(target, c_f, delta, noise, sigma_c2)
                        )
                        expected = (
                            invgamma.logpdf(candidate[b], shape[b], scale=scale[b])
                            - invgamma.logpdf(noise[b], shape[b], scale=scale[b])
                        )
                        np.testing.assert_allclose(actual_ratio, expected, atol=1e-9)

    def test_empirical_draws_and_independent_branches(self) -> None:
        for loading_only in (False, True):
            target = self.target(loading_only)
            c_f, delta, _, _ = self.state(target)
            shape, scale = self.reference(target, c_f, delta)
            keys = jax.random.split(jax.random.key(720 + int(loading_only)), 40_000)
            draws = np.asarray(jax.jit(jax.vmap(lambda key: sample_branch_noise(
                key, target.y_tilde, c_f, delta, target.R, target.branch_sizes,
                target.alpha_y_0, target.beta_y_0,
            )))(keys))
            self.assertEqual(draws.shape, (40_000, len(target.branch_sizes)))
            self.check_ig_draws(draws, shape, scale)
            if not loading_only:
                self.assertLess(abs(np.corrcoef(draws.T)[0, 1]), 0.04)

    def test_inverse_gamma_tail_shapes_and_scale_range(self) -> None:
        shape = jnp.array([0.7, 1.01, 5.2])
        scale = jnp.array([1e-200, 1e200, 0.13])
        keys = jax.random.split(jax.random.key(723), 20_000)
        draws = np.asarray(jax.jit(jax.vmap(
            lambda key: sample_inverse_gamma(key, shape, scale)
        ))(keys))
        self.assertTrue(np.all(np.isfinite(draws)) and np.all(draws > 0))
        normalized = draws / np.asarray(scale)
        for p in (0.1, 0.5, 0.9):
            error = np.abs((normalized < invgamma.ppf(p, shape)).mean(axis=0) - p)
            np.testing.assert_array_less(error, 6 * np.sqrt(p * (1 - p) / len(keys)))

    def test_current_dependencies_keys_and_failure_reporting(self) -> None:
        target = self.target()
        c_f, delta, _, _ = self.state(target)
        key = jax.random.key(724)
        first = update_branch_noise(key, target, c_f, delta)
        np.testing.assert_array_equal(
            first, update_branch_noise(key, target, c_f, delta)
        )
        for coefficients, discrepancy in ((c_f + 0.2, delta), (c_f, delta + 0.2)):
            self.assertFalse(np.allclose(
                first, update_branch_noise(key, target, coefficients, discrepancy)
            ))
        self.assertFalse(np.array_equal(
            first, update_branch_noise(jax.random.split(key)[1], target, c_f, delta)
        ))
        for coefficients, discrepancy in (
            (c_f[:-1], delta), (c_f.at[0].set(jnp.nan), delta),
            (c_f, delta[:-1]), (c_f, delta.at[0].set(jnp.inf)),
        ):
            with self.assertRaises(ValueError):
                update_branch_noise(key, target, coefficients, discrepancy)
        with self.assertRaises(FloatingPointError):
            update_branch_noise(key, replace(target, R=target.R * 1e200), c_f, delta)


class SpatialMeanTest(GibbsFixture):
    def reference(self, target, theta_tilde, Sigma_theta) -> tuple:
        """Condition the independent dense joint Gaussian for mu and theta."""

        n, d = theta_tilde.shape
        H = np.kron(np.ones((n, 1)), np.eye(d))
        m_0 = np.asarray(target.spatial_prior.m_theta_0)
        V_0 = np.asarray(target.spatial_prior.V_theta_0)
        K = np.kron(target.C_theta, Sigma_theta)
        cross = V_0 @ H.T
        V_field = K + H @ V_0 @ H.T
        m_theta = m_0 + cross @ np.linalg.solve(V_field, theta_tilde.ravel() - H @ m_0)
        V_theta = V_0 - cross @ np.linalg.solve(V_field, cross.T)
        return m_theta, V_theta

    def test_dense_moments_and_full_joint_ratios_in_both_modes(self) -> None:
        for bounded in (False, True):
            target = self.target(bounded=bounded)
            theta = target.coordinates.eta_to_theta_tilde(self.eta)
            expected_m, expected_V = self.reference(target, theta, self.Sigma_theta)
            actual = jax.jit(spatial_mean_conditional_moments)(
                theta, target.C_theta, self.Sigma_theta,
                target.spatial_prior.m_theta_0, target.spatial_prior.V_theta_0,
            )
            np.testing.assert_allclose(actual[0], expected_m, atol=1e-12)
            np.testing.assert_allclose(actual[1], expected_V, atol=1e-12)
            c_f, delta, noise, sigma_c2 = self.state(target)
            mu_a = np.array([3.5, -4.2])
            mu_b = np.array([3.7, -3.9])
            actual_ratio = (
                self.joint(target, c_f, delta, noise, sigma_c2, mu_theta=mu_b)
                - self.joint(target, c_f, delta, noise, sigma_c2, mu_theta=mu_a)
            )
            expected = (
                multivariate_normal.logpdf(mu_b, expected_m, expected_V)
                - multivariate_normal.logpdf(mu_a, expected_m, expected_V)
            )
            np.testing.assert_allclose(actual_ratio, expected, atol=1e-9)

    def test_empirical_moments_and_unbounded_draws_with_bounded_sites(self) -> None:
        target = self.target(bounded=True)
        prior = replace(
            target.spatial_prior, m_theta_0=jnp.array([20.0, -20.0]),
            V_theta_0=0.1 * jnp.eye(2),
        )
        target = replace(target, spatial_prior=prior)
        theta = target.coordinates.eta_to_theta_tilde(self.eta)
        m_theta, V_theta = self.reference(target, theta, self.Sigma_theta)
        N = 40_000
        keys = jax.random.split(jax.random.key(730), N)
        draws = np.asarray(jax.jit(jax.vmap(lambda key: sample_spatial_mean(
            key, theta, target.C_theta, self.Sigma_theta,
            prior.m_theta_0, prior.V_theta_0,
        )))(keys))
        self.assertEqual(draws.dtype, np.dtype("float64"))
        mean_se = np.sqrt(np.diag(V_theta) / N)
        cov_se = np.sqrt((
            V_theta**2 + np.outer(np.diag(V_theta), np.diag(V_theta))
        ) / (N - 1))
        np.testing.assert_array_less(np.abs(draws.mean(axis=0) - m_theta) / mean_se, 6)
        np.testing.assert_array_less(
            np.abs(np.cov(draws, rowvar=False) - V_theta) / cov_se, 6
        )
        outside = np.any(
            (draws < target.coordinates.l_tilde) | (draws > target.coordinates.u_tilde),
            axis=1,
        )
        self.assertTrue(np.all(outside))
        checked = update_spatial_mean(keys[0], target, self.eta, self.Sigma_theta)
        np.testing.assert_allclose(checked, draws[0], atol=1e-12)

    def test_current_dependencies_and_key_reproducibility(self) -> None:
        target = self.target()
        key = jax.random.key(732)
        first = update_spatial_mean(key, target, self.eta, self.Sigma_theta)
        np.testing.assert_array_equal(
            first, update_spatial_mean(key, target, self.eta, self.Sigma_theta)
        )
        for eta, Sigma_theta in (
            (self.eta.at[0].add(0.2), self.Sigma_theta),
            (self.eta, self.Sigma_theta * 1.5),
        ):
            self.assertFalse(np.allclose(
                first, update_spatial_mean(key, target, eta, Sigma_theta)
            ))
        self.assertFalse(np.array_equal(
            first, update_spatial_mean(
                jax.random.split(key)[1], target, self.eta, self.Sigma_theta
            )
        ))

    def test_invalid_sites_and_covariance_are_reported(self) -> None:
        target = self.target()
        key = jax.random.key(733)
        for eta, Sigma in (
            (self.eta[:1], self.Sigma_theta),
            (self.eta.at[0, 0].set(jnp.nan), self.Sigma_theta),
            (self.eta, jnp.eye(3)),
            (self.eta, jnp.zeros((2, 2))),
            (self.eta, self.Sigma_theta.at[0, 1].set(0.2)),
            (self.eta, self.Sigma_theta.at[0, 0].set(jnp.inf)),
        ):
            with self.assertRaises(ValueError):
                update_spatial_mean(key, target, eta, Sigma)


class SpatialCovarianceTest(GibbsFixture):
    def reference(self, target, theta, mu_theta) -> tuple:
        """Use a dense NumPy spatial solve to check the sufficient statistic."""

        E = np.asarray(theta) - mu_theta
        return (
            target.spatial_prior.nu_theta_0 + len(theta),
            np.asarray(target.spatial_prior.S_theta_0)
            + E.T @ np.linalg.solve(target.C_theta, E),
        )

    def test_parameters_and_full_joint_ratios_in_both_modes(self) -> None:
        for bounded in (False, True):
            target = self.target(bounded=bounded)
            theta = target.coordinates.eta_to_theta_tilde(self.eta)
            # Spatial covariance conditioning also permits means outside site bounds.
            mu = jnp.array([3.5, -4.2])
            nu, S = self.reference(target, theta, mu)
            actual = jax.jit(spatial_covariance_conditional_parameters)(
                theta, target.C_theta, mu,
                target.spatial_prior.nu_theta_0, target.spatial_prior.S_theta_0,
            )
            np.testing.assert_allclose(actual[0], nu, rtol=1e-13)
            np.testing.assert_allclose(actual[1], S, rtol=1e-12)
            c_f, delta, noise, sigma_c2 = self.state(target)
            candidate = jnp.array([[1.1, 0.2], [0.2, 0.4]])
            actual_ratio = (
                self.joint(target, c_f, delta, noise, sigma_c2,
                           mu_theta=mu, Sigma_theta=candidate)
                - self.joint(target, c_f, delta, noise, sigma_c2, mu_theta=mu)
            )
            expected = (
                invwishart.logpdf(candidate, df=nu, scale=S)
                - invwishart.logpdf(self.Sigma_theta, df=nu, scale=S)
            )
            np.testing.assert_allclose(actual_ratio, expected, rtol=1e-10, atol=1e-9)

    def test_empirical_covariance_and_precision_means(self) -> None:
        target = self.target()
        theta = target.coordinates.eta_to_theta_tilde(self.eta)
        nu, S = self.reference(target, theta, self.mu_theta)
        N = 40_000
        keys = jax.random.split(jax.random.key(740), N)
        draws = np.asarray(jax.jit(jax.vmap(lambda key: sample_spatial_covariance(
            key, theta, target.C_theta, self.mu_theta,
            target.spatial_prior.nu_theta_0, target.spatial_prior.S_theta_0,
        )))(keys))
        self.assertEqual(draws.shape, (N, 2, 2))
        self.assertEqual(draws.dtype, np.dtype("float64"))
        self.assertTrue(np.all(np.isfinite(draws)))
        np.linalg.cholesky(draws)  # Every returned covariance must be SPD.
        expected_mean = invwishart.mean(df=nu, scale=S)
        mean_se = np.sqrt(invwishart.var(df=nu, scale=S) / N)
        np.testing.assert_array_less(
            np.abs(draws.mean(axis=0) - expected_mean) / mean_se, 6
        )
        precision = np.linalg.solve(draws, np.broadcast_to(np.eye(2), draws.shape))
        scale_precision = np.linalg.solve(S, np.eye(2))
        precision_se = np.sqrt(nu * (
            scale_precision**2
            + np.outer(np.diag(scale_precision), np.diag(scale_precision))
        ) / N)
        np.testing.assert_array_less(
            np.abs(precision.mean(axis=0) - nu * scale_precision) / precision_se, 6
        )
        # Any scalar projection has a known inverse-Gamma law.
        a = np.array([1.0, -0.4])
        projected = np.einsum('i,nij,j->n', a, draws, a)
        for p in (0.1, 0.5, 0.9):
            threshold = invgamma.ppf(p, (nu - 1) / 2, scale=a @ S @ a / 2)
            self.assertLess(
                abs((projected < threshold).mean() - p),
                6 * np.sqrt(p * (1 - p) / N),
            )

    def test_one_and_three_dimensions_and_noninteger_df(self) -> None:
        for nu, S in (
            (2.7, np.array([[1.3]])),
            (9.4, np.array([[1.0, 0.2, -0.1],
                            [0.2, 0.8, 0.15], [-0.1, 0.15, 0.6]])),
        ):
            d = len(S)
            N = 20_000
            keys = jax.random.split(jax.random.key(741 + d), N)
            draws = np.asarray(jax.jit(jax.vmap(
                lambda key: sample_inverse_wishart(key, nu, jnp.asarray(S))
            ))(keys))
            np.linalg.cholesky(draws)
            for p in (0.1, 0.5, 0.9):
                thresholds = invgamma.ppf(p, (nu - d + 1) / 2, scale=np.diag(S) / 2)
                errors = np.abs((np.diagonal(draws, axis1=1, axis2=2)
                                 < thresholds).mean(axis=0) - p)
                np.testing.assert_array_less(errors, 6 * np.sqrt(p * (1 - p) / N))

    def test_current_dependencies_keys_and_invalid_states(self) -> None:
        target = self.target(bounded=True)
        key = jax.random.key(745)
        first = update_spatial_covariance(key, target, self.eta, self.mu_theta)
        np.testing.assert_array_equal(
            first, update_spatial_covariance(key, target, self.eta, self.mu_theta)
        )
        theta = target.coordinates.eta_to_theta_tilde(self.eta)
        expected = jax.jit(sample_spatial_covariance)(
            key, theta, target.C_theta, self.mu_theta,
            target.spatial_prior.nu_theta_0, target.spatial_prior.S_theta_0,
        )
        np.testing.assert_allclose(first, expected, rtol=1e-12, atol=1e-12)
        for eta, mu_theta in (
            (self.eta.at[0].add(0.2), self.mu_theta),
            (self.eta, self.mu_theta + 0.2),
        ):
            self.assertFalse(np.allclose(
                first, update_spatial_covariance(key, target, eta, mu_theta)
            ))
        self.assertFalse(np.array_equal(
            first, update_spatial_covariance(
                jax.random.split(key)[1], target, self.eta, self.mu_theta
            )
        ))
        for eta, mu_theta in (
            (self.eta[:1], self.mu_theta),
            (self.eta.at[0, 0].set(jnp.inf), self.mu_theta),
            (self.eta, self.mu_theta[:1]),
            (self.eta, self.mu_theta.at[0].set(jnp.nan)),
        ):
            with self.assertRaises(ValueError):
                update_spatial_covariance(key, target, eta, mu_theta)
        with self.assertRaises(FloatingPointError):
            update_spatial_covariance(key, target, self.eta, jnp.full(2, 1e200))


class CoefficientVarianceTest(GibbsFixture):
    def reference(self, target, theta, c_f) -> tuple:
        """Use the entire joint field/library GP, without conditional algebra."""

        inputs = np.vstack((theta, target.gp.theta_s_tilde))
        diff = (inputs[:, None] - inputs[None, :]) / target.gp.lambda_c
        C_joint = np.exp(-0.5 * np.sum(np.square(diff), axis=-1))
        n = len(theta)
        r, k = target.gp.F_s.shape
        # Match the explicitly configured factor policy for positive jitter.
        C_joint[n:, n:] += target.gp.jitter * np.eye(r)
        F_joint = np.vstack((np.asarray(c_f).reshape(n, k), target.gp.F_s))
        q_joint = np.sum(F_joint * np.linalg.solve(C_joint, F_joint), axis=0)
        shape = np.empty(k)
        scale = np.empty(k)
        start = 0
        for b, size in enumerate(target.branch_sizes):
            shape[start:start + size] = target.alpha_c_0[b] + (r + n) / 2
            scale[start:start + size] = (
                target.beta_c_0[b] + q_joint[start:start + size] / 2
            )
            start += size
        return shape, scale

    def arrays(self, target, eta, c_f) -> tuple:
        """Retrieve variance-independent field moments and fixed library evidence."""

        theta = target.coordinates.eta_to_theta_tilde(eta)
        m, C, _ = target.gp.conditional_moments(theta, jnp.ones(target.gp.F_s.shape[1]))
        return (
            c_f, m, C, target.gp.q_s, target.gp.F_s.shape[0],
            target.branch_sizes, target.alpha_c_0, target.beta_c_0,
        )

    def test_complete_joint_gp_parameters_and_joint_ratios(self) -> None:
        for loading_only in (False, True):
            for bounded in (False, True):
                with self.subTest(loading_only=loading_only, bounded=bounded):
                    target = self.target(loading_only, bounded)
                    c_f, delta, noise, sigma_c2 = self.state(target)
                    theta = target.coordinates.eta_to_theta_tilde(self.eta)
                    shape, scale = self.reference(target, theta, c_f)
                    actual = jax.jit(
                        coefficient_variance_conditional_parameters,
                        static_argnames=("branch_sizes",),
                    )(
                        *self.arrays(target, self.eta, c_f)
                    )
                    np.testing.assert_allclose(actual[0], shape, rtol=1e-13)
                    np.testing.assert_allclose(actual[1], scale, rtol=1e-11)
                    for j in (0, len(sigma_c2) // 2, len(sigma_c2) - 1):
                        candidate = sigma_c2.at[j].multiply(1.7)
                        actual_ratio = (
                            self.joint(target, c_f, delta, noise, candidate)
                            - self.joint(target, c_f, delta, noise, sigma_c2)
                        )
                        expected = (
                            invgamma.logpdf(candidate[j], shape[j], scale=scale[j])
                            - invgamma.logpdf(sigma_c2[j], shape[j], scale=scale[j])
                        )
                        np.testing.assert_allclose(
                            actual_ratio, expected, rtol=1e-10, atol=1e-9
                        )

    def test_unequal_branch_sizes_and_prior_mapping(self) -> None:
        target = replace(self.target(), branch_sizes=(3, 7))
        c_f, delta, noise, sigma_c2 = self.state(target)
        theta = target.coordinates.eta_to_theta_tilde(self.eta)
        shape, scale = self.reference(target, theta, c_f)
        actual = jax.jit(
            coefficient_variance_conditional_parameters,
            static_argnames=("branch_sizes",),
        )(*self.arrays(target, self.eta, c_f))
        np.testing.assert_allclose(actual[0], shape, rtol=1e-13)
        np.testing.assert_allclose(actual[1], scale, rtol=1e-11)
        for j in (0, 2, 3, 9):
            candidate = sigma_c2.at[j].multiply(1.7)
            actual_ratio = (
                self.joint(target, c_f, delta, noise, candidate)
                - self.joint(target, c_f, delta, noise, sigma_c2)
            )
            expected = (
                invgamma.logpdf(candidate[j], shape[j], scale=scale[j])
                - invgamma.logpdf(sigma_c2[j], shape[j], scale=scale[j])
            )
            np.testing.assert_allclose(actual_ratio, expected, atol=1e-9)

    def test_empirical_draws_in_both_branch_modes(self) -> None:
        for loading_only in (False, True):
            target = self.target(loading_only)
            c_f, _, _, _ = self.state(target)
            theta = target.coordinates.eta_to_theta_tilde(self.eta)
            shape, scale = self.reference(target, theta, c_f)
            arrays = self.arrays(target, self.eta, c_f)
            N = 40_000
            keys = jax.random.split(jax.random.key(750 + int(loading_only)), N)
            draws = np.asarray(jax.jit(jax.vmap(
                lambda key: sample_coefficient_variances(key, *arrays)
            ))(keys))
            self.assertEqual(draws.shape, (N, target.gp.F_s.shape[1]))
            self.check_ig_draws(draws, shape, scale)
            precision_se = np.sqrt(shape / N) / scale
            np.testing.assert_array_less(
                np.abs((1 / draws).mean(axis=0) - shape / scale) / precision_se, 6
            )

    def test_current_dependencies_columns_and_fixed_library_cache(self) -> None:
        target = self.target()
        c_f, _, _, _ = self.state(target)
        key = jax.random.key(752)
        first = update_coefficient_variances(key, target, self.eta, c_f)
        np.testing.assert_array_equal(
            first, update_coefficient_variances(key, target, self.eta, c_f)
        )
        expected = jax.jit(
            sample_coefficient_variances, static_argnames=("branch_sizes",),
        )(
            key, *self.arrays(target, self.eta, c_f)
        )
        np.testing.assert_allclose(first, expected, rtol=1e-12)
        for eta, coefficients in (
            (self.eta.at[0].add(0.1), c_f), (self.eta, c_f + 0.1)
        ):
            self.assertFalse(np.allclose(
                first, update_coefficient_variances(key, target, eta, coefficients)
            ))
        self.assertFalse(np.array_equal(
            first, update_coefficient_variances(
                jax.random.split(key)[1], target, self.eta, c_f
            )
        ))
        changed = c_f.at[target.gp.F_s.shape[1] + 2].add(0.2)
        _, old_scale = coefficient_variance_conditional_parameters(
            *self.arrays(target, self.eta, c_f)
        )
        _, new_scale = coefficient_variance_conditional_parameters(
            *self.arrays(target, self.eta, changed)
        )
        np.testing.assert_array_equal(np.delete(old_scale, 2), np.delete(new_scale, 2))
        self.assertNotEqual(float(old_scale[2]), float(new_scale[2]))
        rebuilt_gp = LibraryGP.from_data(
            target.gp.theta_s_tilde, target.gp.F_s + 0.15, target.gp.lambda_c
        )
        rebuilt = replace(target, gp=rebuilt_gp)
        theta = rebuilt.coordinates.eta_to_theta_tilde(self.eta)
        _, expected_scale = self.reference(rebuilt, theta, c_f)
        _, rebuilt_scale = coefficient_variance_conditional_parameters(
            *self.arrays(rebuilt, self.eta, c_f)
        )
        np.testing.assert_allclose(rebuilt_scale, expected_scale, rtol=1e-11)
        self.assertFalse(np.allclose(old_scale, rebuilt_scale))
        self.assertFalse(np.allclose(target.gp.q_s, rebuilt_gp.q_s))

    def test_positive_fixed_jitter_matches_factor_based_joint(self) -> None:
        target = self.target()
        jittered_gp = LibraryGP.from_data(
            target.gp.theta_s_tilde, target.gp.F_s, target.gp.lambda_c, jitter=1e-6
        )
        target = replace(target, gp=jittered_gp)
        c_f, delta, noise, sigma_c2 = self.state(target)
        theta = target.coordinates.eta_to_theta_tilde(self.eta)
        shape, scale = self.reference(target, theta, c_f)
        actual = coefficient_variance_conditional_parameters(
            *self.arrays(target, self.eta, c_f)
        )
        np.testing.assert_allclose(actual[0], shape, rtol=1e-13)
        np.testing.assert_allclose(actual[1], scale, rtol=1e-11)
        candidate = sigma_c2 * 1.5
        ratio = (
            self.joint(target, c_f, delta, noise, candidate)
            - self.joint(target, c_f, delta, noise, sigma_c2)
        )
        expected = np.sum(
            invgamma.logpdf(candidate, shape, scale=scale)
            - invgamma.logpdf(sigma_c2, shape, scale=scale)
        )
        np.testing.assert_allclose(ratio, expected, rtol=1e-10, atol=1e-9)

    def test_invalid_states_and_structural_singularity_are_reported(self) -> None:
        target = self.target()
        c_f, _, _, _ = self.state(target)
        key = jax.random.key(754)
        for eta, coefficients in (
            (self.eta[:1], c_f), (self.eta.at[0, 0].set(jnp.nan), c_f),
            (self.eta.at[1].set(self.eta[0]), c_f),
            (self.eta.at[0].set(target.gp.theta_s_tilde[0]), c_f),
            (self.eta, c_f[:-1]), (self.eta, c_f.at[0].set(jnp.inf)),
        ):
            with self.assertRaises(ValueError):
                update_coefficient_variances(key, target, eta, coefficients)
        with self.assertRaises(FloatingPointError):
            update_coefficient_variances(
                key, target, self.eta, jnp.full(c_f.shape, 1e200)
            )
        jittered = replace(target, gp=LibraryGP.from_data(
            target.gp.theta_s_tilde, target.gp.F_s, target.gp.lambda_c, jitter=1e-6
        ))
        with self.assertRaisesRegex(ValueError, "coincides"):
            update_coefficient_variances(
                key, jittered, self.eta.at[0].set(target.gp.theta_s_tilde[0]), c_f
            )


if __name__ == "__main__":
    unittest.main()
