"""Dense independent references for Stage 3 coefficient-GP calculations."""

import unittest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import multivariate_normal

from bayesiancalibration.gp import (
    LibraryGP,
    fit_library_length_scales,
    profile_log_likelihood,
)


def _numpy_kernel(a: np.ndarray, b: np.ndarray, length: np.ndarray) -> np.ndarray:
    """Independent NumPy reference for the source-note covariance."""

    return np.exp(
        -0.5 * np.sum(((a[:, None, :] - b[None, :, :]) / length) ** 2, axis=2)
    )


class LibraryConditioningTest(unittest.TestCase):
    def setUp(self) -> None:
        self.theta_s = np.array([[-1.0, -0.7], [0.9, -0.4], [0.1, 1.1]])
        self.theta_f = np.array([[-0.3, 0.2], [0.5, -0.1]])
        self.F_s = np.array([[0.2, -0.4], [0.7, 0.1], [-0.3, 0.5]])
        self.lambda_c = np.array([0.8, 1.2])
        self.sigma_c2 = np.array([0.4, 1.7])
        self.gp = LibraryGP.from_data(self.theta_s, self.F_s, self.lambda_c)

    def test_dense_joint_conditional_and_library_density(self) -> None:
        self.gp.validate_field_sites(self.theta_f)
        m_f_given_s, C_f_given_s, Sigma_f_given_s = (
            self.gp.conditional_moments(
                jnp.asarray(self.theta_f), jnp.asarray(self.sigma_c2)
            )
        )
        C_ss = _numpy_kernel(self.theta_s, self.theta_s, self.lambda_c)
        np.testing.assert_allclose(np.diag(self.gp.C_ss), np.ones(3))
        C_fs = _numpy_kernel(self.theta_f, self.theta_s, self.lambda_c)
        C_ff = _numpy_kernel(self.theta_f, self.theta_f, self.lambda_c)
        reference_m = C_fs @ np.linalg.solve(C_ss, self.F_s)
        reference_C = C_ff - C_fs @ np.linalg.solve(C_ss, C_fs.T)
        np.testing.assert_allclose(m_f_given_s, reference_m.reshape(-1), rtol=1e-12)
        np.testing.assert_allclose(C_f_given_s, reference_C, rtol=1e-12)
        np.testing.assert_allclose(
            Sigma_f_given_s, np.kron(reference_C, np.diag(self.sigma_c2)), rtol=1e-12
        )

        c_s = self.F_s.reshape(-1)
        c_f = np.array([0.15, -0.25, 0.6, 0.35])
        C_joint = np.block([[C_ff, C_fs], [C_fs.T, C_ss]])
        Sigma_c = np.diag(self.sigma_c2)
        joint = multivariate_normal.logpdf(
            np.concatenate([c_f, c_s]), cov=np.kron(C_joint, Sigma_c)
        )
        reference_library = multivariate_normal.logpdf(
            c_s, cov=np.kron(C_ss, Sigma_c)
        )
        np.testing.assert_allclose(
            self.gp.library_logpdf(jnp.asarray(self.sigma_c2)), reference_library
        )
        np.testing.assert_allclose(
            joint,
            reference_library
            + multivariate_normal.logpdf(c_f, mean=reference_m.reshape(-1),
                                         cov=np.kron(reference_C, Sigma_c)),
        )

    def test_jit_variance_dependency_and_numerical_jitter(self) -> None:
        compiled = jax.jit(self.gp.conditional_moments)
        m_1, C_1, Sigma_1 = compiled(self.theta_f, self.sigma_c2)
        m_2, C_2, Sigma_2 = compiled(self.theta_f, self.sigma_c2 * 2.0)
        np.testing.assert_allclose(m_1, m_2)
        np.testing.assert_allclose(C_1, C_2)
        np.testing.assert_allclose(Sigma_2, 2.0 * Sigma_1)

        with_jitter = LibraryGP.from_data(
            self.theta_s, self.F_s, self.lambda_c, jitter=1e-10
        )
        np.testing.assert_allclose(with_jitter.C_ss, self.gp.C_ss)
        m_j, C_j, _ = with_jitter.conditional_moments(self.theta_f, self.sigma_c2)
        np.testing.assert_allclose(m_j, m_1, atol=1e-8)
        np.testing.assert_allclose(C_j, C_1, atol=1e-8)
        self.assertEqual(with_jitter.jitter, 1e-10)

    def test_structural_singularity_is_not_hidden_by_jitter(self) -> None:
        with self.assertRaisesRegex(ValueError, "r >= 2"):
            LibraryGP.from_data(self.theta_s[:1], self.F_s[:1], self.lambda_c)
        with self.assertRaisesRegex(ValueError, "Repeated library"):
            LibraryGP.from_data(
                np.vstack([self.theta_s, self.theta_s[0]]),
                np.vstack([self.F_s, self.F_s[0]]),
                self.lambda_c,
                jitter=1e-6,
            )
        with self.assertRaisesRegex(ValueError, "coincides"):
            self.gp.validate_field_sites(self.theta_s[:1])
        with self.assertRaisesRegex(ValueError, "Repeated field"):
            self.gp.validate_field_sites(np.vstack([self.theta_f[0], self.theta_f[0]]))
        with self.assertRaisesRegex(ValueError, "singular"):
            LibraryGP.from_data(
                np.array([[0.0], [1e-9], [1.0]]),
                self.F_s,
                np.array([1.0]),
                jitter=1e-6,
            )


class ProfileFittingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.theta_s = np.array([[-1.8], [-1.2], [-0.6], [0.0], [0.7], [1.3], [2.0]])
        rng = np.random.default_rng(47)
        C_ss = _numpy_kernel(self.theta_s, self.theta_s, np.array([0.85]))
        self.F_s = np.linalg.cholesky(C_ss) @ rng.normal(size=(7, 6))

    def test_profile_matches_full_likelihood_and_variance_maximum(self) -> None:
        xi = jnp.array([-0.3])
        lambda_c = np.exp(np.asarray(xi))
        C_ss = _numpy_kernel(self.theta_s, self.theta_s, lambda_c)
        q = np.sum(self.F_s * np.linalg.solve(C_ss, self.F_s), axis=0)
        sigma_c2_hat = q / self.theta_s.shape[0]
        profile = profile_log_likelihood(xi, self.theta_s, self.F_s)
        full_likelihood = sum(
            multivariate_normal.logpdf(self.F_s[:, j], cov=sigma_c2_hat[j] * C_ss)
            for j in range(self.F_s.shape[1])
        )
        r, k = self.F_s.shape
        np.testing.assert_allclose(
            profile,
            full_likelihood + 0.5 * r * k * (np.log(2.0 * np.pi) + 1.0),
        )

        result = minimize_scalar(
            lambda log_sigma_c2: -multivariate_normal.logpdf(
                self.F_s[:, 0], cov=np.exp(log_sigma_c2) * C_ss
            ),
            bounds=(-5.0, 5.0), method="bounded",
        )
        self.assertTrue(result.success)
        np.testing.assert_allclose(np.exp(result.x), sigma_c2_hat[0], rtol=1e-5)

        loading_only = profile_log_likelihood(xi, self.theta_s, self.F_s[:, :3])
        loading_reference = sum(
            multivariate_normal.logpdf(self.F_s[:, j], cov=sigma_c2_hat[j] * C_ss)
            for j in range(3)
        )
        np.testing.assert_allclose(
            loading_only,
            loading_reference + 0.5 * r * 3 * (np.log(2.0 * np.pi) + 1.0),
        )

    def test_log_length_gradient_and_fit_diagnostics(self) -> None:
        theta_2d = np.column_stack([self.theta_s[:, 0], np.sin(self.theta_s[:, 0])])
        xi = jnp.array([-0.3, 0.1])
        objective = lambda x: profile_log_likelihood(x, theta_2d, self.F_s)
        gradient = jax.jit(jax.grad(objective))(xi)
        h = 1e-5
        finite_difference = np.array([
            (float(objective(xi.at[q].add(h))) - float(objective(xi.at[q].add(-h))))
            / (2.0 * h)
            for q in range(2)
        ])
        np.testing.assert_allclose(gradient, finite_difference, rtol=1e-5, atol=1e-5)

        fit = fit_library_length_scales(
            self.theta_s, self.F_s,
            starts=np.array([[-1.0], [0.2]]),
            gtol=1e-6, ftol=1e-10, maxiter=300,
        )
        self.assertTrue(all(attempt.success for attempt in fit.attempts))
        self.assertEqual(len(fit.attempts), 2)
        np.testing.assert_allclose(fit.lambda_c, [0.85612091], rtol=1e-5)
        self.assertGreater(
            fit.objective,
            float(profile_log_likelihood(jnp.array([-1.0]), self.theta_s, self.F_s)),
        )
        self.assertTrue(bool(jnp.all(fit.profiled_variances > 0)))
        self.assertEqual(fit.gtol, 1e-6)
        self.assertEqual(fit.maxiter, 300)

        gp = LibraryGP.from_data(self.theta_s, self.F_s, fit.lambda_c)
        m_1, C_1, Sigma_1 = gp.conditional_moments(jnp.array([[-0.9]]),
                                                    jnp.ones(6))
        m_2, C_2, Sigma_2 = gp.conditional_moments(jnp.array([[-0.9]]),
                                                    2.0 * jnp.ones(6))
        np.testing.assert_allclose(m_1, m_2)
        np.testing.assert_allclose(C_1, C_2)
        np.testing.assert_allclose(Sigma_2, 2.0 * Sigma_1)

    def test_zero_column_and_invalid_fit_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero coefficient column"):
            fit_library_length_scales(
                self.theta_s, np.zeros_like(self.F_s),
                starts=np.array([[0.0]]), gtol=1e-6, ftol=1e-9, maxiter=100,
            )
        with self.assertRaisesRegex(ValueError, "starts"):
            fit_library_length_scales(
                self.theta_s, self.F_s,
                starts=np.array([]), gtol=1e-6, ftol=1e-9, maxiter=100,
            )
        with self.assertRaisesRegex(ValueError, "singular"):
            fit_library_length_scales(
                self.theta_s, self.F_s,
                starts=np.array([[10.0]]), gtol=1e-6, ftol=1e-9,
                maxiter=100, jitter=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
