"""Stage 6a discrepancy references, joint ratios, and Gaussian draw checks."""

import unittest
from dataclasses import replace

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal

from bayesiancalibration.gibbs import (
    discrepancy_conditional_moments,
    sample_discrepancy,
    update_discrepancy,
)
from bayesiancalibration.gp import LibraryGP
from bayesiancalibration.state import SpatialPrior, ThetaStandardization
from bayesiancalibration.targets import CalibrationTarget
from bayesiancalibration.transforms import SiteCoordinates


class DiscrepancyUpdateTest(unittest.TestCase):
    def _target(
        self, loading_only: bool, bounded: bool = False, default_prior: bool = False
    ) -> CalibrationTarget:
        """Three sites with varying five-column QR factors per active branch."""

        rng = np.random.default_rng(624)
        library = np.array([
            [-1.0, -0.7], [0.9, -0.4], [0.1, 1.1], [-0.8, 0.8]
        ])
        standardization = ThetaStandardization.from_library(library)
        branch_sizes = (5,) if loading_only else (5, 5)
        k = sum(branch_sizes)
        gp = LibraryGP.from_data(
            standardization.to_standardized(library),
            rng.normal(scale=0.3, size=(4, k)), [0.8, 1.2],
        )
        spatial_prior = SpatialPrior.from_standardization(
            standardization, physical_center=[0.1, -0.2],
            V_theta_0=2 * np.eye(2), nu_theta_0=4, S_theta_0=np.eye(2),
        )
        coordinates = SiteCoordinates.from_physical_bounds(
            standardization,
            **({"l": [-1.8, -1.2], "u": [1.7, 1.8]} if bounded else {}),
        )
        R_sites = []
        for i in range(3):
            R_branches = []
            for size in branch_sizes:
                upper = np.triu(rng.normal(scale=0.2, size=(size, size)), 1)
                R_branches.append(upper + np.diag(
                    np.linspace(0.7, 1.3, size) + 0.1 * i
                ))
            R_sites.append(block_diag(*R_branches))
        m_delta_0 = rng.normal(scale=0.15, size=k)
        A = rng.normal(scale=0.1, size=(k, k))
        V_delta_0 = A @ A.T + 0.2 * np.eye(k)
        return CalibrationTarget.from_data(
            gp, coordinates, spatial_prior,
            rng.normal(scale=0.5, size=3 * k), block_diag(*R_sites),
            [[0.0, 0.0], [6.0, 0.0], [0.0, 6.0]], branch_sizes,
            m_delta_0=None if default_prior else m_delta_0,
            V_delta_0=None if default_prior else V_delta_0,
        )

    def _arrays(self, target: CalibrationTarget) -> tuple:
        """Assemble current coefficients/noise without target.observation_arrays."""

        k = target.gp.F_s.shape[1]
        n = target.C_theta.shape[0]
        c_f = 0.3 * np.sin(np.arange(n * k) + 0.4)
        sigma_y2 = np.array([0.07] if len(target.branch_sizes) == 1 else [0.07, 0.29])
        Omega_y = np.diag(np.tile(np.repeat(sigma_y2, target.branch_sizes), n))
        return (
            target.y_tilde, c_f, target.R, Omega_y,
            target.m_delta_0, target.V_delta_0,
        )

    def _observation_space_reference(self, arrays: tuple) -> tuple:
        """Independent Gaussian conditioning in observation space, not precision."""

        y_tilde, c_f, R, Omega_y, m_delta_0, V_delta_0 = map(np.asarray, arrays)
        k = m_delta_0.size
        n = y_tilde.size // k
        H_delta = R @ np.kron(np.ones((n, 1)), np.eye(k))
        V_y = H_delta @ V_delta_0 @ H_delta.T + Omega_y
        cross = V_delta_0 @ H_delta.T
        m_delta = m_delta_0 + cross @ np.linalg.solve(
            V_y, y_tilde - R @ c_f - H_delta @ m_delta_0
        )
        V_delta = V_delta_0 - cross @ np.linalg.solve(V_y, cross.T)
        return m_delta, 0.5 * (V_delta + V_delta.T)

    def test_moments_match_dense_reference_with_configured_and_default_priors(self):
        for loading_only in (False, True):
            for default_prior in (False, True):
                with self.subTest(loading_only=loading_only, default=default_prior):
                    target = self._target(loading_only, default_prior=default_prior)
                    arrays = self._arrays(target)
                    expected_m, expected_V = self._observation_space_reference(arrays)
                    actual_m, actual_V = discrepancy_conditional_moments(*arrays)
                    np.testing.assert_allclose(
                        actual_m, expected_m, rtol=1e-11, atol=1e-14
                    )
                    np.testing.assert_allclose(
                        actual_V, expected_V, rtol=1e-11, atol=1e-14
                    )
                    self.assertEqual(actual_m.shape, (target.gp.F_s.shape[1],))
                    self.assertEqual(actual_V.dtype, jnp.float64)

    def test_single_site_diagonal_closed_form_and_jit(self):
        y_tilde = jnp.array([0.2, -0.4])
        c_f = jnp.array([0.1, 0.3])
        r = jnp.array([0.8, 1.3])
        noise = jnp.array([0.06, 0.2])
        m_delta_0 = jnp.array([0.2, -0.1])
        prior_variances = jnp.array([0.3, 0.7])
        expected_variances = 1 / (1 / prior_variances + r**2 / noise)
        expected_m = expected_variances * (
            m_delta_0 / prior_variances + r * (y_tilde - r * c_f) / noise
        )
        actual_m, actual_V = jax.jit(discrepancy_conditional_moments)(
            y_tilde, c_f, jnp.diag(r), jnp.diag(noise),
            m_delta_0, jnp.diag(prior_variances),
        )
        np.testing.assert_allclose(actual_m, expected_m, rtol=1e-13)
        np.testing.assert_allclose(actual_V, np.diag(expected_variances), rtol=1e-13)

    def test_conditional_ratios_match_full_joint_in_both_coordinate_modes(self):
        eta = jnp.array([[-0.3, 0.2], [0.5, -0.1], [0.0, 0.7]])
        mu_theta = jnp.array([0.2, -0.1])
        Sigma_theta = jnp.array([[0.8, 0.15], [0.15, 0.6]])
        for loading_only in (False, True):
            for bounded in (False, True):
                with self.subTest(loading_only=loading_only, bounded=bounded):
                    target = self._target(loading_only, bounded)
                    arrays = self._arrays(target)
                    expected_m, expected_V = self._observation_space_reference(arrays)
                    k = expected_m.size
                    sigma_y2 = jnp.array([0.07] if loading_only else [0.07, 0.29])
                    sigma_c2 = jnp.linspace(0.4, 1.2, k)
                    delta_a = expected_m + np.linspace(-0.03, 0.02, k)
                    delta_b = expected_m + np.linspace(0.02, -0.05, k)
                    log_joint_a = target.full_joint_uncollapsed(
                        eta, arrays[1], delta_a, sigma_y2,
                        mu_theta, Sigma_theta, sigma_c2,
                    )
                    log_joint_b = target.full_joint_uncollapsed(
                        eta, arrays[1], delta_b, sigma_y2,
                        mu_theta, Sigma_theta, sigma_c2,
                    )
                    expected_ratio = (
                        multivariate_normal.logpdf(delta_b, expected_m, expected_V)
                        - multivariate_normal.logpdf(delta_a, expected_m, expected_V)
                    )
                    np.testing.assert_allclose(
                        log_joint_b - log_joint_a, expected_ratio,
                        rtol=1e-10, atol=1e-10,
                    )

    def test_empirical_moments_in_two_branch_and_loading_only_modes(self):
        N = 40_000
        for loading_only in (False, True):
            with self.subTest(loading_only=loading_only):
                arrays = self._arrays(self._target(loading_only))
                expected_m, expected_V = self._observation_space_reference(arrays)
                keys = jax.random.split(jax.random.key(935 + int(loading_only)), N)
                draws = np.asarray(jax.jit(jax.vmap(
                    lambda key: sample_discrepancy(key, *arrays)
                ))(keys))
                self.assertEqual(draws.shape, (N, expected_m.size))
                self.assertEqual(draws.dtype, np.dtype("float64"))
                mean_se = np.sqrt(np.diag(expected_V) / N)
                cov_se = np.sqrt((
                    expected_V**2
                    + np.outer(np.diag(expected_V), np.diag(expected_V))
                ) / (N - 1))
                np.testing.assert_array_less(
                    np.abs(draws.mean(axis=0) - expected_m) / mean_se, 6.0
                )
                np.testing.assert_array_less(
                    np.abs(np.cov(draws, rowvar=False) - expected_V) / cov_se, 6.0
                )

    def test_key_reproducibility_and_current_state_dependencies(self):
        key = jax.random.key(36)
        for loading_only in (False, True):
            target = self._target(loading_only)
            arrays = self._arrays(target)
            sigma_y2 = jnp.array([0.07] if loading_only else [0.07, 0.29])
            first = update_discrepancy(key, target, arrays[1], sigma_y2)
            np.testing.assert_array_equal(
                first, update_discrepancy(key, target, arrays[1], sigma_y2)
            )
            np.testing.assert_allclose(
                first, jax.jit(sample_discrepancy)(key, *arrays),
                rtol=1e-13, atol=1e-13,
            )
            bounded = self._target(loading_only, bounded=True)
            np.testing.assert_array_equal(
                first, update_discrepancy(key, bounded, arrays[1], sigma_y2)
            )
            key_a, key_b = jax.random.split(key)
            self.assertFalse(np.array_equal(
                update_discrepancy(key_a, target, arrays[1], sigma_y2),
                update_discrepancy(key_b, target, arrays[1], sigma_y2),
            ))
            changed_c_f = list(arrays)
            changed_c_f[1] = arrays[1] + 0.2
            m_old, V_old = discrepancy_conditional_moments(*arrays)
            m_new, V_new = discrepancy_conditional_moments(*changed_c_f)
            self.assertFalse(np.allclose(m_old, m_new))
            np.testing.assert_array_equal(V_old, V_new)
            changed_noise = list(arrays)
            changed_noise[3] = arrays[3] * 1.5
            _, V_noise = discrepancy_conditional_moments(*changed_noise)
            self.assertFalse(np.allclose(V_old, V_noise))
            for c_f, noise in (
                (changed_c_f[1], sigma_y2), (arrays[1], sigma_y2 * 1.5)
            ):
                fresh = update_discrepancy(key, target, c_f, noise)
                self.assertFalse(np.allclose(first, fresh))

    def test_checked_update_rejects_invalid_state_and_nonfinite_draw(self):
        target = self._target(False)
        c_f = jnp.asarray(self._arrays(target)[1])
        sigma_y2 = jnp.array([0.07, 0.29])
        key = jax.random.key(76)
        invalid = (
            (c_f[:-1], sigma_y2),
            (c_f.reshape(3, -1), sigma_y2),
            (c_f.at[0].set(jnp.nan), sigma_y2),
            (c_f.at[0].set(jnp.inf), sigma_y2),
            (c_f, sigma_y2[:1]),
            (c_f, sigma_y2.reshape(2, 1)),
            (c_f, sigma_y2.at[0].set(0)),
            (c_f, sigma_y2.at[1].set(-0.1)),
            (c_f, sigma_y2.at[0].set(jnp.nan)),
            (c_f, sigma_y2.at[1].set(jnp.inf)),
        )
        for coefficients, noise in invalid:
            with self.subTest(coefficients=coefficients, noise=noise):
                with self.assertRaises(ValueError):
                    update_discrepancy(key, target, coefficients, noise)
        # Finite inputs can still overflow; report failure rather than repair.
        overflowing = replace(target, R=target.R * 1e200)
        with self.assertRaisesRegex(FloatingPointError, "nonfinite"):
            update_discrepancy(key, overflowing, c_f, sigma_y2)


if __name__ == "__main__":
    unittest.main()
