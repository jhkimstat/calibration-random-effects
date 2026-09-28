"""Stage 5 precision references and empirical exact-refresh checks."""

import unittest
from dataclasses import replace

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from scipy.linalg import block_diag

from bayesiancalibration.gibbs import (
    refresh_field_coefficients,
    sample_projected_coefficients,
)
from bayesiancalibration.gp import LibraryGP
from bayesiancalibration.linalg import projected_conditional_coefficient_moments
from bayesiancalibration.state import SpatialPrior, ThetaStandardization
from bayesiancalibration.targets import CalibrationTarget
from bayesiancalibration.transforms import SiteCoordinates


class CoefficientRefreshTest(unittest.TestCase):
    def setUp(self) -> None:
        # Three sites, two two-coefficient branches; each R_i is different.
        rng = np.random.default_rng(412)
        A = rng.normal(size=(12, 12)) / 3.0
        self.Sigma = A @ A.T + 0.5 * np.eye(12)
        self.m = rng.normal(size=12)
        self.y = rng.normal(size=12)
        self.R = block_diag(*[
            block_diag(
                np.array([[1.0 + 0.2 * i, 0.15 * (i + 1)], [0.0, 0.7]]),
                np.array([[0.9, -0.1 * (i + 1)], [0.0, 1.3 + 0.1 * i]]),
            )
            for i in range(3)
        ])
        self.d_delta = np.tile([0.03, -0.02, 0.05, 0.01], 3)
        self.Omega = np.diag(np.tile([0.08, 0.08, 0.31, 0.31], 3))

    def _precision_reference(self, arrays: tuple) -> tuple[np.ndarray, np.ndarray]:
        """Independently use the source-note precision form, not V_y."""

        y, m, Sigma, R, d_delta, Omega = map(np.asarray, arrays)
        P_0 = np.linalg.solve(Sigma, np.eye(m.size))
        P_y = np.linalg.solve(Omega, np.eye(y.size))
        P_f = P_0 + R.T @ P_y @ R
        V_f = np.linalg.solve(P_f, np.eye(m.size))
        m_f = np.linalg.solve(P_f, P_0 @ m + R.T @ P_y @ (y - R @ d_delta))
        return m_f, V_f

    def _check_empirical_moments(self, arrays: tuple, seed: int) -> None:
        """Six Monte Carlo standard errors give distribution-based tolerances."""

        m_f, V_f = self._precision_reference(arrays)
        N = 40_000
        keys = jax.random.split(jax.random.key(seed), N)
        draws = np.asarray(jax.jit(jax.vmap(
            lambda key: sample_projected_coefficients(key, *arrays)
        ))(keys))
        self.assertEqual(draws.dtype, np.dtype("float64"))
        self.assertEqual(draws.shape, (N, m_f.size))
        mean_se = np.sqrt(np.diag(V_f) / N)
        cov_se = np.sqrt(
            (V_f**2 + np.outer(np.diag(V_f), np.diag(V_f))) / (N - 1)
        )
        np.testing.assert_array_less(
            np.abs(draws.mean(axis=0) - m_f) / mean_se, 6.0
        )
        np.testing.assert_array_less(
            np.abs(np.cov(draws, rowvar=False) - V_f) / cov_se, 6.0
        )

    def test_analytic_moments_match_source_precision_form(self) -> None:
        arrays = (self.y, self.m, self.Sigma, self.R, self.d_delta, self.Omega)
        expected_m, expected_V = self._precision_reference(arrays)
        actual_m, actual_V = projected_conditional_coefficient_moments(*arrays)
        np.testing.assert_allclose(actual_m, expected_m, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual_V, expected_V, rtol=1e-12, atol=1e-12)

    def test_empirical_moments_with_varying_design_and_branch_noise(self) -> None:
        self._check_empirical_moments(
            (self.y, self.m, self.Sigma, self.R, self.d_delta, self.Omega), 71
        )

    def test_loading_only_empirical_moments(self) -> None:
        indices = np.array([0, 1, 4, 5, 8, 9])
        blocks = np.ix_(indices, indices)
        self._check_empirical_moments((
            self.y[indices], self.m[indices], self.Sigma[blocks],
            self.R[blocks], self.d_delta[indices], self.Omega[blocks],
        ), 72)

    def test_explicit_key_reproducibility_and_jit(self) -> None:
        arrays = (self.y, self.m, self.Sigma, self.R, self.d_delta, self.Omega)
        key = jax.random.key(19)
        first = sample_projected_coefficients(key, *arrays)
        np.testing.assert_array_equal(
            first, sample_projected_coefficients(key, *arrays)
        )
        np.testing.assert_allclose(
            first, jax.jit(sample_projected_coefficients)(key, *arrays),
            rtol=1e-13, atol=1e-13,
        )
        key_a, key_b = jax.random.split(key)
        self.assertFalse(np.array_equal(
            sample_projected_coefficients(key_a, *arrays),
            sample_projected_coefficients(key_b, *arrays),
        ))

    def _target(self, bounded: bool, loading_only: bool = False) -> CalibrationTarget:
        """Small real GP target for checking model-to-draw integration."""

        library = np.array([[-1.0, -0.7], [0.9, -0.4], [0.1, 1.1]])
        standardization = ThetaStandardization.from_library(library)
        F_s = np.array([[0.2, -0.4], [0.7, 0.1], [-0.3, 0.5]])
        gp = LibraryGP.from_data(
            standardization.to_standardized(library), F_s, [0.8, 1.2]
        )
        prior = SpatialPrior.from_standardization(
            standardization, physical_center=[0.1, -0.2],
            V_theta_0=2 * np.eye(2), nu_theta_0=4, S_theta_0=np.eye(2),
        )
        coordinates = SiteCoordinates.from_physical_bounds(
            standardization,
            **({"l": [-1.8, -1.2], "u": [1.7, 1.8]} if bounded else {}),
        )
        R = block_diag([[1.1, 0.2], [0.0, 0.9]], [[1.3, -0.1], [0.0, 1.05]])
        if not loading_only:
            R = np.diag(np.diag(R))  # One coefficient per branch in this fixture.
        return CalibrationTarget.from_data(
            gp, coordinates, prior, [0.12, -0.1, 0.05, 0.18], R,
            [[0.0, 0.0], [6.0, 0.0]], (2,) if loading_only else (1, 1),
        )

    def test_checked_refresh_uses_current_conditioning_in_both_modes(self) -> None:
        key = jax.random.key(83)
        eta = jnp.array([[-0.3, 0.2], [0.5, -0.1]])
        delta = jnp.array([0.03, -0.02])
        sigma_c2 = jnp.array([0.4, 1.2])
        for bounded in (False, True):
            for loading_only in (False, True):
                with self.subTest(bounded=bounded, loading_only=loading_only):
                    target = self._target(bounded, loading_only)
                    sigma_y2 = jnp.array([0.09] if loading_only else [0.09, 0.15])
                    c_f = refresh_field_coefficients(
                        key, target, eta, delta, sigma_y2, sigma_c2
                    )
                    self.assertEqual(c_f.shape, (4,))
                    self.assertTrue(np.all(np.isfinite(c_f)))
                    # Assemble GP moments independently, without the target helpers.
                    theta = np.asarray(target.coordinates.eta_to_theta_tilde(eta))
                    library = np.asarray(target.gp.theta_s_tilde)
                    scales = np.asarray(target.gp.lambda_c)
                    C_fs = np.exp(-0.5 * np.sum(
                        ((theta[:, None] - library[None, :]) / scales)**2, axis=-1
                    ))
                    C_ff = np.exp(-0.5 * np.sum(
                        ((theta[:, None] - theta[None, :]) / scales)**2, axis=-1
                    ))
                    C_ss = np.asarray(target.gp.C_ss)
                    m = (C_fs @ np.linalg.solve(C_ss, target.gp.F_s)).reshape(-1)
                    C = C_ff - C_fs @ np.linalg.solve(C_ss, C_fs.T)
                    Omega = np.diag(np.tile(
                        np.repeat(sigma_y2, target.branch_sizes), 2
                    ))
                    expected = sample_projected_coefficients(
                        key, target.y_tilde, m, np.kron(C, np.diag(sigma_c2)),
                        target.R, np.tile(delta, 2), Omega,
                    )
                    np.testing.assert_allclose(c_f, expected, rtol=1e-12, atol=1e-12)
                    for changed in (
                        (eta.at[0, 0].add(0.1), delta, sigma_y2, sigma_c2),
                        (eta, delta + 0.2, sigma_y2, sigma_c2),
                        (eta, delta, sigma_y2 * 1.5, sigma_c2),
                        (eta, delta, sigma_y2, sigma_c2 * 1.5),
                    ):
                        fresh = refresh_field_coefficients(key, target, *changed)
                        self.assertFalse(np.allclose(c_f, fresh))

    def test_checked_refresh_rejects_invalid_state_and_singular_gp(self) -> None:
        target = self._target(False)
        key = jax.random.key(3)
        eta = jnp.array([[-0.3, 0.2], [0.5, -0.1]])
        delta = jnp.array([0.03, -0.02])
        sigma_y2 = jnp.array([0.09, 0.15])
        sigma_c2 = jnp.array([0.4, 1.2])
        invalid = (
            (eta[:1], delta, sigma_y2, sigma_c2),
            (eta.at[0, 0].set(jnp.nan), delta, sigma_y2, sigma_c2),
            (eta, delta[:1], sigma_y2, sigma_c2),
            (eta, delta.at[0].set(jnp.inf), sigma_y2, sigma_c2),
            (eta, delta, sigma_y2[:1], sigma_c2),
            (eta, delta, sigma_y2.at[0].set(0), sigma_c2),
            (eta, delta, sigma_y2.at[0].set(jnp.nan), sigma_c2),
            (eta, delta, sigma_y2, sigma_c2[:1]),
            (eta, delta, sigma_y2, sigma_c2.at[0].set(-0.1)),
            (eta, delta, sigma_y2, sigma_c2.at[0].set(jnp.inf)),
            (eta.at[1].set(eta[0]), delta, sigma_y2, sigma_c2),
            (eta.at[0].set(target.gp.theta_s_tilde[0]), delta, sigma_y2, sigma_c2),
        )
        for state in invalid:
            with self.subTest(state=state):
                with self.assertRaises(ValueError):
                    refresh_field_coefficients(key, target, *state)
        # A numerical jitter cannot hide structural coincidence at the boundary.
        jittered = replace(target, gp=LibraryGP.from_data(
            target.gp.theta_s_tilde, target.gp.F_s, target.gp.lambda_c, jitter=1e-8
        ))
        with self.assertRaisesRegex(ValueError, "coincides"):
            refresh_field_coefficients(
                key, jittered, eta.at[0].set(target.gp.theta_s_tilde[0]),
                delta, sigma_y2, sigma_c2,
            )


if __name__ == "__main__":
    unittest.main()
