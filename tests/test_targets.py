"""Independent dense posterior references for Stage 4 targets."""

import unittest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from scipy.special import expit
from scipy.stats import invgamma, invwishart, multivariate_normal

from bayesiancalibration.gp import LibraryGP
from bayesiancalibration.linalg import projected_conditional_coefficient_logpdf
from bayesiancalibration.state import SpatialPrior, ThetaStandardization
from bayesiancalibration.targets import CalibrationTarget
from bayesiancalibration.transforms import SiteCoordinates


class CalibrationTargetTest(unittest.TestCase):
    def setUp(self) -> None:
        library_physical = np.array([[-1.0, -0.7], [0.9, -0.4], [0.1, 1.1]])
        self.standardization = ThetaStandardization.from_library(library_physical)
        theta_s_tilde = self.standardization.to_standardized(library_physical)
        self.F_s = np.array([[0.2, -0.4], [0.7, 0.1], [-0.3, 0.5]])
        self.gp = LibraryGP.from_data(theta_s_tilde, self.F_s, [0.8, 1.2])
        self.prior = SpatialPrior.from_standardization(
            self.standardization,
            physical_center=[0.1, -0.2],
            V_theta_0=2.0 * np.eye(2),
            nu_theta_0=4.0,
            S_theta_0=np.array([[1.0, 0.1], [0.1, 0.8]]),
        )
        self.y_tilde = np.array([0.12, -0.10, 0.05, 0.18])
        self.R = np.diag([1.1, 0.9, 1.3, 1.05])
        self.s = np.array([[0.0, 0.0], [6.0, 0.0]])
        self.eta_identity = jnp.array([[-0.3, 0.2], [0.5, -0.1]])
        self.eta_box = jnp.array([[0.2, -0.4], [-0.5, 0.6]])
        self.c_f = jnp.array([0.2, -0.3, 0.35, 0.1])
        self.delta = jnp.array([0.03, -0.02])
        self.sigma_y2 = jnp.array([0.09, 0.15])
        self.mu_theta = jnp.array([0.2, -0.1])
        self.Sigma_theta = jnp.array([[0.8, 0.15], [0.15, 0.6]])
        self.sigma_c2 = jnp.array([0.4, 1.2])

    def _target(self, bounded: bool) -> CalibrationTarget:
        """Prepare the two specified coordinate modes from the same data."""

        if bounded:
            coordinates = SiteCoordinates.from_physical_bounds(
                self.standardization, l=[-1.8, -1.2], u=[1.7, 1.8]
            )
        else:
            coordinates = SiteCoordinates.from_physical_bounds(self.standardization)
        return CalibrationTarget.from_data(
            self.gp, coordinates, self.prior,
            self.y_tilde, self.R, self.s, (1, 1),
        )

    def _reference_terms(
        self, target: CalibrationTarget, eta: jax.Array, sigma_c2: jax.Array
    ) -> tuple[float, float, float]:
        """Build the joint directly with SciPy's dense distributions."""

        theta = np.asarray(target.coordinates.eta_to_theta_tilde(eta))
        m, _, Sigma = target.gp.conditional_moments(jnp.asarray(theta), sigma_c2)
        m = np.asarray(m)
        Sigma = np.asarray(Sigma)
        sigma_c2_np = np.asarray(sigma_c2)
        Omega_y = np.diag(np.tile(np.asarray(self.sigma_y2), 2))
        d_delta = np.tile(np.asarray(self.delta), 2)
        spatial_cov = np.kron(np.asarray(target.C_theta), np.asarray(self.Sigma_theta))

        spatial = multivariate_normal.logpdf(
            theta.reshape(-1),
            mean=np.tile(np.asarray(self.mu_theta), 2), cov=spatial_cov,
        )
        log_jacobian = 0.0
        if target.coordinates.bounded:
            width = np.asarray(target.coordinates.u_tilde - target.coordinates.l_tilde)
            p = expit(np.asarray(eta))
            log_jacobian = float(np.sum(np.log(width) + np.log(p) + np.log1p(-p)))
        other = (
            multivariate_normal.logpdf(
                self.F_s.reshape(-1),
                cov=np.kron(np.asarray(target.gp.C_ss), np.diag(sigma_c2_np)),
            )
            + multivariate_normal.logpdf(
                np.asarray(self.delta),
                mean=np.asarray(target.m_delta_0), cov=np.asarray(target.V_delta_0),
            )
            + np.sum(invgamma.logpdf(
                np.asarray(self.sigma_y2),
                a=np.asarray(target.alpha_y_0), scale=np.asarray(target.beta_y_0),
            ))
            + multivariate_normal.logpdf(
                np.asarray(self.mu_theta), mean=np.asarray(self.prior.m_theta_0),
                cov=np.asarray(self.prior.V_theta_0),
            )
            + invwishart.logpdf(
                np.asarray(self.Sigma_theta), df=self.prior.nu_theta_0,
                scale=np.asarray(self.prior.S_theta_0),
            )
            + np.sum(invgamma.logpdf(
                sigma_c2_np,
                a=np.repeat(target.alpha_c_0, target.branch_sizes),
                scale=np.repeat(target.beta_c_0, target.branch_sizes),
            ))
        )
        observation = multivariate_normal.logpdf(
            np.asarray(self.y_tilde),
            mean=self.R @ (np.asarray(self.c_f) + d_delta), cov=Omega_y,
        )
        coefficient = multivariate_normal.logpdf(
            np.asarray(self.c_f), mean=m, cov=Sigma,
        )
        marginal = multivariate_normal.logpdf(
            np.asarray(self.y_tilde),
            mean=self.R @ (m + d_delta),
            cov=self.R @ Sigma @ self.R.T + Omega_y,
        )
        return (
            float(observation + coefficient + spatial + other + log_jacobian),
            float(marginal + spatial + other + log_jacobian),
            float(other),
        )

    def test_full_densities_match_independent_scipy_in_both_modes(self) -> None:
        for bounded in (False, True):
            with self.subTest(bounded=bounded):
                target = self._target(bounded)
                eta = self.eta_box if bounded else self.eta_identity
                expected_uncoll, expected_coll, _ = self._reference_terms(
                    target, eta, self.sigma_c2
                )
                np.testing.assert_allclose(
                    target.full_joint_uncollapsed(
                        eta, self.c_f, self.delta, self.sigma_y2,
                        self.mu_theta, self.Sigma_theta, self.sigma_c2,
                    ),
                    expected_uncoll,
                )
                np.testing.assert_allclose(
                    target.full_joint_collapsed(
                        eta, self.delta, self.sigma_y2,
                        self.mu_theta, self.Sigma_theta, self.sigma_c2,
                    ),
                    expected_coll,
                )

    def test_joint_factorization_and_theta_only_ratios(self) -> None:
        for bounded in (False, True):
            with self.subTest(bounded=bounded):
                target = self._target(bounded)
                eta = self.eta_box if bounded else self.eta_identity
                proposal = eta.at[0, 0].add(0.15)
                theta = target.coordinates.eta_to_theta_tilde(eta)
                m, _, Sigma = self.gp.conditional_moments(theta, self.sigma_c2)
                Omega_y = np.diag(np.tile(np.asarray(self.sigma_y2), 2))
                d_delta = jnp.tile(self.delta, 2)
                conditional = projected_conditional_coefficient_logpdf(
                    self.c_f, self.y_tilde, m, Sigma, self.R,
                    d_delta, Omega_y,
                )
                full_uncoll = target.full_joint_uncollapsed(
                    eta, self.c_f, self.delta, self.sigma_y2,
                    self.mu_theta, self.Sigma_theta, self.sigma_c2,
                )
                full_coll = target.full_joint_collapsed(
                    eta, self.delta, self.sigma_y2,
                    self.mu_theta, self.Sigma_theta, self.sigma_c2,
                )
                np.testing.assert_allclose(full_uncoll, full_coll + conditional)

                np.testing.assert_allclose(
                    target.full_joint_collapsed(
                        proposal, self.delta, self.sigma_y2,
                        self.mu_theta, self.Sigma_theta, self.sigma_c2,
                    ) - full_coll,
                    target.theta_only_collapsed(
                        proposal, self.delta, self.sigma_y2,
                        self.mu_theta, self.Sigma_theta, self.sigma_c2,
                    ) - target.theta_only_collapsed(
                        eta, self.delta, self.sigma_y2,
                        self.mu_theta, self.Sigma_theta, self.sigma_c2,
                    ),
                )
                np.testing.assert_allclose(
                    target.full_joint_uncollapsed(
                        proposal, self.c_f, self.delta, self.sigma_y2,
                        self.mu_theta, self.Sigma_theta, self.sigma_c2,
                    ) - full_uncoll,
                    target.theta_only_uncollapsed(
                        proposal, self.c_f, self.mu_theta,
                        self.Sigma_theta, self.sigma_c2,
                    ) - target.theta_only_uncollapsed(
                        eta, self.c_f, self.mu_theta,
                        self.Sigma_theta, self.sigma_c2,
                    ),
                )

    def test_one_site_collapsed_ratio_matches_conditional_gaussians(self) -> None:
        for bounded in (False, True):
            with self.subTest(bounded=bounded):
                target = self._target(bounded)
                eta = self.eta_box if bounded else self.eta_identity
                proposal = eta.at[0, 0].add(0.15)

                def reference_conditional_terms(candidate: jax.Array) -> float:
                    theta = np.asarray(target.coordinates.eta_to_theta_tilde(candidate))
                    m, _, Sigma = self.gp.conditional_moments(
                        jnp.asarray(theta), self.sigma_c2
                    )
                    d_delta = np.tile(np.asarray(self.delta), 2)
                    Omega_y = np.diag(np.tile(np.asarray(self.sigma_y2), 2))
                    m_y = self.R @ (np.asarray(m) + d_delta)
                    V_y = self.R @ np.asarray(Sigma) @ self.R.T + Omega_y
                    m_y_cond = m_y[:2] + V_y[:2, 2:] @ np.linalg.solve(
                        V_y[2:, 2:], self.y_tilde[2:] - m_y[2:]
                    )
                    V_y_cond = V_y[:2, :2] - V_y[:2, 2:] @ np.linalg.solve(
                        V_y[2:, 2:], V_y[2:, :2]
                    )

                    spatial_cov = np.kron(
                        np.asarray(target.C_theta), np.asarray(self.Sigma_theta)
                    )
                    mu = np.asarray(self.mu_theta)
                    m_theta_cond = mu + spatial_cov[:2, 2:] @ np.linalg.solve(
                        spatial_cov[2:, 2:], theta[1] - mu
                    )
                    V_theta_cond = spatial_cov[:2, :2] - (
                        spatial_cov[:2, 2:]
                        @ np.linalg.solve(spatial_cov[2:, 2:], spatial_cov[2:, :2])
                    )
                    return float(
                        multivariate_normal.logpdf(
                            self.y_tilde[:2], mean=m_y_cond, cov=V_y_cond
                        )
                        + multivariate_normal.logpdf(
                            theta[0], mean=m_theta_cond, cov=V_theta_cond
                        )
                        + target.coordinates.log_jacobian(candidate)
                    )

                actual_ratio = target.theta_only_collapsed(
                    proposal, self.delta, self.sigma_y2,
                    self.mu_theta, self.Sigma_theta, self.sigma_c2,
                ) - target.theta_only_collapsed(
                    eta, self.delta, self.sigma_y2,
                    self.mu_theta, self.Sigma_theta, self.sigma_c2,
                )
                reference_ratio = (
                    reference_conditional_terms(proposal)
                    - reference_conditional_terms(eta)
                )
                np.testing.assert_allclose(actual_ratio, reference_ratio)

    def test_gradients_match_finite_differences_in_both_modes(self) -> None:
        for bounded in (False, True):
            with self.subTest(bounded=bounded):
                target = self._target(bounded)
                eta = self.eta_box if bounded else self.eta_identity
                densities = (
                    lambda e: target.theta_only_collapsed(
                        e, self.delta, self.sigma_y2,
                        self.mu_theta, self.Sigma_theta, self.sigma_c2,
                    ),
                    lambda e: target.theta_only_uncollapsed(
                        e, self.c_f, self.mu_theta, self.Sigma_theta, self.sigma_c2,
                    ),
                )
                for density in densities:
                    gradient = jax.jit(jax.grad(density))(eta)
                    h = 1e-5
                    reference = np.empty(eta.shape)
                    for i in range(eta.shape[0]):
                        for q in range(eta.shape[1]):
                            plus = eta.at[i, q].add(h)
                            minus = eta.at[i, q].add(-h)
                            reference[i, q] = (
                                float(density(plus)) - float(density(minus))
                            ) / (2.0 * h)
                    np.testing.assert_allclose(
                        gradient, reference, rtol=1e-4, atol=1e-4
                    )

    def test_current_variances_unbounded_mean_and_loading_only(self) -> None:
        target = self._target(True)
        expected_1, _, _ = self._reference_terms(target, self.eta_box, self.sigma_c2)
        expected_2, _, _ = self._reference_terms(
            target, self.eta_box, 2.0 * self.sigma_c2
        )
        actual_1 = target.full_joint_uncollapsed(
            self.eta_box, self.c_f, self.delta, self.sigma_y2,
            self.mu_theta, self.Sigma_theta, self.sigma_c2,
        )
        actual_2 = target.full_joint_uncollapsed(
            self.eta_box, self.c_f, self.delta, self.sigma_y2,
            self.mu_theta, self.Sigma_theta, 2.0 * self.sigma_c2,
        )
        np.testing.assert_allclose(actual_1, expected_1)
        np.testing.assert_allclose(actual_2, expected_2)
        self.assertNotAlmostEqual(float(actual_1), float(actual_2))

        outside_mu = jnp.array([4.0, -4.0])
        self.assertTrue(bool(jnp.isfinite(target.full_joint_collapsed(
            self.eta_box, self.delta, self.sigma_y2,
            outside_mu, self.Sigma_theta, self.sigma_c2,
        ))))

        loading_target = CalibrationTarget.from_data(
            self.gp, SiteCoordinates.from_physical_bounds(self.standardization),
            self.prior, self.y_tilde, self.R, self.s, (2,),
        )
        loading_density = jax.jit(loading_target.full_joint_collapsed)(
            self.eta_identity, self.delta, jnp.array([0.09]),
            self.mu_theta, self.Sigma_theta, self.sigma_c2,
        )
        self.assertTrue(bool(jnp.isfinite(loading_density)))

    def test_fixed_specification_validation(self) -> None:
        coordinates = SiteCoordinates.from_physical_bounds(self.standardization)
        common = (self.gp, coordinates, self.prior, self.y_tilde, self.R, self.s)
        with self.assertRaisesRegex(ValueError, "branch_sizes"):
            CalibrationTarget.from_data(*common, (3,))
        with self.assertRaisesRegex(ValueError, "Repeated spatial"):
            CalibrationTarget.from_data(
                self.gp, coordinates, self.prior, self.y_tilde, self.R,
                np.zeros_like(self.s), (1, 1),
            )
        with self.assertRaisesRegex(ValueError, "alpha_y_0"):
            CalibrationTarget.from_data(
                *common, (1, 1), alpha_y_0=[1.01, -1.0]
            )

    def test_coefficient_prior_arrays_and_validation(self) -> None:
        coordinates = SiteCoordinates.from_physical_bounds(self.standardization)
        common = (self.gp, coordinates, self.prior, self.y_tilde, self.R, self.s)
        for sizes in ((1, 1), (2,)):
            B = len(sizes)
            target = CalibrationTarget.from_data(*common, sizes)
            np.testing.assert_array_equal(target.alpha_c_0, np.full(B, 1.01))
            np.testing.assert_array_equal(target.beta_c_0, np.full(B, 0.01))
            self.assertEqual(target.alpha_c_0.dtype, jnp.float64)
            self.assertEqual(target.beta_c_0.dtype, jnp.float64)
            for name in ("alpha_c_0", "beta_c_0"):
                for invalid in (
                    2.0, [], np.ones(B + 1), np.ones((B, 1)),
                    np.zeros(B), -np.ones(B), np.full(B, np.nan),
                    np.full(B, np.inf),
                ):
                    with self.subTest(sizes=sizes, name=name, invalid=invalid):
                        with self.assertRaisesRegex(ValueError, "alpha_c_0"):
                            CalibrationTarget.from_data(
                                *common, sizes, **{name: invalid}
                            )

    def test_invalid_runtime_variance_is_not_regularized(self) -> None:
        target = self._target(False)
        invalid_sigma_c2 = jnp.array([-0.4, 1.2])
        invalid_noise = jnp.array([0.09, -0.15])
        self.assertTrue(np.isnan(
            float(target.full_joint_collapsed(
                self.eta_identity, self.delta, self.sigma_y2,
                self.mu_theta, self.Sigma_theta, invalid_sigma_c2,
            )),
        ))
        self.assertTrue(np.isnan(
            float(target.full_joint_uncollapsed(
                self.eta_identity, self.c_f, self.delta, invalid_noise,
                self.mu_theta, self.Sigma_theta, self.sigma_c2,
            )),
        ))

    def test_configured_prior_and_spatial_range(self) -> None:
        target = CalibrationTarget.from_data(
            self.gp, SiteCoordinates.from_physical_bounds(self.standardization),
            self.prior, self.y_tilde, self.R, self.s, (1, 1),
            lambda_theta=6.0,
            m_delta_0=[0.1, -0.1],
            V_delta_0=np.diag([0.2, 0.4]),
            alpha_y_0=[2.0, 3.0], beta_y_0=[0.3, 0.4],
            alpha_c_0=[2.5, 3.7], beta_c_0=[0.2, 0.6],
        )
        np.testing.assert_allclose(target.C_theta[0, 1], np.exp(-0.5))
        reference_uncoll, reference_coll, _ = self._reference_terms(
            target, self.eta_identity, self.sigma_c2
        )
        np.testing.assert_allclose(
            target.full_joint_uncollapsed(
                self.eta_identity, self.c_f, self.delta, self.sigma_y2,
                self.mu_theta, self.Sigma_theta, self.sigma_c2,
            ),
            reference_uncoll,
        )
        np.testing.assert_allclose(
            target.full_joint_collapsed(
                self.eta_identity, self.delta, self.sigma_y2,
                self.mu_theta, self.Sigma_theta, self.sigma_c2,
            ),
            reference_coll,
        )


if __name__ == "__main__":
    unittest.main()
