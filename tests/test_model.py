import unittest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from scipy.stats import multivariate_normal

from bayesiancalibration.linalg import (
    projected_collapsed_logpdf,
    projected_conditional_coefficient_logpdf,
    projected_conditional_coefficient_moments,
    projected_joint_logpdf,
    projected_marginal_moments,
)


def _tiny_projected_fixture() -> dict[str, jax.Array]:
    """Return the deterministic Stage 1 fixture with n=3 and k=2."""

    m_f_given_s = np.array([-0.30, 0.20, 0.45, -0.10, 0.15, 0.35])
    covariance_factor = np.array(
        [
            [0.80, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.12, 0.70, 0.00, 0.00, 0.00, 0.00],
            [0.18, -0.08, 0.65, 0.00, 0.00, 0.00],
            [0.00, 0.16, 0.10, 0.75, 0.00, 0.00],
            [0.11, 0.00, 0.14, -0.05, 0.68, 0.00],
            [-0.07, 0.13, 0.00, 0.12, 0.09, 0.72],
        ]
    )
    Sigma_f_given_s = (
        covariance_factor @ covariance_factor.T + 0.20 * np.eye(6)
    )
    R = np.diag([1.20, 0.80, 0.90, 1.40, 1.10, 0.70])
    d_delta = np.tile(np.array([0.03, -0.02]), 3)
    noise_variances = np.tile(np.array([0.08, 0.17]), 3)
    Omega_y = np.diag(noise_variances)
    c_f = np.array([-0.12, 0.28, 0.31, -0.22, 0.19, 0.42])
    projected_residual = np.array([0.05, -0.04, 0.02, 0.07, -0.03, 0.01])
    y_tilde = (
        R @ (c_f + d_delta) + projected_residual
    )

    return {
        "y_tilde": jnp.asarray(y_tilde, dtype=jnp.float64),
        "c_f": jnp.asarray(c_f, dtype=jnp.float64),
        "m_f_given_s": jnp.asarray(m_f_given_s, dtype=jnp.float64),
        "Sigma_f_given_s": jnp.asarray(
            Sigma_f_given_s,
            dtype=jnp.float64,
        ),
        "R": jnp.asarray(
            R,
            dtype=jnp.float64,
        ),
        "d_delta": jnp.asarray(d_delta, dtype=jnp.float64),
        "Omega_y": jnp.asarray(
            Omega_y,
            dtype=jnp.float64,
        ),
    }


def _as_numpy(fixture: dict[str, jax.Array]) -> dict[str, np.ndarray]:
    return {name: np.asarray(value) for name, value in fixture.items()}


class ProjectedGaussianFixtureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = _tiny_projected_fixture()

    def test_fixture_uses_float64_and_unequal_branch_noise(self) -> None:
        for value in self.fixture.values():
            self.assertEqual(value.dtype, jnp.float64)

        noise_diagonal = np.diag(np.asarray(self.fixture["Omega_y"]))
        np.testing.assert_allclose(noise_diagonal[0::2], 0.08)
        np.testing.assert_allclose(noise_diagonal[1::2], 0.17)

    def test_marginal_and_conditional_moments_match_dense_reference(self) -> None:
        values = _as_numpy(self.fixture)
        m_y, V_y = projected_marginal_moments(
            self.fixture["m_f_given_s"],
            self.fixture["Sigma_f_given_s"],
            self.fixture["R"],
            self.fixture["d_delta"],
            self.fixture["Omega_y"],
        )
        m_f, V_f = (
            projected_conditional_coefficient_moments(
                self.fixture["y_tilde"],
                self.fixture["m_f_given_s"],
                self.fixture["Sigma_f_given_s"],
                self.fixture["R"],
                self.fixture["d_delta"],
                self.fixture["Omega_y"],
            )
        )

        reference_m_y = values["R"] @ (
            values["m_f_given_s"] + values["d_delta"]
        )
        reference_V_y = (
            values["R"]
            @ values["Sigma_f_given_s"]
            @ values["R"].T
            + values["Omega_y"]
        )
        Sigma_fy = (
            values["Sigma_f_given_s"]
            @ values["R"].T
        )
        reference_gain = np.linalg.solve(
            reference_V_y,
            Sigma_fy.T,
        ).T
        reference_m_f = values["m_f_given_s"] + (
            reference_gain
            @ (values["y_tilde"] - reference_m_y)
        )
        reference_V_f = (
            values["Sigma_f_given_s"]
            - reference_gain @ Sigma_fy.T
        )

        np.testing.assert_allclose(m_y, reference_m_y)
        np.testing.assert_allclose(
            V_y,
            reference_V_y,
        )
        np.testing.assert_allclose(
            m_f,
            reference_m_f,
        )
        np.testing.assert_allclose(
            V_f,
            reference_V_f,
        )

    def test_joint_marginal_conditional_identity_matches_scipy(self) -> None:
        values = _as_numpy(self.fixture)
        joint_logpdf = projected_joint_logpdf(**self.fixture)
        collapsed_logpdf = projected_collapsed_logpdf(
            self.fixture["y_tilde"],
            self.fixture["m_f_given_s"],
            self.fixture["Sigma_f_given_s"],
            self.fixture["R"],
            self.fixture["d_delta"],
            self.fixture["Omega_y"],
        )
        conditional_logpdf = projected_conditional_coefficient_logpdf(
            **self.fixture
        )

        m_y = values["R"] @ (
            values["m_f_given_s"] + values["d_delta"]
        )
        V_y = (
            values["R"]
            @ values["Sigma_f_given_s"]
            @ values["R"].T
            + values["Omega_y"]
        )
        Sigma_fy = (
            values["Sigma_f_given_s"]
            @ values["R"].T
        )
        full_mean = np.concatenate([values["m_f_given_s"], m_y])
        full_covariance = np.block(
            [
                [values["Sigma_f_given_s"], Sigma_fy],
                [Sigma_fy.T, V_y],
            ]
        )
        full_value = np.concatenate(
            [values["c_f"], values["y_tilde"]]
        )
        reference_joint = multivariate_normal.logpdf(
            full_value,
            mean=full_mean,
            cov=full_covariance,
        )
        reference_collapsed = multivariate_normal.logpdf(
            values["y_tilde"],
            mean=m_y,
            cov=V_y,
        )

        np.testing.assert_allclose(joint_logpdf, reference_joint)
        np.testing.assert_allclose(collapsed_logpdf, reference_collapsed)
        np.testing.assert_allclose(
            joint_logpdf,
            collapsed_logpdf + conditional_logpdf,
        )

    def test_loading_only_subcase_preserves_identity(self) -> None:
        loading_indices = np.array([0, 2, 4])
        loading_fixture = {
            "y_tilde": self.fixture["y_tilde"][loading_indices],
            "c_f": self.fixture["c_f"][loading_indices],
            "m_f_given_s": self.fixture["m_f_given_s"][loading_indices],
            "Sigma_f_given_s": self.fixture["Sigma_f_given_s"][
                np.ix_(loading_indices, loading_indices)
            ],
            "R": self.fixture["R"][
                np.ix_(loading_indices, loading_indices)
            ],
            "d_delta": self.fixture["d_delta"][loading_indices],
            "Omega_y": self.fixture["Omega_y"][
                np.ix_(loading_indices, loading_indices)
            ],
        }

        joint_logpdf = projected_joint_logpdf(**loading_fixture)
        collapsed_logpdf = projected_collapsed_logpdf(
            loading_fixture["y_tilde"],
            loading_fixture["m_f_given_s"],
            loading_fixture["Sigma_f_given_s"],
            loading_fixture["R"],
            loading_fixture["d_delta"],
            loading_fixture["Omega_y"],
        )
        conditional_logpdf = projected_conditional_coefficient_logpdf(
            **loading_fixture
        )

        np.testing.assert_allclose(
            joint_logpdf,
            collapsed_logpdf + conditional_logpdf,
        )
        np.testing.assert_allclose(
            np.diag(np.asarray(loading_fixture["Omega_y"])),
            0.08,
        )

    def test_core_calculations_are_jittable(self) -> None:
        collapsed = jax.jit(projected_collapsed_logpdf)(
            self.fixture["y_tilde"],
            self.fixture["m_f_given_s"],
            self.fixture["Sigma_f_given_s"],
            self.fixture["R"],
            self.fixture["d_delta"],
            self.fixture["Omega_y"],
        )
        m_f, V_f = jax.jit(
            projected_conditional_coefficient_moments
        )(
            self.fixture["y_tilde"],
            self.fixture["m_f_given_s"],
            self.fixture["Sigma_f_given_s"],
            self.fixture["R"],
            self.fixture["d_delta"],
            self.fixture["Omega_y"],
        )

        self.assertTrue(bool(jnp.isfinite(collapsed)))
        self.assertTrue(bool(jnp.all(jnp.isfinite(m_f))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(V_f))))


if __name__ == "__main__":
    unittest.main()
