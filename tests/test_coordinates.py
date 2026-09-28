"""Reference checks for Stage 2 frozen calibration coordinates."""

import unittest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from bayesiancalibration.state import SpatialPrior, ThetaStandardization
from bayesiancalibration.transforms import SiteCoordinates


class ThetaStandardizationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.library = np.array(
            [
                [37500.0, 23800.0, 0.060],
                [38000.0, 24100.0, 0.075],
                [38200.0, 24000.0, 0.081],
                [37900.0, 24400.0, 0.068],
            ],
            dtype=np.float64,
        )

    def test_sample_map_and_inverse_match_numpy(self) -> None:
        frozen = ThetaStandardization.from_library(self.library)
        expected_center = np.mean(self.library, axis=0)
        expected_scales = np.std(self.library, axis=0, ddof=1)
        np.testing.assert_allclose(frozen.theta_bar_dagger, expected_center)
        np.testing.assert_allclose(frozen.D_theta, np.diag(expected_scales))
        self.assertEqual(frozen.source, "library")
        self.assertEqual(frozen.D_theta.dtype, jnp.float64)

        theta_s_tilde = frozen.to_standardized(self.library)
        np.testing.assert_allclose(
            theta_s_tilde,
            (self.library - expected_center) / expected_scales,
        )
        np.testing.assert_allclose(np.mean(theta_s_tilde, axis=0), 0.0, atol=1e-13)
        np.testing.assert_allclose(np.std(theta_s_tilde, axis=0, ddof=1), 1.0)
        np.testing.assert_allclose(frozen.to_physical(theta_s_tilde), self.library)
        arbitrary_tilde = np.array([-0.6, 0.25, 1.8])
        np.testing.assert_allclose(
            frozen.to_physical(arbitrary_tilde),
            expected_center + arbitrary_tilde * expected_scales,
        )

        site = np.array([37750.0, 24250.0, 0.073])
        np.testing.assert_allclose(
            jax.jit(frozen.to_physical)(jax.jit(frozen.to_standardized)(site)),
            site,
        )

    def test_valid_override_is_frozen_and_cannot_hide_bad_library(self) -> None:
        center = np.array([37850.0, 24060.0, 0.071])
        D_theta = np.diag([1000.0, 400.0, 0.02])
        frozen = ThetaStandardization.from_library(
            self.library, theta_bar_dagger=center, D_theta=D_theta
        )
        self.assertEqual(frozen.source, "override")
        np.testing.assert_allclose(
            frozen.to_standardized(self.library),
            (self.library - center) / np.diag(D_theta),
        )
        with self.assertRaisesRegex(ValueError, "spread"):
            ThetaStandardization.from_library(
                np.tile(self.library[0], (3, 1)),
                theta_bar_dagger=center,
                D_theta=D_theta,
            )

    def test_library_and_override_validation(self) -> None:
        bad_libraries = (
            self.library[:1],
            np.column_stack([self.library[:, 0], np.ones(4)]),
            np.column_stack(
                [1e8 + np.array([0.0, 1e-8, 2e-8, 3e-8]), self.library[:, 1]]
            ),
            np.where(np.arange(4)[:, None] == 0, np.inf, self.library),
        )
        for library in bad_libraries:
            with self.subTest(library=library):
                with self.assertRaises(ValueError):
                    ThetaStandardization.from_library(library)

        center = np.mean(self.library, axis=0)
        scale_matrix = np.diag(np.std(self.library, axis=0, ddof=1))
        bad_overrides = (
            {"theta_bar_dagger": center},
            {"D_theta": scale_matrix},
            {"theta_bar_dagger": center[:2], "D_theta": scale_matrix},
            {"theta_bar_dagger": center, "D_theta": np.ones((3, 3))},
            {"theta_bar_dagger": center, "D_theta": np.diag([1.0, 0.0, 1.0])},
            {"theta_bar_dagger": center, "D_theta": np.diag([1.0, 1.0, 1e-320])},
            {"theta_bar_dagger": np.array([np.nan, 1.0, 1.0]), "D_theta": scale_matrix},
        )
        for override in bad_overrides:
            with self.subTest(override=override):
                with self.assertRaises(ValueError):
                    ThetaStandardization.from_library(self.library, **override)


class SiteCoordinatesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.library = np.array(
            [[1.0, 10.0, 0.02], [2.0, 20.0, 0.04], [4.0, 40.0, 0.08]]
        )
        self.frozen = ThetaStandardization.from_library(self.library)

    def test_identity_map_and_prior_center(self) -> None:
        coordinates = SiteCoordinates.from_physical_bounds(self.frozen)
        eta = jnp.array([[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]])
        self.assertFalse(coordinates.bounded)
        np.testing.assert_array_equal(coordinates.eta_to_theta_tilde(eta), eta)
        np.testing.assert_array_equal(coordinates.theta_tilde_to_eta(eta), eta)
        self.assertEqual(float(coordinates.log_jacobian(eta)), 0.0)

        prior = SpatialPrior.from_standardization(self.frozen)
        np.testing.assert_allclose(
            prior.m_theta_0,
            (np.array([37850.0, 24060.0, 0.071]) - np.mean(self.library, axis=0))
            / np.std(self.library, axis=0, ddof=1),
        )
        np.testing.assert_allclose(prior.V_theta_0, 4.0 * np.eye(3))
        np.testing.assert_allclose(prior.S_theta_0, np.eye(3))
        self.assertEqual(prior.nu_theta_0, 5.0)

    def test_box_map_jacobian_and_tails(self) -> None:
        l = np.array([0.5, 8.0, 0.01])
        u = np.array([5.0, 50.0, 0.10])
        coordinates = SiteCoordinates.from_physical_bounds(self.frozen, l=l, u=u)
        self.assertTrue(coordinates.bounded)
        np.testing.assert_allclose(coordinates.l_tilde, self.frozen.to_standardized(l))
        np.testing.assert_allclose(coordinates.u_tilde, self.frozen.to_standardized(u))

        eta = jnp.array([[0.25, -1.2, 1.1], [-0.4, 0.7, -0.3]])
        theta_tilde = jax.jit(coordinates.eta_to_theta_tilde)(eta)
        physical = self.frozen.to_physical(theta_tilde)
        self.assertTrue(bool(jnp.all(physical > l)))
        self.assertTrue(bool(jnp.all(physical < u)))
        np.testing.assert_allclose(coordinates.theta_tilde_to_eta(theta_tilde), eta)

        # Independent finite-difference derivative of each scalar map.
        h = 1e-5
        plus = np.asarray(coordinates.eta_to_theta_tilde(eta + h))
        minus = np.asarray(coordinates.eta_to_theta_tilde(eta - h))
        reference_log_jacobian = np.log((plus - minus) / (2.0 * h)).sum()
        np.testing.assert_allclose(
            jax.jit(coordinates.log_jacobian)(eta),
            reference_log_jacobian,
            rtol=1e-8,
        )

        tails = jnp.array([[-1000.0, 0.0, 1000.0]])
        tail_log_jacobian = jax.jit(coordinates.log_jacobian)(tails)
        self.assertTrue(bool(jnp.isfinite(tail_log_jacobian)))
        np.testing.assert_allclose(
            tail_log_jacobian,
            np.log(np.asarray(coordinates.u_tilde - coordinates.l_tilde)).sum()
            - 2000.0
            - 2.0 * np.log(2.0),
        )
        tail_gradient = jax.grad(coordinates.log_jacobian)(tails)
        self.assertTrue(bool(jnp.all(jnp.isfinite(tail_gradient))))

    def test_bounds_and_nondefault_prior_validation(self) -> None:
        for kwargs in (
            {"l": [0.0, 0.0, 0.0]},
            {"l": [0.0, 0.0, 0.0], "u": [1.0, np.inf, 1.0]},
            {"l": [0.0, 0.0, 0.0], "u": [1.0, 0.0, 1.0]},
            {"l": [0.0, 0.0], "u": [1.0, 1.0]},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    SiteCoordinates.from_physical_bounds(self.frozen, **kwargs)

        two_dimensional = ThetaStandardization.from_library(self.library[:, :2])
        with self.assertRaisesRegex(ValueError, "d != 3"):
            SpatialPrior.from_standardization(two_dimensional)
        explicit = SpatialPrior.from_standardization(
            two_dimensional,
            physical_center=[1.5, 15.0],
            V_theta_0=np.eye(2),
            nu_theta_0=3.0,
            S_theta_0=2.0 * np.eye(2),
        )
        np.testing.assert_allclose(
            explicit.m_theta_0,
            two_dimensional.to_standardized(jnp.array([1.5, 15.0])),
        )
        with self.assertRaisesRegex(ValueError, "positive definite"):
            SpatialPrior.from_standardization(
                two_dimensional,
                physical_center=[1.5, 15.0],
                V_theta_0=np.diag([1.0, -1.0]),
                nu_theta_0=3.0,
                S_theta_0=np.eye(2),
            )


if __name__ == "__main__":
    unittest.main()
