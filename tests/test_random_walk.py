"""Dense proposal checks and quadrature references for Stage 7."""

import unittest
from dataclasses import replace

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal, norm

from bayesiancalibration.gp import LibraryGP
from bayesiancalibration.samplers.metropolis import (
    collapsed_random_walk_sweep,
    update_collapsed_random_walk,
)
from bayesiancalibration.state import SpatialPrior, ThetaStandardization
from bayesiancalibration.targets import CalibrationTarget
from bayesiancalibration.transforms import SiteCoordinates


class CollapsedRandomWalkTest(unittest.TestCase):
    def target_and_state(self, bounded=False, loading_only=False):
        library = np.array([
            [-1.0, -0.7], [0.9, -0.4], [0.1, 1.1], [-0.8, 0.8]
        ])
        standardization = ThetaStandardization.from_library(library)
        sizes = (2,) if loading_only else (2, 1)
        k = sum(sizes)
        gp = LibraryGP.from_data(
            standardization.to_standardized(library),
            np.array([[0.2, -0.4, 0.1], [0.7, 0.1, 0.5],
                      [-0.3, 0.5, -0.1], [0.1, 0.3, 0.2]])[:, :k],
            [0.8, 1.2],
        )
        coordinates = SiteCoordinates.from_physical_bounds(
            standardization,
            **({"l": [-1.8, -1.2], "u": [1.7, 1.8]} if bounded else {}),
        )
        prior = SpatialPrior.from_standardization(
            standardization, physical_center=[0.1, -0.2],
            V_theta_0=2 * np.eye(2), nu_theta_0=4.0, S_theta_0=np.eye(2),
        )
        R_i = block_diag(np.array([[1.1, 0.2], [0.0, 0.9]]), [[1.3]])[:k, :k]
        target = CalibrationTarget.from_data(
            gp, coordinates, prior, np.linspace(-0.1, 0.3, 2 * k),
            block_diag(R_i, 1.2 * R_i), [[0, 0], [6, 0]], sizes,
        )
        state = (
            jnp.array([[-0.3, 0.2], [0.5, -0.1]]),
            jnp.linspace(-0.03, 0.02, k),
            jnp.array([0.09] if loading_only else [0.09, 0.15]),
            jnp.array([0.2, -0.1]),
            jnp.array([[0.8, 0.15], [0.15, 0.6]]),
            jnp.linspace(0.4, 1.2, k),
            jnp.array([[[0.35, 0.1], [0.1, 0.25]],
                       [[0.5, -0.12], [-0.12, 0.3]]]),
        )
        return target, state

    def reference_logdensity(self, target, eta, state):
        """Dense NumPy GP conditioning and SciPy Gaussian densities."""
        _, delta, noise, mean, covariance, variances, _ = state
        theta = np.asarray(target.coordinates.eta_to_theta_tilde(eta))
        library = np.asarray(target.gp.theta_s_tilde)
        length = np.asarray(target.gp.lambda_c)
        diff_ss = (library[:, None] - library[None, :]) / length
        diff_fs = (theta[:, None] - library[None, :]) / length
        diff_ff = (theta[:, None] - theta[None, :]) / length
        C_ss = np.exp(-0.5 * np.sum(diff_ss**2, axis=-1))
        C_fs = np.exp(-0.5 * np.sum(diff_fs**2, axis=-1))
        C_ff = np.exp(-0.5 * np.sum(diff_ff**2, axis=-1))
        m = (C_fs @ np.linalg.solve(C_ss, target.gp.F_s)).reshape(-1)
        C = C_ff - C_fs @ np.linalg.solve(C_ss, C_fs.T)
        R = np.asarray(target.R)
        V_y = (
            R @ np.kron(C, np.diag(variances)) @ R.T
            + np.diag(np.tile(np.repeat(noise, target.branch_sizes), len(theta)))
        )
        m_y = R @ (m + np.tile(delta, len(theta)))
        value = multivariate_normal.logpdf(target.y_tilde, mean=m_y, cov=V_y)
        value += multivariate_normal.logpdf(
            theta.reshape(-1), mean=np.tile(mean, len(theta)),
            cov=np.kron(target.C_theta, covariance),
        )
        if target.coordinates.bounded:
            width = np.asarray(target.coordinates.u_tilde - target.coordinates.l_tilde)
            value += np.sum(
                np.log(width) - np.logaddexp(0, -eta) - np.logaddexp(0, eta)
            )
        return value

    def test_sequential_proposals_and_acceptance_match_dense_reference(self):
        decisions = []
        for bounded in (False, True):
            for loading_only in (False, True):
                target, state = self.target_and_state(bounded, loading_only)
                compiled = jax.jit(lambda key, *args: collapsed_random_walk_sweep(
                    key, target, *args
                ))
                for seed in range(8):
                    key = jax.random.key(810 + seed)
                    eta = state[0]
                    probabilities, accepted = [], []
                    current = self.reference_logdensity(target, eta, state)
                    for i, site_key in enumerate(jax.random.split(key, 2)):
                        proposal_key, accept_key = jax.random.split(site_key)
                        candidate = eta.at[i].set(jax.random.multivariate_normal(
                            proposal_key, eta[i], state[-1][i],
                            dtype=jnp.float64, method="cholesky",
                        ))
                        proposed = self.reference_logdensity(target, candidate, state)
                        probability = np.exp(min(0.0, proposed - current))
                        do_accept = bool(jax.random.bernoulli(accept_key, probability))
                        if do_accept:
                            eta, current = candidate, proposed
                        probabilities.append(probability)
                        accepted.append(do_accept)
                    actual, info = compiled(key, *state)
                    np.testing.assert_allclose(actual, eta, rtol=0, atol=1e-15)
                    np.testing.assert_allclose(
                        info.acceptance_rate, probabilities, rtol=1e-11, atol=1e-13
                    )
                    np.testing.assert_array_equal(info.is_accepted, accepted)
                    np.testing.assert_allclose(info.logdensity, current, rtol=1e-12)
                    decisions.extend(accepted)
        self.assertIn(True, decisions)
        self.assertIn(False, decisions)

    def test_explicit_keys_jit_vmap_and_current_conditioning(self):
        target, state = self.target_and_state(True)
        key = jax.random.key(820)
        compiled = jax.jit(lambda key, *args: collapsed_random_walk_sweep(
            key, target, *args
        ))
        first, info = update_collapsed_random_walk(key, target, *state)
        repeat, repeated_info = compiled(key, *state)
        np.testing.assert_allclose(first, repeat, rtol=0, atol=1e-15)
        np.testing.assert_allclose(info.logdensity, repeated_info.logdensity)
        np.testing.assert_array_equal(info.is_accepted, repeated_info.is_accepted)
        keys = jax.random.split(key, 3)
        draws, infos = jax.jit(jax.vmap(lambda key: compiled(key, *state)))(keys)
        self.assertEqual(draws.dtype, jnp.float64)
        self.assertEqual(draws.shape, (3, 2, 2))
        self.assertEqual(infos.is_accepted.shape, (3, 2))
        self.assertFalse(np.array_equal(draws[0], draws[1]))
        for index in range(6):
            changed = list(state)
            changed[index] = changed[index] * 1.2 + (0.1 if index != 4 else 0)
            result, changed_info = compiled(key, *changed)
            expected = target.theta_only_collapsed(result, *changed[1:6])
            np.testing.assert_allclose(changed_info.logdensity, expected, rtol=1e-12)
            self.assertNotAlmostEqual(
                float(changed_info.logdensity), float(info.logdensity)
            )
        np.testing.assert_array_equal(target.gp.lambda_c, [0.8, 1.2])

    def test_validation_and_structural_singularity(self):
        target, state = self.target_and_state()
        key = jax.random.key(821)
        invalid = (
            (0, state[0][:1]), (0, state[0].at[0, 0].set(jnp.nan)),
            (1, state[1][:1]), (2, state[2].at[0].set(0)),
            (3, state[3][:1]), (4, -state[4]),
            (4, state[4].at[0, 1].set(0)), (5, -state[5]),
            (6, state[6][0]), (6, -state[6]),
            (6, state[6].at[0, 0, 1].set(0)),
            (6, state[6].at[0, 0, 0].set(jnp.inf)),
        )
        for index, value in invalid:
            with self.subTest(index=index):
                changed = list(state)
                changed[index] = value
                with self.assertRaises(ValueError):
                    update_collapsed_random_walk(key, target, *changed)
        for jitter in (0.0, 1e-6):
            gp = LibraryGP.from_data(
                target.gp.theta_s_tilde, target.gp.F_s, target.gp.lambda_c,
                jitter=jitter,
            )
            changed = list(state)
            changed[0] = changed[0].at[0].set(target.gp.theta_s_tilde[0])
            with self.assertRaisesRegex(ValueError, "coincides"):
                update_collapsed_random_walk(key, replace(target, gp=gp), *changed)
        with self.assertRaisesRegex(ValueError, "collapsed log density"):
            update_collapsed_random_walk(
                key, replace(target, y_tilde=jnp.full_like(target.y_tilde, 1e200)),
                *state,
            )

    def test_extreme_proposals_repeat_rejected_states_without_clipping(self):
        for bounded in (False, True):
            target, state = self.target_and_state(bounded)
            changed = list(state)
            changed[-1] = jnp.tile(1e300 * jnp.eye(2), (2, 1, 1))
            result, info = collapsed_random_walk_sweep(
                jax.random.key(822), target, *changed
            )
            np.testing.assert_array_equal(result, state[0])
            np.testing.assert_array_equal(info.is_accepted, [False, False])
            np.testing.assert_array_equal(info.acceptance_rate, [0.0, 0.0])
            self.assertTrue(np.isfinite(float(info.logdensity)))


class RandomWalkPosteriorTest(unittest.TestCase):
    def test_scalar_calibration_posterior_matches_quadrature_in_both_modes(self):
        library = np.array([[-2.0], [0.0], [2.0]])
        standardization = ThetaStandardization.from_library(library)
        theta_s = np.asarray(standardization.to_standardized(library))
        F_s = np.array([[-0.55], [0.2], [1.2]])
        gp = LibraryGP.from_data(theta_s, F_s, [0.65])
        prior = SpatialPrior.from_standardization(
            standardization, physical_center=[0.0],
            V_theta_0=np.eye(1), nu_theta_0=3.0, S_theta_0=np.eye(1),
        )
        C_ss = np.exp(-0.5 * ((theta_s - theta_s.T) / 0.65)**2)
        library_weights = np.linalg.solve(C_ss, F_s)

        def density(theta):
            cross = np.exp(-0.5 * ((theta - theta_s[:, 0]) / 0.65)**2)
            mean = float((cross @ library_weights).item()) + 0.04
            variance = 0.06 + 0.35 * (1 - cross @ np.linalg.solve(C_ss, cross))
            return norm.pdf(0.45, mean, np.sqrt(variance)) * norm.pdf(
                theta, 0.12, np.sqrt(0.45)
            )

        for bounded in (False, True):
            with self.subTest(bounded=bounded):
                coordinates = SiteCoordinates.from_physical_bounds(
                    standardization,
                    **({"l": [-1.5], "u": [1.6]} if bounded else {}),
                )
                target = CalibrationTarget.from_data(
                    gp, coordinates, prior, [0.45], [[1.0]], [[0, 0]], (1,),
                )
                limits = (-0.75, 0.8) if bounded else (-np.inf, np.inf)
                Z = quad(density, *limits, epsabs=1e-11)[0]
                reference = np.array([
                    quad(lambda x: x * density(x), *limits, epsabs=1e-11)[0] / Z,
                    quad(lambda x: x*x * density(x), *limits, epsabs=1e-11)[0] / Z,
                    quad(density, limits[0], 0.1, epsabs=1e-11)[0] / Z,
                ])
                conditioning = (
                    jnp.array([0.04]), jnp.array([0.06]), jnp.array([0.12]),
                    jnp.array([[0.45]]), jnp.array([0.35]), jnp.array([[[0.7]]]),
                )

                def run(key, theta_initial):
                    eta_initial = coordinates.theta_tilde_to_eta(
                        theta_initial.reshape(1, 1)
                    )
                    keys = jax.random.split(key, 26_000)

                    def step(eta, sweep_key):
                        eta_new, info = collapsed_random_walk_sweep(
                            sweep_key, target, eta, *conditioning
                        )
                        return eta_new, coordinates.eta_to_theta_tilde(eta_new)[0, 0]

                    return jax.lax.scan(step, eta_initial, keys)[1]

                chains = np.asarray(jax.jit(jax.vmap(run))(
                    jax.random.split(jax.random.key(830 + int(bounded)), 4),
                    jnp.array([-0.6, -0.2, 0.3, 0.6]),
                ))[:, 2_000:]
                summaries = np.stack(
                    (chains, chains**2, chains <= 0.1), axis=-1
                )
                # Batch means retain rejected states and account for serial
                # correlation. Four independent chains yield 240 batches.
                batches = summaries.reshape(4, 60, 400, 3).mean(axis=2)
                estimates = summaries.mean(axis=(0, 1))
                mcse = batches.reshape(-1, 3).std(axis=0, ddof=1) / np.sqrt(240)
                np.testing.assert_array_less(np.abs(estimates - reference), 6 * mcse)
                for chain in range(4):
                    se = batches[chain].std(axis=0, ddof=1) / np.sqrt(60)
                    np.testing.assert_array_less(
                        np.abs(summaries[chain].mean(axis=0) - reference), 6 * se
                    )
                self.assertGreater(np.mean(np.diff(chains, axis=1) == 0), 0.05)
                if bounded:
                    self.assertTrue(np.all(chains > limits[0]))
                    self.assertTrue(np.all(chains < limits[1]))


if __name__ == "__main__":
    unittest.main()
