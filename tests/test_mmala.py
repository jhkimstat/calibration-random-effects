"""Independent conditional Fisher metrics, asymmetric proposals and posterior checks."""

import unittest

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp
import numpy as np
from scipy.special import expit
from scipy.stats import multivariate_normal

from bayesiancalibration.samplers.mmala import (
    site_observation_conditional_moments, collapsed_mmala_metric,
    mmala_proposal_moments, collapsed_mmala_sweep, update_collapsed_mmala,
)
import test_random_walk as theta_reference
import test_nuts as posterior_reference
from test_gp import _numpy_kernel


def reference_conditional(target, eta, i, inputs):
    """Dense NumPy GP/observation conditioning, independent of project kernels."""

    _, delta, noise, _, _, variances, _ = inputs
    theta = np.asarray(eta)
    if target.coordinates.bounded:
        theta = np.asarray(target.coordinates.l_tilde) + np.asarray(
            target.coordinates.u_tilde-target.coordinates.l_tilde
        ) * expit(theta)
    library, length = np.asarray(target.gp.theta_s_tilde), np.asarray(target.gp.lambda_c)
    kernel = lambda a, b: _numpy_kernel(a, b, length, kernel=target.gp.kernel)
    ss, fs, ff = kernel(library, library), kernel(theta, library), kernel(theta, theta)
    m = (fs @ np.linalg.solve(ss, target.gp.F_s)).reshape(-1)
    C = ff - fs @ np.linalg.solve(ss, fs.T)
    n, k = eta.shape[0], len(variances)
    R = np.asarray(target.R)
    m_y = R @ (m + np.tile(delta, n))
    V_y = R @ np.kron(C, np.diag(variances)) @ R.T + np.diag(
        np.tile(np.repeat(noise, target.branch_sizes), n)
    )
    selected = np.arange(i*k, (i+1)*k)
    other = np.setdiff1d(np.arange(n*k), selected)
    V_io = V_y[np.ix_(selected, other)]
    m_cond = m_y[selected] + V_io @ np.linalg.solve(
        V_y[np.ix_(other, other)], np.asarray(target.y_tilde)[other]-m_y[other]
    )
    V_cond = V_y[np.ix_(selected, selected)] - V_io @ np.linalg.solve(
        V_y[np.ix_(other, other)], V_io.T
    )
    return m_cond, V_cond


def reference_metric(target, eta, i, inputs, ridge):
    """Five-point derivatives of independent conditional moments + prior metric."""

    m, V = reference_conditional(target, eta, i, inputs)
    dm, dV = [], []
    for j in range(eta.shape[1]):
        shift = np.zeros_like(eta)
        shift[i, j] = 1e-4
        values = [reference_conditional(target, eta+a*shift, i, inputs) for a in (-2, -1, 1, 2)]
        dm.append((values[0][0]-8*values[1][0]+8*values[2][0]-values[3][0])/(12e-4))
        dV.append((values[0][1]-8*values[1][1]+8*values[2][1]-values[3][1])/(12e-4))
    dm, dV = np.array(dm).T, np.array(dV)
    A = np.array([np.linalg.solve(V, derivative) for derivative in dV])
    fisher = dm.T @ np.linalg.solve(V, dm) + .5*np.einsum('aij,bji->ab', A, A)
    C = np.asarray(target.C_theta)
    other = np.delete(np.arange(len(eta)), i)
    scale = C[i, i] - C[i, other] @ np.linalg.solve(C[np.ix_(other, other)], C[other, i])
    D, curvature = np.eye(eta.shape[1]), np.zeros(eta.shape[1])
    if target.coordinates.bounded:
        p = expit(eta[i])
        D = np.diag(np.asarray(target.coordinates.u_tilde-target.coordinates.l_tilde)*p*(1-p))
        curvature = 2*p*(1-p)
    G = fisher + D @ np.linalg.solve(scale*np.asarray(inputs[4]), D)
    return G + np.diag(curvature) + ridge*np.eye(eta.shape[1]), dm, dV


class MMALATest(unittest.TestCase):
    def test_conditional_moments_covariance_derivatives_and_dense_metric(self):
        fixture = theta_reference.CollapsedRandomWalkTest()
        covariance_effects = []
        for bounded in (False, True):
            for loading in (False, True):
                target, inputs = fixture.target_and_state(bounded, loading)
                eta, delta, noise, mean, Sigma, variances, _ = inputs
                for i in range(2):
                    m, V = site_observation_conditional_moments(
                        target, eta, i, delta, noise, variances
                    )
                    m_ref, V_ref = reference_conditional(target, np.asarray(eta), i, inputs)
                    np.testing.assert_allclose(m, m_ref, rtol=1e-11, atol=1e-12)
                    np.testing.assert_allclose(V, V_ref, rtol=1e-11, atol=1e-12)
                    expected, dm, dV = reference_metric(target, np.asarray(eta), i, inputs, .07)
                    actual = jax.jit(lambda: collapsed_mmala_metric(
                        target, eta, i, delta, noise, Sigma, variances, .07
                    ))()
                    np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=2e-9)
                    derivative = jax.jacfwd(lambda x: site_observation_conditional_moments(
                        target, eta.at[i].set(x), i, delta, noise, variances
                    ))(eta[i])
                    np.testing.assert_allclose(derivative[0], dm, rtol=2e-8, atol=2e-9)
                    np.testing.assert_allclose(np.moveaxis(derivative[1], -1, 0), dV,
                                               rtol=2e-8, atol=2e-9)
                    self.assertGreater(np.linalg.eigvalsh(actual)[0], .07)
                    covariance_effects.append(np.linalg.norm(dV))
        self.assertGreater(min(covariance_effects), .01)

    def test_scalar_analytic_metric_in_identity_and_nonunit_bounds(self):
        for bounded in (False, True):
            target, batched, _ = posterior_reference.scalar_posterior_start(bounded, 1)
            s = jax.tree.map(lambda x: x[0], batched)
            x = float(target.coordinates.eta_to_theta_tilde(s.eta)[0, 0])
            xs = np.asarray(target.gp.theta_s_tilde)[:, 0]
            C = np.exp(-.5*((xs[:, None]-xs[None, :])/.65)**2)
            f = np.exp(-.5*((x-xs)/.65)**2)
            df = -(x-xs)/.65**2 * f
            m_prime = float(df @ np.linalg.solve(C, np.asarray(target.gp.F_s)[:, 0]))
            V = .06 + .35*(1-f@np.linalg.solve(C, f))
            V_prime = -.7*df @ np.linalg.solve(C, f)
            D, curvature = 1.0, 0.0
            if bounded:
                p = expit(float(s.eta[0, 0]))
                D = 1.55*p*(1-p)
                curvature = 2*p*(1-p)
            expected = (m_prime*D)**2/V + .5*(V_prime*D/V)**2 + D**2/.45 + curvature + .03
            actual = collapsed_mmala_metric(
                target, s.eta, 0, s.delta, s.sigma_y2, s.Sigma_theta, s.sigma_c2, .03
            )
            np.testing.assert_allclose(actual, [[expected]], rtol=1e-11)

    def test_sequential_forward_reverse_proposals_match_independent_reference(self):
        fixture = theta_reference.CollapsedRandomWalkTest()
        corrections, metric_changes, decisions = [], [], []
        epsilon, ridge = .65, .08
        for bounded in (False, True):
            for loading in (False, True):
                target, inputs = fixture.target_and_state(bounded, loading)
                compiled = jax.jit(lambda key: collapsed_mmala_sweep(
                    key, target, *inputs[:-1], epsilon, ridge
                ))
                def density(eta):
                    return fixture.reference_logdensity(target, eta, inputs)
                def moments(eta, i):
                    G, _, _ = reference_metric(target, eta, i, inputs, ridge)
                    grad = []
                    for j in range(eta.shape[1]):
                        shift = np.zeros_like(eta)
                        shift[i, j] = 1e-4
                        grad.append((density(eta-2*shift)-8*density(eta-shift)
                                     +8*density(eta+shift)-density(eta+2*shift))/(12e-4))
                    covariance = epsilon**2*np.linalg.solve(G, np.eye(2))
                    mean = eta[i] + epsilon**2/2*np.linalg.solve(G, grad)
                    return mean, covariance
                for seed in range(6):
                    key = jax.random.key(1300+seed)
                    eta = np.asarray(inputs[0]).copy()
                    probabilities, accepted = [], []
                    for i, site_key in enumerate(jax.random.split(key, 2)):
                        kp, ka = jax.random.split(site_key)
                        m, V = moments(eta, i)
                        candidate = eta.copy()
                        candidate[i] = m + np.linalg.cholesky(V) @ np.asarray(
                            jax.random.normal(kp, (2,), dtype=jnp.float64)
                        )
                        mr, Vr = moments(candidate, i)
                        correction = (multivariate_normal.logpdf(eta[i], mr, Vr)
                                      - multivariate_normal.logpdf(candidate[i], m, V))
                        probability = np.exp(min(0.0, density(candidate)-density(eta)+correction))
                        take = bool(jax.random.bernoulli(ka, probability))
                        if take:
                            eta = candidate
                        probabilities.append(probability)
                        accepted.append(take)
                        corrections.append(correction)
                        metric_changes.append(np.linalg.norm(V-Vr))
                    actual, info = compiled(key)
                    np.testing.assert_allclose(actual, eta, rtol=0, atol=2e-8)
                    np.testing.assert_allclose(info.acceptance_rate, probabilities,
                                               rtol=3e-7, atol=2e-8)
                    np.testing.assert_array_equal(info.is_accepted, accepted)
                    decisions.extend(accepted)
        self.assertIn(True, decisions)
        self.assertIn(False, decisions)
        self.assertGreater(max(np.abs(corrections)), .1)
        self.assertGreater(max(metric_changes), .01)

    def test_rng_current_dependencies_and_checked_failures(self):
        target, inputs = theta_reference.CollapsedRandomWalkTest().target_and_state(True)
        key = jax.random.key(1307)
        kernel = jax.jit(lambda key, *values: collapsed_mmala_sweep(key, target, *values, .5, .07))
        first = kernel(key, *inputs[:-1])
        second = kernel(key, *inputs[:-1])
        for a, b in zip(jax.tree.leaves(first), jax.tree.leaves(second)):
            np.testing.assert_array_equal(a, b)
        keys = jax.random.split(key, 3)
        vectorized = jax.jit(jax.vmap(lambda k: kernel(k, *inputs[:-1])))(keys)
        self.assertFalse(np.array_equal(vectorized[0][0], vectorized[0][1]))
        baseline_metric = collapsed_mmala_metric(
            target, inputs[0], 0, inputs[1], inputs[2], inputs[4], inputs[5], .07
        )
        for index in (0, 2, 4, 5):
            changed = list(inputs[:-1])
            changed[index] = changed[index]*1.25 + (0.1 if index == 0 else 0)
            actual_metric = collapsed_mmala_metric(
                target, changed[0], 0, changed[1], changed[2], changed[4], changed[5], .07
            )
            self.assertFalse(np.allclose(actual_metric, baseline_metric))
        # Delta/mean shift only the gradient; observed complement affects Fisher.
        changed = list(inputs[:-1])
        changed[1] = changed[1] + .2
        self.assertFalse(np.allclose(kernel(key, *changed)[1].logdensity, first[1].logdensity))
        changed[3] = changed[3] + .3
        self.assertFalse(np.allclose(kernel(key, *changed)[1].logdensity, first[1].logdensity))
        from dataclasses import replace
        shifted_data = replace(target, y_tilde=target.y_tilde.at[-1].add(.4))
        changed_metric = collapsed_mmala_metric(
            shifted_data, inputs[0], 0, inputs[1], inputs[2], inputs[4], inputs[5], .07
        )
        self.assertFalse(np.allclose(changed_metric, baseline_metric))
        for ridge in (0, -1, np.inf, np.nan, True, [.07]):
            with self.assertRaises(ValueError):
                update_collapsed_mmala(key, target, *inputs[:-1], .5, ridge)
        for epsilon in (0, -1, np.inf, np.nan, True):
            with self.assertRaises(ValueError):
                update_collapsed_mmala(key, target, *inputs[:-1], epsilon, .07)
        repeated = list(inputs[:-1])
        repeated[0] = repeated[0].at[1].set(repeated[0][0])
        with self.assertRaises(ValueError):
            update_collapsed_mmala(key, target, *repeated, .5, .07)
        actual, info = collapsed_mmala_sweep(key, target, *inputs[:-1], 1e150, .07)
        np.testing.assert_array_equal(actual, inputs[0])
        np.testing.assert_array_equal(info.is_accepted, [False, False])

    def test_joint_posterior_moments_preserved_against_quadrature(self):
        from bayesiancalibration.gibbs import sample_projected_coefficients
        for bounded in (False, True):
            target, state, expected = posterior_reference.scalar_posterior_start(bounded)
            def step(key, s):
                kt, kf = jax.random.split(key)
                eta, _ = collapsed_mmala_sweep(
                    kt, target, s.eta, s.delta, s.sigma_y2, s.mu_theta,
                    s.Sigma_theta, s.sigma_c2, .7, .05,
                )
                m, _, V = target.gp.conditional_moments(
                    target.coordinates.eta_to_theta_tilde(eta), s.sigma_c2
                )
                d, noise = target.observation_arrays(s.delta, s.sigma_y2)
                c = sample_projected_coefficients(kf, target.y_tilde, m, V, target.R, d, noise)
                return s._replace(eta=eta, c_f=c)
            kernel = jax.jit(jax.vmap(step))
            for sweep in range(3):
                keys = jax.random.split(jax.random.key(1310+sweep), state.eta.shape[0])
                state = kernel(keys, state)
                values = posterior_reference.posterior_summaries(target, state)
                se = values.std(axis=0, ddof=1)/np.sqrt(len(values))
                np.testing.assert_array_less(np.abs(values.mean(axis=0)-expected), 6*se)


if __name__ == '__main__':
    unittest.main()
