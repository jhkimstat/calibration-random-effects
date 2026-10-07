"""Independent Matérn values, derivatives, GP fitting and sampler geometry."""

from dataclasses import replace
import unittest

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from scipy.stats import multivariate_normal

from bayesiancalibration.gp import (
    COEFFICIENT_KERNELS, LibraryGP, coefficient_kernel, cv_nlpd, cv_wmse,
    fit_library_length_scales, profile_log_likelihood, squared_exponential_kernel,
)
from bayesiancalibration.samplers.mmala import collapsed_mmala_metric
from test_gp import _numpy_kernel
from test_mmala import reference_metric
import test_random_walk as theta_reference


class CoefficientKernelTest(unittest.TestCase):
    def setUp(self):
        self.theta_s = np.array([[-1., -.7], [.9, -.4], [.1, 1.1], [-.8, .8]])
        self.theta_f = np.array([[-.3, .2], [.5, -.1]])
        self.F_s = np.array([[.2, -.4], [.7, .1], [-.3, .5], [.1, .2]])
        self.length = np.array([.8, 1.2])

    def test_values_ard_scaling_unit_diagonal_and_default(self):
        for kernel in COEFFICIENT_KERNELS:
            with self.subTest(kernel=kernel):
                run = jax.jit(coefficient_kernel, static_argnames='kernel')
                actual = run(jnp.asarray(self.theta_s), jnp.asarray(self.theta_s),
                             jnp.asarray(self.length), kernel=kernel)
                np.testing.assert_allclose(actual, _numpy_kernel(
                    self.theta_s, self.theta_s, self.length, kernel), rtol=1e-10, atol=1e-12)
                np.testing.assert_array_equal(np.diag(actual), np.ones(4))
                cross = run(jnp.asarray(self.theta_f), jnp.asarray(self.theta_s),
                            jnp.asarray(self.length), kernel=kernel)
                np.testing.assert_allclose(cross, _numpy_kernel(
                    self.theta_f, self.theta_s, self.length, kernel), rtol=1e-10, atol=1e-12)
        default = coefficient_kernel(jnp.asarray(self.theta_f), jnp.asarray(self.theta_s), self.length)
        np.testing.assert_allclose(default, _numpy_kernel(
            self.theta_f, self.theta_s, self.length, 'matern32'), rtol=1e-10)
        self.assertEqual(LibraryGP.from_data(self.theta_s, self.F_s, self.length).kernel, 'matern32')
        np.testing.assert_array_equal(
            squared_exponential_kernel(jnp.asarray(self.theta_f), jnp.asarray(self.theta_s), self.length),
            coefficient_kernel(jnp.asarray(self.theta_f), jnp.asarray(self.theta_s), self.length, kernel='se'))

    def test_input_and_length_derivatives_at_zero_and_nonzero_distance(self):
        x = jnp.asarray(self.theta_s[0])
        b = jnp.asarray(self.theta_s)
        for kernel in COEFFICIENT_KERNELS:
            with self.subTest(kernel=kernel), jax.debug_nans(True):
                f = lambda a: coefficient_kernel(a[None], b, self.length, kernel=kernel)[0]
                derivative = jax.jit(jax.jacfwd(f))(x)
                difference = np.asarray(x) - self.theta_s
                r = np.sqrt(np.sum((difference / self.length)**2, axis=-1))
                if kernel == 'se':
                    factor = np.exp(-.5*r**2)
                elif kernel == 'matern32':
                    factor = 3*np.exp(-np.sqrt(3)*r)
                else:
                    factor = (5/3)*(1+np.sqrt(5)*r)*np.exp(-np.sqrt(5)*r)
                expected = -factor[:, None]*difference/self.length**2
                np.testing.assert_allclose(derivative, expected, rtol=1e-8, atol=1e-11)
                hessian = jax.jit(jax.hessian(lambda a: f(a)[0]))(x)
                curvature = {'se':1., 'matern32':3., 'matern52':5/3}[kernel]
                np.testing.assert_allclose(hessian, -curvature*np.diag(1/self.length**2),
                                           rtol=1e-8, atol=1e-11)
                length_gradient = jax.jit(jax.grad(lambda xi: jnp.sum(coefficient_kernel(
                    b, b, jnp.exp(xi), kernel=kernel))))(jnp.log(self.length))
                self.assertTrue(np.all(np.isfinite(length_gradient)))
                # Both first-argument jacfwd and joint-input reverse-mode see
                # self/cross coincidences, without a distance floor.
                joint = jax.jit(jax.grad(lambda a: jnp.sum(coefficient_kernel(
                    a, a, self.length, kernel=kernel))))(b)
                self.assertTrue(np.all(np.isfinite(joint)))
                near = jnp.stack([x, x+jnp.array([1e-9, -1e-9])])
                tiny = jax.jit(jax.jacfwd(lambda a: coefficient_kernel(
                    a, near, self.length, kernel=kernel)))(near)
                self.assertTrue(np.all(np.isfinite(tiny)))

    def test_conditioning_and_library_density_match_dense_bessel_reference(self):
        sigma_c2 = np.array([.4, 1.7])
        for kernel in COEFFICIENT_KERNELS:
            for jitter in (0., 1e-6):
                with self.subTest(kernel=kernel, jitter=jitter):
                    gp = LibraryGP.from_data(self.theta_s, self.F_s, self.length,
                                             kernel=kernel, jitter=jitter)
                    gp.validate_field_sites(self.theta_f)
                    m, C, Sigma = jax.jit(gp.conditional_moments)(self.theta_f, sigma_c2)
                    ss = _numpy_kernel(self.theta_s, self.theta_s, self.length, kernel)+jitter*np.eye(4)
                    fs = _numpy_kernel(self.theta_f, self.theta_s, self.length, kernel)
                    ff = _numpy_kernel(self.theta_f, self.theta_f, self.length, kernel)
                    expected_m = (fs@np.linalg.solve(ss, self.F_s)).reshape(-1)
                    expected_C = ff-fs@np.linalg.solve(ss, fs.T)
                    np.testing.assert_allclose(m, expected_m, rtol=1e-9, atol=1e-11)
                    np.testing.assert_allclose(C, expected_C, rtol=1e-9, atol=1e-11)
                    np.testing.assert_allclose(Sigma, np.kron(expected_C, np.diag(sigma_c2)),
                                               rtol=1e-9, atol=1e-11)
                    expected = multivariate_normal.logpdf(
                        self.F_s.reshape(-1), cov=np.kron(ss, np.diag(sigma_c2)))
                    np.testing.assert_allclose(gp.library_logpdf(sigma_c2), expected, rtol=1e-9)
                    with self.assertRaisesRegex(ValueError, 'coincides'):
                        gp.validate_field_sites(self.theta_s[:1])
                    with self.assertRaisesRegex(ValueError, 'Repeated library'):
                        LibraryGP.from_data(np.vstack([self.theta_s, self.theta_s[0]]),
                                            np.vstack([self.F_s, self.F_s[0]]), self.length,
                                            kernel=kernel, jitter=1e-6)

    def reference_scores(self, xi, kernel, jitter):
        C = _numpy_kernel(self.theta_s, self.theta_s, np.exp(xi), kernel)
        r, k = self.F_s.shape
        full = C+jitter*np.eye(r)
        q = np.sum(self.F_s*np.linalg.solve(full, self.F_s), axis=0)
        profile = -.5*k*np.linalg.slogdet(full)[1]-.5*r*np.log(q/r).sum()
        nlpd, wmse = [], []
        for held in range(r):
            train = np.arange(r) != held
            training = C[np.ix_(train, train)]+jitter*np.eye(r-1)
            cross = C[held, train]
            solved = np.linalg.solve(training, self.F_s[train])
            predicted = cross@solved
            scales = np.sum(self.F_s[train]*solved, axis=0)/(r-1)
            variance = (1-cross@np.linalg.solve(training, cross))*scales
            nlpd.append(-multivariate_normal.logpdf(self.F_s[held], predicted, np.diag(variance)))
            wmse.append(np.sum((self.F_s[held]-predicted)**2/variance))
        return profile, np.mean(nlpd), np.mean(wmse)

    def test_all_fitting_objectives_and_gradients_use_selected_kernel(self):
        xi = np.log(self.length)
        objectives = (profile_log_likelihood, cv_nlpd, cv_wmse)
        for kernel in COEFFICIENT_KERNELS:
            for jitter in (0., 1e-6):
                scores = self.reference_scores(xi, kernel, jitter)
                for index, objective in enumerate(objectives):
                    with self.subTest(kernel=kernel, jitter=jitter, objective=objective.__name__):
                        actual = objective(xi, self.theta_s, self.F_s, kernel=kernel, jitter=jitter)
                        np.testing.assert_allclose(actual, scores[index], rtol=1e-9, atol=1e-10)
                        gradient = jax.jit(jax.grad(lambda z: objective(
                            z, self.theta_s, self.F_s, kernel=kernel, jitter=jitter)))(jnp.asarray(xi))
                        h = 1e-5
                        expected = [(self.reference_scores(xi+h*np.eye(2)[q], kernel, jitter)[index]
                                     -self.reference_scores(xi-h*np.eye(2)[q], kernel, jitter)[index])/(2*h)
                                    for q in range(2)]
                        np.testing.assert_allclose(gradient, expected, rtol=1e-5, atol=1e-7)

    def test_fitted_kernel_records_and_invalid_choices(self):
        for kernel in ('matern32', 'matern52'):
            for method, objective in [('profile',profile_log_likelihood), ('cv_nlpd',cv_nlpd),
                                      ('cv_wmse',cv_wmse)]:
                with self.subTest(kernel=kernel, method=method):
                    fit = fit_library_length_scales(
                        self.theta_s, self.F_s, kernel=kernel, method=method,
                        starts=np.array([[-.5, -.5]]), log_bounds=np.array([[-2., .5], [-2., .5]]),
                        gtol=1e-6, ftol=1e-10, maxiter=200)
                    self.assertEqual(fit.kernel, kernel)
                    self.assertTrue(any(a.success for a in fit.attempts))
                    np.testing.assert_allclose(fit.objective, objective(
                        fit.log_lambda_c, self.theta_s, self.F_s, kernel=kernel), rtol=1e-9)
                    gp = LibraryGP.from_data(self.theta_s, self.F_s, fit.lambda_c, kernel=fit.kernel)
                    np.testing.assert_allclose(fit.profiled_variances, gp.q_s/len(self.theta_s), rtol=1e-9)
        with self.assertRaisesRegex(ValueError, 'kernel'):
            LibraryGP.from_data(self.theta_s, self.F_s, self.length, kernel='invalid')
        with self.assertRaisesRegex(ValueError, 'kernel'):
            fit_library_length_scales(self.theta_s, self.F_s, kernel='invalid',
                                     starts=[[-.5, -.5]], gtol=1e-6, ftol=1e-10, maxiter=10)

    def test_matern_collapsed_gradients_and_mmala_fisher_reference(self):
        for kernel in ('matern32', 'matern52'):
            for bounded in (False, True):
                for loading in (False, True):
                    with self.subTest(kernel=kernel, bounded=bounded, loading=loading):
                        target, inputs = theta_reference.CollapsedRandomWalkTest().target_and_state(bounded, loading)
                        target = replace(target, gp=LibraryGP.from_data(
                            target.gp.theta_s_tilde, target.gp.F_s, target.gp.lambda_c, kernel=kernel))
                        eta, delta, noise, mean, Sigma, variances, _ = inputs
                        value, gradient = jax.jit(jax.value_and_grad(lambda a: target.theta_only_collapsed(
                            a, delta, noise, mean, Sigma, variances)))(eta)
                        self.assertTrue(np.isfinite(value))
                        self.assertTrue(np.all(np.isfinite(gradient)))
                        h = 1e-5
                        for i in range(2):
                            for q in range(2):
                                f = lambda a: target.theta_only_collapsed(a, delta, noise, mean, Sigma, variances)
                                expected = (f(eta.at[i,q].add(h))-f(eta.at[i,q].add(-h)))/(2*h)
                                np.testing.assert_allclose(gradient[i,q], expected, rtol=1e-5, atol=1e-7)
                            expected, _, _ = reference_metric(target, np.asarray(eta), i, inputs, .07)
                            actual = jax.jit(lambda: collapsed_mmala_metric(
                                target, eta, i, delta, noise, Sigma, variances, .07))()
                            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)
                            self.assertGreater(np.linalg.eigvalsh(actual)[0], 0.)
                        # K_theta remains the original spatial SE correlation.
                        np.testing.assert_allclose(target.C_theta[0,1],
                                                   np.exp(-.5*(6/target.lambda_theta)**2), rtol=1e-10)


if __name__ == '__main__':
    unittest.main()
