"""Eta-space mass structures: moment identities, standard driver, Gibbs restart."""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import blackjax
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from blackjax.adaptation.staged_adaptation import build_schedule
from blackjax.mcmc import nuts

from bayesiancalibration import comparison
from bayesiancalibration.adaptation import (
    nuts_window_adapter, kronecker_factors, validate_nuts_mass,
)
from bayesiancalibration.mcmc import (
    initialize_nuts_warmup, run_nuts_warmup, run_fixed_nuts, validate_nuts_chain,
)
from bayesiancalibration.samplers.nuts import nuts_sweep
from bayesiancalibration.run import save_checkpoint, load_checkpoint
import test_mcmc as reference


class NUTSMassTest(unittest.TestCase):
    assert_tree_equal = reference.GibbsSweepTest.assert_tree_equal

    def test_online_factor_scatter_matches_centered_batch_at_every_step(self):
        rng = np.random.default_rng(945)
        draws = rng.normal(size=(31, 3, 2)) + np.array([12., -7.])
        draws[7:10] = draws[6]  # Repeated states contribute to moments.
        init, update, _ = nuts_window_adapter(.8, 'kronecker', (3, 2))
        state = init(jnp.zeros(6), .1)
        step = jax.jit(lambda s, x: update(s, (1, False), x.reshape(-1), jnp.zeros(6), .8))
        for k, draw in enumerate(draws, 1):
            state = step(state, jnp.asarray(draw))
            wc = state.imm_state.wc_state
            residual = draws[:k] - draws[:k].mean(axis=0)
            site = sum(x @ x.T for x in residual)
            param = sum(x.T @ x for x in residual)
            np.testing.assert_allclose(wc.mean.reshape(3, 2), draws[:k].mean(axis=0), atol=1e-14)
            np.testing.assert_allclose(wc.m2_site, site, atol=1e-12)
            np.testing.assert_allclose(wc.m2_param, param, atol=1e-12)
            self.assertEqual(int(wc.sample_size), k)
            self.assertFalse(hasattr(wc, 'm2'))
            self.assertEqual(sum(x.size for x in (wc.mean, wc.m2_site, wc.m2_param)), 6+9+4)
        boundary = jax.jit(lambda s: update(
            s, (1, True), jnp.asarray(draws[-1]).reshape(-1), jnp.zeros(6), .8))(state)
        S = np.cov(
            np.concatenate((draws, draws[-1:])).reshape(32, 6), rowvar=False)
        regularized = 32/37 * S + 5e-3/37*np.eye(6)
        Gamma_site, Gamma_param = kronecker_factors(jnp.asarray(regularized), 3, 2)
        np.testing.assert_allclose(boundary.inverse_mass_matrix,
                                   np.kron(Gamma_site, Gamma_param), atol=1e-13)
        np.testing.assert_array_equal(boundary.imm_state.wc_state.m2_site, np.zeros((3, 3)))
        np.testing.assert_array_equal(boundary.imm_state.wc_state.m2_param, np.zeros((2, 2)))

    def test_partial_trace_reference_and_regularization(self):
        rng = np.random.default_rng(941)
        x = rng.normal(size=(18, 6))
        empirical = np.cov(x, rowvar=False)
        init, update, _ = nuts_window_adapter(.8, 'kronecker', (3, 2))
        state = init(jnp.zeros(6), .1)
        for sample in x:
            state = update(state, (1, False), jnp.array(sample), jnp.zeros(6), .8)
        # Add a final sample to exercise a window boundary and reset.
        last = rng.normal(size=6)
        state = update(state, (1, True), jnp.array(last), jnp.zeros(6), .8)
        empirical = np.cov(np.vstack((x, last)), rowvar=False)
        blocks = empirical.reshape(3, 2, 3, 2)
        site = sum(blocks[:, q, :, q] for q in range(2)) / 2
        param = sum(blocks[i, :, i, :] for i in range(3)) / 3
        site = 19/24 * site + 5e-3/24*np.eye(3)
        param = 19/24 * param + 5e-3/24*np.eye(2)
        Gamma_site = 3*site/np.trace(site)
        np.testing.assert_allclose(state.inverse_mass_matrix,
                                   np.kron(Gamma_site, param), atol=1e-14)
        self.assertEqual(int(state.imm_state.wc_state.sample_size), 0)
        A = np.array([[2., -.4, .1], [-.4, 1., .2], [.1, .2, 3.]])
        B = np.array([[.7, -.2], [-.2, .9]])
        Gs, Gp = kronecker_factors(jnp.array(np.kron(A, B)), 3, 2)
        np.testing.assert_allclose(np.trace(Gs), 3)
        np.testing.assert_allclose(np.kron(Gs, Gp), np.kron(A, B), atol=1e-14)
        validate_nuts_mass(np.kron(Gs, Gp), 6, 'kronecker', (3, 2))
        # Repeated positions still yield an SPD separable metric via the declared ridge.
        repeated = init(jnp.zeros(6), .1)
        for stage in ((1, False), (1, True)):
            repeated = update(repeated, stage, jnp.zeros(6), jnp.zeros(6), .8)
        np.testing.assert_allclose(repeated.inverse_mass_matrix, 5e-3/7*np.eye(6))

    def test_dense_matches_blackjax_window_driver(self):
        density = lambda x: -.5*jnp.dot(x, jnp.linalg.solve(
            jnp.array([[1., -.4], [-.4, .5]]), x))
        key = jax.random.key(942)
        driver = blackjax.window_adaptation(blackjax.nuts, density, is_mass_matrix_diagonal=False)
        expected, _ = jax.jit(lambda: driver.run(key, jnp.zeros(2), num_steps=60))()
        # A single-site Kronecker moment estimator must reduce to dense covariance.
        for structure in ('dense', 'kronecker'):
            init, update, final = nuts_window_adapter(.8, structure, (1, 2))
            def step(carry, xs):
                state, adapt = carry
                k, stage = xs
                s, info = nuts.build_kernel()(k, state, density, adapt.step_size,
                                              adapt.inverse_mass_matrix)
                a = update(adapt, stage, s.position, s.logdensity_grad, info.acceptance_rate)
                return (s, a), None
            (_, adapt), _ = jax.jit(lambda: jax.lax.scan(
                step, (nuts.init(jnp.zeros(2), density), init(jnp.zeros(2), 1.)),
                (jax.random.split(key, 60), build_schedule(60))))()
            step_size, mass = final(adapt)
            np.testing.assert_allclose(mass, expected.parameters['inverse_mass_matrix'], rtol=1e-10)
            np.testing.assert_allclose(step_size, expected.parameters['step_size'], rtol=1e-10)

    def test_eta_kernel_matches_dense_reference_for_both_targets_and_bounds(self):
        for bounded in (False, True):
            target, state, _ = reference.make_fixture(bounded)
            for collapsed in (False, True):
                for structure in ('dense', 'kronecker'):
                    mass = jnp.kron(jnp.array([[1., -.2], [-.2, .8]]),
                                    jnp.array([[.6, .1], [.1, 1.2]]))
                    if structure == 'dense':
                        mass = mass.at[0, 3].add(.03).at[3, 0].add(.03)
                    def density(x):
                        eta = x.reshape(2, 2)
                        if collapsed:
                            return target.theta_only_collapsed(eta, state.delta, state.sigma_y2,
                                state.mu_theta, state.Sigma_theta, state.sigma_c2)
                        return target.theta_only_uncollapsed(eta, state.c_f, state.mu_theta,
                                                            state.Sigma_theta, state.sigma_c2)
                    key = jax.random.key(943)
                    expected, info = jax.jit(lambda: nuts.build_kernel()(
                        key, nuts.init(state.eta.reshape(-1), density), density, .1, mass, 3))()
                    eta, got = jax.jit(lambda: nuts_sweep(
                        key, target, state, .1, mass, max_num_doublings=3, collapsed=collapsed))()
                    np.testing.assert_array_equal(eta.reshape(-1), expected.position)
                    np.testing.assert_array_equal(got.acceptance_rate, info.acceptance_rate)

    def test_structured_warmup_restart_and_freeze(self):
        for structure in ('dense', 'kronecker'):
            for collapsed in (False, True):
                target, state, _ = reference.make_fixture(bounded=collapsed, loading_only=True)
                chain = initialize_nuts_warmup(target, state, jax.random.key(944),
                    num_warmup=24, max_num_doublings=3, collapsed=collapsed,
                    mass_structure=structure)
                complete, draws, info = run_nuts_warmup(target, chain)
                part, _, _ = run_nuts_warmup(target, chain, 12)
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp)/'state.npz'
                    for boundary in (chain, part, complete):
                        save_checkpoint(path, target, boundary)
                        restored, metadata = load_checkpoint(path, target)
                        self.assertEqual(restored.mass_structure, structure)
                        self.assertEqual(metadata['nuts']['coordinates'], 'eta-v1')
                        self.assert_tree_equal(restored.adaptation.state, boundary.adaptation.state, True)
                        if restored.phase == 'warmup':
                            restored, rest, diagnostics = run_nuts_warmup(target, restored)
                            self.assert_tree_equal(rest, jax.tree.map(lambda x: x[boundary.iteration:], draws), True)
                            self.assert_tree_equal(diagnostics, jax.tree.map(lambda x: x[boundary.iteration:], info), True)
                        production, _, _ = run_fixed_nuts(target, restored, 2)
                        self.assert_tree_equal(production.adaptation.state, complete.adaptation.state, True)
                        np.testing.assert_array_equal(production.inverse_mass_matrix, complete.inverse_mass_matrix)
                        self.assertEqual(production.step_size, complete.step_size)
                validate_nuts_mass(complete.inverse_mass_matrix, 4, structure, (2, 2))
                self.assertFalse(np.allclose(complete.inverse_mass_matrix, np.eye(4)))
                with self.assertRaises(ValueError):
                    validate_nuts_chain(target, replace(complete, inverse_mass_matrix=-jnp.eye(4)))

    def test_comparison_dispatch_and_structure_validation(self):
        config = json.loads((Path(__file__).resolve().parents[1] /
                             'experiments/comparison.json').read_text())
        target, state, _ = reference.make_fixture(loading_only=True)
        for structure in ('diagonal', 'dense', 'kronecker'):
            config['nuts_mass_structure'] = structure
            comparison.validate_config(config)
            for method in ('nuts', 'collapsed_nuts'):
                chain = comparison.initialize_chain(target, state, config, method, 0)
                self.assertEqual(chain.mass_structure, structure)
                shape = (4,) if structure == 'diagonal' else (4, 4)
                self.assertEqual(chain.inverse_mass_matrix.shape, shape)
        config['nuts_mass_structure'] = 'invalid'
        with self.assertRaises(ValueError):
            comparison.validate_config(config)

    def test_invalid_structure_and_spd(self):
        for structure in ('dense', 'kronecker'):
            for value in (np.ones(4), -np.eye(4), np.ones((4, 4)),
                          np.eye(4) + np.triu(np.ones((4, 4)), 1)):
                with self.assertRaises(ValueError):
                    validate_nuts_mass(value, 4, structure, (2, 2))
        with self.assertRaises(ValueError):
            nuts_window_adapter(mass_structure='unknown')
        with self.assertRaises(ValueError):
            nuts_window_adapter(mass_structure='kronecker')
        nonseparable = np.diag([1., 2., 3., 7.])
        validate_nuts_mass(nonseparable, 4, 'dense')
        with self.assertRaises(ValueError):
            validate_nuts_mass(nonseparable, 4, 'kronecker', (2, 2))
