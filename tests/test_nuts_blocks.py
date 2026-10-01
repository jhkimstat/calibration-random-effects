"""Sequential conditional NUTS references, ragged blocks and numerical batches."""
from dataclasses import replace
import unittest

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from blackjax.mcmc import nuts
from blackjax.adaptation.staged_adaptation import build_schedule

from bayesiancalibration import mcmc
from bayesiancalibration.adaptation import nuts_window_adapter
from bayesiancalibration.samplers.nuts import nuts_sweep, nuts_blocks
from bayesiancalibration.targets import CalibrationTarget
import test_mcmc as reference


def fixture(bounded=False, loading=False):
    target, state, _ = reference.make_fixture(bounded, loading)
    k = state.sigma_c2.size
    target = CalibrationTarget.from_data(
        target.gp, target.coordinates, target.spatial_prior,
        np.linspace(-.1, .3, 3*k), np.diag(np.linspace(.8, 1.3, 3*k)),
        [[0, 0], [6, 0], [0, 6]], target.branch_sizes, lambda_theta=2.,
        **{name: getattr(target, name) for name in
           ('alpha_y_0', 'beta_y_0', 'alpha_c_0', 'beta_c_0')})
    state = state._replace(eta=jnp.array([[-.3, .2], [.5, -.1], [.1, .7]]),
                           c_f=jnp.linspace(-.15, .25, 3*k))
    return target, state


def density(target, state, eta, collapsed):
    if collapsed:
        return target.theta_only_collapsed(eta, state.delta, state.sigma_y2,
                                           state.mu_theta, state.Sigma_theta, state.sigma_c2)
    return target.theta_only_uncollapsed(eta, state.c_f, state.mu_theta,
                                        state.Sigma_theta, state.sigma_c2)


def sequential_reference(key, target, state, steps, masses, collapsed, stale=False):
    # Independent explicit two-block reference: first two sites, then the last.
    eta, infos = state.eta, []
    for i, ((start, stop), k) in enumerate(zip(((0, 2), (2, 3)), jax.random.split(key, 2))):
        outside = state.eta if stale else eta
        logp = lambda x: density(target, state, outside.at[start:stop].set(
            x.reshape(stop-start, 2)), collapsed)
        selected, info = nuts.build_kernel()(
            k, nuts.init(eta[start:stop].reshape(-1), logp), logp, steps[i], masses[i], 3)
        eta = eta.at[start:stop].set(selected.position.reshape(stop-start, 2))
        infos.append(info)
    return eta, infos


class BlockNUTSTest(unittest.TestCase):
    def equal(self, a, b, exact=False):
        self.assertEqual(jax.tree.structure(a), jax.tree.structure(b))
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
            if hasattr(x, 'dtype') and jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key):
                x, y = jax.random.key_data(x), jax.random.key_data(y)
            if exact:
                np.testing.assert_array_equal(x, y)
            else:
                np.testing.assert_allclose(x, y, rtol=1e-11, atol=1e-12)

    def test_sequential_reference_both_targets_bounds_and_mass_structures(self):
        for bounded in (False, True):
            target, state = fixture(bounded, bounded)
            for collapsed in (False, True):
                for structure in ('diagonal', 'dense', 'kronecker'):
                    with self.subTest(bounded=bounded, collapsed=collapsed, mass=structure):
                        if structure == 'diagonal':
                            masses = (jnp.array([.7, 1.2, .9, 1.1]), jnp.array([.8, 1.3]))
                        else:
                            masses = (jnp.kron(jnp.array([[1., -.2], [-.2, .8]]),
                                               jnp.array([[.7, .1], [.1, 1.3]])),
                                      jnp.array([[.8, .1], [.1, 1.3]]))
                        steps, key = jnp.array([.035, .06]), jax.random.key(782)
                        actual, info = jax.jit(lambda k, st: nuts_sweep(
                            k, target, st, steps, masses, collapsed=collapsed,
                            block_size=2, max_num_doublings=3))(key, state)
                        expected, refs = jax.jit(lambda k, st: sequential_reference(
                            k, target, st, steps, masses, collapsed))(key, state)
                        self.equal(actual, expected)
                        np.testing.assert_allclose(info.acceptance_rate,
                                                   jnp.stack([i.acceptance_rate for i in refs]))
                        np.testing.assert_array_equal(info.is_divergent,
                                                      [i.is_divergent for i in refs])
                        np.testing.assert_allclose(info.logdensity,
                                                   density(target, state, actual, collapsed), rtol=1e-11)
                        stale, _ = jax.jit(lambda k, st: sequential_reference(
                            k, target, st, steps, masses, collapsed, stale=True))(key, state)
                        self.assertGreater(float(jnp.max(jnp.abs(actual[2]-stale[2]))), 1e-10)
                        self.assertEqual(info.step_size.shape, (2,))
                        self.assertEqual(info.inverse_mass_matrix[0].shape, masses[0].shape)
                        self.assertEqual(info.inverse_mass_matrix[1].shape, masses[1].shape)

    def test_one_block_exact_equivalence_and_invalid_sizes(self):
        target, state = fixture()
        key = jax.random.key(783)
        for structure in ('diagonal', 'dense', 'kronecker'):
            a = mcmc.initialize_nuts_warmup(target, state, key, num_warmup=3,
                    initial_step_size=.02, max_num_doublings=2, mass_structure=structure)
            b = mcmc.initialize_nuts_warmup(target, state, key, num_warmup=3,
                    initial_step_size=.02, max_num_doublings=2, mass_structure=structure, block_size=3)
            self.assertIsNone(b.block_size)
            out_a, sa, da = mcmc.SweepRunner(target, a)(a, 2)
            out_b, sb, db = mcmc.SweepRunner(target, b)(b, 2)
            self.equal((sa, da, out_a.key), (sb, db, out_b.key), exact=True)
        self.assertEqual(nuts_blocks(60, 12), tuple((i, i+12) for i in range(0, 60, 12)))
        self.assertEqual(nuts_blocks(5, 2), ((0, 2), (2, 4), (4, 5)))
        for bad in (0, -1, True, 1.5, 4):
            with self.assertRaises(ValueError):
                mcmc.initialize_nuts_warmup(target, state, key, block_size=bad)

    def test_adaptation_batches_freeze_and_ragged_diagnostics(self):
        for collapsed in (False, True):
            for structure in ('diagonal', 'kronecker', 'dense'):
                with self.subTest(collapsed=collapsed, mass=structure):
                    target, state = fixture(collapsed, collapsed)
                    key = jax.random.PRNGKey(784) if collapsed else jax.random.key(784)
                    initial = mcmc.initialize_nuts_warmup(
                        target, state, key, num_warmup=6, initial_step_size=.015,
                        max_num_doublings=2, mass_structure=structure,
                        collapsed=collapsed, block_size=2)
                    runner = mcmc.SweepRunner(target, initial)
                    first, first_samples, first_diag = runner(initial, 2)
                    last, samples, diag = runner(first, 4)
                    self.assertEqual(last.phase, 'sampling')
                    full, all_samples, all_diag = mcmc.run_nuts_warmup(target, initial)
                    self.equal((last.key, last.step_size, last.inverse_mass_matrix, last.adaptation.state),
                               (full.key, full.step_size, full.inverse_mass_matrix, full.adaptation.state),
                               exact=True)
                    self.assertEqual(last.adaptation.completed, full.adaptation.completed)
                    self.equal(jax.tree.map(lambda a, b: jnp.concatenate((a, b)),
                                            first_samples, samples), all_samples, exact=True)
                    self.equal(jax.tree.map(lambda a, b: jnp.concatenate((a, b)),
                                            first_diag, diag), all_diag, exact=True)
                    for i, ((start, stop), window) in enumerate(zip(((0, 2), (2, 3)), first.adaptation.state)):
                        # Each block's BlackJAX engine uses its own selected position/rate.
                        init, update, _ = nuts_window_adapter(.8, structure, (stop-start, 2))
                        expected = init(state.eta[start:stop].reshape(-1), .015)
                        for j in range(2):
                            position = first_samples.eta[j, start:stop].reshape(-1)
                            expected = jax.jit(update)(expected, build_schedule(6)[j],
                                position, jnp.zeros(position.size), first_diag.theta.acceptance_rate[j, i])
                        self.equal(window, expected)
                    retained, _, _ = runner(last, 2)
                    self.equal((retained.step_size, retained.inverse_mass_matrix, retained.adaptation.state),
                               (last.step_size, last.inverse_mass_matrix, last.adaptation.state), exact=True)
                    self.assertEqual(diag.theta.acceptance_rate.shape, (4, 2))
                    mass_shapes = ((4,), (2,)) if structure == 'diagonal' else ((4, 4), (2, 2))
                    for mass, shape in zip(diag.theta.inverse_mass_matrix, mass_shapes):
                        self.assertEqual(mass.shape, (4, *shape))
                    # A compiled runner's static block configuration cannot change mid-run.
                    with self.assertRaises(ValueError):
                        runner(replace(first, block_size=1), 1)

    def test_outer_gibbs_schedule_and_ragged_batch_shapes(self):
        target, state = fixture()
        for collapsed in (False, True):
            chain = mcmc.initialize_nuts_warmup(
                target, state, jax.random.key(785), num_warmup=6,
                initial_step_size=.02, max_num_doublings=3,
                mass_structure='kronecker', block_size=2, collapsed=collapsed)
            # An explicit reference theta hook exercises current outer-Gibbs conditioning.
            def hook(key, conditioned):
                eta, _ = sequential_reference(key, target, conditioned, chain.step_size,
                                               chain.inverse_mass_matrix, collapsed)
                _, info = nuts_sweep(key, target, conditioned, chain.step_size,
                                     chain.inverse_mass_matrix, block_size=2,
                                     collapsed=collapsed, max_num_doublings=3)
                return eta, info
            expected = jax.jit(lambda k, st: mcmc.collapsed_gibbs_sweep(
                k, target, st, None, _theta_transition=hook))(chain.key, state)
            actual = jax.jit(lambda k, st: mcmc.nuts_gibbs_sweep(
                k, target, st, chain.step_size, chain.inverse_mass_matrix,
                block_size=2, collapsed=collapsed, max_num_doublings=3))(chain.key, state)
            self.equal(actual, expected)
            final, samples, info = mcmc.SweepRunner(target, chain)(chain, 2)
            self.assertEqual(final.iteration, 2)
            self.assertEqual(samples.eta.shape, (2, 3, 2))
            self.assertEqual(info.theta.acceptance_rate.shape, (2, 2))
            self.assertEqual(info.theta.inverse_mass_matrix[0].shape, (2, 4, 4))
            self.assertEqual(info.theta.inverse_mass_matrix[1].shape, (2, 2, 2))


if __name__ == '__main__':
    unittest.main()
