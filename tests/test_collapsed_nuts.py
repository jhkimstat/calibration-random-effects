"""Matched NUTS protocol and collapse-specific dependencies."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from bayesiancalibration.mcmc import initialize_nuts_warmup, nuts_gibbs_sweep
from bayesiancalibration.samplers.nuts import nuts_sweep
import test_nuts as reference
import test_mcmc as outer_reference


class CollapsedNUTSTest(reference.NUTSTest):
    """Apply the same kernel/schedule/posterior/batching gates to Stage 12."""

    collapsed = True

    def test_embedded_windows_match_standard_fixed_target_driver(self):
        # Both target choices call the same validated standard adapter. Check
        # resolved defaults/schedule identity rather than repeat Gaussian draws.
        target, state, _ = outer_reference.make_fixture()
        kwargs = dict(num_warmup=1000)
        first = initialize_nuts_warmup(target, state, jax.random.key(1250), **kwargs)
        second = initialize_nuts_warmup(
            target, state, jax.random.key(1250), collapsed=True, **kwargs
        )
        self.assert_tree_equal(first.adaptation.state, second.adaptation.state, True)
        self.assertEqual(first.adaptation.num_warmup, second.adaptation.num_warmup)
        self.assertEqual(first.adaptation.target_accept, second.adaptation.target_accept)
        self.assertEqual(first.max_num_doublings, second.max_num_doublings)
        self.assertEqual(first.divergence_threshold, second.divergence_threshold)
        self.assertEqual(first.step_size, second.step_size)

    def test_cf_independence_and_observation_dependencies(self):
        target, state, _ = outer_reference.make_fixture(True)
        key, mass = jax.random.key(1251), jnp.ones(4)
        kernel = jax.jit(lambda s: nuts_sweep(
            key, target, s, .18, mass, max_num_doublings=4, collapsed=True
        ))
        first = kernel(state)
        self.assert_tree_equal(first, kernel(state._replace(c_f=state.c_f + .4)), True)
        for changed in (
            state._replace(delta=state.delta + .2),
            state._replace(sigma_y2=state.sigma_y2*1.8),
            state._replace(mu_theta=state.mu_theta + .3),
            state._replace(Sigma_theta=state.Sigma_theta*1.4),
            state._replace(sigma_c2=state.sigma_c2*.7),
        ):
            _, info = kernel(changed)
            self.assertNotAlmostEqual(float(info.logdensity), float(first[1].logdensity))

    def test_divergent_repeated_theta_still_refreshes_coefficients(self):
        target, state, _ = outer_reference.make_fixture(True, True)
        updated, _, info = jax.jit(lambda: nuts_gibbs_sweep(
            jax.random.key(1252), target, state, 1e150, jnp.ones(4),
            max_num_doublings=4, collapsed=True,
        ))()
        np.testing.assert_array_equal(updated.eta, state.eta)
        self.assertTrue(bool(info.theta.is_divergent))
        self.assertFalse(np.array_equal(updated.c_f, state.c_f))
        self.assertTrue(np.isfinite(float(info.full_joint_logdensity)))


if __name__ == '__main__':
    unittest.main()
