"""Failure visibility and validation frequency across all five production paths."""

from contextlib import ExitStack
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import tempfile

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from bayesiancalibration import comparison, mcmc
from bayesiancalibration.gp import LibraryGP
from bayesiancalibration.samplers import metropolis, mmala, nuts
from bayesiancalibration.run import save_checkpoint, load_checkpoint
from bayesiancalibration.adaptation import update_random_walk_adaptation
from jax.experimental import checkify
from tests.test_mcmc import make_fixture


class HotPathTest(unittest.TestCase):
    def setUp(self):
        self.target, self.state, self.proposal = make_fixture(loading_only=True)
        self.config = json.loads((Path(__file__).resolve().parents[1]
                                  / "experiments/comparison.json").read_text())
        self.config.update(num_warmup=2, num_initial=2, max_num_doublings=1, initial_proposal_variance=.001,
                           nuts_initial_step_size=.01, mala_epsilon=.01, mmala_epsilon=.01)

    def test_no_static_validation_in_sweeps_and_one_geometry_audit_per_chunk(self):
        for method in comparison.METHODS:
            with self.subTest(method=method):
                initial = comparison.initialize_chain(self.target, self.state, self.config, method, 0)
                runner = mcmc.ChunkRunner(self.target, initial)
                original = LibraryGP.validate_field_sites
                calls = []

                def audit(gp, eta):
                    calls.append(1)
                    return original(gp, eta)

                with ExitStack() as patches:
                    for name in ("validate_random_walk_chain", "validate_mala_chain",
                                 "validate_mmala_chain", "validate_nuts_chain"):
                        patches.enter_context(patch.object(mcmc, name,
                            side_effect=AssertionError("static validation in sampling")))
                    patches.enter_context(patch.object(LibraryGP, "validate_field_sites", audit))
                    runner.compile(initial)
                    self.assertEqual(len(calls), 0)
                    warmed, _, _ = runner(initial, 2)
                    self.assertEqual(len(calls), 1)
                    final, samples, _ = runner(warmed, 3)
                    self.assertEqual(len(calls), 2)
                    self.assertEqual(final.iteration, 5)
                    self.assertTrue(all(np.all(np.isfinite(x)) for x in samples))

    def test_nonfinite_gibbs_draw_fails_without_recording_for_every_method(self):
        for method in comparison.METHODS:
            with self.subTest(method=method):
                initial = comparison.initialize_chain(self.target, self.state, self.config, method, 0)
                with patch.object(mcmc, "sample_projected_coefficients", return_value=
                                  jnp.full_like(self.state.c_f, jnp.nan)):
                    runner = mcmc.ChunkRunner(self.target, initial)
                    with self.assertRaisesRegex(FloatingPointError, "invalid model state"):
                        runner(initial, 2)
                self.assertEqual(initial.iteration, 0)
                np.testing.assert_array_equal(initial.model_state.c_f, self.state.c_f)

    def test_current_gradient_failure_cannot_hide_as_rejection_or_divergence(self):
        for method, module in (("mala", metropolis.mala), ("nuts", nuts.nuts),
                               ("collapsed_nuts", nuts.nuts)):
            with self.subTest(method=method):
                initial = comparison.initialize_chain(self.target, self.state, self.config, method, 0)
                original = module.init

                def invalid(*args):
                    state = original(*args)
                    return state._replace(logdensity_grad=jnp.full_like(state.logdensity_grad, jnp.nan))

                with patch.object(module, "init", side_effect=invalid):
                    runner = mcmc.ChunkRunner(self.target, initial)
                    with self.assertRaisesRegex(FloatingPointError, "current density/gradient"):
                        runner(initial, 1)

    def test_mmala_invalid_metric_fails_inside_actual_proposal(self):
        initial = comparison.initialize_chain(self.target, self.state, self.config, "mmala", 0)
        with patch.object(mmala, "collapsed_mmala_metric", return_value=-jnp.eye(2)):
            runner = mcmc.ChunkRunner(self.target, initial)
            with self.assertRaisesRegex(FloatingPointError, "metric/diffusion"):
                runner(initial, 1)

    def test_nuts_divergences_remain_algorithm_diagnostics(self):
        for collapsed in (False, True):
            initial = mcmc.initialize_nuts_chain(
                self.target, self.state, jax.random.key(702), step_size=1e6,
                inverse_mass_matrix=jnp.ones(self.state.eta.size),
                max_num_doublings=1, collapsed=collapsed,
            )
            final, samples, info = mcmc.ChunkRunner(self.target, initial)(initial, 2)
            self.assertTrue(np.any(info.theta.is_divergent))
            self.assertEqual(final.iteration, 2)
            self.assertTrue(all(np.all(np.isfinite(x)) for x in samples))

    def test_zero_scatter_holds_then_recovers_with_persisted_site_counts(self):
        for method in ("mh", "mala"):
            config = dict(self.config, num_warmup=4, num_initial=2)
            state = self.state._replace(c_f=jnp.zeros_like(self.state.c_f))
            initial = comparison.initialize_chain(self.target, state, config, method, 0)

            def controlled(key, target, state, proposal, **kwargs):
                # First two accepted proposals return identical positions; then move.
                moved = jnp.full(state.eta.shape[0], state.c_f[0] >= 2)
                eta = state.eta + .01 * moved[:, None]
                new_state = state._replace(eta=eta, c_f=state.c_f + 1)
                info_type = metropolis.RandomWalkSweepInfo if method == "mh" else metropolis.MALASweepInfo
                info = info_type(jnp.ones(2), jnp.ones(2, dtype=bool), jnp.array(0.))
                return new_state, jax.random.split(key)[0], mcmc.GibbsSweepInfo(info, jnp.array(0.), moved)

            with patch.object(mcmc, "collapsed_gibbs_sweep", controlled):
                runner = mcmc.ChunkRunner(self.target, initial)
                held, _, _ = runner(initial, 2)
                np.testing.assert_array_equal(held.V_prop, initial.V_prop)
                np.testing.assert_array_equal(held.adaptation.zero_covariance_count, [1, 1])
                np.testing.assert_array_equal(held.adaptation.acceptance_count, [2, 2])
                np.testing.assert_array_equal(held.adaptation.movement_count, [0, 0])
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "held.npz"
                    save_checkpoint(path, self.target, held)
                    restored, _ = load_checkpoint(path, self.target)
                    final, _, _ = runner(restored, 2)
                self.assertEqual(final.phase, "sampling")
                np.testing.assert_array_equal(final.adaptation.acceptance_count, [4, 4])
                np.testing.assert_array_equal(final.adaptation.movement_count, [2, 2])
                np.testing.assert_array_equal(final.adaptation.zero_covariance_count, [1, 1])
                self.assertTrue(np.all(np.diagonal(final.adaptation.moments.m2, axis1=-2, axis2=-1) > 0))

    def test_nonzero_invalid_covariance_is_not_held(self):
        initial = comparison.initialize_chain(self.target, self.state, self.config, "mh", 0)
        moments = initial.adaptation.moments._replace(
            mean=self.state.eta, sample_size=jnp.full(2, 2),
            m2=-jnp.tile(jnp.eye(2), (2, 1, 1)),
        )
        kernel = jax.jit(checkify.checkify(update_random_walk_adaptation))
        error, _ = kernel(moments, self.state.eta, initial.V_prop, True)
        self.assertIn("non-SPD", error.get())

    def test_failed_warmup_writes_counts_and_never_records_production(self):
        target, state, proposal = make_fixture(bounded=True, loading_only=True)
        initial = mcmc.initialize_random_walk_warmup(
            target, state, jax.random.key(962), num_warmup=3, num_initial=1,
            V_prop=proposal * 1e100,
        )
        config = dict(self.config, num_warmup=3, num_initial=1, chunk_sweeps=1)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with self.assertRaises(mcmc.WarmupTuningError):
                comparison.run_chain(target, initial, output, config, {}, source_paths={})
            failure = json.loads((output / "warmup_failure.json").read_text())
            self.assertEqual(failure["movement_count"], [0, 0])
            self.assertEqual(failure["acceptance_count"], [0, 0])
            self.assertEqual(failure["zero_covariance_sites"], [0, 1])
            checkpoint, metadata = load_checkpoint(output / "checkpoint.npz", target)
            self.assertEqual(checkpoint.phase, "warmup")
            self.assertEqual(checkpoint.iteration, 2)
            self.assertEqual(metadata["configuration"]["progress"]["production_seconds"], 0)
            for file in output.glob("draws-*.npz"):
                with np.load(file, allow_pickle=False) as data:
                    self.assertEqual(str(data["phase"]), "warmup")


if __name__ == "__main__":
    unittest.main()
