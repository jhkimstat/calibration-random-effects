"""Fresh-run failure visibility and audit frequency across all five samplers."""

from contextlib import ExitStack
from dataclasses import replace
import json
import unittest
from unittest.mock import patch

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax.experimental import checkify

from bayesiancalibration import mcmc
from bayesiancalibration.adaptation import update_random_walk_adaptation
from bayesiancalibration.gp import LibraryGP
from bayesiancalibration.samplers import metropolis, mmala, nuts
from bayesiancalibration.validation import SamplingError, check_quantity
from tests.test_mcmc import make_fixture


METHODS = ("mh", "mala", "mmala", "nuts", "collapsed_nuts")


def initial_chain(target, state, method, *, warmup=True, num_warmup=2, num_initial=2):
    """Small local scientific fixture; no experiment or restart framework."""
    key = jax.random.key(702)
    proposal = .001 * jnp.tile(jnp.eye(state.eta.shape[1]), (state.eta.shape[0], 1, 1))
    if method == "mh":
        return (mcmc.initialize_random_walk_warmup(
            target, state, key, num_warmup=num_warmup, num_initial=num_initial, V_prop=proposal)
                if warmup else mcmc.initialize_random_walk_chain(target, state, key, proposal))
    if method == "mala":
        return (mcmc.initialize_mala_warmup(
            target, state, key, num_warmup=num_warmup, num_initial=num_initial,
            V_prop=proposal, epsilon=.01)
                if warmup else mcmc.initialize_mala_chain(target, state, key, proposal, .01))
    if method == "mmala":
        return (mcmc.initialize_mmala_warmup(
            target, state, key, num_warmup=num_warmup, epsilon=.01, epsilon_G=.001)
                if warmup else mcmc.initialize_mmala_chain(target, state, key, epsilon=.01, epsilon_G=.001))
    collapsed = method == "collapsed_nuts"
    return (mcmc.initialize_nuts_warmup(
        target, state, key, num_warmup=num_warmup, initial_step_size=.01,
        max_num_doublings=1, collapsed=collapsed)
            if warmup else mcmc.initialize_nuts_chain(
                target, state, key, step_size=.01, inverse_mass_matrix=jnp.ones(state.eta.size),
                max_num_doublings=1, collapsed=collapsed))


class HotPathTest(unittest.TestCase):
    def setUp(self):
        self.target, self.state, self.proposal = make_fixture(loading_only=True)

    def test_no_static_validation_in_sweeps_and_one_geometry_audit_per_batch(self):
        for method in METHODS:
            with self.subTest(method=method):
                initial = initial_chain(self.target, self.state, method)
                runner = mcmc.SweepRunner(self.target, initial)
                original = LibraryGP.validate_field_sites
                calls = []

                def audit(gp, eta):
                    calls.append(1)
                    return original(gp, eta)

                with ExitStack() as patches:
                    for name in ("validate_random_walk_chain", "validate_mala_chain",
                                 "validate_mmala_chain", "validate_nuts_chain", "validate_model_state"):
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

    def test_model_validation_does_not_construct_surrogate_random_walk_chains(self):
        with patch.object(mcmc, "RandomWalkChain", side_effect=AssertionError("surrogate chain")):
            for method in ("mala", "mmala", "nuts", "collapsed_nuts"):
                with self.subTest(method=method):
                    initial_chain(self.target, self.state, method)

    def test_every_gibbs_update_reports_first_invalid_quantity_and_host_sweep(self):
        functions = {
            "delta": "sample_discrepancy", "sigma_y2": "sample_branch_noise",
            "mu_theta": "sample_spatial_mean", "Sigma_theta": "sample_spatial_covariance",
            "sigma_c2": "sample_coefficient_variances", "c_f": "sample_projected_coefficients",
        }
        for name, function in functions.items():
            with self.subTest(update=name):
                initial = replace(initial_chain(self.target, self.state, "mh", warmup=False), iteration=3)
                with patch.object(mcmc, function, return_value=jnp.full_like(getattr(self.state, name), jnp.nan)):
                    runner = mcmc.SweepRunner(self.target, initial)
                    with self.assertRaises(SamplingError) as failed:
                        runner(initial, 2, chain_index=7)
                details = failed.exception.diagnostics
                self.assertEqual((details["method"], details["chain"], details["phase"], details["sweep"]),
                                 ("mh", 7, "sampling", 4))
                self.assertEqual((details["update"], details["quantity"], details["predicate"]),
                                 (name, name, "finite"))
                self.assertEqual(details["batch"], [4, 5])
                json.dumps(details, allow_nan=False)
                self.assertEqual(initial.iteration, 3)
        initial = initial_chain(self.target, self.state, "mh", warmup=False)
        invalid_info = metropolis.RandomWalkSweepInfo(jnp.ones(2), jnp.ones(2, dtype=bool), jnp.array(0.))
        with patch.object(mcmc, "collapsed_random_walk_sweep", return_value=(jnp.full_like(self.state.eta, jnp.nan), invalid_info)):
            with self.assertRaises(SamplingError) as failed:
                mcmc.SweepRunner(self.target, initial)(initial, 1)
        self.assertEqual(failed.exception.diagnostics["update"], "eta")

    def test_sampled_variances_report_nonpositive_draw(self):
        for name, function in (("sigma_y2", "sample_branch_noise"), ("sigma_c2", "sample_coefficient_variances")):
            with self.subTest(quantity=name):
                initial = initial_chain(self.target, self.state, "mh", warmup=False)
                with patch.object(mcmc, function, return_value=jnp.zeros_like(getattr(self.state, name))):
                    with self.assertRaises(SamplingError) as failed:
                        mcmc.SweepRunner(self.target, initial)(initial, 1)
                self.assertEqual(failed.exception.diagnostics["quantity"], name)
                self.assertEqual(failed.exception.diagnostics["predicate"], "positive")

    def test_nonfinite_coefficient_draw_fails_for_every_method(self):
        for method in METHODS:
            with self.subTest(method=method):
                initial = initial_chain(self.target, self.state, method)
                with patch.object(mcmc, "sample_projected_coefficients", return_value=jnp.full_like(self.state.c_f, jnp.nan)):
                    with self.assertRaises(SamplingError) as failed:
                        mcmc.SweepRunner(self.target, initial)(initial, 2)
                self.assertEqual(failed.exception.diagnostics["quantity"], "c_f")
                self.assertEqual(initial.iteration, 0)
                np.testing.assert_array_equal(initial.model_state.c_f, self.state.c_f)

    def test_current_gradient_failure_has_site_or_block_context(self):
        for method, module in (("mala", metropolis.mala), ("nuts", nuts.nuts), ("collapsed_nuts", nuts.nuts)):
            with self.subTest(method=method):
                initial = initial_chain(self.target, self.state, method)
                original = module.init

                def invalid(*args):
                    state = original(*args)
                    return state._replace(logdensity_grad=jnp.full_like(state.logdensity_grad, jnp.nan))

                with patch.object(module, "init", side_effect=invalid):
                    with self.assertRaises(SamplingError) as failed:
                        mcmc.SweepRunner(self.target, initial)(initial, 1)
                self.assertEqual(failed.exception.diagnostics["role"], "current")
                index = "site" if method == "mala" else "block"
                self.assertEqual(failed.exception.diagnostics[index], 0)

    def test_mmala_invalid_metric_identifies_proposal_site(self):
        initial = initial_chain(self.target, self.state, "mmala")
        with patch.object(mmala, "collapsed_mmala_metric", return_value=-jnp.eye(2)):
            with self.assertRaises(SamplingError) as failed:
                mcmc.SweepRunner(self.target, initial)(initial, 1)
        self.assertEqual(failed.exception.diagnostics["quantity"], "gradient_metric_diffusion")
        self.assertEqual(failed.exception.diagnostics["site"], 0)

    def test_nuts_divergences_remain_algorithm_diagnostics(self):
        for collapsed in (False, True):
            initial = mcmc.initialize_nuts_chain(
                self.target, self.state, jax.random.key(702), step_size=1e6,
                inverse_mass_matrix=jnp.ones(self.state.eta.size), max_num_doublings=1, collapsed=collapsed)
            final, samples, info = mcmc.SweepRunner(self.target, initial)(initial, 2)
            self.assertTrue(np.any(info.theta.is_divergent))
            self.assertEqual(final.iteration, 2)
            self.assertTrue(all(np.all(np.isfinite(x)) for x in samples))

    def test_zero_scatter_holds_then_recovers_with_site_counts(self):
        for method in ("mh", "mala"):
            state = self.state._replace(c_f=jnp.zeros_like(self.state.c_f))
            initial = initial_chain(self.target, state, method, num_warmup=4, num_initial=2)

            def controlled(key, target, state, proposal, **kwargs):
                moved = jnp.full(state.eta.shape[0], state.c_f[0] >= 2)
                eta = state.eta + .01 * moved[:, None]
                new_state = state._replace(eta=eta, c_f=state.c_f + 1)
                info_type = metropolis.RandomWalkSweepInfo if method == "mh" else metropolis.MALASweepInfo
                info = info_type(jnp.ones(2), jnp.ones(2, dtype=bool), jnp.array(0.))
                return new_state, jax.random.split(key)[0], mcmc.GibbsSweepInfo(info, jnp.array(0.), moved)

            with patch.object(mcmc, "collapsed_gibbs_sweep", controlled):
                runner = mcmc.SweepRunner(self.target, initial)
                held, _, _ = runner(initial, 2)
                np.testing.assert_array_equal(held.V_prop, initial.V_prop)
                np.testing.assert_array_equal(held.adaptation.zero_covariance_count, [1, 1])
                np.testing.assert_array_equal(held.adaptation.acceptance_count, [2, 2])
                np.testing.assert_array_equal(held.adaptation.movement_count, [0, 0])
                final, _, _ = runner(held, 2)
            self.assertEqual(final.phase, "sampling")
            np.testing.assert_array_equal(final.adaptation.acceptance_count, [4, 4])
            np.testing.assert_array_equal(final.adaptation.movement_count, [2, 2])
            np.testing.assert_array_equal(final.adaptation.zero_covariance_count, [1, 1])

    def test_nonzero_invalid_covariance_is_not_held(self):
        initial = initial_chain(self.target, self.state, "mh")
        moments = initial.adaptation.moments._replace(mean=self.state.eta, sample_size=jnp.full(2, 2),
                    m2=-jnp.tile(jnp.eye(2), (2, 1, 1)))
        kernel = jax.jit(checkify.checkify(update_random_walk_adaptation))
        error, _ = kernel(moments, self.state.eta, initial.V_prop, True)
        self.assertIn("non-SPD", error.get())
        self.assertIn("site=0", error.get())

    def test_adaptation_error_has_host_and_leaf_context(self):
        initial = initial_chain(self.target, self.state, "mh")
        original = mcmc.update_random_walk_adaptation

        def invalid(*args, **kwargs):
            moments, proposal = original(*args, **kwargs)
            return moments, jnp.full_like(proposal, jnp.nan)

        with patch.object(mcmc, "update_random_walk_adaptation", invalid):
            with self.assertRaises(SamplingError) as failed:
                mcmc.SweepRunner(self.target, initial)(initial, 1, chain_index=4)
        details = failed.exception.diagnostics
        self.assertEqual((details["phase"], details["sweep"], details["update"]), ("warmup", 1, "adaptation"))
        self.assertEqual(details["quantity"], "V_prop")
        self.assertEqual(details["failed_value"]["finite_count"], 0)
        self.assertEqual(details["attempted_tuning"]["V_prop"]["finite_count"], 0)
        self.assertEqual(details["previous_tuning"]["V_prop"]["finite_count"], initial.V_prop.size)
        json.dumps(details, allow_nan=False)

    def test_nuts_mass_failure_reports_attempted_values_for_explicit_all_site_block(self):
        initial = initial_chain(self.target, self.state, "nuts")
        # One explicit block uses the all-site adapter representation.
        initial = replace(initial, block_size=self.state.eta.shape[0])
        original = mcmc.nuts_window_adapter
        for invalid_value in (-1., jnp.nan):
            with self.subTest(invalid_value=float(invalid_value)):
                def invalid_adapter(*args, **kwargs):
                    init, update, final = original(*args, **kwargs)

                    def invalid_update(*args):
                        state = update(*args)
                        return state._replace(inverse_mass_matrix=jnp.full_like(state.inverse_mass_matrix, invalid_value))

                    return init, invalid_update, final

                with patch.object(mcmc, "nuts_window_adapter", invalid_adapter):
                    with self.assertRaises(SamplingError) as failed:
                        mcmc.SweepRunner(self.target, initial)(initial, 1)
                details = failed.exception.diagnostics
                self.assertEqual(details["update"], "adaptation")
                self.assertEqual(details["previous_tuning"]["inverse_mass_matrix"]["minimum"], 1.)
                self.assertIn(details["quantity"], details["attempted_tuning"])
                self.assertIn("failed_value", details)
                if np.isfinite(invalid_value):
                    self.assertEqual(details["failed_value"]["minimum"], -1.)
                else:
                    self.assertEqual(details["failed_value"]["finite_count"], 0)
                json.dumps(details, allow_nan=False)

    def test_failed_warmup_reports_counts_and_freeze_context(self):
        target, state, proposal = make_fixture(bounded=True, loading_only=True)
        initial = mcmc.initialize_random_walk_warmup(target, state, jax.random.key(962),
                     num_warmup=3, num_initial=1, V_prop=proposal * 1e100)
        with self.assertRaises(mcmc.WarmupTuningError) as failed:
            mcmc.SweepRunner(target, initial)(initial, 3, chain_index=2)
        details = failed.exception.diagnostics
        self.assertEqual(details["movement_count"], [0, 0])
        self.assertEqual(details["acceptance_count"], [0, 0])
        self.assertEqual(details["zero_covariance_sites"], [0, 1])
        self.assertEqual((details["method"], details["chain"], details["phase"], details["sweep"]),
                         ("mh", 2, "warmup", 3))

    def test_diagnostic_audits_each_state_without_changing_numerical_results(self):
        for method in METHODS:
            with self.subTest(method=method):
                initial = initial_chain(self.target, self.state, method, warmup=False)
                ordinary = mcmc.SweepRunner(self.target, initial)(initial, 2)
                original = LibraryGP.validate_field_sites
                calls = []

                def audit(gp, eta):
                    calls.append(np.asarray(eta))
                    return original(gp, eta)

                with patch.object(LibraryGP, "validate_field_sites", audit):
                    detailed = mcmc.SweepRunner(self.target, initial, diagnostic=True)(initial, 2)
                self.assertEqual(len(calls), 2)
                for a, b in zip(jax.tree.leaves(ordinary[1:]), jax.tree.leaves(detailed[1:])):
                    np.testing.assert_array_equal(a, b)
                np.testing.assert_array_equal(jax.random.key_data(ordinary[0].key), jax.random.key_data(detailed[0].key))

    def test_geometry_failure_identifies_audited_endpoint_and_batch(self):
        initial = initial_chain(self.target, self.state, "mh", warmup=False)
        runner = mcmc.SweepRunner(self.target, initial)
        with patch.object(LibraryGP, "validate_field_sites", side_effect=ValueError("singular unjittered field GP")):
            with self.assertRaises(SamplingError) as failed:
                runner(initial, 3, chain_index=1)
        details = failed.exception.diagnostics
        self.assertEqual((details["update"], details["quantity"], details["sweep"], details["batch"]),
                         ("geometry", "gp_field_covariance", 3, [1, 3]))
        self.assertIn("audited_endpoint_model", details)
        self.assertNotIn("sweep_start_model", details)
        self.assertEqual(initial.iteration, 0)

    def test_failed_coefficient_diagnostics_use_new_theta_and_variances(self):
        initial = initial_chain(self.target, self.state, "mh", warmup=False)
        eta = self.state.eta + .02
        variances = self.state.sigma_c2 * 3
        info = metropolis.RandomWalkSweepInfo(jnp.ones(2), jnp.ones(2, dtype=bool), jnp.array(0.))
        with ExitStack() as patches:
            patches.enter_context(patch.object(mcmc, "sample_coefficient_variances", return_value=variances))
            patches.enter_context(patch.object(mcmc, "collapsed_random_walk_sweep", return_value=(eta, info)))
            patches.enter_context(patch.object(mcmc, "sample_projected_coefficients", return_value=jnp.full_like(self.state.c_f, jnp.nan)))
            with self.assertRaises(SamplingError) as failed:
                mcmc.SweepRunner(self.target, initial, diagnostic=True)(initial, 1)
        details = failed.exception.diagnostics
        np.testing.assert_array_equal(details["failed_update_inputs"]["eta"]["values"], eta)
        np.testing.assert_array_equal(details["failed_update_inputs"]["sigma_c2"]["values"], variances)
        evidence = details["conditional_evidence"]
        self.assertIn("cholesky_diagonal", evidence["Sigma_f_given_s"])
        self.assertIn("cholesky_diagonal", evidence["V_y"])
        self.assertEqual(evidence["unjittered_geometry"], "valid")
        json.dumps(details, allow_nan=False)

    def test_theta_failure_labels_sweep_start_without_claiming_site_geometry(self):
        initial = initial_chain(self.target, self.state, "mala", warmup=False)
        attempted_eta = self.state.eta.at[0, 0].add(.2)

        def failed_later_site(key, target, state, proposal, **kwargs):
            check_quantity(jnp.array(False), update="eta", quantity="density_gradient",
                           role="current", site=1)
            info = metropolis.MALASweepInfo(jnp.ones(2), jnp.ones(2, dtype=bool), jnp.array(0.))
            return (state._replace(eta=attempted_eta), jax.random.split(key)[0],
                    mcmc.GibbsSweepInfo(info, jnp.array(0.), jnp.array([True, False])))

        # Inject a later-site failure after a nominal earlier-site move. The
        # complete attempted sweep is distinct from its intermediate current
        # position, which is unavailable to the host error reporter.
        with patch.object(mcmc, "collapsed_gibbs_sweep", failed_later_site):
            with self.assertRaises(SamplingError) as failed:
                mcmc.SweepRunner(self.target, initial, diagnostic=True)(initial, 1)
        details = failed.exception.diagnostics
        self.assertNotIn("eta", details["failed_update_inputs"])
        self.assertNotIn("conditional_evidence", details)
        evidence = details["theta_position_evidence"]
        np.testing.assert_array_equal(evidence["sweep_start_eta"]["values"], self.state.eta)
        np.testing.assert_array_equal(evidence["attempted_final_eta"]["values"], attempted_eta)
        self.assertEqual(details["site"], 1)
        self.assertFalse(evidence["failed_site_or_block_position_available"])
        self.assertIn("sweep_start_model", details)


if __name__ == "__main__":
    unittest.main()
