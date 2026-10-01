"""Convergence diagnostics detect location, scale and serial-dependence defects."""

import tempfile
import unittest
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
from scipy.stats import norm

from bayesiancalibration.analysis import _rank_normalize, summarize_chains, analyze_method
from tests.test_mcmc import make_fixture


class ChainAnalysisTest(unittest.TestCase):
    def test_tied_blom_scores_and_multivariate_shapes(self):
        values = np.array([[[1.], [1.]], [[2.], [3.]]])
        expected = norm.ppf((np.array([1.5, 1.5, 3., 4.]) - .375) / 4.25)
        np.testing.assert_allclose(_rank_normalize(values).ravel(), expected, atol=1e-15)
        iid = np.random.default_rng(6).normal(size=(4, 1001, 2, 3))
        summary = summarize_chains(iid)
        self.assertEqual(summary["rank_split_folded_rhat"].shape, (2, 3))
        self.assertLess(summary["rank_split_folded_rhat"].max(), 1.02)
        self.assertTrue(np.all(summary["ess_bulk"] > 2000))
        self.assertTrue(np.all(summary["ess_tail"] > 1800))
        np.testing.assert_allclose(summary["mcse_mean"],
                                   summary["sd"] / np.sqrt(summary["ess_mean"]))

    def test_location_and_folded_scale_failures_and_autocorrelation(self):
        rng = np.random.default_rng(91)
        base = rng.normal(size=(4, 2000))
        shifted = base + np.arange(4)[:, None]
        self.assertGreater(float(summarize_chains(shifted)["rank_split_folded_rhat"]), 1.25)
        scaled = base * np.array([.1, 1., 1., 10.])[:, None]
        self.assertGreater(float(summarize_chains(scaled)["rank_split_folded_rhat"]), 1.25)
        ar = base.copy()
        for t in range(1, ar.shape[1]):
            ar[:, t] += .95 * ar[:, t-1]
        self.assertLess(float(summarize_chains(ar)["ess_bulk"]), 700)

    def test_short_constant_and_invalid_inputs(self):
        summary = summarize_chains(np.ones((4, 16)))
        self.assertTrue(summary["constant"])
        self.assertTrue(np.all(summary["constant_by_chain"]))
        for name in ("rank_split_folded_rhat", "ess_bulk", "ess_tail", "mcse_mean"):
            self.assertTrue(np.isnan(summary[name]))
        stuck = summarize_chains(np.broadcast_to(np.arange(4)[:, None], (4, 16)))
        self.assertFalse(stuck["constant"])
        self.assertTrue(np.all(stuck["constant_by_chain"]))
        self.assertFalse(np.isfinite(stuck["rank_split_folded_rhat"]))
        summary = summarize_chains(np.ones((1, 2, 3)))
        self.assertTrue(np.all(np.isnan(summary["rank_split_folded_rhat"])))
        with self.assertRaises(ValueError):
            summarize_chains(np.array([[np.nan]]))

    def test_saved_batches_truth_free_physical_contrasts_and_predictive_mean(self):
        target, state, _ = make_fixture(loading_only=True)
        rng = np.random.default_rng(2)
        with tempfile.TemporaryDirectory() as tmp:
            directories = []
            expected_cf, expected_delta = [], []
            for chain, count in enumerate((10, 12)):
                directory = Path(tmp) / str(chain)
                directory.mkdir()
                directories.append(directory)
                arrays = {f"state.{name}": np.broadcast_to(np.asarray(value), (count,) + value.shape).copy()
                          for name, value in zip(state._fields, state)}
                arrays["state.eta"] += .01 * rng.normal(size=arrays["state.eta"].shape)
                arrays["state.c_f"] += rng.normal(size=arrays["state.c_f"].shape)
                arrays["state.delta"] += rng.normal(size=arrays["state.delta"].shape)
                np.savez_compressed(directory / "draws-000000001.npz", **arrays)
                expected_cf.append(arrays["state.c_f"][:10])
                expected_delta.append(arrays["state.delta"][:10])
            report = analyze_method(target, directories)
            self.assertEqual(report["draws_by_chain"], [10, 12])
            self.assertEqual(report["unused_draws_for_equal_length_diagnostics"], [0, 2])
            self.assertEqual(report["chain_outcomes"], ["incomplete", "incomplete"])
            means = report["summaries"]
            self.assertIn("theta_physical_contrast_to_site_0", means)
            predicted = (np.stack(expected_cf) + np.tile(np.stack(expected_delta), (1, 1, state.eta.shape[0]))) @ np.asarray(target.R).T
            np.testing.assert_allclose(means["projected_observation_mean"]["mean"], predicted.mean(axis=(0, 1)))
            for directory in directories:
                (directory / "draws-000000001.npz").unlink()
            self.assertEqual(analyze_method(target, directories)["status"], "no_retained_draws")


if __name__ == "__main__":
    unittest.main()
