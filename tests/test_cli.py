"""Fresh experiment preparation, option precedence and actual sampler dispatch."""

from dataclasses import replace
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

from bayesiancalibration import cli, experiment, mcmc
from bayesiancalibration.gp import LengthScaleFit


def fresh_fixture(directory: Path):
    """Small two-dimensional observed-only input independent of archived runs."""
    config = experiment.default_config("scientific")
    config["model"].update(
        branch_sizes=[1], physical_center=[0.1, -0.2], V_theta_0=np.eye(2).tolist(),
        nu_theta_0=7., S_theta_0=np.eye(2).tolist(), m_delta_0=[0.], V_delta_0=[[1e-6]],
        alpha_y_0=[3.], beta_y_0=[.2], alpha_c_0=[3.5], beta_c_0=[.3],
    )
    config["library_fit"].update(starts=[[-.8, -.8], [0., 0.]], maxiter=30,
                                log_bounds=[[-2., .5], [-2., .5]])
    arrays = {
        "theta_s_dagger": np.array([[-1., -.7], [.9, -.4], [.1, 1.1], [-.8, .8]]),
        "F_s": np.array([[.2], [.7], [-.3], [.1]]),
        "y_tilde": np.array([-.1, .3]), "R": np.diag([.8, 1.3]),
        "s": np.array([[0., 0.], [6., 0.]]), "metadata": np.asarray('{}'),
    }
    source = directory / "input.npz"
    np.savez(source, **arrays)
    config_path = directory / "science.json"
    config_path.write_text(json.dumps(config))
    return source, config_path, arrays, config


class ConfigurationTest(unittest.TestCase):
    def test_library_fit_methods_and_unknown_science_fields(self):
        config = experiment.default_config("scientific")
        for method in ("profile", "cv_nlpd", "cv_wmse"):
            config["library_fit"]["method"] = method
            experiment.validate_scientific_config(config)
        config["library_fit"]["method"] = "invalid"
        with self.assertRaisesRegex(ValueError, "library_fit.method"):
            experiment.validate_scientific_config(config)
        config["library_fit"]["method"] = "profile"
        config["epsilon"] = .1
        with self.assertRaisesRegex(ValueError, "model and library_fit"):
            experiment.validate_scientific_config(config)

    def test_absent_flags_preserve_file_and_explicit_all_resets_blocking(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nuts.json"
            path.write_text(json.dumps({"method": "nuts", "block_size": 1,
                                        "initial_step_size": .03, "mass_structure": "dense"}))
            parsed = cli.parser().parse_args(["run", "--prepared", "p.npz", "--output", "out",
                                               "--method", "nuts", "--sampler-config", str(path)])
            self.assertNotIn("block_size", vars(parsed))
            self.assertNotIn("initial_step_size", vars(parsed))
            settings = experiment.resolve_sampler_settings("nuts", path)
            self.assertEqual(settings["block_size"], 1)
            self.assertEqual(settings["initial_step_size"], .03)
            self.assertEqual(settings["mass_structure"], "dense")
            parsed = cli.parser().parse_args(["run", "--prepared", "p", "--output", "o",
                                               "--block-size", "all", "--initial-step-size", ".04"])
            settings = experiment.resolve_sampler_settings("nuts", path,
                {"block_size": parsed.block_size, "initial_step_size": parsed.initial_step_size})
            self.assertIsNone(settings["block_size"])
            self.assertEqual(settings["initial_step_size"], .04)

    def test_file_and_cli_equal_settings_reach_equal_actual_tuning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _, _, config = fresh_fixture(root)
            prepared = root / "prepared.npz"
            experiment.prepare_experiment(source, config, prepared)
            target, _, _ = experiment.load_experiment(prepared)
            state = experiment.initial_states(target, 77, 1)[0]
            path = root / "mala.json"
            path.write_text(json.dumps({"method": "mala", "num_initial": 2,
                                        "epsilon": .04, "initial_proposal_variance": .05,
                                        "target_accept": None}))
            file = experiment.resolve_sampler_settings("mala", path)
            overridden = experiment.resolve_sampler_settings("mala", overrides={
                "num_initial": 2, "epsilon": .04, "initial_proposal_variance": .05,
                "target_accept": None})
            self.assertEqual(file, overridden)
            a = experiment.initialize_chain(target, state, "mala", 77, 0, 4, file)
            b = experiment.initialize_chain(target, state, "mala", 77, 0, 4, overridden)
            np.testing.assert_array_equal(a.V_prop, b.V_prop)
            self.assertEqual(a.epsilon, .04)
            self.assertIsNone(a.step_size_adaptation)
            mh = experiment.initialize_chain(target, state, "mh", 77, 0, 100,
                                             experiment.resolve_sampler_settings("mh"))
            mala = experiment.initialize_chain(target, state, "mala", 77, 0, 100,
                                               experiment.resolve_sampler_settings("mala"))
            np.testing.assert_array_equal(mh.V_prop, 1e-6 * np.tile(np.eye(2), (2, 1, 1)))
            np.testing.assert_array_equal(mala.V_prop, np.tile(np.eye(2), (2, 1, 1)))

    def test_invalid_options_fail_before_sampling(self):
        for method, settings in (("mh", {"epsilon": .1}), ("mala", {"epsilon": -1.}),
                                  ("nuts", {"mass_structure": "bad"}),
                                  ("nuts", {"block_size": True}),
                                  ("mmala", {"target_accept": None}),
                                  ("mh", {"typo": 1})):
            with self.subTest(method=method, settings=settings), self.assertRaises(ValueError):
                experiment.resolve_sampler_settings(method, overrides=settings)
        mh = experiment.resolve_sampler_settings("mh")
        with self.assertRaisesRegex(ValueError, "num_initial"):
            experiment.RunOptions(num_warmup=2).validate(mh, 2)
        nuts = experiment.resolve_sampler_settings("nuts", overrides={"block_size": 3})
        with self.assertRaises(ValueError):
            experiment.RunOptions().validate(nuts, 2)
        for options in (experiment.RunOptions(num_samples=None),
                        experiment.RunOptions(num_samples=1, production_seconds=1.),
                        experiment.RunOptions(chains=True), experiment.RunOptions(seed=-1)):
            with self.assertRaises(ValueError):
                options.validate(experiment.resolve_sampler_settings("nuts"), 2)


class FreshExperimentTest(unittest.TestCase):
    def test_truth_is_excluded_and_initial_states_match_across_methods(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _, arrays, config = fresh_fixture(root)
            fit = LengthScaleFit(np.array([.8, 1.2]), np.log([.8, 1.2]), np.array([1.]),
                                 0., (), None, 1e-6, 1e-10, 1000, 0., "test")
            targets, states = [], []
            for i in range(2):
                arrays["theta_f_truth_for_evaluation"] = np.full((2, 2), i * 1e99)
                np.savez(source, **arrays)
                output = root / f"prepared-{i}.npz"
                with patch.object(experiment, "fit_library_length_scales", return_value=fit):
                    experiment.prepare_experiment(source, config, output)
                target, observed, details = experiment.load_experiment(output)
                self.assertNotIn("theta_f_truth_for_evaluation", observed)
                self.assertNotIn("implementation_hashes", details)
                targets.append(target)
                states.append(experiment.initial_states(target, 77, 2))
            for a, b in zip(jax.tree.leaves(states[0]), jax.tree.leaves(states[1])):
                np.testing.assert_array_equal(a, b)
            keys = []
            for method in experiment.METHODS:
                settings = experiment.resolve_sampler_settings(method)
                for index, state in enumerate(states[0]):
                    chain = experiment.initialize_chain(targets[0], state, method, 77, index, 100, settings)
                    for actual, expected in zip(chain.model_state, state):
                        np.testing.assert_array_equal(actual, expected)
                    keys.append(tuple(np.asarray(jax.random.key_data(chain.key))))
            self.assertEqual(len(set(keys)), 10)
            # Changing requested chain count cannot change an existing chain's start.
            larger = experiment.initial_states(targets[0], 77, 3)
            for a, b in zip(jax.tree.leaves(larger[:2]), jax.tree.leaves(states[0])):
                np.testing.assert_array_equal(a, b)

    def test_actual_cli_fit_warmup_and_retained_draws_all_methods(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, science, _, _ = fresh_fixture(root)
            prepared = root / "fixed.npz"
            cli.main(["prepare", "--input", str(source), "--scientific-config", str(science),
                      "--output", str(prepared)])
            config_dir = root / "samplers"
            config_dir.mkdir()
            for method in experiment.METHODS:
                settings = experiment.resolve_sampler_settings(method)
                if method in ("mh", "mala"):
                    settings.update(num_initial=1, initial_proposal_variance=.001)
                if method in ("mala", "mmala"):
                    settings["epsilon"] = .01
                if method in ("nuts", "collapsed_nuts"):
                    settings.update(initial_step_size=.01, max_num_doublings=1)
                (config_dir / (method + ".json")).write_text(json.dumps(settings))
            output = root / "results"
            cli.main(["run", "--prepared", str(prepared), "--output", str(output), "--method", "all",
                      "--chains", "1", "--seed", "77", "--num-warmup", "4", "--num-samples", "2",
                      "--batch-size", "2", "--sampler-config", str(config_dir)])
            resolved = json.loads((output / "experiment.json").read_text())
            self.assertEqual(resolved["run"]["num_samples"], 2)
            for method in experiment.METHODS:
                directory = output / method / "chain-0"
                self.assertFalse((directory / "checkpoint.npz").exists())
                with np.load(next(directory.glob("draws-*.npz")), allow_pickle=False) as draws:
                    self.assertEqual(draws["state.eta"].shape, (2, 2, 2))
                    self.assertEqual(draws["sweep"].tolist(), [5, 6])
                    self.assertTrue(np.all(np.isfinite(draws["full_joint_logdensity"])))
                    self.assertFalse(any("key" in name or "adaptation" in name for name in draws))
                analysis = json.loads((output / method / "analysis.json").read_text())
                self.assertEqual(analysis["status"], "too_short_for_chain_diagnostics")
            with self.assertRaises(FileExistsError):
                cli.main(["run", "--prepared", str(prepared), "--output", str(output), "--method", "nuts"])

    def test_explicit_cli_overrides_are_recorded_and_used_in_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _, _, config = fresh_fixture(root)
            prepared = root / "fixed.npz"
            experiment.prepare_experiment(source, config, prepared)
            sampler = root / "mala.json"
            sampler.write_text(json.dumps({"method": "mala", "num_initial": 1,
                                          "epsilon": .03, "initial_proposal_variance": .001,
                                          "target_accept": .574}))
            output = root / "overridden"
            cli.main(["run", "--prepared", str(prepared), "--output", str(output), "--method", "mala",
                      "--chains", "1", "--seed", "77", "--num-warmup", "4", "--num-samples", "1",
                      "--sampler-config", str(sampler), "--epsilon", ".02", "--target-accept", "none"])
            resolved = json.loads((output / "experiment.json").read_text())
            self.assertEqual(resolved["explicit_sampler_overrides"], {"epsilon": .02, "target_accept": None})
            self.assertEqual(resolved["methods"]["mala"]["epsilon"], .02)
            self.assertIsNone(resolved["methods"]["mala"]["target_accept"])
            result = json.loads((output / "mala" / "chain-0" / "result.json").read_text())
            self.assertEqual(result["final_tuning"]["epsilon"], .02)

    def test_failed_batch_not_recorded_as_retained_draw(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _, _, config = fresh_fixture(root)
            prepared = root / "fixed.npz"
            experiment.prepare_experiment(source, config, prepared)
            target, _, _ = experiment.load_experiment(prepared)
            settings = experiment.resolve_sampler_settings("mmala")
            state = experiment.initial_states(target, 77, 1)[0]
            chain = experiment.initialize_chain(target, state, "mmala", 77, 0, 2, settings)
            options = experiment.RunOptions(seed=77, chains=1, num_warmup=2, num_samples=2)
            output = root / "failed"
            with patch.object(mcmc.SweepRunner, "compile"), patch.object(
                mcmc.SweepRunner, "__call__", side_effect=FloatingPointError("injected delta failure")
            ), self.assertRaises(FloatingPointError):
                experiment.run_chain(target, chain, output, options, settings, 0)
            self.assertFalse(list(output.glob("draws-*.npz")))
            evidence = json.loads((output / "failure.json").read_text())
            self.assertEqual(evidence["phase"], "warmup")
            self.assertEqual(evidence["last_valid_sweep"], 0)

    def test_initial_coefficient_failure_records_phase_chain_and_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _, _, config = fresh_fixture(root)
            prepared = root / "fixed.npz"
            experiment.prepare_experiment(source, config, prepared)
            output = root / "failed-start"
            with patch.object(experiment, "refresh_field_coefficients",
                              side_effect=ValueError("injected initial GP geometry failure")), self.assertRaises(ValueError):
                cli.main(["run", "--prepared", str(prepared), "--output", str(output),
                          "--method", "mmala", "--seed", "77", "--chains", "1",
                          "--num-warmup", "2", "--num-samples", "1"])
            failure = json.loads((output / "failure.json").read_text())
            self.assertEqual(failure["phase"], "initialization")
            self.assertEqual(failure["details"]["chain"], 0)
            self.assertEqual(failure["details"]["seed"], 77)
            self.assertEqual(failure["details"]["update"], "c_f")
            self.assertFalse(list(output.rglob("draws-*.npz")))

    def test_reused_sweeps_match_independent_public_drivers(self):
        """Preserve numerical assertions formerly mixed with job/restart tests."""
        warmups = (mcmc.run_random_walk_warmup, mcmc.run_mala_warmup,
                   mcmc.run_mmala_warmup, mcmc.run_nuts_warmup, mcmc.run_nuts_warmup)
        sampling = (mcmc.run_fixed_random_walk, mcmc.run_fixed_mala,
                    mcmc.run_fixed_mmala, mcmc.run_fixed_nuts, mcmc.run_fixed_nuts)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _, _, config = fresh_fixture(root)
            prepared = root / "fixed.npz"
            experiment.prepare_experiment(source, config, prepared)
            target, _, _ = experiment.load_experiment(prepared)
            state = experiment.initial_states(target, 77, 1)[0]
            for method, warmup, production in zip(experiment.METHODS, warmups, sampling):
                with self.subTest(method=method):
                    settings = experiment.resolve_sampler_settings(method)
                    if method in ("mh", "mala"):
                        settings.update(num_initial=1, initial_proposal_variance=.001)
                    if method in ("mala", "mmala"):
                        settings["epsilon"] = .01
                    if method in ("nuts", "collapsed_nuts"):
                        settings.update(initial_step_size=.01, max_num_doublings=1)
                    initial = experiment.initialize_chain(target, state, method, 77, 0, 4, settings)
                    runner = mcmc.SweepRunner(target, initial)
                    runner.compile(initial)
                    with self.assertRaises(ValueError):
                        runner(initial, 5)
                    first, _, _ = runner(initial, 2)
                    second, _, _ = runner(first, 2)
                    reference, _, _ = warmup(target, initial, 4)
                    actual, draws, info = runner(second, 2)
                    expected, ref_draws, ref_info = production(target, reference, 2)
                    for a, b in zip(jax.tree.leaves((draws, info)), jax.tree.leaves((ref_draws, ref_info))):
                        np.testing.assert_array_equal(a, b)
                    np.testing.assert_array_equal(jax.random.key_data(actual.key), jax.random.key_data(expected.key))
                    self.assertEqual(actual.phase, "sampling")
                    if method == "mmala":
                        with self.assertRaises(ValueError):
                            runner(replace(actual, epsilon_G=.1), 1)

    def test_local_production_time_budget_has_no_restart_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _, _, config = fresh_fixture(root)
            prepared = root / "fixed.npz"
            experiment.prepare_experiment(source, config, prepared)
            target, _, _ = experiment.load_experiment(prepared)
            state = experiment.initial_states(target, 77, 1)[0]
            settings = experiment.resolve_sampler_settings("mmala", overrides={"epsilon": .01})
            chain = experiment.initialize_chain(target, state, "mmala", 77, 0, 2, settings)
            options = experiment.RunOptions(seed=77, chains=1, num_warmup=2, num_samples=None,
                                            production_seconds=2., batch_size=1)
            clock = itertools.count()
            with patch.object(experiment.time, "perf_counter", side_effect=lambda: next(clock)):
                result = experiment.run_chain(target, chain, root / "run", options, settings, 0)
            self.assertEqual(result["retained_draws"], 2)
            self.assertEqual(result["production_seconds"], 2.)
            self.assertEqual(len(list((root / "run").glob("draws-*.npz"))), 2)
            self.assertEqual(len(list((root / "run").glob("warmup-*.npz"))), 2)
            self.assertFalse((root / "run" / "checkpoint.npz").exists())


if __name__ == "__main__":
    unittest.main()
