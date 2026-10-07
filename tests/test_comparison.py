"""Matched starts/streams, reusable kernels, timed commits, and shell dry runs."""

import copy
from dataclasses import replace
import itertools
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from bayesiancalibration import comparison, mcmc
from bayesiancalibration.gp import LengthScaleFit
from bayesiancalibration.run import load_checkpoint, save_checkpoint
from tests.test_mcmc import make_fixture


ROOT = Path(__file__).resolve().parents[1]


class ComparisonTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "experiments/comparison.json").read_text())
        self.config.update(num_warmup=2, num_initial=1, chunk_sweeps=1,
                           production_seconds=2.0, max_num_doublings=1, initial_proposal_variance=.001, mala_initial_proposal_variance=.001,
                           nuts_initial_step_size=.01, mala_epsilon=.01,
                           mmala_epsilon=.01)

    def test_library_fit_method_configuration(self):
        config = copy.deepcopy(self.config)
        comparison.validate_config(config)
        config["library_fit"]["method"] = "cv_nlpd"
        comparison.validate_config(config)
        config["library_fit"]["method"] = "cv_wmse"
        comparison.validate_config(config)
        config["library_fit"]["method"] = "invalid"
        with self.assertRaisesRegex(ValueError, "library_fit.method"):
            comparison.validate_config(config)

    def test_default_mh_variance_preserves_mala_preconditioner(self):
        config = json.loads((ROOT / "experiments/comparison.json").read_text())
        comparison.validate_config(config)
        target, state, _ = make_fixture(loading_only=True)
        mh = comparison.initialize_chain(target, state, config, "mh", 0)
        mala = comparison.initialize_chain(target, state, config, "mala", 0)
        identity = np.tile(np.eye(2), (2, 1, 1))
        np.testing.assert_array_equal(mh.V_prop, 1e-6 * identity)
        np.testing.assert_array_equal(mala.V_prop, identity)

    def test_task_mapping_and_independent_streams(self):
        pairs = [comparison.task_coordinates(i) for i in range(20)]
        self.assertEqual(len(set(pairs)), 20)
        self.assertEqual(pairs[0], ("mh", 0))
        self.assertEqual(pairs[-1], ("collapsed_nuts", 3))
        for bad in (-1, 20, True, 0.5):
            with self.assertRaises(ValueError):
                comparison.task_coordinates(bad)
        target, state, _ = make_fixture(loading_only=True)
        keys = []
        for method in comparison.METHODS:
            for index in range(4):
                chain = comparison.initialize_chain(target, state, self.config, method, index)
                keys.append(tuple(np.asarray(jax.random.key_data(chain.key))))
                for actual, expected in zip(chain.model_state, state):
                    np.testing.assert_array_equal(actual, expected)
        self.assertEqual(len(set(keys)), 20)

    def test_reusable_chunks_match_public_drivers_for_all_methods(self):
        target, state, _ = make_fixture(loading_only=True)
        warmups = (mcmc.run_random_walk_warmup, mcmc.run_mala_warmup,
                   mcmc.run_mmala_warmup, mcmc.run_nuts_warmup, mcmc.run_nuts_warmup)
        productions = (mcmc.run_fixed_random_walk, mcmc.run_fixed_mala,
                       mcmc.run_fixed_mmala, mcmc.run_fixed_nuts, mcmc.run_fixed_nuts)
        for method, warmup, production in zip(comparison.METHODS, warmups, productions):
            with self.subTest(method=method):
                initial = comparison.initialize_chain(target, state, self.config, method, 0)
                runner = mcmc.ChunkRunner(target, initial)
                runner.compile(initial)
                with self.assertRaises(ValueError):
                    runner(initial, 3)
                first, _, _ = runner(initial, 1)
                second, _, _ = runner(first, 1)
                reference, _, _ = warmup(target, initial, 2)
                actual, draws, info = runner(second, 2)
                expected, ref_draws, ref_info = production(target, reference, 2)
                for a, b in zip(jax.tree.leaves((draws, info)),
                                jax.tree.leaves((ref_draws, ref_info))):
                    np.testing.assert_array_equal(a, b)
                np.testing.assert_array_equal(jax.random.key_data(actual.key),
                                              jax.random.key_data(expected.key))
                self.assertEqual(actual.phase, "sampling")
                if method == "mmala":
                    with self.assertRaises(ValueError):
                        runner(replace(actual, epsilon_G=.1), 1)

    def test_committed_budget_restart_and_corrupted_draw_rejection(self):
        target, state, _ = make_fixture(loading_only=True)
        chain = comparison.initialize_chain(target, state, self.config, "mh", 0)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            clock = itertools.count()
            with patch.object(comparison.time, "perf_counter", side_effect=lambda: next(clock)):
                result = comparison.run_chain(target, chain, output, self.config, {}, source_paths={})
            self.assertEqual(result["status"], "budget_complete")
            self.assertEqual(result["production_seconds"], 2.0)
            self.assertEqual(result["iteration"], 4)
            resumed = comparison.run_chain(target, chain, output, self.config, {}, source_paths={})
            self.assertEqual(resumed["iteration"], 4)
            self.assertEqual(resumed["production_seconds"], 2.0)
            saved, metadata = load_checkpoint(output / "checkpoint.npz", target, source_paths={})
            self.assertEqual(saved.iteration, 4)
            chunks = metadata["configuration"]["progress"]["chunks"]
            self.assertEqual([entry["phase"] for entry in chunks],
                             ["warmup", "warmup", "sampling", "sampling"])
            with np.load(output / chunks[-1]["file"], allow_pickle=False) as draws:
                np.testing.assert_array_equal(draws["state.eta"][-1], saved.model_state.eta)
                self.assertEqual(draws["iteration"].tolist(), [4])
            (output / chunks[0]["file"]).write_bytes(b"damaged")
            with self.assertRaisesRegex(ValueError, "damaged"):
                comparison.run_chain(target, chain, output, self.config, {}, source_paths={})

    def test_uncommitted_chunk_replays_after_checkpoint_failure(self):
        target, state, _ = make_fixture(loading_only=True)
        chain = comparison.initialize_chain(target, state, self.config, "mh", 0)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            commits = 0

            def interrupted_save(*args, **kwargs):
                nonlocal commits
                commits += 1
                if commits == 2:
                    raise OSError("simulated interrupted checkpoint")
                save_checkpoint(*args, **kwargs)

            with patch.object(comparison, "save_checkpoint", side_effect=interrupted_save):
                with self.assertRaisesRegex(OSError, "simulated"):
                    comparison.run_chain(target, chain, output, self.config, {},
                                         source_paths={}, check_sweeps=2)
            saved, _ = load_checkpoint(output / "checkpoint.npz", target, source_paths={})
            self.assertEqual(saved.iteration, 1)
            result = comparison.run_chain(target, chain, output, self.config, {},
                                          source_paths={}, check_sweeps=2)
            self.assertEqual(result["status"], "check_complete")
            saved, _ = load_checkpoint(output / "checkpoint.npz", target, source_paths={})
            reference, _, _ = mcmc.run_random_walk_warmup(target, chain, 2)
            for a, b in zip(saved.model_state, reference.model_state):
                np.testing.assert_array_equal(a, b)
            self.assertEqual(result["production_seconds"], 0.0)

    def test_preparation_freezes_inputs_and_ignores_truth(self):
        config = copy.deepcopy(self.config)
        rng = np.random.default_rng(77)
        arrays = dict(theta_s_dagger=np.array([37850., 24060., .071])
                      + rng.normal(size=(8, 3)) * [1000., 1000., .005],
                      F_s=rng.normal(size=(8, 5)), y_tilde=rng.normal(size=10),
                      R=np.eye(10), s=np.array([[0., 0.], [6., 0.]]),
                      metadata=np.asarray('{}'))
        fit = LengthScaleFit(np.ones(3), np.zeros(3), np.ones(5), 0., (),
                             None, 1e-6, 1e-10, 1000, 0., "test")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            states = []
            for i in range(2):
                arrays["theta_f_truth_for_evaluation"] = np.full((2, 3), i * 1e99)
                source, output = root / f"source-{i}.npz", root / f"prepared-{i}.npz"
                np.savez(source, **arrays)
                with patch.object(comparison, "fit_library_length_scales", return_value=fit):
                    comparison.prepare(source, config_path, output)
                target, initial, loaded_config = comparison.load_experiment(output)
                self.assertFalse(target.coordinates.bounded)
                self.assertEqual(loaded_config, config)
                states.append(initial)
            for a, b in zip(jax.tree.leaves(states[0]), jax.tree.leaves(states[1])):
                np.testing.assert_array_equal(a, b)
            with self.assertRaises(FileExistsError):
                comparison.prepare(source, config_path, output)

    def test_shell_syntax_and_print_only_commands(self):
        scripts = (ROOT / "scripts").glob("*.sh")
        for script in scripts:
            subprocess.run(["bash", "-n", str(script)], check=True)
        with tempfile.TemporaryDirectory(prefix="comparison space ") as tmp:
            root = Path(tmp)
            prepared = root / "prepared.npz"
            prepared.touch()
            env = dict(os.environ, PARTITION="example", CPUS="2", MEMORY="8G",
                       CHECK_WALLTIME="01:00:00", PRODUCTION_WALLTIME="12:00:00")
            # A fake sbatch must never be called, even if it is on PATH.
            fake = root / "sbatch"
            fake.write_text('#!/bin/bash\ntouch "' + str(root / 'submitted') + '"\n')
            fake.chmod(0o755)
            env["PATH"] = str(root) + os.pathsep + env["PATH"]
            result = subprocess.run(
                ["bash", str(ROOT / "scripts/print_unity_commands.sh"), str(prepared),
                 str(root / "output")], env=env, check=True, capture_output=True, text=True,
            )
            self.assertIn("--array=0-19%20", result.stdout)
            self.assertIn("--array=0\\,4\\,8\\,12\\,16", result.stdout)
            self.assertFalse((root / "submitted").exists())


if __name__ == "__main__":
    unittest.main()
