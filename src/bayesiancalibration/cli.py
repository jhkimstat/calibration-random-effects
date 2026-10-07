"""Thin command line for preparation and fresh local research experiments."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import jax

from bayesiancalibration import experiment


def _block_size(value: str) -> int | None:
    if value.lower() in ("all", "none", "null"):
        return None
    try:
        return int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use a positive integer or 'all'") from error


def _target_accept(value: str) -> float | None:
    if value.lower() in ("none", "null", "fixed"):
        return None
    try:
        return float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use a probability or 'none' for fixed MALA epsilon") from error


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    preprocessing = commands.add_parser("preprocess", help="Prepare supplied loading-curve CSV/XLSX inputs")
    preprocessing.add_argument("--input-dir", type=Path, required=True)
    preprocessing.add_argument("--output-dir", type=Path, required=True)
    preprocessing.add_argument("--seed", type=int, default=1024)
    preprocessing.add_argument("--candidate-step", type=float, default=0.8)
    prep = commands.add_parser("prepare", help="Fit and freeze fixed numerical inputs for all methods")
    prep.add_argument("--input", type=Path, required=True, help="Preprocessing NPZ with theta_s_dagger,F_s,y_tilde,R,s")
    prep.add_argument("--scientific-config", type=Path, help="JSON model and library_fit settings; defaults to packaged recipe")
    prep.add_argument("--kernel", choices=experiment.COEFFICIENT_KERNELS, default=argparse.SUPPRESS,
                      help="Coefficient GP kernel override; recipe default matern32; spatial GP remains SE")
    prep.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run", help="Initialize, warm up, sample, and summarize a fresh experiment")
    run.add_argument("--prepared", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True, help="A new output directory")
    run.add_argument("--method", nargs="+", choices=(*experiment.METHODS, "all"), default=["mh"])
    run.add_argument("--seed", type=int, default=20260928)
    run.add_argument("--chains", type=int, default=4)
    run.add_argument("--num-warmup", type=int, default=1000)
    length = run.add_mutually_exclusive_group()
    length.add_argument("--num-samples", type=int, help="Retained sweeps per chain; default 1000")
    length.add_argument("--production-seconds", type=float, help="Synchronized production-driver time per chain")
    run.add_argument("--batch-size", type=int, default=10, help="Numerical batch and endpoint geometry-audit spacing")
    run.add_argument("--diagnostic", action="store_true", help="Audit GP geometry each sweep and collect conditional evidence on failure")
    run.add_argument("--sampler-config", type=Path, help="Sampler JSON for one method, or directory of METHOD.json files")
    # SUPPRESS is essential: absent CLI flags must leave file values intact.
    for flag, value_type in (
        ("num-initial", int), ("initial-proposal-variance", float),
        ("epsilon", float), ("epsilon-G", float), ("target-accept", _target_accept),
        ("initial-step-size", float), ("mass-structure", str), ("block-size", _block_size),
        ("max-num-doublings", int), ("divergence-threshold", float),
    ):
        run.add_argument("--" + flag, type=value_type, default=argparse.SUPPRESS,
                         help="Explicit sampler override (must apply to every selected method)")
    return result


def main(argv: list[str] | None = None) -> None:
    arguments = parser().parse_args(argv)
    jax.config.update("jax_enable_x64", True)
    if arguments.command == "preprocess":
        from bayesiancalibration.preprocessing import prepare_synthetic_data, save_prepared
        prepared = prepare_synthetic_data(arguments.input_dir, seed=arguments.seed,
                                          candidate_step=arguments.candidate_step)
        save_prepared(prepared, arguments.output_dir)
        print(json.dumps({"preprocessing": str(arguments.output_dir / "synthetic_preprocessing.npz")}))
        return
    if arguments.command == "prepare":
        scientific = (json.loads(arguments.scientific_config.read_text())
                      if arguments.scientific_config else experiment.default_config("scientific"))
        if hasattr(arguments, "kernel"):
            scientific["library_fit"]["kernel"] = arguments.kernel
        experiment.prepare_experiment(arguments.input, scientific, arguments.output)
        print(json.dumps({"prepared": str(arguments.output)}))
        return
    methods = list(experiment.METHODS) if "all" in arguments.method else arguments.method
    if len(set(methods)) != len(methods) or ("all" in arguments.method and len(arguments.method) > 1):
        raise ValueError("Specify each method once, or use 'all' alone")
    config = arguments.sampler_config
    if config and not config.is_dir() and len(methods) != 1:
        raise ValueError("A single sampler file requires one method; use a directory for multiple methods")
    fields = set().union(*experiment.SAMPLER_FIELDS.values())
    overrides = {name: value for name, value in vars(arguments).items() if name in fields}
    settings = {method: experiment.resolve_sampler_settings(
        method, config / (method + ".json") if config and config.is_dir() else config, overrides,
    ) for method in methods}
    options = experiment.RunOptions(
        seed=arguments.seed, chains=arguments.chains, num_warmup=arguments.num_warmup,
        num_samples=arguments.num_samples if arguments.num_samples is not None else
        (None if arguments.production_seconds is not None else 1000),
        production_seconds=arguments.production_seconds, batch_size=arguments.batch_size,
        diagnostic=arguments.diagnostic,
    )
    target, arrays, preparation = experiment.load_experiment(arguments.prepared)
    for sampler in settings.values():
        options.validate(sampler, target.C_theta.shape[0])
    if arguments.output.exists():
        raise FileExistsError(f"Fresh runs require a new output directory: {arguments.output}")
    arguments.output.mkdir(parents=True)
    experiment.write_json(arguments.output / "experiment.json", {
        "run": asdict(options), "methods": settings,
        "sampler_config_path": str(config) if config else "packaged defaults",
        "explicit_sampler_overrides": overrides, "prepared_input": str(arguments.prepared),
        "preparation": preparation, "environment": experiment.environment(),
        "initialization": "prior mu_theta; spatial eta; Sigma_theta=I; delta=m_delta_0; sigma_y2=1; sigma_c2=mean(F_s**2); conditional c_f",
    })
    try:
        states = experiment.initial_states(target, options.seed, options.chains)
    except Exception as error:
        experiment.write_json(arguments.output / "failure.json", {
            "phase": "initialization", "error_type": type(error).__name__, "message": str(error),
            "details": getattr(error, "diagnostics", {}), "configuration": "experiment.json",
        })
        raise
    for method in methods:
        directories = []
        for index, state in enumerate(states):
            directory = arguments.output / method / f"chain-{index}"
            directories.append(directory)
            try:
                chain = experiment.initialize_chain(target, state, method, options.seed, index,
                                                     options.num_warmup, settings[method])
            except Exception as error:
                directory.mkdir(parents=True)
                experiment.write_json(directory / "failure.json", {
                    "phase": "initialization", "method": method, "chain": index,
                    "error_type": type(error).__name__, "message": str(error),
                    "details": getattr(error, "diagnostics", {}),
                    "run": asdict(options), "sampler": settings[method],
                })
                raise
            experiment.run_chain(target, chain, directory, options, settings[method], index)
        from bayesiancalibration.analysis import analyze_method
        summary = analyze_method(target, directories)
        experiment.write_json(arguments.output / method / "analysis.json", summary)
        print(json.dumps({"method": method, "analysis": summary["status"],
                          "output": str(arguments.output / method)}), flush=True)


if __name__ == "__main__":
    main()
