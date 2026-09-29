"""Portable five-method experiment preparation and resumable timed chains.

Uses the existing Gibbs drivers and checkpoints; no new sampler is defined.
Each committed NPZ chunk retains every full sweep and sampler diagnostic.
The checkpoint is the commit record: files beyond it are orphaned, never
counted as draws, and may be overwritten by deterministic continuation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import tempfile
import time
import uuid

import jax
import jax.numpy as jnp
import numpy as np

from bayesiancalibration.gibbs import refresh_field_coefficients
from bayesiancalibration.gp import LibraryGP, fit_library_length_scales
from bayesiancalibration import mcmc
from bayesiancalibration.run import load_checkpoint, save_checkpoint
from bayesiancalibration.state import CalibrationState, SpatialPrior, ThetaStandardization
from bayesiancalibration.targets import CalibrationTarget
from bayesiancalibration.transforms import SiteCoordinates


METHODS = ("mh", "mala", "mmala", "nuts", "collapsed_nuts")


def task_coordinates(task_id: int) -> tuple[str, int]:
    """Map the agreed 20 array tasks to method and zero-based chain index."""
    if isinstance(task_id, bool) or not isinstance(task_id, int) or not 0 <= task_id < 20:
        raise ValueError("task_id must be an integer in [0, 19]")
    return METHODS[task_id // 4], task_id % 4


def validate_config(config: dict) -> None:
    """Reject incompatible experiment settings before fitting or starting jobs."""
    if config.get("nuts_mass_structure", "diagonal") not in ("diagonal", "kronecker", "dense"):
        raise ValueError("nuts_mass_structure must be diagonal, kronecker, or dense")
    if config["schema_version"] != 1 or config["chains"] != 4:
        raise ValueError("This experiment requires schema 1 and four chains")
    if config["model"]["site_support"] != "unbounded":
        raise ValueError("The agreed comparison uses unbounded site coordinates")
    for name in ("num_warmup", "num_initial", "chunk_sweeps", "max_num_doublings"):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config["num_initial"] > config["num_warmup"]:
        raise ValueError("num_initial cannot exceed num_warmup")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must be a uint32 integer")
    for name in (
        "production_seconds", "mala_epsilon", "mmala_epsilon", "epsilon_G",
        "nuts_initial_step_size", "divergence_threshold", "initial_proposal_variance",
        "mala_target_accept", "mmala_target_accept", "nuts_target_accept",
        "mala_initial_proposal_variance",
    ):
        value = (config.get(name, config["initial_proposal_variance"])
                 if name == "mala_initial_proposal_variance" else config[name])
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
        if name.endswith("target_accept") and value >= 1:
            raise ValueError(f"{name} must be less than one")


def implementation_hashes() -> dict:
    """Fingerprint preparation/runner code as well as numerical kernels."""
    root = Path(__file__).parent
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*.py"))}


def atomic_npz(path: Path, arrays: dict) -> None:
    """Commit draw/preparation archives with the checkpoint's atomic-write policy."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            np.savez_compressed(file, **arrays)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_target(arrays: dict, config: dict) -> CalibrationTarget:
    """Reconstruct the same explicit fixed target on each worker, without truth."""
    model = config["model"]
    standardization = ThetaStandardization.from_library(arrays["theta_s_dagger"])
    gp = LibraryGP.from_data(
        standardization.to_standardized(arrays["theta_s_dagger"]), arrays["F_s"],
        arrays["lambda_c"], jitter=config["library_fit"]["jitter"],
    )
    prior = SpatialPrior.from_standardization(standardization, **{
        name: model[name] for name in
        ("physical_center", "V_theta_0", "nu_theta_0", "S_theta_0")
    })
    return CalibrationTarget.from_data(
        gp, SiteCoordinates.from_physical_bounds(standardization), prior,
        arrays["y_tilde"], arrays["R"], arrays["s"], tuple(model["branch_sizes"]),
        **{name: model[name] for name in (
            "lambda_theta", "m_delta_0", "V_delta_0", "alpha_y_0", "beta_y_0",
            "alpha_c_0", "beta_c_0",
        )},
    )


def prepare(preprocessing: Path, config_path: Path, output: Path) -> None:
    """Fit once and freeze four dispersed full initial states, never using truth.

    Initialization is deliberately explicit: mu_theta is drawn from its prior;
    Sigma_theta=I; eta|mu_theta has covariance C_theta ⊗ I. delta starts at
    its prior mean, sigma_y2 at one, sigma_c2 at library column mean squares,
    and c_f is drawn from its exact observation-conditioned Gaussian. These
    are starting states, not changes to priors or frozen posterior variables.
    No retry, clipping, bounds, or added jitter repairs an invalid state.
    """
    if output.exists():
        raise FileExistsError(f"Refusing to replace frozen experiment: {output}")
    config = json.loads(config_path.read_text())
    validate_config(config)
    with np.load(preprocessing, allow_pickle=False) as source:
        arrays = {name: source[name] for name in
                  ("theta_s_dagger", "F_s", "y_tilde", "R", "s")}
        source_metadata = json.loads(str(source["metadata"].item()))
    standardization = ThetaStandardization.from_library(arrays["theta_s_dagger"])
    start = time.perf_counter()
    fit = fit_library_length_scales(
        standardization.to_standardized(arrays["theta_s_dagger"]), arrays["F_s"],
        **config["library_fit"],
    )
    fit_seconds = time.perf_counter() - start
    arrays["lambda_c"] = np.asarray(fit.lambda_c)
    target = build_target(arrays, config)
    n, d = target.C_theta.shape[0], target.spatial_prior.m_theta_0.size
    L_C = jnp.linalg.cholesky(target.C_theta)
    sigma_c2 = jnp.mean(jnp.square(target.gp.F_s), axis=0)
    key = jax.random.fold_in(jax.random.key(config["seed"]), 0)
    for index, initial_key in enumerate(jax.random.split(key, 4)):
        km, kt, kf = jax.random.split(initial_key, 3)
        mu_theta = jax.random.multivariate_normal(
            km, target.spatial_prior.m_theta_0, target.spatial_prior.V_theta_0,
            dtype=jnp.float64,
        )
        eta = mu_theta + L_C @ jax.random.normal(kt, (n, d), dtype=jnp.float64)
        sigma_y2 = jnp.ones(len(target.branch_sizes), dtype=jnp.float64)
        c_f = refresh_field_coefficients(
            kf, target, eta, target.m_delta_0, sigma_y2, sigma_c2,
        )
        state = CalibrationState(
            eta, c_f, target.m_delta_0, sigma_y2, mu_theta,
            jnp.eye(d, dtype=jnp.float64), sigma_c2,
        )
        mcmc.initialize_random_walk_chain(
            target, state, initial_key, jnp.tile(jnp.eye(d), (n, 1, 1)),
        )
        arrays.update({f"initial.{index}.{name}": np.asarray(value)
                       for name, value in zip(state._fields, state)})
    metadata = {
        "config": config, "fit": asdict(fit), "fit_seconds": fit_seconds,
        "preprocessing_sha256": hashlib.sha256(preprocessing.read_bytes()).hexdigest(),
        "preprocessing_metadata": source_metadata,
        "implementation_hashes": implementation_hashes(),
        "initialization": "prior-mu; C_theta-spatial-eta; Sigma=I; delta=m_delta_0; "
                          "sigma_y2=1; sigma_c2=mean(F_s**2); exact conditional c_f",
    }
    arrays["metadata"] = np.asarray(json.dumps(
        metadata, default=lambda value: np.asarray(value).tolist(), allow_nan=False,
        sort_keys=True,
    ))
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_npz(output, arrays)
    print(json.dumps({"prepared": str(output), "lambda_c": arrays["lambda_c"].tolist(),
                      "fit_seconds": fit_seconds}), flush=True)


def load_experiment(path: Path) -> tuple[CalibrationTarget, list, dict]:
    """Load the frozen input, refusing changed implementation or malformed settings."""
    with np.load(path, allow_pickle=False) as archive:
        arrays = dict(archive)
    metadata = json.loads(str(arrays.pop("metadata").item()))
    config = metadata["config"]
    validate_config(config)
    if metadata["implementation_hashes"] != implementation_hashes():
        raise ValueError("Prepared experiment implementation changed; prepare a new experiment")
    target = build_target(arrays, config)
    states = [CalibrationState(*(jnp.asarray(arrays[f"initial.{i}.{name}"])
                                 for name in CalibrationState._fields)) for i in range(4)]
    return target, states, config


def initialize_chain(target, state, config, method, index):
    """Dispatch to existing warmup factories with separate reproducible streams."""
    if method not in METHODS or not 0 <= index < 4:
        raise ValueError("Invalid method/chain")
    key = jax.random.fold_in(jax.random.key(config["seed"]), 1)
    key = jax.random.fold_in(jax.random.fold_in(key, METHODS.index(method)), index)
    shared = dict(num_warmup=config["num_warmup"])
    if method in ("mh", "mala"):
        n, d = state.eta.shape
        variance = config["initial_proposal_variance"]
        if method == "mala":
            variance = config.get("mala_initial_proposal_variance", variance)
        shared.update(num_initial=config["num_initial"], V_prop=jnp.tile(
            variance * jnp.eye(d), (n, 1, 1)))
        if method == "mh":
            return mcmc.initialize_random_walk_warmup(target, state, key, **shared)
        return mcmc.initialize_mala_warmup(
            target, state, key, **shared, epsilon=config["mala_epsilon"],
            target_accept=config["mala_target_accept"],
        )
    if method == "mmala":
        return mcmc.initialize_mmala_warmup(
            target, state, key, **shared, epsilon=config["mmala_epsilon"],
            epsilon_G=config["epsilon_G"], target_accept=config["mmala_target_accept"],
        )
    return mcmc.initialize_nuts_warmup(
        target, state, key, **shared, initial_step_size=config["nuts_initial_step_size"],
        target_accept=config["nuts_target_accept"],
        max_num_doublings=config["max_num_doublings"],
        divergence_threshold=config["divergence_threshold"], collapsed=method == "collapsed_nuts",
        mass_structure=config.get("nuts_mass_structure", "diagonal"),
    )


def run_chain(target, initial_chain, output: Path, config: dict, identity: dict,
              *, source_paths: dict, check_sweeps: int | None = None,
              stop_requested=lambda: False) -> dict:
    """Commit chunks under a single-writer lock and a cumulative production clock.

    Six hours means synchronized production driver time (including validation
    and host recording), excluding sweep compilation, warmup and disk I/O.
    Finish the last started chunk; overshoot is at most one chunk, reduced to
    one sweep near the budget. A killed uncommitted chunk is replayed. Slurm
    accounting remains the authority for total allocation cost, including
    failed attempts. check_sweeps stops after that many warmup sweeps, without
    changing the production warmup schedule or creating retained samples.
    """
    if check_sweeps is not None and (
        type(check_sweeps) is not int or not 1 <= check_sweeps <= config["num_warmup"]
    ):
        raise ValueError("check_sweeps must lie within the configured warmup")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_locked(target, initial_chain, output, config, identity, source_paths,
                           check_sweeps, stop_requested)


def _run_locked(target, chain, output, config, identity, source_paths, check_sweeps, stop):
    """Keep the draw-file/checkpoint commit sequence inside the worker lock."""
    session_start = time.perf_counter()
    checkpoint = output / "checkpoint.npz"
    signature = {"experiment": identity, "config": config, "check_sweeps": check_sweeps}
    progress = dict(production_seconds=0.0, warmup_seconds=0.0, chunks=[])
    if checkpoint.exists():
        chain, metadata = load_checkpoint(checkpoint, target, source_paths=source_paths)
        saved = metadata["configuration"]
        if saved["signature"] != signature:
            raise ValueError("Checkpoint experiment/configuration/mode does not match")
        progress = saved["progress"]
        previous = 0
        for chunk in progress["chunks"]:
            path = output / chunk["file"]
            if (chunk["start"] != previous + 1 or chunk["end"] < chunk["start"]
                or hashlib.sha256(path.read_bytes()).hexdigest() != chunk["sha256"]):
                raise ValueError("Missing, damaged, or noncontiguous committed draws")
            previous = chunk["end"]
        if previous != chain.iteration:
            raise ValueError("Draw/checkpoint boundary mismatch")
    elif list(output.glob("draws-*.npz")):
        # No boundary was committed; only the original supplied state may resume.
        print("Ignoring uncommitted draws; replaying from the initial boundary", flush=True)

    runner = mcmc.ChunkRunner(target, chain)
    compile_seconds, io_seconds = 0.0, 0.0
    status = "interrupted"
    session_id = uuid.uuid4().hex
    try:
        start = time.perf_counter()
        runner.compile(chain)
        compile_seconds = time.perf_counter() - start
        while not stop():
            if check_sweeps is not None and chain.iteration >= check_sweeps:
                status = "check_complete"
                break
            phase = chain.phase
            count = config["chunk_sweeps"]
            if phase == "warmup":
                count = min(count, config["num_warmup"] - chain.iteration)
                if check_sweeps is not None:
                    count = min(count, check_sweeps - chain.iteration)
            else:
                remaining = config["production_seconds"] - progress["production_seconds"]
                if remaining <= 0:
                    status = "budget_complete"
                    break
                production_draws = chain.iteration - config["num_warmup"]
                if production_draws:
                    seconds_per_draw = progress["production_seconds"] / production_draws
                    count = min(count, max(1, int(remaining / seconds_per_draw)))
            start_iteration = chain.iteration + 1
            start = time.perf_counter()
            candidate, samples, diagnostics = runner(chain, count)
            jax.block_until_ready((samples, diagnostics))
            elapsed = time.perf_counter() - start
            arrays = {f"state.{name}": np.asarray(value)
                      for name, value in zip(samples._fields, samples)}
            arrays.update({f"theta.{name}": np.asarray(value)
                           for name, value in zip(diagnostics.theta._fields, diagnostics.theta)})
            arrays["full_joint_logdensity"] = np.asarray(diagnostics.full_joint_logdensity)
            arrays["site_moved"] = np.asarray(diagnostics.site_moved)
            arrays["iteration"] = np.arange(start_iteration, candidate.iteration + 1)
            arrays["phase"] = np.asarray(phase)
            filename = f"draws-{start_iteration:09d}-{candidate.iteration:09d}.npz"
            start = time.perf_counter()
            atomic_npz(output / filename, arrays)
            chunk = dict(file=filename, start=start_iteration, end=candidate.iteration,
                         phase=phase, seconds=elapsed,
                         sha256=hashlib.sha256((output / filename).read_bytes()).hexdigest())
            candidate_progress = dict(progress)
            candidate_progress["chunks"] = [*progress["chunks"], chunk]
            clock_name = "warmup_seconds" if phase == "warmup" else "production_seconds"
            candidate_progress[clock_name] += elapsed
            previous_counts = progress.get("site_counts", {}).get(phase, {})
            counts = {"movement_count": (
                np.asarray(previous_counts.get("movement_count", 0))
                + np.sum(arrays["site_moved"], axis=0)
            ).tolist()}
            if hasattr(diagnostics.theta, "is_accepted"):
                counts["acceptance_count"] = (
                    np.asarray(previous_counts.get("acceptance_count", 0))
                    + np.sum(arrays["theta.is_accepted"], axis=0)
                ).tolist()
            candidate_progress["site_counts"] = {**progress.get("site_counts", {}), phase: counts}
            if isinstance(candidate.adaptation, mcmc.RandomWalkAdaptationState):
                adapt = candidate.adaptation
                candidate_progress["warmup_tuning"] = {
                    name: np.asarray(getattr(adapt, name)).tolist() for name in
                    ("acceptance_count", "movement_count", "zero_covariance_count")
                }
            save_checkpoint(
                checkpoint, target, candidate, source_paths=source_paths,
                configuration=dict(signature=signature, progress=candidate_progress),
            )
            io_seconds += time.perf_counter() - start
            chain, progress = candidate, candidate_progress
            print(json.dumps({"iteration": chain.iteration, "phase": phase,
                              "chunk_seconds": elapsed,
                              "production_seconds": progress["production_seconds"]}), flush=True)
    except mcmc.WarmupTuningError as error:
        status = "warmup_failed"
        (output / "warmup_failure.json").write_text(
            json.dumps(error.diagnostics, indent=2) + "\n")
        raise
    except BaseException:
        status = "failed"
        raise
    finally:
        session = dict(
            status=status, iteration=chain.iteration, **{k: v for k, v in progress.items()
                                                       if k != "chunks"},
            sweep_compile_seconds=compile_seconds, io_seconds_this_session=io_seconds,
            wall_seconds_this_session=time.perf_counter() - session_start,
            platform=platform.platform(), hostname=platform.node(), backend=jax.default_backend(),
            devices=[str(device) for device in jax.devices()],
            environment={name: os.environ.get(name) for name in (
                "SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_CPUS_PER_TASK",
                "SLURM_JOB_PARTITION", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "XLA_FLAGS",
            )},
        )
        (output / f"session-{session_id}.json").write_text(
            json.dumps(session, indent=2, allow_nan=False) + "\n")
    return session


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--preprocessing", type=Path, required=True)
    prep.add_argument("--config", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--prepared", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--task-id", type=int, required=True)
    run.add_argument("--check-sweeps", type=int)
    run.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command == "run":
        method, index = task_coordinates(args.task_id)
        if args.dry_run:
            print(json.dumps(dict(method=method, chain=index, prepared=str(args.prepared),
                                  mode="check" if args.check_sweeps else "production")))
            return
    jax.config.update("jax_enable_x64", True)
    if args.command == "prepare":
        prepare(args.preprocessing, args.config, args.output)
        return
    target, states, config = load_experiment(args.prepared)
    chain = initialize_chain(target, states[index], config, method, index)
    stopped = False

    def request_stop(signum, frame):
        nonlocal stopped
        stopped = True

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, request_stop)
    identity = dict(method=method, chain=index,
                    prepared_sha256=hashlib.sha256(args.prepared.read_bytes()).hexdigest(),
                    implementation_hashes=implementation_hashes())
    mode = "check" if args.check_sweeps is not None else "production"
    result = run_chain(
        target, chain, args.output / mode / method / f"chain-{index}", config, identity,
        source_paths={"prepared": args.prepared}, check_sweeps=args.check_sweeps,
        stop_requested=lambda: stopped,
    )
    print(json.dumps(result), flush=True)
    if result["status"] == "interrupted":
        raise SystemExit(75)


if __name__ == "__main__":
    main()
