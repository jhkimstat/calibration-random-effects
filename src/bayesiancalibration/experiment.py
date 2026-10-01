"""Small fresh-run research recipe; numerical algorithms live in ``mcmc``.

Prepared archives contain fixed numerical inputs, never restart state. Draw
archives contain successful retained batches, never keys/adaptation state.
Site-major stacking is used throughout: site, then active branch/coefficient.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import metadata, resources
import json
from pathlib import Path
import platform
import time

import jax
import jax.numpy as jnp
import numpy as np

from bayesiancalibration import mcmc
from bayesiancalibration.gibbs import refresh_field_coefficients
from bayesiancalibration.gp import LibraryGP, fit_library_length_scales
from bayesiancalibration.samplers.nuts import nuts_blocks
from bayesiancalibration.state import CalibrationState, SpatialPrior, ThetaStandardization
from bayesiancalibration.targets import CalibrationTarget
from bayesiancalibration.transforms import SiteCoordinates


METHODS = ("mh", "mala", "mmala", "nuts", "collapsed_nuts")
SAMPLER_FIELDS = {
    "mh": {"num_initial", "initial_proposal_variance"},
    "mala": {"num_initial", "initial_proposal_variance", "epsilon", "target_accept"},
    "mmala": {"epsilon", "epsilon_G", "target_accept"},
    "nuts": {"initial_step_size", "target_accept", "mass_structure", "block_size",
             "max_num_doublings", "divergence_threshold"},
}
SAMPLER_FIELDS["collapsed_nuts"] = SAMPLER_FIELDS["nuts"]
PREPARED_FIELDS = ("theta_s_dagger", "F_s", "y_tilde", "R", "s")


def default_config(name: str) -> dict:
    """Read installed JSON resources, so the recipe works outside a checkout."""
    return json.loads(resources.files("bayesiancalibration").joinpath(
        "configs", name + ".json"
    ).read_text())


def write_json(path: Path, value) -> None:
    """Convert numerical summaries to portable JSON, marking undefined values null.

    This small conversion handles arrays/scalars that the standard JSON encoder
    cannot encode, including undefined short-chain convergence diagnostics.
    """
    def plain(item):
        if isinstance(item, dict):
            return {str(k): plain(v) for k, v in item.items()}
        if isinstance(item, (tuple, list)):
            return [plain(v) for v in item]
        if isinstance(item, (np.ndarray, jax.Array)):
            return plain(np.asarray(item).tolist())
        if isinstance(item, np.generic):
            return plain(item.item())
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item
    path.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")


def environment() -> dict:
    """Document the new run without historical source-hash compatibility gates."""
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": {name: metadata.version(name) for name in
                     ("bayesiancalibration", "jax", "blackjax", "numpy", "scipy")},
        "backend": jax.default_backend(), "devices": [str(d) for d in jax.devices()],
        "float64": bool(jax.config.jax_enable_x64),
    }


def validate_scientific_config(config: dict) -> None:
    """Validate the fixed specification; model constructors validate array shapes."""
    if not isinstance(config, dict) or set(config) != {"model", "library_fit"}:
        raise ValueError("Scientific configuration requires only model and library_fit")
    model, fit = config["model"], config["library_fit"]
    model_fields = {"site_support", "branch_sizes", "lambda_theta", "physical_center",
                    "V_theta_0", "nu_theta_0", "S_theta_0", "m_delta_0", "V_delta_0",
                    "alpha_y_0", "beta_y_0", "alpha_c_0", "beta_c_0", "l", "u"}
    required = model_fields - {"l", "u"}
    if not isinstance(model, dict) or not required <= set(model) or set(model) - model_fields:
        raise ValueError("Unknown or missing model settings")
    if model["site_support"] not in ("unbounded", "bounded"):
        raise ValueError("model.site_support must be unbounded or bounded")
    bounded = model["site_support"] == "bounded"
    if bounded != ({"l", "u"} <= set(model)) or (not bounded and ({"l", "u"} & set(model))):
        raise ValueError("Bounded sites require both physical l and u; unbounded sites omit them")
    fit_fields = {"method", "starts", "gtol", "ftol", "maxiter", "log_bounds", "jitter"}
    if not isinstance(fit, dict) or set(fit) - fit_fields or not (fit_fields - {"method"}) <= set(fit):
        raise ValueError("Unknown or missing library_fit settings")
    if fit.get("method", "profile") not in ("profile", "cv_nlpd", "cv_wmse"):
        raise ValueError("library_fit.method must be profile, cv_nlpd, or cv_wmse")


def resolve_sampler_settings(method: str, config_path: Path | None = None,
                             overrides: dict | None = None) -> dict:
    """Resolve once: declared defaults < file values < explicit CLI overrides."""
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    result = default_config(method)
    for source in (json.loads(Path(config_path).read_text()) if config_path else {},
                   overrides or {}):
        if not isinstance(source, dict):
            raise ValueError("Sampler configuration must be a JSON object")
        if source.get("method", method) != method:
            raise ValueError("Sampler file method does not match selected method")
        unknown = set(source) - SAMPLER_FIELDS[method] - {"method"}
        if unknown:
            raise ValueError(f"Unknown or inapplicable {method} settings: {sorted(unknown)}")
        result.update(source)
    result["method"] = method
    for name, value in result.items():
        if name in ("method", "mass_structure", "block_size"):
            continue
        if name in ("num_initial", "max_num_doublings"):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        elif name == "target_accept" and value is None and method == "mala":
            continue  # Explicitly disable MALA epsilon adaptation, retaining covariance adaptation.
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
        elif name == "target_accept" and value >= 1:
            raise ValueError("target_accept must lie in (0,1)")
    if method in ("nuts", "collapsed_nuts"):
        if result["mass_structure"] not in ("diagonal", "kronecker", "dense"):
            raise ValueError("mass_structure must be diagonal, kronecker, or dense")
        block = result["block_size"]
        if block is not None and (type(block) is not int or block < 1):
            raise ValueError("block_size must be null or a positive integer")
    return result


@dataclass(frozen=True)
class RunOptions:
    """Local run controls, separate from fixed science and sampler tuning."""
    seed: int = 20260928
    chains: int = 4
    num_warmup: int = 1000
    num_samples: int | None = 1000
    production_seconds: float | None = None
    batch_size: int = 10
    diagnostic: bool = False

    def validate(self, sampler: dict, num_sites: int) -> None:
        for name in ("chains", "num_warmup", "batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be a uint32 integer")
        if (self.num_samples is None) == (self.production_seconds is None):
            raise ValueError("Specify exactly one of num_samples and production_seconds")
        if self.num_samples is not None and (type(self.num_samples) is not int or self.num_samples < 1):
            raise ValueError("num_samples must be a positive integer")
        if self.production_seconds is not None and (
            isinstance(self.production_seconds, bool) or not np.isfinite(self.production_seconds)
            or self.production_seconds <= 0
        ):
            raise ValueError("production_seconds must be finite and positive")
        if sampler.get("num_initial", 1) > self.num_warmup:
            raise ValueError("num_initial cannot exceed num_warmup")
        if sampler["method"] in ("nuts", "collapsed_nuts"):
            nuts_blocks(num_sites, sampler["block_size"])


def build_target(arrays: dict, config: dict) -> CalibrationTarget:
    """Construct the fixed posterior from observed inputs only, never truth."""
    validate_scientific_config(config)
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
    bounds = {name: model[name] for name in ("l", "u")} if model["site_support"] == "bounded" else {}
    return CalibrationTarget.from_data(
        gp, SiteCoordinates.from_physical_bounds(standardization, **bounds), prior,
        arrays["y_tilde"], arrays["R"], arrays["s"], tuple(model["branch_sizes"]),
        **{name: model[name] for name in (
            "lambda_theta", "m_delta_0", "V_delta_0", "alpha_y_0", "beta_y_0",
            "alpha_c_0", "beta_c_0",
        )},
    )


def prepare_experiment(preprocessing: Path, scientific_config: dict, output: Path) -> dict:
    """Fit the library once and freeze common inputs for every new method/chain."""
    if output.exists():
        raise FileExistsError(f"Refusing to replace prepared inputs: {output}")
    validate_scientific_config(scientific_config)
    scientific_config = {
        "model": scientific_config["model"],
        "library_fit": {"method": "profile", **scientific_config["library_fit"]},
    }
    with np.load(preprocessing, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name], dtype=np.float64) for name in PREPARED_FIELDS}
        source_metadata = json.loads(str(source["metadata"].item())) if "metadata" in source else {}
    standardization = ThetaStandardization.from_library(arrays["theta_s_dagger"])
    start = time.perf_counter()
    fit = fit_library_length_scales(
        standardization.to_standardized(arrays["theta_s_dagger"]), arrays["F_s"],
        **scientific_config["library_fit"],
    )
    arrays["lambda_c"] = np.asarray(fit.lambda_c)
    build_target(arrays, scientific_config)
    details = {
        "scientific_config": scientific_config, "fit": asdict(fit),
        "fit_seconds": time.perf_counter() - start,
        "input": str(preprocessing), "input_metadata": source_metadata,
        "standardization": {
            "source": standardization.source,
            "theta_bar_dagger": np.asarray(standardization.theta_bar_dagger).tolist(),
            "D_theta": np.asarray(standardization.D_theta).tolist(),
            "sample_variance_divisor": len(arrays["theta_s_dagger"]) - 1,
        },
        "environment": environment(), "stacking": "site-major, then branch/coefficient",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays["metadata"] = np.asarray(json.dumps(
        details, default=lambda value: np.asarray(value).tolist(), allow_nan=False,
    ))
    with output.open("xb") as archive:
        np.savez_compressed(archive, **arrays)
    return details


def load_experiment(path: Path) -> tuple[CalibrationTarget, dict, dict]:
    """Read common fixed inputs without historical implementation compatibility."""
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in (*PREPARED_FIELDS, "lambda_c")}
        details = json.loads(str(archive["metadata"].item()))
    return build_target(arrays, details["scientific_config"]), arrays, details


def initial_states(target: CalibrationTarget, seed: int, chains: int) -> list[CalibrationState]:
    """Current unbounded experiment recipe, shared by chain index across methods.

    mu_theta is a prior draw; Sigma_theta=I; eta|mu_theta has C_theta⊗I
    covariance. delta=m_delta_0, sigma_y2=1, sigma_c2=mean(F_s**2);
    c_f is an exact conditional draw. No repair, clipping, or retries occur.
    Bounded automatic initialization is unspecified; use an explicitly supplied
    complete model state with the core numerical API in that mode.
    """
    if target.coordinates.bounded:
        raise ValueError("Automatic bounded initialization is unspecified; supply a complete state to the core API")
    n, d = target.C_theta.shape[0], target.spatial_prior.m_theta_0.size
    L_C = jnp.linalg.cholesky(target.C_theta)
    sigma_c2 = jnp.mean(jnp.square(target.gp.F_s), axis=0)
    initialization_key = jax.random.fold_in(jax.random.key(seed), 0)
    states = []
    for index in range(chains):
        km, kt, kf = jax.random.split(jax.random.fold_in(initialization_key, index), 3)
        mu_theta = jax.random.multivariate_normal(
            km, target.spatial_prior.m_theta_0, target.spatial_prior.V_theta_0, dtype=jnp.float64,
        )
        eta = mu_theta + L_C @ jax.random.normal(kt, (n, d), dtype=jnp.float64)
        sigma_y2 = jnp.ones(len(target.branch_sizes), dtype=jnp.float64)
        try:
            c_f = refresh_field_coefficients(kf, target, eta, target.m_delta_0, sigma_y2, sigma_c2)
        except Exception as error:
            # Preserve the original failure type/policy while attaching the
            # independent initialization clock and exact conditional inputs.
            error.diagnostics = {
                **getattr(error, "diagnostics", {}), "phase": "initialization", "chain": index,
                "update": "c_f", "predicate": str(error), "seed": seed,
                "key_data": np.asarray(jax.random.key_data(kf)).tolist(),
                "eta": np.asarray(eta).tolist(), "mu_theta": np.asarray(mu_theta).tolist(),
                "delta": np.asarray(target.m_delta_0).tolist(),
                "sigma_y2": np.asarray(sigma_y2).tolist(), "sigma_c2": np.asarray(sigma_c2).tolist(),
            }
            raise
        states.append(CalibrationState(
            eta, c_f, target.m_delta_0, sigma_y2, mu_theta, jnp.eye(d), sigma_c2,
        ))
    return states


def initialize_chain(target, state, method: str, seed: int, index: int,
                     num_warmup: int, sampler: dict):
    """Dispatch the existing algorithms with independent method/chain streams."""
    key = jax.random.fold_in(jax.random.key(seed), 1)
    key = jax.random.fold_in(jax.random.fold_in(key, METHODS.index(method)), index)
    settings = {name: value for name, value in sampler.items() if name != "method"}
    settings["num_warmup"] = num_warmup
    if method in ("mh", "mala"):
        n, d = state.eta.shape
        variance = settings.pop("initial_proposal_variance")
        settings["V_prop"] = jnp.tile(variance * jnp.eye(d), (n, 1, 1))
        factory = mcmc.initialize_random_walk_warmup if method == "mh" else mcmc.initialize_mala_warmup
    elif method == "mmala":
        factory = mcmc.initialize_mmala_warmup
    else:
        settings["collapsed"] = method == "collapsed_nuts"
        factory = mcmc.initialize_nuts_warmup
    return factory(target, state, key, **settings)


def draw_arrays(samples, diagnostics, start: int) -> dict:
    """Flatten named diagnostic blocks into pickle-free numeric arrays."""
    arrays = {f"state.{name}": np.asarray(value) for name, value in zip(samples._fields, samples)}
    for name, value in zip(diagnostics.theta._fields, diagnostics.theta):
        if isinstance(value, tuple):
            arrays.update({f"theta.{name}.block_{i}": np.asarray(v) for i, v in enumerate(value)})
        else:
            arrays[f"theta.{name}"] = np.asarray(value)
    arrays["full_joint_logdensity"] = np.asarray(diagnostics.full_joint_logdensity)
    arrays["site_moved"] = np.asarray(diagnostics.site_moved)
    arrays["sweep"] = np.arange(start, start + len(arrays["state.eta"]))
    return arrays


def run_chain(target, chain, output: Path, options: RunOptions,
              sampler: dict, chain_index: int) -> dict:
    """Compile, warm up, and retain successful batches in one fresh local run.

    A time budget measures synchronized sampling/validation/host recording,
    excluding compilation, warmup and file I/O. Finish the last started batch;
    overshoot is at most one batch. Output cannot be used to resume sampling.
    """
    options.validate(sampler, target.C_theta.shape[0])
    output.mkdir(parents=True, exist_ok=False)
    resolved = {"run": asdict(options), "sampler": sampler, "chain": chain_index,
                "environment": environment()}
    write_json(output / "configuration.json", resolved)
    completed, production_seconds, warmup_seconds = 0, 0.0, 0.0
    batch_start = chain.iteration + 1
    phase = chain.phase
    requested = 0
    try:
        runner = mcmc.SweepRunner(target, chain, diagnostic=options.diagnostic)
        runner.compile(chain)
        while chain.phase == "warmup":
            phase, batch_start = "warmup", chain.iteration + 1
            requested = min(options.batch_size, options.num_warmup - chain.iteration)
            start = time.perf_counter()
            candidate, samples, diagnostics = runner(chain, requested, chain_index=chain_index)
            jax.block_until_ready((samples, diagnostics))
            warmup_seconds += time.perf_counter() - start
            # Warmup is reported separately and is never a retained draw.
            np.savez_compressed(output / f"warmup-{batch_start:09d}.npz",
                                **draw_arrays(samples, diagnostics, batch_start))
            chain = candidate
        while ((options.num_samples is not None and completed < options.num_samples)
               or (options.production_seconds is not None and production_seconds < options.production_seconds)):
            phase, batch_start = "sampling", chain.iteration + 1
            requested = options.batch_size
            if options.num_samples is not None:
                requested = min(requested, options.num_samples - completed)
            elif completed:
                remaining = options.production_seconds - production_seconds
                requested = min(requested, max(1, int(remaining * completed / production_seconds)))
            start = time.perf_counter()
            candidate, samples, diagnostics = runner(chain, requested, chain_index=chain_index)
            jax.block_until_ready((samples, diagnostics))
            arrays = draw_arrays(samples, diagnostics, batch_start)
            production_seconds += time.perf_counter() - start
            np.savez_compressed(output / f"draws-{completed + 1:09d}.npz", **arrays)
            chain, completed = candidate, completed + requested
        tuning = {name: getattr(chain, name) for name in
                  ("V_prop", "epsilon", "epsilon_G", "step_size", "inverse_mass_matrix")
                  if hasattr(chain, name)}
        result = {"status": "complete", "retained_draws": completed,
                  "production_seconds": production_seconds, "warmup_seconds": warmup_seconds,
                  "final_tuning": tuning}
        write_json(output / "result.json", result)
        return result
    except Exception as error:
        evidence = {"status": "failed", "method": sampler["method"], "chain": chain_index,
                    "phase": phase, "batch_start": batch_start, "requested_batch_size": requested,
                    "last_valid_sweep": chain.iteration, "retained_draws": completed,
                    "error_type": type(error).__name__, "message": str(error),
                    "details": getattr(error, "diagnostics", {}),
                    "key_data_at_valid_boundary": jax.random.key_data(chain.key),
                    "experiment_configuration": "../../experiment.json",
                    "configuration": resolved}
        write_json(output / "failure.json", evidence)
        raise
