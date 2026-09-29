"""Atomic, pickle-free checkpoints for completed calibration Gibbs sweeps."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from dataclasses import fields, is_dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from blackjax.adaptation.mass_matrix import WelfordAlgorithmState, MassMatrixAdaptationState
from blackjax.adaptation.staged_adaptation import StagedAdaptationState, build_schedule
from blackjax.adaptation.step_size import DualAveragingAdaptationState

from bayesiancalibration.adaptation import (
    MALAStepSizeAdaptationState,
    RandomWalkAdaptationState,
    NUTSAdaptationState,
    KroneckerWelfordState,
    MMALAAdaptationState,
)
from bayesiancalibration.samplers.nuts import nuts_blocks
from bayesiancalibration.mcmc import (
    MALAChain,
    RandomWalkChain,
    validate_mala_chain,
    validate_random_walk_chain,
    NUTSChain,
    validate_nuts_chain,
    MMALAChain,
    validate_mmala_chain,
)
from bayesiancalibration.state import CalibrationState
from bayesiancalibration.targets import CalibrationTarget


_SCHEMA_VERSION = 4
_KEY_PROTOCOL = "split-8-v1"
_JAX_CONFIG_FIELDS = (
    "jax_enable_x64", "jax_default_prng_impl", "jax_threefry_partitionable",
    "jax_default_matmul_precision",
)


def _target_snapshot(target: CalibrationTarget) -> dict[str, np.ndarray]:
    """Flatten the fixed specification, including frozen maps and GP factors.

    A custom traversal keeps this model's dataclass structure in a portable
    array archive without pickle. Fixed factors are fingerprinted alongside
    their dependencies; no changing density/gradient cache is checkpointed.
    """

    arrays = {}

    def visit(name, value):
        if is_dataclass(value):
            for field in fields(value):
                visit(f"{name}.{field.name}", getattr(value, field.name))
        else:
            arrays[name] = np.asarray("None" if value is None else value)

    visit("target", target)
    return arrays


def _array_digest(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash names, shapes, dtypes, and bytes to detect incompatible payloads.

    hashlib provides the standard digest; this framing makes array boundaries
    unambiguous and includes the fixed target and stored PRNG representation.
    """

    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        array = np.asarray(value)
        header = json.dumps([name, array.dtype.str, list(array.shape)]).encode()
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        data = array.tobytes(order="C")
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest()


def _implementation_hashes() -> dict[str, str]:
    """Record the code that defines the target, transitions, and restart format."""

    root = Path(__file__).parent
    names = (
        "state.py", "transforms.py", "linalg.py", "gp.py", "targets.py",
        "gibbs.py", "samplers/metropolis.py", "samplers/nuts.py", "samplers/mmala.py",
        "adaptation.py", "mcmc.py", "run.py",
    )
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in names}


def save_checkpoint(
    path: str | Path,
    target: CalibrationTarget,
    chain: RandomWalkChain | MALAChain | NUTSChain | MMALAChain,
    *,
    configuration: dict | None = None,
    source_paths: Mapping[str, str | Path] | None = None,
) -> None:
    """Atomically save one validated completed chain boundary to an NPZ file.

    Stores every model variable, next-sweep sampler tuning, unused key with its PRNG
    implementation/typed status, iteration/phase, the full fixed target,
    warmup covariance/dual-averaging/window schedules/statistics when present,
    payload/code hashes, versions,
    float64/backend information, and caller configuration. source_paths
    records named statistical-note/data hashes;
    supply these for real runs (synthetic fixtures may leave them empty).

    A temporary archive is flushed in the destination directory before an
    atomic replace, so an interrupted write preserves the old checkpoint.
    Only completed boundaries returned by the chain driver may be saved.
    """

    is_mala, is_nuts = isinstance(chain, MALAChain), isinstance(chain, NUTSChain)
    is_mmala = isinstance(chain, MMALAChain)
    validate = validate_mmala_chain if is_mmala else (
        validate_nuts_chain if is_nuts else (
            validate_mala_chain if is_mala else validate_random_walk_chain
        )
    )
    chain = validate(target, chain)
    arrays = _target_snapshot(target)
    target_digest = _array_digest(arrays)
    arrays.update({f"state.{name}": np.asarray(value)
                   for name, value in zip(CalibrationState._fields, chain.model_state)})
    nuts_metadata = None
    if is_nuts:
        arrays["sampler.step_size"] = np.asarray(chain.step_size, dtype=np.float64)
        if chain.block_size is None:
            arrays["sampler.inverse_mass_matrix"] = np.asarray(chain.inverse_mass_matrix)
        else:
            arrays.update({f"sampler.inverse_mass_matrix.{i}": np.asarray(mass)
                           for i, mass in enumerate(chain.inverse_mass_matrix)})
        nuts_metadata = {
            "max_num_doublings": chain.max_num_doublings,
            "divergence_threshold": chain.divergence_threshold,
            "integrator": "velocity_verlet", "mass_matrix": chain.mass_structure,
            "coordinates": "eta-v1", "block_size": chain.block_size,
            "block_order": "contiguous-sites-v1",
            "theta_key_protocol": "split-blocks-v1" if chain.block_size is not None else "unsplit-v1",
        }
    elif not is_mmala:
        arrays["sampler.V_prop"] = np.asarray(chain.V_prop)
    if is_mala or is_mmala:
        arrays["sampler.epsilon"] = np.asarray(chain.epsilon, dtype=np.float64)
    if is_mmala:
        arrays["sampler.epsilon_G"] = np.asarray(chain.epsilon_G, dtype=np.float64)
    step_size_metadata = None
    step_size = chain.step_size_adaptation if is_mala else (
        chain.adaptation.step_size if is_mmala and chain.adaptation is not None else None
    )
    if step_size is not None:
        arrays["step_size.target_accept"] = np.asarray(
            step_size.target_accept, dtype=np.float64
        )
        arrays["step_size.initial_epsilon"] = np.asarray(
            step_size.initial_epsilon, dtype=np.float64
        )
        arrays.update({f"step_size.{name}": np.asarray(value) for name, value in zip(
            step_size.state._fields, step_size.state
        )})
        step_size_metadata = {"protocol": (
            "mmala-dual-averaging-v1" if is_mmala else "mala-dual-averaging-v1"
        )}
    adaptation_metadata = None
    if is_nuts and chain.adaptation is not None:
        adaptation = chain.adaptation
        windows = adaptation.state if chain.block_size is not None else (adaptation.state,)
        for i, window in enumerate(windows):
            root = f"window.{i}" if chain.block_size is not None else "window"
            for prefix, state in (("ss", window.ss_state), ("wc", window.imm_state.wc_state)):
                arrays.update({f"{root}.{prefix}.{name}": np.asarray(value)
                               for name, value in zip(state._fields, state)})
            arrays[f"{root}.step_size"] = np.asarray(window.step_size)
            arrays[f"{root}.inverse_mass_matrix"] = np.asarray(window.inverse_mass_matrix)
        arrays["window.schedule"] = np.asarray(build_schedule(adaptation.num_warmup))
        adaptation_metadata = {
            "protocol": ("eta-blocked-factor-moments-v1" if chain.block_size is not None
                         else "eta-staged-factor-moments-v2"),
            "num_warmup": adaptation.num_warmup, "completed": adaptation.completed,
            "initial_step_size": adaptation.initial_step_size,
            "target_accept": adaptation.target_accept,
            "buffers": [75, 25, 50], "imm_shrinkage_to_previous": 0.0,
            "dual_averaging": {"t0": 10, "gamma": 0.05, "kappa": 0.75},
        }
    elif is_mmala and chain.adaptation is not None:
        adaptation_metadata = {
            "protocol": "mmala-epsilon-only-v1", "num_warmup": chain.adaptation.num_warmup,
        }
    elif chain.adaptation is not None:
        adaptation = chain.adaptation
        arrays["adaptation.initial_V_prop"] = np.asarray(adaptation.initial_V_prop)
        arrays.update({f"adaptation.{name}": np.asarray(value) for name, value in zip(
            ("mean", "m2", "sample_size"), adaptation.moments
        )})
        arrays.update({f"adaptation.{name}": np.asarray(getattr(adaptation, name))
                       for name in ("acceptance_count", "movement_count", "zero_covariance_count")})
        adaptation_metadata = {
            "protocol": "empirical-covariance-v2",
            "num_warmup": adaptation.num_warmup, "num_initial": adaptation.num_initial,
        }
    arrays["rng.data"] = np.asarray(jax.random.key_data(chain.key))
    packages = {name: version(name) for name in (
        "bayesiancalibration", "jax", "jaxlib", "numpy", "scipy", "blackjax",
    )}
    metadata = {
        "schema_version": _SCHEMA_VERSION,
        "sampler": ("collapsed_nuts" if chain.collapsed else "uncollapsed_nuts")
        if is_nuts else (
            "collapsed_mmala" if is_mmala else (
                "collapsed_mala" if is_mala else "collapsed_random_walk"
            )
        ),
        "nuts": nuts_metadata,
        "key_protocol": _KEY_PROTOCOL,
        "iteration": chain.iteration, "phase": chain.phase,
        "adaptation": adaptation_metadata,
        "step_size_adaptation": step_size_metadata,
        "rng_impl": str(jax.random.key_impl(chain.key)),
        "rng_typed": jax.dtypes.issubdtype(chain.key.dtype, jax.dtypes.prng_key),
        "dtype": "float64", "backend": jax.default_backend(),
        "jax_configuration": {name: jax.config.values[name]
                              for name in _JAX_CONFIG_FIELDS},
        "xla_flags": os.environ.get("XLA_FLAGS"),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": packages, "implementation_hashes": _implementation_hashes(),
        "target_sha256": target_digest, "payload_sha256": _array_digest(arrays),
        "configuration": {} if configuration is None else configuration,
        "source_hashes": {
            name: hashlib.sha256(Path(source).read_bytes()).hexdigest()
            for name, source in ({} if source_paths is None else source_paths).items()
        },
    }
    arrays["metadata"] = np.asarray(
        json.dumps(metadata, sort_keys=True, allow_nan=False)
    )
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as file:
            temporary = Path(file.name)
            np.savez_compressed(file, **arrays)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: str | Path,
    target: CalibrationTarget,
    *,
    source_paths: Mapping[str, str | Path] | None = None,
) -> tuple[RandomWalkChain | MALAChain | NUTSChain | MMALAChain, dict]:
    """Load and validate a completed checkpoint against the supplied target.

    Exact target, code, and numerical-package versions are required. Hardware
    and Python provenance are recorded; bitwise continuation is tested on
    the same backend/environment. Optional source_paths verifies the caller's
    current notes/data against saved hashes. The target is supplied explicitly
    so restart cannot silently change data, bounds, priors, fitted length
    scales, standardization, or numerical policy. No pickled objects are read.
    """

    with np.load(path, allow_pickle=False) as archive:
        try:
            metadata = json.loads(str(archive["metadata"].item()))
            arrays = {name: archive[name] for name in archive.files
                      if name != "metadata"}
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Invalid checkpoint metadata or array payload") from error
    if not isinstance(metadata, dict):
        raise ValueError("Invalid checkpoint metadata")
    if (
        metadata.get("schema_version") != _SCHEMA_VERSION
        or metadata.get("sampler") not in (
            "collapsed_random_walk", "collapsed_mala", "uncollapsed_nuts",
            "collapsed_nuts", "collapsed_mmala",
        )
        or metadata.get("key_protocol") != _KEY_PROTOCOL
        or metadata.get("dtype") != "float64"
    ):
        raise ValueError("Unsupported checkpoint schema, sampler, or key protocol")
    expected_packages = {name: version(name) for name in (
        "bayesiancalibration", "jax", "jaxlib", "numpy", "scipy", "blackjax",
    )}
    if metadata.get("packages") != expected_packages:
        raise ValueError("Checkpoint numerical-package versions do not match")
    if (
        metadata.get("jax_configuration") != {
            name: jax.config.values[name] for name in _JAX_CONFIG_FIELDS
        }
        or metadata.get("xla_flags") != os.environ.get("XLA_FLAGS")
    ):
        raise ValueError("Checkpoint JAX configuration does not match")
    if metadata.get("implementation_hashes") != _implementation_hashes():
        raise ValueError("Checkpoint implementation hashes do not match")
    if metadata.get("payload_sha256") != _array_digest(arrays):
        raise ValueError("Checkpoint payload checksum does not match")
    fixed_arrays = {name: value for name, value in arrays.items()
                    if name.startswith("target.")}
    if (
        metadata.get("target_sha256") != _array_digest(fixed_arrays)
        or metadata.get("target_sha256") != _array_digest(_target_snapshot(target))
    ):
        raise ValueError("Checkpoint fixed target does not match")
    if source_paths is not None:
        hashes = {name: hashlib.sha256(Path(source).read_bytes()).hexdigest()
                  for name, source in source_paths.items()}
        if metadata.get("source_hashes") != hashes:
            raise ValueError("Checkpoint source hashes do not match")
    try:
        model_arrays = [arrays[f"state.{name}"] for name in CalibrationState._fields]
        is_nuts = metadata["sampler"] in ("uncollapsed_nuts", "collapsed_nuts")
        is_mmala = metadata["sampler"] == "collapsed_mmala"
        blocked = is_nuts and metadata["nuts"].get("block_size") is not None
        if is_nuts:
            blocks = nuts_blocks(model_arrays[0].shape[0], metadata["nuts"].get("block_size"))
            if blocked and len(blocks) == 1:
                raise ValueError("Single-block checkpoints must use the all-site representation")
        tuning_names = (("step_size", *(f"inverse_mass_matrix.{i}" for i in range(len(blocks))))
                        if blocked else ("step_size", "inverse_mass_matrix")) if is_nuts else (
            ("epsilon", "epsilon_G") if is_mmala else ("V_prop",)
        )
        expected_tuning = {f"sampler.{name}" for name in tuning_names}
        if metadata["sampler"] == "collapsed_mala":
            expected_tuning.add("sampler.epsilon")
        actual_tuning = {name for name in arrays if name.startswith("sampler.")}
        if actual_tuning != expected_tuning:
            if metadata["sampler"] == "collapsed_random_walk" and "sampler.epsilon" in arrays:
                raise ValueError("Random-walk checkpoint cannot contain MALA tuning")
            raise ValueError("Checkpoint sampler/tuning payload does not match")
        tuning = [arrays[f"sampler.{name}"] for name in tuning_names]
        if any(value.dtype != np.float64 for value in (*model_arrays, *tuning)):
            raise ValueError("Checkpoint model and tuning arrays must be float64")
        key_data = arrays["rng.data"]
        if key_data.dtype != np.uint32 or key_data.ndim != 1:
            raise ValueError("Invalid checkpoint PRNG data")
        if not isinstance(metadata["rng_typed"], bool):
            raise ValueError("Invalid checkpoint PRNG representation")
        key = (
            jax.random.wrap_key_data(jnp.asarray(key_data), impl=metadata["rng_impl"])
            if metadata["rng_typed"] else jnp.asarray(key_data)
        )
        if str(jax.random.key_impl(key)) != metadata["rng_impl"]:
            raise ValueError("Checkpoint PRNG implementation does not match")
        state = CalibrationState(*(jnp.asarray(value) for value in model_arrays))
        if is_nuts:
            config = metadata["nuts"]
            if (not isinstance(config, dict) or config.get("integrator") != "velocity_verlet"
                or config.get("mass_matrix") not in ("diagonal", "dense", "kronecker")
                or config.get("coordinates") != "eta-v1"
                or config.get("block_order") != "contiguous-sites-v1"
                or config.get("theta_key_protocol") != ("split-blocks-v1" if blocked else "unsplit-v1")
                or metadata["step_size_adaptation"] is not None
                or any(name.startswith(("step_size.", "adaptation.")) for name in arrays)
                or "sampler.V_prop" in arrays or "sampler.epsilon" in arrays):
                raise ValueError("Invalid NUTS checkpoint configuration")
            adaptation = None
            am = metadata["adaptation"]
            if am is not None:
                protocol = "eta-blocked-factor-moments-v1" if blocked else "eta-staged-factor-moments-v2"
                if (am.get("protocol") != protocol
                    or am.get("buffers") != [75, 25, 50]
                    or am.get("imm_shrinkage_to_previous") != 0.0
                    or am.get("dual_averaging") != {"t0": 10, "gamma": 0.05, "kappa": 0.75}):
                    raise ValueError("Unsupported NUTS window configuration")
                if not np.array_equal(arrays["window.schedule"], build_schedule(am["num_warmup"])):
                    raise ValueError("NUTS checkpoint window schedule does not match")
                windows, expected_window = [], {"window.schedule"}
                for i in range(len(blocks)):
                    root = f"window.{i}" if blocked else "window"
                    moments_type = (KroneckerWelfordState if config["mass_matrix"] == "kronecker"
                                    else WelfordAlgorithmState)
                    for prefix, typ in (("ss", DualAveragingAdaptationState), ("wc", moments_type)):
                        expected_window.update(f"{root}.{prefix}.{name}" for name in typ._fields)
                    expected_window.update((f"{root}.step_size", f"{root}.inverse_mass_matrix"))
                    ss = DualAveragingAdaptationState(*(
                        jnp.asarray(arrays[f"{root}.ss.{name}"])
                        for name in DualAveragingAdaptationState._fields))
                    wc = moments_type(*(jnp.asarray(arrays[f"{root}.wc.{name}"])
                                        for name in moments_type._fields))
                    mass = jnp.asarray(arrays[f"{root}.inverse_mass_matrix"])
                    windows.append(StagedAdaptationState(
                        ss, MassMatrixAdaptationState(mass, wc),
                        jnp.asarray(arrays[f"{root}.step_size"]), mass))
                if {k for k in arrays if k.startswith("window.")} != expected_window:
                    raise ValueError("NUTS window payload does not match block layout")
                adaptation = NUTSAdaptationState(
                    am["num_warmup"], am["completed"], am["initial_step_size"], am["target_accept"],
                    tuple(windows) if blocked else windows[0])
            elif any(name.startswith("window.") for name in arrays):
                raise ValueError("NUTS window payload requires its schedule")
            if tuning[0].shape != ((len(blocks),) if blocked else ()):
                raise ValueError("NUTS step_size shape does not match block layout")
            chain = NUTSChain(
                state, key, jnp.asarray(tuning[0]) if blocked else float(tuning[0]),
                tuple(jnp.asarray(m) for m in tuning[1:]) if blocked else jnp.asarray(tuning[1]),
                metadata["iteration"], metadata["phase"], adaptation,
                config["max_num_doublings"], config["divergence_threshold"],
                metadata["sampler"] == "collapsed_nuts", config["mass_matrix"], config.get("block_size"),
            )
            return validate_nuts_chain(target, chain), metadata
        if any(name.startswith("window.") for name in arrays) or metadata["nuts"] is not None:
            raise ValueError("Non-NUTS checkpoint cannot contain NUTS configuration")
        V_prop = None if is_mmala else tuning[0]
        adaptation = None
        adaptation_metadata = metadata["adaptation"]
        if is_mmala:
            if ("sampler.V_prop" in arrays
                or any(name.startswith("adaptation.") for name in arrays)):
                raise ValueError("MMALA checkpoint cannot contain covariance adaptation")
            if adaptation_metadata is not None and (
                not isinstance(adaptation_metadata, dict)
                or adaptation_metadata.get("protocol") != "mmala-epsilon-only-v1"
            ):
                raise ValueError("Unsupported MMALA epsilon adaptation protocol")
        elif adaptation_metadata is not None:
            if (
                not isinstance(adaptation_metadata, dict)
                or adaptation_metadata.get("protocol") != "empirical-covariance-v2"
            ):
                raise ValueError("Unsupported checkpoint adaptation protocol")
            mean, m2, initial = (
                arrays["adaptation.mean"], arrays["adaptation.m2"],
                arrays["adaptation.initial_V_prop"],
            )
            if any(value.dtype != np.float64 for value in (mean, m2, initial)):
                raise ValueError("Checkpoint adaptation arrays must be float64")
            count = arrays["adaptation.sample_size"]
            if count.dtype.kind not in "iu":
                raise ValueError("Checkpoint adaptation counts must be integers")
            adaptation = RandomWalkAdaptationState(
                adaptation_metadata["num_warmup"], adaptation_metadata["num_initial"],
                jnp.asarray(initial), WelfordAlgorithmState(
                    jnp.asarray(mean), jnp.asarray(m2), jnp.asarray(count)
                ),
                *(jnp.asarray(arrays[f"adaptation.{name}"]) for name in
                  ("acceptance_count", "movement_count", "zero_covariance_count")),
            )
        elif any(name.startswith("adaptation.") for name in arrays):
            raise ValueError("Checkpoint adaptation payload requires its schedule")
        step_size_adaptation = None
        step_size_metadata = metadata["step_size_adaptation"]
        if step_size_metadata is not None:
            if (
                metadata["sampler"] not in ("collapsed_mala", "collapsed_mmala")
                or not isinstance(step_size_metadata, dict)
                or step_size_metadata.get("protocol") != (
                    "mmala-dual-averaging-v1" if is_mmala else "mala-dual-averaging-v1"
                )
            ):
                raise ValueError("Unsupported checkpoint step-size adaptation protocol")
            target_accept = arrays["step_size.target_accept"]
            initial_epsilon = arrays["step_size.initial_epsilon"]
            statistics = {name: arrays[f"step_size.{name}"]
                          for name in DualAveragingAdaptationState._fields}
            if any(value.shape != () or value.dtype != np.float64 for value in (
                target_accept, initial_epsilon,
                *(value for name, value in statistics.items() if name != "step"),
            )):
                raise ValueError(
                    "Checkpoint dual-averaging arrays must be float64 scalars"
                )
            if (
                statistics["step"].shape != ()
                or statistics["step"].dtype.kind not in "iu"
            ):
                raise ValueError(
                    "Checkpoint dual-averaging step must be an integer scalar"
                )
            step_size_adaptation = MALAStepSizeAdaptationState(
                float(target_accept), float(initial_epsilon),
                DualAveragingAdaptationState(**{
                    name: jnp.asarray(value) for name, value in statistics.items()
                }),
            )
        elif any(name.startswith("step_size.") for name in arrays):
            raise ValueError(
                "Checkpoint dual-averaging payload requires its configuration"
            )
        if is_mmala:
            if (adaptation_metadata is None) != (step_size_adaptation is None):
                raise ValueError("MMALA epsilon adaptation requires its schedule/statistics")
            if any(value.shape != () for value in tuning):
                raise ValueError("MMALA epsilon/ridge must be float64 scalars")
            if step_size_adaptation is not None:
                adaptation = MMALAAdaptationState(
                    adaptation_metadata["num_warmup"], step_size_adaptation
                )
            chain = MMALAChain(
                state, key, float(tuning[0]), float(tuning[1]),
                metadata["iteration"], metadata["phase"], adaptation,
            )
        elif metadata["sampler"] == "collapsed_mala":
            epsilon = arrays["sampler.epsilon"]
            if epsilon.shape != () or epsilon.dtype != np.float64:
                raise ValueError("Checkpoint epsilon must be a float64 scalar")
            chain = MALAChain(
                state, key, jnp.asarray(V_prop), float(epsilon),
                metadata["iteration"], metadata["phase"], adaptation,
                step_size_adaptation,
            )
        else:
            if "sampler.epsilon" in arrays:
                raise ValueError("Random-walk checkpoint cannot contain MALA tuning")
            chain = RandomWalkChain(
                state, key, jnp.asarray(V_prop), metadata["iteration"],
                metadata["phase"], adaptation,
            )
    except (KeyError, TypeError) as error:
        raise ValueError("Missing or invalid checkpoint state") from error
    validate = validate_mmala_chain if isinstance(chain, MMALAChain) else (
        validate_mala_chain if isinstance(chain, MALAChain) else validate_random_walk_chain
    )
    return validate(target, chain), metadata
