"""Complete calibration Gibbs sweeps, warmup, and frozen production."""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from blackjax.adaptation.step_size import dual_averaging_adaptation
from jax import Array
from jax.experimental import checkify

from bayesiancalibration.adaptation import (
    MALAStepSizeAdaptationState,
    RandomWalkAdaptationState,
    initialize_random_walk_adaptation,
    update_random_walk_adaptation,
    validate_random_walk_adaptation,
    validate_mala_step_size_adaptation,
    NUTSAdaptationState,
    nuts_window_adapter,
    validate_nuts_adaptation,
    MMALAAdaptationState,
)
from blackjax.adaptation.staged_adaptation import build_schedule
from bayesiancalibration.gibbs import (
    sample_branch_noise,
    sample_coefficient_variances,
    sample_discrepancy,
    sample_projected_coefficients,
    sample_spatial_covariance,
    sample_spatial_mean,
)
from bayesiancalibration.samplers.metropolis import (
    MALASweepInfo,
    RandomWalkSweepInfo,
    _validate_mala_tuning,
    collapsed_mala_sweep,
    collapsed_random_walk_sweep,
)
from bayesiancalibration.state import CalibrationState
from bayesiancalibration.samplers.nuts import NUTSSweepInfo, nuts_sweep
from bayesiancalibration.samplers.mmala import (
    collapsed_mmala_sweep, validate_mmala_geometry,
)
from bayesiancalibration.targets import CalibrationTarget


@dataclass(frozen=True)
class RandomWalkChain:
    """Restartable state, distinct from fixed target and returned diagnostics.

    model_state holds only model variables; key is the next unused scalar
    JAX key; V_prop (n,d,d) is tuning for the next sweep. iteration counts
    complete sweeps. adaptation holds separate warmup statistics/schedule;
    after warmup it remains frozen with phase='sampling'. Direct fixed-tuning
    initialization uses adaptation=None. Initialize through a checked factory.
    """

    model_state: CalibrationState
    key: Array
    V_prop: Array
    iteration: int = 0
    phase: str = "sampling"
    adaptation: RandomWalkAdaptationState | None = None


@dataclass(frozen=True)
class MALAChain:
    """Restartable MALA model/key/tuning with separate warmup statistics.

    V_prop (n,d,d) excludes epsilon^2. epsilon is a positive scalar for the
    next sweep; optional step_size_adaptation tunes it only during warmup.
    Reuse the site-wise Welford container/schedule;
    its MALA proposal multiplier is 1.0, rather than 2.38^2/d.
    No target densities or gradients are retained as inter-sweep caches.
    """

    model_state: CalibrationState
    key: Array
    V_prop: Array
    epsilon: float
    iteration: int = 0
    phase: str = "sampling"
    adaptation: RandomWalkAdaptationState | None = None
    step_size_adaptation: MALAStepSizeAdaptationState | None = None


class GibbsSweepInfo(NamedTuple):
    """Theta diagnostics and final full uncollapsed joint log density.

    The density is evaluated after the exact c_f refresh. No density or
    gradient here is reused as a cache in a subsequent sweep.
    """

    theta: RandomWalkSweepInfo | MALASweepInfo | NUTSSweepInfo
    full_joint_logdensity: Array
    site_moved: Array  # (n,), actual changes from the previous eta, not acceptance probabilities


class WarmupTuningError(ValueError):
    """A completed MH/MALA warmup cannot freeze a site with zero empirical scatter."""

    def __init__(self, adaptation):
        self.diagnostics = {
            "completed": adaptation.completed,
            "zero_covariance_sites": np.flatnonzero(np.all(
                np.asarray(adaptation.moments.m2) == 0, axis=(-2, -1)
            )).tolist(),
            **{name: np.asarray(getattr(adaptation, name)).tolist() for name in
               ("acceptance_count", "movement_count", "zero_covariance_count")},
        }
        super().__init__(f"Warmup failed: zero empirical covariance; {self.diagnostics}")


def collapsed_gibbs_sweep(
    key: Array,
    target: CalibrationTarget,
    state: CalibrationState,
    V_prop: Array,
    *,
    epsilon: Array | None = None,
    _theta_transition=None,
) -> tuple[CalibrationState, Array, GibbsSweepInfo]:
    """Pure float64 outer sweep; close over target for JIT/vmap/scan.

    Uses the specified delta, noise, mean, covariance, coefficient-variance,
    collapsed-theta, exact-coefficient order. This custom model schedule
    composes the already validated standard-function conditional kernels.
    Every update uses the newest available conditioning variables. Refresh
    c_f immediately after theta, even when every site proposal is rejected.

    Eight subkeys are split once: next unused key, then seven update keys in
    schedule order. This split-8-v1 protocol is part of checkpoint metadata.
    Inputs must be valid; checked orchestration validates outside JIT.
    epsilon=None selects random walk. A declared positive scalar epsilon
    selects MALA with V_prop as its unscaled covariance preconditioner.
    This static choice changes only the collapsed theta transition.
    The internal _theta_transition hook embeds other theta kernels in the
    same model-specific schedule; it receives freshly updated conditioning.
    """

    next_key, kd, ky, km, kS, kc, kt, kf = jax.random.split(key, 8)
    theta_tilde = target.coordinates.eta_to_theta_tilde(state.eta)
    _, Omega_y = target.observation_arrays(state.delta, state.sigma_y2)
    delta = sample_discrepancy(
        kd, target.y_tilde, state.c_f, target.R, Omega_y,
        target.m_delta_0, target.V_delta_0,
    )
    sigma_y2 = sample_branch_noise(
        ky, target.y_tilde, state.c_f, delta, target.R, target.branch_sizes,
        target.alpha_y_0, target.beta_y_0,
    )
    mu_theta = sample_spatial_mean(
        km, theta_tilde, target.C_theta, state.Sigma_theta,
        target.spatial_prior.m_theta_0, target.spatial_prior.V_theta_0,
    )
    Sigma_theta = sample_spatial_covariance(
        kS, theta_tilde, target.C_theta, mu_theta,
        target.spatial_prior.nu_theta_0, target.spatial_prior.S_theta_0,
    )
    r, k = target.gp.F_s.shape
    m_f_given_s, C_f_given_s, _ = target.gp.conditional_moments(
        theta_tilde, jnp.ones(k, dtype=jnp.float64)
    )
    sigma_c2 = sample_coefficient_variances(
        kc, state.c_f, m_f_given_s, C_f_given_s, target.gp.q_s, r,
        target.branch_sizes, target.alpha_c_0, target.beta_c_0,
    )
    if _theta_transition is not None:
        conditioned = CalibrationState(
            state.eta, state.c_f, delta, sigma_y2, mu_theta, Sigma_theta, sigma_c2
        )
        eta, theta_info = _theta_transition(kt, conditioned)
    elif epsilon is None:
        eta, theta_info = collapsed_random_walk_sweep(
            kt, target, state.eta, delta, sigma_y2, mu_theta, Sigma_theta,
            sigma_c2, V_prop,
        )
    else:
        eta, theta_info = collapsed_mala_sweep(
            kt, target, state.eta, delta, sigma_y2, mu_theta, Sigma_theta,
            sigma_c2, V_prop, epsilon,
        )
    m_f_given_s, _, Sigma_f_given_s = target.gp.conditional_moments(
        target.coordinates.eta_to_theta_tilde(eta), sigma_c2
    )
    d_delta, Omega_y = target.observation_arrays(delta, sigma_y2)
    c_f = sample_projected_coefficients(
        kf, target.y_tilde, m_f_given_s, Sigma_f_given_s,
        target.R, d_delta, Omega_y,
    )
    new_state = CalibrationState(
        eta, c_f, delta, sigma_y2, mu_theta, Sigma_theta, sigma_c2
    )
    joint = target.full_joint_uncollapsed(*new_state)
    # Device reductions only: do not repeat factorizations or evaluate another density.
    checkify.debug_check(
        jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in new_state]))
        & jnp.all(sigma_y2 > 0) & jnp.all(sigma_c2 > 0),
        "Gibbs sweep produced invalid model state",
    )
    checkify.debug_check(
        jnp.isfinite(joint) & jnp.isfinite(theta_info.logdensity)
        & jnp.all(jnp.isfinite(theta_info.acceptance_rate)),
        "Gibbs sweep produced nonfinite diagnostics",
    )
    return new_state, next_key, GibbsSweepInfo(theta_info, joint, jnp.any(eta != state.eta, axis=1))


def _checked_call(kernel, *args):
    """Surface JAX's functional errors before advancing the host chain boundary."""
    error, result = kernel(*args)
    message = error.get()
    if message is not None:
        raise FloatingPointError(message)
    return result


def _check_chain_control(chain, chain_type):
    """Only cheap run-control invariants; arrays/configuration were initialized once."""
    if type(chain) is not chain_type:
        raise ValueError(f"Expected {chain_type.__name__}")
    if (isinstance(chain.iteration, bool) or not isinstance(chain.iteration, Integral)
        or chain.iteration < 0 or chain.phase not in ("warmup", "sampling")):
        raise ValueError("Invalid chain iteration/phase")
    adapt = chain.adaptation
    if chain.phase == "warmup":
        if adapt is None or not chain.iteration == adapt.completed < adapt.num_warmup:
            raise ValueError("Warmup schedule/counter mismatch")
    elif adapt is not None and (
        adapt.completed != adapt.num_warmup or chain.iteration < adapt.completed
    ):
        raise ValueError("Production requires completed warmup")


def _checked_adaptation(function):
    """Check only updated numbers; never rebuild schedules or validate old tuning."""
    def update(*args):
        result = function(*args)
        checkify.debug_check(
            jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(result)])),
            "Warmup produced nonfinite adaptation state",
        )
        return result
    return jax.jit(checkify.checkify(update))


def _check_diffusion(epsilon, proposal=1.0):
    """Reject newly adapted epsilon overflow/underflow without another factorization."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        step = np.square(epsilon) / 2
        diffusion = 2 * step * np.asarray(proposal)
    diagonal = np.diagonal(diffusion, axis1=-2, axis2=-1) if diffusion.ndim else diffusion
    if (not np.isfinite(step) or step <= 0 or not np.all(np.isfinite(diffusion))
        or np.any(diagonal <= 0)):
        raise FloatingPointError("Warmup produced invalid epsilon/diffusion")


def validate_random_walk_chain(
    target: CalibrationTarget, chain: RandomWalkChain
) -> RandomWalkChain:
    """Validate and normalize a completed chain boundary outside JIT.

    A model-specific boundary checker is needed because array libraries do
    not know the model's shapes, positive variances, or GP geometry. Used at
    initialization, before/after checked runs, and when saving/loading state.
    No clipping, jitter, repair, or random draws occur during validation.
    """

    if not isinstance(chain, RandomWalkChain):
        raise ValueError("chain must be a RandomWalkChain")
    if not jax.config.jax_enable_x64:
        raise ValueError("Gibbs sampling requires jax_enable_x64=True")
    if (
        isinstance(chain.iteration, bool)
        or not isinstance(chain.iteration, Integral) or chain.iteration < 0
    ):
        raise ValueError("iteration must be a nonnegative integer")
    if chain.phase not in ("warmup", "sampling"):
        raise ValueError("phase must be 'warmup' or 'sampling'")
    if chain.phase == "warmup" and chain.adaptation is None:
        raise ValueError("Warmup phase requires adaptation state")
    try:
        key_data = jax.random.key_data(chain.key)
        jax.random.split(chain.key, 2)
    except (TypeError, ValueError) as error:
        raise ValueError("key must be a scalar JAX PRNG key") from error
    if key_data.ndim != 1:
        raise ValueError("key must be a scalar JAX PRNG key")
    if not isinstance(chain.model_state, CalibrationState):
        raise ValueError("model_state must be a CalibrationState")
    n = target.C_theta.shape[0]
    k = target.gp.F_s.shape[1]
    d = target.gp.theta_s_tilde.shape[1]
    shapes = ((n, d), (n*k,), (k,), (len(target.branch_sizes),),
              (d,), (d, d), (k,))
    checked = []
    for name, value, shape in zip(CalibrationState._fields, chain.model_state, shapes):
        value_np = np.asarray(value, dtype=np.float64)
        if value_np.shape != shape or not np.all(np.isfinite(value_np)):
            raise ValueError(f"{name} must be finite with shape {shape}")
        if name in ("sigma_y2", "sigma_c2") and np.any(value_np <= 0):
            raise ValueError(f"{name} must be positive")
        checked.append(jnp.asarray(value_np))
    state = CalibrationState(*checked)
    for name, value, shape in (
        ("Sigma_theta", state.Sigma_theta, (d, d)),
        ("V_prop", chain.V_prop, (n, d, d)),
    ):
        matrix = np.asarray(value, dtype=np.float64)
        if matrix.shape != shape or not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name} must be finite with shape {shape}")
        if not np.array_equal(matrix, np.swapaxes(matrix, -1, -2)):
            raise ValueError(f"{name} must be symmetric")
        try:
            np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError as error:
            raise ValueError(f"{name} must be positive definite") from error
    adaptation = chain.adaptation
    if adaptation is not None:
        validate_random_walk_adaptation(adaptation, n, d)
        completed = adaptation.completed
        if chain.phase == "warmup":
            if completed >= adaptation.num_warmup or chain.iteration != completed:
                raise ValueError("Warmup phase/counters disagree with its schedule")
        elif completed != adaptation.num_warmup or chain.iteration < completed:
            raise ValueError("Sampling requires a completed warmup schedule")
        elif chain.phase == "sampling" and np.any(np.all(
            np.asarray(adaptation.moments.m2) == 0, axis=(-2, -1)
        )):
            raise WarmupTuningError(adaptation)
        if (
            completed < adaptation.num_initial
            or adaptation.num_initial == adaptation.num_warmup
        ) and not np.array_equal(chain.V_prop, adaptation.initial_V_prop):
            raise ValueError("The initial fixed period must retain initial_V_prop")
    target.gp.validate_field_sites(target.coordinates.eta_to_theta_tilde(state.eta))
    if not np.isfinite(float(target.full_joint_uncollapsed(*state))):
        raise ValueError("Full joint log density must be finite at a chain boundary")
    return RandomWalkChain(
        state, chain.key, jnp.asarray(chain.V_prop, dtype=jnp.float64),
        int(chain.iteration), chain.phase, adaptation,
    )


def initialize_random_walk_chain(
    target: CalibrationTarget, state: CalibrationState, key: Array, V_prop: Array
) -> RandomWalkChain:
    """Validate a supplied initial complete state and fixed proposal tuning."""

    return validate_random_walk_chain(target, RandomWalkChain(state, key, V_prop))


def initialize_random_walk_warmup(
    target: CalibrationTarget,
    state: CalibrationState,
    key: Array,
    *,
    num_warmup: int,
    num_initial: int,
    V_prop: Array | None = None,
) -> RandomWalkChain:
    """Initialize explicitly scheduled warmup; identity is the default tuning.

    Both lengths must be declared, with 1 <= num_initial <= num_warmup.
    V_prop is an optional declared SPD (n,d,d) initial covariance. Covariance
    estimation uses completed warmup states, excluding the initial position.
    """

    n = target.C_theta.shape[0]
    d = target.gp.theta_s_tilde.shape[1]
    if V_prop is None:
        V_prop = jnp.tile(jnp.eye(d, dtype=jnp.float64), (n, 1, 1))
    chain = initialize_random_walk_chain(target, state, key, V_prop)
    adaptation = initialize_random_walk_adaptation(
        num_warmup, num_initial, chain.V_prop
    )
    return validate_random_walk_chain(
        target, replace(chain, phase="warmup", adaptation=adaptation)
    )


def _run_metropolis_chunk(
    target: CalibrationTarget, chain: RandomWalkChain | MALAChain, num_sweeps: int,
    *, kernels=None,
) -> tuple[RandomWalkChain | MALAChain, CalibrationState, GibbsSweepInfo]:
    """Share complete-sweep recording/failure logic across both phases.

    This model-specific orchestration keeps PRNG, coefficient refresh, and
    validation identical in warmup/production. Proposal adaptation occurs
    only after a completed warmup sweep, for use in the next sweep.
    Public drivers prevent chunks from mixing warmup and retained draws.
    """

    is_mala = isinstance(chain, MALAChain)
    if kernels is None:
        kernels = ChunkRunner(target, chain).kernels
    kernel, adapt_kernel = kernels["sweep"], kernels["moments"]
    step_size_adaptation = chain.step_size_adaptation if is_mala else None
    if step_size_adaptation is not None:
        da_update, da_final = kernels["update"], kernels["final"]
    samples, diagnostics = [], []
    for _ in range(num_sweeps):
        epsilon = chain.epsilon if is_mala else None
        state, next_key, info = _checked_call(kernel,
            chain.key, chain.model_state, chain.V_prop, epsilon
        )
        proposal, adaptation, phase = chain.V_prop, chain.adaptation, chain.phase
        if phase == "warmup":
            completed = adaptation.completed + 1
            eligible = (
                completed >= adaptation.num_initial
                and adaptation.num_initial < adaptation.num_warmup
            )
            moments, proposal = _checked_call(adapt_kernel,
                adaptation.moments, state.eta, proposal, jnp.asarray(eligible)
            )
            zero = jnp.all(moments.m2 == 0, axis=(-2, -1))
            adaptation = replace(
                adaptation, moments=moments,
                acceptance_count=adaptation.acceptance_count + info.theta.is_accepted,
                movement_count=adaptation.movement_count + info.site_moved,
                zero_covariance_count=adaptation.zero_covariance_count
                + (zero & eligible & (completed > 1)),
            )
            if completed == adaptation.num_warmup:
                if np.any(np.asarray(zero)):
                    raise WarmupTuningError(adaptation)
                phase = "sampling"
            if step_size_adaptation is not None:
                da_state = _checked_call(da_update,
                    step_size_adaptation.state, jnp.mean(info.theta.acceptance_rate)
                )
                step_size_adaptation = replace(step_size_adaptation, state=da_state)
                epsilon = float(
                    da_final(da_state) if phase == "sampling"
                    else jnp.exp(da_state.log_step_size)
                )
            if is_mala:
                _check_diffusion(epsilon, proposal)
        candidate = replace(
            chain, model_state=state, key=next_key, V_prop=proposal,
            iteration=chain.iteration + 1, phase=phase, adaptation=adaptation,
        )
        if is_mala:
            candidate = replace(
                candidate, epsilon=epsilon, step_size_adaptation=step_size_adaptation
            )
        chain = candidate
        samples.append(chain.model_state)
        diagnostics.append(info)
    target.gp.validate_field_sites(target.coordinates.eta_to_theta_tilde(chain.model_state.eta))
    return (
        chain,
        jax.tree.map(lambda *values: jnp.stack(values), *samples),
        jax.tree.map(lambda *values: jnp.stack(values), *diagnostics),
    )


def run_fixed_random_walk(
    target: CalibrationTarget, chain: RandomWalkChain, num_sweeps: int
) -> tuple[RandomWalkChain, CalibrationState, GibbsSweepInfo]:
    """Run retained complete sweeps with frozen proposal tuning/statistics.

    Requires phase='sampling'. Samples/diagnostics have a leading num_sweeps
    axis. Rejected theta states repeat and c_f refreshes on every sweep.
    The input chain stays unchanged; validation/I/O stay outside JIT.
    """

    if (
        isinstance(num_sweeps, bool) or not isinstance(num_sweeps, Integral)
        or num_sweeps < 1
    ):
        raise ValueError("num_sweeps must be a positive integer")
    _check_chain_control(chain, RandomWalkChain)
    if chain.phase != "sampling":
        raise ValueError("Complete warmup before retained sampling")
    return _run_metropolis_chunk(target, chain, num_sweeps)


def run_random_walk_warmup(
    target: CalibrationTarget,
    chain: RandomWalkChain,
    num_sweeps: int | None = None,
) -> tuple[RandomWalkChain, CalibrationState, GibbsSweepInfo]:
    """Run a warmup chunk, optionally finishing all remaining warmup sweeps.

    Returns warmup history separately from retained production draws. Each
    full sweep uses the previous proposal; its completed eta updates online
    moments including rejections. The final warmup sweep freezes the resulting
    proposal/statistics and sets phase='sampling' without consuming an extra
    key or production draw. Chunks cannot overrun the declared boundary.
    """

    _check_chain_control(chain, RandomWalkChain)
    if chain.phase != "warmup":
        raise ValueError("Warmup requires phase='warmup'")
    remaining = chain.adaptation.num_warmup - chain.adaptation.completed
    if num_sweeps is None:
        num_sweeps = remaining
    if (
        isinstance(num_sweeps, bool) or not isinstance(num_sweeps, Integral)
        or num_sweeps < 1 or num_sweeps > remaining
    ):
        raise ValueError(
            "Warmup chunk must be positive and not exceed remaining sweeps"
        )
    return _run_metropolis_chunk(target, chain, num_sweeps)


def validate_mala_chain(target: CalibrationTarget, chain: MALAChain) -> MALAChain:
    """Reuse model/phase checks, then validate MALA diffusion and gradients.

    These additional checks are required by the gradient-driven proposal;
    finite densities alone do not guarantee usable MALA initialization.
    """

    if not isinstance(chain, MALAChain):
        raise ValueError("chain must be a MALAChain")
    checked = validate_random_walk_chain(target, RandomWalkChain(
        chain.model_state, chain.key, chain.V_prop, chain.iteration,
        chain.phase, chain.adaptation,
    ))
    epsilon = _validate_mala_tuning(chain.epsilon, checked.V_prop)
    step_size_adaptation = chain.step_size_adaptation
    if step_size_adaptation is not None:
        if checked.adaptation is None:
            raise ValueError("Dual averaging requires a declared warmup schedule")
        completed = checked.adaptation.completed
        validate_mala_step_size_adaptation(step_size_adaptation, completed)
        _validate_mala_tuning(
            step_size_adaptation.initial_epsilon, checked.adaptation.initial_V_prop
        )
        if completed == 0:
            expected_epsilon = step_size_adaptation.initial_epsilon
        elif checked.phase == "warmup":
            expected_epsilon = jnp.exp(step_size_adaptation.state.log_step_size)
        else:
            _, _, final = dual_averaging_adaptation(step_size_adaptation.target_accept)
            expected_epsilon = final(step_size_adaptation.state)
        if epsilon != float(expected_epsilon):
            raise ValueError("epsilon does not match its dual-averaging phase/state")
    state = checked.model_state
    gradient = jax.grad(lambda eta: target.theta_only_collapsed(
        eta, state.delta, state.sigma_y2, state.mu_theta, state.Sigma_theta,
        state.sigma_c2,
    ))(state.eta)
    if not np.all(np.isfinite(np.asarray(gradient))):
        raise ValueError("Current collapsed gradient must be finite")
    return MALAChain(
        checked.model_state, checked.key, checked.V_prop, epsilon,
        checked.iteration, checked.phase, checked.adaptation, step_size_adaptation,
    )


def initialize_mala_chain(
    target: CalibrationTarget, state: CalibrationState, key: Array,
    V_prop: Array, epsilon: float,
) -> MALAChain:
    """Validate a complete initial state and declared fixed MALA tuning."""

    return validate_mala_chain(target, MALAChain(state, key, V_prop, epsilon))


def initialize_mala_warmup(
    target: CalibrationTarget, state: CalibrationState, key: Array,
    *, num_warmup: int, num_initial: int, epsilon: float,
    V_prop: Array | None = None,
    target_accept: float | None = 0.574,
) -> MALAChain:
    """Use identity/default SPD tuning and the declared site covariance schedule.

    After the initial period, V_prop is exactly D07's empirical preconditioner
    without random-walk scaling. target_accept in (0,1) enables standard
    BlackJAX dual averaging of epsilon from mean site acceptance probabilities,
    once per completed sweep, including the initial covariance period.
    The default target is 0.574; explicitly use None for fixed epsilon.
    The final averaged epsilon is frozen in production.
    """

    initial = initialize_random_walk_warmup(
        target, state, key, num_warmup=num_warmup, num_initial=num_initial,
        V_prop=V_prop,
    )
    chain = validate_mala_chain(target, MALAChain(
        initial.model_state, initial.key, initial.V_prop, epsilon,
        initial.iteration, initial.phase, initial.adaptation,
    ))
    if target_accept is not None:
        value = np.asarray(target_accept)
        if (
            value.shape != () or value.dtype.kind not in "fiu"
            or not np.isfinite(value) or not 0 < value < 1
        ):
            raise ValueError("target_accept must be a finite scalar in (0,1)")
        da_init, _, _ = dual_averaging_adaptation(float(value))
        state = jax.tree.map(jnp.asarray, da_init(chain.epsilon))
        step_size_adaptation = MALAStepSizeAdaptationState(
            float(value), chain.epsilon, state
        )
        chain = replace(chain, step_size_adaptation=step_size_adaptation)
    return validate_mala_chain(target, chain)


def run_fixed_mala(
    target: CalibrationTarget, chain: MALAChain, num_sweeps: int
) -> tuple[MALAChain, CalibrationState, GibbsSweepInfo]:
    """Run production Gibbs sweeps with fixed epsilon/preconditioners/statistics."""

    if (
        isinstance(num_sweeps, bool) or not isinstance(num_sweeps, Integral)
        or num_sweeps < 1
    ):
        raise ValueError("num_sweeps must be a positive integer")
    _check_chain_control(chain, MALAChain)
    if chain.phase != "sampling":
        raise ValueError("Complete warmup before retained sampling")
    return _run_metropolis_chunk(target, chain, num_sweeps)


def run_mala_warmup(
    target: CalibrationTarget, chain: MALAChain, num_sweeps: int | None = None
) -> tuple[MALAChain, CalibrationState, GibbsSweepInfo]:
    """Run only warmup sweeps; freeze tuning/statistics at the sampling boundary."""

    _check_chain_control(chain, MALAChain)
    if chain.phase != "warmup":
        raise ValueError("Warmup requires phase='warmup'")
    remaining = chain.adaptation.num_warmup - chain.adaptation.completed
    if num_sweeps is None:
        num_sweeps = remaining
    if (
        isinstance(num_sweeps, bool) or not isinstance(num_sweeps, Integral)
        or num_sweeps < 1 or num_sweeps > remaining
    ):
        raise ValueError(
            "Warmup chunk must be positive and not exceed remaining sweeps"
        )
    return _run_metropolis_chunk(target, chain, num_sweeps)


@dataclass(frozen=True)
class NUTSChain:
    """Complete model/key boundary and diagonal all-site NUTS tuning.

    step_size is the library integrator scale; inverse_mass_matrix (n*d,)
    uses BlackJAX's convention. Window statistics are separate and freeze
    in production. Tree/energy limits are explicit computational settings.
    No HMC position-density-gradient cache crosses an outer sweep boundary.
    """

    model_state: CalibrationState
    key: Array
    step_size: float
    inverse_mass_matrix: Array
    iteration: int = 0
    phase: str = "sampling"
    adaptation: NUTSAdaptationState | None = None
    max_num_doublings: int = 10
    divergence_threshold: float = 1000.0
    collapsed: bool = False


def nuts_gibbs_sweep(
    key: Array, target: CalibrationTarget, state: CalibrationState,
    step_size: Array, inverse_mass_matrix: Array,
    *, max_num_doublings: int = 10, divergence_threshold: float = 1000,
    collapsed: bool = False,
) -> tuple[CalibrationState, Array, GibbsSweepInfo]:
    """Use the common outer schedule with matched all-site NUTS kernels.

    The current c_f stays fixed throughout theta; its exact conditional draw
    follows immediately. All other Gibbs updates and split-8-v1 keys match
    the collapsed methods. The private transition hook shares this specified
    model schedule without duplicating coefficient-dependent update logic.
    """

    def transition(theta_key, conditioned):
        return nuts_sweep(
            theta_key, target, conditioned, step_size, inverse_mass_matrix,
            max_num_doublings=max_num_doublings,
            divergence_threshold=divergence_threshold,
            collapsed=collapsed,
        )
    return collapsed_gibbs_sweep(key, target, state, None, _theta_transition=transition)


def validate_nuts_chain(target: CalibrationTarget, chain: NUTSChain) -> NUTSChain:
    """Validate model, finite gradient, tuning and standard-window phase.

    Reuse existing model/PRNG/GP checks with identity site tuning; NUTS has
    its own diagonal all-site mass and window counters. Validation does not
    consume randomness, clip tuning, or retain target gradients.
    """

    if not isinstance(chain, NUTSChain):
        raise ValueError("chain must be a NUTSChain")
    if not isinstance(chain.model_state, CalibrationState):
        raise ValueError("model_state must be a CalibrationState")
    if not isinstance(chain.collapsed, bool):
        raise ValueError("NUTS collapsed must be a boolean target choice")
    n, d = chain.model_state.eta.shape
    identity = jnp.tile(jnp.eye(d, dtype=jnp.float64), (n, 1, 1))
    checked = validate_random_walk_chain(target, RandomWalkChain(
        chain.model_state, chain.key, identity, chain.iteration
    ))
    for name in ("step_size", "divergence_threshold"):
        value = np.asarray(getattr(chain, name))
        if (value.shape != () or value.dtype.kind not in "fiu"
            or not np.isfinite(value) or value <= 0):
            raise ValueError(f"NUTS {name} must be a finite positive scalar")
    if (isinstance(chain.max_num_doublings, bool)
        or not isinstance(chain.max_num_doublings, Integral)
        or chain.max_num_doublings < 1):
        raise ValueError("max_num_doublings must be a positive integer")
    mass = np.asarray(chain.inverse_mass_matrix)
    if mass.shape != (n*d,) or not np.all(np.isfinite(mass)) or np.any(mass <= 0):
        raise ValueError("NUTS inverse_mass_matrix must be positive diagonal (n*d,)")
    if chain.phase not in ("warmup", "sampling"):
        raise ValueError("Invalid NUTS phase")
    adapt = chain.adaptation
    if adapt is None:
        if chain.phase != "sampling":
            raise ValueError("NUTS warmup requires a window schedule")
    else:
        validate_nuts_adaptation(adapt, n*d)
        if chain.phase == "warmup":
            if adapt.completed >= adapt.num_warmup or chain.iteration != adapt.completed:
                raise ValueError("NUTS warmup phase/count mismatch")
            expected_step, expected_mass = adapt.state.step_size, adapt.state.inverse_mass_matrix
        else:
            if adapt.completed != adapt.num_warmup or chain.iteration < adapt.completed:
                raise ValueError("NUTS production requires completed window adaptation")
            _, _, final = nuts_window_adapter(adapt.target_accept)
            expected_step, expected_mass = final(adapt.state)
        if float(chain.step_size) != float(expected_step) or not np.array_equal(
            mass, expected_mass
        ):
            raise ValueError("NUTS tuning disagrees with its window phase/state")
    s = checked.model_state
    def density(eta):
        if chain.collapsed:
            return target.theta_only_collapsed(
                eta, s.delta, s.sigma_y2, s.mu_theta, s.Sigma_theta, s.sigma_c2
            )
        return target.theta_only_uncollapsed(
            eta, s.c_f, s.mu_theta, s.Sigma_theta, s.sigma_c2
        )
    grad = jax.grad(density)(s.eta)
    if not np.all(np.isfinite(np.asarray(grad))):
        raise ValueError("Current NUTS gradient must be finite")
    if adapt is not None:
        adapt = replace(adapt, num_warmup=int(adapt.num_warmup), completed=int(adapt.completed),
                        initial_step_size=float(adapt.initial_step_size),
                        target_accept=float(adapt.target_accept))
    return replace(chain, model_state=s, step_size=float(chain.step_size),
                   inverse_mass_matrix=jnp.asarray(mass, dtype=jnp.float64), adaptation=adapt,
                   iteration=checked.iteration, max_num_doublings=int(chain.max_num_doublings),
                   divergence_threshold=float(chain.divergence_threshold))


def initialize_nuts_chain(
    target: CalibrationTarget, state: CalibrationState, key: Array,
    *, step_size: float, inverse_mass_matrix: Array,
    max_num_doublings: int = 10, divergence_threshold: float = 1000,
    collapsed: bool = False,
) -> NUTSChain:
    """Initialize checked fixed NUTS tuning; select collapse explicitly."""

    return validate_nuts_chain(target, NUTSChain(
        state, key, step_size, inverse_mass_matrix,
        max_num_doublings=max_num_doublings, divergence_threshold=divergence_threshold,
        collapsed=collapsed,
    ))


def initialize_nuts_warmup(
    target: CalibrationTarget, state: CalibrationState, key: Array,
    *, num_warmup: int = 1000, initial_step_size: float = 1.0,
    target_accept: float = 0.8, max_num_doublings: int = 10,
    divergence_threshold: float = 1000,
    collapsed: bool = False,
) -> NUTSChain:
    """Resolve the standard diagonal window-adaptation defaults explicitly.

    Standard schedule buffers are 75/25/50, with BlackJAX short-run handling.
    Initial inverse mass is identity; no empirical-covariance MALA/RW rule
    is reused. A warmup step is one complete Gibbs sweep, with acceptance
    feedback from the NUTS trajectory and current conditional each time.
    """

    init, _, _ = nuts_window_adapter(target_accept)
    adaptation = NUTSAdaptationState(
        num_warmup, 0, initial_step_size, target_accept,
        jax.tree.map(jnp.asarray, init(state.eta.reshape(-1), initial_step_size)),
    )
    chain = NUTSChain(
        state, key, initial_step_size, adaptation.state.inverse_mass_matrix,
        phase="warmup", adaptation=adaptation, max_num_doublings=max_num_doublings,
        divergence_threshold=divergence_threshold,
        collapsed=collapsed,
    )
    return validate_nuts_chain(target, chain)


def _run_nuts_chunk(target, chain, num_sweeps, *, kernels=None):
    """Keep full-sweep recording and standard adaptation on the same clock."""

    if kernels is None:
        kernels = ChunkRunner(target, chain).kernels
    kernel = kernels["sweep"]
    if chain.adaptation is not None:
        update, final, schedule = kernels["update"], kernels["final"], kernels["schedule"]
    samples, diagnostics = [], []
    for _ in range(num_sweeps):
        state, key, info = _checked_call(kernel,
            chain.key, chain.model_state, chain.step_size, chain.inverse_mass_matrix
        )
        step, mass, adapt, phase = (
            chain.step_size, chain.inverse_mass_matrix, chain.adaptation, chain.phase
        )
        if phase == "warmup":
            # The standard Welford recipe ignores grad. Supplying zero avoids
            # recomputing a gradient under the newly refreshed c_f conditional.
            updated = _checked_call(update,
                adapt.state, schedule[adapt.completed], state.eta.reshape(-1),
                jnp.zeros(state.eta.size, dtype=jnp.float64), info.theta.acceptance_rate,
            )
            adapt = replace(adapt, state=updated, completed=adapt.completed + 1)
            if adapt.completed == adapt.num_warmup:
                phase = "sampling"
                step, mass = final(updated)
            else:
                step, mass = updated.step_size, updated.inverse_mass_matrix
            if not np.isfinite(float(step)) or float(step) <= 0 or np.any(np.asarray(mass) <= 0):
                raise FloatingPointError("Warmup produced invalid NUTS step/mass")
        chain = replace(
            chain, model_state=state, key=key, step_size=float(step),
            inverse_mass_matrix=mass, adaptation=adapt, phase=phase,
            iteration=chain.iteration + 1,
        )
        samples.append(chain.model_state)
        diagnostics.append(info)
    target.gp.validate_field_sites(target.coordinates.eta_to_theta_tilde(chain.model_state.eta))
    return (chain, jax.tree.map(lambda *x: jnp.stack(x), *samples),
            jax.tree.map(lambda *x: jnp.stack(x), *diagnostics))


def run_nuts_warmup(
    target: CalibrationTarget, chain: NUTSChain, num_sweeps: int | None = None,
) -> tuple[NUTSChain, CalibrationState, GibbsSweepInfo]:
    """Run a warmup-only chunk; freeze final average step size and mass."""

    _check_chain_control(chain, NUTSChain)
    if chain.phase != "warmup":
        raise ValueError("NUTS warmup requires phase='warmup'")
    remaining = chain.adaptation.num_warmup - chain.adaptation.completed
    if num_sweeps is None:
        num_sweeps = remaining
    if (isinstance(num_sweeps, bool) or not isinstance(num_sweeps, Integral)
        or not 1 <= num_sweeps <= remaining):
        raise ValueError("NUTS warmup chunk must fit remaining schedule")
    return _run_nuts_chunk(target, chain, num_sweeps)


def run_fixed_nuts(
    target: CalibrationTarget, chain: NUTSChain, num_sweeps: int,
) -> tuple[NUTSChain, CalibrationState, GibbsSweepInfo]:
    """Retain complete Gibbs draws with all NUTS tuning/statistics frozen."""

    _check_chain_control(chain, NUTSChain)
    if chain.phase != "sampling":
        raise ValueError("Complete NUTS warmup before retained sampling")
    if (isinstance(num_sweeps, bool) or not isinstance(num_sweeps, Integral)
        or num_sweeps < 1):
        raise ValueError("num_sweeps must be a positive integer")
    return _run_nuts_chunk(target, chain, num_sweeps)


@dataclass(frozen=True)
class MMALAChain:
    """Complete model/key boundary with epsilon and declared metric ridge.

    Position-dependent G_i is always recomputed, including in production.
    Only scalar epsilon is adapted; epsilon_G stays fixed. No empirical
    covariance or changing density, gradient or metric cache is retained.
    """

    model_state: CalibrationState
    key: Array
    epsilon: float
    epsilon_G: float
    iteration: int = 0
    phase: str = "sampling"
    adaptation: MMALAAdaptationState | None = None


def mmala_gibbs_sweep(
    key: Array, target: CalibrationTarget, state: CalibrationState,
    epsilon: Array, epsilon_G: Array,
) -> tuple[CalibrationState, Array, GibbsSweepInfo]:
    """Embed simplified collapsed MMALA in the shared split-8-v1 schedule."""

    def transition(theta_key, conditioned):
        s = conditioned
        return collapsed_mmala_sweep(
            theta_key, target, s.eta, s.delta, s.sigma_y2, s.mu_theta,
            s.Sigma_theta, s.sigma_c2, epsilon, epsilon_G,
        )
    return collapsed_gibbs_sweep(key, target, state, None, _theta_transition=transition)


def validate_mmala_chain(target: CalibrationTarget, chain: MMALAChain) -> MMALAChain:
    """Check complete state/support plus MMALA geometry and epsilon phase."""

    if not isinstance(chain, MMALAChain):
        raise ValueError("chain must be an MMALAChain")
    if not isinstance(chain.model_state, CalibrationState):
        raise ValueError("model_state must be a CalibrationState")
    n, d = chain.model_state.eta.shape
    identity = jnp.tile(jnp.eye(d, dtype=jnp.float64), (n, 1, 1))
    checked = validate_random_walk_chain(target, RandomWalkChain(
        chain.model_state, chain.key, identity, chain.iteration
    ))
    s = checked.model_state
    epsilon, ridge = validate_mmala_geometry(
        target, s.eta, s.delta, s.sigma_y2, s.mu_theta, s.Sigma_theta,
        s.sigma_c2, chain.epsilon, chain.epsilon_G,
    )
    if chain.phase not in ("warmup", "sampling"):
        raise ValueError("Invalid MMALA phase")
    adapt = chain.adaptation
    if adapt is None:
        if chain.phase != "sampling":
            raise ValueError("MMALA warmup requires an epsilon schedule")
    else:
        if not isinstance(adapt, MMALAAdaptationState):
            raise ValueError("MMALA requires epsilon-only adaptation")
        if (isinstance(adapt.num_warmup, bool) or not isinstance(adapt.num_warmup, Integral)
            or adapt.num_warmup < 1 or not 0 <= adapt.completed <= adapt.num_warmup):
            raise ValueError("Invalid MMALA warmup length/count")
        validate_mala_step_size_adaptation(adapt.step_size, adapt.completed)
        if chain.phase == "warmup":
            if adapt.completed >= adapt.num_warmup or chain.iteration != adapt.completed:
                raise ValueError("MMALA warmup phase/count mismatch")
            expected = adapt.step_size.initial_epsilon if adapt.completed == 0 else float(
                jnp.exp(adapt.step_size.state.log_step_size)
            )
        else:
            if adapt.completed != adapt.num_warmup or chain.iteration < adapt.completed:
                raise ValueError("MMALA production requires completed epsilon adaptation")
            _, _, final = dual_averaging_adaptation(adapt.step_size.target_accept)
            expected = float(final(adapt.step_size.state))
        if epsilon != expected:
            raise ValueError("MMALA epsilon disagrees with its adaptation phase/state")
    if adapt is not None:
        adapt = replace(adapt, num_warmup=int(adapt.num_warmup))
    return replace(chain, model_state=s, epsilon=epsilon, epsilon_G=ridge,
                   iteration=checked.iteration, adaptation=adapt)


def initialize_mmala_chain(
    target: CalibrationTarget, state: CalibrationState, key: Array,
    *, epsilon: float, epsilon_G: float,
) -> MMALAChain:
    """Initialize fixed MMALA with an explicit epsilon and positive ridge."""

    return validate_mmala_chain(target, MMALAChain(state, key, epsilon, epsilon_G))


def initialize_mmala_warmup(
    target: CalibrationTarget, state: CalibrationState, key: Array,
    *, num_warmup: int, epsilon: float, epsilon_G: float, target_accept: float = 0.574,
) -> MMALAChain:
    """Tune epsilon only with standard DA and mean site probabilities.

    The final averaged epsilon freezes; G_i(position) keeps changing as part
    of the fixed production kernel. target_accept=0.574 is a tuning heuristic
    shared with MALA. The source note does not specify a universal ridge, so
    epsilon_G is required and recorded, with no silent default or repair.
    """

    chain = initialize_mmala_chain(target, state, key, epsilon=epsilon, epsilon_G=epsilon_G)
    value = np.asarray(target_accept)
    if (value.shape != () or value.dtype.kind not in "fiu" or not np.isfinite(value)
        or not 0 < value < 1):
        raise ValueError("target_accept must be a finite scalar in (0,1)")
    init, _, _ = dual_averaging_adaptation(float(value))
    adaptation = MMALAAdaptationState(num_warmup, MALAStepSizeAdaptationState(
        float(value), chain.epsilon, jax.tree.map(jnp.asarray, init(chain.epsilon))
    ))
    return validate_mmala_chain(target, replace(chain, phase="warmup", adaptation=adaptation))


def _run_mmala_chunk(target, chain, num_sweeps, *, kernels=None):
    """Record complete sweeps and adapt epsilon once on the outer clock."""

    if kernels is None:
        kernels = ChunkRunner(target, chain).kernels
    kernel = kernels["sweep"]
    if chain.adaptation is not None:
        update, final = kernels["update"], kernels["final"]
    samples, diagnostics = [], []
    for _ in range(num_sweeps):
        state, key, info = _checked_call(kernel, chain.key, chain.model_state, chain.epsilon)
        epsilon, adapt, phase = chain.epsilon, chain.adaptation, chain.phase
        if phase == "warmup":
            da = _checked_call(update, adapt.step_size.state, jnp.mean(info.theta.acceptance_rate))
            adapt = replace(adapt, step_size=replace(adapt.step_size, state=da))
            if adapt.completed == adapt.num_warmup:
                phase = "sampling"
                epsilon = float(final(da))
            else:
                epsilon = float(jnp.exp(da.log_step_size))
            _check_diffusion(epsilon)
        chain = replace(
            chain, model_state=state, key=key, epsilon=epsilon, adaptation=adapt,
            phase=phase, iteration=chain.iteration + 1,
        )
        samples.append(chain.model_state)
        diagnostics.append(info)
    target.gp.validate_field_sites(target.coordinates.eta_to_theta_tilde(chain.model_state.eta))
    return (chain, jax.tree.map(lambda *x: jnp.stack(x), *samples),
            jax.tree.map(lambda *x: jnp.stack(x), *diagnostics))


def run_mmala_warmup(
    target: CalibrationTarget, chain: MMALAChain, num_sweeps: int | None = None,
) -> tuple[MMALAChain, CalibrationState, GibbsSweepInfo]:
    """Run a warmup-only chunk, freezing final averaged epsilon at its end."""

    _check_chain_control(chain, MMALAChain)
    if chain.phase != "warmup":
        raise ValueError("MMALA warmup requires phase='warmup'")
    remaining = chain.adaptation.num_warmup - chain.adaptation.completed
    if num_sweeps is None:
        num_sweeps = remaining
    if (isinstance(num_sweeps, bool) or not isinstance(num_sweeps, Integral)
        or not 1 <= num_sweeps <= remaining):
        raise ValueError("MMALA warmup chunk must fit remaining schedule")
    return _run_mmala_chunk(target, chain, num_sweeps)


def run_fixed_mmala(
    target: CalibrationTarget, chain: MMALAChain, num_sweeps: int,
) -> tuple[MMALAChain, CalibrationState, GibbsSweepInfo]:
    """Retain complete draws with frozen epsilon, ridge and DA statistics."""

    _check_chain_control(chain, MMALAChain)
    if chain.phase != "sampling":
        raise ValueError("Complete MMALA warmup before retained sampling")
    if (isinstance(num_sweeps, bool) or not isinstance(num_sweeps, Integral)
        or num_sweeps < 1):
        raise ValueError("num_sweeps must be a positive integer")
    return _run_mmala_chunk(target, chain, num_sweeps)


class ChunkRunner:
    """Reuse JAX functions across checkpoint chunks for one fixed target/config.

    The public one-shot drivers remain available. This orchestration object
    shares their exact transitions and checks, avoiding repeated tracing of
    fresh closures during timed runs. It holds compiled functions, not model
    states or density/gradient caches. Build a new runner after restart.
    """

    @staticmethod
    def _settings(chain):
        """Identify static kernel/adaptation choices that must not change."""
        adapt = chain.adaptation
        da = getattr(chain, "step_size_adaptation", None)
        if isinstance(chain, MMALAChain) and adapt is not None:
            da = adapt.step_size
        return (
            type(chain), getattr(adapt, "num_warmup", None),
            getattr(adapt, "num_initial", None), getattr(adapt, "target_accept", None),
            getattr(da, "target_accept", None), getattr(chain, "epsilon_G", None),
            getattr(chain, "max_num_doublings", None),
            getattr(chain, "divergence_threshold", None), getattr(chain, "collapsed", None),
        )

    def __init__(self, target, chain):
        self.target = target
        self.settings = self._settings(chain)
        self.kernels = {}
        adapt = chain.adaptation
        if isinstance(chain, NUTSChain):
            self.run = _run_nuts_chunk
            self.kernels["sweep"] = jax.jit(lambda key, state, step, mass:
                nuts_gibbs_sweep(
                    key, target, state, step, mass,
                    max_num_doublings=chain.max_num_doublings,
                    divergence_threshold=chain.divergence_threshold,
                    collapsed=chain.collapsed,
                ))
            if adapt is not None:
                _, update, final = nuts_window_adapter(adapt.target_accept)
                self.kernels.update(update=jax.jit(update), final=final,
                                    schedule=build_schedule(adapt.num_warmup))
        elif isinstance(chain, MMALAChain):
            self.run = _run_mmala_chunk
            self.kernels["sweep"] = jax.jit(lambda key, state, epsilon:
                mmala_gibbs_sweep(key, target, state, epsilon, chain.epsilon_G))
            if adapt is not None:
                _, update, final = dual_averaging_adaptation(adapt.step_size.target_accept)
                self.kernels.update(update=jax.jit(update), final=final)
        elif isinstance(chain, (RandomWalkChain, MALAChain)):
            is_mala = isinstance(chain, MALAChain)
            self.run = _run_metropolis_chunk
            self.kernels["sweep"] = jax.jit(lambda key, state, proposal, epsilon:
                collapsed_gibbs_sweep(key, target, state, proposal, epsilon=epsilon))
            self.kernels["moments"] = jax.jit(lambda moments, eta, proposal, eligible:
                update_random_walk_adaptation(
                    moments, eta, proposal, eligible,
                    proposal_scale=1.0 if is_mala else None,
                ))
            da = chain.step_size_adaptation if is_mala else None
            if da is not None:
                _, update, final = dual_averaging_adaptation(da.target_accept)
                self.kernels.update(update=jax.jit(update), final=final)
        else:
            raise TypeError("Unsupported chain type")
        self.kernels["sweep"] = jax.jit(checkify.checkify(self.kernels["sweep"]))
        for name in ("moments", "update"):
            if name in self.kernels:
                self.kernels[name] = _checked_adaptation(self.kernels[name])

    def compile(self, chain):
        """Compile the sweep without consuming keys; adaptation compiles in warmup."""
        _check_chain_control(chain, self.settings[0])
        if self._settings(chain) != self.settings:
            raise ValueError("ChunkRunner configuration changed")
        args = (chain.key, chain.model_state)
        if isinstance(chain, NUTSChain):
            args += (chain.step_size, chain.inverse_mass_matrix)
        elif isinstance(chain, MMALAChain):
            args += (chain.epsilon,)
        else:
            args += (chain.V_prop, getattr(chain, "epsilon", None))
        self.kernels["sweep"].lower(*args).compile()

    def __call__(self, chain, num_sweeps):
        _check_chain_control(chain, self.settings[0])
        if self._settings(chain) != self.settings:
            raise ValueError("ChunkRunner configuration changed")
        if (isinstance(num_sweeps, bool) or not isinstance(num_sweeps, Integral)
            or num_sweeps < 1):
            raise ValueError("num_sweeps must be a positive integer")
        if chain.phase == "warmup" and num_sweeps > (
            chain.adaptation.num_warmup - chain.adaptation.completed
        ):
            raise ValueError("A chunk cannot mix warmup and production")
        return self.run(self.target, chain, num_sweeps, kernels=self.kernels)
