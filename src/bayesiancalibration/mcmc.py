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

from bayesiancalibration.validation import (
    SamplingError, chain_context, check_quantity, validate_control, validate_key,
    validate_model_state, validate_spd,
)

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
    validate_nuts_mass,
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
from bayesiancalibration.samplers.nuts import (
    NUTSSweepInfo, nuts_sweep, nuts_blocks,
)
from bayesiancalibration.samplers.mmala import (
    collapsed_mmala_sweep, validate_mmala_geometry,
)
from bayesiancalibration.targets import CalibrationTarget


@dataclass(frozen=True)
class RandomWalkChain:
    """Local model boundary, distinct from fixed target and returned diagnostics.

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
    """MALA model/key/tuning with separate warmup statistics.

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

    def __init__(self, adaptation, *, chain=None, chain_index=None, batch=None):
        self.diagnostics = {
            **(chain_context(chain, chain_index) if chain is not None else {}),
            "update": "adaptation", "quantity": "V_prop", "role": "next",
            "batch": list(batch) if batch is not None else None,
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
    schedule order. The split-8-v1 protocol is shared by all methods.
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
    check_quantity(jnp.all(jnp.isfinite(delta)), update="delta", quantity="delta")
    sigma_y2 = sample_branch_noise(
        ky, target.y_tilde, state.c_f, delta, target.R, target.branch_sizes,
        target.alpha_y_0, target.beta_y_0,
    )
    check_quantity(jnp.all(jnp.isfinite(sigma_y2)), update="sigma_y2", quantity="sigma_y2")
    check_quantity(jnp.all(sigma_y2 > 0), update="sigma_y2", quantity="sigma_y2", criterion="positive")
    mu_theta = sample_spatial_mean(
        km, theta_tilde, target.C_theta, state.Sigma_theta,
        target.spatial_prior.m_theta_0, target.spatial_prior.V_theta_0,
    )
    check_quantity(jnp.all(jnp.isfinite(mu_theta)), update="mu_theta", quantity="mu_theta")
    Sigma_theta = sample_spatial_covariance(
        kS, theta_tilde, target.C_theta, mu_theta,
        target.spatial_prior.nu_theta_0, target.spatial_prior.S_theta_0,
    )
    check_quantity(jnp.all(jnp.isfinite(Sigma_theta)), update="Sigma_theta", quantity="Sigma_theta")
    r, k = target.gp.F_s.shape
    m_f_given_s, C_f_given_s, _ = target.gp.conditional_moments(
        theta_tilde, jnp.ones(k, dtype=jnp.float64)
    )
    sigma_c2 = sample_coefficient_variances(
        kc, state.c_f, m_f_given_s, C_f_given_s, target.gp.q_s, r,
        target.branch_sizes, target.alpha_c_0, target.beta_c_0,
    )
    check_quantity(jnp.all(jnp.isfinite(sigma_c2)), update="sigma_c2", quantity="sigma_c2")
    check_quantity(jnp.all(sigma_c2 > 0), update="sigma_c2", quantity="sigma_c2", criterion="positive")
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
    check_quantity(jnp.all(jnp.isfinite(eta)), update="eta", quantity="eta", role="selected")
    m_f_given_s, _, Sigma_f_given_s = target.gp.conditional_moments(
        target.coordinates.eta_to_theta_tilde(eta), sigma_c2
    )
    d_delta, Omega_y = target.observation_arrays(delta, sigma_y2)
    c_f = sample_projected_coefficients(
        kf, target.y_tilde, m_f_given_s, Sigma_f_given_s,
        target.R, d_delta, Omega_y,
    )
    check_quantity(jnp.all(jnp.isfinite(c_f)), update="c_f", quantity="c_f")
    new_state = CalibrationState(
        eta, c_f, delta, sigma_y2, mu_theta, Sigma_theta, sigma_c2
    )
    joint = target.full_joint_uncollapsed(*new_state)
    # Device reductions only: do not repeat factors or evaluate another density.
    check_quantity(jnp.isfinite(joint), update="joint", quantity="full_joint_logdensity", role="selected")
    check_quantity(jnp.isfinite(theta_info.logdensity), update="eta", quantity="logdensity", role="selected")
    check_quantity(jnp.all(jnp.isfinite(theta_info.acceptance_rate)), update="eta", quantity="acceptance_rate", role="selected")
    return new_state, next_key, GibbsSweepInfo(theta_info, joint, jnp.any(eta != state.eta, axis=1))


def _checked_call(kernel, *args, chain, chain_index=None, stage="sweep", quantity=None,
                  role=None, block=None, target=None, diagnostic=False, batch=None):
    """Report the first device failure before advancing a host model boundary."""
    error, result = kernel(*args)
    message = error.get()
    if message is not None:
        attempted = (result[0] if isinstance(result, tuple) and isinstance(result[0], CalibrationState) else None)
        attempted_tuning = None
        if stage == "adaptation":
            attempted_tuning = {}
            for path, value in jax.tree_util.tree_flatten_with_path(result)[0]:
                blocked_nuts = (isinstance(chain, NUTSChain)
                                and len(nuts_blocks(chain.model_state.eta.shape[0], chain.block_size)) > 1)
                prefix = "window" if blocked_nuts else "step_size"
                if quantity == "V_prop":
                    prefix = "moments" if path[0].idx == 0 else "V_prop"
                    path = path[1:]
                suffix = ".".join(str(getattr(part, "name", getattr(part, "idx", ""))) for part in path)
                attempted_tuning[prefix + ("." + suffix if suffix else "")] = value
        raise SamplingError(message, chain, chain_index=chain_index,
                            stage=stage, quantity=quantity, role=role, block=block,
                            attempted_state=attempted, target=target, diagnostic=diagnostic,
                            batch=batch, attempted_tuning=attempted_tuning)
    return result


def _audit_geometry(target, chain, *, chain_index=None, batch, diagnostic=False):
    """Apply the existing GP criterion at a labeled completed-state boundary."""
    try:
        target.gp.validate_field_sites(target.coordinates.eta_to_theta_tilde(chain.model_state.eta))
    except (ValueError, np.linalg.LinAlgError) as error:
        raise SamplingError(str(error), chain, chain_index=chain_index,
                            stage="geometry", quantity="gp_field_covariance",
                            role="selected", sweep=chain.iteration, batch=batch,
                            attempted_state=chain.model_state, target=target, diagnostic=diagnostic) from error


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


def _checked_adaptation(function, quantity):
    """Check only updated numbers; never rebuild schedules or validate old tuning."""
    def update(*args):
        result = function(*args)
        for path, value in jax.tree_util.tree_flatten_with_path(result)[0]:
            # Covariance adaptation returns moments and V_prop separately;
            # other named fields retain their statistical names.
            prefix = quantity
            if quantity == "covariance":
                prefix = "moments" if path[0].idx == 0 else "V_prop"
                path = path[1:]
            suffix = ".".join(str(getattr(part, "name", getattr(part, "idx", ""))) for part in path)
            check_quantity(jnp.all(jnp.isfinite(value)), update="adaptation",
                           quantity=prefix + ("." + suffix if suffix else ""), role="next")
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


def _validate_metropolis_boundary(target, chain):
    """Common MH/MALA model, key and covariance-schedule checks without a surrogate chain."""
    validate_control(chain.iteration, chain.phase)
    validate_key(chain.key)
    if chain.phase == "warmup" and chain.adaptation is None:
        raise ValueError("Warmup phase requires adaptation state")
    state = validate_model_state(target, chain.model_state)
    n, d = state.eta.shape
    proposal = validate_spd(chain.V_prop, (n, d, d), "V_prop")
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
    return state, proposal, adaptation


def validate_random_walk_chain(
    target: CalibrationTarget, chain: RandomWalkChain
) -> RandomWalkChain:
    """Validate the model and declared MH tuning at initialization, without repair."""
    if not isinstance(chain, RandomWalkChain):
        raise ValueError("chain must be a RandomWalkChain")
    state, proposal, adaptation = _validate_metropolis_boundary(target, chain)
    return replace(chain, model_state=state, V_prop=proposal,
                   iteration=int(chain.iteration), adaptation=adaptation)


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
    """Initialize scheduled MH warmup; default covariance is 1e-6 I per site.

    Both lengths must be declared, with 1 <= num_initial <= num_warmup.
    V_prop is an optional declared SPD (n,d,d) initial covariance. Covariance
    estimation uses completed warmup states, excluding the initial position.
    """

    n = target.C_theta.shape[0]
    d = target.gp.theta_s_tilde.shape[1]
    if V_prop is None:
        V_prop = 1e-6 * jnp.tile(jnp.eye(d, dtype=jnp.float64), (n, 1, 1))
    chain = initialize_random_walk_chain(target, state, key, V_prop)
    adaptation = initialize_random_walk_adaptation(
        num_warmup, num_initial, chain.V_prop
    )
    return validate_random_walk_chain(
        target, replace(chain, phase="warmup", adaptation=adaptation)
    )


def _run_metropolis_batch(
    target: CalibrationTarget, chain: RandomWalkChain | MALAChain, num_sweeps: int,
    *, kernels=None, diagnostic=False, chain_index=None,
) -> tuple[RandomWalkChain | MALAChain, CalibrationState, GibbsSweepInfo]:
    """Share complete-sweep recording/failure logic across both phases.

    This model-specific orchestration keeps PRNG, coefficient refresh, and
    validation identical in warmup/production. Proposal adaptation occurs
    only after a completed warmup sweep, for use in the next sweep.
    Public drivers prevent batches from mixing warmup and retained draws.
    """

    is_mala = isinstance(chain, MALAChain)
    if kernels is None:
        kernels = SweepRunner(target, chain).kernels
    kernel, adapt_kernel = kernels["sweep"], kernels["moments"]
    step_size_adaptation = chain.step_size_adaptation if is_mala else None
    if step_size_adaptation is not None:
        da_update, da_final = kernels["update"], kernels["final"]
    batch = (chain.iteration + 1, chain.iteration + num_sweeps)
    samples, diagnostics = [], []
    for _ in range(num_sweeps):
        epsilon = chain.epsilon if is_mala else None
        state, next_key, info = _checked_call(kernel,
            chain.key, chain.model_state, chain.V_prop, epsilon,
            chain=chain, chain_index=chain_index, target=target, diagnostic=diagnostic, batch=batch,
        )
        proposal, adaptation, phase = chain.V_prop, chain.adaptation, chain.phase
        if phase == "warmup":
            completed = adaptation.completed + 1
            eligible = (
                completed >= adaptation.num_initial
                and adaptation.num_initial < adaptation.num_warmup
            )
            moments, proposal = _checked_call(adapt_kernel,
                adaptation.moments, state.eta, proposal, jnp.asarray(eligible),
                chain=chain, chain_index=chain_index, stage="adaptation", quantity="V_prop", batch=batch,
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
                    raise WarmupTuningError(adaptation, chain=chain, chain_index=chain_index, batch=batch)
                phase = "sampling"
            if step_size_adaptation is not None:
                da_state = _checked_call(da_update,
                    step_size_adaptation.state, jnp.mean(info.theta.acceptance_rate),
                    chain=chain, chain_index=chain_index, stage="adaptation", quantity="epsilon", batch=batch,
                )
                step_size_adaptation = replace(step_size_adaptation, state=da_state)
                epsilon = float(
                    da_final(da_state) if phase == "sampling"
                    else jnp.exp(da_state.log_step_size)
                )
            if is_mala:
                try:
                    _check_diffusion(epsilon, proposal)
                except FloatingPointError as error:
                    raise SamplingError(str(error), chain, chain_index=chain_index,
                                        stage="adaptation", quantity="epsilon/diffusion", role="next", batch=batch,
                                        attempted_tuning={"epsilon": epsilon, "V_prop": proposal}, failed_value=epsilon) from error
        candidate = replace(
            chain, model_state=state, key=next_key, V_prop=proposal,
            iteration=chain.iteration + 1, phase=phase, adaptation=adaptation,
        )
        if is_mala:
            candidate = replace(
                candidate, epsilon=epsilon, step_size_adaptation=step_size_adaptation
            )
        chain = candidate
        if diagnostic:
            _audit_geometry(target, chain, chain_index=chain_index, batch=batch, diagnostic=diagnostic)
        samples.append(chain.model_state)
        diagnostics.append(info)
    if not diagnostic:
        _audit_geometry(target, chain, chain_index=chain_index, batch=batch, diagnostic=diagnostic)
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
    return _run_metropolis_batch(target, chain, num_sweeps)


def run_random_walk_warmup(
    target: CalibrationTarget,
    chain: RandomWalkChain,
    num_sweeps: int | None = None,
) -> tuple[RandomWalkChain, CalibrationState, GibbsSweepInfo]:
    """Run a warmup batch, optionally finishing all remaining warmup sweeps.

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
            "Warmup batch must be positive and not exceed remaining sweeps"
        )
    return _run_metropolis_batch(target, chain, num_sweeps)


def validate_mala_chain(target: CalibrationTarget, chain: MALAChain) -> MALAChain:
    """Reuse model/phase checks, then validate MALA diffusion and gradients.

    These additional checks are required by the gradient-driven proposal;
    finite densities alone do not guarantee usable MALA initialization.
    """

    if not isinstance(chain, MALAChain):
        raise ValueError("chain must be a MALAChain")
    state, proposal, adaptation = _validate_metropolis_boundary(target, chain)
    checked = replace(chain, model_state=state, V_prop=proposal, adaptation=adaptation)
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

    if V_prop is None:
        n, d = state.eta.shape
        V_prop = jnp.tile(jnp.eye(d, dtype=jnp.float64), (n, 1, 1))
    initial = initialize_mala_chain(target, state, key, V_prop, epsilon)
    adaptation = initialize_random_walk_adaptation(num_warmup, num_initial, initial.V_prop)
    chain = validate_mala_chain(target, replace(initial, phase="warmup", adaptation=adaptation))
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
    return _run_metropolis_batch(target, chain, num_sweeps)


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
            "Warmup batch must be positive and not exceed remaining sweeps"
        )
    return _run_metropolis_batch(target, chain, num_sweeps)


@dataclass(frozen=True)
class NUTSChain:
    """Complete model/key boundary and selectable eta-space NUTS tuning.

    inverse_mass_matrix is (n*d,) for diagonal, (n*d,n*d) for dense or
    Kronecker. All use eta coordinates and BlackJAX's convention. Window
    statistics are separate and freeze in production, as does step_size.
    No HMC position-density-gradient cache crosses an outer sweep boundary.
    """

    model_state: CalibrationState
    key: Array
    step_size: float | Array
    inverse_mass_matrix: Array | tuple[Array, ...]
    iteration: int = 0
    phase: str = "sampling"
    adaptation: NUTSAdaptationState | None = None
    max_num_doublings: int = 10
    divergence_threshold: float = 1000.0
    collapsed: bool = False
    mass_structure: str = "diagonal"
    block_size: int | None = None


def nuts_gibbs_sweep(
    key: Array, target: CalibrationTarget, state: CalibrationState,
    step_size: Array, inverse_mass_matrix: Array,
    *, max_num_doublings: int = 10, divergence_threshold: float = 1000,
    collapsed: bool = False, block_size: int | None = None,
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
            collapsed=collapsed, block_size=block_size,
        )
    return collapsed_gibbs_sweep(key, target, state, None, _theta_transition=transition)


def validate_nuts_chain(target: CalibrationTarget, chain: NUTSChain) -> NUTSChain:
    """Validate model, finite gradient, tuning and standard-window phase.

    Model/PRNG/GP checks are independent of NUTS mass and window counters. Validation does not
    consume randomness, clip tuning, or retain target gradients.
    """

    if not isinstance(chain, NUTSChain):
        raise ValueError("chain must be a NUTSChain")
    if not isinstance(chain.model_state, CalibrationState):
        raise ValueError("model_state must be a CalibrationState")
    if not isinstance(chain.collapsed, bool):
        raise ValueError("NUTS collapsed must be a boolean target choice")
    n, d = chain.model_state.eta.shape
    blocks = nuts_blocks(n, chain.block_size)
    blocked = len(blocks) > 1
    validate_control(chain.iteration, chain.phase)
    validate_key(chain.key)
    state = validate_model_state(target, chain.model_state)
    for name in ("step_size", "divergence_threshold"):
        value = np.asarray(getattr(chain, name))
        shape = (len(blocks),) if name == "step_size" and blocked else ()
        if (value.shape != shape or value.dtype.kind not in "fiu"
            or not np.all(np.isfinite(value)) or np.any(value <= 0)):
            raise ValueError(f"NUTS {name} must have shape {shape} and be finite positive")
    if (isinstance(chain.max_num_doublings, bool)
        or not isinstance(chain.max_num_doublings, Integral)
        or chain.max_num_doublings < 1):
        raise ValueError("max_num_doublings must be a positive integer")
    if blocked and (not isinstance(chain.inverse_mass_matrix, tuple)
                    or len(chain.inverse_mass_matrix) != len(blocks)):
        raise ValueError("Blocked NUTS requires one inverse mass array per block")
    masses = chain.inverse_mass_matrix if blocked else (chain.inverse_mass_matrix,)
    if chain.phase not in ("warmup", "sampling"):
        raise ValueError("Invalid NUTS phase")
    adapt = chain.adaptation
    if adapt is None:
        if chain.phase != "sampling":
            raise ValueError("NUTS warmup requires a window schedule")
        windows = (None,) * len(blocks)
    else:
        if not isinstance(adapt, NUTSAdaptationState):
            raise ValueError("NUTS requires standard window adaptation state")
        if blocked and (not isinstance(adapt.state, tuple)
                        or len(adapt.state) != len(blocks)):
            raise ValueError("Blocked NUTS requires one window state per block")
        windows = adapt.state if blocked else (adapt.state,)
    steps = np.asarray(chain.step_size).reshape(-1)
    for index, ((start, stop), mass, window) in enumerate(zip(blocks, masses, windows)):
        shape = (stop-start, d)
        validate_nuts_mass(mass, (stop-start)*d, chain.mass_structure, shape)
        if adapt is not None:
            validate_nuts_adaptation(replace(adapt, state=window),
                                     (stop-start)*d, chain.mass_structure, shape)
            if chain.phase == "warmup":
                if adapt.completed >= adapt.num_warmup or chain.iteration != adapt.completed:
                    raise ValueError("NUTS warmup phase/count mismatch")
                expected_step, expected_mass = window.step_size, window.inverse_mass_matrix
            else:
                if adapt.completed != adapt.num_warmup or chain.iteration < adapt.completed:
                    raise ValueError("NUTS production requires completed window adaptation")
                _, _, final = nuts_window_adapter(adapt.target_accept, chain.mass_structure, shape)
                expected_step, expected_mass = final(window)
            if float(steps[index]) != float(expected_step) or not np.array_equal(mass, expected_mass):
                raise ValueError("NUTS tuning disagrees with its window phase/state")
    s = state
    def density(eta):
        if chain.collapsed:
            return target.theta_only_collapsed(
                eta, s.delta, s.sigma_y2, s.mu_theta, s.Sigma_theta, s.sigma_c2
            )
        return target.theta_only_uncollapsed(
            eta, s.c_f, s.mu_theta, s.Sigma_theta, s.sigma_c2
        )
    value, grad = jax.value_and_grad(density)(s.eta)
    if not np.isfinite(float(value)) or not np.all(np.isfinite(np.asarray(grad))):
        raise ValueError("Current NUTS density/gradient must be finite")
    if adapt is not None:
        adapt = replace(adapt, num_warmup=int(adapt.num_warmup), completed=int(adapt.completed),
                        initial_step_size=float(adapt.initial_step_size),
                        target_accept=float(adapt.target_accept))
    return replace(chain, model_state=s,
                   step_size=jnp.asarray(steps) if blocked else float(steps[0]),
                   inverse_mass_matrix=(tuple(jnp.asarray(m, dtype=jnp.float64) for m in masses)
                                        if blocked else jnp.asarray(masses[0], dtype=jnp.float64)),
                   adaptation=adapt, block_size=int(chain.block_size) if blocked else None,
                   iteration=int(chain.iteration), max_num_doublings=int(chain.max_num_doublings),
                   divergence_threshold=float(chain.divergence_threshold))


def initialize_nuts_chain(
    target: CalibrationTarget, state: CalibrationState, key: Array,
    *, step_size: float, inverse_mass_matrix: Array,
    max_num_doublings: int = 10, divergence_threshold: float = 1000,
    collapsed: bool = False, mass_structure: str = "diagonal",
    block_size: int | None = None,
) -> NUTSChain:
    """Initialize checked fixed NUTS tuning; select collapse explicitly."""

    return validate_nuts_chain(target, NUTSChain(
        state, key, step_size, inverse_mass_matrix,
        max_num_doublings=max_num_doublings, divergence_threshold=divergence_threshold,
        collapsed=collapsed, mass_structure=mass_structure, block_size=block_size,
    ))


def initialize_nuts_warmup(
    target: CalibrationTarget, state: CalibrationState, key: Array,
    *, num_warmup: int = 1000, initial_step_size: float = 1.0,
    target_accept: float = 0.8, max_num_doublings: int = 10,
    divergence_threshold: float = 1000,
    collapsed: bool = False, mass_structure: str = "diagonal",
    block_size: int | None = None,
) -> NUTSChain:
    """Resolve the selected eta-space window-adaptation defaults explicitly.

    Standard schedule buffers are 75/25/50, with BlackJAX short-run handling.
    Initial inverse mass is identity; no empirical-covariance MALA/RW rule
    is reused. A warmup step is one complete Gibbs sweep, with acceptance
    feedback from the NUTS trajectory and current conditional each time.
    """

    blocks = nuts_blocks(state.eta.shape[0], block_size)
    windows = []
    for start, stop in blocks:
        init, _, _ = nuts_window_adapter(target_accept, mass_structure,
                                         (stop-start, state.eta.shape[1]))
        windows.append(jax.tree.map(jnp.asarray, init(
            state.eta[start:stop].reshape(-1), initial_step_size)))
    blocked = len(blocks) > 1
    adaptation = NUTSAdaptationState(
        num_warmup, 0, initial_step_size, target_accept,
        tuple(windows) if blocked else windows[0],
    )
    chain = NUTSChain(
        state, key, jnp.full((len(blocks),), initial_step_size) if blocked else initial_step_size,
        tuple(w.inverse_mass_matrix for w in windows) if blocked else windows[0].inverse_mass_matrix,
        phase="warmup", adaptation=adaptation, max_num_doublings=max_num_doublings,
        divergence_threshold=divergence_threshold, collapsed=collapsed,
        mass_structure=mass_structure, block_size=block_size,
    )
    return validate_nuts_chain(target, chain)


def _run_nuts_batch(target, chain, num_sweeps, *, kernels=None, diagnostic=False, chain_index=None):
    """Keep full-sweep recording and standard adaptation on the same clock."""

    if kernels is None:
        kernels = SweepRunner(target, chain).kernels
    kernel = kernels["sweep"]
    blocks = nuts_blocks(chain.model_state.eta.shape[0], chain.block_size)
    blocked = len(blocks) > 1
    if chain.adaptation is not None:
        update, final, schedule = kernels["update"], kernels["final"], kernels["schedule"]
    batch = (chain.iteration + 1, chain.iteration + num_sweeps)
    samples, diagnostics = [], []
    for _ in range(num_sweeps):
        state, key, info = _checked_call(kernel,
            chain.key, chain.model_state, chain.step_size, chain.inverse_mass_matrix,
            chain=chain, chain_index=chain_index, target=target, diagnostic=diagnostic, batch=batch,
        )
        step, mass, adapt, phase = (
            chain.step_size, chain.inverse_mass_matrix, chain.adaptation, chain.phase
        )
        if phase == "warmup":
            # The standard Welford recipe ignores grad. Supplying zero avoids
            # recomputing a gradient under the newly refreshed c_f conditional.
            windows = adapt.state if blocked else (adapt.state,)
            updates = update if blocked else (update,)
            finals = final if blocked else (final,)
            rates = info.theta.acceptance_rate if blocked else (info.theta.acceptance_rate,)
            next_windows, next_steps, next_masses = [], [], []
            for block, ((start, stop), window, upd, fin, rate) in enumerate(zip(
                blocks, windows, updates, finals, rates
            )):
                position = state.eta[start:stop].reshape(-1)
                updated = _checked_call(upd, window, schedule[adapt.completed], position,
                                        jnp.zeros(position.size, dtype=jnp.float64), rate,
                                        chain=chain, chain_index=chain_index, stage="adaptation",
                                        quantity="step_size/mass", block=block, batch=batch)
                if adapt.completed + 1 == adapt.num_warmup:
                    block_step, block_mass = fin(updated)
                else:
                    block_step, block_mass = updated.step_size, updated.inverse_mass_matrix
                if not np.isfinite(float(block_step)) or float(block_step) <= 0:
                    raise SamplingError("Warmup produced invalid NUTS step size", chain,
                                        chain_index=chain_index, stage="adaptation",
                                        quantity="step_size", role="next", block=block, batch=batch,
                                        attempted_tuning={"step_size": block_step, "inverse_mass_matrix": block_mass})
                try:
                    validate_nuts_mass(block_mass, position.size, chain.mass_structure,
                                       (stop-start, state.eta.shape[1]))
                except ValueError as error:
                    raise SamplingError(str(error), chain, chain_index=chain_index,
                                        stage="adaptation", quantity="inverse_mass_matrix",
                                        role="next", block=block, batch=batch,
                                        attempted_tuning={"step_size": block_step, "inverse_mass_matrix": block_mass}) from error
                next_windows.append(updated)
                next_steps.append(block_step)
                next_masses.append(block_mass)
            adapt = replace(adapt, state=tuple(next_windows) if blocked else next_windows[0],
                            completed=adapt.completed + 1)
            if adapt.completed == adapt.num_warmup:
                phase = "sampling"
            step = jnp.stack(next_steps) if blocked else float(next_steps[0])
            mass = tuple(next_masses) if blocked else next_masses[0]
        chain = replace(chain, model_state=state, key=key, step_size=step,
                        inverse_mass_matrix=mass, adaptation=adapt, phase=phase,
                        iteration=chain.iteration + 1)
        if diagnostic:
            _audit_geometry(target, chain, chain_index=chain_index, batch=batch, diagnostic=diagnostic)
        samples.append(chain.model_state)
        diagnostics.append(info)
    if not diagnostic:
        _audit_geometry(target, chain, chain_index=chain_index, batch=batch, diagnostic=diagnostic)
    return (chain, jax.tree.map(lambda *x: jnp.stack(x), *samples),
            jax.tree.map(lambda *x: jnp.stack(x), *diagnostics))


def run_nuts_warmup(
    target: CalibrationTarget, chain: NUTSChain, num_sweeps: int | None = None,
) -> tuple[NUTSChain, CalibrationState, GibbsSweepInfo]:
    """Run a warmup-only batch; freeze final average step size and mass."""

    _check_chain_control(chain, NUTSChain)
    if chain.phase != "warmup":
        raise ValueError("NUTS warmup requires phase='warmup'")
    remaining = chain.adaptation.num_warmup - chain.adaptation.completed
    if num_sweeps is None:
        num_sweeps = remaining
    if (isinstance(num_sweeps, bool) or not isinstance(num_sweeps, Integral)
        or not 1 <= num_sweeps <= remaining):
        raise ValueError("NUTS warmup batch must fit remaining schedule")
    return _run_nuts_batch(target, chain, num_sweeps)


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
    return _run_nuts_batch(target, chain, num_sweeps)


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
    validate_control(chain.iteration, chain.phase)
    validate_key(chain.key)
    state = validate_model_state(target, chain.model_state)
    s = state
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
                   iteration=int(chain.iteration), adaptation=adapt)


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


def _run_mmala_batch(target, chain, num_sweeps, *, kernels=None, diagnostic=False, chain_index=None):
    """Record complete sweeps and adapt epsilon once on the outer clock."""

    if kernels is None:
        kernels = SweepRunner(target, chain).kernels
    kernel = kernels["sweep"]
    if chain.adaptation is not None:
        update, final = kernels["update"], kernels["final"]
    batch = (chain.iteration + 1, chain.iteration + num_sweeps)
    samples, diagnostics = [], []
    for _ in range(num_sweeps):
        state, key, info = _checked_call(kernel, chain.key, chain.model_state, chain.epsilon,
                                        chain=chain, chain_index=chain_index, target=target, diagnostic=diagnostic, batch=batch)
        epsilon, adapt, phase = chain.epsilon, chain.adaptation, chain.phase
        if phase == "warmup":
            da = _checked_call(update, adapt.step_size.state, jnp.mean(info.theta.acceptance_rate),
                               chain=chain, chain_index=chain_index, stage="adaptation", quantity="epsilon", batch=batch)
            adapt = replace(adapt, step_size=replace(adapt.step_size, state=da))
            if adapt.completed == adapt.num_warmup:
                phase = "sampling"
                epsilon = float(final(da))
            else:
                epsilon = float(jnp.exp(da.log_step_size))
            try:
                _check_diffusion(epsilon)
            except FloatingPointError as error:
                raise SamplingError(str(error), chain, chain_index=chain_index,
                                    stage="adaptation", quantity="epsilon/diffusion", role="next", batch=batch,
                                    attempted_tuning={"epsilon": epsilon}, failed_value=epsilon) from error
        chain = replace(
            chain, model_state=state, key=key, epsilon=epsilon, adaptation=adapt,
            phase=phase, iteration=chain.iteration + 1,
        )
        if diagnostic:
            _audit_geometry(target, chain, chain_index=chain_index, batch=batch, diagnostic=diagnostic)
        samples.append(chain.model_state)
        diagnostics.append(info)
    if not diagnostic:
        _audit_geometry(target, chain, chain_index=chain_index, batch=batch, diagnostic=diagnostic)
    return (chain, jax.tree.map(lambda *x: jnp.stack(x), *samples),
            jax.tree.map(lambda *x: jnp.stack(x), *diagnostics))


def run_mmala_warmup(
    target: CalibrationTarget, chain: MMALAChain, num_sweeps: int | None = None,
) -> tuple[MMALAChain, CalibrationState, GibbsSweepInfo]:
    """Run a warmup-only batch, freezing final averaged epsilon at its end."""

    _check_chain_control(chain, MMALAChain)
    if chain.phase != "warmup":
        raise ValueError("MMALA warmup requires phase='warmup'")
    remaining = chain.adaptation.num_warmup - chain.adaptation.completed
    if num_sweeps is None:
        num_sweeps = remaining
    if (isinstance(num_sweeps, bool) or not isinstance(num_sweeps, Integral)
        or not 1 <= num_sweeps <= remaining):
        raise ValueError("MMALA warmup batch must fit remaining schedule")
    return _run_mmala_batch(target, chain, num_sweeps)


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
    return _run_mmala_batch(target, chain, num_sweeps)


class SweepRunner:
    """Reuse numerical sweep functions for one fixed target and sampler.

    Holds compiled functions, separate from model/adaptation/diagnostics.
    Local batches never mix warmup with retained sampling. Normal execution
    audits GP geometry once at each batch endpoint; diagnostic execution
    applies the same criterion to every completed state. No I/O or job state.
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
            getattr(chain, "mass_structure", None), getattr(chain, "block_size", None),
        )

    def __init__(self, target, chain, *, diagnostic=False):
        if not isinstance(diagnostic, bool):
            raise ValueError("diagnostic must be a boolean")
        self.diagnostic = diagnostic
        self.target = target
        self.settings = self._settings(chain)
        self.kernels = {}
        adapt = chain.adaptation
        if isinstance(chain, NUTSChain):
            self.run = _run_nuts_batch
            self.kernels["sweep"] = jax.jit(lambda key, state, step, mass:
                nuts_gibbs_sweep(
                    key, target, state, step, mass,
                    max_num_doublings=chain.max_num_doublings,
                    divergence_threshold=chain.divergence_threshold,
                    collapsed=chain.collapsed, block_size=chain.block_size,
                ))
            if adapt is not None:
                blocks = nuts_blocks(chain.model_state.eta.shape[0], chain.block_size)
                adapters = [nuts_window_adapter(
                    adapt.target_accept, chain.mass_structure,
                    (stop-start, chain.model_state.eta.shape[1])) for start, stop in blocks]
                updates = tuple(jax.jit(adapter[1]) for adapter in adapters)
                finals = tuple(adapter[2] for adapter in adapters)
                self.kernels.update(update=updates if len(blocks)>1 else updates[0],
                                    final=finals if len(blocks)>1 else finals[0],
                                    schedule=build_schedule(adapt.num_warmup))
        elif isinstance(chain, MMALAChain):
            self.run = _run_mmala_batch
            self.kernels["sweep"] = jax.jit(lambda key, state, epsilon:
                mmala_gibbs_sweep(key, target, state, epsilon, chain.epsilon_G))
            if adapt is not None:
                _, update, final = dual_averaging_adaptation(adapt.step_size.target_accept)
                self.kernels.update(update=jax.jit(update), final=final)
        elif isinstance(chain, (RandomWalkChain, MALAChain)):
            is_mala = isinstance(chain, MALAChain)
            self.run = _run_metropolis_batch
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
                value = self.kernels[name]
                self.kernels[name] = (tuple(_checked_adaptation(fn, "window") for fn in value)
                                      if isinstance(value, tuple) else _checked_adaptation(
                                          value, "covariance" if name == "moments" else "step_size"))

    def compile(self, chain):
        """Compile the sweep without consuming keys; adaptation compiles in warmup."""
        _check_chain_control(chain, self.settings[0])
        if self._settings(chain) != self.settings:
            raise ValueError("SweepRunner configuration changed")
        args = (chain.key, chain.model_state)
        if isinstance(chain, NUTSChain):
            args += (chain.step_size, chain.inverse_mass_matrix)
        elif isinstance(chain, MMALAChain):
            args += (chain.epsilon,)
        else:
            args += (chain.V_prop, getattr(chain, "epsilon", None))
        self.kernels["sweep"].lower(*args).compile()

    def __call__(self, chain, num_sweeps, *, chain_index=None):
        _check_chain_control(chain, self.settings[0])
        if self._settings(chain) != self.settings:
            raise ValueError("SweepRunner configuration changed")
        if (isinstance(num_sweeps, bool) or not isinstance(num_sweeps, Integral)
            or num_sweeps < 1):
            raise ValueError("num_sweeps must be a positive integer")
        if chain.phase == "warmup" and num_sweeps > (
            chain.adaptation.num_warmup - chain.adaptation.completed
        ):
            raise ValueError("A batch cannot mix warmup and production")
        return self.run(self.target, chain, num_sweeps, kernels=self.kernels,
                        diagnostic=self.diagnostic, chain_index=chain_index)
