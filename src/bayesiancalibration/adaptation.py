"""Warmup covariance, NUTS windows and MALA/MMALA epsilon dual averaging."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np
from blackjax.adaptation.mass_matrix import WelfordAlgorithmState, welford_algorithm
from blackjax.adaptation.step_size import (
    DualAveragingAdaptationState,
    dual_averaging_adaptation,
)
from blackjax.adaptation.metric_recipes import lookup_recipe
from blackjax.adaptation.staged_adaptation import (
    StagedAdaptationState, _make_engine, build_schedule,
)
from jax import Array
from jax.experimental import checkify


def nuts_window_adapter(target_accept: float = 0.8):
    """Embed the exact engine used by BlackJAX 1.6.2 window_adaptation.

    Its high-level driver assumes a fixed density. Gibbs needs its low-level
    init/update/final triple while rebuilding the conditional each sweep.
    The private engine is version-sensitive: package versions are recorded
    and required at restart, and fixed-target driver equivalence is tested.
    Identity diagonal initialization, zero persistence, Stan regularization,
    and dual-averaging defaults all come directly from BlackJAX.
    """

    core = lookup_recipe("welford_diag").build_core()
    return _make_engine(core, target_acceptance_rate=target_accept)


@dataclass(frozen=True)
class NUTSAdaptationState:
    """Standard window statistics, separate from model and NUTS tuning.

    num_warmup defaults to 1000; completed counts full outer sweeps. State is
    BlackJAX's unmodified staged state (diagonal Welford + dual averaging).
    Schedule uses standard 75/25/50 buffers and its short-warmup handling.
    """

    num_warmup: int
    completed: int
    initial_step_size: float
    target_accept: float
    state: StagedAdaptationState


def validate_nuts_adaptation(adaptation: NUTSAdaptationState, size: int) -> None:
    """Enforce this chain's window representation and restart counters.

    The library engine provides updates; these host checks reject malformed
    or inconsistent checkpoint/configuration values without repairing them.
    """

    if not isinstance(adaptation, NUTSAdaptationState):
        raise ValueError("NUTS requires standard window adaptation state")
    for name in ("num_warmup", "completed"):
        value = getattr(adaptation, name)
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError("NUTS window lengths/counters must be integers")
    if not 0 <= adaptation.completed <= adaptation.num_warmup or adaptation.num_warmup < 1:
        raise ValueError("Invalid NUTS warmup length/counter")
    for name in ("initial_step_size", "target_accept"):
        value = np.asarray(getattr(adaptation, name))
        if (value.shape != () or value.dtype.kind not in "fiu"
            or not np.isfinite(value) or value <= 0
            or (name == "target_accept" and value >= 1)):
            raise ValueError(f"Invalid NUTS {name}")
    state = adaptation.state
    if not isinstance(state, StagedAdaptationState):
        raise ValueError("Expected BlackJAX staged adaptation state")
    for value in jax.tree.leaves(state):
        array = np.asarray(value)
        if array.dtype.kind not in "fiu" or not np.all(np.isfinite(array)):
            raise ValueError("NUTS window statistics must be finite numeric arrays")
        if array.dtype.kind == "f" and array.dtype != np.float64:
            raise ValueError("NUTS window statistics must use float64")
    schedule = np.asarray(build_schedule(adaptation.num_warmup))
    prefix = schedule[:adaptation.completed]
    resets = np.flatnonzero(prefix[:, 1])
    last_reset = int(resets[-1] + 1) if resets.size else 0
    count = int(np.sum(prefix[last_reset:, 0]))
    da, metric = state.ss_state, state.imm_state
    if (np.asarray(da.step).shape != () or np.asarray(da.step).dtype.kind not in "iu"
        or int(da.step) != adaptation.completed - last_reset + 1
        or abs(float(da.avg_error)) > 1):
        raise ValueError("NUTS dual-averaging state disagrees with window schedule")
    if any(np.asarray(getattr(da, name)).shape != () for name in da._fields):
        raise ValueError("NUTS dual-averaging statistics must be scalars")
    wc = metric.wc_state
    if (wc.mean.shape != (size,) or wc.m2.shape != (size,)
        or np.asarray(wc.sample_size).shape != ()
        or np.asarray(wc.sample_size).dtype.kind not in "iu"
        or int(wc.sample_size) != count or np.any(np.asarray(wc.m2) < 0)):
        raise ValueError("NUTS Welford statistics disagree with window schedule")
    if (np.asarray(state.step_size).shape != () or float(state.step_size) <= 0
        or state.inverse_mass_matrix.shape != (size,)
        or np.any(np.asarray(state.inverse_mass_matrix) <= 0)
        or not np.array_equal(state.inverse_mass_matrix, metric.inverse_mass_matrix)):
        raise ValueError("Invalid NUTS window step size or inverse mass matrix")
    if adaptation.completed == 0:
        init, _, _ = nuts_window_adapter(adaptation.target_accept)
        expected = init(jnp.zeros(size), adaptation.initial_step_size)
        if any(not np.array_equal(a, b) for a, b in zip(
            jax.tree.leaves(state), jax.tree.leaves(expected)
        )):
            raise ValueError("Empty NUTS window state must match initialization")


@dataclass(frozen=True)
class RandomWalkAdaptationState:
    """Declared warmup schedule and site-wise Welford accumulators.

    moments.mean is (n,d), moments.m2 (n,d,d), and moments.sample_size (n,).
    Each site counts every completed warmup sweep, including rejections.
    Initial positions are excluded; the initial fixed-period states are
    included. initial_V_prop is the declared SPD tuning, shape (n,d,d).
    Counts and statistics stop changing at the sampling boundary.
    """

    num_warmup: int
    num_initial: int
    initial_V_prop: Array
    moments: WelfordAlgorithmState
    acceptance_count: Array  # (n,), MH decisions, including accepted identical positions
    movement_count: Array  # (n,), actual eta changes, including the first transition
    zero_covariance_count: Array  # (n,), eligible updates that retained old tuning

    @property
    def completed(self) -> int:
        return int(self.moments.sample_size[0])


@dataclass(frozen=True)
class MALAStepSizeAdaptationState:
    """Separate MALA/MMALA epsilon configuration and standard BlackJAX DA state.

    target_accept is in (0,1); initial_epsilon is the declared positive scale.
    state has scalar log_step_size/log_step_size_avg/avg_error/mu (float64)
    and integer step. step - 1 counts completed warmup sweeps. Use BlackJAX
    defaults t0=10, gamma=0.05, kappa=0.75; adaptation acts on epsilon itself,
    not the MALA integrator's epsilon^2/2. All statistics freeze in production.
    """

    target_accept: float
    initial_epsilon: float
    state: DualAveragingAdaptationState


@dataclass(frozen=True)
class MMALAAdaptationState:
    """Declared epsilon-only warmup; the metric ridge is fixed chain tuning.

    The source metric changes with position without empirical covariance
    adaptation. Completed sweeps are counted by standard DA's step - 1.
    """

    num_warmup: int
    step_size: MALAStepSizeAdaptationState

    @property
    def completed(self) -> int:
        return int(self.step_size.state.step) - 1


def validate_mala_step_size_adaptation(
    adaptation: MALAStepSizeAdaptationState, completed: int
) -> MALAStepSizeAdaptationState:
    """Check DA configuration/state at a completed warmup/restart boundary.

    Library utilities do not enforce this Gibbs sampler's scalar float64
    representation, configuration, or sweep counters. This checker reports
    incompatible statistics without changing them or implementing DA updates.
    """

    if not isinstance(adaptation, MALAStepSizeAdaptationState):
        raise ValueError("step_size_adaptation must be a MALAStepSizeAdaptationState")
    for name in ("target_accept", "initial_epsilon"):
        value = np.asarray(getattr(adaptation, name))
        if (
            value.shape != () or value.dtype.kind not in "fiu"
            or not np.isfinite(value) or value <= 0
            or (name == "target_accept" and value >= 1)
        ):
            raise ValueError(f"Invalid scalar {name} for MALA dual averaging")
    state = adaptation.state
    if not isinstance(state, DualAveragingAdaptationState):
        raise ValueError("MALA dual averaging requires the standard BlackJAX state")
    for name in state._fields:
        value = np.asarray(getattr(state, name))
        if name == "step":
            if (
                value.shape != () or value.dtype.kind not in "iu"
                or value != completed + 1
            ):
                raise ValueError(
                    "Dual-averaging step must match completed warmup sweeps"
                )
        elif (
            value.shape != () or value.dtype != np.float64 or not np.isfinite(value)
        ):
            raise ValueError("Dual-averaging statistics must be finite float64 scalars")
    init, _, _ = dual_averaging_adaptation(float(adaptation.target_accept))
    initial = init(float(adaptation.initial_epsilon))
    tolerance = 8 * np.finfo(np.float64).eps
    if not np.isclose(state.mu, initial.mu, rtol=tolerance, atol=tolerance):
        raise ValueError("Dual-averaging center does not match initial_epsilon")
    if abs(float(state.avg_error)) > 1:
        raise ValueError("Dual-averaging acceptance error must lie in [-1,1]")
    if completed == 0 and any(
        not np.array_equal(actual, expected) for actual, expected in zip(state, initial)
    ):
        raise ValueError("Empty dual-averaging statistics must match initialization")
    return adaptation


_welford_init, _welford_update, _welford_final = welford_algorithm(False)


def initialize_random_walk_adaptation(
    num_warmup: int, num_initial: int, V_prop: Array
) -> RandomWalkAdaptationState:
    """Initialize a declared positive warmup/initial-period schedule.

    The standard dense Welford estimator is applied independently per site.
    The model-specific schedule requires 1 <= num_initial <= num_warmup;
    if they are equal, all warmup and production use the initial tuning.
    """

    if not jax.config.jax_enable_x64:
        raise ValueError("Random-walk adaptation requires jax_enable_x64=True")
    for name, value in (("num_warmup", num_warmup), ("num_initial", num_initial)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if num_initial > num_warmup:
        raise ValueError("num_initial must not exceed num_warmup")
    proposal = np.asarray(V_prop, dtype=np.float64)
    if (
        proposal.ndim != 3 or proposal.shape[0] < 1 or proposal.shape[1] < 1
        or proposal.shape[1] != proposal.shape[2]
        or not np.all(np.isfinite(proposal))
    ):
        raise ValueError("initial_V_prop must be finite with shape (n,d,d)")
    if not np.array_equal(proposal, np.swapaxes(proposal, -1, -2)):
        raise ValueError("initial_V_prop must be symmetric")
    try:
        np.linalg.cholesky(proposal)
    except np.linalg.LinAlgError as error:
        raise ValueError("initial_V_prop must be positive definite") from error
    n, d, _ = proposal.shape
    moments = jax.vmap(lambda _: _welford_init(d))(jnp.arange(n))
    return RandomWalkAdaptationState(
        int(num_warmup), int(num_initial), jnp.asarray(proposal), moments,
        jnp.zeros(n, dtype=jnp.int64), jnp.zeros(n, dtype=jnp.int64),
        jnp.zeros(n, dtype=jnp.int64),
    )


def update_random_walk_adaptation(
    moments: WelfordAlgorithmState,
    eta: Array,
    V_prop: Array,
    adapt: Array,
    proposal_scale: Array | None = None,
) -> tuple[WelfordAlgorithmState, Array]:
    """Accumulate one completed eta state; return tuning for the next sweep.

    Pure float64 JIT/vmap-compatible adapter for eta (n,d), V_prop (n,d,d).
    BlackJAX supplies the stable dense online mean/covariance estimator.
    Custom code applies the source-note D07 ridge and declared proposal scale,
    which differ from BlackJAX's NUTS mass-matrix regularization.

    adapt is a scalar flag controlled by the declared initial-period boundary.
    Use every completed warmup state, without filtering on acceptance. At
    least two states and a finite positive-trace SPD estimate are required.
    Rank-deficient sample covariances with positive trace receive exactly the
    specified ridge. Exactly zero scatter retains the previous valid tuning
    during adaptive warmup and is counted explicitly. Nonzero invalid estimates
    fail in the checked driver. Zero scatter at warmup end prevents production.
    No extra ridge, flooring, clipping, or random draw is introduced here.
    This function must never be called for retained production sweeps.
    proposal_scale defaults to 2.38^2/d for random walks; MALA supplies 1.0,
    since its epsilon^2 diffusion scale is applied separately by the kernel.
    """

    updated = jax.vmap(_welford_update)(moments, eta)
    # Store a symmetric covariance statistic; this removes only roundoff
    # asymmetry in Welford's outer product, without changing the estimator.
    M2 = 0.5 * (updated.m2 + jnp.swapaxes(updated.m2, -1, -2))
    M2 = jnp.tril(M2) + jnp.swapaxes(jnp.tril(M2, -1), -1, -2)
    updated = updated._replace(m2=M2)

    def proposal_from_covariance(_):
        S_hat, _, _ = jax.vmap(_welford_final)(updated)
        S_hat = 0.5 * (S_hat + jnp.swapaxes(S_hat, -1, -2))
        d = eta.shape[1]
        trace = jnp.trace(S_hat, axis1=-2, axis2=-1)
        P = S_hat + (trace / (1000.0 * d))[:, None, None] * jnp.eye(d)
        scale = 2.38**2 / d if proposal_scale is None else proposal_scale
        proposal = scale * P
        # Mirror one triangle so compiled arithmetic cannot leave a tiny
        # asymmetry at the host's exact covariance-symmetry boundary.
        proposal = jnp.tril(proposal) + jnp.swapaxes(
            jnp.tril(proposal, -1), -1, -2
        )
        # Exactly zero scatter is an explicit tuning issue during warmup.
        # Do not use a tolerance or hold nonzero, indefinite/nonfinite estimates.
        zero = jnp.all(S_hat == 0, axis=(-2, -1))
        proposal = jnp.where(zero[:, None, None], V_prop, proposal)
        checkify.debug_check(
            jnp.all(jnp.isfinite(jnp.linalg.cholesky(proposal))),
            "D07 covariance adaptation produced non-SPD/nonfinite tuning",
        )
        return proposal

    proposal = jax.lax.cond(
        jnp.asarray(adapt) & jnp.all(updated.sample_size > 1),
        proposal_from_covariance, lambda _: V_prop, operand=None,
    )
    return updated, proposal


def validate_random_walk_adaptation(
    adaptation: RandomWalkAdaptationState, n: int, d: int
) -> RandomWalkAdaptationState:
    """Check restartable schedule, counters, and estimator arrays outside JIT.

    Standard array checks cannot enforce this sampler's phase/count rules;
    this boundary rejects invalid saved statistics without modifying them.
    Welford M2 may have roundoff asymmetry, so its symmetry/PSD checks use a
    32-epsilon relative tolerance. This tolerance never changes proposals.
    """

    if not isinstance(adaptation, RandomWalkAdaptationState):
        raise ValueError("adaptation must be a RandomWalkAdaptationState")
    for name in ("num_warmup", "num_initial"):
        value = getattr(adaptation, name)
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if adaptation.num_initial > adaptation.num_warmup:
        raise ValueError("num_initial must not exceed num_warmup")
    initial = np.asarray(adaptation.initial_V_prop)
    if (
        initial.shape != (n, d, d) or initial.dtype != np.float64
        or not np.all(np.isfinite(initial))
        or not np.array_equal(initial, np.swapaxes(initial, -1, -2))
    ):
        raise ValueError("initial_V_prop must be a finite symmetric float64 (n,d,d)")
    try:
        np.linalg.cholesky(initial)
    except np.linalg.LinAlgError as error:
        raise ValueError("initial_V_prop must be positive definite") from error
    mean, m2, count = map(np.asarray, adaptation.moments)
    if (
        mean.shape != (n, d) or m2.shape != (n, d, d)
        or mean.dtype != np.float64 or m2.dtype != np.float64
        or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(m2))
    ):
        raise ValueError("Warmup mean/M2 must be finite float64 (n,d)/(n,d,d)")
    if (
        count.shape != (n,) or count.dtype.kind not in "iu"
        or not np.all(count == count[0])
        or count[0] < 0 or count[0] > adaptation.num_warmup
    ):
        raise ValueError("Warmup site counts must agree and lie within the schedule")
    if count[0] == 0 and (np.any(mean != 0) or np.any(m2 != 0)):
        raise ValueError("Empty warmup statistics must be zero")
    for name in ("acceptance_count", "movement_count", "zero_covariance_count"):
        counts = np.asarray(getattr(adaptation, name))
        if (counts.shape != (n,) or counts.dtype.kind not in "iu"
            or np.any(counts < 0) or np.any(counts > count[0])):
            raise ValueError(f"Invalid warmup {name}")
    if np.any(np.asarray(adaptation.movement_count) > np.asarray(adaptation.acceptance_count)):
        raise ValueError("Movement counts cannot exceed acceptance counts")
    if count[0] < 2 and np.any(m2 != 0):
        raise ValueError("Warmup M2 must be zero before two states")
    tolerance = 32 * np.finfo(np.float64).eps * np.max(np.abs(m2), axis=(-2, -1))
    if np.any(np.max(np.abs(m2 - np.swapaxes(m2, -1, -2)), axis=(-2, -1)) > tolerance):
        raise ValueError("Warmup M2 must be symmetric within roundoff")
    eigenvalues = np.linalg.eigvalsh(0.5 * (m2 + np.swapaxes(m2, -1, -2)))
    if np.any(eigenvalues[:, 0] < -tolerance):
        raise ValueError("Warmup M2 must be positive semidefinite within roundoff")
    return adaptation
