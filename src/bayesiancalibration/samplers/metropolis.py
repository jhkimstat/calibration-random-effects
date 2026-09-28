"""Collapsed random walks and preconditioned MALA in site-major eta.

One sweep visits sites in order, holding all other model variables fixed.
Proposal covariances are declared tuning, separate from model state and
returned diagnostics. A completed collapsed theta sweep must be followed
by the exact field-coefficient refresh before coefficient-dependent updates.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from blackjax.mcmc import mala, random_walk
from jax import Array
from jax.experimental import checkify

from bayesiancalibration.targets import CalibrationTarget


class RandomWalkSweepInfo(NamedTuple):
    """Per-site probabilities/decisions (n,) and final log density (scalar).

    logdensity belongs to the supplied conditioning variables; it is a
    diagnostic, never an initialization cache for the next Gibbs sweep.
    """

    acceptance_rate: Array
    is_accepted: Array
    logdensity: Array


class MALASweepInfo(NamedTuple):
    """Per-site acceptance probabilities/decisions and final collapsed density.

    Shapes are (n,), (n,), and scalar. No density or gradient diagnostic is
    reused across site updates or after Gibbs conditioning changes.
    """

    acceptance_rate: Array
    is_accepted: Array
    logdensity: Array


def collapsed_mala_sweep(
    key: Array,
    target: CalibrationTarget,
    eta: Array,
    delta: Array,
    sigma_y2: Array,
    mu_theta: Array,
    Sigma_theta: Array,
    sigma_c2: Array,
    V_prop: Array,
    epsilon: Array,
) -> tuple[Array, MALASweepInfo]:
    """Pure sequential site MALA using fixed SPD covariance preconditioners.

    Shapes: eta (n,d), delta/sigma_c2 (k,), sigma_y2 (B,), mu_theta (d,),
    Sigma_theta (d,d), V_prop (n,d,d), and positive scalar epsilon. The note's
    proposal has mean eta_i + epsilon^2 V_prop[i] grad_i / 2 and covariance
    epsilon^2 V_prop[i]. V_prop excludes epsilon^2 and random-walk scaling.

    BlackJAX's standard MALA kernel has isotropic diffusion. This adapter
    supplies local Cholesky coordinates eta_i = origin + L_i z, starting
    z at zero, and step_size=epsilon^2/2. Its autodiff and exact asymmetric
    MH correction then implement the specified dense preconditioned proposal.
    The affine determinant is constant through both directions and cancels;
    the target already includes the bounded-coordinate Jacobian. No metric
    volume term, extra ridge, clipping, or gradient repair is introduced.

    Initialize density/gradient anew at each site with the latest other sites
    and current Gibbs inputs. A rejected z=0 maps exactly to the original eta.
    Invalid candidate energies/gradients are rejected by BlackJAX's safe
    energy difference. Current gradients are checked inside each site step
    by the checkified chain driver. Close over target for JIT/vmap/scan.
    """

    def logdensity_fn(position):
        value = target.theta_only_collapsed(
            position, delta, sigma_y2, mu_theta, Sigma_theta, sigma_c2
        )
        # -inf is a legitimate zero-density proposal; NaN/+inf is not.
        checkify.debug_check(~jnp.isnan(value) & ~jnp.isposinf(value),
                             "Metropolis target evaluation produced NaN/+inf")
        return value

    kernel = mala.build_kernel()
    site_keys = jax.random.split(key, eta.shape[0])
    factors = jnp.linalg.cholesky(V_prop)

    def step(position, inputs):
        i, site_key, L = inputs
        origin = position[i]

        def site_logdensity(z):
            return logdensity_fn(position.at[i].set(origin + L @ z))

        state = mala.init(jnp.zeros_like(origin), site_logdensity)
        checkify.debug_check(
            jnp.isfinite(state.logdensity) & jnp.all(jnp.isfinite(state.logdensity_grad)),
            "MALA current density/gradient is nonfinite",
        )
        updated, info = kernel(site_key, state, site_logdensity, epsilon**2 / 2)
        next_position = position.at[i].set(origin + L @ updated.position)
        return next_position, (info.acceptance_rate, info.is_accepted)

    position, (acceptance_rate, is_accepted) = jax.lax.scan(
        step, eta, (jnp.arange(eta.shape[0]), site_keys, factors)
    )
    return position, MALASweepInfo(
        acceptance_rate, is_accepted, logdensity_fn(position)
    )


def collapsed_random_walk_sweep(
    key: Array,
    target: CalibrationTarget,
    eta: Array,
    delta: Array,
    sigma_y2: Array,
    mu_theta: Array,
    Sigma_theta: Array,
    sigma_c2: Array,
    V_prop: Array,
) -> tuple[Array, RandomWalkSweepInfo]:
    """Pure fixed-covariance MH sweep, returning eta (n,d) and diagnostics.

    eta is (n,d), delta/sigma_c2 (k,), sigma_y2 (B,), mu_theta (d,),
    Sigma_theta (d,d), and V_prop (n,d,d). Each V_prop[i] is the complete
    SPD proposal covariance, including any caller-selected scale. Inputs
    must be valid float64 arrays; close over the fixed target when using JIT.

    This model-specific adapter supplies site proposals and sequential
    conditioning to BlackJAX's standard MH kernel. JAX draws the Gaussian;
    BlackJAX splits proposal/acceptance keys and performs accept/reject.
    Fixed eta-space Gaussian proposals are symmetric, so the proposal ratio
    is exactly one. The full joint theta target includes the coordinate
    Jacobian and all inter-site dependencies. No clipping or jitter is used.

    The current density is recomputed on every call. Within a sweep the
    accepted position/density feeds the next site; rejected states repeat.
    Negative-infinite candidate densities undergo ordinary MH rejection.
    NaN/+inf evaluations and invalid current states fail in the checkified
    chain driver; no target-level nonfinite-to-minus-infinity conversion occurs.
    """

    def logdensity_fn(position):
        value = target.theta_only_collapsed(
            position, delta, sigma_y2, mu_theta, Sigma_theta, sigma_c2
        )
        # -inf is a legitimate zero-density proposal; NaN/+inf is not.
        checkify.debug_check(~jnp.isnan(value) & ~jnp.isposinf(value),
                             "Metropolis target evaluation produced NaN/+inf")
        return value

    kernel = random_walk.build_rmh()
    initial_state = random_walk.init(eta, logdensity_fn)
    checkify.debug_check(jnp.isfinite(initial_state.logdensity), "MH current density is nonfinite")
    site_keys = jax.random.split(key, eta.shape[0])

    def step(state, inputs):
        i, site_key, covariance = inputs

        def proposal_generator(proposal_key, position):
            proposal_i = jax.random.multivariate_normal(
                proposal_key, position[i], covariance,
                dtype=jnp.float64, method="cholesky",
            )
            return position.at[i].set(proposal_i)

        next_state, info = kernel(
            site_key, state, logdensity_fn, proposal_generator
        )
        return next_state, (info.acceptance_rate, info.is_accepted)

    final_state, (acceptance_rate, is_accepted) = jax.lax.scan(
        step, initial_state, (jnp.arange(eta.shape[0]), site_keys, V_prop)
    )
    return final_state.position, RandomWalkSweepInfo(
        acceptance_rate, is_accepted, final_state.logdensity
    )


def update_collapsed_random_walk(
    key: Array,
    target: CalibrationTarget,
    eta: Array,
    delta: Array,
    sigma_y2: Array,
    mu_theta: Array,
    Sigma_theta: Array,
    sigma_c2: Array,
    V_prop: Array,
) -> tuple[Array, RandomWalkSweepInfo]:
    """Validate current conditioning/tuning and perform one collapsed sweep.

    Shapes and return values follow collapsed_random_walk_sweep. This host
    boundary rejects invalid initialization, non-SPD proposal covariances,
    and singular unjittered GP geometry even with fixed library jitter.
    It reports unusable final states without repair or redraws. No tuning,
    coefficient refresh, or outer Gibbs sweep is performed here.
    """

    checked = _validate_collapsed_inputs(
        target, eta, delta, sigma_y2, mu_theta, Sigma_theta, sigma_c2, V_prop
    )
    eta_new, info = collapsed_random_walk_sweep(key, target, *checked)
    if (
        not np.all(np.isfinite(np.asarray(eta_new)))
        or not np.isfinite(float(info.logdensity))
        or not np.all(np.isfinite(np.asarray(info.acceptance_rate)))
    ):
        raise FloatingPointError("Collapsed random walk produced unusable values")
    target.gp.validate_field_sites(target.coordinates.eta_to_theta_tilde(eta_new))
    return eta_new, info


def _validate_collapsed_inputs(
    target, eta, delta, sigma_y2, mu_theta, Sigma_theta, sigma_c2, V_prop
):
    """Share this model's shapes, support, and GP checks across MH and MALA.

    Standard array validators cannot enforce calibration-specific dimensions
    or structural GP geometry. No repair or random draws occur here.
    """

    if not jax.config.jax_enable_x64:
        raise ValueError("Collapsed sampling requires jax_enable_x64=True")
    n = target.C_theta.shape[0]
    k = target.gp.F_s.shape[1]
    d = target.gp.theta_s_tilde.shape[1]
    shapes = (
        ("eta", eta, (n, d)), ("delta", delta, (k,)),
        ("sigma_y2", sigma_y2, (len(target.branch_sizes),)),
        ("mu_theta", mu_theta, (d,)),
        ("Sigma_theta", Sigma_theta, (d, d)),
        ("sigma_c2", sigma_c2, (k,)), ("V_prop", V_prop, (n, d, d)),
    )
    checked = []
    for name, value, shape in shapes:
        value_np = np.asarray(value, dtype=np.float64)
        if value_np.shape != shape or not np.all(np.isfinite(value_np)):
            raise ValueError(f"{name} must be finite with shape {shape}")
        if name in ("sigma_y2", "sigma_c2") and np.any(value_np <= 0):
            raise ValueError(f"{name} must be positive")
        if name in ("Sigma_theta", "V_prop"):
            if not np.array_equal(value_np, np.swapaxes(value_np, -1, -2)):
                raise ValueError(f"{name} must be symmetric")
            try:
                np.linalg.cholesky(value_np)
            except np.linalg.LinAlgError as error:
                raise ValueError(f"{name} must be positive definite") from error
        checked.append(jnp.asarray(value_np))

    eta_checked, delta_checked, noise, mean, covariance, variances, proposal = checked
    target.gp.validate_field_sites(
        target.coordinates.eta_to_theta_tilde(eta_checked)
    )
    initial_logdensity = target.theta_only_collapsed(
        eta_checked, delta_checked, noise, mean, covariance, variances
    )
    if not np.isfinite(float(initial_logdensity)):
        raise ValueError("Current collapsed log density must be finite")
    return tuple(checked)


def _validate_mala_tuning(epsilon, V_prop) -> float:
    """Check the note's scalar scale and representable diffusion covariance.

    This boundary rejects overflow/underflow instead of modifying tuning.
    Other proposal shape/SPD checks belong to the shared input validator.
    """

    value = np.asarray(epsilon)
    if value.shape != () or value.dtype.kind not in "fiu":
        raise ValueError("epsilon must be a finite positive scalar")
    value = float(value)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        step_size = np.float64(value) ** 2 / 2
        scaled = (2 * step_size) * np.asarray(V_prop)
    if (
        not np.isfinite(value) or value <= 0
        or not np.isfinite(step_size) or step_size <= 0
        or not np.all(np.isfinite(scaled))
    ):
        raise ValueError("epsilon must give a finite positive diffusion scale")
    try:
        np.linalg.cholesky(scaled)
    except np.linalg.LinAlgError as error:
        raise ValueError("epsilon^2 V_prop must be positive definite") from error
    return value


def update_collapsed_mala(
    key: Array,
    target: CalibrationTarget,
    eta: Array,
    delta: Array,
    sigma_y2: Array,
    mu_theta: Array,
    Sigma_theta: Array,
    sigma_c2: Array,
    V_prop: Array,
    epsilon: Array,
) -> tuple[Array, MALASweepInfo]:
    """Validate current inputs/gradients and run one fixed-tuning MALA sweep.

    Shapes follow collapsed_mala_sweep. Require finite current target and
    gradients, SPD preconditioners and representable positive epsilon^2.
    Final gradients/geometry are checked without repair or redraws. The
    caller must immediately refresh c_f after this collapsed theta update.
    """

    checked = _validate_collapsed_inputs(
        target, eta, delta, sigma_y2, mu_theta, Sigma_theta, sigma_c2, V_prop
    )
    epsilon = _validate_mala_tuning(epsilon, checked[-1])
    density = lambda position: target.theta_only_collapsed(position, *checked[1:6])
    if not np.all(np.isfinite(np.asarray(jax.grad(density)(checked[0])))):
        raise ValueError("Current collapsed gradient must be finite")
    eta_new, info = collapsed_mala_sweep(key, target, *checked, epsilon)
    if (
        not np.all(np.isfinite(np.asarray(eta_new)))
        or not np.isfinite(float(info.logdensity))
        or not np.all(np.isfinite(np.asarray(info.acceptance_rate)))
        or not np.all(np.isfinite(np.asarray(jax.grad(density)(eta_new))))
    ):
        raise FloatingPointError("Collapsed MALA produced unusable values")
    target.gp.validate_field_sites(target.coordinates.eta_to_theta_tilde(eta_new))
    return eta_new, info
