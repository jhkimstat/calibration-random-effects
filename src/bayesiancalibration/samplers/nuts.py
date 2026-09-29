"""Standard BlackJAX NUTS over the complete site-major eta block."""

from typing import NamedTuple

import jax.numpy as jnp
from blackjax.mcmc import nuts
from jax import Array
from jax.experimental import checkify

from bayesiancalibration.state import CalibrationState
from bayesiancalibration.targets import CalibrationTarget


class NUTSSweepInfo(NamedTuple):
    """Scalar NUTS diagnostics; tuning describes this transition, not the next.

    inverse_mass_matrix is (n*d,) for diagonal or (n*d,n*d) for dense and
    Kronecker structures; M^{-1} = Gamma_site ⊗ Gamma_param in the latter. No density/gradient is cached
    between Gibbs sweeps. reached_max_doublings records the expansion cap;
    is_turning/is_divergent distinguish the other termination conditions.
    """

    acceptance_rate: Array
    logdensity: Array
    is_divergent: Array
    is_turning: Array
    num_integration_steps: Array
    num_trajectory_expansions: Array
    reached_max_doublings: Array
    step_size: Array
    inverse_mass_matrix: Array


def nuts_sweep(
    key: Array, target: CalibrationTarget, state: CalibrationState,
    step_size: Array, inverse_mass_matrix: Array,
    *, max_num_doublings: int = 10, divergence_threshold: float = 1000,
    collapsed: bool = False,
) -> tuple[Array, NUTSSweepInfo]:
    """One all-site transition; collapsed selects the integrated c_f target.

    eta is (n,d), flattened site first to (n*d,). Use BlackJAX's complete
    multinomial NUTS tree, velocity Verlet, stopping and selection rules.
    This adapter supplies the model-specific conditional and always calls
    nuts.init anew, because Gibbs updates invalidate density and gradient.
    No endpoint MH step is added. inverse_mass_matrix uses the library's
    convention directly; it is not manually inverted.
    """

    def density(position):
        if collapsed:
            return target.theta_only_collapsed(
                position.reshape(state.eta.shape), state.delta, state.sigma_y2,
                state.mu_theta, state.Sigma_theta, state.sigma_c2,
            )
        return target.theta_only_uncollapsed(
            position.reshape(state.eta.shape), state.c_f, state.mu_theta,
            state.Sigma_theta, state.sigma_c2,
        )

    initial = nuts.init(state.eta.reshape(-1), density)
    checkify.debug_check(
        jnp.isfinite(initial.logdensity) & jnp.all(jnp.isfinite(initial.logdensity_grad)),
        "NUTS current density/gradient is nonfinite",
    )
    updated, info = nuts.build_kernel(divergence_threshold=divergence_threshold)(
        key, initial, density, step_size, inverse_mass_matrix, max_num_doublings
    )
    checkify.debug_check(
        jnp.isfinite(updated.logdensity) & jnp.all(jnp.isfinite(updated.logdensity_grad)),
        "NUTS selected density/gradient is nonfinite",
    )
    return updated.position.reshape(state.eta.shape), NUTSSweepInfo(
        info.acceptance_rate, updated.logdensity, info.is_divergent,
        info.is_turning, info.num_integration_steps, info.num_trajectory_expansions,
        info.num_trajectory_expansions >= max_num_doublings,
        jnp.asarray(step_size), inverse_mass_matrix,
    )
