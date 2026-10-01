"""Standard BlackJAX NUTS over all sites or sequential conditional site blocks."""

from typing import NamedTuple
from numbers import Integral

import jax

import jax.numpy as jnp
from blackjax.mcmc import nuts
from jax import Array
from bayesiancalibration.validation import check_quantity

from bayesiancalibration.state import CalibrationState
from bayesiancalibration.targets import CalibrationTarget


class NUTSSweepInfo(NamedTuple):
    """Scalar (all-site) or per-block NUTS diagnostics; tuning describes this transition, not the next.

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


def nuts_blocks(n: int, block_size: int | None) -> tuple[tuple[int, int], ...]:
    """Static contiguous site slices, including a smaller final block.

    A shared helper keeps kernel, tuning and adaptation block layouts identical.
    None selects the original all-site transition.
    """
    if block_size is None:
        return ((0, n),)
    if (isinstance(block_size, bool) or not isinstance(block_size, Integral)
        or not 1 <= block_size <= n):
        raise ValueError("NUTS block_size must be an integer in [1, number of sites]")
    return tuple((start, min(start + block_size, n)) for start in range(0, n, block_size))


def nuts_sweep(
    key: Array, target: CalibrationTarget, state: CalibrationState,
    step_size: Array, inverse_mass_matrix: Array | tuple[Array, ...],
    *, max_num_doublings: int = 10, divergence_threshold: float = 1000,
    collapsed: bool = False, block_size: int | None = None,
) -> tuple[Array, NUTSSweepInfo]:
    """Sequential complete conditional trees, with current outside-block eta.

    A block counts sites, not scalar coordinates. Multiple blocks use
    random.split(theta_key, number_of_blocks) in fixed site-major order;
    one block retains the original unsplit key and diagnostics. Each tree
    conditions on the latest previous-block results; c_f and other Gibbs
    variables stay fixed throughout this sweep.

    Multi-block diagnostics are (B,) except logdensity (the final complete
    theta target value) and inverse_mass_matrix (a tuple of B possibly unequal
    arrays). step_size is (B,); each mass has its own block dimension.
    """
    blocks = nuts_blocks(state.eta.shape[0], block_size)
    if len(blocks) == 1:
        return _nuts_transition(
            key, target, state, step_size, inverse_mass_matrix,
            max_num_doublings=max_num_doublings,
            divergence_threshold=divergence_threshold, collapsed=collapsed, block_index=0,
        )
    eta, diagnostics = state.eta, []
    for index, ((start, stop), block_key) in enumerate(zip(
        blocks, jax.random.split(key, len(blocks))
    )):
        eta, info = _nuts_transition(
            block_key, target, state._replace(eta=eta), step_size[index],
            inverse_mass_matrix[index], site_slice=(start, stop),
            max_num_doublings=max_num_doublings,
            divergence_threshold=divergence_threshold, collapsed=collapsed, block_index=index,
        )
        diagnostics.append(info)
    values = []
    for name in NUTSSweepInfo._fields:
        if name == "logdensity":
            values.append(diagnostics[-1].logdensity)
        elif name == "inverse_mass_matrix":
            values.append(tuple(info.inverse_mass_matrix for info in diagnostics))
        else:
            values.append(jnp.stack([getattr(info, name) for info in diagnostics]))
    return eta, NUTSSweepInfo(*values)


def _nuts_transition(
    key: Array, target: CalibrationTarget, state: CalibrationState,
    step_size: Array, inverse_mass_matrix: Array,
    *, max_num_doublings: int = 10, divergence_threshold: float = 1000,
    collapsed: bool = False, site_slice: tuple[int, int] | None = None, block_index: int = 0,
) -> tuple[Array, NUTSSweepInfo]:
    """One all-site transition; collapsed selects the integrated c_f target.

    eta is (n,d), flattened site first to (n*d,). Use BlackJAX's complete
    multinomial NUTS tree, velocity Verlet, stopping and selection rules.
    This adapter supplies the model-specific conditional and always calls
    nuts.init anew, because Gibbs updates invalidate density and gradient.
    No endpoint MH step is added. inverse_mass_matrix uses the library's
    convention directly; it is not manually inverted.
    """

    def embed(position):
        if site_slice is None:
            return position.reshape(state.eta.shape)
        start, stop = site_slice
        return state.eta.at[start:stop].set(position.reshape(stop-start, state.eta.shape[1]))

    def density(position):
        if collapsed:
            return target.theta_only_collapsed(
                embed(position), state.delta, state.sigma_y2,
                state.mu_theta, state.Sigma_theta, state.sigma_c2,
            )
        return target.theta_only_uncollapsed(
            embed(position), state.c_f, state.mu_theta,
            state.Sigma_theta, state.sigma_c2,
        )

    position = state.eta if site_slice is None else state.eta[site_slice[0]:site_slice[1]]
    initial = nuts.init(position.reshape(-1), density)
    check_quantity(
        jnp.isfinite(initial.logdensity) & jnp.all(jnp.isfinite(initial.logdensity_grad)),
        update="eta", quantity="density_gradient", role="current", block=block_index,
        description="NUTS current density/gradient is nonfinite",
    )
    updated, info = nuts.build_kernel(divergence_threshold=divergence_threshold)(
        key, initial, density, step_size, inverse_mass_matrix, max_num_doublings
    )
    check_quantity(
        jnp.isfinite(updated.logdensity) & jnp.all(jnp.isfinite(updated.logdensity_grad)),
        update="eta", quantity="density_gradient", role="selected", block=block_index,
        description="NUTS selected density/gradient is nonfinite",
    )
    return embed(updated.position), NUTSSweepInfo(
        info.acceptance_rate, updated.logdensity, info.is_divergent,
        info.is_turning, info.num_integration_steps, info.num_trajectory_expansions,
        info.num_trajectory_expansions >= max_num_doublings,
        jnp.asarray(step_size), inverse_mass_matrix,
    )
