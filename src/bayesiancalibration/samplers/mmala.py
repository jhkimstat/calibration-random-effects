"""Simplified collapsed MMALA with the source note's conditional Fisher metric."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from blackjax.mcmc import random_walk
from jax import Array
from bayesiancalibration.validation import check_quantity

from bayesiancalibration.linalg import projected_marginal_moments
from bayesiancalibration.samplers.metropolis import (
    MALASweepInfo, _validate_collapsed_inputs, _validate_mala_tuning,
)
from bayesiancalibration.targets import CalibrationTarget


def site_observation_conditional_moments(
    target: CalibrationTarget, eta: Array, i: Array,
    delta: Array, sigma_y2: Array, sigma_c2: Array,
) -> tuple[Array, Array]:
    """Return m_y,i|-i (k,) and V_y,i|-i (k,k), conditioning on observed y_-i.

    This model-specific assembly handles site-major indices and varying R_i.
    Standard GP moments and Cholesky solves supply the Gaussian conditional;
    no target term is omitted here. i may be a scalar traced site index.
    """

    n, k = eta.shape[0], target.gp.F_s.shape[1]
    m_f, _, V_f = target.gp.conditional_moments(
        target.coordinates.eta_to_theta_tilde(eta), sigma_c2
    )
    d_delta, Omega_y = target.observation_arrays(delta, sigma_y2)
    m_y, V_y = projected_marginal_moments(m_f, V_f, target.R, d_delta, Omega_y)
    selected = i*k + jnp.arange(k)
    m_i = m_y[selected]
    V_ii = V_y[jnp.ix_(selected, selected)]
    if n == 1:
        return m_i, V_ii
    other_sites = jnp.arange(n-1) + (jnp.arange(n-1) >= i)
    other = (other_sites[:, None]*k + jnp.arange(k)).reshape(-1)
    V_io = V_y[jnp.ix_(selected, other)]
    L_o = jnp.linalg.cholesky(V_y[jnp.ix_(other, other)])
    m_cond = m_i + V_io @ jsp.linalg.cho_solve((L_o, True), target.y_tilde[other]-m_y[other])
    V_cond = V_ii - V_io @ jsp.linalg.cho_solve((L_o, True), V_io.T)
    return m_cond, (V_cond + V_cond.T) / 2


def collapsed_mmala_metric(
    target: CalibrationTarget, eta: Array, i: Array,
    delta: Array, sigma_y2: Array, Sigma_theta: Array, sigma_c2: Array,
    epsilon_G: Array,
) -> Array:
    """Return the precision-like G_i (d,d), not a covariance preconditioner.

    G = conditional-observation Fisher + D V_theta,i|-i^-1 D + Jacobian
    curvature + epsilon_G I. epsilon_G is a declared positive, fixed ridge.
    JAX jacfwd differentiates both conditional mean and covariance in eta_i.
    This custom metric is required by Sampling theta.md; a generic MALA or
    HMC metric does not implement these model-specific conditional terms.
    Identity coordinates have D=I and zero Jacobian curvature. Finite bounds
    use their actual standardized widths and sigmoid probabilities.
    """

    n, d = eta.shape
    def moments(site_eta):
        return site_observation_conditional_moments(
            target, eta.at[i].set(site_eta), i, delta, sigma_y2, sigma_c2
        )
    _, V_y = moments(eta[i])
    J_m, J_V = jax.jacfwd(moments)(eta[i])
    L_y = jnp.linalg.cholesky(V_y)
    solved_derivatives = jax.vmap(lambda derivative: jsp.linalg.cho_solve(
        (L_y, True), derivative
    ))(jnp.moveaxis(J_V, -1, 0))
    fisher = J_m.T @ jsp.linalg.cho_solve((L_y, True), J_m)
    fisher += 0.5 * jnp.einsum("aij,bji->ab", solved_derivatives, solved_derivatives)
    C = target.C_theta
    conditional_scale = C[i, i]
    if n > 1:
        other = jnp.arange(n-1) + (jnp.arange(n-1) >= i)
        cross = C[i, other]
        L_o = jnp.linalg.cholesky(C[jnp.ix_(other, other)])
        conditional_scale -= cross @ jsp.linalg.cho_solve((L_o, True), cross)
    D = jnp.eye(d, dtype=jnp.float64)
    curvature = jnp.zeros(d, dtype=jnp.float64)
    if target.coordinates.bounded:
        p = jax.nn.sigmoid(eta[i])
        width = target.coordinates.u_tilde - target.coordinates.l_tilde
        D = jnp.diag(width*p*(1-p))
        curvature = 2*p*(1-p)
    L_prior = jnp.linalg.cholesky(conditional_scale * Sigma_theta)
    G = (fisher + D @ jsp.linalg.cho_solve((L_prior, True), D)
         + jnp.diag(curvature) + epsilon_G * jnp.eye(d, dtype=jnp.float64))
    return (G + G.T) / 2


def mmala_proposal_moments(
    target: CalibrationTarget, eta: Array, i: Array,
    delta: Array, sigma_y2: Array, mu_theta: Array, Sigma_theta: Array,
    sigma_c2: Array, epsilon: Array, epsilon_G: Array, *, role="current", current_origin=None,
) -> tuple[Array, Array]:
    """Return exact site proposal mean (d,) and covariance (d,d).

    A standard solve constructs G^-1 for the Gaussian sampler/density; no
    explicit inverse, Christoffel drift, or metric-volume target term is used.
    Recompute from the supplied origin for both proposal directions.
    """

    def density(site_eta):
        return target.theta_only_collapsed(
            eta.at[i].set(site_eta), delta, sigma_y2, mu_theta, Sigma_theta, sigma_c2
        )
    gradient = jax.grad(density)(eta[i])
    G = collapsed_mmala_metric(target, eta, i, delta, sigma_y2, Sigma_theta, sigma_c2, epsilon_G)
    L_G = jnp.linalg.cholesky(G)
    mean = eta[i] + epsilon**2 / 2 * jsp.linalg.cho_solve((L_G, True), gradient)
    covariance = epsilon**2 * jsp.linalg.cho_solve((L_G, True), jnp.eye(eta.shape[1]))
    valid = (jnp.all(jnp.isfinite(mean)) & jnp.all(jnp.isfinite(covariance))
             & jnp.all(jnp.diag(covariance) > 0))
    if current_origin is None:
        check_quantity(valid, update="eta", quantity="gradient_metric_diffusion", role=role, site=i,
                       criterion="finite_positive_diffusion",
                       description="MMALA gradient/metric/diffusion is invalid")
    else:
        # BlackJAX calls the same density for both directions. Label the
        # origin using equality only for evidence; proposal decisions use
        # the same unmodified moments and asymmetric density as before.
        check_quantity(valid | ~current_origin, update="eta", quantity="gradient_metric_diffusion",
                       role="current", site=i, criterion="finite_positive_diffusion",
                       description="MMALA gradient/metric/diffusion is invalid")
        check_quantity(valid | current_origin, update="eta", quantity="gradient_metric_diffusion",
                       role="proposal", site=i, criterion="finite_positive_diffusion",
                       description="MMALA gradient/metric/diffusion is invalid")
    return mean, (covariance + covariance.T) / 2


def collapsed_mmala_sweep(
    key: Array, target: CalibrationTarget, eta: Array,
    delta: Array, sigma_y2: Array, mu_theta: Array, Sigma_theta: Array,
    sigma_c2: Array, epsilon: Array, epsilon_G: Array,
) -> tuple[Array, MALASweepInfo]:
    """Sequential site MMALA with exact asymmetric BlackJAX MH correction.

    Shapes: eta (n,d), delta/sigma_c2 (k,), sigma_y2 (B,), mu_theta (d,),
    Sigma_theta (d,d), scalar epsilon/epsilon_G. Ridge and epsilon stay fixed
    through each forward/reverse calculation. Metrics depend on position in
    production and use the latest accepted other sites. Standard Gaussian
    generation, logpdf and MH primitives handle proposals and rejection of
    invalid energies. No changing metric/density cache survives a site step.
    """

    keys = jax.random.split(key, eta.shape[0])
    kernel = random_walk.build_rmh()
    def step(position, inputs):
        i, site_key = inputs
        def density(site_eta, *, role="proposal"):
            value = target.theta_only_collapsed(
                position.at[i].set(site_eta), delta, sigma_y2,
                mu_theta, Sigma_theta, sigma_c2,
            )
            check_quantity(~jnp.isnan(value) & ~jnp.isposinf(value),
                           update="eta", quantity="logdensity", role=role, site=i,
                           criterion="not_nan_or_positive_infinity",
                           description="MMALA target evaluation produced NaN/+inf")
            return value
        def moments(site_eta, *, role, current_origin=None):
            return mmala_proposal_moments(
                target, position.at[i].set(site_eta), i, delta, sigma_y2,
                mu_theta, Sigma_theta, sigma_c2, epsilon, epsilon_G, role=role, current_origin=current_origin,
            )
        def propose(proposal_key, origin):
            mean, covariance = moments(origin, role="current")
            return jax.random.multivariate_normal(
                proposal_key, mean, covariance, dtype=jnp.float64, method="cholesky"
            )
        def proposal_logdensity(origin, destination):
            mean, covariance = moments(origin.position, role="proposal",
                                       current_origin=jnp.all(origin.position == position[i]))
            return jsp.stats.multivariate_normal.logpdf(destination.position, mean, covariance)
        updated, info = kernel(
            site_key, random_walk.init(position[i], lambda site_eta: density(site_eta, role="current")), density,
            propose, proposal_logdensity,
        )
        return position.at[i].set(updated.position), (info.acceptance_rate, info.is_accepted)
    position, (rates, accepted) = jax.lax.scan(step, eta, (jnp.arange(eta.shape[0]), keys))
    density = target.theta_only_collapsed(
        position, delta, sigma_y2, mu_theta, Sigma_theta, sigma_c2
    )
    return position, MALASweepInfo(rates, accepted, density)


def validate_mmala_geometry(
    target, eta, delta, sigma_y2, mu_theta, Sigma_theta, sigma_c2, epsilon, epsilon_G
):
    """Check finite gradients, metrics and representable current diffusion.

    Array/metric libraries cannot validate this model's current site geometry.
    Reject unusable ridge or proposals without repair, clipping or redraws.
    """

    ridge = np.asarray(epsilon_G)
    if (ridge.shape != () or ridge.dtype.kind not in "fiu"
        or not np.isfinite(ridge) or ridge <= 0):
        raise ValueError("epsilon_G must be a declared finite positive scalar")
    epsilon = _validate_mala_tuning(epsilon, np.eye(eta.shape[1]))
    for i in range(eta.shape[0]):
        mean, covariance = mmala_proposal_moments(
            target, eta, i, delta, sigma_y2, mu_theta, Sigma_theta, sigma_c2, epsilon, ridge
        )
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
            raise ValueError("Current MMALA gradient/metric/proposal must be finite")
        try:
            np.linalg.cholesky(np.asarray(covariance))
        except np.linalg.LinAlgError as error:
            raise ValueError("Current MMALA diffusion must be positive definite") from error
    return epsilon, float(ridge)


def update_collapsed_mmala(
    key: Array, target: CalibrationTarget, eta: Array,
    delta: Array, sigma_y2: Array, mu_theta: Array, Sigma_theta: Array,
    sigma_c2: Array, epsilon: Array, epsilon_G: Array,
) -> tuple[Array, MALASweepInfo]:
    """Checked current-input boundary; caller must immediately refresh c_f."""

    n, d = target.C_theta.shape[0], target.gp.theta_s_tilde.shape[1]
    checked = _validate_collapsed_inputs(
        target, eta, delta, sigma_y2, mu_theta, Sigma_theta, sigma_c2,
        jnp.tile(jnp.eye(d, dtype=jnp.float64), (n, 1, 1)),
    )[:-1]
    epsilon, epsilon_G = validate_mmala_geometry(target, *checked, epsilon, epsilon_G)
    position, info = collapsed_mmala_sweep(key, target, *checked, epsilon, epsilon_G)
    if not np.all(np.isfinite(info.acceptance_rate)) or not np.isfinite(info.logdensity):
        raise FloatingPointError("MMALA sweep produced unusable diagnostics")
    target.gp.validate_field_sites(target.coordinates.eta_to_theta_tilde(position))
    validate_mmala_geometry(target, position, *checked[1:], epsilon, epsilon_G)
    return position, info
