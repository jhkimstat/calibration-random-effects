"""Small, auditable Gaussian calculations used by the calibration model.

Model-specific names follow Modeling.md and Sampling.md. For n field sites and
k coefficients per site, vectors are stacked site first, then loading and
unloading coefficients within each site.

Shared projected-model arguments:
    y_tilde: Projected observations, shape (n*k,).
    c_f: Latent field coefficients, shape (n*k,).
    m_f_given_s: Mean of c_f given the fixed library c_s, shape (n*k,).
    Sigma_f_given_s: Full covariance of c_f given c_s, shape (n*k, n*k).
        This derived alias denotes C_f_given_s kron Sigma_c, not the (n, n)
        input covariance C_f_given_s. Dense fixtures may supply a general SPD
        covariance to test the Gaussian identities independently of the GP.
    R: Block-diagonal projection factor, shape (n*k, n*k).
    d_delta: Derived alias for ones(n) kron delta, shape (n*k,), distinct
        from the shared discrepancy delta of shape (k,).
    Omega_y: Derived alias for I_n kron Sigma_y, shape (n*k, n*k), distinct
        from the per-site noise covariance Sigma_y of shape (k, k).

All other conditioning parameters are implicit in these supplied moments.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array


def projected_marginal_moments(
    m_f_given_s: Array,
    Sigma_f_given_s: Array,
    R: Array,
    d_delta: Array,
    Omega_y: Array,
) -> tuple[Array, Array]:
    """Return collapsed observation moments (m_y, V_y).

    The model is c_f | c_s ~ N(m_f_given_s, Sigma_f_given_s) and
    y_tilde | c_f ~ N(R @ (c_f + d_delta), Omega_y). Returned shapes are
    (n*k,) and (n*k, n*k); argument definitions are in the module docstring.
    """

    m_y = R @ (m_f_given_s + d_delta)
    V_y = R @ Sigma_f_given_s @ R.T + Omega_y
    V_y = 0.5 * (V_y + V_y.T)
    return m_y, V_y


def projected_conditional_coefficient_moments(
    y_tilde: Array,
    m_f_given_s: Array,
    Sigma_f_given_s: Array,
    R: Array,
    d_delta: Array,
    Omega_y: Array,
) -> tuple[Array, Array]:
    """Return (m_f, V_f) of c_f conditional on y_tilde and the library.

    Shapes are (n*k,) and (n*k, n*k). See the module docstring for arguments.
    """

    m_y, V_y = projected_marginal_moments(
        m_f_given_s,
        Sigma_f_given_s,
        R,
        d_delta,
        Omega_y,
    )
    L_y = jnp.linalg.cholesky(V_y)  # V_y = L_y @ L_y.T
    # Derived cross-covariance Cov(c_f, y_tilde | c_s), shape (n*k, n*k).
    Sigma_fy = Sigma_f_given_s @ R.T
    V_y_solve_residual = jsp.linalg.cho_solve(
        (L_y, True),
        y_tilde - m_y,
    )
    V_y_solve_Sigma_yf = jsp.linalg.cho_solve(
        (L_y, True),
        Sigma_fy.T,
    )
    m_f = m_f_given_s + Sigma_fy @ V_y_solve_residual
    V_f = Sigma_f_given_s - Sigma_fy @ V_y_solve_Sigma_yf
    V_f = 0.5 * (V_f + V_f.T)
    return m_f, V_f


def projected_joint_logpdf(
    y_tilde: Array,
    c_f: Array,
    m_f_given_s: Array,
    Sigma_f_given_s: Array,
    R: Array,
    d_delta: Array,
    Omega_y: Array,
) -> Array:
    """Return log p(c_f | c_s) + log p(y_tilde | c_f), a scalar.

    Other conditioning variables are implicit. This is the library-conditioned
    joint density, not the full joint density including p(c_s | Sigma_c).
    See the module docstring for argument definitions and shapes.
    """

    # Conditional observation mean, distinct from the collapsed mean m_y.
    m_y_given_cf = R @ (c_f + d_delta)
    return jsp.stats.multivariate_normal.logpdf(
        c_f,
        m_f_given_s,
        Sigma_f_given_s,
    ) + jsp.stats.multivariate_normal.logpdf(
        y_tilde,
        m_y_given_cf,
        Omega_y,
    )


def projected_collapsed_logpdf(
    y_tilde: Array,
    m_f_given_s: Array,
    Sigma_f_given_s: Array,
    R: Array,
    d_delta: Array,
    Omega_y: Array,
) -> Array:
    """Return scalar log p(y_tilde | c_s), integrating out c_f.

    See the module docstring for argument definitions and shapes.
    """

    m_y, V_y = projected_marginal_moments(
        m_f_given_s,
        Sigma_f_given_s,
        R,
        d_delta,
        Omega_y,
    )
    return jsp.stats.multivariate_normal.logpdf(
        y_tilde,
        m_y,
        V_y,
    )


def projected_conditional_coefficient_logpdf(
    c_f: Array,
    y_tilde: Array,
    m_f_given_s: Array,
    Sigma_f_given_s: Array,
    R: Array,
    d_delta: Array,
    Omega_y: Array,
) -> Array:
    """Return scalar log p(c_f | y_tilde, c_s), including normalization.

    See the module docstring for argument definitions and shapes.
    """

    m_f, V_f = (
        projected_conditional_coefficient_moments(
            y_tilde,
            m_f_given_s,
            Sigma_f_given_s,
            R,
            d_delta,
            Omega_y,
        )
    )
    return jsp.stats.multivariate_normal.logpdf(
        c_f,
        m_f,
        V_f,
    )
