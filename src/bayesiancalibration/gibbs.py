"""Conditional Gibbs updates for the projected calibration model.

Coefficient/observation vectors have shape (n*k,) and covariances (nk,nk),
stacked site first, then active branches within each site. Checked host entry
points rebuild moments from current conditioning variables; numerical draw
kernels are pure JAX and support JIT/vmap. No outer sweep lives here.
Shared discrepancy delta has shape (k,), with conditional moments m_delta
and V_delta distinct from the configured prior m_delta_0 and V_delta_0.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jax import Array

from bayesiancalibration.linalg import projected_marginal_moments
from bayesiancalibration.targets import CalibrationTarget


def sample_projected_coefficients(
    key: Array,
    y_tilde: Array,
    m_f_given_s: Array,
    Sigma_f_given_s: Array,
    R: Array,
    d_delta: Array,
    Omega_y: Array,
) -> Array:
    """Draw c_f exactly conditional on projected observations and the library.

    Arguments follow ``linalg.py``; all numerical arrays must be float64 and
    both supplied covariances positive definite. The caller supplies a fresh
    key per update; two independent subkeys draw the prior u and noise e.

    This model-specific correction handles arbitrary varying R_i and avoids
    factoring the subtractive posterior covariance. Standard JAX Gaussian
    draws and Cholesky solves provide the numerical primitives. No jitter or
    singular-covariance fallback is added; use ``refresh_field_coefficients``
    for host validation and failure reporting.
    """

    key_u, key_e = jax.random.split(key)
    u = jax.random.multivariate_normal(
        key_u, m_f_given_s, Sigma_f_given_s,
        dtype=jnp.float64, method="cholesky",
    )
    e = jax.random.multivariate_normal(
        key_e, jnp.zeros_like(y_tilde), Omega_y,
        dtype=jnp.float64, method="cholesky",
    )
    _, V_y = projected_marginal_moments(
        m_f_given_s, Sigma_f_given_s, R, d_delta, Omega_y
    )
    L_y = jnp.linalg.cholesky(V_y)
    y_0 = y_tilde - R @ d_delta
    correction = jsp.linalg.cho_solve((L_y, True), y_0 - R @ u - e)
    return u + Sigma_f_given_s @ R.T @ correction


def refresh_field_coefficients(
    key: Array,
    target: CalibrationTarget,
    eta: Array,
    delta: Array,
    sigma_y2: Array,
    sigma_c2: Array,
) -> Array:
    """Validate current state and return one exact field-coefficient draw.

    ``eta`` is (n,d), ``delta`` and ``sigma_c2`` are (k,), and ``sigma_y2`` is (B,)
    for the active branches. The return is site-major c_f, shape (n*k,).
    Site coordinates may be bounded or unbounded; neither changes this
    Gaussian conditional at fixed theta. mu_theta/Sigma_theta do not enter.

    This host orchestration reports unusable current states before a draw
    can become part of a chain, including singular unjittered GP factors.
    It is not JIT compatible; JIT the numerical kernel separately. The caller
    manages the next update key. No changing-state moments are cached.
    """

    if not jax.config.jax_enable_x64:
        raise ValueError("Coefficient refresh requires jax_enable_x64=True")
    n = target.C_theta.shape[0]
    k = target.gp.F_s.shape[1]
    d = target.gp.theta_s_tilde.shape[1]
    eta_np = np.asarray(eta, dtype=np.float64)
    if eta_np.shape != (n, d) or not np.all(np.isfinite(eta_np)):
        raise ValueError("eta must be finite with shape (n,d)")
    delta_np = np.asarray(delta, dtype=np.float64)
    if delta_np.shape != (k,) or not np.all(np.isfinite(delta_np)):
        raise ValueError("delta must be finite with shape (k,)")
    sigma_y2_np = np.asarray(sigma_y2, dtype=np.float64)
    if (
        sigma_y2_np.shape != (len(target.branch_sizes),)
        or not np.all(np.isfinite(sigma_y2_np)) or np.any(sigma_y2_np <= 0)
    ):
        raise ValueError("sigma_y2 must be finite and positive with shape (B,)")
    sigma_c2_np = np.asarray(sigma_c2, dtype=np.float64)
    if (
        sigma_c2_np.shape != (k,)
        or not np.all(np.isfinite(sigma_c2_np)) or np.any(sigma_c2_np <= 0)
    ):
        raise ValueError("sigma_c2 must be finite and positive with shape (k,)")

    theta_tilde = target.coordinates.eta_to_theta_tilde(jnp.asarray(eta_np))
    target.gp.validate_field_sites(theta_tilde)
    m_f_given_s, _, Sigma_f_given_s = target.gp.conditional_moments(
        theta_tilde, jnp.asarray(sigma_c2_np)
    )
    d_delta, Omega_y = target.observation_arrays(
        jnp.asarray(delta_np), jnp.asarray(sigma_y2_np)
    )
    Sigma_np = np.asarray(Sigma_f_given_s)
    if not np.all(np.isfinite(Sigma_np)):
        raise ValueError("Sigma_f_given_s must be finite")
    try:
        np.linalg.cholesky(Sigma_np)
    except np.linalg.LinAlgError as error:
        raise ValueError("Sigma_f_given_s must be positive definite") from error
    c_f = sample_projected_coefficients(
        key, target.y_tilde, m_f_given_s, Sigma_f_given_s,
        target.R, d_delta, Omega_y,
    )
    if not np.all(np.isfinite(np.asarray(c_f))):
        raise FloatingPointError("Coefficient refresh produced nonfinite values")
    return c_f


def discrepancy_conditional_moments(
    y_tilde: Array,
    c_f: Array,
    R: Array,
    Omega_y: Array,
    m_delta_0: Array,
    V_delta_0: Array,
) -> tuple[Array, Array]:
    """Return discrepancy conditional m_delta (k,) and V_delta (k,k).

    y_tilde/c_f are (n*k,), R/Omega_y are (n*k,n*k), and prior moments
    m_delta_0/V_delta_0 are (k,)/(k,k). Inputs must be valid float64 arrays
    with positive-definite prior/noise covariances. Stacking is site-major.

    This model-specific kernel assembles the source-note precision and
    information vector for a discrepancy shared by all sites. Standard
    Cholesky/triangular solves implement the displayed inverses; the returned
    covariance is needed by the standard Gaussian sampler. No jitter is used.
    """

    k = m_delta_0.shape[0]
    n = y_tilde.shape[0] // k
    I_k = jnp.eye(k, dtype=jnp.float64)
    # H_delta maps one shared delta into projected observations, shape (nk,k).
    H_delta = R @ jnp.tile(I_k, (n, 1))
    L_delta_0 = jnp.linalg.cholesky(V_delta_0)
    L_y = jnp.linalg.cholesky(Omega_y)
    H_whitened = jsp.linalg.solve_triangular(L_y, H_delta, lower=True)
    residual_whitened = jsp.linalg.solve_triangular(
        L_y, y_tilde - R @ c_f, lower=True
    )
    Lambda_delta = (
        jsp.linalg.cho_solve((L_delta_0, True), I_k)
        + H_whitened.T @ H_whitened
    )
    Lambda_delta = 0.5 * (Lambda_delta + Lambda_delta.T)
    h_delta = (
        jsp.linalg.cho_solve((L_delta_0, True), m_delta_0)
        + H_whitened.T @ residual_whitened
    )
    L_delta = jnp.linalg.cholesky(Lambda_delta)
    m_delta = jsp.linalg.cho_solve((L_delta, True), h_delta)
    V_delta = jsp.linalg.cho_solve((L_delta, True), I_k)
    V_delta = 0.5 * (V_delta + V_delta.T)
    return m_delta, V_delta


def sample_discrepancy(
    key: Array,
    y_tilde: Array,
    c_f: Array,
    R: Array,
    Omega_y: Array,
    m_delta_0: Array,
    V_delta_0: Array,
) -> Array:
    """Draw shared delta (k,) from its exact Gaussian full conditional.

    Model conditioning uses ``discrepancy_conditional_moments``; generation
    uses the standard JAX Gaussian sampler. This pure kernel is JIT/vmap
    compatible on valid float64 inputs and consumes one caller-supplied key.
    Use a fresh key for each update; ``update_discrepancy`` validates state.
    """

    m_delta, V_delta = discrepancy_conditional_moments(
        y_tilde, c_f, R, Omega_y, m_delta_0, V_delta_0
    )
    return jax.random.multivariate_normal(
        key, m_delta, V_delta, dtype=jnp.float64, method="cholesky"
    )


def update_discrepancy(
    key: Array,
    target: CalibrationTarget,
    c_f: Array,
    sigma_y2: Array,
) -> Array:
    """Check current coefficients/noise and return one shared delta draw.

    c_f is (n*k,) and sigma_y2 is (B,) for the active branches. Rebuilds
    moments each call, using the configured discrepancy prior.
    Prior moments are target.m_delta_0 (k,) and target.V_delta_0 (k,k).

    Host validation reports invalid states/nonfinite draws before recording
    a chain state. JIT the pure numerical kernels separately. This update
    conditions on current c_f in both site-bounds modes; theta, GP variances,
    and spatial hyperparameters have no additional role at fixed c_f.
    """

    if not jax.config.jax_enable_x64:
        raise ValueError("Discrepancy update requires jax_enable_x64=True")
    n = target.C_theta.shape[0]
    k = target.gp.F_s.shape[1]
    c_f_np = np.asarray(c_f, dtype=np.float64)
    if c_f_np.shape != (n * k,) or not np.all(np.isfinite(c_f_np)):
        raise ValueError("c_f must be finite with shape (n*k,)")
    sigma_y2_np = np.asarray(sigma_y2, dtype=np.float64)
    if (
        sigma_y2_np.shape != (len(target.branch_sizes),)
        or not np.all(np.isfinite(sigma_y2_np)) or np.any(sigma_y2_np <= 0)
    ):
        raise ValueError("sigma_y2 must be finite and positive with shape (B,)")
    _, Omega_y = target.observation_arrays(
        jnp.zeros(k, dtype=jnp.float64), jnp.asarray(sigma_y2_np)
    )
    delta = sample_discrepancy(
        key, target.y_tilde, jnp.asarray(c_f_np), target.R,
        Omega_y, target.m_delta_0, target.V_delta_0,
    )
    if not np.all(np.isfinite(np.asarray(delta))):
        raise FloatingPointError("Discrepancy update produced nonfinite values")
    return delta


def sample_inverse_gamma(key: Array, shape: Array, scale: Array) -> Array:
    """Draw independent shape/scale inverse-Gamma values, matching broadcasts.

    JAX has no inverse-Gamma sampler. Use its standard unit-scale log-Gamma
    sampler and the exact transformation v=scale/Gamma(shape,1). Log space
    avoids prematurely underflowing a small Gamma draw. Valid positive
    float64 inputs are required; no truncation or flooring is applied.
    """

    shape, scale = jnp.broadcast_arrays(
        jnp.asarray(shape, dtype=jnp.float64), jnp.asarray(scale, dtype=jnp.float64)
    )
    return jnp.exp(
        jnp.log(scale) - jax.random.loggamma(key, shape, dtype=jnp.float64)
    )


def branch_noise_conditional_parameters(
    y_tilde: Array,
    c_f: Array,
    delta: Array,
    R: Array,
    branch_sizes: tuple[int, ...],
    alpha_y_0: Array,
    beta_y_0: Array,
) -> tuple[Array, Array]:
    """Return active branch noise IG shapes alpha_y and scales beta_y (B,).

    y_tilde/c_f are site-major (n*k,), delta is (k,), R is (nk,nk), and
    prior parameters are (B,). branch_sizes is a static tuple for JIT, with
    sum k. This model-specific kernel sums only projected residuals within
    each branch; orthogonal QR residuals never enter this conditional.
    """

    k = delta.shape[0]
    n = y_tilde.shape[0] // k
    e_y = (y_tilde - R @ (c_f + jnp.tile(delta, n))).reshape(n, k)
    squared_residuals = []
    start = 0
    for size in branch_sizes:
        squared_residuals.append(jnp.sum(jnp.square(e_y[:, start:start + size])))
        start += size
    alpha_y = alpha_y_0 + 0.5 * n * jnp.asarray(branch_sizes)
    beta_y = beta_y_0 + 0.5 * jnp.stack(squared_residuals)
    return alpha_y, beta_y


def sample_branch_noise(
    key: Array,
    y_tilde: Array,
    c_f: Array,
    delta: Array,
    R: Array,
    branch_sizes: tuple[int, ...],
    alpha_y_0: Array,
    beta_y_0: Array,
) -> Array:
    """Draw all active sigma_y2 (B,) independently, using one fresh key.

    Combines the model's branch conditionals with the standard log-Gamma
    transformation. Valid float64 inputs and static branch_sizes are assumed.
    The pure numerical kernel supports JIT/vmap.
    """

    alpha_y, beta_y = branch_noise_conditional_parameters(
        y_tilde, c_f, delta, R, branch_sizes, alpha_y_0, beta_y_0
    )
    return sample_inverse_gamma(key, alpha_y, beta_y)


def update_branch_noise(
    key: Array, target: CalibrationTarget, c_f: Array, delta: Array
) -> Array:
    """Check current c_f/delta and draw only the active branch variances.

    c_f is (n*k,), delta is (k,), and the return is (B,).
    Prior parameters are target.alpha_y_0 and target.beta_y_0, both (B,).
    Host checks and failure reporting stay outside JIT; no moments are cached.
    """

    if not jax.config.jax_enable_x64:
        raise ValueError("Branch noise update requires jax_enable_x64=True")
    n = target.C_theta.shape[0]
    k = target.gp.F_s.shape[1]
    c_f_np = np.asarray(c_f, dtype=np.float64)
    delta_np = np.asarray(delta, dtype=np.float64)
    if c_f_np.shape != (n * k,) or not np.all(np.isfinite(c_f_np)):
        raise ValueError("c_f must be finite with shape (n*k,)")
    if delta_np.shape != (k,) or not np.all(np.isfinite(delta_np)):
        raise ValueError("delta must be finite with shape (k,)")
    sigma_y2 = sample_branch_noise(
        key, target.y_tilde, jnp.asarray(c_f_np), jnp.asarray(delta_np),
        target.R, target.branch_sizes, target.alpha_y_0, target.beta_y_0,
    )
    if (
        not np.all(np.isfinite(np.asarray(sigma_y2)))
        or np.any(np.asarray(sigma_y2) <= 0)
    ):
        raise FloatingPointError("Branch noise update produced unusable variances")
    return sigma_y2


def _current_theta_tilde(target: CalibrationTarget, eta: Array) -> Array:
    """Share host coordinate checks across the three spatial/GP updates.

    Keeping this boundary in one place makes shape/dtype validation consistent
    while each conditional remains a separate pure numerical kernel.
    """

    if not jax.config.jax_enable_x64:
        raise ValueError("Gibbs updates require jax_enable_x64=True")
    shape = (target.C_theta.shape[0], target.gp.theta_s_tilde.shape[1])
    eta_np = np.asarray(eta, dtype=np.float64)
    if eta_np.shape != shape or not np.all(np.isfinite(eta_np)):
        raise ValueError("eta must be finite with shape (n,d)")
    return target.coordinates.eta_to_theta_tilde(jnp.asarray(eta_np))


def spatial_mean_conditional_moments(
    theta_tilde: Array,
    C_theta: Array,
    Sigma_theta: Array,
    m_theta_0: Array,
    V_theta_0: Array,
) -> tuple[Array, Array]:
    """Return m_theta (d,) and V_theta (d,d) for the unbounded mu_theta.

    theta_tilde is (n,d), C_theta (n,n), Sigma_theta/V_theta_0 (d,d), and
    m_theta_0 (d,). All inputs must be valid float64 arrays with SPD
    covariances. The model-specific precision uses the entire joint spatial
    field, with standard Cholesky solves; no site-independent approximation
    or bound/truncation normalizer is introduced.
    """

    n, d = theta_tilde.shape
    I_d = jnp.eye(d, dtype=jnp.float64)
    L_C = jnp.linalg.cholesky(C_theta)
    L_theta = jnp.linalg.cholesky(Sigma_theta)
    L_theta_0 = jnp.linalg.cholesky(V_theta_0)
    w = jsp.linalg.cho_solve((L_C, True), jnp.ones(n, dtype=jnp.float64))
    a = jnp.sum(w)
    Lambda_theta = (
        jsp.linalg.cho_solve((L_theta_0, True), I_d)
        + a * jsp.linalg.cho_solve((L_theta, True), I_d)
    )
    Lambda_theta = 0.5 * (Lambda_theta + Lambda_theta.T)
    h_theta = (
        jsp.linalg.cho_solve((L_theta_0, True), m_theta_0)
        + jsp.linalg.cho_solve((L_theta, True), theta_tilde.T @ w)
    )
    L_precision = jnp.linalg.cholesky(Lambda_theta)
    m_theta = jsp.linalg.cho_solve((L_precision, True), h_theta)
    V_theta = jsp.linalg.cho_solve((L_precision, True), I_d)
    return m_theta, 0.5 * (V_theta + V_theta.T)


def sample_spatial_mean(
    key: Array,
    theta_tilde: Array,
    C_theta: Array,
    Sigma_theta: Array,
    m_theta_0: Array,
    V_theta_0: Array,
) -> Array:
    """Draw unbounded mu_theta (d,) using the model moments and JAX Gaussian.

    Valid float64 inputs are assumed. This pure kernel supports JIT/vmap,
    requires a fresh key, and applies no site-bound clipping or truncation.
    """

    m_theta, V_theta = spatial_mean_conditional_moments(
        theta_tilde, C_theta, Sigma_theta, m_theta_0, V_theta_0
    )
    return jax.random.multivariate_normal(
        key, m_theta, V_theta, dtype=jnp.float64, method="cholesky"
    )


def update_spatial_mean(
    key: Array, target: CalibrationTarget, eta: Array, Sigma_theta: Array
) -> Array:
    """Validate current sites/covariance and draw the standardized mean (d,).

    This checked host entry point recomputes the conditional from current eta
    and Sigma_theta. The mean remains unbounded in either site-bounds mode.
    """

    theta_tilde = _current_theta_tilde(target, eta)
    d = theta_tilde.shape[1]
    Sigma_np = np.asarray(Sigma_theta, dtype=np.float64)
    if Sigma_np.shape != (d, d) or not np.all(np.isfinite(Sigma_np)):
        raise ValueError("Sigma_theta must be finite with shape (d,d)")
    if not np.array_equal(Sigma_np, Sigma_np.T):
        raise ValueError("Sigma_theta must be symmetric")
    try:
        np.linalg.cholesky(Sigma_np)
    except np.linalg.LinAlgError as error:
        raise ValueError("Sigma_theta must be positive definite") from error
    mu_theta = sample_spatial_mean(
        key, theta_tilde, target.C_theta, jnp.asarray(Sigma_np),
        target.spatial_prior.m_theta_0, target.spatial_prior.V_theta_0,
    )
    if not np.all(np.isfinite(np.asarray(mu_theta))):
        raise FloatingPointError("Spatial mean update produced nonfinite values")
    return mu_theta


def spatial_covariance_conditional_parameters(
    theta_tilde: Array,
    C_theta: Array,
    mu_theta: Array,
    nu_theta_0: float,
    S_theta_0: Array,
) -> tuple[Array, Array]:
    """Return inverse-Wishart nu_theta (scalar), S_theta (d,d).

    theta_tilde is (n,d), C_theta (n,n), mu_theta (d,), and S_theta_0
    (d,d). This model-specific sufficient statistic is the joint-field
    E_theta.T C_theta^{-1} E_theta, computed by a standard triangular solve.
    Valid float64 inputs and SPD matrices are assumed; bounds do not alter
    this conditional under the globally restricted joint spatial prior.
    """

    E_theta = theta_tilde - mu_theta
    L_C = jnp.linalg.cholesky(C_theta)
    E_whitened = jsp.linalg.solve_triangular(L_C, E_theta, lower=True)
    nu_theta = jnp.asarray(nu_theta_0, dtype=jnp.float64) + theta_tilde.shape[0]
    S_theta = S_theta_0 + E_whitened.T @ E_whitened
    return nu_theta, 0.5 * (S_theta + S_theta.T)


def sample_inverse_wishart(key: Array, nu: Array, S: Array) -> Array:
    """Draw IW_d(nu,S) (d,d) with valid float64 S SPD and nu > d-1.

    JAX has no inverse-Wishart sampler; this Bartlett construction preserves
    explicit JAX keys and JIT/vmap support. Normal/chi-square generation and
    triangular solves use standard functions. With A A.T ~ Wishart(nu,I)
    and S=L_S L_S.T, solve T=A^{-1} L_S.T and return T.T T, whose precision
    is Wishart(nu,S^{-1}). This avoids explicit matrix inverses. Real, non-
    integer degrees of freedom are supported; no jitter or redraw is added.
    """

    d = S.shape[0]
    key_normal, key_chi = jax.random.split(key)
    A = jnp.tril(jax.random.normal(key_normal, (d, d), dtype=jnp.float64), -1)
    diagonal = jnp.sqrt(jax.random.chisquare(
        key_chi, nu - jnp.arange(d, dtype=jnp.float64), dtype=jnp.float64
    ))
    A = A + jnp.diag(diagonal)
    L_S = jnp.linalg.cholesky(S)
    T = jsp.linalg.solve_triangular(A, L_S.T, lower=True)
    Sigma = T.T @ T
    return 0.5 * (Sigma + Sigma.T)


def sample_spatial_covariance(
    key: Array,
    theta_tilde: Array,
    C_theta: Array,
    mu_theta: Array,
    nu_theta_0: float,
    S_theta_0: Array,
) -> Array:
    """Draw standardized Sigma_theta from the model's IW conditional.

    Combines the joint-field sufficient statistic with the exact distribution
    sampler above. This pure float64 kernel is JIT/vmap compatible.
    """

    nu_theta, S_theta = spatial_covariance_conditional_parameters(
        theta_tilde, C_theta, mu_theta, nu_theta_0, S_theta_0
    )
    return sample_inverse_wishart(key, nu_theta, S_theta)


def update_spatial_covariance(
    key: Array, target: CalibrationTarget, eta: Array, mu_theta: Array
) -> Array:
    """Validate current sites/mean and draw standardized Sigma_theta (d,d).

    Recomputes the conditional from current eta and unbounded mu_theta. Host
    checks report nonfinite or unusable covariance draws without repair.
    """

    theta_tilde = _current_theta_tilde(target, eta)
    d = theta_tilde.shape[1]
    mu_np = np.asarray(mu_theta, dtype=np.float64)
    if mu_np.shape != (d,) or not np.all(np.isfinite(mu_np)):
        raise ValueError("mu_theta must be finite with shape (d,)")
    Sigma_theta = sample_spatial_covariance(
        key, theta_tilde, target.C_theta, jnp.asarray(mu_np),
        target.spatial_prior.nu_theta_0, target.spatial_prior.S_theta_0,
    )
    Sigma_np = np.asarray(Sigma_theta)
    if not np.all(np.isfinite(Sigma_np)):
        raise FloatingPointError("Spatial covariance update produced nonfinite values")
    try:
        np.linalg.cholesky(Sigma_np)
    except np.linalg.LinAlgError as error:
        raise FloatingPointError(
            "Spatial covariance update produced a nonpositive-definite draw"
        ) from error
    return Sigma_theta


def coefficient_variance_conditional_parameters(
    c_f: Array,
    m_f_given_s: Array,
    C_f_given_s: Array,
    q_s: Array,
    r: int,
    branch_sizes: tuple[int, ...],
    alpha_c_0: Array,
    beta_c_0: Array,
) -> tuple[Array, Array]:
    """Return coefficient IG shapes alpha_c and scales beta_c, both (k,).

    c_f/m_f_given_s are (n*k,), C_f_given_s (n,n), and q_s (k,) contains
    each fixed library quadratic s_j.T C_ss^{-1} s_j. r is the number of
    library runs. This model-specific kernel includes both library and field
    contributions, with the (r+n)/2 shape increment. Standard Cholesky/
    triangular solves whiten field residuals; inputs must be valid float64
    with SPD C_f_given_s and positive alpha_c_0/beta_c_0 of shape (B,).
    branch_sizes maps each branch prior to its contiguous coefficient block
    and is a static tuple for JIT. Current sigma_c2 is not an input.
    """

    n = C_f_given_s.shape[0]
    k = q_s.shape[0]
    E_c = (c_f - m_f_given_s).reshape(n, k)
    L_C = jnp.linalg.cholesky(C_f_given_s)
    E_whitened = jsp.linalg.solve_triangular(L_C, E_c, lower=True)
    alpha_c = jnp.repeat(
        alpha_c_0, jnp.asarray(branch_sizes), total_repeat_length=k
    ) + 0.5 * (r + n)
    beta_c = jnp.repeat(
        beta_c_0, jnp.asarray(branch_sizes), total_repeat_length=k
    ) + 0.5 * (q_s + jnp.sum(jnp.square(E_whitened), axis=0))
    return alpha_c, beta_c


def sample_coefficient_variances(
    key: Array,
    c_f: Array,
    m_f_given_s: Array,
    C_f_given_s: Array,
    q_s: Array,
    r: int,
    branch_sizes: tuple[int, ...],
    alpha_c_0: Array,
    beta_c_0: Array,
) -> Array:
    """Draw independent current sigma_c2 (k,) using the complete joint-GP evidence.

    This pure float64 JIT/vmap kernel combines the model conditional with
    the standard log-Gamma transformation, consuming one fresh caller key.
    Prior parameters are (B,); branch_sizes is static for JIT and maps them
    to the (k,) coefficient variances.
    """

    alpha_c, beta_c = coefficient_variance_conditional_parameters(
        c_f, m_f_given_s, C_f_given_s, q_s, r, branch_sizes, alpha_c_0, beta_c_0
    )
    return sample_inverse_gamma(key, alpha_c, beta_c)


def update_coefficient_variances(
    key: Array, target: CalibrationTarget, eta: Array, c_f: Array
) -> Array:
    """Check current sites/coefficients and draw all active output variances.

    Recompute the GP mean and input conditional covariance from current eta.
    They are independent of sigma_c2, so unit output variances retrieve these from
    the existing GP API. Only fixed library factors/quadratics are reused;
    no field residual or covariance is cached. Structural singularity is
    reported even with configured library jitter. Positive fixed jitter
    retains the GP's explicitly approximate factor-based interpretation.
    """

    theta_tilde = _current_theta_tilde(target, eta)
    n = target.C_theta.shape[0]
    r, k = target.gp.F_s.shape
    c_f_np = np.asarray(c_f, dtype=np.float64)
    if c_f_np.shape != (n * k,) or not np.all(np.isfinite(c_f_np)):
        raise ValueError("c_f must be finite with shape (n*k,)")
    target.gp.validate_field_sites(theta_tilde)
    m_f_given_s, C_f_given_s, _ = target.gp.conditional_moments(
        theta_tilde, jnp.ones(k, dtype=jnp.float64)
    )
    sigma_c2 = sample_coefficient_variances(
        key, jnp.asarray(c_f_np), m_f_given_s, C_f_given_s,
        target.gp.q_s, r, target.branch_sizes, target.alpha_c_0, target.beta_c_0,
    )
    if (
        not np.all(np.isfinite(np.asarray(sigma_c2)))
        or np.any(np.asarray(sigma_c2) <= 0)
    ):
        raise FloatingPointError(
            "Coefficient variance update produced unusable variances"
        )
    return sigma_c2
