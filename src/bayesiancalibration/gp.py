"""Exact zero-mean coefficient GP conditioning and library-only length fitting.

Inputs ``theta_s_tilde`` and ``theta_f_tilde`` are standardized rows of shapes
(r, d) and (n, d). ``F_s`` is the fixed library coefficient view (r, k),
with active branches concatenated within each row. No GP mean or statistical
nugget is added. A supplied jitter is a fixed numerical factorization policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jax import Array
from scipy import __version__ as scipy_version
from scipy.optimize import minimize


def squared_exponential_kernel(
    theta_a_tilde: Array,
    theta_b_tilde: Array,
    lambda_c: Array,
) -> Array:
    """Unit-amplitude ARD kernel, returning an (a, b) Gram matrix.

    Both inputs have trailing dimension d; ``lambda_c`` is positive (d,).
    Input validation belongs to the preparation boundary, outside JIT.
    """

    scaled_difference = (
        theta_a_tilde[:, None, :] - theta_b_tilde[None, :, :]
    ) / lambda_c
    return jnp.exp(-0.5 * jnp.sum(jnp.square(scaled_difference), axis=-1))


def _check_unjittered_covariance(C: np.ndarray, name: str) -> None:
    """Reject numerical singularity before jitter can hide it.

    Eigenvalue checking is a host-side validation step. The 32-epsilon
    relative threshold detects effectively singular input correlations at
    float64 precision; it is not a covariance regularizer.
    """

    eigenvalues = np.linalg.eigvalsh(C)
    if (
        not np.all(np.isfinite(eigenvalues))
        or eigenvalues[0] <= 32.0 * np.finfo(np.float64).eps * eigenvalues[-1]
    ):
        raise ValueError(f"Unjittered {name} is singular or numerically singular")


def _validate_library_inputs(
    theta_s_tilde: Array,
    F_s: Array,
    jitter: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate fixed data once, before numerical JAX kernels are traced."""

    if not jax.config.jax_enable_x64:
        raise RuntimeError("Enable jax_enable_x64 before preparing the GP")
    theta_np = np.asarray(theta_s_tilde, dtype=np.float64)
    F_np = np.asarray(F_s, dtype=np.float64)
    if theta_np.ndim != 2 or theta_np.shape[0] < 2 or theta_np.shape[1] < 1:
        raise ValueError("theta_s_tilde must have shape (r, d), r >= 2, d >= 1")
    if F_np.ndim != 2 or F_np.shape[0] != theta_np.shape[0] or F_np.shape[1] < 1:
        raise ValueError("F_s must have shape (r, k), k >= 1")
    if not np.all(np.isfinite(theta_np)) or not np.all(np.isfinite(F_np)):
        raise ValueError("Library inputs and coefficients must be finite")
    if np.unique(theta_np, axis=0).shape[0] != theta_np.shape[0]:
        raise ValueError("Repeated library inputs make C_ss singular")
    if not np.isfinite(jitter) or jitter < 0:
        raise ValueError("Numerical jitter must be finite and nonnegative")
    return theta_np, F_np


def _library_factor(
    theta_s_tilde: np.ndarray,
    lambda_c: np.ndarray,
    jitter: float,
) -> tuple[Array, Array]:
    """Check structural singularity, then factor at the fixed jitter."""

    theta = jnp.asarray(theta_s_tilde)
    C_ss = squared_exponential_kernel(theta, theta, jnp.asarray(lambda_c))
    _check_unjittered_covariance(np.asarray(C_ss), "C_ss")
    L_ss = jnp.linalg.cholesky(C_ss + jitter * jnp.eye(theta.shape[0]))
    if not bool(jnp.all(jnp.isfinite(L_ss))):
        raise ValueError("C_ss factorization failed at the configured jitter")
    return C_ss, L_ss


@dataclass(frozen=True)
class LibraryGP:
    """Fixed library factors; output variances remain sampled state.

    ``alpha_s = C_ss^{-1} F_s`` at the configured numerical jitter, shape
    (r, k). ``q_s`` has shape (k,) and is used for library likelihoods.
    When jitter is positive, factor-based evaluations approximate the exact
    unjittered model and require a sensitivity check.
    Construct with ``from_data`` to validate and factor the fixed inputs.
    """

    theta_s_tilde: Array
    F_s: Array
    lambda_c: Array
    C_ss: Array
    L_ss: Array
    alpha_s: Array
    q_s: Array
    jitter: float

    @classmethod
    def from_data(
        cls,
        theta_s_tilde: Array,
        F_s: Array,
        lambda_c: Array,
        *,
        jitter: float = 0.0,
    ) -> LibraryGP:
        """Build the exact library GP at supplied fixed length scales."""

        theta_np, F_np = _validate_library_inputs(theta_s_tilde, F_s, jitter)
        lambda_np = np.asarray(lambda_c, dtype=np.float64)
        if lambda_np.shape != (theta_np.shape[1],):
            raise ValueError("lambda_c must have shape (d,)")
        if not np.all(np.isfinite(lambda_np)) or np.any(lambda_np <= 0):
            raise ValueError("lambda_c must be finite and positive")
        C_ss, L_ss = _library_factor(theta_np, lambda_np, jitter)
        F_jax = jnp.asarray(F_np)
        alpha_s = jsp.linalg.cho_solve((L_ss, True), F_jax)
        whitened = jsp.linalg.solve_triangular(L_ss, F_jax, lower=True)
        q_s = jnp.sum(jnp.square(whitened), axis=0)
        return cls(
            jnp.asarray(theta_np), F_jax, jnp.asarray(lambda_np),
            C_ss, L_ss, alpha_s, q_s, float(jitter),
        )

    def validate_field_sites(self, theta_f_tilde: Array) -> None:
        """Reject field/library coincidences and singular field conditionals.

        This host check is for initialization and reference evaluation. The
        pure ``conditional_moments`` kernel can then be JIT compiled for
        continuous proposals, whose exact coincidences have measure zero.
        """

        theta_f = np.asarray(theta_f_tilde, dtype=np.float64)
        d = self.theta_s_tilde.shape[1]
        if theta_f.ndim != 2 or theta_f.shape[1] != d or theta_f.shape[0] < 1:
            raise ValueError("theta_f_tilde must have shape (n, d), n >= 1")
        if not np.all(np.isfinite(theta_f)):
            raise ValueError("theta_f_tilde must be finite")
        if np.unique(theta_f, axis=0).shape[0] != theta_f.shape[0]:
            raise ValueError("Repeated field sites make C_f_given_s singular")
        if np.any(np.all(
            theta_f[:, None, :] == np.asarray(self.theta_s_tilde), axis=-1
        )):
            raise ValueError("Field site coincides with a library input")

        C_fs = np.asarray(squared_exponential_kernel(
            jnp.asarray(theta_f), self.theta_s_tilde, self.lambda_c
        ))
        C_ff = np.asarray(squared_exponential_kernel(
            jnp.asarray(theta_f), jnp.asarray(theta_f), self.lambda_c
        ))
        C_f_given_s = C_ff - C_fs @ np.linalg.solve(np.asarray(self.C_ss), C_fs.T)
        C_f_given_s = 0.5 * (C_f_given_s + C_f_given_s.T)
        _check_unjittered_covariance(C_f_given_s, "C_f_given_s")

    def conditional_moments(
        self,
        theta_f_tilde: Array,
        sigma_c2: Array,
    ) -> tuple[Array, Array, Array]:
        """Return m_f_given_s (nk,), C_f_given_s (n,n), Sigma (nk,nk).

        ``sigma_c2`` is the current positive coefficient-variance vector (k,).
        Constructing ``Sigma_f_given_s`` here avoids a stale covariance when
        sigma_c2 changes. Site-major/branch-within-site stacking uses row-major view.
        """

        C_fs = squared_exponential_kernel(
            theta_f_tilde, self.theta_s_tilde, self.lambda_c
        )
        C_ff = squared_exponential_kernel(
            theta_f_tilde, theta_f_tilde, self.lambda_c
        )
        m_f_given_s = (C_fs @ self.alpha_s).reshape(-1)
        C_f_given_s = C_ff - C_fs @ jsp.linalg.cho_solve(
            (self.L_ss, True), C_fs.T
        )
        C_f_given_s = 0.5 * (C_f_given_s + C_f_given_s.T)
        Sigma_f_given_s = jnp.kron(C_f_given_s, jnp.diag(sigma_c2))
        return m_f_given_s, C_f_given_s, Sigma_f_given_s

    def library_logpdf(self, sigma_c2: Array) -> Array:
        """Return normalized log p(c_s | sigma_c2, lambda_c) from cached factors.

        The separable zero-mean GP permits k scalar output solves. This avoids
        a dense (rk)-dimensional covariance factorization at every sigma_c2 update.
        """

        r, k = self.F_s.shape
        logdet_C_ss = 2.0 * jnp.sum(jnp.log(jnp.diag(self.L_ss)))
        return -0.5 * (
            r * k * jnp.log(2.0 * jnp.pi)
            + k * logdet_C_ss
            + r * jnp.sum(jnp.log(sigma_c2))
            + jnp.sum(self.q_s / sigma_c2)
        )


def profile_log_likelihood(
    log_lambda_c: Array,
    theta_s_tilde: Array,
    F_s: Array,
    *,
    jitter: float = 0.0,
) -> Array:
    """Library-only log profile likelihood, omitting lambda-independent C.

    Each q_j is a zero-mean coefficient-column quadratic form and its
    profiled variance is q_j/r. ``log_lambda_c`` may be traced by JAX.
    Validation of singular candidates occurs in the fitting boundary.
    """

    r, k = F_s.shape
    lambda_c = jnp.exp(log_lambda_c)
    C_ss = squared_exponential_kernel(theta_s_tilde, theta_s_tilde, lambda_c)
    L_ss = jnp.linalg.cholesky(C_ss + jitter * jnp.eye(r))
    whitened = jsp.linalg.solve_triangular(L_ss, F_s, lower=True)
    q_s = jnp.sum(jnp.square(whitened), axis=0)
    return -k * jnp.sum(jnp.log(jnp.diag(L_ss))) - (r / 2.0) * jnp.sum(
        jnp.log(q_s / r)
    )


def _cv_residual_and_variance(
    log_lambda_c: Array,
    theta_s_tilde: Array,
    F_s: Array,
    *,
    jitter: float = 0.0,
) -> tuple[Array, Array]:
    """Share exact fold predictions between NLPD and WMSE without divergence.

    For each held-out row, factor the (r-1, r-1) training correlation,
    estimate each of k output variances from those training rows alone, and
    return residual and predictive variance arrays of shape (r, k).
    Batched triangular solves avoid inverse matrices.
    ``jitter`` follows the same fixed factorization policy as the GP fit.
    """

    theta_s_tilde = jnp.asarray(theta_s_tilde)
    F_s = jnp.asarray(F_s)
    r, k = F_s.shape
    C_ss = squared_exponential_kernel(
        theta_s_tilde, theta_s_tilde, jnp.exp(log_lambda_c)
    )
    rows = jnp.arange(r)
    training = jnp.where(
        jnp.arange(r - 1)[None, :] < rows[:, None],
        jnp.arange(r - 1)[None, :],
        jnp.arange(r - 1)[None, :] + 1,
    )
    C_minus = C_ss[training[:, :, None], training[:, None, :]]
    L_minus = jnp.linalg.cholesky(C_minus + jitter * jnp.eye(r - 1))
    F_minus = F_s[training]
    whitened_F = jsp.linalg.solve_triangular(L_minus, F_minus, lower=True)
    k_j_minus = C_ss[rows[:, None], training]
    whitened_k = jsp.linalg.solve_triangular(
        L_minus, k_j_minus[..., None], lower=True
    )[..., 0]
    c_hat = jnp.einsum("ji,jik->jk", whitened_k, whitened_F)
    sigma_c2_minus = jnp.sum(jnp.square(whitened_F), axis=1) / (r - 1)
    conditional_variance = 1.0 - jnp.sum(jnp.square(whitened_k), axis=1)
    predictive_variance = conditional_variance[:, None] * sigma_c2_minus
    residual = F_s - c_hat
    return residual, predictive_variance


def cv_nlpd(
    log_lambda_c: Array,
    theta_s_tilde: Array,
    F_s: Array,
    *,
    jitter: float = 0.0,
) -> Array:
    """Mean leave-one-run-out NLPD for zero-mean coefficient GP columns."""

    residual, predictive_variance = _cv_residual_and_variance(
        log_lambda_c, theta_s_tilde, F_s, jitter=jitter
    )
    k = F_s.shape[1]
    nlpd = 0.5 * (
        k * jnp.log(2.0 * jnp.pi)
        + jnp.sum(jnp.log(predictive_variance), axis=1)
        + jnp.sum(jnp.square(residual) / predictive_variance, axis=1)
    )
    return jnp.mean(nlpd)


def cv_wmse(
    log_lambda_c: Array,
    theta_s_tilde: Array,
    F_s: Array,
    *,
    jitter: float = 0.0,
) -> Array:
    """Mean foldwise sum of squared errors divided by predictive variances.

    This is exactly the quadratic term inside twice the NLPD, averaged over
    held-out runs. Dividing by k as well would only rescale the objective.
    It is not a proper predictive score: inflated variances can reduce it.
    """

    residual, predictive_variance = _cv_residual_and_variance(
        log_lambda_c, theta_s_tilde, F_s, jitter=jitter
    )
    return jnp.mean(jnp.sum(jnp.square(residual) / predictive_variance, axis=1))


@dataclass(frozen=True)
class FitAttempt:
    """One optimizer start; objective follows the selected fitting method."""

    start: tuple[float, ...]
    optimum: tuple[float, ...] | None
    objective: float | None
    success: bool
    message: str
    iterations: int
    evaluations: int


@dataclass(frozen=True)
class LengthScaleFit:
    """Frozen fitted length scales and reproducible library-fit diagnostics.

    ``profiled_variances`` are full-library q_j/r diagnostics for every
    selection method, never sampled-state sigma_c2.
    The numerical bounds constrain the optimizer search if explicitly given.
    ``objective`` is maximized for ``profile`` and minimized for CV methods.
    """

    lambda_c: Array
    log_lambda_c: Array
    profiled_variances: Array
    objective: float
    attempts: tuple[FitAttempt, ...]
    log_bounds: tuple[tuple[float, float], ...] | None
    gtol: float
    ftol: float
    maxiter: int
    jitter: float
    scipy_version: str
    method: str = "profile"


def fit_library_length_scales(
    theta_s_tilde: Array,
    F_s: Array,
    *,
    starts: Array,
    gtol: float,
    ftol: float,
    maxiter: int,
    log_bounds: Array | None = None,
    jitter: float = 0.0,
    method: str = "profile",
) -> LengthScaleFit:
    """Select length scales by profile likelihood, CV–NLPD, or CV–WMSE.

    ``method='profile'`` maximizes the zero-mean full-library profile log
    likelihood; the CV methods minimize mean leave-one-run-out scores.
    The caller supplies all starts and stopping tolerances. Optional finite
    ``log_bounds`` (d, 2) are an explicit optimizer search restriction, not a
    GP prior. Every candidate is checked before jitter, and invalid starts or
    paths are retained as failed attempts rather than silently regularized.
    """

    theta_np, F_np = _validate_library_inputs(theta_s_tilde, F_s, jitter)
    if method not in ("profile", "cv_nlpd", "cv_wmse"):
        raise ValueError("method must be 'profile', 'cv_nlpd', or 'cv_wmse'")
    if np.any(np.all(F_np == 0.0, axis=0)):
        raise ValueError("A zero coefficient column has no positive variance MLE")
    d = theta_np.shape[1]
    starts_np = np.asarray(starts, dtype=np.float64)
    if starts_np.ndim != 2 or starts_np.shape[1] != d or starts_np.shape[0] == 0:
        raise ValueError("starts must have shape (number_of_starts, d)")
    if not np.all(np.isfinite(starts_np)):
        raise ValueError("Optimizer starts must be finite")
    if not np.isfinite(gtol) or gtol <= 0 or not np.isfinite(ftol) or ftol <= 0:
        raise ValueError("gtol and ftol must be finite and positive")
    if not isinstance(maxiter, int) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer")
    bounds = None
    if log_bounds is not None:
        bounds_np = np.asarray(log_bounds, dtype=np.float64)
        if bounds_np.shape != (d, 2) or not np.all(np.isfinite(bounds_np)):
            raise ValueError("log_bounds must be finite with shape (d, 2)")
        if np.any(bounds_np[:, 0] >= bounds_np[:, 1]):
            raise ValueError("Each log bound must have positive width")
        if np.any(starts_np < bounds_np[:, 0]) or np.any(starts_np > bounds_np[:, 1]):
            raise ValueError("Optimizer starts must lie within log_bounds")
        bounds = tuple((float(lo), float(hi)) for lo, hi in bounds_np)

    theta = jnp.asarray(theta_np)
    F = jnp.asarray(F_np)
    objective = {
        "profile": profile_log_likelihood,
        "cv_nlpd": cv_nlpd,
        "cv_wmse": cv_wmse,
    }[method]
    sign = -1.0 if method == "profile" else 1.0
    evaluate = jax.jit(jax.value_and_grad(
        lambda xi: objective(xi, theta, F, jitter=jitter)
    ))

    # SciPy needs host scalars/arrays; JAX retains the differentiable kernel.
    def minimization_value_and_grad(xi: np.ndarray) -> tuple[float, np.ndarray]:
        with np.errstate(over="ignore", under="ignore"):
            lambda_candidate = np.exp(xi)
        if not np.all(np.isfinite(lambda_candidate)) or np.any(lambda_candidate <= 0):
            raise ValueError("Candidate lambda_c is not finite and positive")
        _library_factor(theta_np, lambda_candidate, jitter)
        value, gradient = evaluate(jnp.asarray(xi))
        value_np = float(value)
        gradient_np = np.asarray(gradient)
        if not np.isfinite(value_np) or not np.all(np.isfinite(gradient_np)):
            raise ValueError(f"{method} objective or gradient is nonfinite")
        return sign * value_np, sign * gradient_np

    attempts: list[FitAttempt] = []
    for start in starts_np:
        try:
            result = minimize(
                minimization_value_and_grad,
                start,
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
                options={"gtol": gtol, "ftol": ftol, "maxiter": maxiter},
            )
            attempts.append(FitAttempt(
                tuple(map(float, start)),
                tuple(map(float, result.x)),
                sign * float(result.fun),
                bool(result.success),
                str(result.message),
                int(result.nit),
                int(result.nfev),
            ))
        except ValueError as error:
            attempts.append(FitAttempt(
                tuple(map(float, start)), None, None, False,
                str(error), 0, 0,
            ))

    successful = [attempt for attempt in attempts if attempt.success]
    if not successful:
        messages = "; ".join(attempt.message for attempt in attempts)
        raise ValueError(f"No length-scale fit converged: {messages}")
    best = (max if method == "profile" else min)(
        successful, key=lambda attempt: attempt.objective
    )
    log_lambda_c = jnp.asarray(best.optimum)
    lambda_c = jnp.exp(log_lambda_c)
    library = LibraryGP.from_data(theta, F, lambda_c, jitter=jitter)
    return LengthScaleFit(
        lambda_c, log_lambda_c, library.q_s / theta.shape[0],
        best.objective, tuple(attempts), bounds,
        float(gtol), float(ftol), maxiter, float(jitter), scipy_version, method,
    )
