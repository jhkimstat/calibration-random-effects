"""Stage 4 posterior densities with one joint spatial field.

Site arrays ``eta`` and ``theta_tilde`` have shape (n, d). Coefficients and
projected observations are stacked site first, then active branches within
each site, giving vectors of shape (n*k,). Fixed library factors and length
scales live in ``LibraryGP``; current positive coefficient variances ``sigma_c2``
are passed to every density evaluation.

Full densities include all normalized reference factors and the fixed-library
likelihood. Under finite site bounds they omit only the single global
restriction normalizer, which is constant across model states. Theta-only
densities omit every term constant during a theta update.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jax import Array

from bayesiancalibration.gp import LibraryGP
from bayesiancalibration.linalg import projected_collapsed_logpdf
from bayesiancalibration.state import SpatialPrior
from bayesiancalibration.transforms import SiteCoordinates


def inverse_gamma_logpdf(value: Array, shape: Array, scale: Array) -> Array:
    """Elementwise normalized shape/scale inverse-Gamma log density.

    The installed JAX stats module has no inverse-Gamma logpdf. This direct
    formula remains differentiable and JIT compatible for positive values.
    """

    return (
        shape * jnp.log(scale)
        - jsp.special.gammaln(shape)
        - (shape + 1.0) * jnp.log(value)
        - scale / value
    )


def inverse_wishart_logpdf(
    Sigma_theta: Array,
    nu_theta_0: float,
    S_theta_0: Array,
) -> Array:
    """Normalized inverse-Wishart log density via Cholesky solves.

    JAX has ``multigammaln`` but no inverse-Wishart distribution. This formula
    follows the source-note degrees-of-freedom/scale parameterization.
    ``Sigma_theta`` and ``S_theta_0`` must be positive definite (d, d).
    """

    d = Sigma_theta.shape[0]
    L_theta = jnp.linalg.cholesky(Sigma_theta)
    L_scale = jnp.linalg.cholesky(S_theta_0)
    logdet_theta = 2.0 * jnp.sum(jnp.log(jnp.diag(L_theta)))
    logdet_scale = 2.0 * jnp.sum(jnp.log(jnp.diag(L_scale)))
    trace_term = jnp.trace(
        jsp.linalg.cho_solve((L_theta, True), S_theta_0)
    )
    return (
        0.5 * nu_theta_0 * logdet_scale
        - 0.5 * nu_theta_0 * d * jnp.log(2.0)
        - jsp.special.multigammaln(nu_theta_0 / 2.0, d)
        - 0.5 * (nu_theta_0 + d + 1.0) * logdet_theta
        - 0.5 * trace_term
    )


@dataclass(frozen=True)
class CalibrationTarget:
    """Fixed target specification, separate from changing model variables.

    ``branch_sizes`` is (k_L, k_U) or (k_L,), with sum k. The spatial
    correlation ``C_theta`` is fixed (n, n); ``Sigma_theta`` is a current
    sampled (d, d) covariance supplied to each density call. Coefficient prior
    parameters alpha_c_0/beta_c_0 are (B,), one per active branch; each applies
    to all coefficient variances in that branch.
    """

    gp: LibraryGP
    coordinates: SiteCoordinates
    spatial_prior: SpatialPrior
    y_tilde: Array
    R: Array
    C_theta: Array
    branch_sizes: tuple[int, ...]
    lambda_theta: float
    m_delta_0: Array
    V_delta_0: Array
    alpha_y_0: Array
    beta_y_0: Array
    alpha_c_0: Array
    beta_c_0: Array

    @classmethod
    def from_data(
        cls,
        gp: LibraryGP,
        coordinates: SiteCoordinates,
        spatial_prior: SpatialPrior,
        y_tilde: Array,
        R: Array,
        s: Array,
        branch_sizes: tuple[int, ...],
        *,
        lambda_theta: float = 12.0,
        m_delta_0: Array | None = None,
        V_delta_0: Array | None = None,
        alpha_y_0: Array | None = None,
        beta_y_0: Array | None = None,
        alpha_c_0: Array | None = None,
        beta_c_0: Array | None = None,
    ) -> CalibrationTarget:
        """Validate fixed data and build the nugget-free spatial correlation."""

        d = gp.theta_s_tilde.shape[1]
        k = gp.F_s.shape[1]
        if coordinates.standardization.theta_bar_dagger.shape != (d,):
            raise ValueError("Coordinate and GP calibration dimensions differ")
        if spatial_prior.m_theta_0.shape != (d,):
            raise ValueError("Spatial prior and GP calibration dimensions differ")
        if (
            len(branch_sizes) not in (1, 2)
            or any(not isinstance(size, int) or size < 1 for size in branch_sizes)
            or sum(branch_sizes) != k
        ):
            raise ValueError(
                "branch_sizes must contain one or two positive blocks summing to k"
            )
        s_np = np.asarray(s, dtype=np.float64)
        if s_np.ndim != 2 or s_np.shape[0] < 1 or s_np.shape[1] < 1:
            raise ValueError("s must have shape (n, spatial_dimension)")
        if not np.all(np.isfinite(s_np)):
            raise ValueError("Spatial locations must be finite")
        if np.unique(s_np, axis=0).shape[0] != s_np.shape[0]:
            raise ValueError("Repeated spatial locations make C_theta singular")
        if not np.isfinite(lambda_theta) or lambda_theta <= 0:
            raise ValueError("lambda_theta must be finite and positive")
        n = s_np.shape[0]
        y_np = np.asarray(y_tilde, dtype=np.float64)
        R_np = np.asarray(R, dtype=np.float64)
        if y_np.shape != (n * k,) or R_np.shape != (n * k, n * k):
            raise ValueError("y_tilde and R must have shapes (n*k,) and (n*k,n*k)")
        if not np.all(np.isfinite(y_np)) or not np.all(np.isfinite(R_np)):
            raise ValueError("Projected observations and R must be finite")

        # The spatial range is isotropic and fixed, unlike the ARD emulator.
        distances = s_np[:, None, :] - s_np[None, :, :]
        C_theta_np = np.exp(
            -0.5 * np.sum(np.square(distances), axis=-1) / lambda_theta**2
        )
        eigenvalues = np.linalg.eigvalsh(C_theta_np)
        if (
            not np.all(np.isfinite(eigenvalues))
            or eigenvalues[0] <= 32.0 * np.finfo(np.float64).eps * eigenvalues[-1]
        ):
            raise ValueError("C_theta is singular or numerically singular")

        m_delta_0_np = (
            np.zeros(k) if m_delta_0 is None
            else np.asarray(m_delta_0, dtype=np.float64)
        )
        V_delta_0_np = (
            1e-6 * np.eye(k)
            if V_delta_0 is None else np.asarray(V_delta_0, dtype=np.float64)
        )
        if m_delta_0_np.shape != (k,) or not np.all(np.isfinite(m_delta_0_np)):
            raise ValueError("m_delta_0 must be finite with shape (k,)")
        if V_delta_0_np.shape != (k, k) or not np.all(np.isfinite(V_delta_0_np)):
            raise ValueError("V_delta_0 must be finite with shape (k,k)")
        if not np.allclose(V_delta_0_np, V_delta_0_np.T, rtol=1e-12, atol=1e-14):
            raise ValueError("V_delta_0 must be symmetric")
        try:
            np.linalg.cholesky(V_delta_0_np)
        except np.linalg.LinAlgError as error:
            raise ValueError("V_delta_0 must be positive definite") from error

        B = len(branch_sizes)
        alpha_y_0_np = (
            np.full(B, 1.01) if alpha_y_0 is None
            else np.asarray(alpha_y_0, dtype=np.float64)
        )
        beta_y_0_np = (
            np.full(B, 0.01) if beta_y_0 is None
            else np.asarray(beta_y_0, dtype=np.float64)
        )
        if (
            alpha_y_0_np.shape != (B,) or beta_y_0_np.shape != (B,)
            or not np.all(np.isfinite(alpha_y_0_np))
            or not np.all(np.isfinite(beta_y_0_np))
            or np.any(alpha_y_0_np <= 0) or np.any(beta_y_0_np <= 0)
        ):
            raise ValueError(
                "alpha_y_0 and beta_y_0 must be positive finite (B,) vectors"
            )
        alpha_c_0_np = (
            np.full(B, 1.01) if alpha_c_0 is None
            else np.asarray(alpha_c_0, dtype=np.float64)
        )
        beta_c_0_np = (
            np.full(B, 0.01) if beta_c_0 is None
            else np.asarray(beta_c_0, dtype=np.float64)
        )
        if (
            alpha_c_0_np.shape != (B,) or beta_c_0_np.shape != (B,)
            or not np.all(np.isfinite(alpha_c_0_np))
            or not np.all(np.isfinite(beta_c_0_np))
            or np.any(alpha_c_0_np <= 0) or np.any(beta_c_0_np <= 0)
        ):
            raise ValueError(
                "alpha_c_0 and beta_c_0 must be positive finite (B,) vectors"
            )
        return cls(
            gp, coordinates, spatial_prior,
            jnp.asarray(y_np), jnp.asarray(R_np), jnp.asarray(C_theta_np),
            tuple(branch_sizes), float(lambda_theta),
            jnp.asarray(m_delta_0_np), jnp.asarray(V_delta_0_np),
            jnp.asarray(alpha_y_0_np), jnp.asarray(beta_y_0_np),
            jnp.asarray(alpha_c_0_np), jnp.asarray(beta_c_0_np),
        )

    def observation_arrays(
        self, delta: Array, sigma_y2: Array
    ) -> tuple[Array, Array]:
        """Build d_delta (nk,) and Omega_y (nk,nk) from current delta/noise.

        Share the model's branch stacking between targets and Gibbs draws.
        This pure kernel checks shapes; callers validate positive noise.
        """

        n = self.C_theta.shape[0]
        delta = jnp.asarray(delta)
        sigma_y2 = jnp.asarray(sigma_y2)
        if delta.shape != (self.gp.F_s.shape[1],):
            raise ValueError("delta must have shape (k,)")
        if sigma_y2.shape != (len(self.branch_sizes),):
            raise ValueError("sigma_y2 must have shape (B,)")
        d_delta = jnp.tile(delta, n)
        per_coefficient = jnp.repeat(
            sigma_y2,
            jnp.asarray(self.branch_sizes),
            total_repeat_length=self.gp.F_s.shape[1],
        )
        Omega_y = jnp.kron(jnp.eye(n), jnp.diag(per_coefficient))
        return d_delta, Omega_y

    def _conditioning_and_spatial(
        self,
        eta: Array,
        mu_theta: Array,
        Sigma_theta: Array,
        sigma_c2: Array,
    ) -> tuple[Array, Array, Array]:
        """Share one GP conditioning and spatial prior across both targets."""

        theta_tilde = self.coordinates.eta_to_theta_tilde(eta)
        m_f_given_s, _, Sigma_f_given_s = self.gp.conditional_moments(
            theta_tilde, sigma_c2
        )
        n = self.C_theta.shape[0]
        spatial_logpdf = jsp.stats.multivariate_normal.logpdf(
            theta_tilde.reshape(-1),
            jnp.tile(mu_theta, n),
            jnp.kron(self.C_theta, Sigma_theta),
        )
        return m_f_given_s, Sigma_f_given_s, (
            spatial_logpdf + self.coordinates.log_jacobian(eta)
        )

    def _other_prior_logpdf(
        self,
        delta: Array,
        sigma_y2: Array,
        mu_theta: Array,
        Sigma_theta: Array,
        sigma_c2: Array,
    ) -> Array:
        """Terms constant in eta, retained in each full joint density."""

        return (
            self.gp.library_logpdf(sigma_c2)
            + jsp.stats.multivariate_normal.logpdf(
                delta, self.m_delta_0, self.V_delta_0
            )
            + jnp.sum(inverse_gamma_logpdf(
                sigma_y2, self.alpha_y_0, self.beta_y_0
            ))
            + jsp.stats.multivariate_normal.logpdf(
                mu_theta, self.spatial_prior.m_theta_0,
                self.spatial_prior.V_theta_0,
            )
            + inverse_wishart_logpdf(
                Sigma_theta, self.spatial_prior.nu_theta_0,
                self.spatial_prior.S_theta_0,
            )
            + jnp.sum(inverse_gamma_logpdf(
                sigma_c2,
                jnp.repeat(
                    self.alpha_c_0, jnp.asarray(self.branch_sizes),
                    total_repeat_length=self.gp.F_s.shape[1],
                ),
                jnp.repeat(
                    self.beta_c_0, jnp.asarray(self.branch_sizes),
                    total_repeat_length=self.gp.F_s.shape[1],
                ),
            ))
        )

    def theta_only_collapsed(
        self,
        eta: Array,
        delta: Array,
        sigma_y2: Array,
        mu_theta: Array,
        Sigma_theta: Array,
        sigma_c2: Array,
    ) -> Array:
        """Log theta target after integrating out c_f, including log Jacobian."""

        m_f_given_s, Sigma_f_given_s, spatial = self._conditioning_and_spatial(
            eta, mu_theta, Sigma_theta, sigma_c2
        )
        d_delta, Omega_y = self.observation_arrays(delta, sigma_y2)
        value = projected_collapsed_logpdf(
            self.y_tilde, m_f_given_s, Sigma_f_given_s,
            self.R, d_delta, Omega_y,
        ) + spatial
        return value

    def theta_only_uncollapsed(
        self,
        eta: Array,
        c_f: Array,
        mu_theta: Array,
        Sigma_theta: Array,
        sigma_c2: Array,
    ) -> Array:
        """Log theta target with c_f fixed, including log Jacobian."""

        m_f_given_s, Sigma_f_given_s, spatial = self._conditioning_and_spatial(
            eta, mu_theta, Sigma_theta, sigma_c2
        )
        value = jsp.stats.multivariate_normal.logpdf(
            c_f, m_f_given_s, Sigma_f_given_s
        ) + spatial
        return value

    def full_joint_collapsed(
        self,
        eta: Array,
        delta: Array,
        sigma_y2: Array,
        mu_theta: Array,
        Sigma_theta: Array,
        sigma_c2: Array,
    ) -> Array:
        """Log p(y,c_s,eta,hyperparameters), up to global box constant."""

        value = self.theta_only_collapsed(
            eta, delta, sigma_y2, mu_theta, Sigma_theta, sigma_c2
        ) + self._other_prior_logpdf(
            delta, sigma_y2, mu_theta, Sigma_theta, sigma_c2
        )
        return value

    def full_joint_uncollapsed(
        self,
        eta: Array,
        c_f: Array,
        delta: Array,
        sigma_y2: Array,
        mu_theta: Array,
        Sigma_theta: Array,
        sigma_c2: Array,
    ) -> Array:
        """Log p(y,c_f,c_s,eta,hyperparameters), up to box constant."""

        d_delta, Omega_y = self.observation_arrays(delta, sigma_y2)
        observation = jsp.stats.multivariate_normal.logpdf(
            self.y_tilde, self.R @ (c_f + d_delta), Omega_y
        )
        value = (
            self.theta_only_uncollapsed(eta, c_f, mu_theta, Sigma_theta, sigma_c2)
            + observation
            + self._other_prior_logpdf(
                delta, sigma_y2, mu_theta, Sigma_theta, sigma_c2
            )
        )
        return value
