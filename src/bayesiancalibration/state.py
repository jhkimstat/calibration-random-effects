"""Frozen calibration-coordinate specification, prepared outside JIT.

Library inputs ``theta_s_dagger`` have shape (r, d), with one row per run.
The frozen map stores the source-note mean ``theta_bar_dagger`` (d,) and
diagonal sample-scale matrix ``D_theta`` (d, d), both in physical units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


@dataclass(frozen=True)
class ThetaStandardization:
    """Immutable library-derived affine map; construct with ``from_library``.

    ``source`` records whether the frozen location and scales came from the
    library sample or an explicit override. The JAX arrays are immutable;
    field values cannot change this map during sampling.
    """

    theta_bar_dagger: Array
    D_theta: Array
    source: Literal["library", "override"]

    @classmethod
    def from_library(
        cls,
        theta_s_dagger: Array,
        *,
        theta_bar_dagger: Array | None = None,
        D_theta: Array | None = None,
    ) -> ThetaStandardization:
        """Validate an (r, d) library and freeze its float64 coordinate map.

        Automatic sample scales use ``ddof=1``. A coordinate is rejected when
        its sample standard deviation is at most 32 float64 ulps of the
        largest absolute library value in that coordinate. This scale-aware
        threshold catches subtraction dominated by rounding without imposing
        a physical-unit floor. Explicit overrides never bypass this check.
        Both override values must be supplied together; ``D_theta`` must be
        exactly diagonal.
        """

        if not jax.config.jax_enable_x64:
            raise RuntimeError("Enable jax_enable_x64 before preparing the library")
        theta_s_dagger_np = np.asarray(theta_s_dagger, dtype=np.float64)
        if theta_s_dagger_np.ndim != 2 or theta_s_dagger_np.shape[1] == 0:
            raise ValueError("theta_s_dagger must have shape (r, d) with d >= 1")
        r, d = theta_s_dagger_np.shape
        if r < 2:
            raise ValueError("Library standardization requires r >= 2")
        if not np.all(np.isfinite(theta_s_dagger_np)):
            raise ValueError("theta_s_dagger must be finite")

        sample_mean = np.mean(theta_s_dagger_np, axis=0)
        sample_scale = np.std(theta_s_dagger_np, axis=0, ddof=1)
        spread_threshold = (
            32.0
            * np.finfo(np.float64).eps
            * np.max(np.abs(theta_s_dagger_np), axis=0)
        )
        if (
            not np.all(np.isfinite(sample_mean))
            or not np.all(np.isfinite(sample_scale))
            or np.any(sample_scale <= spread_threshold)
        ):
            raise ValueError("Library has zero or numerically negligible spread")

        if (theta_bar_dagger is None) != (D_theta is None):
            raise ValueError("Supply theta_bar_dagger and D_theta together")
        if theta_bar_dagger is None:
            center = sample_mean
            scales = sample_scale
            source: Literal["library", "override"] = "library"
        else:
            center = np.asarray(theta_bar_dagger, dtype=np.float64)
            scale_matrix = np.asarray(D_theta, dtype=np.float64)
            if center.shape != (d,) or scale_matrix.shape != (d, d):
                raise ValueError("Override shapes must be (d,) and (d, d)")
            if not np.all(np.isfinite(center)) or not np.all(np.isfinite(scale_matrix)):
                raise ValueError("Overrides must be finite")
            if not np.array_equal(scale_matrix, np.diag(np.diag(scale_matrix))):
                raise ValueError("D_theta must be diagonal")
            scales = np.diag(scale_matrix)
            source = "override"

        if np.any(scales <= 0):
            raise ValueError("D_theta scales must be positive and numerically usable")
        with np.errstate(over="ignore"):
            usable_reciprocals = np.all(np.isfinite(1.0 / scales))
        if not usable_reciprocals:
            raise ValueError("D_theta scales must be positive and numerically usable")
        standardized_library = (theta_s_dagger_np - center) / scales
        if not np.all(np.isfinite(standardized_library)):
            raise ValueError("Frozen map makes standardized library inputs nonfinite")
        return cls(
            theta_bar_dagger=jnp.asarray(center, dtype=jnp.float64),
            D_theta=jnp.diag(jnp.asarray(scales, dtype=jnp.float64)),
            source=source,
        )

    def to_standardized(self, theta: Array) -> Array:
        """Apply D_theta^{-1}(theta - theta_bar_dagger), with trailing d axis."""

        theta = jnp.asarray(theta, dtype=jnp.float64)
        if theta.ndim == 0 or theta.shape[-1] != self.theta_bar_dagger.shape[0]:
            raise ValueError("theta must have trailing dimension d")
        return (theta - self.theta_bar_dagger) / jnp.diag(self.D_theta)

    def to_physical(self, theta_tilde: Array) -> Array:
        """Apply theta_bar_dagger + D_theta theta_tilde, with trailing d axis."""

        theta_tilde = jnp.asarray(theta_tilde, dtype=jnp.float64)
        if (
            theta_tilde.ndim == 0
            or theta_tilde.shape[-1] != self.theta_bar_dagger.shape[0]
        ):
            raise ValueError("theta_tilde must have trailing dimension d")
        return self.theta_bar_dagger + theta_tilde * jnp.diag(self.D_theta)


@dataclass(frozen=True)
class SpatialPrior:
    """Standardized reference prior parameters, independent of site bounds."""

    m_theta_0: Array  # (d,)
    V_theta_0: Array  # (d, d), prior covariance of mu_theta
    nu_theta_0: float  # inverse-Wishart degrees of freedom
    S_theta_0: Array  # (d, d), inverse-Wishart scale

    @classmethod
    def from_standardization(
        cls,
        standardization: ThetaStandardization,
        *,
        physical_center: Array | None = None,
        V_theta_0: Array | None = None,
        nu_theta_0: float | None = None,
        S_theta_0: Array | None = None,
    ) -> SpatialPrior:
        """Convert the physical mean center; keep covariance priors standardized.

        The source-note defaults apply only for d=3. Other dimensions require
        all four prior settings explicitly.
        """

        d = standardization.theta_bar_dagger.shape[0]
        if d == 3:
            if physical_center is None:
                physical_center = [37850.0, 24060.0, 0.071]
            if V_theta_0 is None:
                V_theta_0 = 4.0 * np.eye(d)
            if nu_theta_0 is None:
                nu_theta_0 = 5.0
            if S_theta_0 is None:
                S_theta_0 = np.eye(d)
        elif any(
            value is None
            for value in (physical_center, V_theta_0, nu_theta_0, S_theta_0)
        ):
            raise ValueError("Explicit spatial prior settings are required for d != 3")

        center = np.asarray(physical_center, dtype=np.float64)
        V = np.asarray(V_theta_0, dtype=np.float64)
        S = np.asarray(S_theta_0, dtype=np.float64)
        nu = float(nu_theta_0)
        if center.shape != (d,) or not np.all(np.isfinite(center)):
            raise ValueError("physical_center must be a finite (d,) vector")
        for name, matrix in (("V_theta_0", V), ("S_theta_0", S)):
            if matrix.shape != (d, d) or not np.all(np.isfinite(matrix)):
                raise ValueError(f"{name} must be a finite (d, d) matrix")
            if not np.allclose(matrix, matrix.T, rtol=0.0, atol=0.0):
                raise ValueError(f"{name} must be symmetric")
            try:
                np.linalg.cholesky(matrix)
            except np.linalg.LinAlgError as error:
                raise ValueError(f"{name} must be positive definite") from error
        if not np.isfinite(nu) or nu <= d - 1:
            raise ValueError("nu_theta_0 must exceed d - 1")
        m_theta_0 = standardization.to_standardized(jnp.asarray(center))
        if not bool(jnp.all(jnp.isfinite(m_theta_0))):
            raise ValueError("Physical prior center is not usable under the frozen map")
        return cls(m_theta_0, jnp.asarray(V), nu, jnp.asarray(S))


class CalibrationState(NamedTuple):
    """Changing model variables at a completed Gibbs-sweep boundary.

    eta is (n,d); c_f is site-major (n*k,); delta and sigma_c2 are (k,);
    sigma_y2 is (B,); mu_theta is (d,) and Sigma_theta is (d,d).
    PRNG state, proposal tuning, adaptation, and diagnostics live separately.
    Sites derive standardized/physical values from eta and the frozen map.
    """

    eta: Array
    c_f: Array
    delta: Array
    sigma_y2: Array
    mu_theta: Array
    Sigma_theta: Array
    sigma_c2: Array
