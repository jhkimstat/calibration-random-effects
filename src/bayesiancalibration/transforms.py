"""Site-coordinate maps for standardized calibration parameters.

Site arrays have shape (..., d), normally (n, d). The same frozen library map
is used for library inputs, sites, physical bounds, and the prior mean center.
The spatial mean ``mu_theta`` remains unbounded in either site mode.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.nn as jnn
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jax import Array

from bayesiancalibration.state import ThetaStandardization


@dataclass(frozen=True)
class SiteCoordinates:
    """Identity or finite-box map for site ``eta`` and ``theta_tilde``.

    For bounded sites, ``l_tilde`` and ``u_tilde`` are shape-(d,) bounds in
    standardized coordinates. An unbounded instance stores both as ``None``.
    Construct through ``from_physical_bounds`` to validate and freeze them.
    """

    standardization: ThetaStandardization
    l_tilde: Array | None
    u_tilde: Array | None

    @classmethod
    def from_physical_bounds(
        cls,
        standardization: ThetaStandardization,
        *,
        l: Array | None = None,
        u: Array | None = None,
    ) -> SiteCoordinates:
        """Prepare either no bounds or a finite componentwise physical box."""

        if l is None and u is None:
            return cls(standardization, None, None)
        if l is None or u is None:
            raise ValueError("Both physical bounds l and u are required")
        d = standardization.theta_bar_dagger.shape[0]
        l_np = np.asarray(l, dtype=np.float64)
        u_np = np.asarray(u, dtype=np.float64)
        if l_np.shape != (d,) or u_np.shape != (d,):
            raise ValueError("Physical bounds must have shape (d,)")
        if not np.all(np.isfinite(l_np)) or not np.all(np.isfinite(u_np)):
            raise ValueError("Physical bounds must be finite")
        if np.any(l_np >= u_np):
            raise ValueError("Each physical lower bound must be below its upper bound")

        l_tilde = standardization.to_standardized(jnp.asarray(l_np))
        u_tilde = standardization.to_standardized(jnp.asarray(u_np))
        width = u_tilde - l_tilde
        if not bool(jnp.all(jnp.isfinite(width))) or not bool(jnp.all(width > 0)):
            raise ValueError("Standardized bounds must have finite positive widths")
        return cls(standardization, l_tilde, u_tilde)

    @property
    def bounded(self) -> bool:
        return self.l_tilde is not None

    def eta_to_theta_tilde(self, eta: Array) -> Array:
        """Map unconstrained site coordinates to standardized coordinates."""

        eta = jnp.asarray(eta, dtype=jnp.float64)
        if (
            eta.ndim == 0
            or eta.shape[-1] != self.standardization.theta_bar_dagger.shape[0]
        ):
            raise ValueError("eta must have trailing dimension d")
        if not self.bounded:
            return eta
        return self.l_tilde + (self.u_tilde - self.l_tilde) * jnn.sigmoid(eta)

    def theta_tilde_to_eta(self, theta_tilde: Array) -> Array:
        """Inverse map on the box interior; endpoints have infinite eta."""

        theta_tilde = jnp.asarray(theta_tilde, dtype=jnp.float64)
        if (
            theta_tilde.ndim == 0
            or theta_tilde.shape[-1] != self.standardization.theta_bar_dagger.shape[0]
        ):
            raise ValueError("theta_tilde must have trailing dimension d")
        if not self.bounded:
            return theta_tilde
        unit = (theta_tilde - self.l_tilde) / (self.u_tilde - self.l_tilde)
        return jsp.special.logit(unit)

    def log_jacobian(self, eta: Array) -> Array:
        """Sum log|d theta_tilde / d eta| over all site coordinates."""

        eta = jnp.asarray(eta, dtype=jnp.float64)
        if (
            eta.ndim == 0
            or eta.shape[-1] != self.standardization.theta_bar_dagger.shape[0]
        ):
            raise ValueError("eta must have trailing dimension d")
        if not self.bounded:
            return jnp.zeros((), dtype=eta.dtype)
        return jnp.sum(
            jnp.log(self.u_tilde - self.l_tilde)
            - jnn.softplus(-eta)
            - jnn.softplus(eta)
        )
