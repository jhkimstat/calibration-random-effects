"""Loading-curve preparation mathematics, without readers, plots or file I/O.

Inputs are observed loads: noise generation belongs to the experiment recipe.
Library/run and field/site axes lead all curve arrays; coefficients and QR
observations use site/run-major stacking with five loading coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import qr, solve_triangular
from scipy.stats import linregress


_KNOTS = np.array([0.0] * 4 + [1.0 / 3.0, 2.0 / 3.0] + [1.0] * 4)


def loading_spline_basis(t: np.ndarray) -> np.ndarray:
    """Match R splines::bs cubic, two knots, and intercept=FALSE exactly.

    SciPy returns six columns for the complete knot vector. Dropping its
    first column reproduces R's five-column convention, including endpoints.
    """

    t = np.asarray(t, dtype=np.float64)
    if t.ndim != 1 or not np.all(np.isfinite(t)) or np.any((t < 0) | (t > 1)):
        raise ValueError("Normalized loading depth must be finite in [0,1]")
    return BSpline.design_matrix(t, _KNOTS, 3, extrapolate=False).toarray()[:, 1:]


def match_loading_offset(
    library_depth: np.ndarray,
    library_load: np.ndarray,
    field_depth: np.ndarray,
    field_load: np.ndarray,
    *,
    window: float = 4.0,
    candidate_step: float = 0.8,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Match local OLS slopes from observed depths in each four-unit window.

    Depth may be uneven, repeated, out of order, and different for each row.
    Accept either a shared (m,) depth vector or row-specific (r,m)/(n,m)
    matrices. Evaluate a regular offset grid over the common full-window
    library range, with spacing at most candidate_step. Prefer the smallest
    evaluated offset in an exact tie. Interpolation is reserved for shifting
    library loads after alignment.
    """

    h_s = np.asarray(library_depth, dtype=np.float64)
    y_s = np.asarray(library_load, dtype=np.float64)
    h_f = np.asarray(field_depth, dtype=np.float64)
    y_f = np.asarray(field_load, dtype=np.float64)
    if (y_s.ndim != 2 or y_f.ndim != 2 or y_s.shape[0] < 1 or y_f.shape[0] < 1
            or h_s.ndim not in (1, 2) or h_f.ndim not in (1, 2)
            or not np.isfinite(window) or window <= 0
            or not np.isfinite(candidate_step) or candidate_step <= 0
            or not all(np.all(np.isfinite(a)) for a in (h_s, h_f, y_s, y_f))):
        raise ValueError("Slope matching requires finite curves, window, and candidate_step")
    if h_s.ndim == 1:
        h_s = np.broadcast_to(h_s, y_s.shape)
    if h_f.ndim == 1:
        h_f = np.broadcast_to(h_f, y_f.shape)
    if h_s.shape != y_s.shape or h_f.shape != y_f.shape:
        raise ValueError("Depth and load must have matching row and sample shapes")

    # SciPy supplies the OLS fit. This local adapter additionally checks the
    # model's window and near-zero depth spread, which linregress does not.
    def local_slope(h: np.ndarray, y: np.ndarray, start: float) -> float:
        inside = (h >= start) & (h <= start + window)
        x = h[inside]
        if x.size < 2:
            return np.nan
        threshold = 32.0 * np.finfo(np.float64).eps * max(1.0, np.max(np.abs(x)))
        if np.std(x) <= threshold:
            return np.nan
        slope = float(linregress(x, y[inside]).slope)
        return slope if np.isfinite(slope) else np.nan

    field_slopes = np.array([
        local_slope(h, y, float(np.min(h))) for h, y in zip(h_f, y_f)
    ])
    if not np.all(np.isfinite(field_slopes)):
        raise ValueError("Field slope cannot be calculated: local depth variance ≈ 0")
    mean_field_slope = float(field_slopes.mean())

    lower = float(np.max(np.min(h_s, axis=1)))
    upper = float(np.min(np.max(h_s, axis=1) - window))
    if lower > upper:
        raise ValueError("No common complete library slope window")
    intervals = int(np.ceil((upper - lower) / candidate_step))
    offsets = np.linspace(lower, upper, intervals + 1)
    mean_library_slopes = np.empty(offsets.size, dtype=np.float64)
    for index, h0 in enumerate(offsets):
        slopes = np.array([
            local_slope(h, y, float(h0)) for h, y in zip(h_s, y_s)
        ])
        mean_library_slopes[index] = (
            float(slopes.mean()) if np.all(np.isfinite(slopes)) else np.nan
        )
    errors = np.abs(mean_library_slopes - mean_field_slope)
    if not np.any(np.isfinite(errors)):
        raise ValueError("No library offset has estimable local slopes for every run")
    # np.nanargmin returns the earliest candidate for an exact score tie.
    offset = float(offsets[np.nanargmin(errors)])
    return offset, offsets, errors, mean_library_slopes, field_slopes



@dataclass(frozen=True)
class LoadingProjection:
    """Numerical preparation output for shared sorted grids, with k=5.

    F_s: (r,5), y_tilde: (n*5,), R: (n*5,n*5), and curve arrays retain
    run/site-major order. The slope matcher itself also accepts irregular,
    unsorted row-specific grids; this QR pipeline requires shared sorted grids.
    """

    F_s: np.ndarray
    y_tilde: np.ndarray
    R: np.ndarray
    library_depth: np.ndarray
    library_t: np.ndarray
    library_load_shifted: np.ndarray
    field_t: np.ndarray
    slope_offsets: np.ndarray
    slope_errors: np.ndarray
    mean_library_slopes: np.ndarray
    field_slopes: np.ndarray
    h0: float
    library_fit_rms: float
    field_projection_discarded_rms: float


def project_loading_curves(
    library_depth: np.ndarray, library_load: np.ndarray,
    field_depth: np.ndarray, field_load: np.ndarray, *, candidate_step: float = 0.8,
) -> LoadingProjection:
    """Apply the specified alignment, R spline fit and field QR projection.

    Depths have shapes (m_s,) and (m_f,), loads (r,m_s) and (n,m_f).
    This model-specific composition makes the preparation equations readable
    independently of the synthetic loader, without changing their calculations.
    The fixed 4-unit window and candidate grid have the existing interpretation.
    """
    h_s, h_f = (np.asarray(value, dtype=np.float64)
                for value in (library_depth, field_depth))
    y_s, y_f = (np.asarray(value, dtype=np.float64)
                for value in (library_load, field_load))
    if (h_s.ndim != 1 or h_f.ndim != 1 or h_s.size < 2 or h_f.size < 2
            or y_s.ndim != 2 or y_f.ndim != 2 or y_s.shape[1] != h_s.size
            or y_f.shape[1] != h_f.size or y_s.shape[0] < 1 or y_f.shape[0] < 1
            or not all(np.all(np.isfinite(value)) for value in (h_s, h_f, y_s, y_f))
            or np.any(np.diff(h_s) <= 0) or np.any(np.diff(h_f) <= 0)):
        raise ValueError("QR preparation requires finite curves on shared sorted depth grids")
    h0, offsets, errors, slopes_s, slopes_f = match_loading_offset(
        h_s, y_s, h_f, y_f, candidate_step=candidate_step,
    )
    # Baseline interpolation is separate from raw-point slope estimation.
    keep = h_s > h0 + 1e-10
    h_s_shifted = np.concatenate(([0.0], h_s[keep] - h0))
    origin_load = np.array([np.interp(h0, h_s, row) for row in y_s])
    y_s_shifted = np.concatenate((
        np.zeros((y_s.shape[0], 1)), y_s[:, keep] - origin_load[:, None]
    ), axis=1)
    t_s = h_s_shifted / (h_s[-1] - h0)
    t_f = (h_f - h_f[0]) / (h_f[-1] - h_f[0])
    Phi_s, Phi_f = loading_spline_basis(t_s), loading_spline_basis(t_f)
    Q_s, R_s = qr(Phi_s, mode="economic")
    Q_f, R_f = qr(Phi_f, mode="economic")
    if np.linalg.matrix_rank(R_s) != 5 or np.linalg.matrix_rank(R_f) != 5:
        raise ValueError("Loading spline design must have full column rank")
    F_s = solve_triangular(R_s, Q_s.T @ y_s_shifted.T).T
    signs = np.where(np.diag(R_f) < 0, -1.0, 1.0)
    Q_f *= signs
    R_f = signs[:, None] * R_f
    y_tilde = (Q_f.T @ y_f.T).T.reshape(-1)
    R = np.kron(np.eye(y_f.shape[0]), R_f)
    return LoadingProjection(
        F_s, y_tilde, R, h_s_shifted, t_s, y_s_shifted, t_f,
        offsets, errors, slopes_s, slopes_f, h0,
        float(np.sqrt(np.mean((Phi_s @ F_s.T - y_s_shifted.T)**2))),
        float(np.sqrt(np.mean((Q_f @ (Q_f.T @ y_f.T) - y_f.T)**2))),
    )
