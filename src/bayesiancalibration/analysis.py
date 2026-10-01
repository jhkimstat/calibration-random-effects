"""Retained-chain summaries, separate from sampling and numerical validation.

Arrays have shape (chain, draw, ...). Rank normalization, folding and splitting
follow Vehtari et al. (2021); BlackJAX supplies R-hat and autocorrelation ESS.
See https://mc-stan.org/posterior/reference/rhat.html and the reference source
https://github.com/stan-dev/posterior/blob/master/R/convergence.R .
No diagnostic is fed back into transitions or warmup.
"""

from __future__ import annotations

import json
from pathlib import Path

import blackjax.diagnostics as diagnostics
import jax
import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata

from bayesiancalibration.state import CalibrationState


def _split(values: np.ndarray) -> np.ndarray:
    """Split chain halves, excluding the middle draw for odd lengths."""
    half = values.shape[1] // 2
    return np.concatenate((values[:, :half], values[:, -half:]), axis=0)


def _rank_normalize(values: np.ndarray) -> np.ndarray:
    """Pool ranks per quantity with average ties and Blom's normal scores.

    This composition is needed because BlackJAX's basic diagnostics do not
    rank-normalize. The denominator is S + 1/4, as in posterior's z_scale.
    """
    pooled = values.reshape((-1, values.shape[-1]))
    ranks = rankdata(pooled, axis=0, method="average")
    return ndtri((ranks - 3/8) / (pooled.shape[0] + 1/4)).reshape(values.shape)


def summarize_chains(values: np.ndarray) -> dict:
    """Summarize one array-valued quantity without treating smoke runs as converged.

    Return the maximum rank/folded split R-hat, bulk ESS, minimum 5%/95%
    indicator ESS and MCSE of the mean using raw split-chain ESS. At least
    two original chains and eight draws each are required here for diagnostics;
    that small numerical minimum is not an adequacy or convergence criterion.
    Undefined diagnostics (e.g. constant quantities) are NaN and saved as null.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 2 or min(values.shape[:2]) < 1 or not np.all(np.isfinite(values)):
        raise ValueError("Retained values must be finite (chain,draw,...) arrays")
    shape = values.shape[2:]
    flat = values.reshape(values.shape[:2] + (-1,))
    sufficient = values.shape[0] >= 2 and values.shape[1] >= 8
    sd = (np.std(values, axis=(0, 1), ddof=1)
          if values.shape[0] * values.shape[1] > 1 else np.full(shape, np.nan))
    result = {
        "mean": np.mean(values, axis=(0, 1)), "sd": sd,
        "quantiles_05_50_95": np.quantile(values, [.05, .5, .95], axis=(0, 1)),
        "rank_split_folded_rhat": np.full(shape, np.nan),
        "ess_bulk": np.full(shape, np.nan), "ess_tail": np.full(shape, np.nan),
        "ess_mean": np.full(shape, np.nan), "mcse_mean": np.full(shape, np.nan),
        "mcse_mean_over_sd": np.full(shape, np.nan),
        "constant": np.all(flat == flat[0, 0], axis=(0, 1)).reshape(shape),
        "constant_by_chain": np.all(flat == flat[:, :1], axis=1).reshape((values.shape[0],) + shape),
    }
    if not sufficient:
        return result
    if not jax.config.jax_enable_x64:
        raise RuntimeError("Enable jax_enable_x64 before chain analysis")
    split = _split(flat)
    ranked = _rank_normalize(split)
    # Fold around the median of the original pooled draws before splitting.
    folded = _rank_normalize(_split(np.abs(flat - np.median(flat, axis=(0, 1)))))
    # Constant quantities can make the reference diagnostics undefined. Keep
    # explicit immobility flags and null diagnostics in the saved report.
    with np.errstate(invalid="ignore", divide="ignore"):
        rhat = np.maximum(np.asarray(diagnostics.potential_scale_reduction(ranked)),
                          np.asarray(diagnostics.potential_scale_reduction(folded)))
        bulk = np.asarray(diagnostics.effective_sample_size(ranked))
        q05, q95 = np.quantile(flat, [.05, .95], axis=(0, 1))
        tail = np.minimum(
            np.asarray(diagnostics.effective_sample_size(_split((flat <= q05).astype(float)))),
            np.asarray(diagnostics.effective_sample_size(_split((flat <= q95).astype(float)))),
        )
        mean_ess = np.asarray(diagnostics.effective_sample_size(split))
    for name, value in (("rank_split_folded_rhat", rhat), ("ess_bulk", bulk),
                        ("ess_tail", tail), ("ess_mean", mean_ess)):
        result[name] = np.where(result["constant"], np.nan, np.asarray(value).reshape(shape))
    with np.errstate(invalid="ignore", divide="ignore"):
        result["mcse_mean"] = sd / np.sqrt(result["ess_mean"])
        result["mcse_mean_over_sd"] = result["mcse_mean"] / sd
    return result


def analyze_method(target, chain_directories: list[Path]) -> dict:
    """Analyze all supplied chains using the common retained prefix when unequal.

    Output counts preserve unequal timed lengths explicitly. Every model
    variable, physical site parameter, site-minus-site-0 contrast and projected
    conditional observation mean is summarized. The latter is a predictive
    mean functional; no new predictive-noise draws or PRNG keys are generated.
    """
    if not chain_directories:
        raise ValueError("At least one chain directory is required")
    traces, counts, outcomes = [], [], []
    for directory in chain_directories:
        pieces = {name: [] for name in CalibrationState._fields}
        if (directory / "failure.json").exists():
            outcomes.append("failed")
        elif (directory / "result.json").exists():
            outcomes.append(json.loads((directory / "result.json").read_text())["status"])
        else:
            outcomes.append("incomplete")
        for path in sorted(directory.glob("draws-*.npz")):
            with np.load(path, allow_pickle=False) as archive:
                for name in pieces:
                    pieces[name].append(archive[f"state.{name}"])
        counts.append(sum(len(part) for part in pieces["eta"]))
        traces.append({name: np.concatenate(parts) for name, parts in pieces.items()}
                      if pieces["eta"] else None)
    count = min(counts)
    if count == 0:
        return {"status": "no_retained_draws", "draws_by_chain": counts,
                "chain_outcomes": outcomes, "draws_per_chain_analyzed": 0}
    stacked = {name: np.stack([trace[name][:count] for trace in traces])
               for name in CalibrationState._fields}
    theta_tilde = target.coordinates.eta_to_theta_tilde(stacked["eta"])
    stacked["theta_physical"] = np.asarray(
        target.coordinates.standardization.to_physical(theta_tilde))
    if stacked["eta"].shape[2] > 1:
        stacked["theta_physical_contrast_to_site_0"] = (
            stacked["theta_physical"][:, :, 1:] - stacked["theta_physical"][:, :, :1])
    n = target.C_theta.shape[0]
    stacked["projected_observation_mean"] = (
        stacked["c_f"] + np.tile(stacked["delta"], (1, 1, n))) @ np.asarray(target.R).T
    Sigma = stacked["Sigma_theta"]
    scales = np.sqrt(np.diagonal(Sigma, axis1=-2, axis2=-1))
    stacked["spatial_sd"] = scales
    if scales.shape[-1] > 1:
        row, col = np.triu_indices(scales.shape[-1], k=1)
        stacked["spatial_correlations"] = Sigma[..., row, col] / (scales[..., row] * scales[..., col])
    summaries = {name: summarize_chains(values) for name, values in stacked.items()}
    sufficient = len(traces) >= 2 and count >= 8
    return {
        "status": "diagnostics_available" if sufficient else "too_short_for_chain_diagnostics",
        "diagnostic_definition": "rank-normalized split/folded R-hat; bulk and 5%/95% tail ESS; raw split ESS mean MCSE",
        "draws_by_chain": counts, "draws_per_chain_analyzed": count,
        "unused_draws_for_equal_length_diagnostics": [number - count for number in counts],
        "chain_outcomes": outcomes, "summaries": summaries,
        "interpretation": "Diagnostic availability is not convergence; short execution checks do not establish posterior agreement or sampler rankings.",
    }
