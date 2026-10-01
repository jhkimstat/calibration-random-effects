"""Model boundary validation and focused numerical failure evidence.

These checks report invalid quantities without changing support, proposals,
random keys or draws. Shape/SPD checks run at initialization; inexpensive
named reductions can be functionalized by JAX checkify in each Gibbs sweep.
"""

from numbers import Integral
import re

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import checkify

from bayesiancalibration.state import CalibrationState


def check_quantity(predicate, *, update, quantity, role="draw", site=None,
                   block=None, criterion="finite", description=""):
    """Name a device predicate without extra density evaluations or factors.

    A custom helper keeps machine-readable context consistent across traced
    kernels; checkify supports traced site/block indices as format arguments.
    debug_check has no effect in an unchecked pure numerical kernel.
    """
    message = (f"[update={update} quantity={quantity} role={role} "
               f"predicate={criterion}")
    arguments = {}
    for name, value in (("site", site), ("block", block)):
        if value is not None:
            message += " " + name + "={" + name + "}"
            arguments[name] = jnp.asarray(value)
    checkify.debug_check(predicate, message + "] " + description, **arguments)


def validate_control(iteration, phase):
    """Validate local sweep counters independently of sampler adaptation."""
    if (isinstance(iteration, bool) or not isinstance(iteration, Integral)
        or iteration < 0):
        raise ValueError("iteration must be a nonnegative integer")
    if phase not in ("warmup", "sampling"):
        raise ValueError("phase must be 'warmup' or 'sampling'")


def validate_key(key):
    """Require one scalar JAX key without consuming it."""
    try:
        data = jax.random.key_data(key)
        jax.random.split(key, 2)
    except (TypeError, ValueError) as error:
        raise ValueError("key must be a scalar JAX PRNG key") from error
    if data.ndim != 1:
        raise ValueError("key must be a scalar JAX PRNG key")


def validate_spd(value, shape, name):
    """Normalize a declared symmetric positive-definite float64 array."""
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != shape or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    if not np.array_equal(matrix, np.swapaxes(matrix, -1, -2)):
        raise ValueError(f"{name} must be symmetric")
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be positive definite") from error
    return jnp.asarray(matrix)


def validate_model_state(target, state):
    """Validate model shapes/support, field GP and full joint outside JIT.

    Model-specific dimensions and GP geometry cannot be checked by a generic
    array validator. No sampler or artificial proposal is constructed here.
    """
    if not jax.config.jax_enable_x64:
        raise ValueError("Gibbs sampling requires jax_enable_x64=True")
    if not isinstance(state, CalibrationState):
        raise ValueError("model_state must be a CalibrationState")
    n = target.C_theta.shape[0]
    k = target.gp.F_s.shape[1]
    d = target.gp.theta_s_tilde.shape[1]
    shapes = ((n, d), (n*k,), (k,), (len(target.branch_sizes),),
              (d,), (d, d), (k,))
    checked = []
    for name, value, shape in zip(CalibrationState._fields, state, shapes):
        array = np.asarray(value, dtype=np.float64)
        if array.shape != shape or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite with shape {shape}")
        if name in ("sigma_y2", "sigma_c2") and np.any(array <= 0):
            raise ValueError(f"{name} must be positive")
        checked.append(jnp.asarray(array))
    state = CalibrationState(*checked)
    validate_spd(state.Sigma_theta, (d, d), "Sigma_theta")
    target.gp.validate_field_sites(target.coordinates.eta_to_theta_tilde(state.eta))
    if not np.isfinite(float(target.full_joint_uncollapsed(*state))):
        raise ValueError("Full joint log density must be finite at a chain boundary")
    return state


def chain_context(chain, chain_index=None, *, sweep=None):
    """Host labels used by errors; they never enter numerical kernels."""
    method = {"RandomWalkChain": "mh", "MALAChain": "mala",
              "MMALAChain": "mmala", "NUTSChain": "nuts"}[type(chain).__name__]
    if method == "nuts" and chain.collapsed:
        method = "collapsed_nuts"
    return {"method": method, "chain": chain_index, "phase": chain.phase,
            "sweep": chain.iteration + 1 if sweep is None else sweep}


def _array_summary(value):
    """Summarize failed-run context instead of dumping large model matrices."""
    array = np.asarray(value)
    finite = array[np.isfinite(array)]
    return {"shape": list(array.shape), "finite_count": int(finite.size),
            "size": int(array.size),
            "minimum": float(finite.min()) if finite.size else None,
            "maximum": float(finite.max()) if finite.size else None}


_UPDATE_ORDER = ("delta", "sigma_y2", "mu_theta", "Sigma_theta", "sigma_c2", "eta", "c_f")
_CONDITIONING = {
    "delta": ("c_f", "sigma_y2"), "sigma_y2": ("c_f", "delta"),
    "mu_theta": ("eta", "Sigma_theta"), "Sigma_theta": ("eta", "mu_theta"),
    "sigma_c2": ("eta", "c_f"),
    "eta": ("c_f", "delta", "sigma_y2", "mu_theta", "Sigma_theta", "sigma_c2"),
    "c_f": ("eta", "delta", "sigma_y2", "sigma_c2"),
}


def _update_inputs(previous, attempted, update):
    """Recover the newest conditioning used before the named failed update."""
    if update not in _UPDATE_ORDER:
        return attempted
    index = _UPDATE_ORDER.index(update)
    return CalibrationState(*(getattr(attempted if _UPDATE_ORDER.index(name) < index else previous, name)
                              for name in CalibrationState._fields))


def _reported_array(value):
    """Store focused small inputs and summaries, encoding nonfinite values as null."""
    result = _array_summary(value)
    array = np.asarray(value)
    if array.size <= 256:
        result["values"] = np.where(np.isfinite(array), array, None).tolist()
    return result


def _matrix_evidence(value):
    """Failure-only spectra/factor evidence; no predicate feeds back into sampling."""
    result = _reported_array(value)
    matrix = np.asarray(value)
    if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1] and np.all(np.isfinite(matrix)):
        eigenvalues = np.linalg.eigvalsh((matrix + matrix.T) / 2)
        result["eigenvalue_minimum"] = float(eigenvalues.min())
        result["eigenvalue_maximum"] = float(eigenvalues.max())
        try:
            result["cholesky_diagonal"] = _reported_array(np.diag(np.linalg.cholesky(matrix)))
        except np.linalg.LinAlgError as error:
            result["cholesky_error"] = str(error)
    return result


def _conditional_evidence(target, inputs, update):
    """Recompute the failed update's specified parameters only after failure.

    Reuse the existing conditional kernels, with latest conditioning inputs.
    This diagnostic work draws no random numbers and cannot change a result.
    """
    from bayesiancalibration import gibbs

    theta_tilde = target.coordinates.eta_to_theta_tilde(inputs.eta)
    prior = target.spatial_prior
    if update == "delta":
        _, noise = target.observation_arrays(inputs.delta, inputs.sigma_y2)
        mean, covariance = gibbs.discrepancy_conditional_moments(
            target.y_tilde, inputs.c_f, target.R, noise, target.m_delta_0, target.V_delta_0)
        return {"m_delta": _reported_array(mean), "V_delta": _matrix_evidence(covariance)}
    if update == "sigma_y2":
        shape, scale = gibbs.branch_noise_conditional_parameters(
            target.y_tilde, inputs.c_f, inputs.delta, target.R, target.branch_sizes,
            target.alpha_y_0, target.beta_y_0)
        return {"alpha_y": _reported_array(shape), "beta_y": _reported_array(scale)}
    if update == "mu_theta":
        mean, covariance = gibbs.spatial_mean_conditional_moments(
            theta_tilde, target.C_theta, inputs.Sigma_theta, prior.m_theta_0, prior.V_theta_0)
        return {"m_theta": _reported_array(mean), "V_theta": _matrix_evidence(covariance)}
    if update == "Sigma_theta":
        nu, scale = gibbs.spatial_covariance_conditional_parameters(
            theta_tilde, target.C_theta, inputs.mu_theta, prior.nu_theta_0, prior.S_theta_0)
        return {"nu_theta": _reported_array(nu), "S_theta": _matrix_evidence(scale)}
    if update in ("sigma_c2", "c_f", "eta", "geometry"):
        k = target.gp.F_s.shape[1]
        variances = jnp.ones(k, dtype=jnp.float64) if update == "sigma_c2" else inputs.sigma_c2
        mean, field, covariance = target.gp.conditional_moments(theta_tilde, variances)
        result = {"m_f_given_s": _reported_array(mean),
                  "C_f_given_s": _matrix_evidence(field),
                  "Sigma_f_given_s": _matrix_evidence(covariance)}
        try:
            target.gp.validate_field_sites(theta_tilde)
            result["unjittered_geometry"] = "valid"
        except ValueError as error:
            result["unjittered_geometry"] = str(error)
        if update == "sigma_c2":
            shape, scale = gibbs.coefficient_variance_conditional_parameters(
                inputs.c_f, mean, field, target.gp.q_s, target.gp.F_s.shape[0],
                target.branch_sizes, target.alpha_c_0, target.beta_c_0)
            result.update(alpha_c=_reported_array(shape), beta_c=_reported_array(scale))
        else:
            from bayesiancalibration.linalg import projected_marginal_moments
            discrepancy, noise = target.observation_arrays(inputs.delta, inputs.sigma_y2)
            mean_y, covariance_y = projected_marginal_moments(mean, covariance, target.R, discrepancy, noise)
            result.update(m_y=_reported_array(mean_y), V_y=_matrix_evidence(covariance_y),
                          Omega_y=_matrix_evidence(noise))
        return result
    return {}


class SamplingError(FloatingPointError):
    """Fatal checked execution error with JSON-safe diagnostic evidence.

    Evidence is deliberately a report, not a serialized state for restart.
    It describes the input to the failed sweep and the first failed predicate.
    """

    def __init__(self, message, chain, *, chain_index=None, stage="sweep",
                 quantity=None, role=None, block=None, sweep=None, batch=None,
                 attempted_state=None, target=None, diagnostic=False,
                 attempted_tuning=None, failed_value=None):
        details = chain_context(chain, chain_index, sweep=sweep)
        details.update(update=stage, quantity=quantity, role=role,
                       site=None, block=block, predicate=message)
        match = re.search(r"\[([^\]]+)\]", message)
        if match:
            for name, value in re.findall(r"(\w+)=([^\s]+)", match.group(1)):
                details[name] = int(value) if name in ("site", "block") else value
        details["message"] = message
        details["batch"] = list(batch) if batch is not None else None
        details["key_data"] = np.asarray(jax.random.key_data(chain.key)).tolist()
        model_label = "audited_endpoint_model" if details["update"] == "geometry" else "sweep_start_model"
        details[model_label] = {
            name: _array_summary(value)
            for name, value in zip(CalibrationState._fields, chain.model_state)
        }
        if attempted_state is not None:
            update = details["update"]
            inputs = _update_inputs(chain.model_state, attempted_state, update)
            details["failed_update_inputs"] = {
                name: _reported_array(getattr(inputs, name))
                for name in _CONDITIONING.get(update, CalibrationState._fields)
            }
            if update == "eta":
                # A sequential site/block kernel may already have accepted
                # earlier positions before failure. Its intermediate current
                # or proposal eta is not returned by the complete sweep.
                details["theta_position_evidence"] = {
                    "sweep_start_eta": _reported_array(chain.model_state.eta),
                    "attempted_final_eta": _reported_array(attempted_state.eta),
                    "failed_site_or_block_position_available": False,
                }
            if details["quantity"] in CalibrationState._fields:
                details["failed_value"] = _reported_array(getattr(attempted_state, details["quantity"]))
            if diagnostic and target is not None and update != "eta":
                try:
                    details["conditional_evidence"] = _conditional_evidence(target, inputs, update)
                except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                    details["conditional_evidence_error"] = str(error)
        tuning_label = "audited_endpoint_tuning" if details["update"] == "geometry" else "previous_tuning"
        details[tuning_label] = {}
        for name in ("V_prop", "epsilon", "epsilon_G", "step_size", "inverse_mass_matrix"):
            if hasattr(chain, name):
                value = getattr(chain, name)
                details[tuning_label][name] = ([_array_summary(v) for v in value]
                                               if isinstance(value, tuple) else _array_summary(value))
        if attempted_tuning is not None:
            details["attempted_tuning"] = {
                name: _reported_array(value) for name, value in attempted_tuning.items()
            }
            if details["quantity"] in attempted_tuning:
                details["failed_value"] = _reported_array(attempted_tuning[details["quantity"]])
        if failed_value is not None:
            details["failed_value"] = _reported_array(failed_value)
        if chain.adaptation is not None:
            details["adaptation"] = {
                "completed": int(chain.adaptation.completed),
                "num_warmup": int(chain.adaptation.num_warmup),
            }
        self.diagnostics = details
        labels = " ".join(f"{name}={details[name]}" for name in
                          ("method", "chain", "phase", "sweep", "update", "quantity", "role", "site", "block"))
        super().__init__(f"{labels}: {message}")
