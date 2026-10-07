# Bayesian calibration

Python/JAX research implementation of the spatial calibration model and its seven-update Gibbs sampler. The available theta transitions are collapsed MH, MALA, simplified MMALA, uncollapsed NUTS, and collapsed NUTS. Numerical calculations use float64 and explicit JAX PRNG keys.

The method specification, decisions, and validation record are in [docs/implementation-plan.md](docs/implementation-plan.md). Public reproduction means generating comparable results from fresh inputs under the same model and experiment protocol. Completed short runs demonstrate execution; posterior agreement requires adequate chains and assessed Monte Carlo uncertainty.

## Install and inspect the interface

The supported environment is Python 3.13 or newer, with BlackJAX 1.6.2. The lock file records the dependency resolution; validation currently uses Python 3.13.15, JAX 0.11.2, and a CPU backend.

```sh
uv sync --locked --python 3.13 --extra preprocessing
uv run bayesiancalibration --help
uv run bayesiancalibration run --help
```

The `preprocessing` extra supplies XLSX reading and plotting. It is unnecessary when numerical inputs are already available. `python -m bayesiancalibration` exposes the same CLI after installation. The CLI enables JAX float64; callers using the numerical API should enable `jax_enable_x64` themselves.

## A fresh experiment

Prepare loading curves, fit the library once, and share those fixed inputs across methods:

```sh
uv run bayesiancalibration preprocess \
  --input-dir /path/to/source-curves --output-dir outputs/preprocessing \
  --seed 1024 --candidate-step 0.8
uv run bayesiancalibration prepare \
  --input outputs/preprocessing/synthetic_preprocessing.npz \
  --output outputs/prepared.npz
uv run bayesiancalibration run \
  --prepared outputs/prepared.npz --output outputs/comparison \
  --method all --seed 20260928 --chains 4 --num-warmup 1000 \
  --production-seconds 21600 --batch-size 10
```

The last command expresses the existing six-production-hour-per-chain protocol and executes chains locally in sequence. To request a fixed number of retained sweeps, replace `--production-seconds` with `--num-samples`. The default is 1,000 warmup sweeps and 1,000 retained sweeps per chain. Production time measures synchronized numerical execution, validation, and host array recording; it excludes sweep compilation, warmup, and file I/O. The last started batch is completed.

Each preparation file and run directory must be new. The workflow has no checkpoint loading, restart mode, historical compatibility gate, scheduler, or worker locking. Draw files are ordinary research outputs. Initialization is shared by chain index across methods; method/chain sampling streams are independent. Warmup adaptation freezes before retained sampling.

## Inputs and preparation

Source data are supplied separately. The loading-only CSV/XLSX reader expects these files in `--input-dir`:

| File | Layout |
| --- | --- |
| `2026BKA.xlsx` | `Sheet1`; header `Case`, columns containing `Ea`, `Em`, `etr`; 20 rows with Case IDs 1–20 |
| `BKA5_2_depth.csv`, `BKA5_2_load.csv` | Header followed by 20 numeric rows × 501 depth/load values |
| `syn_theta_spatial_depth.csv`, `syn_theta_spatial_load.csv` | Header followed by 60 numeric rows × 501 depth/load values |
| `theta_spatial.csv` | Header `EA,EM,etr`; 60 rows of evaluation truth |

Within each source, curves share a strictly increasing depth grid from 0 to 400. Row order determines library runs and field sites. The field coordinates are three rows at y = 12, 6, 0, with x = 0, 6, …, 114. Units are unspecified; numeric loads are unchanged. Truth is retained for evaluation in the preprocessing archive and never enters fitting or the target.

The synthetic experiment adds independent N(0,1) field noise with seed 1024 before slope matching and projection. Alignment matches mean local OLS slopes over a four-depth-unit window on a regular candidate grid, choosing the smallest offset in an exact tie. Library loads are baseline-shifted and depths normalized after alignment. Both sources use the five-column cubic spline basis corresponding to R `splines::bs(..., knots=c(1/3,2/3), intercept=FALSE)`. Library coefficients use full-rank QR; field projection uses positive-diagonal QR factors. A slope-error plot and recipe/provenance metadata accompany the numerical archive.

For other datasets, supply a pickle-free NPZ directly to `prepare` with the following observed arrays. Use `data.project_loading_curves` for the numerical loading projection without source readers, synthetic noise, truth, or plotting.

| Array | Shape and meaning |
| --- | --- |
| `theta_s_dagger` | `(r,d)` physical simulator-library parameters |
| `F_s` | `(r,k)` fixed fitted library coefficients |
| `y_tilde` | `(n*k,)` projected field observations |
| `R` | `(n*k,n*k)` block-diagonal field QR factors |
| `s` | `(n,p)` field spatial coordinates |
| `metadata` | Optional scalar JSON string documenting input provenance |

The stacking convention is **site/run first, then active branch, then coefficient**. With branch sizes `(5,5)`, a site's loading coefficients precede its unloading coefficients. No truth array is required by `prepare`.

`prepare --scientific-config FILE.json` selects the shared model and library-fitting specification. The installed default is [scientific.json](src/bayesiancalibration/configs/scientific.json): loading only, three calibration parameters, unbounded sites, spatial length scale 12, physical mean-prior center `[37850,24060,0.071]`, standardized mean covariance `4I`, inverse-Wishart parameters `(5,I)`, discrepancy prior `(0,1e-6 I)`, and inverse-gamma variance parameters `(1.01,0.01)`. Different dimensions/branches require a corresponding complete scientific specification.

Library sample mean and sample standard deviation (divisor `r-1`) are frozen before fitting. The default profile fit uses three log-length-scale starts `[-1,-1,-1]`, `[0,0,0]`, `[1,1,1]`, `gtol=1e-6`, `ftol=1e-10`, and `maxiter=1000`. `cv_nlpd` and `cv_wmse` remain available. The default has no GP nugget/jitter, no fitted input bounds, and inferred coefficient variances. The prepared archive records the selected length scales, standardization, scientific settings, fit diagnostics, provenance, and environment.

The coefficient GP kernel is selected by `library_fit.kernel`: `se`, `matern32` (the default), or `matern52`. Each has unit amplitude, with sampled `sigma_c2` carrying output variance. All use the standardized ARD distance `r = sqrt(sum_q((x_q-x'_q)/lambda_c[q])^2)`. SE is `exp(-r²/2)`; Matérn 3/2 is `(1+sqrt(3)r)exp(-sqrt(3)r)`; Matérn 5/2 is `(1+sqrt(5)r+5r²/3)exp(-sqrt(5)r)`. These are radial kernels using anisotropic length scales, following the [standard Matérn convention](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html). The spatial GP `K_theta` retains its SE kernel and fixed range 12.

An explicit preparation flag overrides the scientific file's coefficient kernel:

```sh
bayesiancalibration prepare --input inputs.npz --output prepared-matern52.npz --kernel matern52
```

Without `--kernel`, the file selection is preserved; an omitted `library_fit.kernel` defaults to `matern32` during fresh preparation. The selected kernel is used consistently for profile/CV fitting, library factors, field conditioning and geometry checks, and is frozen in prepared metadata. Changing kernels requires fresh preparation and length-scale fitting. Prepared archives without an explicit fitted-kernel record must be prepared again; they are not interpreted using the new default. In the numerical API, pass `kernel="se"`, `kernel="matern32"`, or `kernel="matern52"` to `LibraryGP.from_data`, `fit_library_length_scales`, and the fitting objectives; `squared_exponential_kernel` remains explicitly SE.

## Sampler files and CLI overrides

Run controls belong to the CLI: paths, method, seed, chain count, warmup and production lengths, batch size, diagnostic mode, and sampler-file selection. Sampler files contain only settings applicable to their method. Packaged files provide the defaults:

| Method | Sampler settings |
| --- | --- |
| [mh](src/bayesiancalibration/configs/mh.json) | `num_initial=100`, `initial_proposal_variance=1e-6` |
| [mala](src/bayesiancalibration/configs/mala.json) | `num_initial=100`, `initial_proposal_variance=1`, `epsilon=0.1`, `target_accept=0.574` |
| [mmala](src/bayesiancalibration/configs/mmala.json) | `epsilon=0.1`, `epsilon_G=1e-6`, `target_accept=0.574` |
| [nuts](src/bayesiancalibration/configs/nuts.json), [collapsed_nuts](src/bayesiancalibration/configs/collapsed_nuts.json) | `initial_step_size=1`, `target_accept=0.8`, `mass_structure="diagonal"`, `block_size=null`, `max_num_doublings=10`, `divergence_threshold=1000` |

Resolution is **packaged defaults → file values → explicitly supplied CLI values**. A file may specify a subset of applicable settings and an optional matching `method`. Absent CLI flags preserve file values. Unknown or method-inapplicable settings are rejected. Settings are resolved and validated once before initialization; model/preparation settings do not belong in sampler files.

For example, save `mala.json`:

```json
{
  "method": "mala",
  "num_initial": 100,
  "initial_proposal_variance": 1.0,
  "epsilon": 0.03,
  "target_accept": 0.574
}
```

```sh
uv run bayesiancalibration run \
  --prepared outputs/prepared.npz --output outputs/mala-fixed \
  --method mala --sampler-config mala.json \
  --epsilon 0.02 --target-accept none
```

Here epsilon remains 0.02 while MALA covariance adaptation continues. NUTS mass choices are `diagonal`, `kronecker`, and `dense`; `--block-size all` explicitly overrides a blocked file setting with all-site updating. A sampler directory supplies `METHOD.json` for each selected method. CLI overrides must apply to every selected method. `num_initial` must fit within warmup and NUTS block sizes must fit the prepared site count.

## Validation and outputs

Normal execution validates inputs and initialization, names each Gibbs update, checks finite outputs and positive sampled variances, validates theta/adaptation quantities, and audits GP geometry at each numerical batch endpoint. `--diagnostic` applies the same geometry criterion to every completed sweep and collects conditional/factor evidence on failure. It adds no jitter, clipping, redraw, retry, or proposal correction. Ordinary candidate rejection, NUTS divergence, and tree limits retain their existing meanings.

`experiment.json` records effective settings, explicit overrides, preparation, initialization recipe, and environment. Each method/chain directory contains `configuration.json`, separate `warmup-*.npz` and `draws-*.npz` files, and `result.json` with timings/counts/final tuning. Numeric archives contain `state.FIELD`, theta diagnostics (`theta.FIELD`, or `theta.FIELD.block_I` for ragged blocks), full joint density, site movement, and full sweep indices. A failing batch is not saved as valid draws. `failure.json` records method/chain/phase, sweep or audit endpoint, update/quantity/role/site/block where applicable, failed predicate, and focused diagnostic evidence. Candidate arrays and complete NUTS trajectories are not retained.

After successful chains, `analysis.json` reports model variables, physical theta, site contrasts, spatial scales/correlations, and projected conditional observation means. It includes rank-normalized split/folded R-hat, bulk/tail ESS, and mean MCSE, following [the published diagnostic definitions](https://mc-stan.org/posterior/reference/rhat.html). Unequal timed lengths use a reported common prefix. At least two chains and eight draws each are required to compute these diagnostics here; that minimum does not establish convergence. Undefined/short-chain diagnostics are recorded as `null`; pooled and per-chain constant flags identify immobility. Predictive means are summarized without generating new observation-noise draws.

For a production comparison, inspect retained acceptance/movement, NUTS divergences and tree-cap frequencies alongside the chain summaries. The plan records proposed screening targets (R-hat < 1.01, bulk/tail ESS ≥ 400, mean MCSE/SD ≤ 0.05); these are review criteria, not automatic success gates. Compare posterior summaries with Monte Carlo uncertainty before ranking efficiency. Pooled ESS/time uses summed production time across chains from `result.json`, with any unused common-prefix draws reported; compilation, warmup and output costs are assessed separately. Full predictive-noise checks, plots and an adequate production comparison remain experiment work.

Automatic CLI initialization currently implements the declared **unbounded** experiment recipe: prior mean draw, `Sigma_theta=I`, spatial eta draw, `delta=m_delta_0`, `sigma_y2=1`, `sigma_c2=mean(F_s**2)`, then an exact `c_f` conditional draw. The numerical API also supports bounded site transformations with an explicitly supplied complete initial state; a bounded automatic initialization recipe remains unspecified. GP degeneracy remedies remain an open decision. Fresh numerical failures must be diagnosed before interpreting posterior comparisons.

## Reading order and tests

| Responsibility | Modules |
| --- | --- |
| Numerical preparation | `data.py` (`loading_spline_basis`, `match_loading_offset`, `project_loading_curves`) |
| Fixed state, coordinates, GP | `state.py`, `transforms.py`, `gp.py` |
| Posterior and exact conditionals | `targets.py`, `gibbs.py` |
| Theta transitions | `samplers/metropolis.py`, `samplers/mmala.py`, `samplers/nuts.py` |
| Adaptation formulas and NUTS mass estimation | `adaptation.py` |
| Complete sweep and numerical loops | `mcmc.py`: delta → sigma_y2 → mu_theta → Sigma_theta → sigma_c2 → eta → c_f |
| Runtime validation and error context | `validation.py` |
| Source I/O and fresh-run interface | `preprocessing.py`, `experiment.py`, `cli.py` |
| Retained-chain assessment | `analysis.py` |

The exact coefficient refresh immediately follows the theta transition, including rejection. `SweepRunner` reuses compiled functions for a fixed target/sampler without experiment I/O. Model state, tuning, adaptation state, and diagnostics stay distinct. Warmup and production calls cannot share one batch.

```sh
uv run python -m unittest discover -s tests -v
```

Mathematical/reference tests use fresh fixtures and run without external data. To include the supplied loading-curve integration tests, set `BAYESIANCALIBRATION_TEST_DATA` to a directory with the files above and install the preprocessing extra. Short CLI tests cover all five methods, configuration precedence, numerical batching equivalence, and injected failures. Full production comparisons and posterior/predictive agreement remain separate experiment work.
